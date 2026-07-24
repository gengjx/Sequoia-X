"""多因子选股策略：IC加权合成综合因子分，选全市场Top N。

与规则式策略的区别：
  - 规则式(如均线放量)：固定条件筛选(布尔)，选出的票"全等"
  - 多因子：所有股票统一打分排序(连续值)，选出Top N，区分度更强

因子权重来自IC评估（有效因子按IC加权），定期重评估自动更新。
无效因子(IC接近0)自动剔除，避免噪音。
"""

import numpy as np
import pandas as pd

from sequoia_x.analysis.factor import compute_factors, cross_section_rank
from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.strategy.base import BaseStrategy
from sequoia_x.strategy.registry import register_strategy

logger = get_logger(__name__)


# IC加权默认权重（来自因子评估实测IC，定期重评估更新）
# 仅保留IC>0.02的有效因子，按IC值归一化为权重
_DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    # 正向因子（IC>0）
    "vol_20": 0.0698,       # 低波动溢价
    "rev_20": 0.0641,       # 20日反转
    "atr_pct": 0.0625,      # 低ATR占比
    "rev_5": 0.0489,        # 5日反转
    "vol_shrink_p": 0.0408, # 缩量企稳
    "rev_10": 0.0357,       # 10日反转
    "flag_tight": 0.0203,   # 旗形收敛
    # 负向因子（IC<0，方向已取负，故权重为正）
    "turnover": 0.0978,     # 低换手（IC-0.098最强，取负后正加权）
    "mom_60": 0.0764,       # 低60日动量（反转逻辑）
    "rps_120": 0.071,       # 低120日动量
    "mom_20": 0.0641,       # 低20日动量
    "amihud": 0.0516,       # 低Amihud非流动性
    "mom_10": 0.0511,       # 低10日动量
}


@register_strategy("multi_factor")
class MultiFactorStrategy(BaseStrategy):
    """多因子IC加权选股策略。

    选股逻辑：
    1. 全市场计算30因子截面值
    2. 横截面百分位排名（标准化0-100）
    3. IC加权合成综合因子分
    4. 选Top N（默认选入数量与规则式策略可比）

    Attributes:
        webhook_key: 'multi_factor'，路由到专属飞书机器人。
    """

    _TOP_N: int = 50  # 选入数量（与规则式策略平均选股量可比）

    def __init__(self, engine: DataEngine, settings: Settings,
                 factor_weights: dict[str, float] | None = None) -> None:
        super().__init__(engine, settings)
        # 权重优先级：外部注入 > DB动态加载 > 默认硬编码兜底
        if factor_weights:
            self._weights = factor_weights
        else:
            self._weights = self._load_db_weights()
        # 三态权重（市场状态自适应）
        self._state_weights: dict[str, dict[str, float]] = self._load_state_weights()

    def _load_db_weights(self) -> dict[str, float]:
        """从DB加载最新因子IC权重，DB空则用默认值兜底。

        IC评估引擎每次运行会写回DB，使选股自适应最新市场数据。
        """
        try:
            db_weights = self.engine.load_factor_weights()
            if db_weights:
                weights = {k: v["weight"] for k, v in db_weights.items() if v["weight"] != 0}
                logger.info(
                    f"因子权重已从DB加载（{len(weights)}个因子，"
                    f"最近更新：{next(iter(db_weights.values())).get('updated_at', '?')}）"
                )
                return weights
        except Exception as e:
            logger.warning(f"加载因子权重失败，使用默认值：{e!r}")
        return dict(_DEFAULT_FACTOR_WEIGHTS)

    def _load_state_weights(self) -> dict[str, dict[str, float]]:
        """从DB加载三态市场状态因子权重（bull/neutral/bear各一套）。"""
        try:
            return self.engine.load_market_factor_weights()
        except Exception:
            return {}

    def get_weights_for_state(self, market_state: str) -> dict[str, float]:
        """获取指定市场状态下的因子权重，无则回退到全局权重。"""
        state_w = self._state_weights.get(market_state)
        if state_w:
            logger.info(f"多因子使用{market_state}状态权重（{len(state_w)}个因子）")
            return state_w
        # 回退：bear→neutral→全局
        for fallback in ("neutral", "bull", "bear"):
            if fallback != market_state and fallback in self._state_weights:
                logger.info(f"多因子{market_state}权重缺失，回退到{fallback}（{len(self._state_weights[fallback])}个因子）")
                return self._state_weights[fallback]
        logger.info(f"多因子使用全局权重（{len(self._weights)}个因子）")
        return self._weights

    def run(self) -> list[str]:
        """执行多因子选股，返回综合因子分Top N的股票代码。"""
        symbols = list(self._shared_daily.keys()) if self._shared_daily else self.engine.get_local_symbols()

        # ── 选股池硬过滤：ST/退市预警 ──
        symbols = self._filter_universe(symbols)

        # P1: 市场状态自适应——根据当前市场状态选择对应权重
        market_state = self._detect_market_state()
        active_weights = self.get_weights_for_state(market_state)
        # ML 合成因子是市场状态无关的全局信号，三态权重表可能不含，
        # 从全局权重补充注入（达标时非零、未达标为0则不注入）
        if self._weights.get("ml_score"):
            active_weights["ml_score"] = self._weights["ml_score"]

        # 预加载财报数据（批量查一次，避免逐只查库）
        finance_map = self._load_finance_map(symbols)
        # 预加载资金流向（最近一天）
        fund_flow_map = self._load_fund_flow_map()
        # 预加载龙虎榜（近30天上榜次数+净买入额）
        lhb_map = self._load_lhb_map()
        # 预加载北向持股历史（用于北向因子）
        north_map = self._load_north_map()

        # 采集全市场因子截面（含质量因子+资金因子）+ 趋势确认数据
        rows = []
        trend_info: dict[str, dict] = {}
        for sym in symbols:
            df = self.get_daily(sym)
            if df is None or len(df) < 60:
                continue
            try:
                factors = compute_factors(
                    df,
                    finance=finance_map.get(sym),
                    fund_flow=fund_flow_map.get(sym),
                    lhb_data=lhb_map.get(sym),
                    north_hold=north_map.get(sym),
                )
                factors["symbol"] = sym
                rows.append(factors)

                # 趋势确认指标（避免纯反转因子选出的"飞刀"股）
                close = df["close"]
                ma20 = close.iloc[-20:].mean()
                ma60 = close.iloc[-60:].mean()
                ret_5d = close.iloc[-1] / close.iloc[-6] - 1 if len(close) >= 6 else 0
                ret_20d = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0
                trend_info[sym] = {
                    "above_ma20": close.iloc[-1] > ma20,
                    "ma20_above_ma60": ma20 > ma60,
                    "ret_5d": float(ret_5d) if ret_5d == ret_5d else 0,
                    "ret_20d": float(ret_20d) if ret_20d == ret_20d else 0,
                    "stabilized": float(ret_5d) > -0.03 if ret_5d == ret_5d else False,
                }
            except Exception:
                continue
        if not rows:
            logger.info("MultiFactorStrategy 无可用股票")
            return []

        df_factors = pd.DataFrame(rows).set_index("symbol")

        # ── 注入ML因子分（如果权重中包含ml_score）──
        if "ml_score" in active_weights:
            ml_scores = self._compute_ml_scores(df_factors.index.tolist())
            if ml_scores:
                ml_series = pd.Series(ml_scores, name="ml_score")
                df_factors = df_factors.join(ml_series, how="left")
                logger.info(f"ML因子注入：{len(ml_scores)}只股票获得ml_score")

        # 横截面百分位排名（消除量纲）
        valid_factors = [f for f in active_weights if f in df_factors.columns]
        df_rank = cross_section_rank(df_factors[valid_factors])

        # 带符号IC加权合成综合因子分（权重再平衡：防止单类因子主导）
        weights = self._rebalance_weights({f: active_weights[f] for f in valid_factors})
        total_w = sum(abs(w) for w in weights.values())
        if total_w == 0:
            logger.warning("多因子权重全为0，无法选股")
            return []

        df_rank["composite"] = sum(
            df_rank[f].fillna(0) * w for f, w in weights.items()
        ) / total_w

        # ── 趋势确认过滤：排除下降趋势中的股票 ──
        # 纯反转因子会选出"跌多了的股票"，但这些股票经常继续跌。
        # 要求MA20>MA60（中期上升趋势）+ 价格站稳MA20 + 近5日企稳。
        confirmed = {sym for sym, t in trend_info.items()
                     if t["ma20_above_ma60"] and t["above_ma20"] and t["stabilized"]}
        if len(confirmed) < self._TOP_N * 2:
            # 候选不足时放宽
            confirmed = {sym for sym, t in trend_info.items()
                         if t["ma20_above_ma60"] and t["above_ma20"]}
        if len(confirmed) < self._TOP_N:
            confirmed = {sym for sym, t in trend_info.items() if t["above_ma20"]}

        before_filter = len(df_rank)
        df_rank = df_rank[df_rank.index.isin(confirmed)]
        filtered_out = before_filter - len(df_rank)

        # ── 动量修正加分：趋势确认股中，已企稳反弹的优先 ──
        for sym in df_rank.index:
            t = trend_info.get(sym, {})
            ret5 = t.get("ret_5d", 0)
            ret20 = t.get("ret_20d", 0)
            if ret5 > 0 and ret20 > -0.05:
                df_rank.loc[sym, "composite"] += 0.08
            elif ret5 > 0:
                df_rank.loc[sym, "composite"] += 0.04
            if ret20 > 0.05:
                df_rank.loc[sym, "composite"] += 0.03

        # 选Top N
        top = df_rank.nlargest(self._TOP_N, "composite")
        selected = top.index.tolist()

        logger.info(
            f"MultiFactorStrategy 选出 {len(selected)} 只 "
            f"（{len(valid_factors)}因子IC加权，趋势过滤淘汰{filtered_out}只，综合分"
            f"{top['composite'].min():.1f}~{top['composite'].max():.1f}）"
        )
        return selected

    def _compute_ml_scores(self, symbols: list[str]) -> dict[str, float]:
        """读取 ml_scores 月度快照（不实时训练）。

        ML 引擎在 monthly_sweep 中月度训练一次，写全市场预测快照到 ml_scores 表。
        决策层只读当日快照，避免每次决策都重训模型。
        """
        import sqlite3
        try:
            with sqlite3.connect(self.engine.db_path) as conn:
                # 取最新 run_date 的快照
                row = conn.execute(
                    "SELECT MAX(run_date) FROM ml_scores"
                ).fetchone()
                if not row or not row[0]:
                    logger.info("ML因子：无快照（月度训练未运行），跳过注入")
                    return {}
                run_date = row[0]
                rows = conn.execute(
                    "SELECT symbol, ml_score FROM ml_scores WHERE run_date=?",
                    (run_date,),
                ).fetchall()
            scores = {r[0]: float(r[1]) for r in rows
                      if r[1] is not None and r[1] == r[1]}
            logger.info(f"ML因子快照读取：{len(scores)}只（run_date={run_date}）")
            return scores
        except Exception as e:
            logger.warning(f"ML因子快照读取失败（表可能不存在）：{e!r}")
            return {}

    def _filter_universe(self, symbols: list[str]) -> list[str]:
        """选股池硬过滤：剔除ST/退市预警股。"""
        import sqlite3
        with sqlite3.connect(self.engine.db_path) as conn:
            st_rows = conn.execute(
                "SELECT symbol FROM stock_basic "
                "WHERE name LIKE 'ST%' OR name LIKE '%*ST%' OR name LIKE '%退%'"
            ).fetchall()
            st_set = {r[0] for r in st_rows}

        n_before = len(symbols)
        filtered = [s for s in symbols if s not in st_set]
        if n_before != len(filtered):
            logger.info(f"选股池过滤：剔除{n_before - len(filtered)}只ST/退市预警股，剩余{len(filtered)}只")
        return filtered

    def _rebalance_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """权重再平衡：单因子上限 + 大类约束。

        解决IC自动权重过度集中问题：
          - 单因子绝对权重 ≤ 15%（防流动性主导）
          - 流动性大类总权重 ≤ 25%（防僵尸股入选）
          - 质量/估值从流动性节余自然获得更大占比
        """
        LIQ = {"turnover", "amihud", "volume_ratio", "turn_ratio", "liq_rank", "turn_surge"}

        total_abs = sum(abs(w) for w in weights.values())
        if total_abs == 0:
            return weights

        # 单因子上限 15%
        cap = total_abs * 0.15
        adjusted = {}
        for f, w in weights.items():
            adjusted[f] = cap * (1 if w > 0 else -1) if abs(w) > cap else w

        # 流动性大类 ≤ 25%
        liq_wt = sum(abs(adjusted.get(f, 0)) for f in LIQ) / total_abs
        if liq_wt > 0.25:
            shrink = 0.25 / liq_wt
            for f in LIQ:
                if f in adjusted:
                    adjusted[f] *= shrink
            logger.info(f"权重再平衡：流动性{liq_wt*100:.0f}%→25%")

        qual_wt = sum(abs(adjusted.get(f, 0)) for f in ("np_margin", "roe", "gp_margin", "rev_growth", "profit_growth")) / total_abs
        val_wt = sum(abs(adjusted.get(f, 0)) for f in ("pe_ratio", "pb_ratio")) / total_abs

        # 质量估值下限：质量≥15% + 估值≥10%（P0调整：折中值）
        # 质量因子方向稳定、路径风险极低，是降低止损率的关键
        QUAL_FLOOR = 0.15
        VAL_FLOOR = 0.10
        if qual_wt < QUAL_FLOOR:
            for f in ("np_margin", "rev_growth", "gp_margin", "roe", "profit_growth"):
                if f in adjusted and adjusted[f] != 0:
                    adjusted[f] *= QUAL_FLOOR / max(qual_wt, 0.01)
                    break
            logger.info(f"权重再平衡：质量{qual_wt*100:.0f}%→{QUAL_FLOOR*100:.0f}%")
        if val_wt < VAL_FLOOR:
            for f in ("pb_ratio", "pe_ratio"):
                if f in adjusted and adjusted[f] != 0:
                    adjusted[f] *= VAL_FLOOR / max(val_wt, 0.01)
                    break
            logger.info(f"权重再平衡：估值{val_wt*100:.0f}%→{VAL_FLOOR*100:.0f}%")

        logger.info(f"权重分布：流动性{liq_wt*100:.0f}% 质量{qual_wt*100:.0f}% 估值{val_wt*100:.0f}%")
        return adjusted

    def get_factor_weights(self) -> dict[str, float]:
        """返回当前因子权重（供前端展示）。"""
        return dict(self._weights)

    def _detect_market_state(self) -> str:
        """检测当前市场状态（bull/neutral/bear）。

        用全市场近20日等权收益中位数判断：
          >3% bull, <-3% bear, 中间 neutral
        """
        import numpy as np
        try:
            symbols = list(self._shared_daily.keys())[:200] if self._shared_daily else self.engine.get_local_symbols()[:200]
            rets = []
            for sym in symbols:
                df = self.get_daily(sym)
                if df is None or len(df) < 22 or "close" not in df.columns:
                    continue
                r = (df["close"].iloc[-1] / df["close"].iloc[-21] - 1)
                if r == r:  # not NaN
                    rets.append(float(r))
            if len(rets) < 50:
                return "neutral"
            median_ret = float(np.median(rets))
            if median_ret > 0.03:
                state = "bull"
            elif median_ret < -0.03:
                state = "bear"
            else:
                state = "neutral"
            logger.info(f"市场状态检测：{state}（全市场中位20日收益{median_ret*100:+.2f}%）")
            return state
        except Exception as e:
            logger.warning(f"市场状态检测失败，默认neutral：{e!r}")
            return "neutral"

    def _load_finance_map(self, symbols: list[str]) -> dict[str, dict]:
        """批量加载财报数据，返回 {symbol: {roe, np_margin, ...}}。"""
        try:
            import sqlite3
            finance_map = {}
            with sqlite3.connect(self.engine.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT symbol, roe, np_margin, gp_margin, yoy_eps, yoy_pni, "
                    "yoy_ni, asset_turn, inv_turn, nr_turn "
                    "FROM stock_finance WHERE symbol IN ({}) "
                    "AND stat_date = (SELECT MAX(stat_date) FROM stock_finance f2 WHERE f2.symbol = stock_finance.symbol)".format(
                        ",".join("?" * len(symbols))
                    ) if len(symbols) <= 900 else
                    "SELECT symbol, roe, np_margin, gp_margin, yoy_eps, yoy_pni, "
                    "yoy_ni, asset_turn, inv_turn, nr_turn "
                    "FROM stock_finance WHERE stat_date IN (SELECT MAX(stat_date) FROM stock_finance)",
                    symbols if len(symbols) <= 900 else []
                ).fetchall()
                for r in rows:
                    finance_map[r["symbol"]] = dict(r)
            if finance_map:
                logger.info(f"财报数据加载：{len(finance_map)}/{len(symbols)}只")
            return finance_map
        except Exception as e:
            logger.warning(f"加载财报数据失败：{e!r}")
            return {}

    def _load_fund_flow_map(self) -> dict[str, dict]:
        """加载最近一天的主力资金流向，返回 {symbol: {main_net, main_pct}}。"""
        try:
            import sqlite3
            flow_map = {}
            with sqlite3.connect(self.engine.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT symbol, main_net, main_pct, super_net, big_net FROM fund_flow "
                    "WHERE date = (SELECT MAX(date) FROM fund_flow)"
                ).fetchall()
                for r in rows:
                    flow_map[r["symbol"]] = {
                        "main_net": r["main_net"], "main_pct": r["main_pct"],
                        "super_net": r["super_net"], "big_net": r["big_net"],
                    }
            if flow_map:
                logger.info(f"资金流向加载：{len(flow_map)}只")
            return flow_map
        except Exception as e:
            logger.warning(f"加载资金流向失败：{e!r}")
            return {}

    def _load_lhb_map(self) -> dict[str, dict]:
        """加载近30天龙虎榜数据，返回 {symbol: {count, net_buy}}。"""
        try:
            import sqlite3
            lhb_map = {}
            with sqlite3.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, COUNT(*) as cnt, SUM(net_buy) as total_net "
                    "FROM lhb_detail WHERE date >= date('now', '-30 days') "
                    "GROUP BY symbol"
                ).fetchall()
                for r in rows:
                    lhb_map[r[0]] = {"count": r[1], "net_buy": r[2] or 0}
            if lhb_map:
                logger.info(f"龙虎榜数据加载：{len(lhb_map)}只")
            return lhb_map
        except Exception as e:
            logger.warning(f"加载龙虎榜失败：{e!r}")
            return {}

    def _load_north_map(self) -> dict[str, "pd.DataFrame"]:
        """加载北向资金持股历史，返回 {symbol: DataFrame(date, hold_pct)}。

        北向数据高度集中于中大市值，小盘股无数据则不返回（compute_factors 中返回 nan）。
        """
        try:
            import sqlite3
            north_map: dict[str, pd.DataFrame] = {}
            with sqlite3.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, date, hold_pct FROM north_hold "
                    "WHERE hold_pct IS NOT NULL ORDER BY symbol, date"
                ).fetchall()
            tmp: dict[str, list] = {}
            for r in rows:
                tmp.setdefault(r[0], []).append({"date": str(r[1]), "hold_pct": float(r[2])})
            for sym, recs in tmp.items():
                north_map[sym] = pd.DataFrame(recs)
            if north_map:
                logger.info(f"北向持股加载：{len(north_map)}只")
            return north_map
        except Exception as e:
            logger.warning(f"加载北向持股失败（表可能不存在）：{e!r}")
            return {}
