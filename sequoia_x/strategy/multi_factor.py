"""多因子选股策略：IC加权合成综合因子分，选全市场Top N。

与规则式策略的区别：
  - 规则式(如均线放量)：固定条件筛选(布尔)，选出的票"全等"
  - 多因子：所有股票统一打分排序(连续值)，选出Top N，区分度更强

因子权重来自IC评估（有效因子按IC加权），定期重评估自动更新。
无效因子(IC接近0)自动剔除，避免噪音。
"""

import numpy as np
import pandas as pd

from sequoia_x.analysis.factor import (
    compute_factors,
    cross_section_rank,
    _ml_stability_penalty,
    _recent_ml_ic_mean,
)
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




def _pit_asof(seq: list[tuple], as_of_date: str) -> dict | None:
    """从已排序的 [(date, {fields}), ...] 取 ≤ as_of_date 的最新一条 fields。

    回测 PIT as-of 查找：杜绝未来函数。复用 evaluate_factor_ic 验证过的模式。
    """
    if not seq:
        return None
    import bisect
    dates_list = [d for d, _ in seq]
    idx = bisect.bisect_right(dates_list, as_of_date) - 1
    return seq[idx][1] if idx >= 0 else None

@register_strategy("multi_factor")
class MultiFactorStrategy(BaseStrategy):
    """多因子IC加权选股策略。

    选股逻辑：

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
        # 回测注入的 as-of ML 快照（{symbol: score}）；None=读最新 run_date 快照
        self._ml_scores_asof: dict[str, float] | None = None
        # 回测 PIT 快照历史（preload_snapshot_history 一次性加载）；None=实盘走最新值
        self._snapshot_history: dict | None = None

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

    def preload_snapshot_history(self) -> None:
        """一次性加载全部快照因子的完整历史（回测 PIT 模式）。

        复用 evaluate_factor_ic 验证过的 {symbol: [(date, {fields})]} 排序结构。
        加载后 run(as_of_date=...) 走 as-of 截断路径；不调用则 run() 走实盘最新值。
        """
        import sqlite3
        sh: dict[str, dict] = {}
        db = self.engine.db_path
        with sqlite3.connect(db) as conn:
            # 财报（按 stat_date 排序）
            try:
                rows = conn.execute(
                    "SELECT symbol, stat_date, roe, np_margin, gp_margin, yoy_eps, yoy_pni, "
                    "yoy_ni, asset_turn, inv_turn, nr_turn, cfo_to_or, cfo_to_np "
                    "FROM stock_finance ORDER BY symbol, stat_date"
                ).fetchall()
                sh["finance"] = {}
                for r in rows:
                    sh["finance"].setdefault(r[0], []).append((r[1], {
                        "roe": r[2], "np_margin": r[3], "gp_margin": r[4],
                        "yoy_eps": r[5], "yoy_pni": r[6], "yoy_ni": r[7],
                        "asset_turn": r[8], "inv_turn": r[9], "nr_turn": r[10],
                        "cfo_to_or": r[11], "cfo_to_np": r[12],
                    }))
            except Exception as e:
                sh["finance"] = {}
            # 资金流向
            try:
                rows = conn.execute(
                    "SELECT symbol, date, main_net, main_pct, super_net, big_net "
                    "FROM fund_flow ORDER BY symbol, date"
                ).fetchall()
                sh["fund_flow"] = {}
                for r in rows:
                    sh["fund_flow"].setdefault(r[0], []).append((r[1], {
                        "main_net": r[2], "main_pct": r[3], "super_net": r[4], "big_net": r[5],
                    }))
            except Exception:
                sh["fund_flow"] = {}
            # 龙虎榜
            try:
                rows = conn.execute(
                    "SELECT symbol, date, net_buy FROM lhb_detail ORDER BY symbol, date"
                ).fetchall()
                sh["lhb"] = {}
                for r in rows:
                    sh["lhb"].setdefault(r[0], []).append((r[1], float(r[2]) if r[2] else 0.0))
            except Exception:
                sh["lhb"] = {}
            # 北向持股（全序列，按 date 排序）
            try:
                rows = conn.execute(
                    "SELECT symbol, date, hold_pct FROM north_hold "
                    "WHERE hold_pct IS NOT NULL ORDER BY symbol, date"
                ).fetchall()
                sh["north"] = {}
                tmp: dict[str, list] = {}
                for r in rows:
                    tmp.setdefault(r[0], []).append({"date": str(r[1]), "hold_pct": float(r[2])})
                for sym, recs in tmp.items():
                    sh["north"][sym] = recs  # list[dict]，as-of 时取 <=today 前缀
            except Exception:
                sh["north"] = {}
            # 融资融券
            try:
                rows = conn.execute(
                    "SELECT symbol, date, rzye, rzbuy, rqlts, rqye "
                    "FROM margin_detail ORDER BY symbol, date"
                ).fetchall()
                sh["margin"] = {}
                for r in rows:
                    sh["margin"].setdefault(r[0], []).append((r[1], {
                        "rzye": r[2], "rzbuy": r[3], "rqlts": r[4], "rqye": r[5],
                    }))
            except Exception:
                sh["margin"] = {}
            # 基金持仓
            try:
                rows = conn.execute(
                    "SELECT symbol, report_date, fund_count, change_pct "
                    "FROM fund_hold ORDER BY symbol, report_date"
                ).fetchall()
                sh["fund_hold"] = {}
                for r in rows:
                    sh["fund_hold"].setdefault(r[0], []).append((r[1], {
                        "fund_count": r[2], "change_pct": r[3],
                    }))
            except Exception:
                sh["fund_hold"] = {}
            # 大宗交易（折溢率，近30天窗口）
            try:
                rows = conn.execute(
                    "SELECT symbol, date, discount, amount FROM block_trade ORDER BY symbol, date"
                ).fetchall()
                sh["block"] = {}
                for r in rows:
                    sh["block"].setdefault(r[0], []).append((r[1], {
                        "discount": r[2], "amount": r[3],
                    }))
            except Exception:
                sh["block"] = {}
            # 股东户数（季频，筹码集中度）
            try:
                rows = conn.execute(
                    "SELECT symbol, end_date, holder_num, holder_change, avg_value "
                    "FROM holder_count ORDER BY symbol, end_date"
                ).fetchall()
                sh["holder"] = {}
                for r in rows:
                    sh["holder"].setdefault(r[0], []).append((r[1], {
                        "holder_num": r[2], "holder_change": r[3], "avg_value": r[4],
                    }))
            except Exception:
                sh["holder"] = {}
            # 沪深300指数收益率（全序列）
            try:
                rows = conn.execute(
                    "SELECT date, close FROM index_daily WHERE symbol='000300' ORDER BY date"
                ).fetchall()
                if len(rows) >= 60:
                    _df = pd.DataFrame(rows, columns=["date", "close"])
                    _df["date"] = _df["date"].astype(str)
                    _ret = _df["close"].pct_change()
                    sh["index_ret"] = pd.Series(_ret.values, index=_df["date"].values).dropna()
                else:
                    sh["index_ret"] = None
            except Exception:
                sh["index_ret"] = None
            # 宏观（M2/社融，按 month 排序）
            try:
                sh["macro_m2"] = [(r[0], r[1]) for r in conn.execute(
                    "SELECT month, m2_yoy FROM macro_money ORDER BY month"
                ) if r[1] is not None]
                sh["macro_sf"] = [(r[0], r[1]) for r in conn.execute(
                    "SELECT month, sf_total FROM macro_sf ORDER BY month"
                ) if r[1] is not None]
            except Exception:
                sh["macro_m2"] = []
                sh["macro_sf"] = []
        self._snapshot_history = sh
        logger.info(
            f"快照历史预加载完成：财报{len(sh.get('finance',{}))} 资金流{len(sh.get('fund_flow',{}))} "
            f"龙虎榜{len(sh.get('lhb',{}))} 北向{len(sh.get('north',{}))} 融资融券{len(sh.get('margin',{}))} "
            f"基金持仓{len(sh.get('fund_hold',{}))}"
        )

    def run(self, as_of_date: str | None = None) -> list[str]:
        """执行多因子选股，返回综合因子分Top N的股票代码。

        Args:
            as_of_date: PIT 截止日期（YYYY-MM-DD）。None=实盘走最新快照；
                        非 None（回测）时用 preload_snapshot_history 预加载的
                        历史按 as-of 截断，杜绝未来函数。
        """
        symbols = list(self._shared_daily.keys()) if self._shared_daily else self.engine.get_local_symbols()

        # ── 选股池硬过滤：ST/退市预警 ──
        symbols = self._filter_universe(symbols)

        # P1: 市场状态自适应——根据当前市场状态选择对应权重
        market_state = self._detect_market_state(as_of_date=as_of_date)
        active_weights = self.get_weights_for_state(market_state)
        # ML 合成因子是市场状态无关的全局信号，三态权重表可能不含，
        # 从全局权重补充注入（达标时非零、未达标为0则不注入）
        # P10b: 按近期样本外 IC 稳定性施加软惩罚（as-of PIT）——
        # 强周期满权重保 alpha，衰退期自动降至下限抑制噪声。
        base_ml = self._weights.get("ml_score")
        if base_ml:
            if not hasattr(self, "_ml_ic_cache"):
                self._ml_ic_cache = {}
            cache_key = as_of_date or "__live__"
            recent_ic = self._ml_ic_cache.get(cache_key, "__miss__")
            if recent_ic == "__miss__":
                recent_ic = _recent_ml_ic_mean(self.engine.db_path, as_of_date=as_of_date)
                self._ml_ic_cache[cache_key] = recent_ic
            penalty = _ml_stability_penalty(recent_ic) if recent_ic is not None else 1.0
            active_weights["ml_score"] = round(base_ml * penalty, 4)

        # ── 快照因子加载：回测走 as-of 截断（杜绝未来函数），实盘走最新值 ──
        if as_of_date and self._snapshot_history:
            sh = self._snapshot_history
            finance_map = self._build_asof_finance(symbols, as_of_date)
            fund_flow_map = self._build_asof_fund_flow(as_of_date)
            lhb_map = self._build_asof_lhb(as_of_date)
            north_map = self._build_asof_north(as_of_date)
            margin_map = self._build_asof_margin(as_of_date)
            fund_hold_map = self._build_asof_fund_hold(as_of_date)
            block_map = self._build_asof_block(as_of_date)
            holder_map = self._build_asof_holder(as_of_date)
            index_ret = sh.get("index_ret")
            if index_ret is not None:
                index_ret = index_ret[index_ret.index <= as_of_date]
        else:
            # 实盘路径：读最新快照（零改动）
            finance_map = self._load_finance_map(symbols)
            fund_flow_map = self._load_fund_flow_map()
            lhb_map = self._load_lhb_map()
            north_map = self._load_north_map()
            margin_map = self._load_margin_map()
            fund_hold_map = self._load_fund_hold_map()
            block_map = self._load_block_map()
            holder_map = self._load_holder_map()
            index_ret = self._load_index_ret()

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
                    margin=margin_map.get(sym),
                    fund_hold=fund_hold_map.get(sym),
                    block_data=block_map.get(sym),
                    holder_data=holder_map.get(sym),
                    index_ret=index_ret,
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
        """读取 ML 因子月度快照（不实时训练）。

        优先级：回测注入的 as-of 快照（_ml_scores_asof）> DB 最新 run_date 快照。
        回测注入确保"严禁未来函数"——回测日 D 的 ML 预测来自 ≤ D 的训练。
        """
        # 回测 as-of 注入优先（杜绝未来函数）
        if self._ml_scores_asof is not None:
            scores = {s: float(v) for s, v in self._ml_scores_asof.items()
                      if v is not None and v == v}
            return scores

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

    def _load_margin_map(self) -> dict[str, dict]:
        """加载最新日融资融券明细。返回 {symbol: {rzye, rzbuy, rqlts, rqye}}。"""
        try:
            import sqlite3
            with sqlite3.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, rzye, rzbuy, rqlts, rqye FROM margin_detail "
                    "WHERE date=(SELECT MAX(date) FROM margin_detail)"
                ).fetchall()
            result = {}
            for r in rows:
                result[r[0]] = {"rzye": r[1], "rzbuy": r[2], "rqlts": r[3], "rqye": r[4]}
            return result
        except Exception as e:
            logger.debug(f"融资融券加载失败: {e}")
            return {}

    def _load_fund_hold_map(self) -> dict[str, dict]:
        """加载最新季度基金持仓。返回 {symbol: {fund_count, change_pct}}。"""
        try:
            import sqlite3
            with sqlite3.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, fund_count, change_pct FROM fund_hold "
                    "WHERE report_date=(SELECT MAX(report_date) FROM fund_hold)"
                ).fetchall()
            result = {}
            for r in rows:
                result[r[0]] = {"fund_count": r[1], "change_pct": r[2]}
            return result
        except Exception as e:
            logger.debug(f"基金持仓加载失败: {e}")
            return {}

    def _load_index_ret(self) -> "pd.Series | None":
        """加载沪深300指数收益率序列（用于 beta_300 / rel_strength_300）。"""
        try:
            import sqlite3
            with sqlite3.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT date, close FROM index_daily "
                    "WHERE symbol='000300' ORDER BY date"
                ).fetchall()
            if len(rows) < 60:
                return None
            s = pd.DataFrame(rows, columns=["date", "close"])
            rets = s["close"].pct_change()
            series = pd.Series(rets.values, index=s["date"].astype(str).values).dropna()
            return series
        except Exception as e:
            logger.debug(f"沪深300指数加载失败: {e}")
            return None


    # ───────────────────────────────────────────────────────────────
    # PIT as-of 快照构建（回测专用，从 _snapshot_history 截断）
    # ───────────────────────────────────────────────────────────────

    def _build_asof_finance(self, symbols: list[str], as_of_date: str) -> dict[str, dict]:
        """从预加载财报历史按 as-of 截断，返回 {symbol: {roe,...}}。"""
        seq_map = self._snapshot_history.get("finance", {})
        result: dict[str, dict] = {}
        for sym in symbols:
            fields = _pit_asof(seq_map.get(sym, []), as_of_date)
            if fields:
                result[sym] = fields
        return result

    def _build_asof_fund_flow(self, as_of_date: str) -> dict[str, dict]:
        """资金流向 as-of：取 ≤ today 最新一条。"""
        seq_map = self._snapshot_history.get("fund_flow", {})
        result: dict[str, dict] = {}
        for sym, seq in seq_map.items():
            fields = _pit_asof(seq, as_of_date)
            if fields:
                result[sym] = fields
        return result

    def _build_asof_lhb(self, as_of_date: str) -> dict[str, dict]:
        """龙虎榜 as-of：(today-30, today] 窗口的 count + net_buy 之和。"""
        import datetime as _dt
        seq_map = self._snapshot_history.get("lhb", {})
        result: dict[str, dict] = {}
        try:
            ref = _dt.datetime.strptime(as_of_date, "%Y-%m-%d")
            ws = (ref - _dt.timedelta(days=30)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            ws = "1900-01-01"
        for sym, seq in seq_map.items():
            window = [(d, nb) for d, nb in seq if ws <= d <= as_of_date]
            if window:
                result[sym] = {"count": float(len(window)),
                               "net_buy": float(sum(nb for _, nb in window))}
        return result

    def _build_asof_north(self, as_of_date: str) -> dict[str, "pd.DataFrame"]:
        """北向持股 as-of：取 ≤ today 全序列（nb_inflow 需 21 行历史）。"""
        raw = self._snapshot_history.get("north", {})
        result: dict[str, pd.DataFrame] = {}
        for sym, recs in raw.items():
            filt = [r for r in recs if r["date"] <= as_of_date]
            if filt:
                result[sym] = pd.DataFrame(filt)
        return result

    def _build_asof_margin(self, as_of_date: str) -> dict[str, dict]:
        """融资融券 as-of：取 ≤ today 最新一条。"""
        seq_map = self._snapshot_history.get("margin", {})
        result: dict[str, dict] = {}
        for sym, seq in seq_map.items():
            fields = _pit_asof(seq, as_of_date)
            if fields:
                result[sym] = fields
        return result

    def _load_holder_map(self) -> dict[str, dict]:
        """加载最新季度股东户数（实盘）。返回 {symbol: {holder_num, holder_change, avg_value}}。"""
        import sqlite3
        try:
            with sqlite3.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, holder_num, holder_change, avg_value FROM holder_count "
                    "WHERE end_date=(SELECT MAX(end_date) FROM holder_count)"
                ).fetchall()
            return {r[0]: {"holder_num": r[1], "holder_change": r[2], "avg_value": r[3]}
                    for r in rows}
        except Exception as e:
            logger.debug(f"股东户数加载失败: {e}")
            return {}

    def _build_asof_holder(self, as_of_date: str) -> dict[str, dict]:
        """股东户数 as-of：取 ≤ today 最新季度快照（季频 PIT）。"""
        seq_map = self._snapshot_history.get("holder", {})
        result = {}
        for sym, seq in seq_map.items():
            fields = _pit_asof(seq, as_of_date)
            if fields:
                result[sym] = fields
        return result

    def _load_block_map(self) -> dict[str, dict]:
        """加载近30天大宗交易（实盘最新日）。返回 {symbol: {discount, count}}。"""
        import sqlite3
        try:
            with sqlite3.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, date, discount, amount FROM block_trade "
                    "WHERE date >= date('now', '-30 days') ORDER BY symbol, date"
                ).fetchall()
            result = {}
            tmp = {}
            for r in rows:
                tmp.setdefault(r[0], []).append((r[2] or 0, r[3] or 0))
            for sym, recs in tmp.items():
                total_amt = sum(amt for _, amt in recs if amt and amt > 0)
                if total_amt > 0:
                    disc = sum(d * (amt if amt and amt > 0 else 0) for d, amt in recs) / total_amt
                else:
                    disc = sum(d for d, _ in recs) / len(recs) if recs else 0
                result[sym] = {"discount": disc, "count": len(recs)}
            return result
        except Exception as e:
            logger.debug(f"大宗交易加载失败: {e}")
            return {}

    def _build_asof_block(self, as_of_date: str) -> dict[str, dict]:
        """大宗交易 as-of：取 ≤ today 近30天加权折价率。"""
        import datetime as _dt
        seq_map = self._snapshot_history.get("block", {})
        result = {}
        cutoff = (_dt.datetime.strptime(as_of_date, "%Y-%m-%d") - _dt.timedelta(days=30)).strftime("%Y-%m-%d")
        for sym, seq in seq_map.items():
            recent = [(rec[0], rec[1].get("discount", 0), rec[1].get("amount", 0))
                      for rec in seq if cutoff <= rec[0] <= as_of_date]
            if not recent:
                continue
            total_amt = sum(amt for _, _, amt in recent if amt and amt > 0)
            if total_amt > 0:
                disc = sum(d * (amt if amt and amt > 0 else 0) for _, d, amt in recent) / total_amt
            else:
                disc = sum(d for _, d, _ in recent) / len(recent)
            result[sym] = {"discount": disc, "count": len(recent)}
        return result

    def _build_asof_fund_hold(self, as_of_date: str) -> dict[str, dict]:
        """基金持仓 as-of：取 report_date ≤ today 最新季度。"""
        seq_map = self._snapshot_history.get("fund_hold", {})
        result: dict[str, dict] = {}
        for sym, seq in seq_map.items():
            fields = _pit_asof(seq, as_of_date)
            if fields:
                result[sym] = fields
        return result

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

    def _detect_market_state(self, as_of_date: str | None = None) -> str:
        """检测当前市场状态（bull/neutral/bear）。

        用全市场近20日等权收益中位数判断：
          >3% bull, <-3% bear, 中间 neutral

        P8: 宏观流动性（M2/社融）作为辅助确认层，对阈值做方向性偏置：
          M2同比>9%或社融放量→放宽bull门槛（更易确认多头）；
          M2同比<7%或社融收缩→收紧bear门槛（更易确认空头）。
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
            # 宏观流动性偏置（放宽/收紧状态门槛）
            macro_bias = self._load_macro_bias(as_of_date=as_of_date)
            bull_thr = 0.02 if macro_bias == "bull" else 0.03
            bear_thr = -0.02 if macro_bias == "bear" else -0.03
            if median_ret > bull_thr:
                state = "bull"
            elif median_ret < bear_thr:
                state = "bear"
            else:
                state = "neutral"
            logger.info(f"市场状态检测：{state}（中位20日收益{median_ret*100:+.2f}%, 宏观偏置={macro_bias}）")
            return state
        except Exception as e:
            logger.warning(f"市场状态检测失败，默认neutral：{e!r}")
            return "neutral"

    def _load_macro_bias(self, as_of_date: str | None = None) -> str | None:
        """加载宏观流动性偏置（M2/社融），返回 'bull'/'bear'/None。

        as_of_date 非 None（回测）时从预加载历史按 PIT 取 ≤ as_of_date 最新值。
        """
        bias = None
        try:
            if as_of_date and self._snapshot_history:
                sh = self._snapshot_history
                m2_seq = sh.get("macro_m2", [])
                sf_seq = sh.get("macro_sf", [])
                # M2: 取 month <= as_of_date[:7] 最新
                m2_month = as_of_date[:7]
                m2_valid = [(m, v) for m, v in m2_seq if m <= m2_month]
                m2_yoy = m2_valid[-1][1] if m2_valid else None
                # 社融：取最近两个月 <= as_of_date
                sf_month = as_of_date[:7]
                sf_valid = [(m, v) for m, v in sf_seq if m <= sf_month]
                sf_rows = sf_valid[-2:] if len(sf_valid) >= 2 else sf_valid
            else:
                import sqlite3
                with sqlite3.connect(self.engine.db_path) as conn:
                    m2_row = conn.execute(
                        "SELECT m2_yoy FROM macro_money ORDER BY month DESC LIMIT 1"
                    ).fetchone()
                    sf_rows = conn.execute(
                        "SELECT sf_total FROM macro_sf ORDER BY month DESC LIMIT 2"
                    ).fetchall()
                    sf_rows = [(None, r[0]) for r in sf_rows]  # 对齐结构
                m2_yoy = m2_row[0] if m2_row else None
            if m2_yoy is not None and m2_yoy > 9.0:
                bias = "bull"
            elif m2_yoy is not None and m2_yoy < 7.0:
                bias = "bear"
            # 社融环比：最近月 vs 上月，放量→bull、收缩→bear
            # sf_vals = [最新月, 上月]（无论哪个分支，最新在前）
            if len(sf_rows) >= 2:
                # as-of 路径是升序（最新在 -1），DB 路径是降序（最新在 0）
                if as_of_date and self._snapshot_history:
                    cur_sf, prev_sf = sf_rows[-1][1], sf_rows[-2][1]
                else:
                    cur_sf, prev_sf = sf_rows[0][1], sf_rows[1][1]
                if cur_sf is not None and prev_sf is not None and prev_sf != 0:
                    if cur_sf > prev_sf * 1.2:
                        bias = bias or "bull"
                    elif cur_sf < prev_sf * 0.8:
                        bias = bias or "bear"
            return bias
        except Exception as e:
            logger.debug(f"宏观偏置加载失败: {e}")
            return None

    def _load_finance_map(self, symbols: list[str]) -> dict[str, dict]:
        """批量加载财报数据，返回 {symbol: {roe, np_margin, ...}}。"""
        try:
            import sqlite3
            finance_map = {}
            with sqlite3.connect(self.engine.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT symbol, roe, np_margin, gp_margin, yoy_eps, yoy_pni, "
                    "yoy_ni, asset_turn, inv_turn, nr_turn, cfo_to_or, cfo_to_np "
                    "FROM stock_finance WHERE symbol IN ({}) "
                    "AND stat_date = (SELECT MAX(stat_date) FROM stock_finance f2 WHERE f2.symbol = stock_finance.symbol)".format(
                        ",".join("?" * len(symbols))
                    ) if len(symbols) <= 900 else
                    "SELECT symbol, roe, np_margin, gp_margin, yoy_eps, yoy_pni, "
                    "yoy_ni, asset_turn, inv_turn, nr_turn, cfo_to_or, cfo_to_np "
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
