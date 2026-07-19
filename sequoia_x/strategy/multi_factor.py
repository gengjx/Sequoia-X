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

        # P1: 市场状态自适应——根据当前市场状态选择对应权重
        market_state = self._detect_market_state()
        active_weights = self.get_weights_for_state(market_state)

        # 预加载财报数据（批量查一次，避免逐只查库）
        finance_map = self._load_finance_map(symbols)
        # 预加载资金流向（最近一天）
        fund_flow_map = self._load_fund_flow_map()
        # 预加载龙虎榜（近30天上榜次数+净买入额）
        lhb_map = self._load_lhb_map()

        # 采集全市场因子截面（含质量因子+资金因子）
        rows = []
        for sym in symbols:
            df = self.get_daily(sym)
            if df is None or len(df) < 20:
                continue
            try:
                factors = compute_factors(
                    df,
                    finance=finance_map.get(sym),
                    fund_flow=fund_flow_map.get(sym),
                    lhb_data=lhb_map.get(sym),
                )
                factors["symbol"] = sym
                rows.append(factors)
            except Exception:
                continue
        if not rows:
            logger.info("MultiFactorStrategy 无可用股票")
            return []

        df_factors = pd.DataFrame(rows).set_index("symbol")

        # 横截面百分位排名（消除量纲）
        valid_factors = [f for f in active_weights if f in df_factors.columns]
        df_rank = cross_section_rank(df_factors[valid_factors])

        # 带符号IC加权合成综合因子分
        weights = {f: active_weights[f] for f in valid_factors}
        total_w = sum(abs(w) for w in weights.values())
        if total_w == 0:
            logger.warning("多因子权重全为0，无法选股")
            return []

        df_rank["composite"] = sum(
            df_rank[f] * w for f, w in weights.items()
        ) / total_w

        # 选Top N
        top = df_rank.nlargest(self._TOP_N, "composite")
        selected = top.index.tolist()

        logger.info(
            f"MultiFactorStrategy 选出 {len(selected)} 只 "
            f"（{len(valid_factors)}因子IC加权，综合分"
            f"{top['composite'].min():.1f}~{top['composite'].max():.1f}）"
        )
        return selected

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
                    "SELECT symbol, roe, np_margin, gp_margin, yoy_eps, yoy_pni "
                    "FROM stock_finance WHERE symbol IN ({}) "
                    "AND stat_date = (SELECT MAX(stat_date) FROM stock_finance f2 WHERE f2.symbol = stock_finance.symbol)".format(
                        ",".join("?" * len(symbols))
                    ) if len(symbols) <= 900 else
                    "SELECT symbol, roe, np_margin, gp_margin, yoy_eps, yoy_pni "
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
                    "SELECT symbol, main_net, main_pct FROM fund_flow "
                    "WHERE date = (SELECT MAX(date) FROM fund_flow)"
                ).fetchall()
                for r in rows:
                    flow_map[r["symbol"]] = {"main_net": r["main_net"], "main_pct": r["main_pct"]}
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
