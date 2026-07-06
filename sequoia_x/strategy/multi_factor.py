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

    webhook_key: str = "multi_factor"
    _TOP_N: int = 50  # 选入数量（与规则式策略平均选股量可比）

    def __init__(self, engine: DataEngine, settings: Settings,
                 factor_weights: dict[str, float] | None = None) -> None:
        super().__init__(engine, settings)
        # 权重优先级：外部注入 > DB动态加载 > 默认硬编码兜底
        if factor_weights:
            self._weights = factor_weights
        else:
            self._weights = self._load_db_weights()

    def _load_db_weights(self) -> dict[str, float]:
        """从DB加载最新因子IC权重，DB空则用默认值兜底。

        IC评估引擎每次运行会写回DB，使选股自适应最新市场数据。
        """
        try:
            db_weights = self.engine.load_factor_weights()
            if db_weights:
                weights = {k: v["weight"] for k, v in db_weights.items()}
                logger.info(
                    f"因子权重已从DB加载（{len(weights)}个因子，"
                    f"最近更新：{next(iter(db_weights.values())).get('updated_at', '?')}）"
                )
                return weights
        except Exception as e:
            logger.warning(f"加载因子权重失败，使用默认值：{e!r}")
        return dict(_DEFAULT_FACTOR_WEIGHTS)

    def run(self) -> list[str]:
        """执行多因子选股，返回综合因子分Top N的股票代码。"""
        symbols = list(self._shared_daily.keys()) if self._shared_daily else self.engine.get_local_symbols()

        # 采集全市场因子截面
        rows = []
        for sym in symbols:
            df = self.get_daily(sym)
            if df is None or len(df) < 20:
                continue
            try:
                factors = compute_factors(df)
                factors["symbol"] = sym
                rows.append(factors)
            except Exception:
                continue
        if not rows:
            logger.info("MultiFactorStrategy 无可用股票")
            return []

        df_factors = pd.DataFrame(rows).set_index("symbol")

        # 横截面百分位排名（消除量纲）
        valid_factors = [f for f in self._weights if f in df_factors.columns]
        df_rank = cross_section_rank(df_factors[valid_factors])

        # IC加权合成综合因子分
        weights = {f: self._weights[f] for f in valid_factors}
        total_w = sum(weights.values())
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
