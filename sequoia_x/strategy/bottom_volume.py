"""底部放量策略：超跌后恐慌底放量企稳，左侧反转信号。"""

import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy
from sequoia_x.strategy.registry import register_strategy

logger = get_logger(__name__)


@register_strategy("bottom")
class BottomVolumeStrategy(BaseStrategy):
    """底部放量选股策略。

    填补 sequoia-x 无"底部反转"缺口——现有策略全是右侧（强势多头）选股，
    本策略捕捉超跌后恐慌底放量企稳的反转机会，经典的左侧抄底模式。
    风险较高，仅作信号提示，需配合严格止损（设在近期低点下方）。

    选股条件（全部向量化，严禁 iterrows）：
    1. 持续下跌：近20日高点至今日跌幅 > 15%（超跌确认）
    2. 异动放量：当日 volume > 5日均量 × 3（恐慌底放量）
    3. 阳线企稳：当日 close > open（收阳，买方承接）
    4. 下影线支撑：当日下影线长度 > 实体长度 × 2（买方防守有效）
    5. 流动性：当日 turnover > 100,000,000

    Attributes:
        webhook_key: 路由到 'bottom' 专属飞书机器人。
    """

    _MIN_BARS: int = 20  # 至少需要 20 根 K 线

    def run(self) -> list[str]:
        """遍历全市场，返回满足底部放量条件的股票代码列表。"""
        if self._shared_daily is not None:
            symbols = list(self._shared_daily.keys())
        else:
            symbols = self.engine.get_local_symbols()
        selected: list[str] = []

        for symbol in symbols:
            try:
                df = self.get_daily(symbol)
                if len(df) < self._MIN_BARS:
                    continue

                df["vol_ma5"] = df["volume"].rolling(5).mean()
                df["high_20"] = df["high"].rolling(20).max()

                last = df.iloc[-1]
                if pd.isna(last["vol_ma5"]) or pd.isna(last["high_20"]):
                    continue

                # 条件 1：持续下跌（近20日高点至今跌幅 > 15%）
                drawdown = (last["high_20"] - last["close"]) / last["high_20"] * 100
                oversold = drawdown > 15.0

                # 条件 2：异动放量（当日量 > 5日均量 × 3）
                volume_surge = last["volume"] > last["vol_ma5"] * 3

                # 条件 3：阳线企稳（收阳）
                is_yang = last["close"] > last["open"]

                # 条件 4：下影线支撑（下影线 > 实体 × 2）
                body = abs(last["close"] - last["open"])
                lower_shadow = min(last["open"], last["close"]) - last["low"]
                long_lower_shadow = body > 0 and lower_shadow > body * 2

                # 条件 5：流动性
                liquid = last["turnover"] > 100_000_000

                if oversold and volume_surge and is_yang and long_lower_shadow and liquid:
                    selected.append(symbol)

            except Exception as exc:
                logger.warning(f"[{symbol}] BottomVolumeStrategy 计算失败：{exc}")
                continue

        logger.info(f"BottomVolumeStrategy 选出 {len(selected)} 只股票")
        return selected
