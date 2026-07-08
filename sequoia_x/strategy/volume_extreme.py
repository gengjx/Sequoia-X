"""地量/天量策略：换手率极值择时，A股经典量能信号。

两类信号：
  地量见底：换手率创60日新低 + 价格企稳（缩量跌不动）→ 底部反转
  天量见顶：换手率创20日新高 + 高位滞涨（放量不涨）→ 破位预警

依赖 baostock 的 turn（换手率%）字段。
"""

import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


class VolumeExtremeStrategy(BaseStrategy):
    """地量/天量策略：换手率极值择时。

    Attributes:
        webhook_key: 路由到 'volume_extreme' 专属飞书机器人。
    """

    webhook_key: str = "volume_extreme"
    _MIN_BARS: int = 60  # 至少60根K线（需要60日换手率数据）

    def run(self) -> list[str]:
        """遍历全市场，返回满足地量见底条件的股票代码列表。"""
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

                if "turn" not in df.columns:
                    continue

                turn = df["turn"].astype(float)
                close = df["close"].astype(float)
                last = df.iloc[-1]

                if pd.isna(turn.iloc[-1]):
                    continue

                # ════════ 地量见底信号 ════════
                # 条件1：换手率创60日新低（排除当日）
                turn_min60 = turn.iloc[-61:-1].min()
                if pd.isna(turn_min60) or turn_min60 <= 0:
                    continue
                is_extreme_low = turn.iloc[-1] <= turn_min60 * 1.1  # 10%容差

                # 条件2：价格企稳（近5日跌幅 < 3%，跌不动了）
                if len(close) >= 6:
                    ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
                    is_stable = ret_5d > -3.0
                else:
                    continue

                # 条件3：流动性（成交额 > 5000万，排除僵尸股）
                liquid = last["turnover"] > 50_000_000

                # 条件4：非涨停（排除封死涨停的低换手假信号）
                if "pct_chg" in df.columns and pd.notna(last.get("pct_chg")):
                    not_limit_up = last["pct_chg"] < 9.5
                else:
                    not_limit_up = True

                if is_extreme_low and is_stable and liquid and not_limit_up:
                    selected.append(symbol)

            except Exception:
                continue

        logger.info(f"VolumeExtremeStrategy（地量见底）：选出 {len(selected)} 只")
        return selected
