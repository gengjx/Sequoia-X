"""缩量回踩策略：上升趋势中回踩均线支撑 + 缩量企稳，右侧低吸买点。"""

import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


class ShrinkPullbackStrategy(BaseStrategy):
    """缩量回踩选股策略。

    填补 sequoia-x 无"回踩买点"缺口——现有策略全是突破/启动信号，
    本策略捕捉上升趋势中价格回踩均线、量能萎缩抛压衰竭后的企稳反弹机会，
    实战中是胜率最高的右侧低吸入场点。

    选股条件（全部向量化，严禁 iterrows）：
    1. 多头排列：MA5 > MA10 > MA20（趋势确认）
    2. 回踩支撑：close 距 MA5 ≤ 1.5% 或距 MA10 ≤ 3%（触及均线支撑）
    3. 量能萎缩：近3日 volume 均值 < 5日 volume 均值 × 0.7（抛压衰竭）
    4. 支撑有效：当日 low ≥ MA10（未跌破均线支撑）
    5. 流动性：当日 turnover > 100,000,000

    Attributes:
        webhook_key: 路由到 'pullback' 专属飞书机器人。
    """

    webhook_key: str = "pullback"
    _MIN_BARS: int = 20  # 至少需要 20 根 K 线（MA20 + 5日量窗）

    def run(self) -> list[str]:
        """遍历全市场，返回满足缩量回踩条件的股票代码列表。"""
        symbols = self.engine.get_local_symbols()
        selected: list[str] = []

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < self._MIN_BARS:
                    continue

                # 向量化计算均线与成交量均值
                df["ma5"] = df["close"].rolling(5).mean()
                df["ma10"] = df["close"].rolling(10).mean()
                df["ma20"] = df["close"].rolling(20).mean()
                df["vol_ma5"] = df["volume"].rolling(5).mean()

                last = df.iloc[-1]
                if pd.isna(last["ma20"]) or pd.isna(last["vol_ma5"]):
                    continue

                # 条件 1：多头排列（MA5 > MA10 > MA20）
                bullish = last["ma5"] > last["ma10"] > last["ma20"]

                # 条件 2：回踩支撑（close 接近 MA5 或 MA10）
                near_ma5 = abs(last["close"] - last["ma5"]) / last["ma5"] <= 0.015
                near_ma10 = abs(last["close"] - last["ma10"]) / last["ma10"] <= 0.03
                pullback = near_ma5 or near_ma10

                # 条件 3：量能萎缩（近3日均量 < 5日均量 × 0.7）
                recent_vol = df["volume"].tail(3).mean()
                shrink = recent_vol < last["vol_ma5"] * 0.7

                # 条件 4：支撑有效（当日 low 未跌破 MA10）
                support_held = last["low"] >= last["ma10"]

                # 条件 5：流动性（成交额过亿）
                liquid = last["turnover"] > 100_000_000

                if bullish and pullback and shrink and support_held and liquid:
                    selected.append(symbol)

            except Exception as exc:
                logger.warning(f"[{symbol}] ShrinkPullbackStrategy 计算失败：{exc}")
                continue

        logger.info(f"ShrinkPullbackStrategy 选出 {len(selected)} 只股票")
        return selected
