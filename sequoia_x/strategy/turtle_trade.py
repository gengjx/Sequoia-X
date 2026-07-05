"""海龟交易策略：20日新高突破 + 成交额过亿 + 动量阳线过滤。"""

import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


class TurtleTradeStrategy(BaseStrategy):
    """海龟交易策略（A股防诱多改良版）。

    选股条件（向量化，严禁 iterrows）：
    1. 突破新高：今日 close > 前20个交易日 high 的最大值
    2. 流动性：今日 turnover > 100,000,000
    3. 防诱多过滤：今日必须是实体阳线（今日 close > 今日 open），且必须真涨（今日 close > 昨日 close）

    Attributes:
        webhook_key: 路由到 'turtle' 专属飞书机器人。
    """

    webhook_key: str = "turtle"
    _MIN_BARS: int = 21  # 至少需要 21 根 K 线（20日窗口 + 当日）

    def _get_market_caps(self, symbols: list[str]) -> dict[str, float]:
        """获取候选股票的流通市值（仅用于结果排序，不影响选股逻辑）。

        优先用本地 stock_market_cap 缓存（东财市值，5350只），毫秒级。
        无缓存时回退用最近日成交额(turnover)近似排序——流通市值与成交额
        高度相关，排序结果足够接近。baostock串行查询38s且周末无数据，弃用。
        """
        import sqlite3
        market_caps: dict[str, float] = {}
        if not symbols:
            return market_caps
        placeholders = ",".join("?" * len(symbols))
        try:
            with sqlite3.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    f"SELECT symbol, market_cap FROM stock_market_cap "
                    f"WHERE symbol IN ({placeholders})",
                    symbols,
                ).fetchall()
            for sym, mc in rows:
                if mc:
                    market_caps[sym] = mc
        except Exception:
            pass
        # 缺失的用最近日成交额近似（成交额与市值正相关，排序够用）
        missing = [s for s in symbols if s not in market_caps]
        if missing:
            try:
                with sqlite3.connect(self.engine.db_path) as conn:
                    last_date = conn.execute(
                        "SELECT MAX(date) FROM stock_daily"
                    ).fetchone()[0]
                    if last_date:
                        ph = ",".join("?" * len(missing))
                        rows = conn.execute(
                            f"SELECT symbol, turnover FROM stock_daily "
                            f"WHERE date=? AND symbol IN ({ph})",
                            [last_date] + missing,
                        ).fetchall()
                        for sym, turn in rows:
                            if turn:
                                market_caps[sym] = turn
            except Exception:
                pass
        return market_caps

    def run(self) -> list[str]:
        """
        遍历全市场，返回满足海龟突破条件的股票代码列表。
        """
        symbols = self.engine.get_local_symbols()
        candidates: list[str] = []

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < self._MIN_BARS:
                    continue

                # 向量化：前20日 high 的滚动最大值（不含当日，shift(1) 后取 rolling(20)）
                df["high_20"] = df["high"].shift(1).rolling(20).max()

                last = df.iloc[-1]
                prev = df.iloc[-2]  # 获取昨日数据，用于对比

                if pd.isna(last["high_20"]):
                    continue

                # 核心条件 1：突破前 20 天最高点
                breakout = last["close"] > last["high_20"]
                # 核心条件 2：流动性过亿
                liquid = last["turnover"] > 100_000_000

                # 【新增防守条件】拒绝郑州煤电式的高开低走大阴线！
                is_yang = last["close"] > last["open"]   # 实体必须是阳线（红柱）
                is_up = last["close"] > prev["close"]    # 必须是真涨，不能是假阳线

                if breakout and liquid and is_yang and is_up:
                    candidates.append(symbol)

            except Exception as exc:
                logger.warning(f"[{symbol}] TurtleTradeStrategy 计算失败：{exc}")
                continue

        # 按流通市值从大到小排序
        if candidates:
            market_caps = self._get_market_caps(candidates)
            candidates.sort(key=lambda s: market_caps.get(s, 0), reverse=True)

        logger.info(f"TurtleTradeStrategy 选出 {len(candidates)} 只股票")
        return candidates