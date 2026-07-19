"""龙虎榜跟买策略：机构/游资净买入 → T+1跟随，A股短线事件驱动策略。

核心逻辑：
  龙虎榜 = 当日涨跌幅异常的股票（涨停/异动），公开机构席位买卖明细。
  当机构净买入额显著（>5000万），说明聪明资金看好，次日跟随买入。

信号条件：
  1. 最近1天上龙虎榜
  2. 净买入额 > 5000万（机构看好）
  3. 涨幅 < 9.8%（排除已涨停买不进的）
  4. 非ST/退市
  5. 净买入占比 > 5%（主力资金明确流入）

回测限制：龙虎榜历史数据仅24天，无法做5.3年回测。
采用信号验证模式：每日记录→T+1/T+5验证命中率，与竞价验证一致。
"""

from __future__ import annotations

import sqlite3

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy
from sequoia_x.strategy.registry import register_strategy

logger = get_logger(__name__)


@register_strategy("lhb_follow")
class LhbFollowStrategy(BaseStrategy):
    """龙虎榜跟买策略：聪明资金跟随。"""


    # 信号阈值
    MIN_NET_BUY = 5000_0000       # 净买入≥5000万
    MIN_NET_RATIO = 5.0           # 净买入占比≥5%
    MAX_PCT_CHG = 9.8             # 排除已涨停（买不进）
    MIN_PCT_CHG = -3.0            # 排除暴跌（异动卖出）

    def run(self) -> list[str]:
        """选出最近龙虎榜机构净买入的股票。"""
        with sqlite3.connect(self.engine.db_path) as conn:
            # 取最近一个龙虎榜日期
            row = conn.execute(
                "SELECT MAX(date) FROM lhb_detail"
            ).fetchone()
            if not row or not row[0]:
                logger.info("龙虎榜跟买：无龙虎榜数据")
                return []
            latest_date = row[0]

            rows = conn.execute(
                """SELECT symbol, name, net_buy, net_ratio, pct_chg, reason
                   FROM lhb_detail
                   WHERE date = ?
                   AND net_buy > ?
                   AND net_ratio > ?
                   AND pct_chg < ?
                   AND pct_chg > ?
                   AND name NOT LIKE 'ST%'
                   AND name NOT LIKE '%*ST%'
                   AND name NOT LIKE '%退%'
                   ORDER BY net_buy DESC
                   LIMIT 30""",
                (latest_date, self.MIN_NET_BUY, self.MIN_NET_RATIO,
                 self.MAX_PCT_CHG, self.MIN_PCT_CHG)
            ).fetchall()

        # 二次过滤：用日K确认趋势（MA20>MA60，不接飞刀）
        confirmed = []
        for r in rows:
            sym, name, net_buy, net_ratio, pct_chg, reason = r
            df = self.get_daily(sym)
            if df is None or len(df) < 60:
                continue
            close = df["close"]
            ma20 = close.iloc[-20:].mean()
            ma60 = close.iloc[-60:].mean()
            # 趋势确认：价格站稳MA20 + 中期趋势向上
            if close.iloc[-1] > ma20 and ma20 > ma60:
                confirmed.append(sym)

        logger.info(
            f"龙虎榜跟买：{latest_date} 筛选{len(rows)}只 → "
            f"趋势确认{len(confirmed)}只"
        )
        return confirmed
