"""板块轮动策略：买入近期最强板块的龙头股，A股中期趋势跟随策略。

核心逻辑：
  A股有明显的板块轮动效应：资金在不同板块间切换，持续2-4周。
  策略计算每个板块的20日加权平均涨幅，选出Top板块中的最强个股。

信号条件：
  1. 按板块分组计算20日加权涨幅（市值加权）
  2. 选出Top5强势板块
  3. 每个板块内选出涨幅最强+成交活跃的龙头
  4. 趋势确认：MA20>MA60 + 价格站稳MA20

与dragon_head的区别：
  dragon_head用东财实时板块排名（需网络请求，当天数据）
  sector_rotation用本地日K历史计算（可回测，无网络依赖）
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy
from sequoia_x.strategy.registry import register_strategy

logger = get_logger(__name__)


@register_strategy("sector_rotation")
class SectorRotationStrategy(BaseStrategy):
    """板块轮动策略：强势板块+龙头个股。"""


    TOP_SECTORS = 5           # 选前5个强势板块
    PER_SECTOR = 3            # 每板块选3只龙头
    LOOKBACK = 20             # 20日板块动量
    MIN_TURNOVER = 1e8        # 日均成交额≥1亿

    def run(self) -> list[str]:
        """选出强势板块中的龙头个股。"""
        # 加载板块映射
        with sqlite3.connect(self.engine.db_path) as conn:
            board_rows = conn.execute(
                "SELECT symbol, board FROM stock_board_em"
            ).fetchall()
            if not board_rows:
                logger.info("板块轮动：无板块映射数据")
                return []
            board_map = {r[0]: r[1] for r in board_rows if r[1]}

        # 计算每只股票的20日涨幅 + 最新成交额
        stock_data = []
        for sym, board in board_map.items():
            df = self.get_daily(sym)
            if df is None or len(df) < 60:
                continue
            close = df["close"]
            if len(close) < self.LOOKBACK + 1:
                continue

            ret_20d = close.iloc[-1] / close.iloc[-(self.LOOKBACK + 1)] - 1
            ma20 = close.iloc[-20:].mean()
            ma60 = close.iloc[-60:].mean()
            avg_turnover = df["turnover"].iloc[-5:].mean() if "turnover" in df.columns else 0

            # 趋势确认
            above_ma20 = close.iloc[-1] > ma20
            ma20_above_ma60 = ma20 > ma60

            stock_data.append({
                "symbol": sym, "board": board,
                "ret_20d": float(ret_20d) if ret_20d == ret_20d else 0,
                "avg_turnover": float(avg_turnover) if avg_turnover == avg_turnover else 0,
                "above_ma20": above_ma20,
                "ma20_above_ma60": ma20_above_ma60,
            })

        if len(stock_data) < 100:
            logger.info(f"板块轮动：有效股票不足({len(stock_data)})")
            return []

        df_stocks = pd.DataFrame(stock_data)

        # ── Step 1: 板块动量排名 ──
        # 按板块中位数涨幅排名（中位数比均值更抗极端值）
        sector_ret = df_stocks.groupby("board")["ret_20d"].agg(["median", "count"])
        sector_ret = sector_ret[sector_ret["count"] >= 3]  # 板块至少3只成分股
        sector_ret = sector_ret.sort_values("median", ascending=False)

        top_sectors = sector_ret.head(self.TOP_SECTORS).index.tolist()
        top3_str = ", ".join(f"{s}({sector_ret.loc[s, 'median']*100:.1f}%)" for s in top_sectors[:3])
        logger.info(f"板块轮动：Top{self.TOP_SECTORS}板块 {top3_str}")

        # ── Step 2: 每板块选龙头 ──
        selected = []
        for sector in top_sectors:
            sector_stocks = df_stocks[
                (df_stocks["board"] == sector) &
                (df_stocks["avg_turnover"] >= self.MIN_TURNOVER) &
                (df_stocks["above_ma20"]) &
                (df_stocks["ma20_above_ma60"])
            ].nlargest(self.PER_SECTOR, "ret_20d")

            selected.extend(sector_stocks["symbol"].tolist())

        # 去ST
        with sqlite3.connect(self.engine.db_path) as conn:
            st_rows = conn.execute(
                "SELECT symbol FROM stock_basic "
                "WHERE name LIKE 'ST%' OR name LIKE '%*ST%' OR name LIKE '%退%'"
            ).fetchall()
            st_set = {r[0] for r in st_rows}
        selected = [s for s in selected if s not in st_set]

        logger.info(f"板块轮动：选出{len(selected)}只（{len(top_sectors)}板块×{self.PER_SECTOR}龙头）")
        return selected
