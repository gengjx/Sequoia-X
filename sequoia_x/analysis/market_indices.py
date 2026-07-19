"""大盘分析 - 指数结构（MarketAnalyzer mixin 组件）。

由 :class:`sequoia_x.analysis.market.MarketAnalyzer` 多重继承组合，
不单独实例化；方法通过 ``self.db_path`` 等访问门面状态。
"""

from __future__ import annotations

import pandas as pd

from sequoia_x.analysis.market_common import (
    _INDEX_DEFS,
)
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class IndicesMixin:
    """指数结构计算逻辑（MarketAnalyzer 的 mixin 组件）。"""

    def _fetch_indices(self, latest: str) -> list[dict]:
        from datetime import date as _date
        from datetime import timedelta as _td

        import baostock as bs

        # 仅拉取近 ~45 个交易日（约 65 自然日），足够计算涨跌幅与 20 日支撑/压力
        start_date = (_date.fromisoformat(latest) - _td(days=65)).strftime("%Y-%m-%d")

        bs.login()
        results: list[dict] = []
        try:
            for key, bs_code, name in _INDEX_DEFS:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,volume,amount",
                    start_date=start_date,
                    end_date=latest,
                    frequency="d",
                    adjustflag="3",  # 指数不复权
                )
                if rs.error_code != "0":
                    continue
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if len(rows) < 2:
                    continue
                df = pd.DataFrame(rows, columns=rs.fields)
                for col in ["open", "high", "low", "close", "amount"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"]).reset_index(drop=True)
                if len(df) < 2:
                    continue
                cur = df.iloc[-1]
                prv = df.iloc[-2]
                change_pct = round((cur["close"] - prv["close"]) / prv["close"] * 100, 2)
                amplitude = round((cur["high"] - cur["low"]) / prv["close"] * 100, 2)
                # 支撑/压力：近 5 日高低点（贴近实战的短期关键位）
                window = df.tail(5)
                support = round(float(window["low"].min()), 2)
                resistance = round(float(window["high"].max()), 2)
                results.append({
                    "key": key,
                    "name": name,
                    "latest": round(float(cur["close"]), 2),
                    "change_pct": change_pct,
                    "open": round(float(cur["open"]), 2),
                    "high": round(float(cur["high"]), 2),
                    "low": round(float(cur["low"]), 2),
                    "amplitude": amplitude,
                    "turnover_yi": round(float(cur["amount"]) / 1e8, 0),
                    "support": support,
                    "resistance": resistance,
                })
        finally:
            bs.logout()
        return results

    # ------------------------------------------------------------------
    # 3. 板块主线
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 股票元数据缓存（名称 / 上市日 → 涨跌停限幅与新股过滤）
    # ------------------------------------------------------------------
