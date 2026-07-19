"""Web 服务 - 股票浏览（WebServices mixin 组件）。

由 :class:`sequoia_x.web.services.WebServices` 多重继承组合，不单独实例化；
方法通过 ``self.engine`` / ``self.settings`` / ``self._task_store`` 等访问门面状态。
"""

from __future__ import annotations

import logging
import sqlite3

from sequoia_x.analysis.stock_analysis import StockAnalyzer
from sequoia_x.web.services_common import (
    _to_xueqiu_code,
)

logger = logging.getLogger(__name__)


class StockMixin:
    """股票浏览（WebServices 的 mixin 组件）。"""

    def search_symbols(self, query: str, limit: int = 20) -> list[str]:
        with sqlite3.connect(self.engine.db_path, timeout=5) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM stock_daily WHERE symbol LIKE ? ORDER BY symbol LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [r[0] for r in rows]

    def get_stock_data(self, symbol: str, days: int = 120) -> dict:
        """返回K线数据（后复权）+ 复权系数，前端可切换后复权/不复权视图。

        DB存后复权数据（运算正确口径），展示层需标注并提供不复权视图。
        复权系数 = 东财昨收真实价(f60) / DB最新日后复权收盘（同日纯系数）。
        """
        df = self.engine.get_ohlcv(symbol)
        if df.empty:
            return {"rows": [], "adjust_ratio": None, "real_latest": None, "price_type": "hfq", "real_available": False}
        df = df.sort_values("date").tail(days)
        hfq_latest = float(df.iloc[-1]["close"])
        adjust_ratio = None
        real_latest = None
        try:
            real_latest, prev_close = StockAnalyzer._fetch_price_quote(symbol)
            if prev_close and hfq_latest and prev_close > 0:
                adjust_ratio = prev_close / hfq_latest
        except Exception:
            pass
        rows = df[["date", "open", "high", "low", "close", "volume", "turnover"]].to_dict("records")
        return {
            "rows": rows,
            "adjust_ratio": round(adjust_ratio, 6) if adjust_ratio else None,
            "real_latest": round(real_latest, 2) if real_latest else None,
            "price_type": "hfq",
            "real_available": adjust_ratio is not None,
        }

    def get_stock_summary(self, symbol: str) -> dict | None:
        df = self.engine.get_ohlcv(symbol)
        if df.empty:
            return None
        df = df.sort_values("date")
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change_pct = round((latest["close"] - prev["close"]) / prev["close"] * 100, 2) if prev["close"] else 0
        # 真实价（东财昨收，与DB最新日同日），DB存后复权，展示需双口径
        real_price = None
        adjust_ratio = None
        try:
            real_latest, prev_close = StockAnalyzer._fetch_price_quote(symbol)
            if prev_close and latest["close"] and prev_close > 0:
                real_price = round(prev_close, 2)
                adjust_ratio = round(prev_close / latest["close"], 6)
        except Exception:
            pass
        return {
            "symbol": symbol,
            "latest_close": round(latest["close"], 2),      # 后复权
            "real_price": real_price,                         # 真实不复权
            "adjust_ratio": adjust_ratio,                     # 后复权→真实系数
            "latest_date": latest["date"],
            "change_pct": change_pct,
            "total_rows": len(df),
            "date_range": [df.iloc[0]["date"], df.iloc[-1]["date"]],
            "xueqiu_code": _to_xueqiu_code(symbol),
        }

    # -- System methods --
