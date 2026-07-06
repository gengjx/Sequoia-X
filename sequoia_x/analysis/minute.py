"""分钟K线采集器：按关注池从东财 pull2his 拉取1/5/15分钟K线。

数据源：东财 push2his（免费，0.1s/只，当天240根1分钟线全覆盖）。
不全量回填5205只（太重），按关注池拉取（竞价A级+持仓+决策池，通常50-100只）。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


def _to_eastmoney_secid(symbol: str) -> str:
    """纯数字代码 → 东财 secid（6/9开头→1.沪，其余→0.深）。"""
    return f"1.{symbol}" if symbol.startswith(("6", "9", "5")) else f"0.{symbol}"


def fetch_minute_klines(symbol: str, klt: int = 1, days: int = 1) -> list[dict]:
    """从东财拉取单只股票的分钟K线。

    Args:
        symbol: 纯数字代码
        klt: K线周期 1=1分钟 5=5分钟 15=15分钟 60=小时
        days: 拉取天数（beg=end往前推days天）
    Returns:
        [{symbol, datetime, open, high, low, close, volume, amount}, ...]
    """
    import requests

    secid = _to_eastmoney_secid(symbol)
    today = datetime.now().strftime("%Y%m%d")
    from datetime import timedelta
    beg = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
    klines = []
    for attempt in range(3):
        try:
            r = requests.get(
                "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                params={
                    "secid": secid,
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
                    "klt": klt, "fqt": "1",  # 后复权
                    "beg": beg, "end": today,
                },
                headers=headers, timeout=8,
            )
            klines = r.json().get("data", {}).get("klines") or []
            if klines:
                break
        except Exception as e:
            if attempt < 2:
                import time as _t
                _t.sleep(0.5 * (attempt + 1))
            else:
                logger.warning(f"分钟K拉取失败 {symbol}(重试3次): {e!r}")
    if not klines:
        return []

    rows = []
    for kl in klines:
        parts = kl.split(",")
        if len(parts) < 8:
            continue
        try:
            rows.append({
                "symbol": symbol,
                "datetime": parts[0],
                "open": float(parts[1]),
                "high": float(parts[2]),
                "low": float(parts[3]),
                "close": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]),
            })
        except (ValueError, IndexError):
            continue
    return rows


def build_watchlist(db_path: str) -> list[dict]:
    """构建关注池：竞价A级 + 持仓 + 决策池候选。

    Returns:
        [{symbol, source}, ...] source标注来源(竞价/持仓/决策)
    """
    watchlist: dict[str, str] = {}
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(db_path) as conn:
        # 1. 竞价A级（今天的竞价强票）
        try:
            for row in conn.execute(
                "SELECT symbol FROM auction_snap WHERE grade='A' "
                "AND date=(SELECT MAX(date) FROM auction_snap)"
            ).fetchall():
                watchlist[row[0]] = "竞价A"
        except sqlite3.OperationalError:
            pass

        # 2. 持仓股
        for row in conn.execute(
            "SELECT symbol FROM portfolio_holding"
        ).fetchall():
            if row[0]:
                watchlist[row[0]] = watchlist.get(row[0], "持仓")

        # 3. 决策A/B级（用最新的决策快照，若有缓存表）
        # 注：决策结果目前不入库，暂只取竞价+持仓
    result = [{"symbol": s, "source": src} for s, src in watchlist.items()]
    logger.info(f"关注池构建：{len(result)}只（竞价A+持仓）")
    return result


class MinuteCollector:
    """分钟K线批量采集器。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def collect_watchlist(self, klt: int = 1, days: int = 1) -> dict:
        """采集关注池全量分钟K线并落库。

        Args:
            klt: 1=1分钟 5=5分钟
            days: 拉取天数
        Returns:
            {total, collected, failed, symbols}
        """
        from sequoia_x.data.engine import DataEngine

        watchlist = build_watchlist(self.db_path)
        if not watchlist:
            return {"total": 0, "collected": 0, "failed": 0, "symbols": [], "msg": "关注池为空"}

        engine = DataEngine.__new__(DataEngine)
        engine.db_path = self.db_path

        all_rows = []
        failed = []
        for w in watchlist:
            rows = fetch_minute_klines(w["symbol"], klt=klt, days=days)
            if rows:
                all_rows.extend(rows)
            else:
                failed.append(w["symbol"])

        saved = engine.save_minute_klines(all_rows) if all_rows else 0
        logger.info(f"分钟K采集完成：关注池{len(watchlist)}只，落库{saved}条，失败{len(failed)}只")

        return {
            "total": len(watchlist),
            "collected": saved,
            "failed": len(failed),
            "failed_symbols": failed,
            "symbols": [{"symbol": w["symbol"], "source": w["source"]} for w in watchlist],
        }
