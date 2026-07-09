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


def _fetch_minute_sina(symbol: str, klt: int = 1, datalen: int = 300) -> list[dict]:
    """从新浪拉取分钟K线（备用源）。

    新浪返回不复权数据，格式：[{day, open, high, low, close, volume}, ...]
    """
    import requests

    sina_symbol = ("sh" if symbol.startswith(("6", "9", "5")) else "sz") + symbol
    # 新浪 scale 参数：5→5分钟, 15→15分钟, 60→60分钟
    scale_map = {1: 1, 5: 5, 15: 15, 60: 60}
    scale = scale_map.get(klt, 5)

    try:
        r = requests.get(
            "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData",
            params={"symbol": sina_symbol, "scale": scale, "datalen": datalen},
            timeout=8,
        )
        data = r.json()
    except Exception as e:
        logger.warning(f"新浪分钟K拉取失败 {symbol}: {e!r}")
        return []

    if not data:
        return []

    rows = []
    for item in data:
        try:
            rows.append({
                "symbol": symbol,
                "datetime": item.get("day", ""),
                "open": float(item.get("open", 0)),
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "close": float(item.get("close", 0)),
                "volume": float(item.get("volume", 0)),
                "amount": 0,  # 新浪不返回成交额
            })
        except (ValueError, TypeError):
            continue
    return rows


def fetch_minute_klines(symbol: str, klt: int = 1, days: int = 1) -> list[dict]:
    """拉取单只股票的分钟K线（双源冗余：东财优先 → 新浪备用）。

    Args:
        symbol: 纯数字代码
        klt: K线周期 1=1分钟 5=5分钟 15=15分钟 60=小时
        days: 拉取天数（东财用日期范围，新浪用datalen条数）
    Returns:
        [{symbol, datetime, open, high, low, close, volume, amount}, ...]
    """
    # 先试东财
    rows = _fetch_minute_eastmoney(symbol, klt, days)
    if rows:
        return rows

    # 东财失败 → 新浪备用
    logger.info(f"东财分钟K失败 {symbol}，切换新浪备用源")
    datalen = {1: 240, 5: 48, 15: 16, 60: 4}.get(klt, 48) * days
    return _fetch_minute_sina(symbol, klt, datalen)


def _fetch_minute_eastmoney(symbol: str, klt: int = 1, days: int = 1) -> list[dict]:
    """从东财 push2his 拉取单只股票的分钟K线（主源）。"""
    from sequoia_x.core.rate_limiter import em_get

    secid = _to_eastmoney_secid(symbol)
    today = datetime.now().strftime("%Y%m%d")
    from datetime import timedelta
    beg = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
    klines = []
    for attempt in range(3):
        try:
            r = em_get(
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

        # 3. 今日决策池（盘后选出的票，盘中实时监控突破/异动）
        try:
            for row in conn.execute(
                "SELECT symbol, source FROM decision_pool WHERE date=? "
                "ORDER BY score DESC", (today,)
            ).fetchall():
                src = row[1] or "决策"
                watchlist[row[0]] = watchlist.get(row[0], src)
        except sqlite3.OperationalError:
            pass  # 表不存在时跳过

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
