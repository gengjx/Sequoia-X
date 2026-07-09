"""资金流向历史回填引擎：东财 push2his 个股历史资金流。

数据源：push2his.eastmoney.com/api/qt/stock/fflow/daykline/get
每只股票返回最近N天的日级资金流向（主力/超大单/大单/中单/小单净额）。

回填策略：
  - 优先回填决策池+持仓+多因子选出的股票（高价值）
  - 支持并发回填（6线程）
  - 失败自动跳过，不阻塞整体
"""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}


def _to_secid(symbol: str) -> str:
    """股票代码转东财secid（沪市1.，深市0.）。"""
    if symbol.startswith(("6", "9")):
        return f"1.{symbol}"
    return f"0.{symbol}"


def fetch_fund_flow_history(symbol: str, days: int = 60) -> list[dict]:
    """获取单只股票最近N天的资金流向。

    Returns:
        [{date, main_net, small_net, mid_net, big_net, super_net}, ...]
    """
    secid = _to_secid(symbol)
    try:
        from sequoia_x.core.rate_limiter import _rate_limiter
        if not _rate_limiter.eastmoney_acquire():
            return []
        r = requests.get(
            "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
            params={
                "lmt": days,
                "klt": 101,  # 日K
                "secid": secid,
                "fields1": "f1,f2,f3,f7",
                "fields2": "f51,f52,f53,f54,f55,f56",
            },
            headers=_HEADERS,
            timeout=10,
        )
        data = r.json().get("data", {})
        klines = data.get("klines", [])
        result = []
        for line in klines:
            parts = line.split(",")
            if len(parts) < 6:
                continue
            result.append({
                "symbol": symbol,
                "date": parts[0],
                "main_net": float(parts[1]),   # 主力净流入
                "small_net": float(parts[2]),  # 小单净流入
                "mid_net": float(parts[3]),    # 中单净流入
                "big_net": float(parts[4]),    # 大单净流入
                "super_net": float(parts[5]),  # 超大单净流入
            })
        _rate_limiter.eastmoney_success()
        return result
    except Exception as e:
        _rate_limiter.eastmoney_failure()
        logger.debug(f"资金流向历史 {symbol} 获取失败: {e}")
        return []


def backfill_fund_flow_history(db_path: str, symbols: list[str],
                               days: int = 60, n_workers: int = 6) -> dict:
    """批量回填资金流向历史。

    Args:
        db_path: 数据库路径
        symbols: 股票代码列表
        days: 回填天数
        n_workers: 并发线程数

    Returns:
        {total, success, failed, rows, elapsed}
    """
    t0 = time.time()
    total = len(symbols)
    success = 0
    failed = 0
    all_rows: list[tuple] = []

    # 探测东财 push2his 是否可用
    from sequoia_x.core.rate_limiter import RateLimiter
    if not RateLimiter.probe_eastmoney("push2his"):
        logger.error("东财 push2his 被封，资金流向回填取消（push2delay仍可用，稍后重试）")
        return {"total": total, "success": 0, "failed": total, "rows": 0, "elapsed": 0, "blocked": True}

    logger.info(f"资金流向历史回填：{total}只，{days}天，{n_workers}线程")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(fetch_fund_flow_history, sym, days): sym for sym in symbols}
        for i, future in enumerate(as_completed(futures)):
            sym = futures[future]
            try:
                rows = future.result()
                if rows:
                    all_rows.extend(rows)
                    success += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed
                eta = (total - i - 1) / speed if speed > 0 else 0
                logger.info(f"资金流向回填进度：{i+1}/{total} ({speed:.0f}只/秒 ETA {eta:.0f}s)")

    # 批量写入
    if all_rows:
        conn = sqlite3.connect(db_path, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executemany(
                """INSERT OR REPLACE INTO fund_flow
                   (symbol, date, main_net, main_pct, super_net, big_net, mid_net, small_net)
                   VALUES (?, ?, ?, 0, ?, ?, ?, ?)""",
                [(r["symbol"], r["date"], r["main_net"], r["super_net"],
                  r["big_net"], r["mid_net"], r["small_net"]) for r in all_rows],
            )
        finally:
            conn.close()

    elapsed = time.time() - t0
    logger.info(f"资金流向回填完成：成功{success} 失败{failed} 写入{len(all_rows)}行 耗时{elapsed:.0f}s")
    return {
        "total": total, "success": success, "failed": failed,
        "rows": len(all_rows), "elapsed": round(elapsed, 1),
    }
