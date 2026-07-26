"""股东户数数据同步：akshare stock_zh_a_gdhs_detail_em（逐只历史）。

季频全市场股东户数，筹码集中度核心信号。
13年历史(2013-2026)，全市场5500只≈35分钟，不耗baostock额度。
"""

from __future__ import annotations

import sqlite3

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


def _ensure_table(db_path: str) -> None:
    """幂等建表。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS holder_count ("
            "symbol TEXT NOT NULL, end_date TEXT NOT NULL, "
            "holder_num REAL, holder_change REAL, avg_value REAL, avg_shares REAL, "
            "PRIMARY KEY (symbol, end_date))"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_holder_sym_dt ON holder_count(symbol, end_date)"
        )
        conn.commit()


def sync_holder_detail(db_path: str, symbol: str) -> int:
    """同步单只股票的股东户数历史到 DB（幂等 UPSERT）。

    Args:
        db_path: 数据库路径
        symbol: 股票代码（6位数字）

    Returns:
        写入行数
    """
    import akshare as ak

    _ensure_table(db_path)

    try:
        df = ak.stock_zh_a_gdhs_detail_em(symbol=symbol)
    except Exception as exc:
        logger.debug(f"股东户数拉取失败 {symbol}: {exc}")
        return 0

    if df is None or len(df) == 0:
        return 0

    def _num(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    rows = []
    for _, r in df.iterrows():
        end_date = r.get("股东户数统计截止日")
        if end_date is None:
            continue
        end_date = str(end_date)[:10]  # 统一 YYYY-MM-DD
        rows.append((
            symbol,
            end_date,
            _num(r.get("股东户数-本次")),
            _num(r.get("股东户数-增减比例")),
            _num(r.get("户均持股市值")),
            _num(r.get("户均持股数量")),
        ))

    if not rows:
        return 0

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO holder_count "
            "(symbol, end_date, holder_num, holder_change, avg_value, avg_shares) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    return len(rows)


def backfill_holder_count(db_path: str, symbols: list[str],
                          batch_pause: int = 100, pause_sec: float = 0.3) -> dict:
    """批量回补股东户数历史。

    Args:
        db_path: 数据库路径
        symbols: 股票代码列表
        batch_pause: 每处理N只暂停一次
        pause_sec: 暂停秒数

    Returns:
        {total, success, failed, rows, elapsed}
    """
    import time

    _ensure_table(db_path)
    t0 = time.time()
    success = 0
    failed = 0
    total_rows = 0
    for i, sym in enumerate(symbols):
        rows = sync_holder_detail(db_path, sym)
        if rows > 0:
            success += 1
            total_rows += rows
        else:
            failed += 1
        if (i + 1) % batch_pause == 0:
            elapsed = time.time() - t0
            logger.info(
                f"股东户数回补: {i+1}/{len(symbols)}只, "
                f"成功{success}/{total_rows}行, 耗时{elapsed:.0f}s"
            )
            time.sleep(pause_sec)
    elapsed = time.time() - t0
    logger.info(f"股东户数回补完成: {success}/{len(symbols)}只, {total_rows}行, {elapsed:.0f}s")
    return {"total": len(symbols), "success": success, "failed": failed,
            "rows": total_rows, "elapsed": round(elapsed, 1)}
