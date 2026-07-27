"""解禁数据同步：akshare stock_restricted_release_detail_em（全市场解禁明细）。

解禁日期是公司提前公告的未来事件，用作风控过滤器（回避未来N天大额解禁）。
5年历史10秒回补，不耗baostock额度。
"""

from __future__ import annotations

import sqlite3

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


def _ensure_table(db_path: str) -> None:
    """幂等建表。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS restricted_release ("
            "symbol TEXT NOT NULL, release_date TEXT NOT NULL, "
            "unlock_ratio REAL, unlock_value REAL, "
            "PRIMARY KEY (symbol, release_date))"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_restricted_date ON restricted_release(release_date)"
        )
        conn.commit()


def sync_restricted_release(db_path: str, start_date: str, end_date: str) -> int:
    """按日期范围拉全市场解禁明细入库（幂等 UPSERT）。

    Args:
        db_path: 数据库路径
        start_date: 起始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD

    Returns:
        写入行数
    """
    import akshare as ak

    _ensure_table(db_path)

    try:
        df = ak.stock_restricted_release_detail_em(start_date=start_date, end_date=end_date)
    except Exception as exc:
        logger.debug(f"解禁拉取失败 {start_date}~{end_date}: {exc}")
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
        sym = str(r.get("股票代码", "")).strip()
        if not sym:
            continue
        release_date = str(r.get("解禁时间", ""))[:10]  # YYYY-MM-DD
        if not release_date or release_date == "NaT":
            continue
        rows.append((
            sym,
            release_date,
            _num(r.get("占解禁前流通市值比例")),
            _num(r.get("实际解禁市值")),
        ))

    if not rows:
        return 0

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO restricted_release "
            "(symbol, release_date, unlock_ratio, unlock_value) "
            "VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()
    return len(rows)


def backfill_restricted_release(db_path: str, start_year: int = 2022, end_year: int = 2026) -> dict:
    """按年拉历史解禁数据（供回测 PIT 验证）。

    Args:
        db_path: 数据库路径
        start_year: 起始年
        end_year: 结束年

    Returns:
        {years, total_rows, elapsed}
    """
    import time

    _ensure_table(db_path)
    t0 = time.time()
    total_rows = 0
    years_done = 0
    for year in range(start_year, end_year + 1):
        sd = f"{year}0101"
        ed = f"{year}1231"
        rows = sync_restricted_release(db_path, sd, ed)
        total_rows += rows
        years_done += 1
        logger.info(f"解禁回补 {year}: {rows}行")
    elapsed = time.time() - t0
    logger.info(f"解禁回补完成: {years_done}年, {total_rows}行, {elapsed:.0f}s")
    return {"years": years_done, "total_rows": total_rows, "elapsed": round(elapsed, 1)}
