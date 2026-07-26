"""大宗交易数据同步：akshare stock_dzjy_mrtj（含折溢率）。

日频全市场大宗交易明细，折溢率是机构接货的核心信号。
不限额度（akshare 东财源），回补历史约4分钟。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


def sync_block_trade(db_path: str, date_str: str) -> int:
    """同步单日大宗交易明细到 DB（幂等 UPSERT）。

    Args:
        db_path: 数据库路径
        date_str: 日期 YYYYMMDD 格式

    Returns:
        写入行数
    """
    import akshare as ak

    _ensure_table(db_path)
    db_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    try:
        df = ak.stock_dzjy_mrtj(start_date=date_str, end_date=date_str)
    except Exception as exc:
        logger.debug(f"大宗交易拉取失败 {date_str}: {exc}")
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
        sym = str(r.get("证券代码", "")).strip()
        if not sym:
            continue
        rows.append((
            sym,
            db_date,
            _num(r.get("收盘价")),
            _num(r.get("成交价")),
            _num(r.get("折溢率")),
            _num(r.get("成交量")),
            _num(r.get("成交额")),
            str(r.get("买方营业部", ""))[:120],
            str(r.get("卖方营业部", ""))[:120],
        ))

    if not rows:
        return 0

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO block_trade "
            "(symbol, date, close, trade_price, discount, volume, amount, buyer, seller) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    return len(rows)


def backfill_block_trade(db_path: str, trading_days: list[str],
                         batch_pause: int = 100, pause_sec: float = 0.5) -> dict:
    """批量回补大宗交易历史。

    Args:
        db_path: 数据库路径
        trading_days: 交易日列表（YYYYMMDD 格式）
        batch_pause: 每处理N天暂停一次
        pause_sec: 暂停秒数

    Returns:
        {total_days, success_days, rows, elapsed}
    """
    import time

    t0 = time.time()
    success = 0
    total_rows = 0
    for i, d in enumerate(trading_days):
        rows = sync_block_trade(db_path, d)
        if rows > 0:
            success += 1
            total_rows += rows
        if (i + 1) % batch_pause == 0:
            elapsed = time.time() - t0
            logger.info(
                f"大宗交易回补: {i+1}/{len(trading_days)}天, "
                f"成功{success}/{total_rows}行, 耗时{elapsed:.0f}s"
            )
            time.sleep(pause_sec)
    elapsed = time.time() - t0
    logger.info(f"大宗交易回补完成: {success}/{len(trading_days)}天, {total_rows}行, {elapsed:.0f}s")
    return {"total_days": len(trading_days), "success_days": success,
            "rows": total_rows, "elapsed": round(elapsed, 1)}


def _ensure_table(db_path: str) -> None:
    """幂等建表。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS block_trade ("
            "symbol TEXT NOT NULL, date TEXT NOT NULL, "
            "close REAL, trade_price REAL, discount REAL, "
            "volume REAL, amount REAL, buyer TEXT, seller TEXT, "
            "PRIMARY KEY (symbol, date, buyer, seller))"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_block_trade_sym_dt ON block_trade(symbol, date)"
        )
        conn.commit()
