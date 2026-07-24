"""北向资金（沪深港通）持股历史采集引擎。

数据源：akshare stock_hsgt_individual_em（东财沪深港通个股持股明细）。
北向资金代表外资/机构资金方向，持股占比变化是强机构信号。

采集范围：限定 circ_mv 前 ~800 只（沪深300+中证500），北向持仓高度集中于
中大市值，小盘股持股可忽略并默认 0，避免 5000 次 API 调用。

失败降级：表不存在或 API 不可用时返回空，不阻断流程。
"""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# akshare 列名映射 → DB 列名
_COL_MAP = {
    "持股日期": "date",
    "持股数量": "hold_share",
    "持股市值": "hold_value",
    "持股数量占A股百分比": "hold_pct",
}


def _ensure_table(db_path: str) -> None:
    """幂等建表。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS north_hold ("
            "symbol TEXT NOT NULL, date TEXT NOT NULL, "
            "hold_share REAL, hold_value REAL, hold_pct REAL, "
            "PRIMARY KEY (symbol, date))"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_north_hold_symbol ON north_hold(symbol)"
        )
        conn.commit()


def fetch_north_hold(symbol: str) -> list[dict]:
    """获取单只股票的北向持股历史。

    Returns:
        [{symbol, date, hold_share, hold_value, hold_pct}, ...]
    """
    try:
        import akshare as ak
        df = ak.stock_hsgt_individual_em(symbol=symbol)
        if df is None or len(df) == 0:
            return []
        df = df.rename(columns=_COL_MAP)
        # 只保留映射成功的列
        keep = [c for c in ["date", "hold_share", "hold_value", "hold_pct"] if c in df.columns]
        df = df[keep].copy()
        result: list[dict] = []
        for _, row in df.iterrows():
            rec = {"symbol": symbol, "date": str(row.get("date"))}
            for c in ("hold_share", "hold_value", "hold_pct"):
                if c in keep:
                    v = row.get(c)
                    try:
                        rec[c] = float(v) if v == v else None  # NaN check
                    except (TypeError, ValueError):
                        rec[c] = None
                else:
                    rec[c] = None
            result.append(rec)
        return result
    except Exception as e:
        logger.debug(f"北向持股 {symbol} 获取失败: {e}")
        return []


def backfill_north_hold(
    db_path: str,
    symbols: list[str] | None = None,
    top_n: int = 800,
    n_workers: int = 3,
    rate_limit_per_sec: float = 0.5,
) -> dict:
    """批量回填北向持股历史。

    Args:
        db_path: 数据库路径
        symbols: 指定股票列表，None=自动选 circ_mv 前 top_n 只
        top_n: 自动选股时的市值排名上限
        n_workers: 并发线程数
        rate_limit_per_sec: 每只股票之间的间隔秒数（避免触发限流）

    Returns:
        {total, success, failed, rows, elapsed}
    """
    t0 = time.time()
    _ensure_table(db_path)

    # 选股：指定列表 或 circ_mv 前 top_n
    if symbols is None:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT symbol FROM stock_market_cap "
                "WHERE circ_mv IS NOT NULL AND circ_mv > 0 "
                "ORDER BY circ_mv DESC LIMIT ?",
                (top_n,),
            ).fetchall()
            symbols = [r[0] for r in rows]

    total = len(symbols)
    if total == 0:
        logger.warning("北向回填：无候选股票（stock_market_cap 表可能为空）")
        return {"total": 0, "success": 0, "failed": 0, "rows": 0, "elapsed": 0}

    logger.info(f"北向持股回填：{total} 只股票，{n_workers} 线程")

    success = 0
    failed = 0
    all_rows: list[tuple] = []

    def _worker(sym: str):
        time.sleep(rate_limit_per_sec)
        return sym, fetch_north_hold(sym)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, s): s for s in symbols}
        done = 0
        for fut in as_completed(futures):
            done += 1
            sym = futures[fut]
            try:
                _, records = fut.result()
                if records:
                    success += 1
                    for rec in records:
                        all_rows.append((
                            rec["symbol"], rec["date"],
                            rec.get("hold_share"), rec.get("hold_value"),
                            rec.get("hold_pct"),
                        ))
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logger.debug(f"北向回填 {sym} 异常: {e}")
            if done % 100 == 0:
                logger.info(f"北向回填进度：{done}/{total}，成功{success}，失败{failed}")

    # 批量写入（UPSERT）
    rows_written = 0
    if all_rows:
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO north_hold "
                "(symbol, date, hold_share, hold_value, hold_pct) "
                "VALUES (?,?,?,?,?)",
                all_rows,
            )
            conn.commit()
            rows_written = len(all_rows)

    elapsed = time.time() - t0
    logger.info(
        f"北向持股回填完成：{success}/{total} 成功，{rows_written} 行，"
        f"耗时 {elapsed:.0f}s"
    )
    return {
        "total": total,
        "success": success,
        "failed": failed,
        "rows": rows_written,
        "elapsed": round(elapsed, 1),
    }


def select_top_symbols(db_path: str, top_n: int = 800) -> list[str]:
    """选取 circ_mv 前 N 只股票（供回补和日同步使用）。"""
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT symbol FROM stock_market_cap "
                "WHERE circ_mv IS NOT NULL AND circ_mv > 0 "
                "ORDER BY circ_mv DESC LIMIT ?",
                (top_n,),
            ).fetchall()
            return [r[0] for r in rows]
    except Exception:
        return []
