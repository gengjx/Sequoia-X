"""融资融券明细采集引擎。

数据源：akshare stock_margin_detail_sse（上交所）+ stock_margin_detail_szse（深交所）。
融资融券代表杠杆资金方向——融资余额=看多杠杆，融券余量=看空杠杆。
2024-08北向资金个股明细永久断供后，融资融券是最佳机构/杠杆资金替代信号。

采集范围：两所全部标的（每日 ~4000 只），按交易日采集。
历史深度：2021年至今（仅交易日有数据，节假日返回空）。
"""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


def fetch_margin_detail(date_str: str) -> list[dict]:
    """获取某日全市场融资融券明细（沪+深合并）。

    Args:
        date_str: 日期 YYYYMMDD 格式（必须是交易日，节假日返回空）

    Returns:
        [{symbol, date, rzye, rzbuy, rzrepay, rqlts, rqsell, rqrepay, rqye}, ...]
    """
    import akshare as ak
    result: list[dict] = []
    db_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    # 上交所
    try:
        df = ak.stock_margin_detail_sse(date=date_str)
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                code = str(row.get("标的证券代码", "")).strip()
                # 跳过 ETF/基金（510/159开头），只保留股票
                if not code or code.startswith(("510", "511", "512", "513", "515",
                                                 "516", "518", "561", "159", "501", "502")):
                    continue
                result.append({
                    "symbol": code,
                    "date": db_date,
                    "rzye": _f(row.get("融资余额")),
                    "rzbuy": _f(row.get("融资买入额")),
                    "rzrepay": _f(row.get("融资偿还额")),
                    "rqlts": _f(row.get("融券余量")),
                    "rqsell": _f(row.get("融券卖出量")),
                    "rqrepay": _f(row.get("融券偿还量")),
                    "rqye": None,  # 上交所无此列
                })
    except Exception as e:
        logger.debug(f"融资融券 SSE {date_str}: {e}")

    # 深交所
    try:
        df = ak.stock_margin_detail_szse(date=date_str)
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                code = str(row.get("证券代码", "")).strip()
                if not code:
                    continue
                result.append({
                    "symbol": code,
                    "date": db_date,
                    "rzye": _f(row.get("融资余额")),
                    "rzbuy": _f(row.get("融资买入额")),
                    "rzrepay": None,  # 深交所无此列
                    "rqlts": _f(row.get("融券余量")),
                    "rqsell": _f(row.get("融券卖出量")),
                    "rqrepay": None,
                    "rqye": _f(row.get("融券余额")),
                })
    except Exception as e:
        logger.debug(f"融资融券 SZSE {date_str}: {e}")

    return result


def _f(v) -> float | None:
    """安全转 float。"""
    try:
        return float(v) if v is not None and v == v else None
    except (TypeError, ValueError):
        return None


def sync_margin_detail(db_path: str, date_str: str) -> int:
    """同步单日融资融券明细到 DB（幂等 UPSERT）。

    Returns:
        写入行数
    """
    records = fetch_margin_detail(date_str)
    if not records:
        return 0
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO margin_detail "
            "(symbol, date, rzye, rzbuy, rzrepay, rqlts, rqsell, rqrepay, rqye) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(r["symbol"], r["date"], r["rzye"], r["rzbuy"], r["rzrepay"],
              r["rqlts"], r["rqsell"], r["rqrepay"], r["rqye"]) for r in records],
        )
        conn.commit()
    return len(records)


def backfill_margin_detail(db_path: str, trading_days: list[str],
                           batch_pause: int = 50, pause_sec: float = 1.0) -> dict:
    """批量回补融资融券历史。

    Args:
        db_path: 数据库路径
        trading_days: 交易日列表（YYYYMMDD 格式）
        batch_pause: 每处理N只暂停一次
        pause_sec: 暂停秒数

    Returns:
        {total_days, success_days, rows, elapsed}
    """
    t0 = time.time()
    success = 0
    total_rows = 0
    for i, date_str in enumerate(trading_days):
        rows = sync_margin_detail(db_path, date_str)
        if rows > 0:
            success += 1
            total_rows += rows
        if (i + 1) % batch_pause == 0:
            elapsed = time.time() - t0
            logger.info(
                f"融资融券回补: {i+1}/{len(trading_days)}天, "
                f"成功{success}/{total_rows}行, 耗时{elapsed:.0f}s"
            )
            time.sleep(pause_sec)
    elapsed = time.time() - t0
    logger.info(f"融资融券回补完成: {success}/{len(trading_days)}天, {total_rows}行, {elapsed:.0f}s")
    return {"total_days": len(trading_days), "success_days": success,
            "rows": total_rows, "elapsed": round(elapsed, 1)}
