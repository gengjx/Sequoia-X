"""另类数据同步引擎：基金持仓 + 沪深300指数 + 宏观(M2/社融)。

数据源：akshare（新浪/东财宏观），非 push2his，不受东财神资金流接口熔断影响。
同步周期：
  - 基金持仓：季度（每季财报发布后更新）
  - 沪深300指数：日（盘后增量）
  - M2/社融：月（每月中旬发布上月数据）
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


def _f(v) -> float | None:
    try:
        return float(v) if v is not None and v == v else None
    except (TypeError, ValueError):
        return None


def _parse_month(s: str) -> str:
    """统一月份格式为 YYYY-MM。"""
    s = str(s).strip()
    # "2026年06月份" → 2026-06
    m = re.match(r"(\d{4})年(\d{1,2})月", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    # "202601" → 2026-01
    m = re.match(r"(\d{4})(\d{2})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return s


# ════════════════════════════════════════
# 基金持仓（季度）
# ════════════════════════════════════════
def sync_fund_hold(db_path: str, report_date: str | None = None) -> int:
    """同步单季度基金持仓数据。

    Args:
        report_date: 报告期 YYYYMMDD（如 "20251231"），None=最近4个季度全量拉

    Returns:
        写入行数
    """
    import akshare as ak
    dates = [report_date] if report_date else _recent_report_dates()
    total = 0
    for d in dates:
        try:
            df = ak.stock_report_fund_hold(date=d)
            if df is None or len(df) == 0:
                continue
            rows = []
            for _, r in df.iterrows():
                rows.append((
                    str(r.get("股票代码", "")).strip(),
                    f"{d[:4]}-{d[4:6]}-{d[6:8]}",
                    r.get("股票简称", ""),
                    int(_f(r.get("持有基金家数")) or 0),
                    _f(r.get("持股总数")),
                    _f(r.get("持股市值")),
                    str(r.get("持股变化", "")),
                    _f(r.get("持股变动数值")),
                    _f(r.get("持股变动比例")),
                ))
            if rows:
                with sqlite3.connect(db_path) as conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO fund_hold "
                        "(symbol, report_date, name, fund_count, hold_shares, "
                        "hold_value, change_dir, change_shares, change_pct) "
                        "VALUES (?,?,?,?,?,?,?,?,?)", rows,
                    )
                    conn.commit()
                total += len(rows)
                logger.info(f"基金持仓 {d}: {len(rows)}行")
        except Exception as e:
            logger.warning(f"基金持仓 {d} 失败: {e!r}")
    return total


def _recent_report_dates(n: int = 8) -> list[str]:
    """最近N个报告期（季末），YYYYMMDD。"""
    today = datetime.now()
    quarters = []
    y, m = today.year, today.month
    for _ in range(n * 2):  # 多跑几轮确保够
        qm = (m - 1) // 3 * 3  # 季末月：3/6/9/12
        if qm == 0:
            qm = 12
            y -= 1
        if qm in (3, 6, 9, 12):
            quarters.append(f"{y}{qm:02d}30" if qm != 12 else f"{y}1231")
            if qm == 12:
                y -= 1
        m = qm - 1
        if m <= 0:
            m = 12
            y -= 1
        if len(quarters) >= n:
            break
    return quarters[:n]


# ════════════════════════════════════════
# 沪深300指数（日）
# ════════════════════════════════════════
def sync_index_daily(db_path: str, symbol: str = "sh000300") -> int:
    """同步沪深300指数日线（全量拉取，幂等UPSERT）。

    数据源：新浪 stock_zh_index_daily，历史2002年至今。
    """
    import akshare as ak
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        if df is None or len(df) == 0:
            return 0
        rows = []
        for _, r in df.iterrows():
            d = str(r["date"])[:10]
            rows.append((
                symbol.replace("sh", "").replace("sz", ""),
                d,
                _f(r.get("open")), _f(r.get("high")),
                _f(r.get("low")), _f(r.get("close")),
                _f(r.get("volume")),
            ))
        if rows:
            with sqlite3.connect(db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO index_daily "
                    "(symbol, date, open, high, low, close, volume) "
                    "VALUES (?,?,?,?,?,?,?)", rows,
                )
                conn.commit()
        logger.info(f"指数 {symbol}: {len(rows)}行")
        return len(rows)
    except Exception as e:
        logger.warning(f"指数同步 {symbol} 失败: {e!r}")
        return 0


# ════════════════════════════════════════════════════════
# 宏观数据（M2 + 社融，月频）
# ════════════════════════════════════════════════════════
def sync_macro_money_supply(db_path: str) -> int:
    """同步货币供应量（M2/M1/M0），全量拉取幂等UPSERT。"""
    import akshare as ak
    try:
        df = ak.macro_china_money_supply()
        if df is None or len(df) == 0:
            return 0
        rows = []
        for _, r in df.iterrows():
            month = _parse_month(r.get("月份", ""))
            if len(month) < 7:
                continue
            rows.append((
                month,
                _f(r.get("货币和准货币(M2)-数量(亿元)")),
                _f(r.get("货币和准货币(M2)-同比增长")),
                _f(r.get("货币(M1)-数量(亿元)")),
                _f(r.get("货币(M1)-同比增长")),
                _f(r.get("流通中的现金(M0)-数量(亿元)")),
                _f(r.get("流通中的现金(M0)-同比增长")),
            ))
        if rows:
            with sqlite3.connect(db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO macro_money "
                    "(month, m2, m2_yoy, m1, m1_yoy, m0, m0_yoy) "
                    "VALUES (?,?,?,?,?,?,?)", rows,
                )
                conn.commit()
        logger.info(f"M2/货币供应: {len(rows)}行")
        return len(rows)
    except Exception as e:
        logger.warning(f"M2同步失败: {e!r}")
        return 0


def sync_macro_social_finance(db_path: str) -> int:
    """同步社会融资规模增量，全量拉取幂等UPSERT。"""
    import akshare as ak
    try:
        df = ak.macro_china_shrzgm()
        if df is None or len(df) == 0:
            return 0
        rows = []
        for _, r in df.iterrows():
            month = _parse_month(r.get("月份", ""))
            if len(month) < 7:
                continue
            rows.append((
                month,
                _f(r.get("社会融资规模增量")),
                _f(r.get("其中-人民币贷款")),
                _f(r.get("其中-委托贷款")),
                _f(r.get("其中-信托贷款")),
                _f(r.get("其中-企业债券")),
                _f(r.get("其中-非金融企业境内股票融资")),
            ))
        if rows:
            with sqlite3.connect(db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO macro_sf "
                    "(month, sf_total, rmb_loan, entrust_loan, trust_loan, "
                    "corp_bond, equity_finance) "
                    "VALUES (?,?,?,?,?,?,?)", rows,
                )
                conn.commit()
        logger.info(f"社融: {len(rows)}行")
        return len(rows)
    except Exception as e:
        logger.warning(f"社融同步失败: {e!r}")
        return 0
