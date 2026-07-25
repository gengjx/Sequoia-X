"""财报历史回补（带额度保护+黑名单检测+自动重连）。

离线一次性任务：回补 stock_finance 表至 5 年深度（20 季度）。
走系统 rate_limiter，到 baostock 日限额或检测到黑名单时自动停止，不空转。

用法：
  PYTHONPATH=. .venv/bin/python -u backfill_finance.py

进度日志：data/.api_rate_limit.json（额度计数）+ stdout
"""
import os, sys, sqlite3, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import baostock as bs
from sequoia_x.data.finance_sync import _recent_quarters
from sequoia_x.core.config import Settings
from sequoia_x.core.rate_limiter import _rate_limiter, BAOSTOCK_DAILY_LIMIT

settings = Settings()
db = settings.db_path
N_QUARTERS = 20


def _sf(v):
    try:
        return float(v) if v not in (None, "", "nan") else None
    except (ValueError, TypeError):
        return None


def is_blacklisted(error_msg: str) -> bool:
    """检测 baostock 黑名单/超额错误。"""
    msg = str(error_msg or "")
    return ("黑名单" in msg or "10001011" in msg
            or "exceed" in msg.lower())


def ensure_login():
    """确保 baostock 已登录。返回 True/False/None（None=黑名单）。"""
    for attempt in range(5):
        try:
            lg = bs.login()
            if lg.error_code == "0":
                return True
            if is_blacklisted(lg.error_msg):
                print(f"!! baostock 黑名单：{lg.error_msg}，停止回补", flush=True)
                return None
        except Exception:
            pass
        time.sleep(3 * (attempt + 1))
    return False


def fetch_one(symbol, quarters):
    """采集单只股票。返回 (records, ok)；黑名单/超额时 ok=False。"""
    bs_code = "sh." + symbol if symbol.startswith(("6", "9")) else "sz." + symbol
    records = []
    for year, quarter in quarters:
        for label in ["profit", "growth", "operation"]:
            # ── 额度检查：每次 query 前 check + consume ──
            if not _rate_limiter.baostock_check(1):
                print(f"!! baostock 日限额达 {BAOSTOCK_DAILY_LIMIT}，停止回补", flush=True)
                return records, False
            _rate_limiter.baostock_consume(1)
            for retry in range(3):
                try:
                    if label == "profit":
                        rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                    elif label == "growth":
                        rs = bs.query_growth_data(code=bs_code, year=year, quarter=quarter)
                    else:
                        rs = bs.query_operation_data(code=bs_code, year=year, quarter=quarter)
                    if is_blacklisted(getattr(rs, "error_msg", "")):
                        print("!! baostock 查询返回黑名单，停止", flush=True)
                        return records, False
                    while rs.next():
                        row = rs.get_row_data()
                        rec = {"symbol": symbol}
                        rec["stat_date"] = row[1] if len(row) > 1 else f"{year}-Q{quarter}"
                        rec["report_date"] = row[2] if len(row) > 2 else None
                        rec["roe"] = _sf(row[3]) if len(row) > 3 else None
                        rec["np_margin"] = _sf(row[4]) if len(row) > 4 else None
                        rec["gp_margin"] = _sf(row[5]) if len(row) > 5 else None
                        rec["net_profit"] = _sf(row[6]) if len(row) > 6 else None
                        rec["eps_ttm"] = _sf(row[7]) if len(row) > 7 else None
                        rec["revenue"] = _sf(row[8]) if len(row) > 8 else None
                        records.append(rec)
                    break
                except Exception:
                    if retry < 2:
                        time.sleep(2)
                        ok = ensure_login()
                        if ok is None:
                            return records, False
    return records, True


# ── 主流程 ──
with sqlite3.connect(db) as conn:
    all_symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM stock_daily ORDER BY symbol").fetchall()]
    have_full = {r[0] for r in conn.execute(
        "SELECT symbol FROM stock_finance GROUP BY symbol "
        "HAVING COUNT(DISTINCT stat_date) >= 18").fetchall()}

todo = [s for s in all_symbols if s not in have_full]
quarters = _recent_quarters(N_QUARTERS)
status = _rate_limiter.baostock_status()
print(f"财报回补启动: {time.strftime('%H:%M:%S')}", flush=True)
print(f"待补{len(todo)}只（已完成{len(have_full)}只跳过）", flush=True)
print(f"baostock 额度: 已用 {status['used']}/{BAOSTOCK_DAILY_LIMIT}，"
      f"剩余 {status['remaining']}", flush=True)

ok = ensure_login()
if ok is None:
    print("!! baostock 当前在黑名单中，无法回补。请24小时后重试。", flush=True)
    sys.exit(1)
if not ok:
    print("baostock 登录失败，退出", flush=True)
    sys.exit(1)

sql = ("INSERT OR REPLACE INTO stock_finance "
       "(symbol,stat_date,report_date,roe,np_margin,gp_margin,net_profit,eps_ttm,revenue,"
       "yoy_equity,yoy_asset,yoy_ni,yoy_eps,yoy_pni,nr_turn,inv_turn,asset_turn) "
       "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")

t0 = time.time()
done = 0
halted = False

for sym in todo:
    try:
        records, ok = fetch_one(sym, quarters)
        if not ok:
            halted = True
            break
        if records:
            for r in records:
                for k in ["yoy_equity", "yoy_asset", "yoy_ni", "yoy_eps", "yoy_pni",
                          "nr_turn", "inv_turn", "asset_turn"]:
                    r.setdefault(k, None)
            with sqlite3.connect(db, timeout=30) as conn:
                conn.executemany(sql, [(
                    r["symbol"], r.get("stat_date"), r.get("report_date"),
                    r.get("roe"), r.get("np_margin"), r.get("gp_margin"),
                    r.get("net_profit"), r.get("eps_ttm"), r.get("revenue"),
                    r.get("yoy_equity"), r.get("yoy_asset"), r.get("yoy_ni"),
                    r.get("yoy_eps"), r.get("yoy_pni"),
                    r.get("nr_turn"), r.get("inv_turn"), r.get("asset_turn"),
                ) for r in records])
                conn.commit()
    except Exception as e:
        print(f"  {sym} 异常: {e!r}", flush=True)
        if is_blacklisted(repr(e)):
            halted = True
            break
        ensure_login()
    done += 1
    if done % 20 == 0 or done >= len(todo):
        elapsed = time.time() - t0
        with sqlite3.connect(db, timeout=30) as conn:
            full = conn.execute(
                "SELECT COUNT(*) FROM (SELECT symbol FROM stock_finance "
                "GROUP BY symbol HAVING COUNT(DISTINCT stat_date)>=18)").fetchone()[0]
        st = _rate_limiter.baostock_status()
        eta = elapsed / done * (len(todo) - done) / 60
        print(f"  {done}/{len(todo)}只, ≥18季:{full}只, "
              f"额度{st['used']}/{BAOSTOCK_DAILY_LIMIT}({st['usage_pct']}%), "
              f"ETA~{eta:.0f}min", flush=True)

bs.logout()
with sqlite3.connect(db, timeout=30) as conn:
    qc = conn.execute("SELECT COUNT(DISTINCT stat_date) FROM stock_finance").fetchone()[0]
    full = conn.execute(
        "SELECT COUNT(*) FROM (SELECT symbol FROM stock_finance "
        "GROUP BY SYMBOL HAVING COUNT(DISTINCT stat_date)>=18)").fetchone()[0]
reason = "额度耗尽/黑名单自动停止" if halted else "全部完成"
print(f"\n财报回补结束({reason}): ≥18季 {full}只, {qc}季度", flush=True)
