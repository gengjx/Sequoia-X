"""融资融券历史回补（akshare 沪+深合并，按交易日采集）。

离线任务：回补 margin_detail 表历史。
数据源走 akshare（非东财 push2his），不受东财神熔断影响。
仅交易日有数据，节假日返回空自动跳过。

用法：
  PYTHONPATH=. .venv/bin/python -u backfill_margin.py            # 回补1年
  PYTHONPATH=. .venv/bin/python -u backfill_margin.py --days 250 # 指定天数
"""
import os, sys, sqlite3, time, argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from sequoia_x.data.margin_sync import sync_margin_detail
from sequoia_x.core.config import Settings

settings = Settings()
db = settings.db_path


def get_trading_days(days: int) -> list[str]:
    """从 stock_daily 取最近 N 个交易日（YYYYMMDD 格式）。"""
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT ?", (days,)
        ).fetchall()
    return [r[0].replace("-", "") for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=250, help="回补交易日数(默认250≈1年)")
    args = ap.parse_args()

    trading_days = get_trading_days(args.days)
    # 跳过已有的日期
    with sqlite3.connect(db) as conn:
        have = {r[0] for r in conn.execute(
            "SELECT DISTINCT date FROM margin_detail").fetchall()}
    todo = [d for d in trading_days if f"{d[:4]}-{d[4:6]}-{d[6:8]}" not in have]
    todo.sort()  # 升序，从旧到新

    print(f"融资融券回补启动: {time.strftime('%H:%M:%S')}", flush=True)
    print(f"交易日 {len(trading_days)}天, 已有{len(have)}, 待补{len(todo)}天", flush=True)

    t0 = time.time()
    success = 0
    total_rows = 0
    for i, date_str in enumerate(todo):
        rows = sync_margin_detail(db, date_str)
        if rows > 0:
            success += 1
            total_rows += rows
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(todo) - i - 1) / 60
            print(f"  {i+1}/{len(todo)}天, 成功{success}/{total_rows}行, "
                  f"ETA~{eta:.0f}min", flush=True)
        time.sleep(0.5)  # 限速，防封

    elapsed = time.time() - t0
    with sqlite3.connect(db) as conn:
        final = conn.execute("SELECT COUNT(*) FROM margin_detail").fetchone()[0]
        days_cov = conn.execute("SELECT COUNT(DISTINCT date) FROM margin_detail").fetchone()[0]
    print(f"\n融资融券回补完成: {success}/{len(todo)}天, 新增{total_rows}行, 耗时{elapsed:.0f}s",
          flush=True)
    print(f"margin_detail 总计 {final:,}行, {days_cov}个交易日", flush=True)


if __name__ == "__main__":
    main()
