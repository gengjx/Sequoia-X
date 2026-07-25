"""资金流向历史回补（高价值池 + 慢速限流 + 熔断自动停止）。

离线任务：回补 fund_flow 表至 1 年深度（250 交易日）。
限定高价值池（circ_mv 前800 + 持仓 + 决策池），而非全市场——东财 push2his
限流极严，全量 5000 只会快速触发 IP 封禁。

特点：
  - 串行执行（非并发），每只间隔 1.5 秒（低于东财 0.4s 最小间隔的并发触发线）
  - 单批最多 80 只，到熔断阈值(连续失败10次)立即停止，避免长时间空转
  - 幂等：已有 ≥200 行的股票跳过，支持多轮增量补

用法：
  PYTHONPATH=. .venv/bin/python -u backfill_fundflow.py
  PYTHONPATH=. .venv/bin/python -u backfill_fundflow.py --batch 60 --interval 2.0
"""
import os, sys, sqlite3, time, argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from sequoia_x.data.fund_flow_history import fetch_fund_flow_history
from sequoia_x.core.config import Settings
from sequoia_x.core.rate_limiter import _rate_limiter, MAX_CONSECUTIVE_FAILURES

settings = Settings()
db = settings.db_path
DAYS = 250
MIN_DEPTH = 200  # 已有≥200行视为足够，跳过


def get_target_symbols() -> list[str]:
    """高价值池：circ_mv 前800 + 持仓 + 决策池。"""
    with sqlite3.connect(db) as conn:
        top800 = {r[0] for r in conn.execute(
            "SELECT symbol FROM stock_market_cap WHERE circ_mv>0 "
            "ORDER BY circ_mv DESC LIMIT 800").fetchall()}
        holdings = {r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM paper_holdings").fetchall()}
        pool = {r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM decision_pool").fetchall()}
        return sorted(top800 | holdings | pool)


def get_todo(target: list[str], max_consec_fail: int = 8) -> list[str]:
    """筛选待补：缺失 或 深度<MIN_DEPTH。"""
    with sqlite3.connect(db) as conn:
        have = dict(conn.execute(
            "SELECT symbol, COUNT(*) FROM fund_flow GROUP BY symbol").fetchall())
    todo = [s for s in target if have.get(s, 0) < MIN_DEPTH]
    return todo


def write_rows(rows: list[dict]) -> int:
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executemany(
            "INSERT OR REPLACE INTO fund_flow "
            "(symbol, date, main_net, main_pct, super_net, big_net, mid_net, small_net) "
            "VALUES (?, ?, ?, 0, ?, ?, ?, ?)",
            [(r["symbol"], r["date"], r["main_net"], r["super_net"],
              r["big_net"], r["mid_net"], r["small_net"]) for r in rows],
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=80, help="单批最多处理只数")
    ap.add_argument("--interval", type=float, default=1.5, help="每只间隔秒")
    ap.add_argument("--max-fail", type=int, default=8, help="连续失败N次停止")
    ap.add_argument("--wait", action="store_true", help="熔断中时等待解除再跑(每5分钟检查)")
    args = ap.parse_args()

    target = get_target_symbols()
    todo = get_todo(target)
    print(f"资金流回补启动: {time.strftime('%H:%M:%S')}", flush=True)
    print(f"目标池 {len(target)}只, 待补 {len(todo)}只 (深度<{MIN_DEPTH}行)", flush=True)
    print(f"参数: batch={args.batch} interval={args.interval}s max_fail={args.max_fail}", flush=True)

    # 熔断预检查：--wait 时轮询等待解除，否则直接退出
    while _rate_limiter._is_circuit_breaker("eastmoney"):
        cb = _rate_limiter._circuit_until.get("eastmoney", 0)
        remain = max(0, int(cb - time.time()))
        if not args.wait:
            print(f"!! 东财熔断中(剩{remain//60}分)，加 --wait 可等待解除", flush=True)
            sys.exit(1)
        print(f"东财熔断中(剩{remain//60}分{remain%60}秒)，5分钟后重试...", flush=True)
        time.sleep(300)
    print("东财熔断已解除，开始回补", flush=True)

    t0 = time.time()
    done = 0
    success = 0
    fail = 0
    consec_fail = 0
    halted = False

    for sym in todo[:args.batch]:
        rows = fetch_fund_flow_history(sym, days=DAYS)
        if rows:
            n = write_rows(rows)
            success += 1
            consec_fail = 0
            done += 1
        else:
            fail += 1
            consec_fail += 1
            done += 1
            if consec_fail >= args.max_fail:
                print(f"!! 连续失败 {consec_fail} 次，停止本轮（防长时间空转）", flush=True)
                halted = True
                break
        time.sleep(args.interval)
        if done % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {done}只: 成功{success} 失败{fail} "
                  f"连续失败{consec_fail} 耗时{elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    # 最终统计
    with sqlite3.connect(db) as conn:
        total_rows = conn.execute("SELECT COUNT(*) FROM fund_flow").fetchone()[0]
        deep = conn.execute(
            f"SELECT COUNT(*) FROM (SELECT symbol FROM fund_flow "
            f"GROUP BY symbol HAVING COUNT(*)>={MIN_DEPTH})").fetchone()[0]
    print(f"\n资金流回补结束({'熔断/连续失败停止' if halted else '完成'}): "
          f"成功{success} 失败{fail} 耗时{elapsed:.0f}s", flush=True)
    print(f"fund_flow 总计 {total_rows:,}行, 深度≥{MIN_DEPTH}行: {deep}只", flush=True)


if __name__ == "__main__":
    main()
