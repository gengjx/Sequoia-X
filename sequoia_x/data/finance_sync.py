"""财报全量采集引擎：多进程并行从 baostock 采集全市场财报。

每只股票采集最近6个季度的3类财报（profit/growth/operation），
多进程并行加速。支持增量采集（跳过已有的）。
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from multiprocessing import Pool

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


def _recent_quarters(n: int = 6) -> list[tuple[int, int]]:
    """生成最近 n 个 (year, quarter) 列表（倒序）。"""
    now = datetime.now()
    year, month = now.year, now.month
    quarters: list[tuple[int, int]] = []
    q = (month - 1) // 3  # 0-3
    for _ in range(n):
        if q == 0:
            year -= 1
            q = 4
        quarters.append((year, q))
        q -= 1
    return quarters


def _to_baostock_code(symbol: str) -> str:
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    return f"{prefix}.{symbol}"


def _sf(v) -> float | None:
    """安全转float。"""
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# 多进程 worker
def _fetch_batch(args: tuple) -> list[dict]:
    """worker：独立 login，批量采集一批股票的财报。"""
    import baostock as bs

    symbols, quarters = args
    bs.login()
    results: list[dict] = []
    cols = [
        "symbol", "stat_date", "report_date", "roe", "np_margin", "gp_margin",
        "net_profit", "eps_ttm", "revenue", "yoy_equity", "yoy_asset", "yoy_ni",
        "yoy_eps", "yoy_pni", "nr_turn", "inv_turn", "asset_turn",
    ]

    for symbol in symbols:
        bs_code = _to_baostock_code(symbol)
        for year, quarter in quarters:
            rec = {"symbol": symbol, "stat_date": None}
            try:
                rp = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                if rp.error_code != "0":
                    continue
                p = None
                while rp.next():
                    p = rp.get_row_data()
                if not p:
                    continue
                rec["report_date"], rec["stat_date"] = p[1], p[2]
                rec["roe"] = _sf(p[3])
                rec["np_margin"] = _sf(p[4])
                rec["gp_margin"] = _sf(p[5])
                rec["net_profit"] = _sf(p[6])
                rec["eps_ttm"] = _sf(p[7])
                rec["revenue"] = _sf(p[8])

                rg = bs.query_growth_data(code=bs_code, year=year, quarter=quarter)
                while rg.next():
                    g = rg.get_row_data()
                    rec["yoy_equity"] = _sf(g[3])
                    rec["yoy_asset"] = _sf(g[4])
                    rec["yoy_ni"] = _sf(g[5])
                    rec["yoy_eps"] = _sf(g[6])
                    rec["yoy_pni"] = _sf(g[7])

                ro = bs.query_operation_data(code=bs_code, year=year, quarter=quarter)
                while ro.next():
                    o = ro.get_row_data()
                    rec["nr_turn"] = _sf(o[3])
                    rec["inv_turn"] = _sf(o[5])
                    rec["asset_turn"] = _sf(o[8])

                results.append(rec)
            except Exception:
                continue

    bs.logout()

    # 批量入库
    if results:
        db_path = Settings().db_path
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO stock_finance ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                [tuple(rec.get(c) for c in cols) for rec in results],
            )
            conn.commit()
    return results


class FinanceSync:
    """财报全量采集引擎。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.db_path = self.settings.db_path

    def sync_all(self, n_quarters: int = 6, n_workers: int = 3,
                 batch_size: int = 50, max_stocks: int | None = None) -> dict:
        """全量采集财报。

        Args:
            n_quarters: 采集最近几个季度
            n_workers: 并行进程数
            batch_size: 每个 worker 分多少只股票
            max_stocks: 最多采几只（None=全部）

        Returns:
            {total, fetched, skipped, failed, elapsed}
        """
        t0 = time.time()

        # 建表（确保存在）
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS stock_finance ("
                "symbol TEXT NOT NULL, stat_date TEXT NOT NULL, report_date TEXT,"
                "roe REAL, np_margin REAL, gp_margin REAL, net_profit REAL, eps_ttm REAL, revenue REAL,"
                "yoy_equity REAL, yoy_asset REAL, yoy_ni REAL, yoy_eps REAL, yoy_pni REAL,"
                "nr_turn REAL, inv_turn REAL, asset_turn REAL,"
                "PRIMARY KEY (symbol, stat_date))"
            )
            conn.commit()

        # 获取全市场代码
        with sqlite3.connect(self.db_path) as conn:
            all_symbols = [r[0] for r in conn.execute(
                "SELECT symbol FROM stock_basic ORDER BY symbol"
            ).fetchall()]
            # 已有财报的跳过
            have = {r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM stock_finance"
            ).fetchall()}

        missing = [s for s in all_symbols if s not in have]

        # ── baostock 额度预算保护 ──
        # 每只股票 = n_quarters 季度 × 3 类财报(profit/growth/operation) = 18 次查询
        from sequoia_x.core.rate_limiter import _rate_limiter
        calls_per_stock = n_quarters * 3
        rl_status = _rate_limiter.baostock_status()
        budget_stocks = rl_status["remaining"] // calls_per_stock if calls_per_stock > 0 else 0
        if budget_stocks == 0:
            logger.error(
                f"baostock 额度已耗尽，财报采集取消。"
                f"已用 {rl_status['used']}/{rl_status['limit']}，"
                f"明天自动重置"
            )
            return {
                "total": len(all_symbols),
                "fetched": 0,
                "skipped": len(all_symbols) - len(missing),
                "failed": len(missing),
                "elapsed": 0,
                "rate_limited": True,
            }
        if max_stocks is None or max_stocks > budget_stocks:
            if len(missing) > budget_stocks:
                logger.warning(
                    f"额度保护：需采集 {len(missing)} 只 × {calls_per_stock} 次查询，"
                    f"超出 baostock 剩余额度({rl_status['remaining']})，"
                    f"截断为 {budget_stocks} 只，剩余明天继续"
                )
            max_stocks = min(max_stocks or budget_stocks, budget_stocks)

        if max_stocks:
            missing = missing[:max_stocks]

        logger.info(
            f"财报全量采集：全市场 {len(all_symbols)} 只，"
            f"已有 {len(all_symbols) - len(missing)} 只，"
            f"需采集 {len(missing)} 只，{n_workers} 进程并行"
        )

        if not missing:
            return {
                "total": len(all_symbols),
                "fetched": 0,
                "skipped": len(all_symbols),
                "failed": 0,
                "elapsed": 0,
            }

        quarters = _recent_quarters(n_quarters)

        # 分块
        chunks = []
        for i in range(0, len(missing), batch_size):
            chunk = missing[i:i + batch_size]
            chunks.append((chunk, quarters))

        # 多进程并行
        total_fetched = 0
        with Pool(n_workers) as pool:
            for i, batch_result in enumerate(pool.imap_unordered(_fetch_batch, chunks)):
                total_fetched += len(batch_result)
                done = (i + 1) * batch_size
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                eta = (len(missing) - done) / speed if speed > 0 else 0
                logger.info(
                    f"财报采集进度：{min(done, len(missing))}/{len(missing)} 只 "
                    f"({speed:.0f} 只/秒，ETA {eta:.0f}s)"
                )

        elapsed = time.time() - t0

        # 消耗 baostock 额度计数（实际采集数 × 每只查询次数）
        from sequoia_x.core.rate_limiter import _rate_limiter as _rl2
        _rl2.baostock_consume(len(missing) * calls_per_stock)

        # 验证
        with sqlite3.connect(self.db_path) as conn:
            final_count = conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM stock_finance"
            ).fetchone()[0]

        logger.info(
            f"财报全量采集完成：本次采集 {len(missing)} 只，"
            f"共写入 {total_fetched} 季，"
            f"覆盖率 {final_count}/{len(all_symbols)} "
            f"({final_count / len(all_symbols) * 100:.1f}%)，耗时 {elapsed:.0f}s"
        )

        return {
            "total": len(all_symbols),
            "fetched": len(missing),
            "skipped": len(all_symbols) - len(missing),
            "records": total_fetched,
            "coverage": final_count,
            "coverage_pct": round(final_count / len(all_symbols) * 100, 1),
            "failed": len(missing) - (final_count - (len(all_symbols) - len(missing))),
            "elapsed": round(elapsed, 0),
        }


def backfill_finance_history(settings=None, n_quarters: int = 20, max_stocks: int | None = None) -> dict:
    """一次性离线回补财报历史（默认 20 个季度 ≈ 5 年）。

    与每日增量同步（n_quarters=6）不同，本函数拉取更长时间窗口的历史财报，
    使基本面因子的 IC/t-stat 评估有充足样本。

    用法：
        python -c "from sequoia_x.data.finance_sync import backfill_finance_history; backfill_finance_history()"
    或通过 Web 任务触发。复用 baostock 额度保护，超额自动截断分批。
    """
    syncer = FinanceSync(settings)
    return syncer.sync_all(n_quarters=n_quarters, n_workers=3, max_stocks=max_stocks)
