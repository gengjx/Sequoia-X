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


# ── akshare 东财财报摘要 → stock_finance 字段映射 ──
# 键=(选项,指标)，值=(目标列名, 除数)。除数 100=百分数转小数，1=原值。
_AK_MAP: dict[tuple[str, str], tuple[str, float]] = {
    ("盈利能力", "净资产收益率(ROE)"): ("roe", 100.0),
    ("盈利能力", "销售净利率"): ("np_margin", 100.0),
    ("盈利能力", "毛利率"): ("gp_margin", 100.0),
    ("常用指标", "归母净利润"): ("net_profit", 1.0),
    ("每股指标", "基本每股收益"): ("eps_ttm", 1.0),
    ("常用指标", "营业总收入"): ("revenue", 1.0),
    ("成长能力", "归属母公司净利润增长率"): ("yoy_ni", 100.0),
    ("成长能力", "营业总收入增长率"): ("yoy_eps", 100.0),
    ("营运能力", "总资产周转率"): ("asset_turn", 1.0),
    ("营运能力", "存货周转率"): ("inv_turn", 1.0),
    ("营运能力", "应收账款周转率"): ("nr_turn", 1.0),
    ("收益质量", "经营性现金净流量/营业总收入"): ("cfo_to_or", 1.0),
    ("收益质量", "经营活动净现金/归属母公司的净利润"): ("cfo_to_np", 1.0),
    # 偿债能力 + 杜邦杠杆（财务风险分类，同一 API 返回，零额外调用）
    ("常用指标", "资产负债率"): ("liability_to_asset", 100.0),
   ("财务风险", "权益乘数"): ("equity_multiplier", 1.0),
}


# 多进程 worker
def _fetch_batch(args: tuple) -> list[dict]:
    """worker：独立 login，批量采集一批股票的财报。"""
    import baostock as bs

    from sequoia_x.core.rate_limiter import _rate_limiter

    symbols, quarters = args
    bs.login()
    _rate_limiter.baostock_try_consume(1)  # login 算1次额度
    results: list[dict] = []
    cols = [
        "symbol", "stat_date", "report_date", "roe", "np_margin", "gp_margin",
        "net_profit", "eps_ttm", "revenue", "yoy_equity", "yoy_asset", "yoy_ni",
        "yoy_eps", "yoy_pni", "nr_turn", "inv_turn", "asset_turn",
        "cfo_to_or", "cfo_to_np", "cfo_to_gr", "tangible_ratio",
        "liability_to_asset", "equity_multiplier",
    ]

    quota_exhausted = False
    for symbol in symbols:
        if quota_exhausted:
            break
        bs_code = _to_baostock_code(symbol)
        for year, quarter in quarters:
            if not _rate_limiter.baostock_try_consume(4):  # 每季4类query
                quota_exhausted = True
                break
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

                rc = bs.query_cash_flow_data(code=bs_code, year=year, quarter=quarter)
                while rc.next():
                    c = rc.get_row_data()
                    rec["tangible_ratio"] = _sf(c[5])
                    rec["cfo_to_or"] = _sf(c[7])
                    rec["cfo_to_np"] = _sf(c[8])
                    rec["cfo_to_gr"] = _sf(c[9])

                # 资产负债表（query_balance_data）：资产负债率 = 负债总额/资产总额
                rb = bs.query_balance_data(code=bs_code, year=year, quarter=quarter)
                while rb.next():
                    b = rb.get_row_data()
                    rec["liability_to_asset"] = _sf(b[7])

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
                 batch_size: int = 50, max_stocks: int | None = None,
                 deep_sync: bool = False) -> dict:
        """全量采集财报。

        Args:
            n_quarters: 采集最近几个季度
            n_workers: 并行进程数
            batch_size: 每个 worker 分多少只股票
            max_stocks: 最多采几只（None=全部）
            deep_sync: 增量补深模式——除完全缺失外，也采集
                季度数 < n_quarters 的股票（INSERT OR REPLACE 自动补缺，
                不删除已有数据）

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
                "cfo_to_or REAL, cfo_to_np REAL, cfo_to_gr REAL, tangible_ratio REAL,"
                "liability_to_asset REAL, equity_multiplier REAL,"
                "PRIMARY KEY (symbol, stat_date))"
            )
            # 幂等迁移：现有表补现金流比率列（P15 盈利质量因子）
            for col in ("cfo_to_or", "cfo_to_np", "cfo_to_gr", "tangible_ratio"):
                try:
                    conn.execute(
                        f"ALTER TABLE stock_finance ADD COLUMN {col} REAL"
                    )
                except sqlite3.OperationalError:
                    pass  # 列已存在
            # 幂等迁移：资产负债率 + 权益乘数（P18 偿债能力+杜邦杠杆因子）
            for col in ("liability_to_asset", "equity_multiplier"):
                try:
                    conn.execute(
                        f"ALTER TABLE stock_finance ADD COLUMN {col} REAL"
                    )
                except sqlite3.OperationalError:
                    pass
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

        if deep_sync:
            # 增量补深：完全缺失 + 季度数不足的股票都纳入采集
            # INSERT OR REPLACE 天然补缺，绝不删除已有数据
            with sqlite3.connect(self.db_path) as conn:
                thin = {r[0] for r in conn.execute(
                    "SELECT symbol FROM stock_finance GROUP BY symbol "
                    "HAVING COUNT(*) < ?",
                    (n_quarters,),
                ).fetchall()}
            missing = sorted(
                (s for s in all_symbols if s not in have or s in thin)
            )
        else:
            missing = [s for s in all_symbols if s not in have]

        # ── baostock 额度预算保护 ──
        # 每只股票 = n_quarters 季度 × 4 类财报(profit/growth/operation/cashflow)
        from sequoia_x.core.rate_limiter import _rate_limiter
        calls_per_stock = n_quarters * 4
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

        # baostock 额度已在 worker 内逐次实时扣减（baostock_try_consume），
        # 无需事后批量记账

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

    # ── akshare 路径（不依赖 baostock 额度，天然增量补缺）──
    def collect_one_akshare(self, symbol: str, n_quarters: int = 20) -> int:
        """用 akshare 东财财报摘要采集单只股票，INSERT OR REPLACE 写入。

        不依赖 baostock 额度；天然增量补缺（不删除已有数据）。
        返回写入的季度数。
        """
        import re

        import akshare as ak

        try:
            df = ak.stock_financial_abstract(symbol=symbol)
        except Exception as e:
            logger.warning(f"akshare 财报采集失败 {symbol}: {e!r}")
            return 0
        if df is None or df.empty or "指标" not in df.columns:
            return 0

        period_cols = [
            c for c in df.columns
            if c not in ("选项", "指标") and re.match(r"^\d{8}$", str(c))
        ]
        if not period_cols:
            return 0
        period_cols = sorted(period_cols, reverse=True)[:n_quarters]

        # 构建 (选项,指标) → 行，去重保留首个（常见指标与分类指标重复）
        lookup: dict[tuple[str, str], object] = {}
        for _, row in df.iterrows():
            key = (str(row.get("选项", "")), str(row.get("指标", "")))
            if key not in lookup:
                lookup[key] = row

        records: list[dict] = []
        for col in period_cols:
            stat_date = f"{col[:4]}-{col[4:6]}-{col[6:]}"
            rec: dict = {"symbol": symbol, "stat_date": stat_date, "report_date": stat_date}
            for (cat, ind), (field, divisor) in _AK_MAP.items():
                row = lookup.get((cat, ind))
                if row is None:
                    continue
                fval = _sf(row.get(col))
                if fval is not None:
                    rec[field] = fval / divisor
            if "yoy_ni" in rec:
                rec["yoy_pni"] = rec["yoy_ni"]
            if "cfo_to_or" in rec:
                rec["cfo_to_gr"] = rec["cfo_to_or"]
            if len(rec) > 3:
                records.append(rec)

        if not records:
            return 0

        cols = [
            "symbol", "stat_date", "report_date", "roe", "np_margin", "gp_margin",
            "net_profit", "eps_ttm", "revenue", "yoy_equity", "yoy_asset", "yoy_ni",
            "yoy_eps", "yoy_pni", "nr_turn", "inv_turn", "asset_turn",
            "cfo_to_or", "cfo_to_np", "cfo_to_gr", "tangible_ratio",
            "liability_to_asset", "equity_multiplier",
        ]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO stock_finance ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                [tuple(rec.get(c) for c in cols) for rec in records],
            )
            conn.commit()
        return len(records)

    def sync_all_akshare(self, n_quarters: int = 20,
                         max_stocks: int | None = None,
                         delay: float = 0.3) -> dict:
        """用 akshare 全量/增量采集财报（不依赖 baostock 额度）。

        增量逻辑：跳过已有 ≥ n_quarters 季度数据的股票（深度已够）。
        幂等：INSERT OR REPLACE，绝不删除已有数据。
        """
        t0 = time.time()

        with sqlite3.connect(self.db_path) as conn:
            all_symbols = [r[0] for r in conn.execute(
                "SELECT symbol FROM stock_basic ORDER BY symbol"
            ).fetchall()]
            full = {r[0] for r in conn.execute(
                "SELECT symbol FROM stock_finance GROUP BY symbol "
                "HAVING COUNT(*) >= ?",
                (n_quarters,),
            ).fetchall()}

        todo = [s for s in all_symbols if s not in full]
        if max_stocks:
            todo = todo[:max_stocks]

        logger.info(
            f"akshare 财报采集：全市场 {len(all_symbols)} 只，"
            f"深度已够 {len(full)} 只，待采 {len(todo)} 只"
        )

        done = 0
        written = 0
        for sym in todo:
            written += self.collect_one_akshare(sym, n_quarters=n_quarters)
            done += 1
            time.sleep(delay)
            if done % 50 == 0:
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                eta = (len(todo) - done) / speed if speed > 0 else 0
                logger.info(
                    f"akshare 财报进度：{done}/{len(todo)} 只，"
                    f"写入 {written} 季，ETA {eta:.0f}s"
                )

        elapsed = time.time() - t0
        with sqlite3.connect(self.db_path) as conn:
            final_count = conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM stock_finance"
            ).fetchone()[0]

        logger.info(
            f"akshare 财报采集完成：本次 {done} 只，写入 {written} 季，"
            f"覆盖率 {final_count}/{len(all_symbols)} "
            f"({final_count / len(all_symbols) * 100:.1f}%)，耗时 {elapsed:.0f}s"
        )
        return {
            "total": len(all_symbols),
            "fetched": done,
            "written": written,
            "coverage": final_count,
            "coverage_pct": round(final_count / len(all_symbols) * 100, 1),
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


def _stat_date_to_yq(stat_date: str | None) -> tuple[int, int] | None:
    """YYYY-MM-DD -> (year, quarter)。"""
    if not stat_date or len(stat_date) < 7:
        return None
    try:
        y = int(stat_date[:4])
        m = int(stat_date[5:7])
        return (y, (m - 1) // 3 + 1)
    except (ValueError, IndexError):
        return None


def _fetch_cashflow_batch(args: tuple) -> int:
    """worker：仅拉现金流比率，逐行 UPDATE。"""
    import baostock as bs

    from sequoia_x.core.rate_limiter import _rate_limiter

    (rows,) = args
    bs.login()
    _rate_limiter.baostock_try_consume(1)
    updates: list[tuple] = []
    for symbol, stat_date in rows:
        if not _rate_limiter.baostock_try_consume(1):
            break
        yq = _stat_date_to_yq(stat_date)
        if not yq:
            continue
        year, quarter = yq
        bs_code = _to_baostock_code(symbol)
        try:
            rc = bs.query_cash_flow_data(code=bs_code, year=year, quarter=quarter)
            while rc.next():
                c = rc.get_row_data()
                updates.append((
                    _sf(c[5]),  # tangible_ratio
                    _sf(c[7]),  # cfo_to_or
                    _sf(c[8]),  # cfo_to_np
                    _sf(c[9]),  # cfo_to_gr
                    symbol, stat_date,
                ))
                break  # 单季度单行
        except Exception:
            continue
    bs.logout()

    if updates:
        db_path = Settings().db_path
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                "UPDATE stock_finance SET tangible_ratio=?, cfo_to_or=?, "
                "cfo_to_np=?, cfo_to_gr=? WHERE symbol=? AND stat_date=?",
                updates,
            )
            conn.commit()
    return len(updates)


def backfill_cash_flow(settings=None, batch_size: int = 200,
                       max_rows: int | None = None, n_workers: int = 3) -> dict:
    """增量回补现金流比率（仅 cfo_to_or IS NULL 的行）。

    P15 专用：已有 profit/growth/operation 但缺现金流的行，专项补
    query_cash_flow_data（每行 1 次调用），比全量重拉省 75% 额度。

    用法：
        python -c "from sequoia_x.data.finance_sync import backfill_cash_flow; backfill_cash_flow()"
    """
    syncer = FinanceSync(settings)
    t0 = time.time()

    with sqlite3.connect(syncer.db_path) as conn:
        null_rows = conn.execute(
            "SELECT symbol, stat_date FROM stock_finance "
            "WHERE cfo_to_or IS NULL ORDER BY symbol, stat_date"
        ).fetchall()

    total_null = len(null_rows)
    if total_null == 0:
        logger.info("现金流回补：无缺失行，跳过")
        return {"total": 0, "fetched": 0, "elapsed": 0}

    # baostock 额度保护（每行 1 次调用）
    from sequoia_x.core.rate_limiter import _rate_limiter
    rl_status = _rate_limiter.baostock_status()
    budget = rl_status["remaining"]
    if budget == 0:
        logger.error(
            f"baostock 额度已耗尽({rl_status['used']}/{rl_status['limit']})，"
            f"现金流回补取消，明天 00:00 重置后继续"
        )
        return {"total": total_null, "fetched": 0, "rate_limited": True, "elapsed": 0}

    todo = null_rows[:max_rows] if max_rows else null_rows
    if len(todo) > budget:
        logger.warning(
            f"额度保护：需回补 {len(todo)} 行，超出 baostock 剩余 {budget}，截断"
        )
        todo = todo[:budget]

    logger.info(
        f"现金流回补：缺失 {total_null} 行，本次回补 {len(todo)} 行，{n_workers} 进程并行"
    )

    chunks = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    chunk_args = [(c,) for c in chunks]
    total_fetched = 0
    with Pool(n_workers) as pool:
        for i, cnt in enumerate(pool.imap_unordered(_fetch_cashflow_batch, chunk_args)):
            total_fetched += cnt
            done = min((i + 1) * batch_size, len(todo))
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            eta = (len(todo) - done) / speed if speed > 0 else 0
            logger.info(
                f"现金流回补进度：{done}/{len(todo)} 行 "
                f"({speed:.0f} 行/秒，ETA {eta:.0f}s)"
            )

    # baostock 额度已在 worker 内逐次实时扣减
    elapsed = time.time() - t0

    with sqlite3.connect(syncer.db_path) as conn:
        remaining_null = conn.execute(
            "SELECT COUNT(*) FROM stock_finance WHERE cfo_to_or IS NULL"
        ).fetchone()[0]

    logger.info(
        f"现金流回补完成：本次 {total_fetched}/{len(todo)} 行，"
        f"剩余缺失 {remaining_null} 行，耗时 {elapsed:.0f}s"
    )
    return {
        "total": total_null,
        "fetched": total_fetched,
        "remaining": remaining_null,
        "elapsed": round(elapsed, 0),
    }
