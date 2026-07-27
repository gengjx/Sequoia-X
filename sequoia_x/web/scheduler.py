"""轻量定时调度器：基于 threading.Timer，无外部依赖。

A股交易时段调度（Asia/Shanghai）：
  09:25  → 集合竞价扫描 + 飞书推送
  18:00  → 日K数据自动同步（解决手动同步痛点）
  19:00  → 策略评估权重刷新（可选）

非交易日（周末/节假日）自动跳过。
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class AuctionScheduler:
    """后台调度线程：每日定时执行竞价扫描与数据同步。"""

    # 定时任务表：(hour, minute, task_name)
    SCHEDULE = [
        (0, 15, "backfill_offline"),  # 凌晨baostock额度恢复后离线回补（现金流+缺失财报）
        (9, 25, "auction_scan"),
        (9, 30, "intraday_scan_start"),  # 启动盘中持仓监控
        (21, 0, "sync_daily"),       # 避开baostock盘后高峰(18-21点拥堵)
        (21, 5, "sync_lhb"),         # 龙虎榜数据同步（紧跟日K之后）
        (21, 6, "sync_fund_flow"),   # 主力资金流向同步
        (21, 7, "sync_margin"),      # 融资融券增量同步（杠杆资金方向，北向断供替代）
        (21, 8, "sync_valuation"),  # PE/PB估值同步（东财快照，全市场3秒）
        (21, 9, "sync_altdata"),  # 另类数据：沪深300+基金持仓+M2/社融增量
        (21, 12, "refresh_factor_ic"),  # 因子IC权重刷新（滚动6个月窗口）
        (21, 30, "auction_verify"),  # 同步完成后验证T+1命中
        (21, 40, "paper_trade"),  # 模拟盘：盘后选股→买入→卖出闭环
        (22, 0, "daily_report"),  # 每日任务执行汇总 → 飞书推送
        (22, 30, "monthly_sweep"),  # 每月1号参数扫描验证（非1号自动跳过）
    ]

    def __init__(self, settings: Settings, db_path: str) -> None:
        self.settings = settings
        self.db_path = db_path
        self.engine = None  # 延迟初始化
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_run: dict[str, str] = {}  # {task_name: "YYYY-MM-DD"}
        self._init_task_log_table()
        self._load_last_run()
        self._cleanup_zombies()
        self._last_wal_checkpoint = 0.0
        self._last_attribution: dict | None = None

    def _init_task_log_table(self) -> None:
        """创建任务执行记录表。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS task_log ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "task_name TEXT NOT NULL, "
                    "run_date TEXT NOT NULL, "
                    "status TEXT NOT NULL, "  # running/success/failed
                    "started_at TEXT, "
                    "finished_at TEXT, "
                    "elapsed_sec REAL, "
                    "result_summary TEXT, "
                    "error_msg TEXT)"
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"task_log 表初始化失败：{e!r}")

    def _load_last_run(self) -> None:
        """从 DB 恢复今日已执行的任务（重启不重复执行）。"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT task_name FROM task_log "
                    "WHERE run_date=? AND status='success'", (today,)
                ).fetchall()
            for r in rows:
                self._last_run[f"{r[0]}_{today}"] = "recovered"
            if rows:
                logger.info(f"调度器恢复：今日已完成 {len(rows)} 个任务，跳过重跑")
        except Exception as e:
            logger.warning(f"调度器恢复失败：{e!r}")

    def _log_task(self, task: str, status: str, started_at: str,
                  finished_at: str, elapsed: float, summary: str = "", error: str = "") -> None:
        """记录任务执行结果到 DB。"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO task_log (task_name, run_date, status, started_at, finished_at, "
                    "elapsed_sec, result_summary, error_msg) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (task, today, status, started_at, finished_at, elapsed, summary[:500], error[:500]),
                )
                conn.commit()
        except Exception:
            pass

    def _wal_checkpoint(self) -> None:
        """强制 WAL checkpoint：将WAL日志写入主DB文件并截断WAL。

        解决两个问题：
        1. WAL文件持续膨胀（不checkpoint会越来越大）
        2. SQLite WAL读缓存不一致（写入后读取可能拿到旧值）
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA busy_timeout=5000")
                result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                # result = (busy, log_frames, checkpointed_frames)
            self._last_wal_checkpoint = time.time()
            if result and result[1] > 0:
                logger.info(f"WAL checkpoint: {result[1]}帧已刷新")
        except Exception as e:
            logger.debug(f"WAL checkpoint: {e!r}")

    def _cleanup_zombies(self) -> None:
        """启动时清理残留的 multiprocessing 子进程。"""
        try:
            import subprocess
            # 找到属于本项目的 multiprocessing spawn/resource_tracker 进程
            result = subprocess.run(
                ["pgrep", "-f", "sequoia-x.*multiprocessing"],
                capture_output=True, text=True, timeout=5,
            )
            pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
            # 排除自己
            my_pid = str(os.getpid())
            killed = 0
            for pid in pids:
                if pid and pid != my_pid:
                    try:
                        os.kill(int(pid), 9)
                        killed += 1
                    except (ProcessLookupError, ValueError):
                        pass
            if killed:
                logger.warning(f"僵尸进程清理：kill {killed} 个残留 multiprocessing 进程")
        except Exception:
            pass  # pgrep 不存在时静默跳过

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="auction-scheduler")
        self._thread.start()
        logger.info("竞价调度器已启动（09:25竞价扫描 / 18:00数据同步）")

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        from datetime import timedelta as _td
        while not self._stop.is_set():
            today = datetime.now().strftime("%Y-%m-%d")

            for hour, minute, task in self.SCHEDULE:
                key = f"{task}_{today}"
                if self._last_run.get(key):
                    continue
                # 每个任务独立刷新now（前序长任务可能阻塞数十分钟）
                now = datetime.now()
                sched_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                in_window = now.hour == hour and abs(now.minute - minute) <= 10
                # 补跑：任务时间已过>10min但今日未执行（被前序长任务阻塞）
                # 排除时效性任务（竞价扫描/盘中轮询不应补跑）
                is_overdue = (
                    now > sched_dt + _td(minutes=10)
                    and task not in ("auction_scan", "intraday_scan_start")
                    and now < sched_dt + _td(hours=2)
                )
                if (in_window or is_overdue) and self._is_trading_day(today):
                        logger.info(f"触发定时任务：{task}")
                        t_start = time.time()
                        started_at = now.strftime("%H:%M:%S")
                        try:
                            summary = self._run_task_with_result(task)
                            elapsed = time.time() - t_start
                            self._last_run[key] = now.strftime("%H:%M")
                            self._log_task(task, "success", started_at,
                                          now.strftime("%H:%M:%S"), elapsed, str(summary)[:200])
                            logger.info(f"定时任务 {task} 完成 ({elapsed:.0f}s)")
                        except Exception as e:
                            elapsed = time.time() - t_start
                            self._log_task(task, "failed", started_at,
                                          now.strftime("%H:%M:%S"), elapsed, error=str(e))
                            # 标记已尝试，避免在时间窗口内每分钟重复失败
                            self._last_run[key] = now.strftime("%H:%M")
                            logger.warning(f"定时任务 {task} 失败 ({elapsed:.0f}s)：{e!r}")

            # 定时 WAL checkpoint（每30分钟，防止WAL膨胀+缓存不一致）
            if time.time() - self._last_wal_checkpoint > 1800:
                self._wal_checkpoint()

            # 每分钟检查一次
            self._stop.wait(60)

    def _is_trading_day(self, date_str: str) -> bool:
        """简化判定：周末跳过（节假日需手动补，或接交易日历）。"""
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return d.weekday() < 5  # 0-4 = 周一至周五

    def _run_task(self, task: str) -> None:
        if task == "auction_scan":
            self._auction_scan()
        elif task == "sync_daily":
            self._sync_daily()
        elif task == "auction_verify":
            self._auction_verify()
        elif task == "intraday_scan_start":
            self._intraday_loop()
        elif task == "paper_trade":
            self._paper_trade()
        elif task == "sync_lhb":
            self._sync_lhb()
        elif task == "sync_fund_flow":
            self._sync_fund_flow()
        elif task == "sync_margin":
            self._sync_margin()
        elif task == "sync_valuation":
            self._sync_valuation()
        elif task == "sync_altdata":
            self._sync_altdata()
        elif task == "refresh_factor_ic":
            self._refresh_factor_ic()
        elif task == "daily_report":
            self._daily_report()
        elif task == "monthly_sweep":
            self._monthly_sweep()
        elif task == "backfill_offline":
            self._backfill_offline()

    def _run_task_with_result(self, task: str) -> str:
        """执行任务并返回结果摘要（供日志记录）。"""
        self._run_task(task)
        return f"{task} executed"

    def _auction_scan(self) -> None:
        """竞价扫描 + 飞书推送。"""
        from sequoia_x.analysis.auction import AuctionScanner
        from sequoia_x.notify.feishu import FeishuNotifier

        scanner = AuctionScanner(self.db_path)
        notifier = FeishuNotifier(self.settings)
        result = scanner.scan(top_n=50, push=True, notifier=notifier)
        if result.get("error"):
            logger.warning(f"竞价扫描失败：{result['error']}")
        else:
            logger.info(f"竞价扫描完成并推送：A{result.get('grade_a',0)} B{result.get('grade_b',0)}")

    def _sync_daily(self) -> None:
        """日K数据自动同步：东财批量优先(3秒)，baostock兜底。"""
        from sequoia_x.data.engine import DataEngine
        try:
            engine = DataEngine(self.settings)
            n = engine.sync_today_eastmoney()
            if n < 100:
                logger.warning(f"东财增量仅{n}只，切换baostock兜底...")
                n = engine.sync_today_bulk()
            logger.info(f"定时数据同步完成：写入 {n} 只")
        except Exception as e:
            logger.warning(f"定时数据同步失败：{e!r}")

    def _backfill_offline(self) -> None:
        """凌晨离线回补：baostock 额度恢复后自动跑（00:15 触发）。

        优先级1：现金流比率回补（P15，每行1次调用，最高性价比）
        优先级2：缺失财报股票补全（剩余额度）
        每个工作日凌晨跑一批，连续几天补完。
        补完后自动跳过（无 NULL 行 / 无缺失股），零开销。
        """
        try:
            from sequoia_x.data.finance_sync import backfill_cash_flow
            result = backfill_cash_flow(settings=self.settings)
            logger.info(f"现金流回补完成：{result}")
            if result.get("rate_limited"):
                return  # 额度耗尽，不跑步骤2（省额度给高优先级）
        except Exception as e:
            logger.warning(f"现金流回补失败：{e!r}")
            return

        try:
            from sequoia_x.data.finance_sync import FinanceSync
            syncer = FinanceSync(self.settings)
            result = syncer.sync_all(n_quarters=20, max_stocks=300)
            logger.info(f"财报缺失股补全完成：{result}")
        except Exception as e:
            logger.warning(f"财报缺失股补全失败：{e!r}")

    def _sync_lhb(self) -> None:
        """龙虎榜数据自动同步（日K同步后执行）。"""
        from sequoia_x.data.engine import DataEngine
        try:
            engine = DataEngine(self.settings)
            n = engine.sync_lhb()
            logger.info(f"龙虎榜同步完成：{n} 只个股明细")
            # 席位明细较慢（逐股请求），后台执行不阻塞
            n2 = engine.sync_lhb_seats()
            logger.info(f"龙虎榜席位同步完成：{n2} 行")
        except Exception as e:
            logger.warning(f"龙虎榜同步失败：{e!r}")

    def _monthly_sweep(self) -> None:
        """闭环3：每月1号自动跑参数扫描，验证当前参数是否仍最优。

        非每月1号自动跳过。扫描结果写入DB日志，如最优参数变化则飞书通知。
        """
        today = datetime.now()
        if today.day != 1:
            return  # 非每月1号跳过

        logger.info("月度参数扫描：开始验证当前模拟盘参数是否仍最优...")
        try:
            from sequoia_x.analysis.paper_replay import PaperReplayEngine
            engine = PaperReplayEngine(self.db_path)
            result = engine.sweep(sample_size=300, progress_callback=None)

            best = result.get("best", {})
            baseline = result.get("baseline", {})

            # 如果最优参数与当前基线差异大，飞书通知
            sharpe_diff = best.get("sharpe", 0) - baseline.get("sharpe", 0)
            if sharpe_diff > 0.3:
                from sequoia_x.notify.feishu import FeishuNotifier
                notifier = FeishuNotifier(self.settings)
                notifier.send(
                    f"📊 月度参数扫描完成\n"
                    f"当前参数夏普: {baseline.get('sharpe', '?')}\n"
                    f"最优参数夏普: {best.get('sharpe', '?')}\n"
                    f"提升: +{sharpe_diff:.2f}\n"
                    f"建议参数: 止损{best.get('stop_loss')}% 止盈{best.get('take_profit', '∞')}% "
                    f"调仓{best.get('rebalance_interval')}天\n"
                    f"年化: {baseline.get('annual_return', '?')}% → {best.get('annual_return', '?')}%"
                )
                logger.info(f"月度扫描：发现更优参数，已飞书通知（夏普+{sharpe_diff:.2f}）")
            else:
                logger.info(f"月度扫描：当前参数仍接近最优（夏普差{sharpe_diff:.2f}），无需调整")
        except Exception as e:
            logger.warning(f"月度参数扫描失败：{e!r}")

        # 策略质量重评 + 样本外衰减刷新（walk-forward）→ 写 strategy_weights.oos_decay
        # 避免质量分长期停留在手动跑的旧值，使决策加成反映最新样本外延续性
        try:
            from sequoia_x.analysis.strategy_eval import StrategyEvaluator
            from sequoia_x.data.engine import DataEngine
            eval_engine = DataEngine(self.settings)
            evaluator = StrategyEvaluator(eval_engine, self.settings)
            evaluator.evaluate(hold_days=20, sample_size=300)
            logger.info("月度策略质量重评完成：oos_decay 已刷新写入 strategy_weights")
        except Exception as e:
            logger.warning(f"月度策略质量重评失败（不影响参数扫描结果）：{e!r}")

        # ML 因子合成月度训练（LightGBM/Ridge）→ 写 ml_scores 快照 + factor_weights.ml_score
        # 与因子IC刷新、oos_decay 同批次，不进每日闭环
        try:
            from sequoia_x.analysis.ml_factor import MLFactorEngine
            ml_engine = MLFactorEngine(self.db_path)
            ml_result = ml_engine.compute_ml_score()
            logger.info(
                f"月度ML训练完成：IC={ml_result.get('ic_mean', 0)} "
                f"ICIR={ml_result.get('icir', 0)} 有效={ml_result.get('valid', False)} "
                f"模型={ml_result.get('model_version', '?')}"
            )
        except Exception as e:
            logger.warning(f"月度ML训练失败（不影响其他任务）：{e!r}")

        # 因子归因分析（采样500只，63个月截面，约15秒）→ 回答"赚的钱来自哪个因子"
        # 只展示不自动改权重（小样本归因可能误判强因子），飞书通知供人工审阅
        try:
            from sequoia_x.analysis.attribution import AttributionAnalyzer
            from sequoia_x.data.engine import DataEngine
            attr_engine = DataEngine(self.settings)
            analyzer = AttributionAnalyzer(attr_engine, self.settings)
            attr_result = analyzer.analyze(hold_days=20, sample_size=500)
            self._last_attribution = attr_result

            # 飞书通知归因摘要
            drivers = attr_result.get("top_drivers", [])
            drags = attr_result.get("top_drags", [])
            cat_summary = attr_result.get("category_summary", [])[:3]
            strat_ret = attr_result.get("strategy_annual_return", 0)
            bench_ret = attr_result.get("benchmark_annual_return", 0)
            alpha = attr_result.get("alpha", 0)

            driver_str = " ".join(drivers[:3]) if drivers else "无"
            drag_str = " ".join(drags[:3]) if drags else "无"
            cat_str = " ".join(
                f"{c['category']}({c['total_contribution']:+.2f})"
                for c in cat_summary
            ) if cat_summary else "无"

            from sequoia_x.notify.feishu import FeishuNotifier
            notifier = FeishuNotifier(self.settings)
            notifier.send(
                "📊 月度因子归因\n"
                f"策略年化 {strat_ret}% vs 基准 {bench_ret}% (Alpha {alpha}%)\n"
                f"🟢 收益驱动: {driver_str}\n"
                f"🔴 收益拖累: {drag_str}\n"
                f"📊 大类贡献: {cat_str}"
            )
            logger.info(f"月度归因完成：Alpha={alpha}%, 驱动={drivers[:3]}")
        except Exception as e:
            logger.warning(f"月度归因失败（不影响其他任务）：{e!r}")

        # 策略增量贡献计算（月度）→ 写 strategy_weights.marginal_alpha + 飞书通知
        try:
            from sequoia_x.analysis.combo_backtest import ComboBacktester
            from sequoia_x.data.engine import DataEngine
            cb_engine = DataEngine(self.settings)
            cb = ComboBacktester(cb_engine, self.settings)
            marginal = cb.compute_strategy_marginal(hold_days=20, sample_size=500)
            if marginal:
                # 写入 DB（在现有 strategy_weights 行上更新 marginal_alpha 列）
                import sqlite3 as _sql3
                import time as _time_mod
                now_str = _time_mod.strftime("%Y-%m-%d %H:%M:%S")
                with _sql3.connect(self.db_path) as conn:
                    for skey, info in marginal.items():
                        conn.execute(
                            "UPDATE strategy_weights SET marginal_alpha=?, updated_at=? "
                            "WHERE strategy_key=?",
                            (info["marginal_alpha_pp"], now_str, skey),
                        )
                        if conn.total_changes == 0:
                            conn.execute(
                                "INSERT OR IGNORE INTO strategy_weights "
                                "(strategy_key, quality_score, marginal_alpha, updated_at) "
                                "VALUES (?, 0, ?, ?)",
                                (skey, info["marginal_alpha_pp"], now_str),
                            )
                    conn.commit()

                # 飞书通知（按 marginal_alpha 降序前5）
                _labels = {
                    "multi_factor": "多因子", "bottom": "底部放量", "flag": "高位旗形",
                    "ma_volume": "均线放量", "pullback": "缩量回踩", "volume_extreme": "地量见底",
                    "turtle": "海龟突破", "shakeout": "涨停洗盘", "limit_down": "上升趋势跌停",
                    "dragon": "板块龙头", "rps": "RPS强势", "lhb_follow": "龙虎榜跟买",
                    "sector_rotation": "板块轮动",
                }
                top5 = sorted(marginal.items(), key=lambda x: -x[1]["marginal_alpha_pp"])[:5]
                parts = [
                    f"{_labels.get(k, k)}({v['marginal_alpha_pp']:+.1f}pp)"
                    for k, v in top5
                ]
                from sequoia_x.notify.feishu import FeishuNotifier as _FN2
                _FN2(self.settings).send(
                    "📈 策略增量贡献\n" + " ".join(parts)
                )
                logger.info(f"月度策略增量贡献完成：{len(marginal)}个策略")
        except Exception as e:
            logger.warning(f"月度策略增量贡献失败（不影响其他任务）：{e!r}")

    def _daily_report(self) -> str:
        """每日任务执行汇总：统计今天所有定时任务的成功/失败/耗时，发飞书。"""
        from sequoia_x.notify.feishu import FeishuNotifier
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT task_name, status, elapsed_sec, started_at, finished_at, error_msg "
                    "FROM task_log WHERE run_date=? ORDER BY id", (today,)
                ).fetchall()

            # 任务中文名映射
            name_map = {
                "auction_scan": "竞价扫描",
                "intraday_scan_start": "盘中持仓监控",
                "backfill_offline": "离线数据回补",
                "sync_daily": "日K同步",
                "sync_lhb": "龙虎榜同步",
                "sync_fund_flow": "资金流向同步",
                "sync_margin": "融资融券同步",
                "sync_altdata": "另类数据同步",
                "refresh_factor_ic": "因子IC刷新",
                "auction_verify": "竞价T+1验证",
                "paper_trade": "模拟盘闭环",
            }

            success = sum(1 for r in rows if r[1] == "success")
            failed = sum(1 for r in rows if r[1] == "failed")
            total = len(rows)

            lines = [f"**📊 每日任务执行汇总（{today}）**\n"]
            lines.append(f"总计 {total} 个任务：✅成功 {success}  ❌失败 {failed}\n")

            for r in rows:
                name = name_map.get(r[0], r[0])
                status_icon = "✅" if r[1] == "success" else "❌"
                elapsed = f"{r[2]:.0f}s" if r[2] else "-"
                time_range = ""
                if r[3] and r[4]:
                    time_range = f" {r[3]}~{r[4]}"
                line = f"{status_icon} {name} ({elapsed}){time_range}"
                if r[5]:
                    line += f"\n   ⚠️ {r[5][:80]}"
                lines.append(line)

            # ── 数据同步全景状态 ──
            lines.append("")
            lines.append("---")
            lines.append("**📦 数据同步状态**")
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # 日K
                    latest_k = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0] or "-"
                    k_rows = conn.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0]
                    lines.append(f"日K: {latest_k} | {k_rows:,}行")

                    # 财报
                    fin_full = conn.execute(
                        "SELECT COUNT(*) FROM (SELECT symbol FROM stock_finance "
                        "GROUP BY symbol HAVING COUNT(DISTINCT stat_date)>=18)").fetchone()[0]
                    fin_sym = conn.execute("SELECT COUNT(DISTINCT symbol) FROM stock_finance").fetchone()[0]
                    total_sym = conn.execute("SELECT COUNT(*) FROM stock_basic").fetchone()[0] or 5000
                    lines.append(f"财报: ≥5年{fin_full}/{total_sym} ({fin_full/total_sym*100:.0f}%) | {fin_sym}只")

                    # 龙虎榜
                    lhb_days = conn.execute("SELECT COUNT(DISTINCT date) FROM lhb_detail").fetchone()[0]
                    lhb_max = conn.execute("SELECT MAX(date) FROM lhb_detail").fetchone()[0] or "-"
                    lines.append(f"龙虎榜: {lhb_max} | {lhb_days}交易日")

                    # 融资融券
                    mg_days = conn.execute("SELECT COUNT(DISTINCT date) FROM margin_detail").fetchone()[0]
                    mg_max = conn.execute("SELECT MAX(date) FROM margin_detail").fetchone()[0] or "-"
                    mg_sym = conn.execute("SELECT COUNT(DISTINCT symbol) FROM margin_detail").fetchone()[0]
                    lines.append(f"融资融券: {mg_max} | {mg_days}天/{mg_sym}只")

                    # 资金流
                    ff_days = conn.execute("SELECT COUNT(DISTINCT date) FROM fund_flow").fetchone()[0]
                    ff_max = conn.execute("SELECT MAX(date) FROM fund_flow").fetchone()[0] or "-"
                    ff_sym = conn.execute("SELECT COUNT(DISTINCT symbol) FROM fund_flow").fetchone()[0]
                    lines.append(f"资金流: {ff_max} | {ff_days}天/{ff_sym}只")

                    # 北向（标注断供）
                    nb_max = conn.execute("SELECT MAX(date) FROM north_hold").fetchone()[0] or "-"
                    lines.append(f"北向: {nb_max}（2024-08后断供）")

                    # 沪深300指数
                    idx_max = conn.execute("SELECT MAX(date) FROM index_daily").fetchone()[0] or "-"
                    lines.append(f"沪深300: {idx_max}")

                    # 基金持仓
                    fh_qtr = conn.execute("SELECT COUNT(DISTINCT report_date) FROM fund_hold").fetchone()[0]
                    fh_max = conn.execute("SELECT MAX(report_date) FROM fund_hold").fetchone()[0] or "-"
                    lines.append(f"基金持仓: {fh_max} | {fh_qtr}季度")

                    # 宏观
                    m2_max = conn.execute("SELECT MAX(month) FROM macro_money").fetchone()[0] or "-"
                    sf_max = conn.execute("SELECT MAX(month) FROM macro_sf").fetchone()[0] or "-"
                    lines.append(f"M2: {m2_max} | 社融: {sf_max}")
            except Exception as e:
                lines.append(f"数据状态查询失败: {e}")

            # 限流器状态
            try:
                from sequoia_x.core.rate_limiter import _rate_limiter
                rl = _rate_limiter.baostock_status()
                em_cb = _rate_limiter._circuit_until.get("eastmoney", 0)
                em_remain = max(0, int(em_cb - time.time()))
                em_status = f"熔断(剩{em_remain//60}分)" if em_remain > 0 else "正常"
                lines.append(f"---")
                lines.append(f"baostock额度: {rl['used']}/{rl['limit']} ({rl['usage_pct']:.0f}%)")
                lines.append(f"东财: {em_status}")
            except Exception:
                pass

            # 因子归因（月度任务，当日执行过则展示）
            if self._last_attribution:
                attr = self._last_attribution
                drivers = attr.get("top_drivers", [])
                alpha = attr.get("alpha", 0)
                driver_str = "/".join(drivers[:3]) if drivers else "无"
                lines.append(f"---")
                lines.append(f"因子归因: Alpha {alpha}% | 驱动: {driver_str}")
            else:
                lines.append(f"因子归因: 月度任务（每月1号执行）")

            content = "\n".join(lines)
            notifier = FeishuNotifier(self.settings)
            notifier.send_text("每日任务汇总", content)
            logger.info(f"每日汇总已推送飞书：{success}成功/{failed}失败/{total}总计")
            return f"{success}成功/{failed}失败"
        except Exception as e:
            logger.warning(f"每日汇总失败：{e!r}")
            return f"失败: {e}"

    def _sync_altdata(self) -> None:
        """另类数据增量同步：沪深300指数(日) + 基金持仓(季) + M2/社融(月)。

        各源全量拉取幂等UPSERT，增量数据自动覆盖。约3-5秒。
        """
        from sequoia_x.data.altdata_sync import (
            sync_index_daily, sync_fund_hold,
            sync_macro_money_supply, sync_macro_social_finance,
        )
        try:
            # 沪深300（日频，全量拉取约1秒）
            n = sync_index_daily(self.db_path, "sh000300")
            logger.info(f"沪深300指数同步: {n}行")
        except Exception as e:
            logger.warning(f"沪深300同步失败: {e!r}")
        try:
            # M2/社融（月频，全量拉取<1秒）
            sync_macro_money_supply(self.db_path)
            sync_macro_social_finance(self.db_path)
        except Exception as e:
            logger.warning(f"宏观数据同步失败: {e!r}")
        try:
            # 基金持仓（季频，取最近2季度约10秒）
            from sequoia_x.data.altdata_sync import _recent_report_dates
            for d in _recent_report_dates(2):
                sync_fund_hold(self.db_path, d)
        except Exception as e:
            logger.warning(f"基金持仓同步失败: {e!r}")

    def _refresh_factor_ic(self) -> None:
        """因子IC权重自动刷新（滚动6个月窗口 + 三态权重）。

        每天盘后执行：
          1. 用最近6个月数据重算因子IC
          2. 写入带符号权重到DB
          3. 更新三态市场状态权重
          4. 飞书通知刷新结果
        """
        try:
            from sequoia_x.data.engine import DataEngine
            from sequoia_x.analysis.factor import evaluate_factor_ic
            engine = DataEngine(self.settings)
            result = evaluate_factor_ic(
                engine, hold_days=20, sample_size=500, rolling_months=12
            )
            factors = result.get("factors", [])
            effective = [f for f in factors if abs(f.get("ic_mean", 0)) > 0.015
                         and abs(f.get("icir", 0)) > 0.3]
            logger.info(f"因子IC刷新完成：{len(effective)}/{len(factors)}个有效因子（滚动6个月）")
        except Exception as e:
            logger.warning(f"因子IC刷新失败：{e!r}")

    def _sync_margin(self) -> None:
        """融资融券增量同步（盘后执行，akshare 沪+深，不受东财push2his熔断影响）。

        同步当日全市场融资融券明细到 margin_detail 表（幂等UPSERT）。
        """
        try:
            from sequoia_x.data.margin_sync import sync_margin_detail
            today = datetime.now().strftime("%Y%m%d")
            n = sync_margin_detail(self.db_path, today)
            logger.info(f"融资融券同步完成：{n} 行")
        except Exception as e:
            logger.warning(f"融资融券同步失败：{e!r}")

    def _sync_valuation(self) -> None:
        """PE/PB估值同步（东财快照，全市场）。"""
        from sequoia_x.data.engine import DataEngine
        try:
            engine = DataEngine(self.settings)
            n = engine.sync_valuation()
            logger.info(f"估值同步完成：{n} 只更新PE/PB")
        except Exception as e:
            logger.warning(f"估值同步失败：{e!r}")

    def _sync_fund_flow(self) -> None:
        """主力资金流向同步（盘后执行）+ 历史回填。"""
        from sequoia_x.data.engine import DataEngine
        try:
            engine = DataEngine(self.settings)
            n = engine.sync_fund_flow()
            logger.info(f"资金流向同步完成：{n} 行")
        except Exception as e:
            logger.warning(f"资金流向同步失败：{e!r}")

        # 历史回填（增量回补~1年，供基本面/资金类因子 IC 评估）
        try:
            import sqlite3
            from sequoia_x.data.fund_flow_history import backfill_fund_flow_history
            with sqlite3.connect(self.db_path) as conn:
                symbols = [r[0] for r in conn.execute(
                    "SELECT DISTINCT symbol FROM ("
                    "SELECT symbol FROM decision_pool UNION "
                    "SELECT symbol FROM lhb_detail UNION "
                    "SELECT symbol FROM paper_holdings UNION "
                    "SELECT symbol FROM stock_market_cap "
                    "WHERE circ_mv IS NOT NULL ORDER BY circ_mv DESC LIMIT 500) "
                    "ORDER BY symbol LIMIT 500"
                ).fetchall()]
            if symbols:
                # 串行慢速回补（n_workers=1），避免东财神 IP 风控触发熔断
                result = backfill_fund_flow_history(
                    self.db_path, symbols, days=250, n_workers=1
                )
                logger.info(f"资金流向历史回填：{result}")
        except Exception as e:
            logger.warning(f"资金流向历史回填失败：{e!r}")

        # 北向资金已断供（2024-08沪深港通新规取消个股明细公开），跳过同步
        logger.debug("北向持股跳过同步（2024-08后数据源永久断供）")

    def _auction_verify(self) -> None:
        """竞价T+1命中验证（数据同步后执行）。"""
        from sequoia_x.analysis.auction import AuctionScanner
        try:
            scanner = AuctionScanner(self.db_path)
            result = scanner.verify_t1()
            verified = result.get("verified", 0)
            summary = result.get("summary", {})
            if verified:
                logger.info(f"竞价T+1验证完成：{verified}条，命中率 {summary}")
            else:
                logger.info(f"竞价验证跳过：{result.get('msg', '无新数据')}")
        except Exception as e:
            logger.warning(f"竞价T+1验证失败：{e!r}")

    def _paper_trade(self) -> None:
        """模拟盘盘后自动执行：数据同步后跑决策→自动买卖→飞书推送绩效。

        流程：持仓扫描卖出 → 全策略决策 → 自动买入 → 绩效快照 → 飞书推送。
        """
        from sequoia_x.core.config import Settings
        from sequoia_x.web.services import WebServices
        from sequoia_x.notify.feishu import FeishuNotifier

        try:
            settings = self.settings
            engine_mod = __import__("sequoia_x.data.engine", fromlist=["DataEngine"])
            data_engine = engine_mod.DataEngine(settings)
            services = WebServices(settings, data_engine)
            notifier = FeishuNotifier(settings)

            # Step1: 持仓扫描 → 自动卖出（启用止损写回：ATR收紧次日生效）
            pos_result = services.scan_positions(apply_stop_move=True)

            # Step1.5: 止损收敛兜底 — 过宽止损(>12%)按 ATR(2.5×, 封顶8-15%)重算写回
            try:
                from sequoia_x.analysis.stop_loss import calc_atr_stop
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    wide_rows = conn.execute(
                        "SELECT id, symbol, entry_price, stop_loss FROM paper_holdings "
                        "WHERE shares > 0 AND entry_price > 0 AND stop_loss > 0 "
                        "AND (entry_price - stop_loss) / entry_price > 0.12"
                    ).fetchall()
                converged = 0
                for row in wide_rows:
                    df = data_engine.get_ohlcv(row["symbol"])
                    atr_stop = calc_atr_stop(df, row["entry_price"])
                    if atr_stop > row["stop_loss"]:
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute(
                                "UPDATE paper_holdings SET stop_loss=? WHERE id=?",
                                (round(atr_stop, 2), row["id"]),
                            )
                            conn.commit()
                        converged += 1
                if converged:
                    logger.info(f"止损收敛兜底：{converged} 只过宽持仓按 ATR 收紧至 ≤15%")
            except Exception as e:
                logger.warning(f"止损收敛兜底失败：{e!r}")

            sell_result = services.paper_auto_sell(pos_result.get("signals", []))

            # Step2: 多因子为核心的决策（废弃策略已自动排除）
            from sequoia_x.strategy.registry import ACTIVE_STRATEGY_KEYS
            decision = services.generate_decision(
                strategy_keys=ACTIVE_STRATEGY_KEYS, capital=100000, min_score=50,
                exclude_markets=None, exclude_st=True,
                max_candidates=60, include_auction=False,
            )

            # Step3: 自动买入
            buy_result = services.paper_auto_buy(decision)

            # 买入后同步：paper_holdings → portfolio_holding
            services.sync_paper_to_portfolio()

            # Step4: 记录日度净值
            try:
                services.paper_record_nav()
            except Exception:
                pass

            # Step5: 绩效
            perf = services.paper_performance()

            # Step6: 飞书推送
            sharpe_str = f" 夏普{perf.sharpe_ratio}" if perf.sharpe_ratio else ""
            alpha_str = f" 超额{perf.alpha:+.1f}%" if perf.alpha else ""
            summary = (
                f"📊 模拟盘日报\n"
                f"总资产 ¥{perf.total_assets:,.0f} (收益{perf.total_return_pct:+.2f}% 年化{perf.annual_return:+.1f}%)\n"
                f"现金 ¥{perf.cash:,.0f} | 持仓 ¥{perf.market_value:,.0f} ({perf.holding_count}只)\n"
                f"今日：卖出{len(sell_result.get('sold',[]))}笔 买入{len(buy_result.get('bought',[]))}只\n"
                f"累计：{perf.total_trades}笔 胜率{perf.win_rate}% 盈亏比{perf.profit_factor if perf.profit_factor<999 else '∞'}"
                f" 回撤{perf.max_drawdown_pct:.1f}%\n"
                f"风控状态：{perf.risk_circuit}"
                f"{sharpe_str}{alpha_str}"
            )
            fh = decision.get("factor_health")
            if fh and fh.get("is_degraded"):
                _wl_parts = []
                for src_name, info in fh.get("sources", {}).items():
                    if info.get("coverage", 1.0) < 0.20:
                        _labels = {
                            "finance": "财报", "fund_flow": "资金流", "lhb": "龙虎榜",
                            "north": "北向", "margin": "融资融券", "fund_hold": "基金持仓",
                            "block": "大宗交易", "holder": "股东户数", "index": "沪深300",
                        }
                        _wl_parts.append(
                            f"{_labels.get(src_name, src_name)}{info['coverage']*100:.0f}%"
                        )
                if _wl_parts:
                    summary += f"\n⚠️ 因子健康度：{' '.join(_wl_parts)} [{len(_wl_parts)}源降级]"
            try:
                notifier.send_text("模拟盘日报", summary)
            except Exception as e:
                logger.warning(f"飞书日报推送失败：{e!r}")

            # 记录每日决策快照到 paper_decision_log
            try:
                import sqlite3 as _sql
                from datetime import datetime as _dt
                today_str = _dt.now().strftime("%Y-%m-%d")
                now_str = _dt.now().isoformat()
                ms = decision.get("market_state", {})
                bought_syms = ",".join([b.get("symbol", "") for b in (buy_result.get("bought") or [])])
                with _sql.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO paper_decision_log "
                        "(run_date, market_state, market_score, pool_size, buy_count, bought_stocks, "
                        "sell_count, total_assets, total_return_pct, reason, created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            today_str,
                            ms.get("state", ""),
                            ms.get("score", 0),
                            decision.get("pool_size", 0),
                            len(buy_result.get("bought") or []),
                            bought_syms,
                            len(sell_result.get("sold") or []),
                            perf.total_assets,
                            perf.total_return_pct,
                            buy_result.get("reason", ""),
                            now_str,
                        ),
                    )
                    conn.commit()
            except Exception:
                pass

            logger.info(f"模拟盘执行完成：买{len(buy_result.get('bought',[]))} 卖{len(sell_result.get('sold',[]))} 收益{perf.total_return_pct}%")
        except Exception as e:
            logger.warning(f"模拟盘自动执行失败：{e!r}")

    def _intraday_loop(self) -> None:
        """盘中持仓监控：9:30-15:00 每60秒扫描模拟盘持仓，触发止损/止盈/减仓。

        已移除关注池信号扫描（IntradayScanner），仅保留持仓盯盘+自动卖出。
        原因：关注池扫描消耗大量东财请求且信号质量低，盘中核心需求是持仓风控。
        """
        from sequoia_x.notify.feishu import FeishuNotifier
        try:
            notifier = FeishuNotifier(self.settings)
            count = 0
            while not self._stop.is_set():
                now = datetime.now()
                # 仅交易时段 9:30-11:30 / 13:00-15:00
                hour_min = now.hour * 100 + now.minute
                in_session = (930 <= hour_min <= 1130) or (1300 <= hour_min <= 1500)
                if not in_session:
                    if hour_min > 1500:
                        break  # 收盘退出
                    self._stop.wait(60)
                    continue

                # 持仓盘中盯盘：止损/止盈/减仓 → 自动卖出 + 飞书推送
                try:
                    self._position_monitor(notifier)
                    count += 1
                except Exception as e:
                    logger.warning(f"持仓监控异常：{e!r}")

                self._stop.wait(60)  # 每60秒一轮
            logger.info(f"盘中持仓监控结束，共执行 {count} 轮")
        except Exception as e:
            logger.warning(f"盘中持仓监控启动失败：{e!r}")

    def _position_monitor(self, notifier=None) -> None:
        """持仓盘中实时盯盘：批量快照 + 实时MA + 信号推送 + 自动卖出。

        danger 信号（止损清仓/止盈/减仓）→ 自动执行模拟盘卖出 + 飞书推送。
        """
        from sequoia_x.analysis.position import PositionTracker
        try:
            from sequoia_x.data.engine import DataEngine
            from sequoia_x.analysis.paper_trade import PaperTradeEngine
            engine = DataEngine(self.settings)
            tracker = PositionTracker(engine, self.settings)
            signals = tracker.scan_intraday(notifier=notifier)
            # danger=止损/止盈（清仓），warn=弱信号（减仓）
            danger = [s for s in signals if s.signal_level in ("danger", "warn")
                       and s.action not in ("持有", "数据缺失", "移动止损")]
            if not danger:
                return

            logger.warning(f"持仓盘中预警：{len(danger)}只触发止损/清仓信号")

            # ── 盘中实时卖出：danger 信号 → 模拟盘执行 ──
            sell_signals = []
            for sig in danger:
                action = sig.action
                if action in ("止损清仓", "止盈", "减仓/清仓", "减仓半仓"):
                    sell_signals.append({
                        "symbol": sig.symbol,
                        "action": action,
                        "shares": sig.shares,
                        "new_stop": sig.new_stop,
                        "reasons": sig.reasons,
                        "realtime_price": getattr(sig, "price", 0),
                    })

            if sell_signals:
                pe = PaperTradeEngine(self.settings)
                result = pe.auto_sell_intraday(sell_signals)
                if result.get("sold"):
                    sold_list = result["sold"]
                    logger.info(f"盘中实时卖出执行：{len(sold_list)}笔")
                    # 同步 paper_holdings → portfolio_holding（保持两表一致）
                    try:
                        from sequoia_x.data.engine import DataEngine as _DE
                        from sequoia_x.web.services import WebServices as _WS
                        _svc = _WS(self.settings, _DE(self.settings))
                        _svc.sync_paper_to_portfolio()
                    except Exception:
                        pass
                    # 飞书推送卖出结果
                    if notifier:
                        for s in sold_list:
                            notifier.send(
                                f"🔴 盘中实时卖出 `{s['symbol']}`\n"
                                f"▶ 卖出价 {s['price']}\n"
                                f"▶ {s['shares']}股 金额{s['amount']:.0f}元\n"
                                f"▶ 盈亏{s.get('pnl',0):+.0f}({s.get('pnl_pct',0):+.1f}%)\n"
                                f"▶ 原因：{', '.join(s.get('reasons', [])[:2])}"
                            )
        except Exception as e:
            logger.warning(f"持仓监控失败：{e!r}")
