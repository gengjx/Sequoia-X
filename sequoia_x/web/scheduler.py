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
        (9, 25, "auction_scan"),
        (9, 30, "intraday_scan_start"),  # 启动盘中轮询
        (21, 0, "sync_daily"),       # 避开baostock盘后高峰(18-21点拥堵)
        (21, 5, "sync_lhb"),         # 龙虎榜数据同步（紧跟日K之后）
        (21, 6, "sync_fund_flow"),   # 主力资金流向同步
        (21, 7, "sync_valuation"),  # PE/PB估值同步（东财快照，全市场3秒）
        (21, 10, "refresh_factor_ic"),  # 因子IC权重刷新（滚动6个月窗口）
        (21, 30, "auction_verify"),  # 同步完成后验证T+1命中
        (21, 40, "paper_trade"),  # 模拟盘：盘后选股→买入→卖出闭环
        (22, 0, "daily_report"),  # 每日任务执行汇总 → 飞书推送
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
        while not self._stop.is_set():
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")

            for hour, minute, task in self.SCHEDULE:
                key = f"{task}_{today}"
                if self._last_run.get(key):
                    continue
                # 命中调度时间窗口（±10分钟容忍，避免错过）
                if now.hour == hour and abs(now.minute - minute) <= 10:
                    if self._is_trading_day(today):
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
        elif task == "sync_valuation":
            self._sync_valuation()
        elif task == "refresh_factor_ic":
            self._refresh_factor_ic()
        elif task == "daily_report":
            self._daily_report()

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
        """日K数据自动同步。"""
        from sequoia_x.data.engine import DataEngine
        try:
            engine = DataEngine(self.settings)
            n = engine.sync_today_bulk()
            logger.info(f"定时数据同步完成：写入 {n} 只")
        except Exception as e:
            logger.warning(f"定时数据同步失败：{e!r}")

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
                "intraday_scan_start": "盘中信号轮询",
                "sync_daily": "日K同步",
                "sync_lhb": "龙虎榜同步",
                "sync_fund_flow": "资金流向同步",
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

            # 补充系统状态
            lines.append("")
            latest_k = ""
            fin_cov = 0
            total_stocks = 0
            try:
                with sqlite3.connect(self.db_path) as conn:
                    latest_k = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0] or "-"
                    fin_cov = conn.execute("SELECT COUNT(DISTINCT symbol) FROM stock_finance").fetchone()[0]
                    total_stocks = conn.execute("SELECT COUNT(*) FROM stock_basic").fetchone()[0]
            except Exception:
                pass
            lines.append(f"---\n日K最新: {latest_k}")
            lines.append(f"财报覆盖: {fin_cov}/{total_stocks} ({fin_cov/total_stocks*100:.0f}%)")

            # 限流器状态
            try:
                from sequoia_x.core.rate_limiter import _rate_limiter
                rl = _rate_limiter.baostock_status()
                lines.append(f"baostock额度: {rl['used']}/{rl['limit']} ({rl['usage_pct']:.0f}%)")
            except Exception:
                pass

            content = "\n".join(lines)
            notifier = FeishuNotifier(self.settings)
            notifier.send_text("每日任务汇总", content)
            logger.info(f"每日汇总已推送飞书：{success}成功/{failed}失败/{total}总计")
            return f"{success}成功/{failed}失败"
        except Exception as e:
            logger.warning(f"每日汇总失败：{e!r}")
            return f"失败: {e}"

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
                engine, hold_days=20, sample_size=500, rolling_months=6
            )
            factors = result.get("factors", [])
            effective = [f for f in factors if abs(f.get("ic_mean", 0)) > 0.015
                         and abs(f.get("icir", 0)) > 0.3]
            logger.info(f"因子IC刷新完成：{len(effective)}/{len(factors)}个有效因子（滚动6个月）")
        except Exception as e:
            logger.warning(f"因子IC刷新失败：{e!r}")

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

        # 历史回填（增量，只补充最近60天）
        try:
            import sqlite3
            from sequoia_x.data.fund_flow_history import backfill_fund_flow_history
            with sqlite3.connect(self.db_path) as conn:
                symbols = [r[0] for r in conn.execute(
                    "SELECT DISTINCT symbol FROM ("
                    "SELECT symbol FROM decision_pool UNION "
                    "SELECT symbol FROM lhb_detail) "
                    "ORDER BY symbol LIMIT 500"
                ).fetchall()]
            if symbols:
                result = backfill_fund_flow_history(
                    self.db_path, symbols, days=60, n_workers=3
                )
                logger.info(f"资金流向历史回填：{result}")
        except Exception as e:
            logger.warning(f"资金流向历史回填失败：{e!r}")

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

            # Step1: 持仓扫描 → 自动卖出
            pos_result = services.scan_positions(apply_stop_move=False)
            sell_result = services.paper_auto_sell(pos_result.get("signals", []))

            # Step2: 全策略决策
            decision = services.generate_decision(
                strategy_keys=None, capital=100000, min_score=50,
                exclude_markets=None, exclude_st=True,
                max_candidates=60, include_auction=False,
            )

            # Step3: 自动买入
            buy_result = services.paper_auto_buy(decision)

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
            try:
                notifier.send_text(summary)
            except Exception:
                pass
            logger.info(f"模拟盘执行完成：买{len(buy_result.get('bought',[]))} 卖{len(sell_result.get('sold',[]))} 收益{perf.total_return_pct}%")
        except Exception as e:
            logger.warning(f"模拟盘自动执行失败：{e!r}")

    def _intraday_loop(self) -> None:
        """盘中信号轮询：9:30-15:00 每60秒扫描关注池。"""
        from sequoia_x.analysis.intraday_scanner import IntradayScanner
        from sequoia_x.notify.feishu import FeishuNotifier
        try:
            scanner = IntradayScanner(self.db_path)
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
                try:
                    signals = scanner.scan_once(notifier=notifier)
                    count += len(signals)
                except Exception as e:
                    logger.warning(f"盘中扫描异常：{e!r}")

                # 持仓盘中盯盘（每轮同步扫描，复用同一循环）
                try:
                    self._position_monitor(notifier)
                except Exception as e:
                    logger.warning(f"持仓监控异常：{e!r}")

                self._stop.wait(60)  # 每60秒一轮
            logger.info(f"盘中轮询结束，累计推送信号 {count} 个")
        except Exception as e:
            logger.warning(f"盘中轮询启动失败：{e!r}")

    def _position_monitor(self, notifier=None) -> None:
        """持仓盘中实时盯盘：批量快照 + 实时MA + 信号推送。"""
        from sequoia_x.analysis.position import PositionTracker
        try:
            from sequoia_x.data.engine import DataEngine
            engine = DataEngine(self.settings)
            tracker = PositionTracker(engine, self.settings)
            signals = tracker.scan_intraday(notifier=notifier)
            danger = [s for s in signals if s.signal_level == "danger"]
            if danger:
                logger.warning(f"持仓盘中预警：{len(danger)}只触发止损/清仓信号")
        except Exception as e:
            logger.warning(f"持仓监控失败：{e!r}")
