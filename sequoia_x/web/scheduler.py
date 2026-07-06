"""轻量定时调度器：基于 threading.Timer，无外部依赖。

A股交易时段调度（Asia/Shanghai）：
  09:25  → 集合竞价扫描 + 飞书推送
  18:00  → 日K数据自动同步（解决手动同步痛点）
  19:00  → 策略评估权重刷新（可选）

非交易日（周末/节假日）自动跳过。
"""

from __future__ import annotations

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
        (18, 0, "sync_daily"),
        (18, 30, "auction_verify"),
    ]

    def __init__(self, settings: Settings, db_path: str) -> None:
        self.settings = settings
        self.db_path = db_path
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_run: dict[str, str] = {}  # {task_name: "YYYY-MM-DD"}

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
                        try:
                            self._run_task(task)
                            self._last_run[key] = now.strftime("%H:%M")
                        except Exception as e:
                            logger.warning(f"定时任务 {task} 失败：{e!r}")

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
                self._stop.wait(60)  # 每60秒一轮
            logger.info(f"盘中轮询结束，累计推送信号 {count} 个")
        except Exception as e:
            logger.warning(f"盘中轮询启动失败：{e!r}")
