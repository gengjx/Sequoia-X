"""全局数据源限流器：保护 baostock/东财 API 不被超频/超量调用。

问题背景：
  - baostock 每日 5 万次调用限额（login+query 都算）
  - 东财 push2his/push2delay 高频请求会被封 IP（1~24小时）
  - 多个任务各自独立调用，没有全局协调

限流策略：
  1. 日额度：baostock 每日上限 40000 次（留 20% 安全余量）
  2. 频率限制：东财每秒最多 3 次请求（避免触发风控）
  3. 失败退避：连续失败时指数退避，超过阈值自动熔断
  4. 状态持久化：额度计数写入文件，重启不重置
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# ── 限额配置 ──
BAOSTOCK_DAILY_LIMIT = 40_000      # baostock 日限额（官方5万，留20%余量）
EASTMONEY_MIN_INTERVAL = 0.4       # 东财请求最小间隔秒（2.5次/秒）
MAX_CONSECUTIVE_FAILURES = 10      # 连续失败熔断阈值
CIRCUIT_BREAKER_COOLDOWN = 3600    # 熔断冷却时间（秒）

_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", ".api_rate_limit.json"
)


class RateLimiter:
    """全局数据源限流器（单例）。"""

    _instance: RateLimiter | None = None
    _lock = threading.Lock()

    def __new__(cls) -> RateLimiter:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._baostock_count = 0
        self._baostock_date = datetime.now().strftime("%Y-%m-%d")
        self._eastmoney_last_call = 0.0
        self._eastmoney_lock = threading.Lock()
        self._failures: dict[str, int] = {}  # source -> consecutive failures
        self._circuit_until: dict[str, float] = {}  # source -> timestamp
        self._state_lock = threading.Lock()
        self._load_state()

    def _load_state(self) -> None:
        """从文件恢复日额度计数。"""
        try:
            if os.path.exists(_STATE_FILE):
                with open(_STATE_FILE) as f:
                    state = json.load(f)
                today = datetime.now().strftime("%Y-%m-%d")
                if state.get("date") == today:
                    self._baostock_count = state.get("baostock_count", 0)
                    logger.info(f"限流器恢复：baostock今日已用 {self._baostock_count}/{BAOSTOCK_DAILY_LIMIT}")
        except Exception:
            pass

    def _save_state(self) -> None:
        """持久化日额度计数。"""
        try:
            Path(_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(_STATE_FILE, "w") as f:
                json.dump({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "baostock_count": self._baostock_count,
                }, f)
        except Exception:
            pass

    # ════════════════════════════════════════
    # baostock 限流
    # ════════════════════════════════════════

    def baostock_check(self, estimated_calls: int = 1) -> bool:
        """检查 baostock 是否还能调用。

        Args:
            estimated_calls: 预计本次操作需要几次调用

        Returns:
            True=可以调用，False=已达限额
        """
        self._check_date_rollover()
        with self._state_lock:
            if self._baostock_count + estimated_calls > BAOSTOCK_DAILY_LIMIT:
                logger.warning(
                    f"baostock 日限额保护：今日已用 {self._baostock_count}/"
                    f"{BAOSTOCK_DAILY_LIMIT}，本次需 {estimated_calls}，拒绝调用"
                )
                return False
            return True

    def baostock_consume(self, count: int = 1) -> None:
        """记录 baostock 调用消耗。"""
        self._check_date_rollover()
        with self._state_lock:
            self._baostock_count += count
            self._save_state()

    def baostock_status(self) -> dict:
        """获取 baostock 当前限额状态。"""
        self._check_date_rollover()
        return {
            "used": self._baostock_count,
            "limit": BAOSTOCK_DAILY_LIMIT,
            "remaining": max(0, BAOSTOCK_DAILY_LIMIT - self._baostock_count),
            "usage_pct": round(self._baostock_count / BAOSTOCK_DAILY_LIMIT * 100, 1),
        }

    # ════════════════════════════════════════
    # 东财限流
    # ════════════════════════════════════════

    def eastmoney_acquire(self) -> bool:
        """获取东财调用许可（带频率限制）。

        Returns:
            True=可以调用，False=熔断中
        """
        if self._is_circuit_breaker("eastmoney"):
            return False

        with self._eastmoney_lock:
            now = time.time()
            elapsed = now - self._eastmoney_last_call
            if elapsed < EASTMONEY_MIN_INTERVAL:
                time.sleep(EASTMONEY_MIN_INTERVAL - elapsed)
            self._eastmoney_last_call = time.time()
            return True

    def eastmoney_success(self) -> None:
        """记录东财调用成功。"""
        with self._state_lock:
            self._failures["eastmoney"] = 0

    def eastmoney_failure(self) -> None:
        """记录东财调用失败（达到阈值触发熔断）。"""
        with self._state_lock:
            self._failures["eastmoney"] = self._failures.get("eastmoney", 0) + 1
            if self._failures["eastmoney"] >= MAX_CONSECUTIVE_FAILURES:
                self._circuit_until["eastmoney"] = time.time() + CIRCUIT_BREAKER_COOLDOWN
                logger.warning(
                    f"东财熔断：连续失败 {self._failures['eastmoney']} 次，"
                    f"冷却 {CIRCUIT_BREAKER_COOLDOWN // 60} 分钟"
                )

    # ════════════════════════════════════════
    # 通用熔断
    # ════════════════════════════════════════

    def _is_circuit_breaker(self, source: str) -> bool:
        """检查某数据源是否在熔断冷却中。"""
        until = self._circuit_until.get(source, 0)
        if time.time() < until:
            remaining = int(until - time.time())
            logger.warning(f"{source} 熔断中，剩余冷却 {remaining // 60} 分钟")
            return True
        return False

    # ════════════════════════════════════════
    # 内部方法
    # ════════════════════════════════════════

    def _check_date_rollover(self) -> None:
        """日期切换时重置日额度。"""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._baostock_date:
            self._baostock_count = 0
            self._baostock_date = today
            self._save_state()
            logger.info(f"baostock 日额度重置（{today}）")

    def status(self) -> dict:
        """获取所有数据源限流状态。"""
        return {
            "baostock": self.baostock_status(),
            "eastmoney": {
                "failures": self._failures.get("eastmoney", 0),
                "circuit_breaker": self._is_circuit_breaker("eastmoney"),
            },
        }


# 全局单例
_rate_limiter = RateLimiter()
