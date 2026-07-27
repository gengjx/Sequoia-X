"""全局数据源限流器：保护 baostock/东财 API 不被超频/超量调用。

问题背景：
  - baostock 每日 5 万次调用限额（login+query 都算）
  - 东财 push2his/push2delay 高频请求会被封 IP（1~24小时）
  - 多个任务各自独立调用，没有全局协调

限流策略：
  1. 日额度：baostock 每日上限 48000 次（官方5万，留4%余量给增量任务）
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
BAOSTOCK_DAILY_LIMIT = 48_000      # baostock 日限额（官方5万，留4%余量给增量任务）
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
        self._save_state()  # 确保状态文件存在

    def _load_state(self) -> None:
        """从文件恢复日额度计数。"""
        try:
            if os.path.exists(_STATE_FILE):
                with open(_STATE_FILE) as f:
                    state = json.load(f)
                today = datetime.now().strftime("%Y-%m-%d")
                if state.get("date") == today:
                    self._baostock_count = state.get("baostock_count", 0)
                    self._failures["eastmoney"] = state.get("em_failures", 0)
                    cb_until = state.get("em_circuit_until", 0)
                    if cb_until > time.time():
                        self._circuit_until["eastmoney"] = cb_until
                    logger.info(
                        f"限流器恢复：baostock今日已用 {self._baostock_count}/{BAOSTOCK_DAILY_LIMIT}"
                        + (f"，东财熔断中(剩余{int(cb_until - time.time())}s)" if cb_until > time.time() else "")
                    )
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
                    "em_failures": self._failures.get("eastmoney", 0),
                    "em_circuit_until": self._circuit_until.get("eastmoney", 0),
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

    def baostock_try_consume(self, n: int = 1) -> bool:
        """原子检查并扣减 baostock 额度（跨进程安全，fcntl 文件锁）。

        多进程 Pool worker 和并发脚本共享同一个文件计数，
        每次真实 API 调用前扣减，彻底消除多进程计数不共享和事后批量记账的缺陷。

        Args:
            n: 本次预计消耗的次数

        Returns:
            True=已扣减成功可调用，False=已达限额应停止
        """
        import fcntl

        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._baostock_date:
            self._check_date_rollover()

        try:
            with open(_STATE_FILE, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    raw = f.read()
                    state = json.loads(raw) if raw.strip() else {}
                    if state.get("date") != today:
                        state["date"] = today
                        state["baostock_count"] = 0
                    count = state.get("baostock_count", 0)
                    if count + n > BAOSTOCK_DAILY_LIMIT:
                        return False
                    count += n
                    state["baostock_count"] = count
                    f.seek(0)
                    f.truncate()
                    f.write(json.dumps(state))
                    f.flush()
                    os.fsync(f.fileno())
                    with self._state_lock:
                        self._baostock_count = count
                        self._baostock_date = today
                    return True
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            return True

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
            self._save_state()

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
            self._save_state()

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

    @staticmethod
    def probe_eastmoney(endpoint: str = "push2his") -> bool:
        """探测东财某接口是否可用（发1个轻量请求）。

        Args:
            endpoint: "push2his" | "push2delay"

        Returns:
            True=可用, False=被封
        """
        import requests as _req
        if endpoint == "push2his":
            url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            params = {"secid": "1.600519", "klt": "101", "fqt": "1", "beg": "20260101", "end": "20260102", "fields1": "f1", "fields2": "f51"}
        else:
            url = "https://push2delay.eastmoney.com/api/qt/stock/get"
            params = {"secid": "0.000001", "fields": "f43"}
        try:
            r = _req.get(url, params=params, timeout=6)
            return r.status_code == 200
        except Exception:
            return False


def em_get(url: str, **kwargs) -> "requests.Response":
    """东财 HTTP GET 封装：自动频率控制 + 熔断 + 失败计数。

    所有东财请求都应走这个函数，统一限流。

    Raises:
        ConnectionError: 熔断中
        requests.RequestException: 底层请求失败
    """
    import requests as _req
    if not _rate_limiter.eastmoney_acquire():
        raise ConnectionError("东财熔断中，请稍后重试")
    kwargs.setdefault("timeout", 10)
    try:
        resp = _req.get(url, **kwargs)
        if resp.status_code == 200:
            _rate_limiter.eastmoney_success()
            return resp
        else:
            _rate_limiter.eastmoney_failure()
            resp.raise_for_status()
            return resp
    except Exception:
        _rate_limiter.eastmoney_failure()
        raise


# 全局单例
_rate_limiter = RateLimiter()
