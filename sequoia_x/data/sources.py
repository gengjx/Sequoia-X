"""数据源抽象层：统一多源行情接入契约 + 降级编排。

将散落在 :class:`DataEngine` 中的 baostock / 东财 / 腾讯 / 新浪 / akshare 直连调用，
收敛为统一的「能力协议 + Broker 编排」架构：

- :class:`DataSource`：基础协议（name + health 健康检查）
- :class:`DailySource` / :class:`MinuteSource` / :class:`LhbSource` /
  :class:`FundFlowSource` / :class:`ValuationSource`：按能力分项，数据源按需实现
- :class:`DataBroker`：按 ``source_order`` 依次尝试、自动降级、异常隔离、来源标记

设计要点
--------
1. **契约先行**：新数据源只需实现对应 Protocol 并 ``register`` 到 broker，即可被
   统一调度，避免再向业务文件直接散布 ``requests.get`` / ``baostock`` 调用。
2. **降级隔离**：单个 provider 抛异常或返回空不影响后续 provider；全部失败返回
   空结构并标记 ``source=None``，由调用方决定是否报错。
3. **不接管 RateLimiter**：现有 ``core.rate_limiter`` 是源特定的命令式配额/熔断器，
   保持由具体 Provider 自行调用；Broker 只负责编排顺序与健康检查，避免破坏既有配额。
4. **渐进迁移**：现有 ``DataEngine`` 可通过适配器包装为 Provider 接入 broker；
   engine 内部 43 处直连调用的全量迁移是独立大工程，本模块提供标准接入点。

典型用法
--------
>>> broker = get_broker()
>>> broker.register(BaostockProvider())
>>> df, src = broker.fetch_daily("000001", "2024-01-01", "2024-06-01")
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 能力协议（数据源按需实现）
# ---------------------------------------------------------------------------
@runtime_checkable
class DataSource(Protocol):
    """数据源基础协议：具备名称与健康检查。"""

    name: str

    def health(self) -> bool:
        """源是否可用（如网络可达、配额未耗尽）。返回 False 则 Broker 跳过。"""
        ...


@runtime_checkable
class DailySource(DataSource, Protocol):
    """日 K 线能力（含复权）。"""

    def fetch_daily(
        self, symbol: str, start: str, end: str, adjust: str = "hfq"
    ) -> pd.DataFrame:
        """返回列含 open/high/low/close/volume/turnover 的日 K DataFrame（可能为空）。"""
        ...


@runtime_checkable
class MinuteSource(DataSource, Protocol):
    """分钟 K 线能力。"""

    def fetch_minute(self, symbol: str, date: str) -> pd.DataFrame:
        ...


@runtime_checkable
class LhbSource(DataSource, Protocol):
    """龙虎榜能力。"""

    def fetch_lhb(self, date_str: str) -> pd.DataFrame:
        ...


@runtime_checkable
class FundFlowSource(DataSource, Protocol):
    """资金流能力。"""

    def fetch_fund_flow(self, date_str: str) -> pd.DataFrame:
        ...


@runtime_checkable
class ValuationSource(DataSource, Protocol):
    """估值（PE/PB/市值）快照能力。"""

    def fetch_valuation(self) -> pd.DataFrame:
        ...


# ---------------------------------------------------------------------------
# 编排层
# ---------------------------------------------------------------------------
@dataclass
class FetchResult:
    """单次抓取结果，标记实际命中的源（全失败时 source=None）。"""

    data: pd.DataFrame
    source: str | None


class DataBroker:
    """多源数据 Broker：按注册顺序降级抓取，异常隔离。

    线程安全（注册与抓取均加锁）。各 ``fetch_*`` 返回 :class:`FetchResult`，
    ``data`` 为空 DataFrame 时表示所有源均失败。
    """

    def __init__(self) -> None:
        self._providers: list[DataSource] = []
        self._lock = threading.Lock()

    def register(self, provider: DataSource) -> None:
        """注册一个数据源（靠前的优先级更高）。"""
        with self._lock:
            self._providers.append(provider)

    @property
    def providers(self) -> list[DataSource]:
        with self._lock:
            return list(self._providers)

    def _try(
        self,
        capability: type,
        method: str,
        *args: object,
        **kwargs: object,
    ) -> FetchResult:
        """按顺序尝试具备 *capability* 的 provider，首个非空成功即返回。"""
        with self._lock:
            candidates = [p for p in self._providers if isinstance(p, capability)]
        for p in candidates:
            try:
                if not p.health():
                    logger.debug(f"数据源 {p.name} 健康检查未通过，跳过")
                    continue
                result = getattr(p, method)(*args, **kwargs)
                if result is None:
                    continue
                if isinstance(result, pd.DataFrame):
                    if result.empty:
                        continue
                    return FetchResult(data=result, source=p.name)
                return FetchResult(data=result, source=p.name)
            except Exception as exc:  # noqa: BLE001 — 隔离单源故障
                logger.warning(f"数据源 {p.name}.{method} 异常：{exc}")
                continue
        logger.info(f"所有 {capability.__name__} 源均未提供 {method} 数据")
        return FetchResult(data=pd.DataFrame(), source=None)

    def fetch_daily(
        self, symbol: str, start: str, end: str, adjust: str = "hfq"
    ) -> FetchResult:
        return self._try(DailySource, "fetch_daily", symbol, start, end, adjust)

    def fetch_minute(self, symbol: str, date: str) -> FetchResult:
        return self._try(MinuteSource, "fetch_minute", symbol, date)

    def fetch_lhb(self, date_str: str) -> FetchResult:
        return self._try(LhbSource, "fetch_lhb", date_str)

    def fetch_fund_flow(self, date_str: str) -> FetchResult:
        return self._try(FundFlowSource, "fetch_fund_flow", date_str)

    def fetch_valuation(self) -> FetchResult:
        return self._try(ValuationSource, "fetch_valuation")


# ---------------------------------------------------------------------------
# 进程级单例
# ---------------------------------------------------------------------------
_broker: DataBroker | None = None
_broker_lock = threading.Lock()


def get_broker() -> DataBroker:
    """返回进程级唯一 DataBroker 单例。

    首次调用创建空 broker；具体 Provider 应在应用启动时（如 ``main.py`` /
    ``web/app.py``）通过 ``register`` 注入。保持惰性，便于测试替换。
    """
    global _broker
    if _broker is None:
        with _broker_lock:
            if _broker is None:
                _broker = DataBroker()
    return _broker


def reset_broker() -> None:
    """重置单例（仅供测试使用）。"""
    global _broker
    with _broker_lock:
        _broker = None
