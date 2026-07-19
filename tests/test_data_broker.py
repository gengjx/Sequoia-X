"""DataBroker 多源降级编排测试。"""

from __future__ import annotations

import pandas as pd

from sequoia_x.data.sources import DataBroker, FetchResult, get_broker, reset_broker


class _FakeDailySource:
    """可控的日 K 源桩：按预设行为返回数据 / 抛异常 / 健康检查失败。"""

    def __init__(
        self,
        name: str,
        data: pd.DataFrame | None = None,
        healthy: bool = True,
        exc: Exception | None = None,
    ) -> None:
        self.name = name
        self._data = data
        self._healthy = healthy
        self._exc = exc
        self.calls = 0

    def health(self) -> bool:
        return self._healthy

    def fetch_daily(self, symbol: str, start: str, end: str, adjust: str = "hfq") -> pd.DataFrame:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._data if self._data is not None else pd.DataFrame()


def _df() -> pd.DataFrame:
    return pd.DataFrame({"open": [1.0], "high": [2.0], "close": [1.5]})


def test_first_healthy_source_wins():
    """首个健康且非空的源命中，后续源不被调用。"""
    a = _FakeDailySource("baostock", data=_df())
    b = _FakeDailySource("eastmoney", data=_df())
    broker = DataBroker()
    broker.register(a)
    broker.register(b)

    result = broker.fetch_daily("000001", "2024-01-01", "2024-06-01")

    assert result.source == "baostock"
    assert not result.data.empty
    assert a.calls == 1
    assert b.calls == 0  # 首源命中，次源未触发


def test_fallback_on_empty():
    """首源返回空 DataFrame 时降级到下一个源。"""
    a = _FakeDailySource("baostock", data=pd.DataFrame())  # 空
    b = _FakeDailySource("eastmoney", data=_df())
    broker = DataBroker()
    broker.register(a)
    broker.register(b)

    result = broker.fetch_daily("000001", "2024-01-01", "2024-06-01")

    assert result.source == "eastmoney"
    assert a.calls == 1 and b.calls == 1


def test_fallback_on_exception():
    """单源抛异常被隔离，降级到下一个源。"""
    a = _FakeDailySource("baostock", exc=ConnectionError("baostock down"))
    b = _FakeDailySource("eastmoney", data=_df())
    broker = DataBroker()
    broker.register(a)
    broker.register(b)

    result = broker.fetch_daily("000001", "2024-01-01", "2024-06-01")

    assert result.source == "eastmoney"  # 异常被隔离，未中断


def test_unhealthy_source_skipped():
    """健康检查失败的源被跳过，其 fetch_daily 不被调用。"""
    a = _FakeDailySource("baostock", healthy=False, data=_df())
    b = _FakeDailySource("eastmoney", data=_df())
    broker = DataBroker()
    broker.register(a)
    broker.register(b)

    result = broker.fetch_daily("000001", "2024-01-01", "2024-06-01")

    assert result.source == "eastmoney"
    assert a.calls == 0  # 不健康 → 未调用


def test_all_fail_returns_empty():
    """所有源均失败时返回空 DataFrame 且 source=None。"""
    a = _FakeDailySource("baostock", exc=RuntimeError())
    b = _FakeDailySource("eastmoney", data=pd.DataFrame())
    broker = DataBroker()
    broker.register(a)
    broker.register(b)

    result = broker.fetch_daily("000001", "2024-01-01", "2024-06-01")

    assert isinstance(result, FetchResult)
    assert result.data.empty
    assert result.source is None


def test_capability_filtering():
    """只调用实现了对应能力协议的 provider。"""
    class DailyOnly:
        name = "daily_only"
        def health(self): return True
        def fetch_daily(self, symbol, start, end, adjust="hfq"): return _df()

    class MinuteOnly:
        name = "minute_only"
        def health(self): return True
        def fetch_minute(self, symbol, date): return _df()

    broker = DataBroker()
    broker.register(DailyOnly())  # type: ignore[arg-type]
    broker.register(MinuteOnly())  # type: ignore[arg-type]

    daily = broker.fetch_daily("000001", "2024-01-01", "2024-06-01")
    minute = broker.fetch_minute("000001", "2024-07-01")

    assert daily.source == "daily_only"
    assert minute.source == "minute_only"


def test_broker_singleton_is_process_global():
    """get_broker 返回进程级单例，reset 后重建。"""
    reset_broker()
    b1 = get_broker()
    b2 = get_broker()
    assert b1 is b2
    reset_broker()
    b3 = get_broker()
    assert b3 is not b1
    reset_broker()
