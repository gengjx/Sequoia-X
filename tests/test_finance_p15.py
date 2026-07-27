"""P15 盈利质量因子测试：现金流比率（CFOToOR / CFOToNP）。

覆盖：
  - compute_factors 从 finance dict 正确提取 cfo_yield/earnings_quality
  - 缺数据返回 None
  - FACTOR_META 注册两个新因子
  - finance_sync 列表含现金流列 + query_cash_flow_data 调用
  - _stat_date_to_yq 日期→季度解析正确
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from sequoia_x.analysis.factor import FACTOR_META, compute_factors
from sequoia_x.data.finance_sync import _stat_date_to_yq, _fetch_batch


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _ohlcv(n: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes = [10.0 * (1.005 ** i) for i in range(n)]
    close = pd.Series(closes, index=dates)
    high = close * 1.01
    low = close * 0.99
    op = close.shift(1).fillna(close)
    vol = pd.Series([1e6] * n, index=dates)
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close,
                         "volume": vol, "turnover": vol * close})


# ===========================================================================
# compute_factors 现金流因子
# ===========================================================================
class TestCashFlowFactors:
    def test_factor_meta_registered(self):
        assert "cfo_yield" in FACTOR_META
        assert "earnings_quality" in FACTOR_META
        assert FACTOR_META["cfo_yield"]["category"] == "盈利质量"
        assert FACTOR_META["earnings_quality"]["category"] == "盈利质量"

    def test_cashflow_factors_when_provided(self):
        """传入 finance dict 含 cfo_to_or/cfo_to_np 时因子被填充。"""
        finance = {"roe": 15.0, "np_margin": 20.0, "gp_margin": 40.0,
                   "yoy_eps": 10.0, "yoy_pni": 25.0,
                   "cfo_to_or": 0.12, "cfo_to_np": 1.05}
        f = compute_factors(_ohlcv(20), finance=finance)
        assert f["cfo_yield"] == pytest.approx(0.12)
        assert f["earnings_quality"] == pytest.approx(1.05)

    def test_cashflow_factors_none_when_absent(self):
        """未传 finance 时现金流因子为 None。"""
        f = compute_factors(_ohlcv(20))
        assert f["cfo_yield"] is None
        assert f["earnings_quality"] is None

    def test_cashflow_factors_none_when_values_missing(self):
        """finance dict 有但缺现金流字段 → None。"""
        finance = {"roe": 15.0, "np_margin": 20.0}
        f = compute_factors(_ohlcv(20), finance=finance)
        assert f["cfo_yield"] is None
        assert f["earnings_quality"] is None

    def test_cashflow_factors_none_when_null(self):
        """finance dict 现金流字段为 None → None。"""
        finance = {"roe": 15.0, "cfo_to_or": None, "cfo_to_np": None}
        f = compute_factors(_ohlcv(20), finance=finance)
        assert f["cfo_yield"] is None
        assert f["earnings_quality"] is None


# ===========================================================================
# finance_sync 现金流列 + worker
# ===========================================================================
class TestFinanceSyncCashFlow:
    def test_cols_include_cashflow(self):
        """_fetch_batch 的 cols 列表含 4 个现金流列。"""
        src = inspect.getsource(_fetch_batch)
        for col in ("cfo_to_or", "cfo_to_np", "cfo_to_gr", "tangible_ratio"):
            assert col in src

    def test_worker_queries_cash_flow(self):
        """worker 调用 query_cash_flow_data。"""
        src = inspect.getsource(_fetch_batch)
        assert "query_cash_flow_data" in src


# ===========================================================================
# 日期 → 季度解析
# ===========================================================================
class TestStatDateToYQ:
    @pytest.mark.parametrize("stat_date, expected", [
        ("2021-12-31", (2021, 4)),
        ("2024-03-31", (2024, 1)),
        ("2024-06-30", (2024, 2)),
        ("2024-09-30", (2024, 3)),
        ("2025-01-01", (2025, 1)),
    ])
    def test_valid_dates(self, stat_date, expected):
        assert _stat_date_to_yq(stat_date) == expected

    @pytest.mark.parametrize("bad", [None, "", "bad", "2021"])
    def test_invalid_returns_none(self, bad):
        assert _stat_date_to_yq(bad) is None
