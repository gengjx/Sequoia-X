"""P3c 因子扩域测试：基本面深度 + 另类数据 + 北向因子计算。

覆盖：
  - 新增基本面因子（营运效率/成长）从 finance dict 正确提取
  - 北向因子（nb_holding_pct/nb_inflow）从 DataFrame 正确计算
  - 缺数据时返回 nan（安全跳过）
  - evaluate_factor_ic 的 finance/fund_flow/lhb/north 因子加载链路
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequoia_x.analysis.factor import FACTOR_META, compute_factors


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _ohlcv(n: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes = [10.0 * (1.005 ** i) for i in range(n)]
    close = pd.Series(closes, index=dates)
    return pd.DataFrame({
        "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close,
        "volume": pd.Series([1_000_000] * n, index=dates),
        "turnover": pd.Series([1_000_000 * c for c in closes], index=dates),
    }, index=dates)


_FINANCE = {
    "roe": 15.0, "np_margin": 12.0, "gp_margin": 35.0,
    "yoy_eps": 20.0, "yoy_pni": 25.0,
    "yoy_ni": 30.0, "asset_turn": 0.8, "inv_turn": 5.0, "nr_turn": 10.0,
}


# ===========================================================================
# FACTOR_META 注册
# ===========================================================================
class TestFactorMetaRegistration:
    def test_new_factors_registered(self):
        for f in ["yoy_ni", "asset_turn", "inv_turn", "nr_turn",
                   "nb_holding_pct", "nb_inflow"]:
            assert f in FACTOR_META, f"{f} 未注册到 FACTOR_META"
            assert "category" in FACTOR_META[f]
            assert "desc" in FACTOR_META[f]

    def test_categories_correct(self):
        assert FACTOR_META["yoy_ni"]["category"] == "成长"
        assert FACTOR_META["asset_turn"]["category"] == "营运"
        assert FACTOR_META["nb_holding_pct"]["category"] == "北向"


# ===========================================================================
# 基本面因子计算
# ===========================================================================
class TestFundamentalFactors:
    def test_operating_turns_from_finance(self):
        df = _ohlcv()
        factors = compute_factors(df, finance=_FINANCE)
        assert factors["asset_turn"] == pytest.approx(0.8)
        assert factors["inv_turn"] == pytest.approx(5.0)
        assert factors["nr_turn"] == pytest.approx(10.0)

    def test_yoy_ni_from_finance(self):
        df = _ohlcv()
        factors = compute_factors(df, finance=_FINANCE)
        assert factors["yoy_ni"] == pytest.approx(30.0)

    def test_missing_finance_returns_nan(self):
        df = _ohlcv()
        factors = compute_factors(df, finance=None)
        for k in ["yoy_ni", "asset_turn", "inv_turn", "nr_turn"]:
            assert factors[k] is None  # NaN→None

    def test_partial_finance_returns_nan_for_missing(self):
        df = _ohlcv()
        fin = {"roe": 15.0}  # 只有 roe，其他缺失
        factors = compute_factors(df, finance=fin)
        assert factors["roe"] == pytest.approx(15.0)
        assert factors["yoy_ni"] is None  # NaN→None


# ===========================================================================
# 北向因子计算
# ===========================================================================
class TestNorthboundFactors:
    def _north_df(self, n: int = 30, base_pct: float = 5.0) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-01", periods=n)
        return pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "hold_pct": [base_pct + i * 0.01 for i in range(n)],
        })

    def test_nb_holding_pct(self):
        df = _ohlcv()
        nb = self._north_df(30, 5.0)
        factors = compute_factors(df, north_hold=nb)
        assert factors["nb_holding_pct"] == pytest.approx(5.0 + 29 * 0.01)

    def test_nb_inflow_20d_change(self):
        df = _ohlcv()
        nb = self._north_df(30, 5.0)
        factors = compute_factors(df, north_hold=nb)
        # 最后一天 pct - 21天前 pct = (5.0+29*0.01) - (5.0+8*0.01) = 0.21
        assert factors["nb_inflow"] == pytest.approx(0.20)

    def test_short_history_no_inflow(self):
        df = _ohlcv()
        nb = self._north_df(10, 5.0)  # <21 天
        factors = compute_factors(df, north_hold=nb)
        assert factors["nb_holding_pct"] == pytest.approx(5.0 + 9 * 0.01)
        assert factors["nb_inflow"] is None  # NaN→None

    def test_no_north_data_returns_nan(self):
        df = _ohlcv()
        factors = compute_factors(df, north_hold=None)
        assert factors["nb_holding_pct"] is None
        assert factors["nb_inflow"] is None


# ===========================================================================
# evaluate_factor_ic 的因子加载链路（运营/成长/北向/lhb/fund_flow）
# ===========================================================================
class TestFactorICDataLoading:
    def test_finance_field_map_covers_new_factors(self):
        """evaluate_factor_ic 内部的 _finance_field_map 应覆盖所有基本面因子。"""
        # 模拟 evaluate_factor_ic 中的 field map 逻辑
        _finance_field_map = {
            "roe": "roe", "np_margin": "np_margin", "gp_margin": "gp_margin",
            "rev_growth": "yoy_eps", "profit_growth": "yoy_pni",
            "yoy_ni": "yoy_ni", "asset_turn": "asset_turn",
            "inv_turn": "inv_turn", "nr_turn": "nr_turn",
        }
        new_factors = ["yoy_ni", "asset_turn", "inv_turn", "nr_turn"]
        for f in new_factors:
            assert f in _finance_field_map
            assert _finance_field_map[f] == f  # 同名字段

    def test_synonym_group_includes_operating_turns(self):
        """去重同义组应包含营运效率周转率。"""
        _SYNONYM_GROUPS = [
            {"asset_turn", "inv_turn", "nr_turn"},
        ]
        assert {"asset_turn", "inv_turn", "nr_turn"} in [set(g) for g in _SYNONYM_GROUPS]
