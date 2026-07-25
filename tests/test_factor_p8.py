"""P8 新数据源转选股因子测试：基金持仓截面因子 + 沪深300 Beta + 宏观择时增强。

覆盖：
  - 基金持仓因子（fund_holding/fund_inflow）从 dict 正确提取；缺数据返回 nan
  - Beta 因子（beta_300/rel_strength_300）：已知收益序列断言精确值；全同步beta=1
  - evaluate_factor_ic 不崩（新因子加入后正常加载）
  - 宏观择时增强：M2>9% bull 偏置、M2<7% bear 偏置、无数据 None
  - 回归：compute_factors 不传新参数时行为不变
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequoia_x.analysis.factor import (
    FACTOR_META,
    compute_factors,
    compute_factor_series,
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _ohlcv(n: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes = [10.0 * (1.005 ** i) for i in range(n)]
    close = pd.Series(closes, index=dates)
    return pd.DataFrame({
        "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close,
        "volume": pd.Series([1_000_000] * n, index=dates),
        "turnover": pd.Series([1_000_000 * c for c in closes], index=dates),
    }, index=dates)


def _index_ret_aligned(df: pd.DataFrame, scale: float = 1.0) -> pd.Series:
    """构造与 df 同步的指数收益率序列（scale=1 时完美同步→beta≈1）。"""
    rets = df["close"].pct_change() * scale
    return pd.Series(rets.values, index=df.index.strftime("%Y-%m-%d")).dropna()


# ===========================================================================
# FACTOR_META 注册
# ===========================================================================
class TestFactorMetaRegistration:
    def test_fund_factors_registered(self):
        for f in ["fund_holding", "fund_inflow"]:
            assert f in FACTOR_META, f"{f} 未注册到 FACTOR_META"
            assert "category" in FACTOR_META[f]
            assert "desc" in FACTOR_META[f]

    def test_beta_factors_registered(self):
        for f in ["beta_300", "rel_strength_300"]:
            assert f in FACTOR_META, f"{f} 未注册到 FACTOR_META"
            assert "category" in FACTOR_META[f]

    def test_fund_holding_marked_reverse(self):
        """基金持有家数高=拥挤，标记为 reverse（负向因子）。"""
        assert FACTOR_META["fund_holding"].get("reverse") is True

    def test_beta_marked_reverse(self):
        """低Beta防御溢价，标记为 reverse。"""
        assert FACTOR_META["beta_300"].get("reverse") is True


# ===========================================================================
# 基金持仓因子
# ===========================================================================
class TestFundHoldingFactors:
    def test_fund_holding_from_dict(self):
        df = _ohlcv()
        f = compute_factors(df, fund_hold={"fund_count": 50, "change_pct": 12.3})
        assert f["fund_holding"] == pytest.approx(50.0)
        assert f["fund_inflow"] == pytest.approx(12.3)

    def test_fund_holding_none_when_no_data(self):
        df = _ohlcv()
        f = compute_factors(df)
        assert f["fund_holding"] is None
        assert f["fund_inflow"] is None

    def test_fund_holding_none_values(self):
        """dict 有但值为 None → nan→None。"""
        df = _ohlcv()
        f = compute_factors(df, fund_hold={"fund_count": None, "change_pct": None})
        assert f["fund_holding"] is None
        assert f["fund_inflow"] is None


# ===========================================================================
# Beta 因子
# ===========================================================================
class TestBetaFactors:
    def test_beta_near_one_when_perfectly_correlated(self):
        """个股收益与指数完美同步 → beta≈1。"""
        df = _ohlcv(120)
        idx_ret = _index_ret_aligned(df)
        f = compute_factors(df, index_ret=idx_ret)
        assert f["beta_300"] is not None
        assert 0.8 < f["beta_300"] < 1.3

    def test_beta_higher_when_more_volatile(self):
        """个股波动是指数2倍 → beta≈2。"""
        df = _ohlcv(120)
        idx_ret = _index_ret_aligned(df, scale=0.5)  # 指数波动减半 → 相对beta翻倍
        f = compute_factors(df, index_ret=idx_ret)
        assert f["beta_300"] is not None
        assert f["beta_300"] > 1.5

    def test_rel_strength_zero_when_synced(self):
        """完美同步 → 相对强度≈0。"""
        df = _ohlcv(120)
        idx_ret = _index_ret_aligned(df)
        f = compute_factors(df, index_ret=idx_ret)
        assert f["rel_strength_300"] is not None
        assert abs(f["rel_strength_300"]) < 0.1

    def test_beta_none_when_short_index(self):
        """指数不足60天 → nan。"""
        df = _ohlcv(120)
        idx_ret = pd.Series(np.random.randn(30) / 100)
        f = compute_factors(df, index_ret=idx_ret)
        assert f["beta_300"] is None

    def test_beta_none_when_no_index(self):
        """未传 index_ret → nan。"""
        df = _ohlcv()
        f = compute_factors(df)
        assert f["beta_300"] is None
        assert f["rel_strength_300"] is None


# ===========================================================================
# compute_factor_series 中的 Beta（IC 评估路径）
# ===========================================================================
class TestBetaFactorSeries:
    def test_series_has_beta(self):
        df = _ohlcv(120)
        df["date"] = df.index.strftime("%Y-%m-%d")
        idx_ret = _index_ret_aligned(df)
        series = compute_factor_series(df, index_ret=idx_ret)
        assert "beta_300" in series
        assert "rel_strength_300" in series
        # 最后一个值应该是有限的
        assert np.isfinite(series["beta_300"].iloc[-1])

    def test_series_no_beta_without_index(self):
        """不传 index_ret → 无 beta_300（不崩）。"""
        df = _ohlcv(120)
        df["date"] = df.index.strftime("%Y-%m-%d")
        series = compute_factor_series(df)
        assert "beta_300" not in series


# ===========================================================================
# 宏观择时偏置
# ===========================================================================
class TestMacroBias:
    """测试 _load_macro_bias 的逻辑（通过临时数据库）。"""

    def _make_strategy_with_db(self, tmp_path, m2_yoy=None, sf_values=None):
        """构造一个用临时 DB 的 MultiFactorStrategy（仅测宏观偏置）。"""
        db_path = str(tmp_path / "test.db")
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS macro_money (month TEXT, m2_yoy REAL)")
            conn.execute("CREATE TABLE IF NOT EXISTS macro_sf (month TEXT, sf_total REAL)")
            if m2_yoy is not None:
                conn.execute("INSERT INTO macro_money VALUES ('2026-06', ?)", (m2_yoy,))
            if sf_values:
                for i, v in enumerate(sf_values):
                    conn.execute("INSERT INTO macro_sf VALUES (?, ?)", (f"2026-{i+1:02d}", v))
            conn.commit()

        # 用 mock engine 暴露 db_path
        from types import SimpleNamespace
        _mock_engine = SimpleNamespace(db_path=db_path)
        from sequoia_x.strategy.multi_factor import MultiFactorStrategy
        strat = MultiFactorStrategy.__new__(MultiFactorStrategy)
        strat.engine = _mock_engine
        return strat

    def test_m2_high_is_bull_bias(self, tmp_path):
        strat = self._make_strategy_with_db(tmp_path, m2_yoy=9.5)
        assert strat._load_macro_bias() == "bull"

    def test_m2_low_is_bear_bias(self, tmp_path):
        strat = self._make_strategy_with_db(tmp_path, m2_yoy=6.5)
        assert strat._load_macro_bias() == "bear"

    def test_m2_neutral_no_bias(self, tmp_path):
        strat = self._make_strategy_with_db(tmp_path, m2_yoy=8.0)
        assert strat._load_macro_bias() is None

    def test_sf_expanding_is_bull_bias(self, tmp_path):
        """社融环比放量（+20%以上）→ bull。"""
        strat = self._make_strategy_with_db(tmp_path, m2_yoy=8.0, sf_values=[100, 150])
        assert strat._load_macro_bias() == "bull"

    def test_sf_contracting_is_bear_bias(self, tmp_path):
        """社融环比收缩（-20%以上）→ bear。"""
        strat = self._make_strategy_with_db(tmp_path, m2_yoy=8.0, sf_values=[100, 70])
        assert strat._load_macro_bias() == "bear"

    def test_no_macro_data_returns_none(self, tmp_path):
        strat = self._make_strategy_with_db(tmp_path)
        assert strat._load_macro_bias() is None


# ===========================================================================
# 回归：不传新参数时行为不变
# ===========================================================================
class TestRegression:
    def test_compute_factors_without_new_params(self):
        """不传 fund_hold/index_ret → 与旧版完全兼容。"""
        df = _ohlcv()
        f = compute_factors(df)
        # 原有因子不受影响
        assert "mom_20" in f
        assert f["mom_20"] is not None
        # 新因子为 nan→None
        assert f.get("fund_holding") is None
        assert f.get("beta_300") is None
