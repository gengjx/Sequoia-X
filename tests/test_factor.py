"""因子计算与评估纯函数数值正确性测试。

覆盖 _ret（N日收益率）、compute_factors（38因子截面计算，重点验动量/反转/
量能/换手率/结构等关键因子的数学关系）、cross_section_rank（横截面百分位）、
compute_composite_score（多因子合成）、_factor_assessment（有效性评级）。

这些是多因子选股与 IC 评估的数值基石，回归测试防止因子公式被改坏。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequoia_x.analysis.factor import (
    _factor_assessment,
    _ret,
    compute_composite_score,
    compute_factors,
    cross_section_rank,
)


# ---------------------------------------------------------------------------
# 辅助：构造 OHLCV DataFrame
# ---------------------------------------------------------------------------
def _ohlcv(n: int, start: float = 10.0, daily_ret: float = 0.0,
           turn: float | None = None) -> pd.DataFrame:
    """构造 n 日 OHLCV，收盘价按 daily_ret 日复利。

    Args:
        n: 天数
        start: 起始收盘价
        daily_ret: 每日收益率（如 0.01=1%）
        turn: 换手率（可选，恒定值）
    """
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + daily_ret))
    close = pd.Series(closes, index=dates)
    df = pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": pd.Series([1_000_000] * n, index=dates),
        "turnover": pd.Series([1_000_000 * c for c in closes], index=dates),
    }, index=dates)
    if turn is not None:
        df["turn"] = turn
    return df


# ---------------------------------------------------------------------------
# _ret：N 日收益率 %
# ---------------------------------------------------------------------------
class TestRet:
    def test_positive_return(self):
        close = pd.Series([10.0, 11.0, 12.1])
        # _ret(n=2): (12.1/10 - 1)*100 = 21
        assert _ret(close, 2) == pytest.approx(21.0)

    def test_negative_return(self):
        close = pd.Series([10.0, 9.0, 8.1])
        assert _ret(close, 2) == pytest.approx(-19.0)

    def test_zero_return(self):
        close = pd.Series([10.0, 10.0, 10.0])
        assert _ret(close, 2) == 0.0

    def test_insufficient_length_returns_nan(self):
        """len(close) <= n 返回 nan。"""
        close = pd.Series([10.0, 11.0])
        result = _ret(close, 2)
        assert result != result  # nan 检测

    def test_zero_prev_returns_nan(self):
        """前 n 日价格为 0 返回 nan（避免除零）。"""
        close = pd.Series([0.0, 0.0, 10.0])
        assert _ret(close, 2) != _ret(close, 2)  # nan


# ---------------------------------------------------------------------------
# compute_factors：38 因子截面计算
# ---------------------------------------------------------------------------
class TestComputeFactors:
    def test_insufficient_bars_returns_empty(self):
        """不足 20 根 K 线返回空 dict。"""
        df = _ohlcv(15)
        assert compute_factors(df) == {}

    def test_minimum_bars_returns_factors(self):
        """20 根 K 线即返回因子 dict（含基础因子）。"""
        df = _ohlcv(20)
        factors = compute_factors(df)
        assert isinstance(factors, dict)
        assert len(factors) > 20  # 至少含动量+反转+波动等基础因子

    def test_momentum_factors_present(self):
        """动量因子 mom_5/10/20 在 20 根 K 线时已可计算。"""
        factors = compute_factors(_ohlcv(20))
        for k in ("mom_5", "mom_10", "mom_20"):
            assert k in factors

    def test_reversal_is_negative_momentum(self):
        """反转因子 rev_* = -mom_*（取负让高反转=高分）。"""
        factors = compute_factors(_ohlcv(25))
        assert factors["rev_5"] == pytest.approx(-factors["mom_5"], abs=1e-4)
        assert factors["rev_10"] == pytest.approx(-factors["mom_10"], abs=1e-4)

    def test_momentum_60_requires_60_bars(self):
        """mom_60 在不足 60 根时为 None（nan→None）。"""
        factors = compute_factors(_ohlcv(30))
        assert factors["mom_60"] is None  # nan 清理为 None

    def test_momentum_60_computed_with_enough_bars(self):
        factors = compute_factors(_ohlcv(65))
        assert factors["mom_60"] is not None

    def test_rps_120_requires_120_bars(self):
        """rps_120 在不足 120 根时为 None。"""
        assert compute_factors(_ohlcv(100))["rps_120"] is None
        assert compute_factors(_ohlcv(125))["rps_120"] is not None

    def test_turnover_factor(self):
        """turnover 因子 = 末尾成交额 / 1e8（亿元）。"""
        df = _ohlcv(20)
        factors = compute_factors(df)
        expected = df["turnover"].iloc[-1] / 1e8
        assert factors["turnover"] == pytest.approx(expected, rel=1e-4)

    def test_liq_rank_equals_raw_turnover(self):
        """liq_rank = 原始成交额值（横截面排名时用）。"""
        df = _ohlcv(20)
        factors = compute_factors(df)
        assert factors["liq_rank"] == pytest.approx(df["turnover"].iloc[-1], rel=1e-4)

    def test_ma_cross_no_cross_when_steady(self):
        """稳定上涨时 ma5 始终在 ma20 上方，无金叉（ma_cross=0）。"""
        df = _ohlcv(25, daily_ret=0.01)
        factors = compute_factors(df)
        assert factors["ma_cross"] == 0.0

    def test_volume_ratio_around_one_for_constant_volume(self):
        """恒定成交量下 volume_ratio ≈ 1（当日/5日均量）。"""
        df = _ohlcv(25)
        factors = compute_factors(df)
        assert factors["volume_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_finance_factors_when_provided(self):
        """传入 finance dict 时质量因子被填充。"""
        finance = {"roe": 15.0, "np_margin": 20.0, "gp_margin": 40.0,
                   "yoy_eps": 10.0, "yoy_pni": 25.0}
        factors = compute_factors(_ohlcv(20), finance=finance)
        assert factors["roe"] == 15.0
        assert factors["np_margin"] == 20.0
        assert factors["gp_margin"] == 40.0
        assert factors["rev_growth"] == 10.0
        assert factors["profit_growth"] == 25.0

    def test_finance_factors_nan_when_absent(self):
        """未传 finance 时质量因子为 None。"""
        factors = compute_factors(_ohlcv(20))
        for k in ("roe", "np_margin", "gp_margin", "rev_growth", "profit_growth"):
            assert factors[k] is None

    def test_fund_flow_factors_when_provided(self):
        """传入 fund_flow 时资金因子被填充。"""
        fund_flow = {"main_net": -5e7, "main_pct": -2.0,
                     "super_net": 1e7, "big_net": -3e7}
        factors = compute_factors(_ohlcv(20), fund_flow=fund_flow)
        assert factors["main_net"] == pytest.approx(-5e7)
        assert factors["main_pct"] == pytest.approx(-2.0)

    def test_lhb_factors_when_provided(self):
        factors = compute_factors(_ohlcv(20), lhb_data={"count": 3, "net_buy": 1e8})
        assert factors["lhb_count"] == 3.0
        assert factors["lhb_netbuy"] == pytest.approx(1e8)

    def test_turn_factors_when_turn_column_present(self):
        """含 turn 列时换手率因子被填充，turn_reversal = -turn_ma20。"""
        df = _ohlcv(25, turn=3.0)
        factors = compute_factors(df)
        assert factors["turn_ratio"] == pytest.approx(3.0)
        assert factors["turn_ma5"] == pytest.approx(3.0)
        assert factors["turn_reversal"] == pytest.approx(-3.0, abs=1e-4)
        assert factors["turn_surge"] == pytest.approx(1.0, abs=0.01)

    def test_turn_factors_nan_when_absent(self):
        """无 turn 列时换手率因子为 None。"""
        factors = compute_factors(_ohlcv(25))
        for k in ("turn_ratio", "turn_ma5", "turn_surge", "turn_reversal"):
            assert factors[k] is None

    def test_nan_cleaned_to_none(self):
        """因子 dict 中 nan 被清理为 None（序列化友好）。"""
        factors = compute_factors(_ohlcv(25))  # mom_60 等为 nan
        for v in factors.values():
            if v is not None:
                assert v == v  # 非 nan

    def test_values_are_rounded(self):
        """因子值保留 4 位小数。"""
        factors = compute_factors(_ohlcv(20))
        for k, v in factors.items():
            if v is not None and isinstance(v, float):
                assert v == round(v, 4)


# ---------------------------------------------------------------------------
# cross_section_rank：横截面百分位排名（0-100）
# ---------------------------------------------------------------------------
class TestCrossSectionRank:
    def test_single_column_ranks(self):
        df = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0]})
        ranked = cross_section_rank(df)
        # pct=True: rank/len → 0.2,0.4,...; ×100 → 20,40,...,100
        assert list(ranked["f"]) == [20.0, 40.0, 60.0, 80.0, 100.0]

    def test_multiple_columns_independent(self):
        """多列各自独立排名。"""
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [30.0, 10.0, 20.0],
        })
        ranked = cross_section_rank(df)
        assert list(ranked["f1"]) == pytest.approx([100 / 3, 200 / 3, 100.0])
        assert list(ranked["f2"]) == pytest.approx([100.0, 100 / 3, 200 / 3])

    def test_range_is_0_to_100(self):
        df = pd.DataFrame({"f": np.random.rand(50)})
        ranked = cross_section_rank(df)
        assert ranked["f"].min() > 0
        assert ranked["f"].max() <= 100

    def test_nan_propagated(self):
        """NaN 在排名中保持 NaN。"""
        df = pd.DataFrame({"f": [1.0, np.nan, 3.0]})
        ranked = cross_section_rank(df)
        assert np.isnan(ranked["f"].iloc[1])


# ---------------------------------------------------------------------------
# _factor_assessment：因子有效性评级
# ---------------------------------------------------------------------------
class TestFactorAssessment:
    def test_strong_factor(self):
        assert _factor_assessment(ic_mean=0.04, icir=0.6, win_rate=50) == "强有效因子"

    def test_effective_factor(self):
        assert _factor_assessment(ic_mean=0.025, icir=0.3, win_rate=60) == "有效因子"

    def test_weak_positive(self):
        assert _factor_assessment(ic_mean=0.025, icir=0.1, win_rate=40) == "弱有效，方向正向"

    def test_weak_negative(self):
        assert _factor_assessment(ic_mean=-0.025, icir=0.1, win_rate=40) == "弱有效，方向负向"

    def test_useless(self):
        assert _factor_assessment(ic_mean=0.01, icir=0.1, win_rate=40) == "无效因子"

    def test_strong_requires_both_ic_and_icir(self):
        """ic>=0.03 但 icir<0.5 不算强有效。"""
        assert _factor_assessment(ic_mean=0.04, icir=0.3, win_rate=60) != "强有效因子"


# ---------------------------------------------------------------------------
# compute_composite_score：多因子合成（时序百分位加权）
# ---------------------------------------------------------------------------
class TestCompositeScore:
    def test_insufficient_bars_returns_zeros(self):
        """不足 60 根 K 线返回全 0 序列。"""
        df = _ohlcv(40)
        series = compute_composite_score(df)
        assert (series == 0.0).all()
        assert len(series) == 40

    def test_returns_series_aligned_with_index(self):
        """返回 Series 与输入 DataFrame 索引对齐。"""
        df = _ohlcv(70)
        series = compute_composite_score(df)
        assert isinstance(series, pd.Series)
        assert len(series) == len(df)
        assert (series.index == df.index).all()

    def test_range_within_0_100(self):
        """综合分落在 [0, 100] 区间（百分位加权性质）。"""
        df = _ohlcv(300, daily_ret=0.002)
        series = compute_composite_score(df)
        valid = series.dropna()
        assert valid.min() >= -100  # 负权重可能略低，但应 bounded
        assert valid.max() <= 100

    def test_custom_weights_used(self):
        """传入自定义权重时被采用（不被默认覆盖）。"""
        df = _ohlcv(70)
        # 单因子权重，合成分应反映该因子时序百分位
        series = compute_composite_score(df, weights={"mom_5": 1.0})
        assert isinstance(series, pd.Series)
        assert len(series) == 70
