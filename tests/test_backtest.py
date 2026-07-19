"""回测核心纯函数数值正确性测试。

覆盖 signal_score / fear_greed_score（盘面评分加权）、_compute_ic（Rank IC）、
_compute_quantiles（分档单调性）、_ic_assessment（有效性评级）。
这些是策略质量定级与决策中枢的数值基石，回归测试防止公式被无意改坏。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sequoia_x.analysis.backtest import (
    SignalBacktester,
    _ic_assessment,
    fear_greed_score,
    signal_score,
)


# ---------------------------------------------------------------------------
# signal_score：0-100 加权（宽度45% + 涨跌幅30% + 涨跌停比25%）
# ---------------------------------------------------------------------------
def test_signal_score_all_bullish_is_100():
    """全涨盘面应得满分 100。"""
    assert signal_score(up_ratio=100, avg_change=5, limit_up=10, limit_down=0) == 100


def test_signal_score_all_bearish_is_0():
    """全跌盘面应得 0 分。"""
    assert signal_score(up_ratio=0, avg_change=-5, limit_up=0, limit_down=10) == 0


def test_signal_score_balanced_is_50():
    """平衡盘面（各因子均取中值）应得 50 分。"""
    assert signal_score(up_ratio=50, avg_change=0, limit_up=5, limit_down=5) == 50


def test_signal_score_clamps_avg_change():
    """avg_change 超出 [-5, +5] 区间被钳制，不影响 change_score 边界。"""
    # avg_change=20 远超上界，change_score 钳到 100
    s_high = signal_score(up_ratio=50, avg_change=20, limit_up=5, limit_down=5)
    # avg_change=-20 远低于下界，change_score 钳到 0
    s_low = signal_score(up_ratio=50, avg_change=-20, limit_up=5, limit_down=5)
    assert s_high > s_low


def test_signal_score_zero_limit_uses_neutral_50():
    """无涨跌停数据（lim=0）时 limit_score 取中性 50，避免除零。"""
    s = signal_score(up_ratio=50, avg_change=0, limit_up=0, limit_down=0)
    # breadth=50, change=50, limit=50 → 50
    assert s == 50


# ---------------------------------------------------------------------------
# fear_greed_score：五因子加权（0-100）
# ---------------------------------------------------------------------------
def test_fear_greed_extreme_greed_high_score():
    """极端贪婪场景得分高（接近满分）。"""
    score = fear_greed_score(
        limit_up=10, limit_down=0, up_ratio=100,
        nh=10, nl=0, decided=10, ma20_pct=80, ma60_pct=80,
    )
    assert score == 93  # 0.2*100+0.25*100+0.2*100+0.2*80+0.15*80


def test_fear_greed_extreme_fear_low_score():
    """极端恐惧场景得分低。"""
    score = fear_greed_score(
        limit_up=0, limit_down=10, up_ratio=0,
        nh=0, nl=10, decided=10, ma20_pct=20, ma60_pct=20,
    )
    assert score == 7  # 0.2*0+0.25*0+0.2*0+0.2*20+0.15*20


def test_fear_greed_clamps_nhnl_component():
    """nh-nl 分量超出 [-10, +10] 被钳到 [0, 100]。"""
    # nh=100, nl=0, decided=100 → nhnl_pct=100 → s3 钳到 100
    score = fear_greed_score(
        limit_up=5, limit_down=5, up_ratio=50,
        nh=100, nl=0, decided=100, ma20_pct=50, ma60_pct=50,
    )
    assert 0 <= score <= 100


def test_fear_greed_zero_decided_no_division_error():
    """decided=0 时不除零（nhnl_pct 取 0）。"""
    score = fear_greed_score(
        limit_up=5, limit_down=5, up_ratio=50,
        nh=0, nl=0, decided=0, ma20_pct=50, ma60_pct=50,
    )
    # decided=0 不除零；各因子均取 50 → 得分恰好 50
    assert score == 50


# ---------------------------------------------------------------------------
# _compute_ic：Rank IC（Spearman）正确性
# ---------------------------------------------------------------------------
def _make_daily(factor: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    """构造 index 为日期字符串（跨两月）的 daily DataFrame，供 IC 月度聚合。"""
    dates = [f"2024-01-{i:02d}" for i in range(1, 16)] + [f"2024-02-{i:02d}" for i in range(1, 16)]
    return pd.DataFrame({"f": factor, "ret": target}, index=dates[: len(factor)])


def test_compute_ic_perfect_positive_is_one():
    """因子与收益完美正相关 → Rank IC ≈ 1.0。"""
    x = np.arange(30, dtype=float)
    daily = _make_daily(x, x)
    ic = SignalBacktester._compute_ic(daily, "f", "ret")
    assert ic["ic_mean"] == 1.0
    assert ic["assessment"] in ("强有效因子", "有效因子", "弱有效，可优化权重")


def test_compute_ic_perfect_negative_is_minus_one():
    """因子与收益完美负相关 → Rank IC ≈ -1.0。"""
    x = np.arange(30, dtype=float)
    daily = _make_daily(x, -x)
    ic = SignalBacktester._compute_ic(daily, "f", "ret")
    assert ic["ic_mean"] == -1.0


def test_compute_ic_insufficient_samples_returns_error():
    """样本 < 30 时返回 error 标记，不计算 IC。"""
    daily = _make_daily(np.arange(10.0), np.arange(10.0))
    ic = SignalBacktester._compute_ic(daily, "f", "ret")
    assert ic == {"error": "样本不足"}


# ---------------------------------------------------------------------------
# _compute_quantiles：分档单调性
# ---------------------------------------------------------------------------
def test_compute_quantiles_monotonic_factor_monotonic_return():
    """因子单调且收益随因子单调递增 → 各档 next_avg_return 单调递增。"""
    n = 50
    factor = np.arange(n, dtype=float)
    daily = pd.DataFrame(
        {"f": factor, "next_avg_chg": factor * 0.1},
        index=[f"2024-01-{(i % 28) + 1:02d}" for i in range(n)],
    )
    quantiles = SignalBacktester._compute_quantiles(daily, "f", n_bins=5)
    assert len(quantiles) > 0
    returns = [q["next_avg_return"] for q in quantiles]
    assert returns == sorted(returns), f"分档收益应单调递增，实际 {returns}"


def test_compute_quantiles_insufficient_samples_returns_empty():
    """样本不足 n_bins*5 时返回空列表。"""
    daily = pd.DataFrame(
        {"f": np.arange(10.0), "next_avg_chg": np.arange(10.0)},
        index=[f"2024-01-{i + 1:02d}" for i in range(10)],
    )
    assert SignalBacktester._compute_quantiles(daily, "f", n_bins=5) == []


# ---------------------------------------------------------------------------
# _ic_assessment：有效性评级阈值
# ---------------------------------------------------------------------------
def test_ic_assessment_strong():
    assert _ic_assessment(ic_mean=0.1, icir=1.2, win_rate=75) == "强有效因子"


def test_ic_assessment_effective():
    assert _ic_assessment(ic_mean=0.05, icir=0.6, win_rate=60) == "有效因子"


def test_ic_assessment_weak():
    assert _ic_assessment(ic_mean=0.04, icir=0.3, win_rate=52) == "弱有效，可优化权重"


def test_ic_assessment_useless():
    assert _ic_assessment(ic_mean=0.01, icir=0.1, win_rate=40) == "预测力不足"
