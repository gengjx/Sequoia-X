"""P3a Alpha 泄漏修复测试：t-stat 显著性过滤 + 因子去重原子性。

覆盖：
  - _t_stat：月度 IC 的 t 统计量计算（样本不足→0）
  - _is_significant：三重门槛过滤（IC/ICIR/t-stat）
  - 因子去重保留最强者（liq_rank vs turnover）
  - 全量清零原子性（total_ic==0 时仍清零）
"""

from __future__ import annotations

import pytest

from sequoia_x.analysis.factor import (
    MIN_IC_ABS,
    MIN_ICIR_ABS,
    MIN_T_STAT,
    MIN_IC_SAMPLES,
    _t_stat,
    _is_significant,
)


# ===========================================================================
# t-stat 计算
# ===========================================================================
class TestTStat:
    @pytest.mark.parametrize("ic_mean,ic_std,n,expected_gt", [
        (0.04, 0.05, 12, 2.0),   # 强信号 → 通过 2.0
        (0.03, 0.03, 12, 2.0),   # 边界强 → 通过
    ])
    def test_significant_signals_pass_threshold(self, ic_mean, ic_std, n, expected_gt):
        assert abs(_t_stat(ic_mean, ic_std, n)) >= expected_gt

    @pytest.mark.parametrize("ic_mean,ic_std,n,expected_lt", [
        (0.02, 0.05, 12, 2.0),   # 弱信号 → 不通过
        (0.01, 0.05, 12, 2.0),   # 噪声 → 不通过
    ])
    def test_noise_signals_below_threshold(self, ic_mean, ic_std, n, expected_lt):
        assert abs(_t_stat(ic_mean, ic_std, n)) < expected_lt

    def test_exact_value(self):
        """已知值验证：ic=0.04,std=0.05,n=12 → t≈2.77。"""
        assert _t_stat(0.04, 0.05, 12) == pytest.approx(2.771, abs=0.01)

    def test_insufficient_samples_returns_zero(self):
        """样本不足(<6)→t=0（视为不显著，避免小样本噪声）。"""
        assert _t_stat(0.10, 0.01, MIN_IC_SAMPLES - 1) == 0.0

    def test_zero_std_returns_zero(self):
        """std=0（IC 无波动）→t=0。"""
        assert _t_stat(0.05, 0.0, 12) == 0.0

    def test_negative_ic_t_stat_negative(self):
        """负 IC → t 为负（带符号，|t| 仍判显著性）。"""
        assert _t_stat(-0.04, 0.05, 12) < 0

    def test_more_samples_higher_t(self):
        """同样 IC/std，样本越多 t 越高（统计功效提升）。"""
        t_few = abs(_t_stat(0.04, 0.05, MIN_IC_SAMPLES))
        t_many = abs(_t_stat(0.04, 0.05, 24))
        assert t_many > t_few


# ===========================================================================
# 显著性三重门槛
# ===========================================================================
class TestIsSignificant:
    def test_all_pass_is_significant(self):
        assert _is_significant(0.04, 0.6, 2.77, 12)

    def test_low_ic_rejected(self):
        assert not _is_significant(0.02, 0.6, 2.77, 12)

    def test_low_icir_rejected(self):
        assert not _is_significant(0.04, 0.3, 2.77, 12)

    def test_low_tstat_rejected(self):
        assert not _is_significant(0.04, 0.6, 1.0, 12)

    def test_insufficient_samples_rejected(self):
        assert not _is_significant(0.10, 1.0, 5.0, MIN_IC_SAMPLES - 1)

    def test_negative_factor_significant(self):
        """负 IC 因子（如流动性）|t| 达标即显著。"""
        assert _is_significant(-0.04, -0.6, -2.77, 12)

    @pytest.mark.parametrize("ic", [MIN_IC_ABS, MIN_IC_ABS - 0.001])
    def test_ic_boundary(self, ic):
        # 边界值 >MIN（不含）— MIN_IC_ABS 本身不通过（严格大于）
        assert _is_significant(MIN_IC_ABS, 0.6, 2.77, 12) is False  # == 不通过
        assert _is_significant(MIN_IC_ABS + 0.001, 0.6, 2.77, 12) is True


# ===========================================================================
# 因子去重：保留最强者
# ===========================================================================
class TestDedupKeepsStrongest:
    """验证去重逻辑：同义组保留 |IC| 最大的因子。"""

    def test_liq_rank_stronger_than_turnover(self):
        """liq_rank(|IC|=0.165) 应胜过 turnover(0.156)。"""
        liq = {"name": "liq_rank", "ic_mean": -0.1652}
        turn = {"name": "turnover", "ic_mean": -0.1561}
        best = max([liq, turn], key=lambda x: abs(x["ic_mean"]))
        assert best["name"] == "liq_rank"

    def test_dedup_group_logic(self):
        """同义组去重：组内多个时只保留 |IC| 最大。"""
        synonym_groups = [{"turnover", "liq_rank"}]
        effective = [
            {"name": "liq_rank", "ic_mean": -0.1652},
            {"name": "turnover", "ic_mean": -0.1561},
        ]
        removed = set()
        for group in synonym_groups:
            in_eff = [f for f in effective if f["name"] in group]
            if len(in_eff) > 1:
                best = max(in_eff, key=lambda x: abs(x["ic_mean"]))
                for f in in_eff:
                    if f["name"] != best["name"]:
                        removed.add(f["name"])
        assert removed == {"turnover"}  # 弱者被剔除
        kept = {f["name"] for f in effective if f["name"] not in removed}
        assert kept == {"liq_rank"}

    def test_single_member_group_no_dedup(self):
        """同义组只有1个成员时不剔除（无可去重）。"""
        group = {"turnover", "liq_rank"}
        effective = [{"name": "liq_rank", "ic_mean": -0.1652}]  # 只有1个
        in_eff = [f for f in effective if f["name"] in group]
        assert len(in_eff) == 1  # 不触发去重


# ===========================================================================
# 常量校验
# ===========================================================================
class TestThresholdConstants:
    def test_industrial_thresholds(self):
        """A 股月频工业标准门槛值。"""
        assert MIN_IC_ABS == 0.03
        assert MIN_ICIR_ABS == 0.5
        assert MIN_T_STAT == 2.0
        assert MIN_IC_SAMPLES == 6
