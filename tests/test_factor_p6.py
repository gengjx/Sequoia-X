"""P6 因子拥挤度衰减测试：行业集中度 HHI → 软连续权重衰减。

覆盖：
  - _industry_hhi：行业集中度计算（全集中→1.0、分散→低值、精确分布）
  - _crowding_penalty：软连续衰减（SAFE 以下不衰减、MAX 以上保留下限、线性中点）
  - _compute_crowding：多头侧方向（IC≥0→top、IC<0→bottom）+ 多月均值
  - 衰减穿透归一化：拥挤因子相对权重压缩、分散因子不变
  - 持久化：factor_weights.crowding 落库 + load 读回
"""

from __future__ import annotations

import sqlite3

import pytest

from sequoia_x.analysis.factor import (
    CROWDING_FLOOR,
    CROWDING_MAX,
    CROWDING_SAFE,
    _compute_crowding,
    _crowding_penalty,
    _industry_hhi,
)
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine


# ===========================================================================
# _industry_hhi：行业集中度计算
# ===========================================================================
class TestIndustryHhi:
    def test_all_same_industry_is_one(self):
        """全部同一行业 → HHI=1.0（完全集中）。"""
        ind = {s: "银行" for s in ["A", "B", "C", "D"]}
        assert _industry_hhi(["A", "B", "C", "D"], ind) == pytest.approx(1.0)

    def test_completely_dispersed_is_low(self):
        """每只票不同行业 → HHI=Σ(1/n)²=n*(1/n)²=1/n（极度分散）。"""
        ind = {"A": "I1", "B": "I2", "C": "I3", "D": "I4"}
        hhi = _industry_hhi(["A", "B", "C", "D"], ind)
        assert hhi == pytest.approx(0.25)  # 1/4

    def test_known_distribution(self):
        """已知分布：4银行+2科技+2消费 → HHI=(0.5)²+(0.25)²+(0.25)²=0.375。"""
        ind = {"A": "银行", "B": "银行", "C": "银行", "D": "银行",
               "E": "科技", "F": "科技", "G": "消费", "H": "消费"}
        hhi = _industry_hhi(["A", "B", "C", "D", "E", "F", "G", "H"], ind)
        assert hhi == pytest.approx(0.375)

    def test_partial_coverage(self):
        """部分股票无行业数据 → 只统计有数据的。"""
        ind = {"A": "银行", "B": "银行"}
        # C 无行业 → 只算 A,B（同行业 → 1.0）
        assert _industry_hhi(["A", "B", "C"], ind) == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        """空列表 → 0（不拥挤，不惩罚）。"""
        assert _industry_hhi([], {"A": "银行"}) == 0.0

    def test_no_industry_map_returns_zero(self):
        assert _industry_hhi(["A", "B"], {}) == 0.0

    def test_all_unknown_industry_returns_zero(self):
        """所有股票都不在 industry_map → 0。"""
        assert _industry_hhi(["X", "Y"], {"A": "银行"}) == 0.0


# ===========================================================================
# _crowding_penalty：软连续衰减
# ===========================================================================
class TestCrowdingPenalty:
    def test_below_safe_no_decay(self):
        """HHI ≤ SAFE(0.08) → 不衰减（乘子=1.0）。"""
        assert _crowding_penalty(0.0) == 1.0
        assert _crowding_penalty(CROWDING_SAFE) == 1.0
        assert _crowding_penalty(0.05) == 1.0

    def test_above_max_hits_floor(self):
        """HHI ≥ MAX(0.30) → 衰减到 FLOOR(0.30)。"""
        assert _crowding_penalty(CROWDING_MAX) == pytest.approx(CROWDING_FLOOR)
        assert _crowding_penalty(0.50) == pytest.approx(CROWDING_FLOOR)
        assert _crowding_penalty(1.0) == pytest.approx(CROWDING_FLOOR)

    def test_linear_midpoint(self):
        """HHI=0.19（SAFE~MAX 中点）→ 0.65。"""
        # frac = (0.19-0.08)/(0.30-0.08) = 0.5 → 1.0-(1-0.3)*0.5 = 0.65
        assert _crowding_penalty(0.19) == pytest.approx(0.65)

    def test_monotonic_decreasing(self):
        """拥挤度越高，衰减乘子越小（单调递减）。"""
        vals = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        penalties = [_crowding_penalty(v) for v in vals]
        for i in range(len(penalties) - 1):
            assert penalties[i] >= penalties[i + 1]

    def test_floor_not_broken(self):
        """任何 crowding 值都不会让乘子低于 FLOOR。"""
        for c in [-0.1, 0.0, 0.5, 1.0, 5.0]:
            assert _crowding_penalty(c) >= CROWDING_FLOOR - 1e-9

    def test_range_in_unit(self):
        """乘子恒在 [FLOOR, 1.0]。"""
        for c in [0.0, 0.01, 0.08, 0.10, 0.19, 0.30, 0.99]:
            p = _crowding_penalty(c)
            assert CROWDING_FLOOR - 1e-9 <= p <= 1.0 + 1e-9


# ===========================================================================
# _compute_crowding：多头侧方向 + 多月均值
# ===========================================================================
class TestComputeCrowding:
    @staticmethod
    def _batch(factor_vals, symbols, month):
        """构造一个月的 records batch。"""
        return [
            {"symbol": s, "mom": fv, "rev": -fv, "fwd_return": 0.01}
            for s, fv in zip(symbols, factor_vals)
        ]

    @staticmethod
    def _industry_map(symbols):
        """前半分行业 A，后半行业 B（可控制集中度）。"""
        n = len(symbols)
        return {s: "A" if i < n // 2 else "B" for i, s in enumerate(symbols)}

    def test_positive_ic_takes_top(self):
        """IC>0 的因子取 top20% 多头侧。"""
        syms = [f"S{i:03d}" for i in range(100)]
        # top20%（mom 最高）全在前 20 只，若它们全在行业 A → HHI 高
        fv = list(range(100))  # mom=S000=0 ... S099=99
        ind = {s: "A" if i < 20 else "B" for i, s in enumerate(syms)}
        records = {"2025-01": self._batch(fv, syms, "2025-01")}
        crowding = _compute_crowding(records, ["2025-01"], {"mom": 0.05}, ind)
        # top20 = S080~S099，全在行业 B → HHI=1.0
        assert "mom" in crowding
        assert crowding["mom"] == pytest.approx(1.0)

    def test_negative_ic_takes_bottom(self):
        """IC<0 的因子取 bottom20%（低值侧）——修正旧 nlargest 取错侧。"""
        syms = [f"S{i:03d}" for i in range(100)]
        fv = list(range(100))
        ind = {s: "A" if i < 20 else "B" for i, s in enumerate(syms)}
        records = {"2025-01": self._batch(fv, syms, "2025-01")}
        # rev = -fv → rev 最低的是 S099(rev=-99)
        # IC<0 → bottom20% = rev 最低 = S080~S099 → 全在行业 B → HHI=1.0
        crowding = _compute_crowding(records, ["2025-01"], {"rev": -0.05}, ind)
        assert "rev" in crowding
        assert crowding["rev"] == pytest.approx(1.0)

    def test_multi_month_average(self):
        """多个月 HHI 取均值降噪。"""
        syms = [f"S{i:03d}" for i in range(100)]
        fv = list(range(100))
        # 月1：top20 全行业 A（HHI=1.0）；月2：top20 全行业 B（HHI=1.0）
        ind1 = {s: "A" if i < 80 else "B" for i, s in enumerate(syms)}  # top=S080~099→B
        ind2 = {s: "A" if i >= 80 else "B" for i, s in enumerate(syms)}  # top→A
        r1 = self._batch(fv, syms, "2025-01")
        r2 = self._batch(fv, syms, "2025-02")
        # 用统一的 industry_map 无法逐月不同，故用单行业测试均值语义
        ind = {s: "BANK" for s in syms}  # 全同 → 每月 HHI=1.0
        records = {"2025-01": r1, "2025-02": r2}
        crowding = _compute_crowding(records, ["2025-01", "2025-02"], {"mom": 0.05}, ind)
        assert crowding["mom"] == pytest.approx(1.0)  # 两月均值=1.0

    def test_insufficient_data_skipped(self):
        """batch < 50 → 跳过，不在结果中。"""
        syms = [f"S{i:03d}" for i in range(30)]
        records = {"2025-01": self._batch(list(range(30)), syms, "2025-01")}
        crowding = _compute_crowding(records, ["2025-01"], {"mom": 0.05}, self._industry_map(syms))
        assert crowding == {}

    def test_empty_months(self):
        assert _compute_crowding({}, [], {"mom": 0.05}, {"A": "B"}) == {}

    def test_no_industry_map(self):
        syms = [f"S{i:03d}" for i in range(100)]
        records = {"2025-01": self._batch(list(range(100)), syms, "2025-01")}
        assert _compute_crowding(records, ["2025-01"], {"mom": 0.05}, {}) == {}


# ===========================================================================
# 衰减穿透归一化
# ===========================================================================
class TestDecayThroughNormalization:
    def test_crowded_factor_weight_reduced(self):
        """拥挤因子 A 的权重被压缩、分散因子 B 不变（乘子效应）。"""
        # A 拥挤 HHI=0.30（→penalty 0.30），B 分散 HHI=0.05（→penalty 1.0）
        pen_a = _crowding_penalty(0.30)
        pen_b = _crowding_penalty(0.05)
        # 假设 IC 相同，原始权重各 0.5
        raw_a, raw_b = 0.5, 0.5
        decayed_a = raw_a * pen_a
        decayed_b = raw_b * pen_b
        total = decayed_a + decayed_b
        # 归一化后相对权重
        norm_a = decayed_a / total
        norm_b = decayed_b / total
        # A 的相对权重应 < B
        assert norm_a < norm_b
        # B 应占大头
        assert norm_b > 0.5

    def test_no_penalty_equal_weights(self):
        """都不拥挤 → penalty=1.0 → 相对权重不变。"""
        pen_a = _crowding_penalty(0.03)
        pen_b = _crowding_penalty(0.04)
        raw_a, raw_b = 0.5, 0.5
        total = raw_a * pen_a + raw_b * pen_b
        assert (raw_a * pen_a) / total == pytest.approx(0.5)


# ===========================================================================
# 持久化：factor_weights.crowding 落库 + 读回
# ===========================================================================
class TestPersistence:
    def test_crowding_saved_and_loaded(self, tmp_path):
        """save_factor_weights 写入 crowding，load_factor_weights 读回。"""
        db = str(tmp_path / "p6.db")
        eng = DataEngine(Settings(db_path=db))
        eng.save_factor_weights([
            {"factor_name": "mom_5", "category": "动量", "ic_mean": 0.05,
             "icir": 0.8, "win_rate": 60, "weight": 0.4, "crowding": 0.25},
            {"factor_name": "rev_5", "category": "动量", "ic_mean": -0.04,
             "icir": -0.7, "win_rate": 45, "weight": -0.3, "crowding": 0.10},
        ])
        loaded = eng.load_factor_weights()
        assert loaded["mom_5"]["crowding"] == pytest.approx(0.25)
        assert loaded["rev_5"]["crowding"] == pytest.approx(0.10)

    def test_crowding_defaults_zero_for_old_data(self, tmp_path):
        """无 crowding 字段的旧权重 → 默认 0（不惩罚）。"""
        db = str(tmp_path / "p6_old.db")
        eng = DataEngine(Settings(db_path=db))
        eng.save_factor_weights([
            {"factor_name": "mom_5", "category": "动量", "ic_mean": 0.05,
             "icir": 0.8, "win_rate": 60, "weight": 0.4},  # 无 crowding 键
        ])
        loaded = eng.load_factor_weights()
        assert loaded["mom_5"]["crowding"] == 0

    def test_migration_idempotent(self, tmp_path):
        """重复初始化不重复加列。"""
        db = str(tmp_path / "p6_mig.db")
        DataEngine(Settings(db_path=db))
        DataEngine(Settings(db_path=db))  # 再次初始化
        cols = [
            r[1] for r in
            sqlite3.connect(db).execute("PRAGMA table_info(factor_weights)").fetchall()
        ]
        assert cols.count("crowding") == 1
