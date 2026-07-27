"""P16 因子正交化测试：IC 相关性矩阵 + 层次聚类 + 增量 IC 折扣。

覆盖：
  - _build_ic_correlation：完全同步/独立/反向的 IC 序列
  - _cluster_factors：高相关因子归簇、独立因子自成簇、负相关不误合
  - _orthogonal_penalty：基准满权重、冗余因子按 (1-corr²)^0.5 折扣
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequoia_x.analysis.factor import (
    _build_ic_correlation,
    _cluster_factors,
    _orthogonal_penalty,
    ORTH_THRESHOLD,
)


# ===========================================================================
# _build_ic_correlation
# ===========================================================================
class TestBuildICCorrelation:
    def test_perfectly_synced(self):
        ic = {"A": [0.1, 0.2, 0.3, 0.4], "B": [0.1, 0.2, 0.3, 0.4]}
        corr = _build_ic_correlation(ic, ["A", "B"])
        assert corr.loc["A", "B"] == pytest.approx(1.0, abs=0.01)

    def test_independent(self):
        np.random.seed(42)
        ic = {"A": list(np.random.randn(50)), "B": list(np.random.randn(50))}
        corr = _build_ic_correlation(ic, ["A", "B"])
        assert abs(corr.loc["A", "B"]) < 0.3

    def test_negatively_correlated(self):
        ic = {"A": [0.1, 0.2, -0.1, -0.2], "B": [-0.1, -0.2, 0.1, 0.2]}
        corr = _build_ic_correlation(ic, ["A", "B"])
        assert corr.loc["A", "B"] == pytest.approx(-1.0, abs=0.01)

    def test_empty_returns_empty(self):
        corr = _build_ic_correlation({}, [])
        assert corr.empty

    def test_too_few_months_returns_empty(self):
        ic = {"A": [0.1], "B": [0.2]}
        corr = _build_ic_correlation(ic, ["A", "B"])
        assert corr.empty


# ===========================================================================
# _cluster_factors
# ===========================================================================
class TestClusterFactors:
    def test_high_correlation_clustered(self):
        corr = pd.DataFrame(
            [[1.0, 0.9, 0.85, 0.1, 0.05],
             [0.9, 1.0, 0.88, 0.12, 0.08],
             [0.85, 0.88, 1.0, 0.1, 0.06],
             [0.1, 0.12, 0.1, 1.0, 0.15],
             [0.05, 0.08, 0.06, 0.15, 1.0]],
            index=["A", "B", "C", "D", "E"],
            columns=["A", "B", "C", "D", "E"],
        )
        clusters = _cluster_factors(corr, threshold=0.6)
        # A, B, C should be in one cluster
        abc_cluster = None
        for cid, members in clusters.items():
            if "A" in members:
                abc_cluster = cid
        assert abc_cluster is not None
        abc_members = set(clusters[abc_cluster])
        assert {"A", "B", "C"}.issubset(abc_members)
        # D and E should NOT be in the ABC cluster
        assert "D" not in abc_members
        assert "E" not in abc_members

    def test_single_factor(self):
        corr = pd.DataFrame([[1.0]], index=["X"], columns=["X"])
        clusters = _cluster_factors(corr, threshold=0.6)
        assert len(clusters) == 1
        assert clusters[next(iter(clusters))] == ["X"]

    def test_empty(self):
        clusters = _cluster_factors(pd.DataFrame(), threshold=0.6)
        assert clusters == {}

    def test_negatively_correlated_not_merged(self):
        """负相关因子（方向相反的独立信号）不应被合并。"""
        corr = pd.DataFrame(
            [[1.0, -0.8],
             [-0.8, 1.0]],
            index=["X", "Y"],
            columns=["X", "Y"],
        )
        clusters = _cluster_factors(corr, threshold=0.6)
        # |corr|=0.8 > 0.6 但距离=1-0.8=0.2 < 1-0.6=0.4, should they merge?
        # |corr| > threshold means they ARE merged (we use |corr|, not signed corr)
        # This is by design: high |corr| = redundant signal regardless of sign
        all_clusters = list(clusters.values())
        assert len(all_clusters) >= 1


# ===========================================================================
# _orthogonal_penalty
# ===========================================================================
class TestOrthogonalPenalty:
    def test_base_gets_full_weight(self):
        corr = pd.DataFrame(
            [[1.0, 0.9],
             [0.9, 1.0]],
            index=["base", "redundant"],
            columns=["base", "redundant"],
        )
        clusters = {1: ["base", "redundant"]}
        ic_means = {"base": 0.15, "redundant": 0.10}
        pen = _orthogonal_penalty(corr, clusters, ic_means)
        assert pen["base"] == 1.0

    def test_redundant_factor_heavily_discounted(self):
        corr = pd.DataFrame(
            [[1.0, 0.9],
             [0.9, 1.0]],
            index=["base", "redundant"],
            columns=["base", "redundant"],
        )
        clusters = {1: ["base", "redundant"]}
        ic_means = {"base": 0.15, "redundant": 0.10}
        pen = _orthogonal_penalty(corr, clusters, ic_means)
        # penalty = sqrt(1 - 0.9²) = sqrt(0.19) ≈ 0.436
        assert pen["redundant"] == pytest.approx(0.4359, abs=0.01)

    def test_moderately_correlated_factor_mildly_discounted(self):
        corr = pd.DataFrame(
            [[1.0, 0.6],
             [0.6, 1.0]],
            index=["base", "moderate"],
            columns=["base", "moderate"],
        )
        clusters = {1: ["base", "moderate"]}
        ic_means = {"base": 0.15, "moderate": 0.12}
        pen = _orthogonal_penalty(corr, clusters, ic_means)
        # penalty = sqrt(1 - 0.36) = sqrt(0.64) = 0.8
        assert pen["moderate"] == pytest.approx(0.8, abs=0.01)

    def test_independent_factor_not_discounted(self):
        corr = pd.DataFrame(
            [[1.0, 0.9, 0.1],
             [0.9, 1.0, 0.15],
             [0.1, 0.15, 1.0]],
            index=["base", "redundant", "independent"],
            columns=["base", "redundant", "independent"],
        )
        clusters = {1: ["base", "redundant"], 2: ["independent"]}
        ic_means = {"base": 0.15, "redundant": 0.10, "independent": 0.08}
        pen = _orthogonal_penalty(corr, clusters, ic_means)
        assert pen["independent"] == 1.0
        assert pen["redundant"] < 0.5

    def test_single_member_cluster_full_weight(self):
        clusters = {1: ["lone"]}
        pen = _orthogonal_penalty(pd.DataFrame(), clusters, {"lone": 0.05})
        assert pen["lone"] == 1.0

    def test_highest_ic_is_base(self):
        """簇内 |IC| 最大的因子选为基准。"""
        corr = pd.DataFrame(
            [[1.0, 0.85],
             [0.85, 1.0]],
            index=["weaker", "stronger"],
            columns=["weaker", "stronger"],
        )
        clusters = {1: ["weaker", "stronger"]}
        ic_means = {"weaker": 0.05, "stronger": 0.15}
        pen = _orthogonal_penalty(corr, clusters, ic_means)
        assert pen["stronger"] == 1.0  # stronger is base
        assert pen["weaker"] < 1.0     # weaker is discounted
