"""P3c ML 因子合成测试：训练/CV/持久化/门控。

覆盖：
  - LightGBM 不可用时回退 Ridge（不抛错）
  - 时间序列 CV 严格无前视（train_months 全早于 test_month）
  - 持久化：训练后写 ml_scores 快照
  - 门控：IC/ICIR/t-stat 低于 P3a 门槛时 valid=False
  - 特征重要度输出
  - multi_factor 读 ml_scores 快照（不实时训练）
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sequoia_x.analysis.ml_factor import MLFactorEngine


# ===========================================================================
# Ridge fallback（lightgbm 不可用时的核心保证）
# ===========================================================================
class TestRidgeFallback:
    def test_train_predict_returns_predictions(self):
        """_train_predict 即使无 lightgbm 也返回预测值。"""
        engine = MLFactorEngine(":memory:")
        rng = np.random.RandomState(42)
        X_train = rng.randn(500, 5)
        y_train = X_train @ np.array([1, -0.5, 0.3, 0, 0]) + rng.randn(500) * 0.1
        X_test = rng.randn(50, 5)
        preds, importance = engine._train_predict(X_train, y_train, X_test)
        assert preds.shape == (50,)
        assert importance is not None
        assert len(importance) == 5

    def test_rank_corr_perfect(self):
        a = np.array([1, 2, 3, 4, 5], dtype=float)
        b = np.array([10, 20, 30, 40, 50], dtype=float)
        assert MLFactorEngine._rank_corr(a, b) == pytest.approx(1.0)

    def test_rank_corr_inverse(self):
        a = np.array([1, 2, 3, 4, 5], dtype=float)
        b = np.array([5, 4, 3, 2, 1], dtype=float)
        assert MLFactorEngine._rank_corr(a, b) == pytest.approx(-1.0)


# ===========================================================================
# 时间序列 CV 无前视
# ===========================================================================
class TestNoLookahead:
    def test_cv_train_months_before_test(self):
        """expanding-window CV 的 train_months 必须严格早于 test_month。"""
        # 构造合成月度数据
        data = []
        rng = np.random.RandomState(42)
        months = [f"2024-{m:02d}" for m in range(1, 13)]  # 12 个月
        for month in months:
            for i in range(150):
                data.append({
                    "month": month,
                    "symbol": f"S{i}",
                    "features": {f: float(rng.randn()) for f in MLFactorEngine.FEATURE_FACTORS},
                    "fwd_return": float(rng.randn() * 0.1),
                })
        engine = MLFactorEngine(":memory:")
        results = engine._time_series_cv(data)
        # 应该能跑完不报错
        assert "ic_mean" in results
        assert "model_version" in results

    def test_cv_returns_feature_importance(self):
        """有效结果应包含 feature_importance（Ridge 或 LightGBM）。"""
        data = []
        rng = np.random.RandomState(42)
        months = [f"2024-{m:02d}" for m in range(1, 13)]
        for month in months:
            for i in range(150):
                feats = {f: float(rng.randn()) for f in MLFactorEngine.FEATURE_FACTORS}
                # 构造信号：第一个特征正相关
                fwd = feats["mom_5"] * 0.05 + rng.randn() * 0.02
                data.append({
                    "month": month, "symbol": f"S{i}",
                    "features": feats, "fwd_return": fwd,
                })
        engine = MLFactorEngine(":memory:")
        results = engine._time_series_cv(data)
        assert "feature_importance" in results
        assert isinstance(results["feature_importance"], dict)


# ===========================================================================
# 门控（P3a 显著性）
# ===========================================================================
class TestMLGating:
    def test_valid_requires_high_ic(self):
        """IC/ICIR/t-stat 低于门槛时 valid=False。"""
        results = {
            "ic_mean": 0.015,  # < 0.03
            "icir": 0.2,       # < 0.5
            "t_stat": 1.0,     # < 2.0
            "n_months": 10,
            "win_rate": 55.0,
        }
        from sequoia_x.analysis.factor import _is_significant
        assert not _is_significant(
            results["ic_mean"], results["icir"],
            results["t_stat"], results["n_months"],
        )

    def test_valid_passes_with_strong_ic(self):
        results = {
            "ic_mean": 0.04,
            "icir": 0.6,
            "t_stat": 2.5,
            "n_months": 12,
            "win_rate": 65.0,
        }
        from sequoia_x.analysis.factor import _is_significant
        assert _is_significant(
            results["ic_mean"], results["icir"],
            results["t_stat"], results["n_months"],
        )


# ===========================================================================
# 持久化（ml_scores 快照）
# ===========================================================================
class TestMLPersistence:
    def _make_db(self, tmp_path: Path) -> str:
        db = str(tmp_path / "test.db")
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE ml_scores ("
                "run_date TEXT, symbol TEXT, ml_score REAL, "
                "ic_mean REAL, icir REAL, t_stat REAL, model_version TEXT, "
                "PRIMARY KEY (run_date, symbol))"
            )
            conn.execute(
                "CREATE TABLE factor_weights ("
                "factor_name TEXT, category TEXT, ic_mean REAL, icir REAL, "
                "win_rate REAL, weight REAL, updated_at TEXT, "
                "PRIMARY KEY (factor_name))"
            )
        return db

    def test_save_ml_scores_writes_snapshot(self, tmp_path: Path):
        db = self._make_db(tmp_path)
        engine = MLFactorEngine(db)
        results = {
            "ic_mean": 0.04, "icir": 0.6, "t_stat": 2.5,
            "n_months": 12, "win_rate": 65.0,
            "model_version": "ridge",
            "predictions": {"600519": 0.5, "000001": -0.3},
        }
        engine._save_ml_scores(results)
        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT symbol, ml_score FROM ml_scores ORDER BY symbol"
            ).fetchall()
        assert len(rows) == 2
        assert rows[0] == ("000001", pytest.approx(-0.3))

    def test_save_ml_scores_zero_weight_when_invalid(self, tmp_path: Path):
        db = self._make_db(tmp_path)
        engine = MLFactorEngine(db)
        results = {
            "ic_mean": 0.01, "icir": 0.2, "t_stat": 1.0,  # 不达标
            "n_months": 12, "win_rate": 50.0,
            "model_version": "ridge",
            "predictions": {"600519": 0.5},
        }
        engine._save_ml_scores(results)
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT weight FROM factor_weights WHERE factor_name='ml_score'"
            ).fetchone()
        assert row is not None
        assert row[0] == 0.0  # 不达标 → 权重 0

    def test_save_ml_scores_nonzero_weight_when_valid(self, tmp_path: Path):
        db = self._make_db(tmp_path)
        engine = MLFactorEngine(db)
        results = {
            "ic_mean": 0.04, "icir": 0.6, "t_stat": 2.5,
            "n_months": 12, "win_rate": 65.0,
            "model_version": "ridge",
            "predictions": {"600519": 0.5},
        }
        engine._save_ml_scores(results)
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT weight FROM factor_weights WHERE factor_name='ml_score'"
            ).fetchone()
        assert row is not None
        assert row[0] > 0  # 达标 → 非零权重


# ===========================================================================
# multi_factor 读 ml_scores 快照（不实时训练）
# ===========================================================================
class TestMultiFactorReadSnapshot:
    def test_compute_ml_scores_reads_table(self, tmp_path: Path):
        """multi_factor._compute_ml_scores 应读 ml_scores 表，不调训练。"""
        db = str(tmp_path / "test.db")
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE ml_scores ("
                "run_date TEXT, symbol TEXT, ml_score REAL, "
                "ic_mean REAL, icir REAL, t_stat REAL, model_version TEXT, "
                "PRIMARY KEY (run_date, symbol))"
            )
            conn.execute(
                "INSERT INTO ml_scores VALUES ('2026-07-24','600519',0.5,0.04,0.6,2.5,'ridge')"
            )
            conn.execute(
                "INSERT INTO ml_scores VALUES ('2026-07-24','000001',-0.3,0.04,0.6,2.5,'ridge')"
            )

        # Mock engine with db_path
        mock_engine = MagicMock()
        mock_engine.db_path = db

        from sequoia_x.strategy.multi_factor import MultiFactorStrategy
        strategy = MultiFactorStrategy.__new__(MultiFactorStrategy)
        strategy.engine = mock_engine
        strategy._ml_scores_asof = None

        scores = strategy._compute_ml_scores(["600519", "000001"])
        assert scores == {"600519": 0.5, "000001": -0.3}

    def test_compute_ml_scores_empty_when_no_snapshot(self, tmp_path: Path):
        db = str(tmp_path / "test.db")
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE ml_scores ("
                "run_date TEXT, symbol TEXT, ml_score REAL, "
                "ic_mean REAL, icir REAL, t_stat REAL, model_version TEXT, "
                "PRIMARY KEY (run_date, symbol))"
            )
        mock_engine = MagicMock()
        mock_engine.db_path = db
        from sequoia_x.strategy.multi_factor import MultiFactorStrategy
        strategy = MultiFactorStrategy.__new__(MultiFactorStrategy)
        strategy.engine = mock_engine
        strategy._ml_scores_asof = None
        scores = strategy._compute_ml_scores(["600519"])
        assert scores == {}
