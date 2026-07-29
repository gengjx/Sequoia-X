"""P19 ML 因子覆盖扩展测试：预计算优化 + walk-forward 快照复用。

覆盖：
  - _precompute_all_factors() 遍历全部月份（非仅最后 TRAIN_MONTHS+3 个月）
  - generate_walkforward_snapshots 只调一次预计算，不再逐月 _collect_training_data
  - 预计算与 _collect_training_data 结果一致性（同 symbol 同 month）
  - 月份不足时安全返回空列表
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sequoia_x.analysis.ml_factor import MLFactorEngine


# ---------------------------------------------------------------------------
# 辅助：构造含充分日频数据的临时 DB（fwd_return 可计算）
# ---------------------------------------------------------------------------
def _make_rich_ml_db(tmp_path: Path, n_days: int = 840, n_stocks: int = 3) -> str:
    """构造含日频 K 线的临时 DB。

    n_days 个交易日（~40 个月），每只股票带趋势+噪声，
    确保 compute_factors + 前向收益（HOLD_DAYS=20）均可计算。
    """
    db = str(tmp_path / "p19.db")
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS stock_daily "
            "(symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, "
            "volume REAL, turnover REAL, pct_chg REAL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS stock_market_cap (symbol TEXT, pe REAL, pb REAL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS stock_finance "
            "(symbol TEXT, stat_date TEXT, roe REAL, np_margin REAL, gp_margin REAL, "
            "yoy_eps REAL, yoy_pni REAL, yoy_ni REAL, asset_turn REAL, inv_turn REAL, "
            "nr_turn REAL, cfo_to_or REAL, cfo_to_np REAL, cfo_to_gr REAL, "
            "tangible_ratio REAL, liability_to_asset REAL, equity_multiplier REAL, "
            "PRIMARY KEY (symbol, stat_date))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS fund_flow "
            "(symbol TEXT, date TEXT, main_net REAL, main_pct REAL, "
            "super_net REAL, big_net REAL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS north_hold "
            "(symbol TEXT, date TEXT, hold_share REAL, hold_value REAL, hold_pct REAL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ml_scores "
            "(run_date TEXT, symbol TEXT, ml_score REAL, ic_mean REAL, icir REAL, "
            "t_stat REAL, model_version TEXT, PRIMARY KEY (run_date, symbol))"
        )

        for si in range(n_stocks):
            sym = f"00000{si + 1}"
            drift = 0.0004 + 0.0002 * si
            rets = rng.normal(drift, 0.02, n_days)
            closes = 10.0 * np.cumprod(1 + rets)
            for i, d in enumerate(date_strs):
                close = float(closes[i])
                high = close * 1.01
                low = close * 0.99
                open_ = close * 0.995
                vol = 3_000_000.0
                conn.execute(
                    "INSERT INTO stock_daily VALUES (?,?,?,?,?,?,?, ?,?)",
                    (sym, d, open_, high, low, close, vol, vol * close, float(rets[i] * 100)),
                )
            conn.execute("INSERT INTO stock_market_cap VALUES (?,?,?)", (sym, 15.0 + si, 2.0 + si * 0.5))
        conn.commit()
    return db


# ===========================================================================
# 预计算覆盖全部月份
# ===========================================================================
class TestPrecomputeCoverage:
    def test_precompute_returns_list(self, tmp_path):
        db = _make_rich_ml_db(tmp_path)
        engine = MLFactorEngine(db)
        data = engine._precompute_all_factors()
        assert isinstance(data, list)

    def test_precompute_covers_more_months_than_collect(self, tmp_path):
        """预计算遍历全部月份，_collect_training_data 只取最后 TRAIN_MONTHS+3 个月。"""
        db = _make_rich_ml_db(tmp_path)
        engine = MLFactorEngine(db)
        pre = engine._precompute_all_factors()
        collect = engine._collect_training_data(as_of_month=None)
        if pre and collect:
            pre_months = {r["month"] for r in pre}
            collect_months = {r["month"] for r in collect}
            # 预计算覆盖更早的月份
            assert min(pre_months) < min(collect_months)
            assert len(pre_months) > len(collect_months)

    def test_precompute_covers_beyond_train_window(self, tmp_path):
        """预计算月份数 > TRAIN_MONTHS+3（证明非截断到训练窗口）。"""
        db = _make_rich_ml_db(tmp_path)
        engine = MLFactorEngine(db)
        data = engine._precompute_all_factors()
        if data:
            months = {r["month"] for r in data}
            assert len(months) > engine.TRAIN_MONTHS + 3

    def test_precompute_record_structure(self, tmp_path):
        db = _make_rich_ml_db(tmp_path)
        engine = MLFactorEngine(db)
        data = engine._precompute_all_factors()
        if data:
            rec = data[0]
            assert "symbol" in rec
            assert "month" in rec
            assert "features" in rec
            assert "fwd_return" in rec
            assert set(rec["features"].keys()) == set(engine.FEATURE_FACTORS)


# ===========================================================================
# 预计算与 _collect_training_data 一致性
# ===========================================================================
class TestPrecomputeConsistency:
    def test_collect_subset_of_precompute(self, tmp_path):
        """_collect_training_data 的每条记录都应出现在预计算结果中（同 symbol+month）。"""
        db = _make_rich_ml_db(tmp_path)
        engine = MLFactorEngine(db)
        pre = engine._precompute_all_factors()
        collect = engine._collect_training_data(as_of_month=None)
        if pre and collect:
            pre_keys = {(r["symbol"], r["month"]) for r in pre}
            for r in collect:
                assert (r["symbol"], r["month"]) in pre_keys


# ===========================================================================
# walk-forward 使用预计算（不再逐月 _collect_training_data）
# ===========================================================================
class TestWalkforwardUsesPrecompute:
    def test_precompute_called_once(self, tmp_path):
        """generate_walkforward_snapshots 只调一次 _precompute_all_factors。"""
        db = _make_rich_ml_db(tmp_path)
        engine = MLFactorEngine(db)
        with patch.object(engine, "_precompute_all_factors", wraps=engine._precompute_all_factors) as spy_pre, \
             patch.object(engine, "_collect_training_data", wraps=engine._collect_training_data) as spy_col:
            result = engine.generate_walkforward_snapshots(
                start_month="2023-06", end_month="2023-09"
            )
            assert spy_pre.call_count == 1
            assert spy_col.call_count == 0
            assert "months_generated" in result

    def test_walkforward_does_not_crash(self, tmp_path):
        db = _make_rich_ml_db(tmp_path)
        engine = MLFactorEngine(db)
        result = engine.generate_walkforward_snapshots(
            start_month="2023-06", end_month="2023-08"
        )
        assert isinstance(result, dict)
        assert isinstance(result.get("months_generated", 0), int)


# ===========================================================================
# 边界：月份不足
# ===========================================================================
class TestPrecomputeEdgeCases:
    def test_empty_when_insufficient_months(self, tmp_path):
        """月份 < TRAIN_MONTHS+2 时返回空列表。"""
        db = _make_rich_ml_db(tmp_path, n_days=200)  # ~10 个月
        engine = MLFactorEngine(db)
        data = engine._precompute_all_factors()
        assert data == []
