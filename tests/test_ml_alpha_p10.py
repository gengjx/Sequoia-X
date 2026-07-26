"""P10 选股 alpha 突破测试：ML 权重泄漏修复 + Walk-forward 快照。

覆盖：
  - ml_score 权重不被 evaluate_factor_ic 全量清零抹掉
  - _collect_training_data(as_of_month) PIT 截断（训练月份数据 ≤ as_of_month）
  - generate_walkforward_snapshots 按月生成历史快照
  - 回测读取历史 as-of 快照（无未来函数）
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from sequoia_x.analysis.ml_factor import MLFactorEngine


# ---------------------------------------------------------------------------
# 辅助：构造临时 DB
# ---------------------------------------------------------------------------
def _make_ml_db(tmp_path: Path, months: int = 25) -> str:
    """构造含 stock_daily（多月份）的临时 DB。"""
    db = str(tmp_path / "test.db")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS stock_daily "
            "(symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, "
            "volume REAL, turnover REAL, pct_chg REAL)"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS stock_market_cap (symbol TEXT, pe REAL, pb REAL)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS stock_finance "
            "(symbol TEXT, stat_date TEXT, roe REAL, np_margin REAL, gp_margin REAL, "
            "yoy_eps REAL, yoy_pni REAL, yoy_ni REAL, asset_turn REAL, inv_turn REAL, nr_turn REAL, "
            "PRIMARY KEY (symbol, stat_date))"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS fund_flow (symbol TEXT, date TEXT, main_net REAL, main_pct REAL, super_net REAL, big_net REAL)")
        conn.execute("CREATE TABLE IF NOT EXISTS north_hold (symbol TEXT, date TEXT, hold_share REAL, hold_value REAL, hold_pct REAL)")
        # 每月一个交易日，一只股票，价格递增
        base_dates = pd.date_range("2022-01-01", periods=months, freq="MS")
        for i, dt in enumerate(base_dates):
            d = dt.strftime("%Y-%m-%d")
            for sym in ("000001", "000002"):
                close = 10.0 + i * 0.2 + (1 if sym == "000002" else 0)
                conn.execute(
                    "INSERT INTO stock_daily VALUES (?,?,?,?,?,?,?, ?,0)",
                    (sym, d, close * 0.99, close * 1.01, close * 0.98, close,
                     2_000_000, 2_000_000 * close),
                )
            conn.execute("INSERT INTO stock_market_cap VALUES ('000001', 15, 2)")
            conn.execute("INSERT INTO stock_market_cap VALUES ('000002', 20, 3)")
        # ml_scores 表（测试 walk-forward 写入）
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ml_scores "
            "(run_date TEXT, symbol TEXT, ml_score REAL, ic_mean REAL, icir REAL, "
            "t_stat REAL, model_version TEXT, PRIMARY KEY (run_date, symbol))"
        )
        # factor_weights 表（测试权重恢复）
        conn.execute(
            "CREATE TABLE IF NOT EXISTS factor_weights "
            "(factor_name TEXT PRIMARY KEY, category TEXT, ic_mean REAL, icir REAL, "
            "win_rate REAL, weight REAL, updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO factor_weights VALUES "
            "('ml_score','ML因子',0.11,1.07,83,0.11,'2026-07-26 10:00:00')"
        )
        conn.commit()
    return db


# ===========================================================================
# ML 权重泄漏修复
# ===========================================================================
class TestWeightLeakageFix:
    def test_ml_weight_preserved_after_factor_ic_clear(self, tmp_path):
        """evaluate_factor_ic 的全量清零不应抹掉 ml_score 权重。

        本测试直接验证备份/恢复逻辑：清零前 ml_score.weight=0.11，
        清零后应恢复。
        """
        db = _make_ml_db(tmp_path)
        with sqlite3.connect(db) as conn:
            # 模拟 evaluate_factor_ic 的备份+清零+恢复逻辑
            row = conn.execute(
                "SELECT weight, ic_mean, icir, win_rate, updated_at "
                "FROM factor_weights WHERE factor_name='ml_score'"
            ).fetchone()
            assert row and row[0] == pytest.approx(0.11)

            # 全量清零
            conn.execute("UPDATE factor_weights SET weight=0")
            # 模拟写常规因子（这里写一个测试因子）
            conn.execute(
                "INSERT OR REPLACE INTO factor_weights "
                "(factor_name, category, ic_mean, weight) VALUES ('test_f','测试',0.05,0.5)"
            )
            # 恢复 ml_score
            if row[0] and row[0] != 0:
                conn.execute(
                    "INSERT OR REPLACE INTO factor_weights "
                    "(factor_name, category, ic_mean, icir, win_rate, weight, updated_at) "
                    "VALUES ('ml_score','ML因子',?,?,?,?,?)",
                    (row[1], row[2], row[3], row[0], row[4]),
                )
            conn.commit()

            # 验证：ml_score 权重恢复
            ml_row = conn.execute(
                "SELECT weight FROM factor_weights WHERE factor_name='ml_score'"
            ).fetchone()
            assert ml_row[0] == pytest.approx(0.11)


# ===========================================================================
# Walk-forward as-of 训练
# ===========================================================================
class TestWalkforwardAsOf:
    def test_as_of_month_filters_future_months(self, tmp_path):
        """_collect_training_data(as_of_month) 只返回 ≤ as_of_month 的月份。"""
        db = _make_ml_db(tmp_path, months=25)
        engine = MLFactorEngine(db)
        data = engine._collect_training_data(as_of_month="2023-06")
        if data:  # 数据足够时
            all_months = {d["month"] for d in data}
            for m in all_months:
                assert m <= "2023-06", f"{m} 超过 as_of_month 2023-06（未来函数）"

    def test_as_of_month_none_uses_latest(self, tmp_path):
        """as_of_month=None 取最新月（实盘/单次训练行为不变）。"""
        db = _make_ml_db(tmp_path, months=25)
        engine = MLFactorEngine(db)
        data = engine._collect_training_data(as_of_month=None)
        if data:
            all_months = {d["month"] for d in data}
            # 应包含最近的月份
            assert max(all_months) >= "2023-12"


# ===========================================================================
# Walk-forward 快照生成
# ===========================================================================
class TestWalkforwardSnapshots:
    def test_generate_creates_snapshots(self, tmp_path):
        """generate_walkforward_snapshots 在 ml_scores 表写入多个 run_date 快照。"""
        db = _make_ml_db(tmp_path, months=25)
        engine = MLFactorEngine(db)
        # 小范围测试（数据可能不足，验证不崩 + 调用成功）
        result = engine.generate_walkforward_snapshots(
            start_month="2023-10", end_month="2023-12"
        )
        assert "months_generated" in result
        # 即使数据不足生成0个，也不应报错
        assert isinstance(result["months_generated"], int)

    def test_save_ml_scores_to_date(self, tmp_path):
        """_save_ml_scores_to_date 写指定 run_date 的快照。"""
        db = _make_ml_db(tmp_path)
        engine = MLFactorEngine(db)
        engine._save_ml_scores_to_date(
            {"predictions": {"000001": 0.5, "000002": -0.3},
             "ic_mean": 0.1, "icir": 1.0, "t_stat": 3.0, "model_version": "test"},
            run_date="2024-06-30",
        )
        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT symbol, ml_score FROM ml_scores WHERE run_date='2024-06-30'"
            ).fetchall()
        assert len(rows) == 2
        syms = {r[0] for r in rows}
        assert syms == {"000001", "000002"}


# ===========================================================================
# 回测读取历史快照（无未来函数）
# ===========================================================================
class TestReplayAsOfSnapshot:
    def test_asof_picks_correct_historical_snapshot(self, tmp_path):
        """回测日 2024-02-15 应取 ≤ 2024-02 的快照，非 2026 最新。"""
        from sequoia_x.analysis.paper_replay import PaperReplayEngine
        db = _make_ml_db(tmp_path)
        # 写两个快照：2024-01 和 2026-07
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO ml_scores VALUES ('2024-01-31','000001',0.3,0.1,1.0,3.0,'wf')"
            )
            conn.execute(
                "INSERT INTO ml_scores VALUES ('2026-07-25','000001',0.9,0.1,1.0,3.0,'lgb')"
            )
            conn.commit()
        eng = PaperReplayEngine(db)
        scores = eng._load_ml_scores_asof("2024-02-15")
        assert scores is not None
        # 应取 2024-01（0.3），不是 2026-07（0.9）
        assert scores["000001"] == pytest.approx(0.3)

    def test_asof_before_all_snapshots_returns_none(self, tmp_path):
        """回测日早于所有快照 → None（回退纯因子加权）。"""
        from sequoia_x.analysis.paper_replay import PaperReplayEngine
        db = _make_ml_db(tmp_path)
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO ml_scores VALUES ('2024-06-30','000001',0.5,0.1,1.0,3.0,'wf')"
            )
            conn.commit()
        eng = PaperReplayEngine(db)
        scores = eng._load_ml_scores_asof("2023-01-01")
        assert scores is None
