"""P8b 回测引擎可信化测试：PIT as-of 快照 + 真实沪深300基准 + 全市场多种子。

覆盖：
  - PIT 正确性：回测日取到 ≤ today 的最新快照（非未来值）
  - as-of 边界：today 早于所有快照 → None/nan（不报错、不前视）
  - 基准正确性：回测 benchmark == 沪深300 累乘曲线（非采样股等权）
  - strategy 复用：多次 _select_top 复用同一实例（不重建）
  - 全市场：sample_size=0 处理全部 symbol
  - 多种子聚合：run_validation 返回 per-seed + summary
  - 实盘路径不退化：as_of_date=None 读最新值
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sequoia_x.analysis.paper_replay import PaperReplayEngine
from sequoia_x.strategy.multi_factor import MultiFactorStrategy, _pit_asof


# ---------------------------------------------------------------------------
# 辅助：构造带财报 + 指数 + K线数据的临时 DB
# ---------------------------------------------------------------------------
def _make_pit_db(tmp_path: Path) -> str:
    """构造含两期财报（2021-12 / 2026-01）+ 沪深300 + K线的 DB。"""
    db = str(tmp_path / "test.db")
    with sqlite3.connect(db) as conn:
        # 基本表
        conn.execute("CREATE TABLE IF NOT EXISTS stock_basic (symbol TEXT, name TEXT, ipo_date TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS stock_daily (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover REAL, pct_chg REAL, tradestatus INTEGER)")
        conn.execute("CREATE TABLE IF NOT EXISTS index_daily (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")

        # stock_finance：两期 ROE
        conn.execute(
            "CREATE TABLE IF NOT EXISTS stock_finance "
            "(symbol TEXT, stat_date TEXT, roe REAL, np_margin REAL, gp_margin REAL, "
            "yoy_eps REAL, yoy_pni REAL, yoy_ni REAL, asset_turn REAL, inv_turn REAL, nr_turn REAL, "
            "PRIMARY KEY (symbol, stat_date))"
        )
        conn.execute("INSERT INTO stock_finance VALUES ('000001','2021-12-31',8.0,0,0,0,0,0,0,0,0)")
        conn.execute("INSERT INTO stock_finance VALUES ('000001','2026-01-31',30.0,0,0,0,0,0,0,0,0)")

        # 沪深300指数（构造已知收益率序列）
        for i in range(80):
            dt = f"2025-01-{i+1:02d}"
            close = 100.0 * (1.001 ** i)
            conn.execute("INSERT INTO index_daily VALUES ('000300', ?, 0,0,0,?,0)", (dt, close))

        # K线（80天，逐日上涨，满足 60 天最低要求）
        for i in range(80):
            dt = f"2025-01-{i+1:02d}"
            close = 10.0 * (1.001 ** i)
            conn.execute(
                "INSERT INTO stock_daily VALUES ('000001', ?, ?,?,?,?, ?, ?, 0.1, 1)",
                (dt, close * 0.99, close * 1.01, close * 0.98, close,
                 1_000_000, 1_000_000 * close),
            )
        conn.execute("INSERT INTO stock_basic VALUES ('000001','平安银行','2020-01-01')")
        conn.commit()
    return db


# ===========================================================================
# PIT as-of 正确性
# ===========================================================================
class TestPitAsOf:
    def test_pit_asof_picks_correct_finance(self, tmp_path):
        """回测日 2022-03-15 应取 2021-12 的 ROE=8.0（非 2026 的 30.0）。"""
        db = _make_pit_db(tmp_path)
        from sequoia_x.core.config import Settings
        from sequoia_x.data.engine import DataEngine
        from types import SimpleNamespace
        settings = Settings()
        engine = DataEngine(settings)
        engine.db_path = db
        strat = MultiFactorStrategy(engine, settings)
        strat.preload_snapshot_history()
        finance_map = strat._build_asof_finance(["000001"], "2022-03-15")
        assert finance_map["000001"]["roe"] == pytest.approx(8.0)

    def test_pit_asof_picks_latest_finance(self, tmp_path):
        """回测日 2026-02-01 应取 2026-01 的 ROE=30.0。"""
        db = _make_pit_db(tmp_path)
        from sequoia_x.core.config import Settings
        from sequoia_x.data.engine import DataEngine
        settings = Settings()
        engine = DataEngine(settings)
        engine.db_path = db
        strat = MultiFactorStrategy(engine, settings)
        strat.preload_snapshot_history()
        finance_map = strat._build_asof_finance(["000001"], "2026-02-01")
        assert finance_map["000001"]["roe"] == pytest.approx(30.0)

    def test_asof_before_all_snapshots_returns_empty(self, tmp_path):
        """today 早于所有快照日期 → 无数据（不报错）。"""
        db = _make_pit_db(tmp_path)
        from sequoia_x.core.config import Settings
        from sequoia_x.data.engine import DataEngine
        settings = Settings()
        engine = DataEngine(settings)
        engine.db_path = db
        strat = MultiFactorStrategy(engine, settings)
        strat.preload_snapshot_history()
        # 2020-01 早于所有财报 stat_date
        finance_map = strat._build_asof_finance(["000001"], "2020-01-01")
        assert "000001" not in finance_map

    def test_pit_asof_helper_correct(self):
        """模块级 _pit_asof 的 bisect 逻辑正确。"""
        seq = [("2022-01", {"v": 1}), ("2022-03", {"v": 2}), ("2022-06", {"v": 3})]
        assert _pit_asof(seq, "2022-02") == {"v": 1}
        assert _pit_asof(seq, "2022-04") == {"v": 2}
        assert _pit_asof(seq, "2022-07") == {"v": 3}
        assert _pit_asof(seq, "2021-01") is None  # 早于所有
        assert _pit_asof([], "2022-01") is None   # 空序列


# ===========================================================================
# 基准正确性（真实沪深300）
# ===========================================================================
class TestBenchmark:
    def test_benchmark_uses_real_csi300(self, tmp_path):
        """回测 benchmark 应基于 index_daily(000300)，非采样股等权。"""
        db = _make_pit_db(tmp_path)
        engine = PaperReplayEngine(db)
        idx_ret = engine._load_index_returns()
        assert idx_ret is not None
        assert len(idx_ret) >= 60
        # 指数递增 → 日收益率为正
        assert idx_ret.iloc[0] > 0

    def test_benchmark_not_sample_equal_weight(self, tmp_path):
        """不同 seed 的 benchmark 曲线应该完全相同（真实指数不随 seed 变化）。"""
        db = _make_pit_db(tmp_path)
        eng1 = PaperReplayEngine(db)
        res1 = eng1.replay(start_date="2025-01-02", end_date="2025-01-30",
                           sample_size=0, seed=42)
        eng2 = PaperReplayEngine(db)
        res2 = eng2.replay(start_date="2025-01-02", end_date="2025-01-30",
                           sample_size=0, seed=999)
        bm1 = res1["benchmark"]
        bm2 = res2["benchmark"]
        assert len(bm1) == len(bm2)
        for a, b in zip(bm1, bm2):
            assert a["nav"] == pytest.approx(b["nav"], rel=1e-6)


# ===========================================================================
# Strategy 复用
# ===========================================================================
class TestStrategyReuse:
    def test_get_replay_strategy_caches_instance(self, tmp_path):
        """多次调用复用同一 MultiFactorStrategy 实例。"""
        db = _make_pit_db(tmp_path)
        engine = PaperReplayEngine(db)
        s1 = engine._get_replay_strategy()
        s2 = engine._get_replay_strategy()
        assert id(s1) == id(s2)

    def test_replay_strategy_has_snapshot_history(self, tmp_path):
        """复用的 strategy 已预加载快照历史。"""
        db = _make_pit_db(tmp_path)
        engine = PaperReplayEngine(db)
        strat = engine._get_replay_strategy()
        assert strat._snapshot_history is not None
        assert "finance" in strat._snapshot_history


# ===========================================================================
# 全市场
# ===========================================================================
class TestFullMarket:
    def test_sample_size_zero_uses_all(self, tmp_path):
        """sample_size=0（falsy）→ 不采样，用全部 symbol。"""
        db = _make_pit_db(tmp_path)
        engine = PaperReplayEngine(db)
        res = engine.replay(start_date="2025-01-02", end_date="2025-01-10",
                            sample_size=0, seed=42)
        assert res["config"]["sample_size"] >= 1  # 至少处理到 000001


# ===========================================================================
# 多种子聚合
# ===========================================================================
class TestRunValidation:
    def test_returns_per_seed_and_summary(self, tmp_path):
        """run_validation 返回 per_seed 列表 + summary 的 mean/std 字段。"""
        db = _make_pit_db(tmp_path)
        engine = PaperReplayEngine(db)
        res = engine.run_validation(
            start_date="2025-01-02", end_date="2025-01-10",
            seeds=(1, 2),
        )
        assert "per_seed" in res
        assert len(res["per_seed"]) == 2
        assert "summary" in res
        for k in ("annual_return_mean", "annual_return_std",
                   "sharpe_mean", "sharpe_std",
                   "max_drawdown_mean", "alpha_mean"):
            assert k in res["summary"]


# ===========================================================================
# 实盘路径不退化
# ===========================================================================
class TestLivePathNoRegression:
    def test_as_of_none_uses_latest(self, tmp_path):
        """as_of_date=None 时实盘走最新快照（非 as-of）。"""
        db = _make_pit_db(tmp_path)
        from sequoia_x.core.config import Settings
        from sequoia_x.data.engine import DataEngine
        settings = Settings()
        engine = DataEngine(settings)
        engine.db_path = db
        strat = MultiFactorStrategy(engine, settings)
        strat.preload_snapshot_history()
        # as_of_date=None → 走 _load_finance_map（最新值 30.0）
        finance_map = strat._load_finance_map(["000001"])
        assert finance_map["000001"]["roe"] == pytest.approx(30.0)
