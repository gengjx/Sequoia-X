"""P4 回测引擎重构测试：消除回测与实盘脱节。

覆盖：
  - 成本一致性：回测用 apply_trading_costs（非 RTC），与实盘逐笔对账
  - 参数同源：回测参数从实盘组件导入（MAX_HOLD_DAYS 等）
  - 选股统一：_select_top 复用 multi_factor（含 ML/北向/趋势确认）
  - 无未来函数：ML as-of 快照严格 ≤ 回测日
  - ML walk-forward IC 指标
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from sequoia_x.analysis.paper_replay import PaperReplayEngine
from sequoia_x.analysis.paper_trade import apply_trading_costs


# ===========================================================================
# 成本一致性
# ===========================================================================
class TestCostUnification:
    def test_no_rtc_constant_in_use(self):
        """RTC 常量已废弃，所有成本走 apply_trading_costs。"""
        import sequoia_x.analysis.paper_replay as pr
        assert not hasattr(pr, "RTC") or "RTC" not in dir(pr)

    def test_buy_cost_matches_apply_trading_costs(self):
        """回测买入成本 == apply_trading_costs('buy')。"""
        price, shares = 10.0, 1000
        tc = apply_trading_costs("buy", price, shares)
        # net_cash 是现金流出总额
        assert tc["net_cash"] > price * shares  # 含佣金+滑点
        assert tc["fill_price"] > price  # 买入滑点偏高
        assert tc["commission"] == pytest.approx(tc["amount"] * 0.00025)

    def test_sell_cost_matches_apply_trading_costs(self):
        """回测卖出成本 == apply_trading_costs('sell')，含印花税。"""
        price, shares = 10.0, 1000
        tc = apply_trading_costs("sell", price, shares)
        assert tc["net_cash"] < price * shares  # 扣佣金+印花税+滑点
        assert tc["fill_price"] < price  # 卖出滑点偏低
        assert tc["stamp_duty"] == pytest.approx(tc["amount"] * 0.0005)
        assert tc["commission"] == pytest.approx(tc["amount"] * 0.00025)

    def test_buy_sell_roundtrip_cost_approx_02pct(self):
        """买卖一来回成本约 0.2%（实盘口径）。"""
        price, shares = 10.0, 1000
        buy = apply_trading_costs("buy", price, shares)
        sell = apply_trading_costs("sell", buy["fill_price"], shares)
        roundtrip = (buy["net_cash"] - sell["net_cash"]) / (price * shares)
        assert 0.0015 < roundtrip < 0.003  # ~0.15%-0.3%


# ===========================================================================
# 参数同源
# ===========================================================================
class TestParamSourcing:
    def test_max_hold_days_imported_from_paper_trade(self):
        """超时强平天数从 paper_trade 导入。"""
        import sequoia_x.analysis.paper_replay as pr
        from sequoia_x.analysis.paper_trade import MAX_HOLD_DAYS
        assert pr.MAX_HOLD_DAYS == MAX_HOLD_DAYS

    def test_import_apply_trading_costs(self):
        """回测模块成功导入 apply_trading_costs。"""
        import sequoia_x.analysis.paper_replay as pr
        assert pr.apply_trading_costs is apply_trading_costs


# ===========================================================================
# ML as-of 无未来函数
# ===========================================================================
class TestNoLookahead:
    def _make_db(self, tmp_path: Path) -> str:
        db = str(tmp_path / "test.db")
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE ml_scores ("
                "run_date TEXT, symbol TEXT, ml_score REAL, "
                "ic_mean REAL, icir REAL, t_stat REAL, model_version TEXT, "
                "PRIMARY KEY (run_date, symbol))"
            )
            # 快照1: 2026-01-15
            conn.execute(
                "INSERT INTO ml_scores VALUES ('2026-01-15','600519',0.5,0.1,1.0,3.0,'lgb')"
            )
            conn.execute(
                "INSERT INTO ml_scores VALUES ('2026-01-15','000001',-0.3,0.1,1.0,3.0,'lgb')"
            )
            # 快照2: 2026-03-15
            conn.execute(
                "INSERT INTO ml_scores VALUES ('2026-03-15','600519',0.8,0.1,1.0,3.0,'lgb')"
            )
        return db

    def test_asof_picks_latest_before_today(self, tmp_path: Path):
        """回测日 2026-02-01 应取快照 2026-01-15（≤ today 的最新）。"""
        db = self._make_db(tmp_path)
        engine = PaperReplayEngine(db)
        scores = engine._load_ml_scores_asof("2026-02-01")
        assert scores is not None
        # 应是 1月快照（0.5），不是3月快照（0.8）
        assert scores["600519"] == 0.5

    def test_asof_picks_correct_snapshot(self, tmp_path: Path):
        """回测日 2026-04-01 应取快照 2026-03-15。"""
        db = self._make_db(tmp_path)
        engine = PaperReplayEngine(db)
        scores = engine._load_ml_scores_asof("2026-04-01")
        assert scores is not None
        assert scores["600519"] == 0.8

    def test_no_future_snapshot(self, tmp_path: Path):
        """回测日 2026-01-01（早于所有快照）→ 无快照，回退 None。"""
        db = self._make_db(tmp_path)
        engine = PaperReplayEngine(db)
        scores = engine._load_ml_scores_asof("2026-01-01")
        assert scores is None

    def test_empty_db_returns_none(self, tmp_path: Path):
        """无 ml_scores 数据 → None（回退纯因子加权）。"""
        db = str(tmp_path / "test.db")
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE ml_scores ("
                "run_date TEXT, symbol TEXT, ml_score REAL, "
                "ic_mean REAL, icir REAL, t_stat REAL, model_version TEXT, "
                "PRIMARY KEY (run_date, symbol))"
            )
        engine = PaperReplayEngine(db)
        assert engine._load_ml_scores_asof("2026-06-01") is None

    def test_ml_scores_asof_injected_into_multi_factor(self):
        """_ml_scores_asof 注入后，multi_factor 不读 DB 最新快照。"""
        from sequoia_x.strategy.multi_factor import MultiFactorStrategy
        strat = MultiFactorStrategy.__new__(MultiFactorStrategy)
        strat._ml_scores_asof = {"600519": 0.42}
        strat.engine = MagicMock()
        scores = strat._compute_ml_scores(["600519"])
        assert scores == {"600519": 0.42}
        # 不应调用 DB（验证 DB 没被读）
        strat.engine.db_path.__str__.assert_not_called()


# ===========================================================================
# ML walk-forward IC 指标
# ===========================================================================
class TestMLWalkForward:
    def test_metrics_has_ml_ic_fields(self):
        """_calc_metrics 返回 dict 含 ML IC 字段。"""
        engine = PaperReplayEngine(":memory:")
        nav = [{"date": f"2026-01-{i:02d}", "nav": 1.0 + i * 0.001, "daily_return": 0.1}
               for i in range(1, 30)]
        bench = [{"date": d["date"], "nav": 1.0} for d in nav]
        m = engine._calc_metrics(nav, bench, [], 100000.0)
        assert "ml_ic_mean" in m
        assert "ml_icir" in m
        assert "ml_ic_months" in m
        # 无 ML 数据时为 0
        assert m["ml_ic_mean"] == 0.0

    def test_ml_ic_series_with_positive_ic(self):
        """构造正 IC 交易序列，验证 ml_ic_mean > 0。"""
        engine = PaperReplayEngine(":memory:")
        # 构造 ML score 与收益正相关的交易
        trades = []
        for i in range(15):
            trades.append({"side": "buy", "date": "2026-01-05", "symbol": f"S{i}"})
            # 高 score → 正收益
            trades.append({
                "side": "sell", "date": "2026-01-20", "symbol": f"S{i}",
                "pnl_pct": i * 2.0,  # 收益递增
            })
        # 设置 mock 快照：score 递增
        engine._ml_snapshots_cache = {
            "2026-01-04": {f"S{i}": float(i) for i in range(15)}
        }
        ic_series = engine._compute_ml_ic_series(
            [{"date": "2026-01-05", "nav": 1.0}], trades
        )
        assert len(ic_series) >= 1
        assert ic_series[0] > 0  # 正相关
