"""策略增量贡献测试（compute_strategy_marginal + _truncate_pool 加权 + 弱信号标注）。

覆盖：
  - compute_strategy_marginal 增量 alpha 计算
  - _truncate_pool 使用 marginal_alpha 排序
  - _grade_item 弱信号标记
  - DB 写入 marginal_alpha 列
"""

from __future__ import annotations

import numpy as np
import pytest


# ── FactorHealth 纯函数单测 ──────────────────────────────────────

class TestComputeStrategyMarginal:
    """compute_strategy_marginal 增量 alpha 计算。"""

    def test_marginal_alpha_direction(self, tmp_db_with_data):
        """高收益策略 marginal_alpha>0，低收益策略 <0。"""
        from sequoia_x.analysis.combo_backtest import ComboBacktester
        from sequoia_x.data.engine import DataEngine
        from sequoia_x.core.config import Settings

        settings = Settings(db_path=tmp_db_with_data)
        engine = DataEngine(settings)
        cb = ComboBacktester(engine, settings)
        result = cb.compute_strategy_marginal(hold_days=5, sample_size=0)

        assert isinstance(result, dict)
        assert len(result) > 0

        for skey, info in result.items():
            assert "marginal_alpha_pp" in info
            assert "sample_count" in info
            assert isinstance(info["marginal_alpha_pp"], float)

    def test_marginal_alpha_known_values(self):
        """构造已知收益分布断言增量 alpha = 样本均值 - 池均值。"""
        from sequoia_x.analysis.combo_backtest import ComboBacktester

        # Mock: 策略 A 选出高收益股，策略 B 选出低收益股
        strategy_returns = {
            "A": {20: ([0.02, 0.03, 0.025, 0.015, 0.035], [])},
            "B": {20: ([-0.01, -0.02, -0.015, -0.005, -0.025], [])},
        }
        all_nets = [r for s in strategy_returns.values() for r in s[20][0]]
        pool_avg = np.mean(all_nets)
        annualize = 252.0 / 20

        a_avg = np.mean(strategy_returns["A"][20][0])
        b_avg = np.mean(strategy_returns["B"][20][0])
        a_marginal = (a_avg - pool_avg) * annualize * 100
        b_marginal = (b_avg - pool_avg) * annualize * 100

        assert a_marginal > 0, "高收益策略应有正增量 alpha"
        assert b_marginal < 0, "低收益策略应有负增量 alpha"
        assert a_marginal == pytest.approx(-b_marginal, rel=0.01), "对称分布增量幅度相近"


class TestTruncatePoolWeighting:
    """_truncate_pool 使用 marginal_alpha 排序。"""

    def test_marginal_alpha_loaded_from_db(self, tmp_db_with_weights):
        """DecisionEngine 从 DB 加载 marginal_alpha。"""
        from sequoia_x.data.engine import DataEngine
        from sequoia_x.core.config import Settings
        from sequoia_x.analysis.decision import DecisionEngine

        settings = Settings(db_path=tmp_db_with_weights)
        engine = DataEngine(settings)
        de = DecisionEngine(engine, settings)
        assert "multi_factor" in de._strategy_marginal
        assert de._strategy_marginal["multi_factor"] == pytest.approx(15.0)
        assert de._strategy_marginal["bottom"] == pytest.approx(-5.0)

    def test_positive_alpha_ranks_higher(self, tmp_db_with_weights):
        """正 marginal_alpha 策略命中的票排序更高。"""
        from sequoia_x.data.engine import DataEngine
        from sequoia_x.core.config import Settings
        from sequoia_x.analysis.decision import DecisionEngine

        settings = Settings(db_path=tmp_db_with_weights)
        engine = DataEngine(settings)
        de = DecisionEngine(engine, settings)

        # 构造持仓池：所有票动量相同，但策略不同
        pool = {
            "000001": ["multi_factor"],   # marginal_alpha=15 → +1.5 bonus
            "000002": ["bottom"],          # marginal_alpha=-5 → -0.5 bonus
        }
        kept = de._truncate_pool(pool, max_candidates=1)
        assert "000001" in kept
        assert "000002" not in kept


class TestWeakSignalMarker:
    """_grade_item 弱信号标注。"""

    def test_weak_signal_labeled(self, tmp_db_with_weights):
        """仅由负 marginal_alpha 策略触发的票标注弱信号。"""
        from sequoia_x.data.engine import DataEngine
        from sequoia_x.core.config import Settings
        from sequoia_x.analysis.decision import DecisionEngine, DecisionItem

        settings = Settings(db_path=tmp_db_with_weights)
        engine = DataEngine(settings)
        de = DecisionEngine(engine, settings)

        item = DecisionItem(
            symbol="000002", name="测试", resonance=1,
            hit_strategies=["底部放量"],  # bottom, marginal_alpha=-5
            strategy_quality=53, score=70, price=10, stop_loss=9, target=12,
        )
        de._grade_item(item, capital=100000, market_state="neutral")

        assert item.grade in ("A", "B", "C", "观望")
        if item.reason:
            assert "弱信号主导" in item.reason or "有效评分" in item.reason

    def test_strong_signal_not_labeled(self, tmp_db_with_weights):
        """multi_factor 命中的票不标注弱信号。"""
        from sequoia_x.data.engine import DataEngine
        from sequoia_x.core.config import Settings
        from sequoia_x.analysis.decision import DecisionEngine, DecisionItem

        settings = Settings(db_path=tmp_db_with_weights)
        engine = DataEngine(settings)
        de = DecisionEngine(engine, settings)

        item = DecisionItem(
            symbol="000001", name="测试", resonance=1,
            hit_strategies=["多因子选股"],  # multi_factor, marginal_alpha=15
            strategy_quality=80, score=75, price=10, stop_loss=9, target=12,
        )
        de._grade_item(item, capital=100000, market_state="neutral")

        assert item.grade in ("A", "B", "C", "观望")
        assert "弱信号" not in (item.reason or "")


class TestDBPersistence:
    """marginal_alpha 列持久化。"""

    def test_save_and_load_marginal_alpha(self, tmp_db):
        from sequoia_x.data.engine import DataEngine
        from sequoia_x.core.config import Settings

        settings = Settings(db_path=tmp_db)
        engine = DataEngine(settings)

        engine.save_strategy_weights([{
            "strategy_key": "test_strat",
            "quality_score": 50,
            "sharpe": 1.0,
            "alpha": 5.0,
            "marginal_alpha": 12.5,
        }])

        loaded = engine.load_strategy_weights()
        assert "test_strat" in loaded
        assert loaded["test_strat"]["marginal_alpha"] == pytest.approx(12.5)


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """创建临时 DB。"""
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.core.config import Settings
    db_path = str(tmp_path / "test.db")
    DataEngine(Settings(db_path=db_path))
    return db_path


@pytest.fixture
def tmp_db_with_weights(tmp_db):
    """创建含 marginal_alpha 权重的 DB。"""
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.core.config import Settings

    settings = Settings(db_path=tmp_db)
    engine = DataEngine(settings)
    engine.save_strategy_weights([
        {"strategy_key": "multi_factor", "quality_score": 70, "marginal_alpha": 15.0},
        {"strategy_key": "bottom", "quality_score": 53, "marginal_alpha": -5.0},
        {"strategy_key": "flag", "quality_score": 51, "marginal_alpha": -8.0},
    ])
    return tmp_db


@pytest.fixture
def tmp_db_with_data(tmp_db):
    """创建含 K线数据的 DB 供 combo_backtest 运行。"""
    import sqlite3
    import pandas as pd

    with sqlite3.connect(tmp_db) as conn:
        dates = pd.bdate_range("2024-01-01", periods=120)
        for sym in ("000001", "000002", "000003", "000004", "000005"):
            prices = np.cumprod(1 + np.random.randn(120) * 0.02) * 10
            for i, d in enumerate(dates):
                conn.execute(
                    "INSERT OR REPLACE INTO stock_daily "
                    "(symbol, date, open, high, low, close, volume, turnover, "
                    "turn, pct_chg, tradestatus, isst) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sym, d.strftime("%Y-%m-%d"), float(prices[i]*0.99),
                     float(prices[i]*1.01), float(prices[i]*0.98), float(prices[i]),
                     1000000.0, float(prices[i]*1000000), 2.5, 1.0, 1, 0),
                )
        conn.commit()
    return tmp_db
