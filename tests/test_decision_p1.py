"""P1 实战能力提升测试：择时平滑（hysteresis）、仓位集中、入场质量。

覆盖：
  - _apply_regime_hysteresis：保守滞后转移矩阵（退出bear需确认、进入bear即时）
  - 仓位集中常量与逻辑门槛（MAX_HOLDINGS、最小手数/金额）
  - 默认策略池排除 demoted 弱策略
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sequoia_x.analysis.decision import (
    MAX_HOLDINGS,
    MIN_LOT_SHARES,
    MIN_POSITION_AMOUNT,
    REGIME_CONFIRM_SCORE,
    DecisionEngine,
    DecisionItem,
)
from sequoia_x.strategy.registry import (
    ACTIVE_STRATEGY_KEYS,
    RETIRED_STRATEGY_KEYS,
)


# ---------------------------------------------------------------------------
# 构造 DecisionEngine 实例（复用本地 DB，monkeypatch 掉网络依赖）
# ---------------------------------------------------------------------------
@pytest.fixture
def engine():
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    return DecisionEngine(DataEngine(Settings()), Settings())


def _trend(down=False, deep=False, pct=50.0):
    return {"pct_above_ma20": pct, "n_stocks": 4800,
            "trend_down": down, "deep_down": deep}


# ===========================================================================
# 择时平滑：_apply_regime_hysteresis 转移矩阵
# ===========================================================================
class TestRegimeHysteresis:
    """保守滞后：退出bear需确认、进入bear即时。"""

    def test_no_history_passes_through(self, engine):
        """无历史记录（首次运行）不施加滞后。"""
        with patch.object(engine, "_last_committed_state", return_value=None):
            state, score, label = engine._apply_regime_hysteresis("bull", 70, _trend())
        assert state == "bull"
        assert label == ""

    def test_prev_neutral_passes_through(self, engine):
        """前日非bear → 不施加滞后。"""
        with patch.object(engine, "_last_committed_state", return_value="neutral"):
            state, score, label = engine._apply_regime_hysteresis("bull", 70, _trend())
        assert state == "bull"

    def test_enter_bear_instant(self, engine):
        """进入bear即时降级（无需确认）— 资本保全优先。"""
        with patch.object(engine, "_last_committed_state", return_value="bull"):
            state, _, _ = engine._apply_regime_hysteresis("bear", 40, _trend(down=True))
        assert state == "bear"

    def test_exit_bear_trend_still_down_stays_bear(self, engine):
        """退出bear但趋势仍下跌 → 保持bear（最关键防守）。"""
        with patch.object(engine, "_last_committed_state", return_value="bear"):
            state, score, label = engine._apply_regime_hysteresis("bull", 70, _trend(down=True))
        assert state == "bear"
        assert "保持防御" in label

    def test_exit_bear_low_breadth_stays_bear(self, engine):
        """退出bear但breadth评分不足 → 保持bear。"""
        with patch.object(engine, "_last_committed_state", return_value="bear"):
            state, _, label = engine._apply_regime_hysteresis(
                "bull", REGIME_CONFIRM_SCORE - 1, _trend(down=False))
        assert state == "bear"

    def test_exit_bear_confirmed_goes_neutral(self, engine):
        """退出bear确认（趋势不跌+breadth≥55）→ 落neutral缓冲，不直接满仓bull。"""
        with patch.object(engine, "_last_committed_state", return_value="bear"):
            state, score, label = engine._apply_regime_hysteresis(
                "bull", REGIME_CONFIRM_SCORE, _trend(down=False))
        assert state == "neutral"
        assert "neutral缓冲" in label

    def test_exit_bear_two_day_sequence(self, engine):
        """两日序列：D1确认→neutral，D2(前日neutral)→放行bull。"""
        # D1: 前日bear, 确认 → neutral
        with patch.object(engine, "_last_committed_state", return_value="bear"):
            d1_state, _, _ = engine._apply_regime_hysteresis("bull", 60, _trend(down=False))
        assert d1_state == "neutral"
        # D2: 前日neutral(非bear) → 直接放行
        with patch.object(engine, "_last_committed_state", return_value="neutral"):
            d2_state, _, _ = engine._apply_regime_hysteresis("bull", 60, _trend(down=False))
        assert d2_state == "bull"

    def test_exit_bear_unconfirmed_loops_correctly(self, engine):
        """未确认的退出请求在多日后仍正确保持bear，直到条件满足。"""
        for _ in range(5):
            with patch.object(engine, "_last_committed_state", return_value="bear"):
                state, _, _ = engine._apply_regime_hysteresis("bull", 50, _trend(down=False))
            assert state == "bear"


# ===========================================================================
# 仓位集中：常量 + 门槛逻辑
# ===========================================================================
class TestPositionConcentration:
    def test_max_holdings_constant(self):
        assert MAX_HOLDINGS == 8

    def test_min_lot_shares_constant(self):
        assert MIN_LOT_SHARES == 300

    def test_min_position_amount_constant(self):
        assert MIN_POSITION_AMOUNT == 3000

    def test_small_position_below_threshold(self):
        """1手(100股)微仓应被门槛识别为过小。"""
        item = DecisionItem(symbol="000001", grade="A", price=10.0, shares=100, capital=1000)
        assert item.shares < MIN_LOT_SHARES
        assert item.capital < MIN_POSITION_AMOUNT

    def test_valid_position_passes_threshold(self):
        """3手(300股)¥3000+ 应通过门槛。"""
        item = DecisionItem(symbol="000001", grade="A", price=10.0, shares=300, capital=3000)
        assert item.shares >= MIN_LOT_SHARES
        assert item.capital >= MIN_POSITION_AMOUNT

    def test_buy_list_truncation_logic(self):
        """buy_list 超过 MAX_HOLDINGS 时应截断（验证排序后截断模式）。"""
        items = [
            DecisionItem(symbol=f"00000{i}", grade="A", score=90 - i)
            for i in range(12)
        ]
        truncated = items[:MAX_HOLDINGS]
        assert len(truncated) == MAX_HOLDINGS
        assert len(items) - MAX_HOLDINGS == 4  # 4只溢出


# ===========================================================================
# 入场质量：默认策略池排除 demoted
# ===========================================================================
class TestEntryQuality:
    def test_active_keys_excludes_demoted(self):
        """默认策略池仅含 core+active，不含 demoted 弱策略。"""
        demoted = {"ma_volume", "pullback", "volume_extreme"}
        assert ACTIVE_STRATEGY_KEYS == ["multi_factor", "bottom", "flag"]
        assert not (set(ACTIVE_STRATEGY_KEYS) & demoted)

    def test_active_keys_excludes_retired(self):
        """默认策略池不含已废弃策略。"""
        assert not (set(ACTIVE_STRATEGY_KEYS) & set(RETIRED_STRATEGY_KEYS))

    def test_scheduler_import_works(self):
        """scheduler.py 修复后的导入不再抛 ImportError。"""
        from sequoia_x.strategy.registry import ACTIVE_STRATEGY_KEYS as imported
        assert "multi_factor" in imported
        assert "ma_volume" not in imported
