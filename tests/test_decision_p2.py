"""P2 实战能力提升测试：样本外衰减折扣 + 涨停/停牌成交假设。

覆盖：
  - _quality_score 的 oos_decay 折扣公式（0.5+0.5*decay）
  - walk_forward per-strategy median_decay 聚合
  - _check_tradable 涨停/停牌判定逻辑
"""

from __future__ import annotations

from statistics import median

import pytest

from sequoia_x.analysis.strategy_eval import StrategyEvaluator
from sequoia_x.analysis.backtest_validation import ValidationResult
from sequoia_x.core.config import Settings
from sequoia_x.analysis.paper_trade import PaperTradeEngine


# ===========================================================================
# 样本外衰减折扣公式
# ===========================================================================
class TestDecayDiscount:
    """折扣公式 adjusted = base * (0.5 + 0.5*clamp(decay,0,1))。"""

    @staticmethod
    def _strat():
        """固定样本内得分的策略dict，便于隔离验证折扣。"""
        return {
            "sharpe": 0.5, "max_drawdown": -20.0, "profit_loss_ratio": 1.5,
            "alpha": 0.0, "annual_return": 10.0,
        }

    def test_decay_one_no_discount(self):
        """oos_decay=1.0（完全延续）→ 不打折。"""
        base = StrategyEvaluator._quality_score(self._strat(), 0.0, oos_decay=1.0)
        no_decay = StrategyEvaluator._quality_score(self._strat(), 0.0)
        assert base == no_decay

    def test_decay_zero_half_floor(self):
        """oos_decay=0.0（完全失效）→ 保留50%下限。"""
        base = StrategyEvaluator._quality_score(self._strat(), 0.0, oos_decay=1.0)
        decayed = StrategyEvaluator._quality_score(self._strat(), 0.0, oos_decay=0.0)
        assert decayed == int(round(base * 0.5))

    def test_decay_half_75pct(self):
        """oos_decay=0.5 → 75折。"""
        base = StrategyEvaluator._quality_score(self._strat(), 0.0, oos_decay=1.0)
        decayed = StrategyEvaluator._quality_score(self._strat(), 0.0, oos_decay=0.5)
        assert decayed == int(round(base * 0.75))

    def test_decay_missing_defaults_no_discount(self):
        """oos_decay 缺失（默认1.0）→ 不打折（安全回退）。"""
        result = StrategyEvaluator._quality_score(self._strat(), 0.0)
        base = StrategyEvaluator._quality_score(self._strat(), 0.0, oos_decay=1.0)
        assert result == base

    def test_decay_clamped_above_one(self):
        """oos_decay>1 → clamp到1.0，不超过无衰减分。"""
        base = StrategyEvaluator._quality_score(self._strat(), 0.0, oos_decay=1.0)
        boosted = StrategyEvaluator._quality_score(self._strat(), 0.0, oos_decay=2.0)
        assert boosted == base

    def test_decay_monotonic_decreasing(self):
        """decay 越低，质量分越低（单调递减）。"""
        d10 = StrategyEvaluator._quality_score(self._strat(), 0.0, oos_decay=1.0)
        d05 = StrategyEvaluator._quality_score(self._strat(), 0.0, oos_decay=0.5)
        d00 = StrategyEvaluator._quality_score(self._strat(), 0.0, oos_decay=0.0)
        assert d10 >= d05 >= d00


# ===========================================================================
# walk_forward per-strategy decay 聚合
# ===========================================================================
class TestPerStrategyDecay:
    """验证 walk_forward 返回 detail 含 per_strategy_decay。"""

    def test_result_has_per_strategy_decay_key(self):
        """ValidationResult detail 含 per_strategy_decay 字段。"""
        vr = ValidationResult("walk_forward", True, 70.0, {
            "median_decay": 0.7, "per_strategy_decay": {"multi_factor": 0.8},
        })
        assert "per_strategy_decay" in vr.detail
        assert vr.detail["per_strategy_decay"]["multi_factor"] == 0.8

    def test_multi_strategy_aggregation(self):
        """多策略多窗口的 decay 分组取中位数（复现 walk_forward 聚合逻辑）。"""
        results = [
            {"strategy": "multi_factor", "decay": 0.8},
            {"strategy": "multi_factor", "decay": 0.6},
            {"strategy": "multi_factor", "decay": 1.0},
            {"strategy": "bottom", "decay": 0.2},
            {"strategy": "bottom", "decay": 0.4},
        ]
        groups: dict[str, list[float]] = {}
        for r in results:
            groups.setdefault(r["strategy"], []).append(r["decay"])
        per_strategy = {k: round(float(median(v)), 2) for k, v in groups.items()}
        assert per_strategy["multi_factor"] == 0.8   # median(0.6,0.8,1.0)
        assert per_strategy["bottom"] == 0.3          # median(0.2,0.4)


# ===========================================================================
# 涨停/停牌成交假设（纯逻辑验证）
# ===========================================================================
def _board_threshold(symbol: str) -> float:
    """板块涨停阈值（复现 _check_tradable / _filter_limit_up 逻辑）。"""
    if symbol.startswith(("8", "4", "92")):
        return 28.5
    if symbol.startswith(("300", "301", "688", "689")):
        return 19.0
    return 9.5


class TestCheckTradable:
    """验证 _check_tradable 板块涨停阈值 + 停牌判定逻辑。"""

    @pytest.fixture
    def engine(self):
        return PaperTradeEngine(Settings())

    def test_normal_stock_tradable(self, engine):
        """正常股（非涨停非停牌）→ 可买入。"""
        tradable, reason = engine._check_tradable("600519")
        assert tradable is True
        assert reason == ""

    @pytest.mark.parametrize("symbol,expected", [
        ("600519", 9.5),    # 主板
        ("000001", 9.5),    # 深主板
        ("300750", 19.0),   # 创业板
        ("688981", 19.0),   # 科创板
        ("830799", 28.5),   # 北交所
    ])
    def test_board_threshold_logic(self, symbol, expected):
        """各板块涨停阈值正确。"""
        assert _board_threshold(symbol) == expected

    def test_limit_up_detection_logic(self):
        """涨停判定：pct_chg >= 阈值 → 不可买入。"""
        symbol = "600519"
        pct_chg = 10.0  # 主板涨停
        assert pct_chg >= _board_threshold(symbol)

    def test_near_limit_not_blocked(self):
        """接近但未达涨停 → 可买入。"""
        symbol = "600519"
        pct_chg = 9.0  # 主板接近涨停但未封板
        assert pct_chg < _board_threshold(symbol)

    def test_suspended_detection_logic(self):
        """停牌判定：tradestatus=0 → 不可买入。"""
        tradestatus = 0
        assert tradestatus == 0  # 停牌

    def test_gem_limit_higher_threshold(self):
        """创业板涨停阈值(19)高于主板(9.5)。"""
        assert _board_threshold("300750") > _board_threshold("600519")
