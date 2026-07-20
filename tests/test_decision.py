"""决策中枢纯函数与 staticmethod 数值/逻辑正确性测试。

覆盖 quality_tier/quality_bonus/quality_label/detect_strategy_conflicts
（质量分层与冲突检测）、_has_hard_risk（硬风险淘汰）、_filter_pool
（市场板块过滤）、_allocate_capital（ATR 风险预算+行业暴露约束）、
_build_summary/_to_dict/_empty_result（结果组装）。

这些是策略选股→质量过滤→共振定级→仓位分配链路的决策基石，
回归测试防止定级/风控/资金规则被无意改坏。
"""

from __future__ import annotations

import pytest

from sequoia_x.analysis.decision import (
    DecisionEngine,
    DecisionItem,
    detect_strategy_conflicts,
    quality_bonus,
    quality_label,
    quality_tier,
)


# ---------------------------------------------------------------------------
# quality_tier：质量分 → S/A/B/C/D 分层（阈值边界）
# ---------------------------------------------------------------------------
class TestQualityTier:
    def test_s_tier_threshold(self):
        assert quality_tier(80) == "S"
        assert quality_tier(100) == "S"

    def test_a_tier_range(self):
        assert quality_tier(65) == "A"
        assert quality_tier(79) == "A"

    def test_b_tier_range(self):
        assert quality_tier(45) == "B"
        assert quality_tier(64) == "B"

    def test_c_tier_range(self):
        assert quality_tier(30) == "C"
        assert quality_tier(44) == "C"

    def test_d_tier_low(self):
        assert quality_tier(29) == "D"
        assert quality_tier(0) == "D"

    @pytest.mark.parametrize("score,expected", [
        (80, "S"), (65, "A"), (45, "B"), (30, "C"), (29, "D"),
    ])
    def test_boundary_is_inclusive_lower(self, score, expected):
        """各档下限（含）归入该档。"""
        assert quality_tier(score) == expected


# ---------------------------------------------------------------------------
# quality_bonus：命中策略质量加成（S+5/A+3/B+1/C+0/D-3）
# ---------------------------------------------------------------------------
class TestQualityBonus:
    def test_empty_hit_zero(self):
        assert quality_bonus([]) == 0

    def test_single_known_strategy(self):
        # multi_factor=80→S→+5（用中文名反查）
        assert quality_bonus(["多因子选股"]) == 5

    def test_multiple_strategies_sum(self):
        # 多因子(S+5) + 底部放量(A+3) = 8
        assert quality_bonus(["多因子选股", "底部放量"]) == 8

    def test_d_tier_strategy_negative(self):
        # 已废弃策略 D 级 → -3
        bonus = quality_bonus(["海龟突破"])  # turtle=7→D
        assert bonus == -3

    def test_unknown_strategy_uses_default_40_c_tier(self):
        """未知策略名兜底质量分 40（C 级，加成 0）。"""
        assert quality_bonus(["不存在的策略"]) == 0

    def test_known_english_key_also_works(self):
        """quality_label 直接用英文 key 查，确保英文 key 也可命中。"""
        # quality_bonus 经 name_to_key 反查，英文 key 原样查 STRATEGY_QUALITY
        assert quality_bonus(["multi_factor"]) == 5


# ---------------------------------------------------------------------------
# quality_label：最高质量分层（用于前端展示，S>A>B>C>D）
# ---------------------------------------------------------------------------
class TestQualityLabel:
    def test_empty_returns_empty(self):
        assert quality_label([]) == ""

    def test_single_strategy(self):
        assert quality_label(["multi_factor"]) == "S"

    def test_returns_best_tier(self):
        """多个策略取最高层级（S 最优）。"""
        # multi_factor=S, bottom=A → 最高 S
        assert quality_label(["multi_factor", "bottom"]) == "S"

    def test_picks_among_mixed(self):
        # turtle=D, flag=B → 最高 B
        assert quality_label(["turtle", "flag"]) == "B"

    def test_unknown_strategy_default_c(self):
        assert quality_label(["未知策略"]) == "C"


# ---------------------------------------------------------------------------
# detect_strategy_conflicts：互斥策略组合检测
# ---------------------------------------------------------------------------
class TestDetectStrategyConflicts:
    def test_no_conflict_empty(self):
        assert detect_strategy_conflicts([]) == []

    def test_no_conflict_compatible(self):
        assert detect_strategy_conflicts(["multi_factor", "bottom"]) == []

    def test_bottom_vs_flag_conflict(self):
        result = detect_strategy_conflicts(["bottom", "flag"])
        assert len(result) == 1
        assert "底部放量" in result[0]
        assert "高位旗形" in result[0]

    def test_bottom_vs_rps_conflict(self):
        result = detect_strategy_conflicts(["bottom", "rps"])
        assert len(result) == 1

    def test_pullback_vs_ma_volume_conflict(self):
        result = detect_strategy_conflicts(["pullback", "ma_volume"])
        assert len(result) == 1
        assert "缩量" in result[0]

    def test_multiple_conflicts(self):
        """同时触发多个互斥组合。"""
        result = detect_strategy_conflicts(["bottom", "flag", "rps"])
        # bottom+flag + bottom+rps = 2 个冲突
        assert len(result) == 2

    def test_unrelated_keys_no_conflict(self):
        assert detect_strategy_conflicts(["multi_factor", "auction", "lhb_follow"]) == []


# ---------------------------------------------------------------------------
# _has_hard_risk：硬性风险淘汰（PE泡沫/退市/主力净流出>5% 等）
# ---------------------------------------------------------------------------
class TestHasHardRisk:
    def test_no_risk(self):
        assert DecisionEngine._has_hard_risk([]) is False
        assert DecisionEngine._has_hard_risk(["温和上涨"]) is False

    def test_pe_bubble(self):
        assert DecisionEngine._has_hard_risk(["PE=120，估值过高"]) is True

    def test_bubble_keyword(self):
        assert DecisionEngine._has_hard_risk(["存在估值泡沫"]) is True

    def test_delisting_risk(self):
        assert DecisionEngine._has_hard_risk(["退市风险警示"]) is True

    def test_fundamental_deterioration(self):
        assert DecisionEngine._has_hard_risk(["基本面恶化"]) is True

    def test_severe_overbought(self):
        assert DecisionEngine._has_hard_risk(["严重超买"]) is True

    def test_main_force_outflow_under_threshold(self):
        """主力净流出 ≤5% 不触发硬淘汰。"""
        assert DecisionEngine._has_hard_risk(["主力资金净流出3%"]) is False

    def test_main_force_outflow_over_threshold(self):
        """主力净流出 >5% 触发硬淘汰。"""
        assert DecisionEngine._has_hard_risk(["主力资金净流出8%"]) is True

    def test_main_force_outflow_boundary_5_pct_not_triggered(self):
        """恰好 5% 不触发（>5 才触发）。"""
        assert DecisionEngine._has_hard_risk(["主力资金净流出5%"]) is False


# ---------------------------------------------------------------------------
# _filter_pool：市场板块过滤（创业板/科创板/北交所/主板）
# ---------------------------------------------------------------------------
class TestFilterPool:
    def test_empty_pool(self):
        assert DecisionEngine._filter_pool({}, [], False) == {}

    def test_no_exclusion_keeps_all(self):
        pool = {"000001": ["s1"], "300001": ["s2"], "688001": ["s3"]}
        assert DecisionEngine._filter_pool(pool, [], False) == pool

    def test_exclude_chinext(self):
        pool = {"000001": ["s1"], "300001": ["s2"], "301001": ["s3"]}
        result = DecisionEngine._filter_pool(pool, ["chinext"], False)
        assert set(result.keys()) == {"000001"}

    def test_exclude_star(self):
        pool = {"688001": ["s1"], "689001": ["s2"], "000001": ["s3"]}
        result = DecisionEngine._filter_pool(pool, ["star"], False)
        assert set(result.keys()) == {"000001"}

    def test_exclude_bse(self):
        pool = {"830001": ["s1"], "430001": ["s2"], "000001": ["s3"]}
        result = DecisionEngine._filter_pool(pool, ["bse"], False)
        assert set(result.keys()) == {"000001"}

    def test_exclude_multiple_markets(self):
        pool = {
            "000001": ["s"],   # main 保留
            "300001": ["s"],   # chinext 剔除
            "688001": ["s"],   # star 剔除
            "830001": ["s"],   # bse 剔除
        }
        result = DecisionEngine._filter_pool(pool, ["chinext", "star", "bse"], False)
        assert set(result.keys()) == {"000001"}

    def test_strategy_list_preserved(self):
        """过滤保留原策略列表，不改动命中策略。"""
        pool = {"000001": ["s1", "s2", "s3"]}
        result = DecisionEngine._filter_pool(pool, [], False)
        assert result["000001"] == ["s1", "s2", "s3"]


# ---------------------------------------------------------------------------
# _allocate_capital：ATR 风险预算 + 行业暴露约束
# ---------------------------------------------------------------------------
def _buy_item(symbol="000001", price=10.0, stop=9.0, position_pct=20.0, industry="银行"):
    """构造买入条目辅助函数。"""
    return DecisionItem(
        symbol=symbol, grade="A", price=price, stop_loss=stop,
        position_pct=position_pct, industry=industry,
    )


class TestAllocateCapital:
    def test_basic_allocation_within_risk_budget(self):
        """单只票按 ATR 风险预算分配，单笔风险=资金×1.5%。"""
        capital = 100000.0
        items = [_buy_item(price=10.0, stop=9.0, position_pct=20.0)]  # 风险/share=1
        DecisionEngine._allocate_capital(items, capital)
        # 风险金额=1500，每 share 风险=1 → 最多 1500 股，按100整取=1500
        # 资金上限=20000/10=2000 股，取小=1500
        assert items[0].shares == 1500
        assert items[0].capital == pytest.approx(15000, abs=1)

    def test_total_capital_capped_at_80_pct(self):
        """总占用不超过资金 80%（留现金）。"""
        capital = 100000.0
        # 5 只票每只要 20% = 100%，应被 80% 上限截断
        items = [
            _buy_item(symbol=f"00000{i}", price=10.0, stop=9.5, position_pct=20.0,
                      industry=f"行业{i}")
            for i in range(5)
        ]
        DecisionEngine._allocate_capital(items, capital)
        total = sum(i.capital for i in items)
        assert total <= 80000 + 1  # 80% = 80000，容差1元（取整）

    def test_industry_concentration_cap(self):
        """单行业累计资金 ≤ 30% 上限，超限削减。"""
        capital = 100000.0
        industry_limit = 30000  # 30%
        # 同行业 4 只票，每只本应 15000，累计 60000 超 30% 上限
        items = [
            _buy_item(symbol=f"00000{i}", price=10.0, stop=9.0, position_pct=15.0,
                      industry="银行")
            for i in range(4)
        ]
        DecisionEngine._allocate_capital(items, capital)
        total_in_industry = sum(i.capital for i in items)
        assert total_in_industry <= industry_limit + 100  # 取整容差

    def test_zero_price_skipped(self):
        """price<=0 的条目跳过分配。"""
        items = [_buy_item(price=0, stop=0)]
        DecisionEngine._allocate_capital(items, 100000.0)
        assert items[0].shares == 0

    def test_stop_above_price_skipped(self):
        """止损价>=买入价（risk_per_share<=0）跳过分配。"""
        items = [_buy_item(price=10.0, stop=10.5)]  # 止损高于买价
        DecisionEngine._allocate_capital(items, 100000.0)
        assert items[0].shares == 0

    def test_shares_rounded_to_100(self):
        """A股按手交易，股数向下取整到 100。"""
        items = [_buy_item(price=10.0, stop=9.9, position_pct=20.0)]  # 风险/share=0.1
        DecisionEngine._allocate_capital(items, 100000.0)
        assert items[0].shares % 100 == 0


# ---------------------------------------------------------------------------
# _build_summary / _to_dict / _empty_result：结果组装
# ---------------------------------------------------------------------------
class TestResultAssembly:
    def test_empty_result_structure(self):
        r = DecisionEngine._empty_result(100000.0)
        assert r["buy_list"] == []
        assert r["reject_list"] == []
        assert r["summary"]["cash_ratio"] == 100
        assert r["summary"]["position_ratio"] == 0
        assert r["summary"]["capital"] == 100000.0

    def test_to_dict_contains_all_fields(self):
        item = DecisionItem(symbol="000001", name="平安银行", grade="A")
        d = DecisionEngine._to_dict(item)
        assert d["symbol"] == "000001"
        assert d["name"] == "平安银行"
        assert d["grade"] == "A"
        # 关键字段齐全
        for key in ("resonance", "hit_strategies", "score", "price",
                    "stop_loss", "position_pct", "shares", "capital"):
            assert key in d

    def test_build_summary_capital_ratios(self):
        """cash_ratio + position_ratio = 100。"""
        items = [
            DecisionItem(symbol="000001", grade="A", capital=30000, industry="银行"),
            DecisionItem(symbol="000002", grade="B", capital=20000, industry="地产"),
        ]
        s = DecisionEngine._build_summary(items, [], 100000.0)
        assert s["used_capital"] == 50000
        assert s["cash_ratio"] == 50.0
        assert s["position_ratio"] == 50.0
        assert s["buy_count"] == 2

    def test_build_summary_grade_count(self):
        items = [
            DecisionItem(symbol="1", grade="A", industry="X"),
            DecisionItem(symbol="2", grade="A", industry="X"),
            DecisionItem(symbol="3", grade="B", industry="Y"),
        ]
        s = DecisionEngine._build_summary(items, ["reject"], 100000.0)
        assert s["grade_count"]["A"] == 2
        assert s["grade_count"]["B"] == 1
        assert s["grade_count"]["C"] == 0
        assert s["reject_count"] == 1

    def test_build_summary_top_industry(self):
        items = [
            DecisionItem(symbol="1", grade="A", capital=10000, industry="银行"),
            DecisionItem(symbol="2", grade="A", capital=10000, industry="银行"),
            DecisionItem(symbol="3", grade="B", capital=10000, industry="地产"),
        ]
        s = DecisionEngine._build_summary(items, [], 100000.0)
        assert s["top_industry"] == "银行"
        assert s["industries"][0] == {"name": "银行", "count": 2}

    def test_build_summary_empty_buy_list(self):
        s = DecisionEngine._build_summary([], [], 100000.0)
        assert s["used_capital"] == 0
        assert s["cash_ratio"] == 100.0
        assert s["top_industry"] == ""
        assert s["concentration"] == 0
