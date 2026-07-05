"""交易决策中枢：融合多策略选股 + 个股六层评分，输出可执行的买卖清单。

决策四步漏斗：
  1. 多策略汇总去重 → 候选池（统计共振度）
  2. 质量过滤 → 六层评分过滤淘汰劣质标的
  3. 共振定级 → 决策矩阵（共振度 × 综合评分）→ A/B/C 档
  4. 资金分配 → ATR 风险预算按优先级分配仓位

输出：买入清单（评级/策略/评分/价格/止损/目标/仓位/股数）+ 淘汰清单 + 组合摘要
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine

logger = get_logger(__name__)


@dataclass
class DecisionItem:
    """单只股票的决策条目。"""
    symbol: str
    name: str = ""
    grade: str = ""          # A / B / C / 淘汰
    resonance: int = 0       # 命中策略数
    hit_strategies: list[str] = field(default_factory=list)  # 命中策略中文名
    score: int = 0           # 个股分析综合评分
    action: str = ""
    price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    position_pct: float = 0.0    # 建议仓位占比
    shares: int = 0              # 建议股数
    capital: float = 0.0         # 占用资金
    sub_scores: dict = field(default_factory=dict)
    industry: str = ""
    risks: list[str] = field(default_factory=list)
    reason: str = ""             # 定级理由
    reject_reason: str = ""      # 淘汰原因


class DecisionEngine:
    """决策中枢：策略选股 → 质量过滤 → 共振定级 → 仓位分配。"""

    # 仓位档位（占总资金）
    GRADE_POSITION = {
        "A": (0.15, 0.20),   # A级：15-20%
        "B": (0.08, 0.10),   # B级：8-10%
        "C": (0.03, 0.05),   # C级：3-5%（试探）
    }

    def __init__(self, engine: DataEngine, settings: Settings) -> None:
        self.engine = engine
        self.settings = settings

    def generate(
        self,
        strategy_results: dict[str, list[str]],
        analyze_fn: Callable[[str], dict],
        capital: float = 100000.0,
        min_score: int = 50,
        max_candidates: int = 40,
    ) -> dict:
        """生成买卖决策清单。

        Args:
            strategy_results: {策略key: [股票代码]}，来自多策略并行运行结果
            analyze_fn: 个股分析函数（StockAnalyzer.analyze 的引用），复用其 5min 缓存
            capital: 总资金（元）
            min_score: 综合评分下限，低于此值淘汰

        Returns:
            {buy_list, watch_list, reject_list, summary}
        """
        # ── Step 1: 多策略汇总去重 ──
        from sequoia_x.web.services import STRATEGY_META
        pool: dict[str, list[str]] = {}   # {symbol: [策略key]}
        for skey, symbols in strategy_results.items():
            sname = STRATEGY_META.get(skey, {}).get("name_cn", skey)
            for sym in symbols:
                pool.setdefault(sym, []).append(sname)

        logger.info(f"决策中枢：候选池 {len(pool)} 只（来自 {len(strategy_results)} 个策略）")
        if not pool:
            return self._empty_result(capital)

        # 按共振度排序，截断候选池（共振高的优先分析，控制冷启动耗时）
        if len(pool) > max_candidates:
            sorted_syms = sorted(pool.keys(), key=lambda x: -len(pool[x]))
            truncated = {k: pool[k] for k in sorted_syms[:max_candidates]}
            logger.info(f"候选池截断：{len(pool)} → {max_candidates}（按共振度优先）")
            pool = truncated

        # 预热东财快照（analyze_fn 内部会刷新，但预热后并发无竞争）
        analyzer = self._get_analyzer(analyze_fn)
        if analyzer:
            analyzer._refresh_quote_cache()

        # ── Step 2: 并行个股分析（质量过滤）──
        items: list[DecisionItem] = []
        for sym, strat_names in pool.items():
            try:
                report = analyze_fn(sym)
                if report.get("error"):
                    continue
                rec = report.get("recommendation", {})
                score = rec.get("score", 0)
                risks = report.get("risks", [])
                risks_real = [r for r in risks if "暂无" not in r]

                item = DecisionItem(
                    symbol=sym,
                    name=report.get("name", sym),
                    resonance=len(strat_names),
                    hit_strategies=strat_names,
                    score=score,
                    action=rec.get("action", ""),
                    price=report.get("price", 0),
                    stop_loss=rec.get("stop_loss", 0),
                    target=rec.get("target", 0),
                    sub_scores=rec.get("sub_scores", {}),
                    industry=report.get("fundamental", {}).get("industry", ""),
                    risks=risks_real,
                )

                # 质量过滤
                if score < min_score:
                    item.grade = "淘汰"
                    item.reject_reason = f"综合评分 {score}<{min_score}，趋势偏弱"
                elif self._has_hard_risk(risks_real):
                    item.grade = "淘汰"
                    item.reject_reason = f"高风险信号：{risks_real[0][:30]}"
                else:
                    self._grade_item(item, capital)

                items.append(item)
            except Exception as e:
                logger.warning(f"决策分析 {sym} 失败：{e!r}")

        # ── Step 3: 定级后排序（评级→共振→评分）──
        grade_order = {"A": 0, "B": 1, "C": 2, "观望": 3, "淘汰": 4}
        items.sort(key=lambda x: (grade_order.get(x.grade, 9), -x.resonance, -x.score))

        buy_list = [i for i in items if i.grade in ("A", "B", "C")]
        reject_list = [i for i in items if i.grade == "淘汰"]

        # ── Step 4: 资金分配（仓位不超过总资金）──
        self._allocate_capital(buy_list, capital)

        # 仓位 0%（ATR 约束下无法建整手）的降级为观望
        cannot_buy = [i for i in buy_list if i.shares == 0]
        for i in cannot_buy:
            i.grade = "观望"
            i.action = "观望"
            i.reason = f"评分{i.score}但当前价位风险预算下无法建整手，建议观望等回调"
        buy_list = [i for i in buy_list if i.shares > 0]
        reject_list = cannot_buy + reject_list

        return {
            "buy_list": [self._to_dict(i) for i in buy_list],
            "reject_list": [self._to_dict(i) for i in reject_list],
            "summary": self._build_summary(buy_list, reject_list, capital),
            "strategy_count": len(strategy_results),
            "pool_size": len(pool),
        }

    # ------------------------------------------------------------------
    # 定级逻辑
    # ------------------------------------------------------------------
    def _grade_item(self, item: DecisionItem, capital: float) -> None:
        """决策矩阵定级：共振度 × 综合评分。"""
        r, s = item.resonance, item.score
        if r >= 3 and s >= 65:
            item.grade = "A"
            item.reason = f"{r}策略共振 + 高评分{s}，多维度共振·重点参与"
            item.position_pct = self.GRADE_POSITION["A"][0]
        elif r >= 2 and s >= 65:
            item.grade = "B"
            item.reason = f"{r}策略共振 + 高评分{s}，趋势确认"
            item.position_pct = self.GRADE_POSITION["B"][0]
        elif r >= 2 and s >= 50:
            item.grade = "B"
            item.reason = f"{r}策略共振 + 中性评分{ s}，逢低关注"
            item.position_pct = self.GRADE_POSITION["B"][0]
        elif s >= 65:
            item.grade = "C"
            item.reason = f"单策略 + 高评分{s}，小仓位试探"
            item.position_pct = self.GRADE_POSITION["C"][0]
        elif s >= 50:
            item.grade = "C"
            item.reason = f"单策略 + 中性评分{ s}，观望为主"
            item.position_pct = self.GRADE_POSITION["C"][0] * 0.5
        else:
            item.grade = "淘汰"
            item.reject_reason = f"评分 {s} 不足"
        item.action = {"A": "重点买入", "B": "逢低建仓", "C": "小仓试探/观望"}.get(item.grade, "")

    @staticmethod
    def _has_hard_risk(risks: list[str]) -> bool:
        """硬性风险淘汰：估值泡沫 / 主力大幅出货 / 基本面恶化。"""
        for r in risks:
            if any(k in r for k in ("PE=", "泡沫", "基本面恶化", "退市风险", "严重超买")):
                return True
            if "主力资金净流出" in r:
                # 净流出幅度 >5% 才硬淘汰
                try:
                    import re
                    m = re.search(r"([\d.]+)%", r)
                    if m and float(m.group(1)) > 5:
                        return True
                except Exception:
                    pass
        return False

    # ------------------------------------------------------------------
    # 资金分配
    # ------------------------------------------------------------------
    @staticmethod
    def _allocate_capital(buy_list: list[DecisionItem], capital: float) -> None:
        """按 ATR 风险预算分配仓位（单笔风险 1.5%），总量不超过总资金。"""
        used_capital = 0.0
        max_capital = capital * 0.8  # 最高占用 80%，留现金
        for item in buy_list:
            if used_capital >= max_capital or item.price <= 0:
                continue
            # 按档位仓位占比 × 总资金
            target_capital = item.position_pct * capital
            # ATR 风险预算约束（单笔最大亏损 = 资金 × 1.5%）
            risk_amount = capital * 0.015
            risk_per_share = item.price - item.stop_loss
            if risk_per_share > 0:
                max_shares_by_risk = int(risk_amount / risk_per_share / 100) * 100
                max_shares_by_capital = int(target_capital / item.price / 100) * 100
                shares = max(0, min(max_shares_by_risk, max_shares_by_capital))
                # 不超剩余可用资金
                if shares * item.price > max_capital - used_capital:
                    shares = int((max_capital - used_capital) / item.price / 100) * 100
                item.shares = shares
                item.capital = round(shares * item.price, 0)
                used_capital += item.capital
            item.position_pct = round(item.capital / capital * 100, 1)

    # ------------------------------------------------------------------
    # 摘要 & 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _build_summary(buy_list: list, reject_list: list, capital: float) -> dict:
        used = sum(i.capital for i in buy_list)
        grade_count = {"A": 0, "B": 0, "C": 0}
        industries: dict[str, int] = {}
        for i in buy_list:
            grade_count[i.grade] = grade_count.get(i.grade, 0) + 1
            if i.industry:
                industries[i.industry] = industries.get(i.industry, 0) + 1
        top_industries = sorted(industries.items(), key=lambda x: -x[1])[:5]
        max_conc = round(top_industries[0][1] / len(buy_list) * 100) if buy_list and top_industries else 0
        return {
            "capital": capital,
            "used_capital": round(used, 0),
            "cash_ratio": round((capital - used) / capital * 100, 1),
            "position_ratio": round(used / capital * 100, 1),
            "grade_count": grade_count,
            "buy_count": len(buy_list),
            "reject_count": len(reject_list),
            "industries": [{"name": n, "count": c} for n, c in top_industries],
            "top_industry": top_industries[0][0] if top_industries else "",
            "concentration": max_conc,
        }

    @staticmethod
    def _to_dict(item: DecisionItem) -> dict:
        return {
            "symbol": item.symbol, "name": item.name, "grade": item.grade,
            "resonance": item.resonance, "hit_strategies": item.hit_strategies,
            "score": item.score, "action": item.action, "price": item.price,
            "stop_loss": item.stop_loss, "target": item.target,
            "position_pct": item.position_pct, "shares": item.shares, "capital": item.capital,
            "sub_scores": item.sub_scores, "industry": item.industry, "risks": item.risks,
            "reason": item.reason, "reject_reason": item.reject_reason,
        }

    @staticmethod
    def _empty_result(capital: float) -> dict:
        return {
            "buy_list": [], "reject_list": [],
            "summary": {"capital": capital, "used_capital": 0, "cash_ratio": 100, "position_ratio": 0,
                        "grade_count": {"A": 0, "B": 0, "C": 0}, "buy_count": 0, "reject_count": 0},
            "strategy_count": 0, "pool_size": 0,
        }

    @staticmethod
    def _get_analyzer(analyze_fn) -> object | None:
        """从闭包/绑定对象提取 StockAnalyzer 实例（用于预热快照）。"""
        obj = getattr(analyze_fn, "__self__", None)
        return obj
