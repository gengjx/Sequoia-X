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
        exclude_markets: list[str] | None = None,
        exclude_st: bool = False,
        market_fn: Callable[[], dict] | None = None,
        max_industry_pct: float = 0.30,
    ) -> dict:
        """max_industry_pct: 单行业最大资金占比（默认30%），超出降级观望。"""
        """生成买卖决策清单。

        Args:
            strategy_results: {策略key: [股票代码]}，来自多策略并行运行结果
            analyze_fn: 个股分析函数（StockAnalyzer.analyze 的引用），复用其 5min 缓存
            capital: 总资金（元）
            min_score: 综合评分下限，低于此值淘汰
            exclude_markets: 剔除的市场板块，如 ['chinext','star','bse']
            exclude_st: 是否剔除 ST/*ST 股票
            market_fn: 大盘分析函数（返回 report dict），用于市场状态择时

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

        # 市场板块 + ST 过滤（在分析前剔除，节省财报采集时间）
        if exclude_markets or exclude_st:
            pool = self._filter_pool(pool, exclude_markets or [], exclude_st)

        if not pool:
            return self._empty_result(capital)

        # 市场状态择时层：获取大盘评分，判定牛/震荡/熊
        market_state, market_score, market_label = self._get_market_state(market_fn)
        logger.info(f"市场状态：{market_state}（评分{market_score} {market_label}）")

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
                # ST 过滤（分析后拿到股票名称才能判断）
                if exclude_st and "ST" in report.get("name", "").upper():
                    items.append(DecisionItem(
                        symbol=sym, name=report.get("name", sym), grade="淘汰",
                        reject_reason="ST/*ST股票已剔除",
                        score=report.get("recommendation", {}).get("score", 0),
                    ))
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
                    self._grade_item(item, capital, market_state)

                items.append(item)
            except Exception as e:
                logger.warning(f"决策分析 {sym} 失败：{e!r}")

        # ── Step 3: 定级后排序（评级→共振→评分）──
        grade_order = {"A": 0, "B": 1, "C": 2, "观望": 3, "淘汰": 4}
        items.sort(key=lambda x: (grade_order.get(x.grade, 9), -x.resonance, -x.score))

        buy_list = [i for i in items if i.grade in ("A", "B", "C")]
        reject_list = [i for i in items if i.grade == "淘汰"]

        # ── Step 4: 资金分配（仓位不超过总资金）──
        # 市场状态仓位缩放：熊市×0.5、震荡×0.8、牛市×1.0
        position_scale = {"bull": 1.0, "neutral": 0.8, "bear": 0.5}.get(market_state, 0.8)
        scaled_capital = capital * position_scale
        self._allocate_capital(buy_list, scaled_capital, max_industry_pct)

        # 仓位 0%（ATR 约束下无法建整手）的降级为观望
        cannot_buy = [i for i in buy_list if i.shares == 0]
        for i in cannot_buy:
            i.grade = "观望"
            i.action = "观望"
            if not i.reason:
                i.reason = f"评分{i.score}但当前价位风险预算下无法建整手，建议观望等回调"
        buy_list = [i for i in buy_list if i.shares > 0]
        reject_list = cannot_buy + reject_list

        return {
            "buy_list": [self._to_dict(i) for i in buy_list],
            "reject_list": [self._to_dict(i) for i in reject_list],
            "summary": self._build_summary(buy_list, reject_list, capital),
            "market_state": {"state": market_state, "score": market_score,
                             "label": market_label, "position_scale": position_scale},
            "strategy_count": len(strategy_results),
            "pool_size": len(pool),
        }

    # ------------------------------------------------------------------
    # 市场板块过滤
    # ------------------------------------------------------------------
    @staticmethod
    def _filter_pool(
        pool: dict[str, list[str]], exclude_markets: list[str], exclude_st: bool,
    ) -> dict[str, list[str]]:
        """按市场板块过滤候选池（ST 过滤延迟到分析阶段，因需股票名称）。

        exclude_markets 支持值：chinext(创业板300/301) star(科创板688/689) bse(北交所8/4)
        """
        def _market_of(symbol: str) -> str:
            if symbol.startswith(("300", "301")):
                return "chinext"
            if symbol.startswith(("688", "689")):
                return "star"
            if symbol.startswith(("8", "4")):
                return "bse"
            return "main"

        filtered = {}
        excluded = 0
        for sym, strats in pool.items():
            if _market_of(sym) in exclude_markets:
                excluded += 1
                continue
            filtered[sym] = strats
        if excluded:
            logger.info(f"市场板块过滤：剔除 {excluded} 只（{','.join(exclude_markets)}）")
        return filtered

    # ------------------------------------------------------------------
    # 定级逻辑
    # ------------------------------------------------------------------
    def _grade_item(self, item: DecisionItem, capital: float, market_state: str = "neutral") -> None:
        """数据驱动定级矩阵（基于共振回测实测结论）。

        回测事实（497只×10天持有期）：
          - 单策略(共振1): +1.22% 胜率47% ← 最稳健
          - 2策略共振: +1.08% 胜率44%
          - 3+共振: -2.07% 胜率27% ← 过热见顶，反而亏损
        故：3+共振降级为风险预警（非重仓），单策略高评分提升。

        market_state: bull/neutral/bear，影响评分门槛和仓位系数。
        """
        r, s = item.resonance, item.score
        # 市场状态调整评分门槛（牛市放宽、熊市收紧）
        threshold_map = {"bull": 60, "neutral": 50, "bear": 55}
        hi_threshold = {"bull": 72, "neutral": 65, "bear": 70}
        t_low = threshold_map.get(market_state, 50)
        t_hi = hi_threshold.get(market_state, 65)

        # 3+共振：过热风险，降级处理
        if r >= 3:
            item.grade = "淘汰"
            item.reject_reason = f"{r}策略共振→过热见顶风险（回测{r}共振10天-2.07%胜率27%）"
            return

        # 单策略高评分：回测最稳健，提升评级
        if r == 1 and s >= t_hi:
            item.grade = "A"
            item.reason = f"单策略+高评分{s}（回测单策略最稳健+1.22%），数据支持重点参与"
            item.position_pct = self.GRADE_POSITION["A"][0]
        elif r == 2 and s >= t_hi:
            item.grade = "B"
            item.reason = f"{r}策略共振+高评分{s}，趋势确认"
            item.position_pct = self.GRADE_POSITION["B"][0]
        elif s >= t_hi:
            item.grade = "B"
            item.reason = f"评分{s}达标，可逢低建仓"
            item.position_pct = self.GRADE_POSITION["B"][0]
        elif s >= t_low:
            item.grade = "C"
            item.reason = f"评分{s}中性，小仓试探"
            item.position_pct = self.GRADE_POSITION["C"][0]
        else:
            item.grade = "淘汰"
            item.reject_reason = f"评分{s}<{t_low}（{market_state}市场门槛）"
            return
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
    def _allocate_capital(buy_list: list[DecisionItem], capital: float,
                          max_industry_pct: float = 0.30) -> None:
        """按 ATR 风险预算分配仓位（单笔风险 1.5%），总量不超过总资金。

        行业暴露控制：单行业累计资金 ≤ max_industry_pct × capital（默认30%），
        超出则削减该票仓位；削减至0的由调用方降级为观望。A股单行业政策/黑天鹅
        风险集中，硬约束避免组合被单一板块拖垮。
        """
        used_capital = 0.0
        max_capital = capital * 0.8  # 最高占用 80%，留现金
        industry_capital: dict[str, float] = {}
        industry_limit = capital * max_industry_pct
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
                # 行业暴露硬约束：单行业 ≤ max_industry_pct
                ind = item.industry or "其他"
                ind_used = industry_capital.get(ind, 0)
                if ind_used + shares * item.price > industry_limit:
                    allowed = max(0, industry_limit - ind_used)
                    shares = int(allowed / item.price / 100) * 100
                    if shares == 0:
                        item.reason = (
                            f"{ind}行业已占用{ind_used / capital * 100:.0f}%≥"
                            f"{max_industry_pct * 100:.0f}%上限，降级观望分散风险"
                        )
                item.shares = shares
                item.capital = round(shares * item.price, 0)
                used_capital += item.capital
                industry_capital[ind] = ind_used + item.capital
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
    def _get_market_state(market_fn) -> tuple[str, int, str]:
        """从大盘分析报告提取市场状态（bull/neutral/bear）。

        基于 market.py 的 signal_score（0-100）：
          >=55 bull（偏暖/强势）放宽门槛、满仓
          45-54 neutral（中性）标准门槛、仓位×0.8
          <45 bear（偏冷/弱势）收紧门槛、仓位×0.5
        """
        if market_fn is None:
            return "neutral", 50, "未接入大盘数据"
        try:
            report = market_fn()
            score = report.get("overview", {}).get("signal_score", 50)
            label = report.get("overview", {}).get("signal_label", "")
            if score >= 55:
                state = "bull"
            elif score >= 45:
                state = "neutral"
            else:
                state = "bear"
            return state, score, label
        except Exception as e:
            logger.warning(f"大盘状态获取失败，默认neutral：{e!r}")
            return "neutral", 50, f"获取失败({e})"

    @staticmethod
    def _get_analyzer(analyze_fn) -> object | None:
        """从闭包/绑定对象提取 StockAnalyzer 实例（用于预热快照）。"""
        obj = getattr(analyze_fn, "__self__", None)
        return obj
