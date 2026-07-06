"""交易决策中枢：融合多策略选股 + 个股六层评分，输出可执行的买卖清单。

决策四步漏斗：
  1. 多策略汇总去重 → 候选池（统计共振度）
  2. 质量过滤 → 六层评分过滤淘汰劣质标的
  3. 共振定级 → 决策矩阵（共振度 × 综合评分）→ A/B/C 档
  4. 资金分配 → ATR 风险预算按优先级分配仓位

输出：买入清单（评级/策略/评分/价格/止损/目标/仓位/股数）+ 淘汰清单 + 组合摘要
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine

logger = get_logger(__name__)


# ------------------------------------------------------------------
# 策略质量分层（基于 strategy_eval 评估引擎实测数据）
# 综合维度：夏普(风险调整) + 卡玛(回撤调整) + 选股IC(相对alpha) + 回撤控制
# ⚠️ 阶段一硬编码，阶段二改为评估引擎定期重评估自动刷新（避免过拟合）
# ------------------------------------------------------------------
# 策略质量默认权重（DB无数据时的兜底值，基于一次评估快照）
# 运行策略评估后会自动刷新写入DB，决策引擎启动时从DB加载最新值
_DEFAULT_STRATEGY_QUALITY: dict[str, int] = {
    "flag": 85,        # S级 夏普0.68最高，唯一跑赢基准(+0.87% alpha)
    "pullback": 78,    # A级 IC0.061最强，回撤-19.99%最小，风控最优
    "bottom": 55,      # B级 样本不足保守中性（条件极严格）
    "dragon": 50,      # B级 不支持向量化回测，板块维度有独立价值
    "ma_volume": 48,   # B级 IC0.034正向选股力，但夏普偏低
    "rps": 40,         # C级 夏普0.39，但IC-0.054追高风险
    "turtle": 28,      # C级 夏普-0.37，IC-0.009，选股alpha不足
    "shakeout": 18,    # D级 夏普-0.59，年化-29.8%
    "limit_down": 10,  # D级 夏普-1.03，年化-34.8%（最差）
}

# 运行时动态权重（DecisionEngine.__init__ 从DB加载，覆盖默认值）
STRATEGY_QUALITY: dict[str, int] = dict(_DEFAULT_STRATEGY_QUALITY)

# 质量分 → 分层 → 定级加成
_TIER_BONUS = {"S": 5, "A": 3, "B": 1, "C": 0, "D": -3}


def quality_tier(score: int) -> str:
    """质量分 → S/A/B/C/D 分层。"""
    if score >= 80:
        return "S"
    if score >= 65:
        return "A"
    if score >= 45:
        return "B"
    if score >= 30:
        return "C"
    return "D"


def quality_bonus(hit_strategies: list[str]) -> float:
    """命中策略的质量加成分（用于定级矩阵）。

    S级策略命中+5，A级+3，B级+1，C级0，D级-3。
    让决策中枢区分"命中好策略"与"命中差策略"。
    """
    return sum(
        _TIER_BONUS.get(quality_tier(STRATEGY_QUALITY.get(s, 40)), 0)
        for s in hit_strategies
    )


def quality_label(hit_strategies: list[str]) -> str:
    """命中策略中最高的质量分层（用于前端展示）。"""
    if not hit_strategies:
        return ""
    tiers = [quality_tier(STRATEGY_QUALITY.get(s, 40)) for s in hit_strategies]
    order = ["S", "A", "B", "C", "D"]
    return min(tiers, key=lambda t: order.index(t))



@dataclass
class DecisionItem:
    """单只股票的决策条目。"""
    symbol: str
    name: str = ""
    grade: str = ""          # A / B / C / 淘汰
    resonance: int = 0       # 命中策略数
    hit_strategies: list[str] = field(default_factory=list)  # 命中策略中文名
    strategy_quality: float = 0.0   # 命中策略平均质量分(0-100)
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
        self._load_weights()

    def _load_weights(self) -> None:
        """从DB加载最新策略质量权重（覆盖默认值），DB空则用默认兜底。

        策略评估引擎每次运行会写回DB，使决策中枢自适应最新市场数据。
        """
        global STRATEGY_QUALITY
        try:
            db_weights = self.engine.load_strategy_weights()
            if db_weights:
                for key, w in db_weights.items():
                    STRATEGY_QUALITY[key] = w["quality_score"]
                logger.info(
                    f"策略权重已从DB加载（{len(db_weights)}个策略，"
                    f"最近更新：{next(iter(db_weights.values())).get('updated_at', '?')}）"
                )
        except Exception as e:
            logger.warning(f"加载策略权重失败，使用默认值：{e!r}")

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

        # 分层截断候选池（对齐回测价值，非纯共振度排序）
        # 回测事实：单策略+1.22%/47%最优，2策略+1.08%/44%，3+策略-2.07%/27%过热见顶
        # 故：3+共振直接淘汰 → 剩余按动量+共振bonus预筛
        pool_pre = Counter(len(v) for v in pool.values())  # 截断前分布
        pool = self._truncate_pool(pool, max_candidates)
        pool_post = Counter(len(v) for v in pool.values())  # 截断后分布

        # 预热东财快照（analyze_fn 内部会刷新，但预热后并发无竞争）
        analyzer = self._get_analyzer(analyze_fn)
        if analyzer:
            analyzer._refresh_quote_cache()

        # 批量预采财报：一次 login 集中采完缺财报的票，避免分析时逐只串行握手
        # baostock 季报是冷启动主要瓶颈（每只~10-20s），预采后个股分析财报读库秒级
        prefetch = {"missing": 0, "fetched": 0, "skipped": 0}
        if analyzer and hasattr(analyzer, "batch_prefetch_finance"):
            prefetch = analyzer.batch_prefetch_finance(list(pool.keys()))

        # ── Step 2: 并行个股分析（网络密集，串行→并行提速）──
        # 财报已预采落库，个股分析的主要耗时是东财网络请求（真实价/龙虎榜），可安全并发。
        # 过滤+定级（CPU密集）保持串行，逻辑清晰且无并发风险。
        import time as _t
        _t0 = _t.time()
        from concurrent.futures import ThreadPoolExecutor, as_completed
        syms_list = list(pool.keys())
        reports: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(syms_list))) as ex:
            futs = {ex.submit(analyze_fn, sym): sym for sym in syms_list}
            for fut in as_completed(futs):
                sym = futs[fut]
                try:
                    rep = fut.result()
                    if rep and not rep.get("error"):
                        reports[sym] = rep
                except Exception as e:
                    logger.warning(f"决策分析 {sym} 失败：{e!r}")
        logger.info(f"个股分析并行完成：{len(reports)}/{len(syms_list)} 只，耗时{_t.time() - _t0:.1f}s")

        # 串行：过滤 + 定级（毫秒级）
        items: list[DecisionItem] = []
        for sym, strat_names in pool.items():
            report = reports.get(sym)
            if not report:
                continue
            try:
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
                    strategy_quality=round(sum(STRATEGY_QUALITY.get(x, 40) for x in strat_names) / max(len(strat_names), 1), 1),
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
                logger.warning(f"决策定级 {sym} 失败：{e!r}")

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
            "pool_funnel": {
                "total_before": sum(pool_pre.values()),
                "total_after": sum(pool_post.values()),
                "resonance_before": {str(k): pool_pre.get(k, 0) for k in (1, 2, 3)},
                "resonance_after": {str(k): pool_post.get(k, 0) for k in (1, 2, 3)},
                "r3_dropped": pool_pre.get(3, 0),
                "r1_kept": pool_post.get(1, 0),
                "r2_kept": pool_post.get(2, 0),
                "max_candidates": max_candidates,
            },
            "finance_prefetch": prefetch,
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
        """数据驱动定级矩阵（共振度 × 策略质量 × 综合评分）。

        回测事实（497只×30月）：
          - 共振度：3+共振过热见顶（IC-0.0316有效负向），单策略最稳健
          - 策略质量：高位旗形夏普0.68(唯一正超额) >> 上升趋势跌停-1.03
        故：effective_score = 综合评分 + 策略质量加成，让决策中枢"识货"。

        market_state: bull/neutral/bear，影响评分门槛和仓位系数。
        """
        r = item.resonance
        # 策略质量加成：命中S级策略+5，D级-3
        q_bonus = quality_bonus(item.hit_strategies)
        es = item.score + q_bonus  # effective_score 有效评分

        # 市场状态调整评分门槛（牛市放宽、熊市收紧）
        threshold_map = {"bull": 60, "neutral": 50, "bear": 55}
        hi_threshold = {"bull": 72, "neutral": 65, "bear": 70}
        t_low = threshold_map.get(market_state, 50)
        t_hi = hi_threshold.get(market_state, 65)

        q_str = f"（质量加成{q_bonus:+d}，有效{es}）" if q_bonus else ""

        # 3+共振：过热风险，降级处理
        if r >= 3:
            item.grade = "淘汰"
            item.reject_reason = f"{r}策略共振→过热见顶风险（回测共振因子IC-0.0316有效负向）"
            return

        # 单策略高质量+高评分：回测最优组合，提升评级
        if r == 1 and es >= t_hi:
            item.grade = "A"
            item.reason = f"单策略+有效评分{es}{q_str}，重点参与"
            item.position_pct = self.GRADE_POSITION["A"][0]
        elif r == 2 and es >= t_hi:
            item.grade = "B"
            item.reason = f"{r}策略共振+有效评分{es}{q_str}，趋势确认"
            item.position_pct = self.GRADE_POSITION["B"][0]
        elif es >= t_hi:
            item.grade = "B"
            item.reason = f"有效评分{es}{q_str}达标，可逢低建仓"
            item.position_pct = self.GRADE_POSITION["B"][0]
        elif es >= t_low:
            item.grade = "C"
            item.reason = f"有效评分{es}{q_str}中性，小仓试探"
            item.position_pct = self.GRADE_POSITION["C"][0]
        else:
            item.grade = "淘汰"
            item.reject_reason = f"有效评分{es}<{t_low}（{market_state}市场门槛）{q_str}"
            return
        item.action = {"A": "重点买入", "B": "逢低建仓", "C": "小仓试探/观望"}.get(item.grade, "")

    def _truncate_pool(self, pool: dict[str, list[str]], max_candidates: int) -> dict[str, list[str]]:
        """截断候选池（数据驱动，对齐回测价值）。

        回测事实（497只×10天持有期）：
          - 单策略(共振1): +1.22% 胜率47% ← 收益&胜率双优
          - 2策略共振: +1.08% 胜率44%
          - 3+共振: -2.07% 胜率27% ← 过热见顶，反而亏损

        截断策略（让市场动量来选，非人为设定共振档优先级）：
          1. 3+共振直接淘汰（回测亏损，省财报采集）
          2. 剩余票按"动量分位 + 共振bonus"统一排序取前N
             - 动量是趋势跟随核心因子，决定谁先进分析池
             - 共振每多1个策略 +10分（多策略确认bonus，但不足以压制高动量票）
             - 避免共振2占满名额把回测最优的单策略高动量票挤出
        """
        if len(pool) <= max_candidates:
            return pool

        candidates = {k: v for k, v in pool.items() if len(v) < 3}
        dropped = len(pool) - len(candidates)
        if dropped:
            logger.info(f"分层截断：剔除 {dropped} 只3+共振（过热见顶，回测-2.07%）")

        if len(candidates) <= max_candidates:
            return candidates

        kept = self._top_by_momentum_bonus(candidates, max_candidates)
        kept_res = Counter(len(v) for v in kept.values())
        logger.info(
            f"候选池截断：{len(pool)} → {len(kept)} "
            f"（动量+共振预筛：单策略{kept_res.get(1, 0)} 共振2+{sum(v for k, v in kept_res.items() if k >= 2)}）"
        )
        return kept

    def _top_by_momentum_bonus(self, pool: dict[str, list[str]], n: int) -> dict[str, list[str]]:
        """按"动量分位(0-100) + 策略质量bonus"统一排序取前n。

        纯量价计算不依赖财报，毫秒级。动量分位用近20日涨幅在全市场排名；
        策略质量bonus：高质量策略(>40分)正加权，低质量(<40分)负加权，
        让高位旗形/缩量回踩等优质策略票更容易进分析池，
        淘汰涨停洗盘/上升趋势跌停等劣质策略票（回测夏普-0.59/-1.03）。
        """
        import sqlite3
        from collections import defaultdict
        try:
            with sqlite3.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, close FROM stock_daily WHERE date >= "
                    "(SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT 1 OFFSET 19) "
                    "ORDER BY symbol, date",
                ).fetchall()
            by_sym = defaultdict(list)
            for sym, close in rows:
                by_sym[sym].append(close)
            all_mom = {}
            for sym, closes in by_sym.items():
                if len(closes) >= 2 and closes[0]:
                    all_mom[sym] = (closes[-1] / closes[0] - 1) * 100
            import numpy as np
            vals = np.array(sorted(all_mom.values()))

            def _pctile(m):
                if not len(vals):
                    return 50.0
                return float((vals <= m).sum() / len(vals) * 100)

            scored = {}
            for sym, strats in pool.items():
                # 策略质量bonus：(质量分-40)/3，高质量正加权，低质量负加权
                q_bonus = sum((STRATEGY_QUALITY.get(st, 40) - 40) / 3 for st in strats)
                scored[sym] = _pctile(all_mom.get(sym, 0)) + q_bonus
            ranked = sorted(pool.keys(), key=lambda x: -scored.get(x, -999))[:n]
            return {k: pool[k] for k in ranked}
        except Exception as e:
            logger.warning(f"动量预筛失败，退化为随意取前{n}：{e!r}")
            keys = list(pool.keys())[:n]
            return {k: pool[k] for k in keys}

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
            "strategy_quality": item.strategy_quality,
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
        """从闭包/绑定对象提取 StockAnalyzer 实例（用于预热快照）。

        兼容两种传入方式：
          - StockAnalyzer.analyze（绑定方法）→ __self__ 即 analyzer
          - services.analyze_stock（带缓存包装）→ __self__ 是 WebServices，
            需取其底层 _stock_analyzer
        """
        obj = getattr(analyze_fn, "__self__", None)
        if obj is None:
            return None
        if hasattr(obj, "_get_stock_analyzer"):
            return obj._get_stock_analyzer()
        if hasattr(obj, "_refresh_quote_cache"):
            return obj
        return None
