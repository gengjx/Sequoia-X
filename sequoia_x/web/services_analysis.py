"""Web 服务 - 个股与组合分析（WebServices mixin 组件）。

由 :class:`sequoia_x.web.services.WebServices` 多重继承组合，不单独实例化；
方法通过 ``self.engine`` / ``self.settings`` / ``self._task_store`` 等访问门面状态。
"""

from __future__ import annotations

import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor

from sequoia_x.analysis.decision import DecisionEngine
from sequoia_x.analysis.market import MarketAnalyzer
from sequoia_x.analysis.stock_analysis import StockAnalyzer
from sequoia_x.strategy.registry import (
    ACTIVE_STRATEGY_KEYS,
    RETIRED_STRATEGY_KEYS,
    STRATEGY_REGISTRY,
)

logger = logging.getLogger(__name__)


class AnalysisMixin:
    """个股与组合分析（WebServices 的 mixin 组件）。"""

    def analyze_stock(self, symbol: str) -> dict:
        """同步分析个股，返回六层结构化决策报告（当日有效缓存）。

        A股T+1：日K收盘后技术指标/财报日内不变，分析结果当日有效。
        缓存 key 含日期，自然跨日失效；盘中实时价变化不影响决策逻辑。
        """
        data_date = self._data_date()
        cached = self._stock_result_cache.get(symbol)
        if cached and cached[0].get("data_date") == data_date:
            return cached[1]
        result = self._get_stock_analyzer().analyze(symbol)
        self._stock_result_cache[symbol] = ({"data_date": data_date}, result)
        return result

    def analyze_portfolio(self, symbols: list[str]) -> dict:
        """批量分析多只股票，返回 {stocks, summary}（持仓体检）。

        - 预热东财全市场快照缓存（一次请求），避免并发刷新竞争。
        - ThreadPoolExecutor 并行分析（baostock 财报内部已加锁串行化）。
        - 聚合组合体检摘要：评分分布 / 行业集中度 / 加权估值 / 风险预警。
        """
        from concurrent.futures import as_completed
        symbols = [str(x).strip() for x in symbols if str(x).strip()]
        if not symbols:
            return {"stocks": [], "summary": {}}

        # 预热快照缓存
        self._get_stock_analyzer()._refresh_quote_cache()

        results: list[dict] = []
        workers = min(4, len(symbols))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.analyze_stock, sym): sym for sym in symbols}
            for fut in as_completed(futs):
                sym = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({"symbol": sym, "error": str(e), "recommendation": {"score": 0}})

        results.sort(key=lambda x: x.get("recommendation", {}).get("score", 0), reverse=True)
        summary = self._portfolio_summary(results)
        logging.getLogger(__name__).info(f"批量分析完成：{len(symbols)} 只，平均评分 {summary.get('avg_score')}")
        return {"stocks": results, "summary": summary}

    def _portfolio_summary(self, results: list[dict]) -> dict:
        """聚合组合体检摘要。"""
        valid = [r for r in results if not r.get("error")]
        if not valid:
            return {"count": 0}
        scores = [r.get("recommendation", {}).get("score", 0) for r in valid]
        avg = round(sum(scores) / len(scores)) if scores else 0
        strong = sum(1 for s in scores if s >= 65)
        neutral = sum(1 for s in scores if 50 <= s < 65)
        weak = sum(1 for s in scores if s < 50)

        # 行业集中度
        industry_map: dict[str, int] = {}
        for r in valid:
            ind = r.get("fundamental", {}).get("industry") or "未知"
            industry_map[ind] = industry_map.get(ind, 0) + 1
        industries = sorted(industry_map.items(), key=lambda x: -x[1])[:5]
        top_industry = industries[0] if industries else ("", 0)
        concentration = round(top_industry[1] / len(valid) * 100) if valid else 0

        # 市值加权 PE/PB
        total_cap = 0.0
        w_pe = 0.0
        w_pb = 0.0
        for r in valid:
            f = r.get("fundamental", {})
            cap = f.get("market_cap") or 0
            pe = f.get("pe")
            pb = f.get("pb")
            if cap and pe and pe > 0:
                w_pe += pe * cap
                total_cap += cap
            if cap and pb and pb > 0:
                w_pb += pb * cap
        avg_pe = round(w_pe / total_cap, 1) if total_cap else None
        avg_pb = round(w_pb / total_cap, 2) if total_cap else None

        # 风险预警（评分<50 或 有多条风险提示）
        alerts = []
        for r in valid:
            rec = r.get("recommendation", {})
            risks = r.get("risks", [])
            if rec.get("score", 100) < 50 or len([x for x in risks if "暂无" not in x]) >= 3:
                alerts.append({
                    "symbol": r.get("symbol"),
                    "name": r.get("name"),
                    "score": rec.get("score", 0),
                    "action": rec.get("action", ""),
                    "top_risk": next((x for x in risks if "暂无" not in x), ""),
                })

        return {
            "count": len(valid),
            "avg_score": avg,
            "distribution": {"strong": strong, "neutral": neutral, "weak": weak},
            "industries": [{"name": n, "count": c} for n, c in industries],
            "top_industry": top_industry[0],
            "concentration": concentration,
            "weighted_pe": avg_pe,
            "weighted_pb": avg_pb,
            "alerts": alerts,
        }

    def generate_decision(
        self, strategy_keys: list[str] | None = None,
        capital: float = 100000.0, min_score: int = 50,
        exclude_markets: list[str] | None = None, exclude_st: bool = False,
        max_candidates: int | None = None,
        include_auction: bool = False,
    ) -> dict:
        """交易决策中枢：多策略选股 → 质量过滤 → 共振定级 → 仓位分配。

        并行运行选定策略，汇总去重后跑个股分析，输出可执行买卖清单（10分钟缓存）。
        """
        import time
        now = time.time()
        # 缓存 key 包含策略+过滤参数，避免不同条件复用错误结果
        # 缓存key含全部决策参数：不同分析池上限(60/120/不限)选出的股票集合不同，
        # 必须独立缓存，否则选60分析后选120会错误命中返回60的结果
        mc = max_candidates or 60
        cache_key = f"{','.join(sorted(strategy_keys or []))}|{capital}|{min_score}|{','.join(sorted(exclude_markets or []))}|{exclude_st}|mc{mc}|au{int(include_auction)}"
        data_date = self._data_date()
        # 快速检查缓存（无锁）
        cached = self._decision_cache.get(cache_key)
        if cached and cached[0].get("data_date") == data_date:
            return cached[1]
        # Double-checked locking：并发请求只跑一次，后续命中缓存
        with self._decision_lock:
            # 锁内二次检查（前一个请求可能已完成写入缓存）
            cached = self._decision_cache.get(cache_key)
            if cached and cached[0].get("data_date") == data_date:
                return cached[1]
            if strategy_keys is None:
                # 默认：多因子为核心（5.3年回测年化+22.6%，唯一穿越牛熊）
                # 仅保留 core+active 策略（multi_factor/bottom/flag），
                # demoted(ma_volume/pullback/volume_extreme) 弱于基准，退出默认决策池。
                strategy_keys = list(ACTIVE_STRATEGY_KEYS)

            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Step1: 并行运行选定策略（data_date 来自上方缓存检查，闭包复用）
            # 每策略独立读K线+结果缓存；共享DF实测在3M行下groupby慢+内存复制开销，无净收益
            strategy_results: dict[str, list[str]] = {}
            strategy_health: dict[str, object] = {}

            def _run_strategy(key: str) -> tuple[str, list[str]]:
                # 缓存命中：同一data_date内策略结果不变（K线数据没变，选股结果一致）
                cached = self._result_cache.get(key)
                if cached and cached[0] == data_date:
                    return key, cached[1]
                cls = STRATEGY_REGISTRY.get(key)
                if not cls:
                    return key, []
                try:
                    strat = cls(engine=self.engine, settings=self.settings)
                    results = strat.run()
                    if hasattr(strat, "last_health") and strat.last_health:
                        strategy_health[key] = strat.last_health
                    self._result_cache[key] = (data_date, results)
                    return key, results
                except Exception as e:
                    logging.getLogger(__name__).warning(f"策略 {key} 运行失败：{e!r}")
                    return key, [] 

            workers = min(4, len(strategy_keys) or 1)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_run_strategy, k) for k in strategy_keys]
                for fut in as_completed(futs):
                    k, syms = fut.result()
                    strategy_results[k] = syms

            # Step2-4: 决策引擎融合
            analyzer = self._get_stock_analyzer()
            engine = DecisionEngine(self.engine, self.settings)
            # 大盘状态择时：复用已缓存的market报告；无缓存则实时分析并自动缓存（绑定data_date）
            # 大盘分析需拉东财496板块数据~30s，决策内缓存后同data_date复用秒级
            def _market_fn():
                report = self._market_report_cache.get(data_date)
                if report:
                    return report
                report = self._get_analyzer().analyze()
                if report and report.get("date"):
                    self._market_report_cache[report["date"]] = report
                    logging.getLogger(__name__).info("大盘分析已完成并缓存，后续决策复用")
                return report
            # 竞价→决策联动：纳入今日竞价A级票作为盘中候选（竞价定方向+技术面确认）
            if include_auction:
                auction_syms = self._get_auction_a_grade()
                if auction_syms:
                    strategy_results["auction"] = auction_syms
                    logging.getLogger(__name__).info(f"竞价联动：注入{len(auction_syms)}只A级票到决策池")

            result = engine.generate(
                strategy_results=strategy_results,
                analyze_fn=self.analyze_stock,
                capital=capital,
                min_score=min_score,
                exclude_markets=exclude_markets,
                exclude_st=exclude_st,
                market_fn=_market_fn,
                max_candidates=max_candidates or 60,
            )
            result["strategies_run"] = {
                k: len(v) for k, v in strategy_results.items()
            }
            _health = strategy_health.get("multi_factor")
            if _health:
                result["factor_health"] = _health.to_summary()
                result["degraded"] = _health.to_summary().get("is_degraded", False)
            else:
                result["factor_health"] = None
                result["degraded"] = False
            self._decision_cache[cache_key] = ({"data_date": data_date}, result)
        logging.getLogger(__name__).info('决策缓存写入 key=' + cache_key[:40])
        logging.getLogger(__name__).info(
            f"决策生成完成：候选{result['pool_size']}只 → 买入{result['summary']['buy_count']}只"
        )
        return result

    def _get_analyzer(self) -> MarketAnalyzer:
        if self._market_analyzer is None:
            self._market_analyzer = MarketAnalyzer(self.settings)
        return self._market_analyzer

    def _get_stock_analyzer(self) -> StockAnalyzer:
        if self._stock_analyzer is None:
            self._stock_analyzer = StockAnalyzer(self.settings)
        return self._stock_analyzer

    def _data_date(self) -> str:
        """获取数据库K线最新交易日，作为缓存版本依据。

        K线是决策/分析的真正数据源，盘后同步数据后 MAX(date) 变化，
        以此为缓存版本可自动失效（盘后刷新），无需绑定自然日。
        结果缓存60s（MAX(date)扫3M行~80ms，但数据同步是低频操作无需每次查）。
        """
        import time
        now = time.time()
        cached = getattr(self, "_data_date_cache", None)
        if cached and now - cached[0] < 60:
            return cached[1]
        try:
            with sqlite3.connect(self.settings.db_path) as conn:
                r = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()
                val = r[0] if r and r[0] else ""
            self._data_date_cache = (now, val)
            return val
        except Exception:
            return ""
