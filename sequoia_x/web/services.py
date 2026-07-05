"""Web 服务层：桥接现有 engine/settings/strategies 到 Web API。"""

import collections
import logging
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.analysis.market import MarketAnalyzer
from sequoia_x.analysis.backtest import SignalBacktester
from sequoia_x.analysis.stock_analysis import StockAnalyzer
from sequoia_x.analysis.decision import DecisionEngine
from sequoia_x.analysis.position import PositionTracker
from sequoia_x.analysis.combo_backtest import ComboBacktester, SIGNAL_FUNCS
from sequoia_x.notify.feishu import FeishuNotifier
from sequoia_x.data.engine import DataEngine
from sequoia_x.strategy.base import BaseStrategy
from sequoia_x.strategy.high_tight_flag import HighTightFlagStrategy
from sequoia_x.strategy.limit_up_shakeout import LimitUpShakeoutStrategy
from sequoia_x.strategy.ma_volume import MaVolumeStrategy
from sequoia_x.strategy.rps_breakout import RpsBreakoutStrategy
from sequoia_x.strategy.turtle_trade import TurtleTradeStrategy
from sequoia_x.strategy.uptrend_limit_down import UptrendLimitDownStrategy
from sequoia_x.strategy.shrink_pullback import ShrinkPullbackStrategy
from sequoia_x.strategy.dragon_head import DragonHeadStrategy
from sequoia_x.strategy.bottom_volume import BottomVolumeStrategy


# ---------------------------------------------------------------------------
# Strategy registry & metadata
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    cls.webhook_key: cls
    for cls in [
        MaVolumeStrategy,
        TurtleTradeStrategy,
        HighTightFlagStrategy,
        LimitUpShakeoutStrategy,
        UptrendLimitDownStrategy,
        RpsBreakoutStrategy,
        ShrinkPullbackStrategy,
        DragonHeadStrategy,
        BottomVolumeStrategy,
    ]
}

STRATEGY_META: dict[str, dict] = {
    "ma_volume": {
        "name": "MaVolume",
        "name_cn": "均线放量",
        "description": "5日均线上穿20日均线（金叉）且成交量放大1.5倍",
        "min_bars": 20,
    },
    "turtle": {
        "name": "TurtleTrade",
        "name_cn": "海龟突破",
        "description": "20日新高突破 + 成交额过亿 + 阳线防诱多",
        "min_bars": 21,
    },
    "flag": {
        "name": "HighTightFlag",
        "name_cn": "高位旗形",
        "description": "40日涨幅>60% + 10日窄幅震荡 + 缩量整理",
        "min_bars": 40,
    },
    "shakeout": {
        "name": "LimitUpShakeout",
        "name_cn": "涨停洗盘",
        "description": "昨日涨停 + 今日阴线放量 + 不破涨停支撑",
        "min_bars": 5,
    },
    "limit_down": {
        "name": "UptrendLimitDown",
        "name_cn": "上升趋势跌停",
        "description": "MA20>MA60上升趋势 + 今日跌停 + 放量",
        "min_bars": 60,
    },
    "rps": {
        "name": "RpsBreakout",
        "name_cn": "RPS相对强度",
        "description": "120日涨幅排名前10% + 接近120日新高",
        "min_bars": 120,
    },
    "pullback": {
        "name": "ShrinkPullback",
        "name_cn": "缩量回踩",
        "description": "上升趋势回踩均线支撑 + 缩量企稳，右侧低吸买点",
        "min_bars": 20,
    },
    "dragon": {
        "name": "DragonHead",
        "name_cn": "板块龙头",
        "description": "领涨板块内跑赢板块+成交过亿的强势龙头",
        "min_bars": 2,
    },
    "bottom": {
        "name": "BottomVolume",
        "name_cn": "底部放量",
        "description": "超跌15%+异动放量3倍+下影线阳线，左侧反转信号",
        "min_bars": 20,
    },
}


# ---------------------------------------------------------------------------
# Background task tracking
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class TaskRecord:
    task_id: str
    strategy_key: str
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    results: list[str] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Log ring buffer handler
# ---------------------------------------------------------------------------

class RingBufferHandler(logging.Handler):
    """自定义 logging handler，将日志存入环形缓冲区。"""

    def __init__(self, maxlen: int = 500):
        super().__init__()
        self.buffer: collections.deque[str] = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(self.format(record))


# ---------------------------------------------------------------------------
# Xueqiu code helper (reused from feishu.py)
# ---------------------------------------------------------------------------

def _to_xueqiu_code(code: str) -> str:
    if code.startswith("6"):
        return f"SH{code}"
    elif code.startswith(("4", "8")):
        return f"BJ{code}"
    return f"SZ{code}"


# ---------------------------------------------------------------------------
# WebServices
# ---------------------------------------------------------------------------

class WebServices:
    """桥接 Web 层到现有 DataEngine / Settings / strategies。"""

    def __init__(self, settings: Settings, engine: DataEngine) -> None:
        self.settings = settings
        self.engine = engine
        self._task_store: dict[str, TaskRecord] = {}
        self._result_cache: dict[str, list[str]] = {}
        self._market_report_cache: dict[str, dict] = {}
        self._market_analyzer: MarketAnalyzer | None = None
        self._stock_analyzer: StockAnalyzer | None = None
        self._stock_result_cache: dict[str, tuple[dict, dict]] = {}
        self._decision_cache: dict[str, tuple[dict, dict]] = {}
        self._decision_cache_ts: float = 0.0
        self._backtest_cache: dict | None = None
        self._position_tracker: PositionTracker | None = None
        self._executor = ThreadPoolExecutor(max_workers=2)

    # -- Strategy methods --

    def list_strategies(self) -> list[dict]:
        result = []
        for key, meta in STRATEGY_META.items():
            webhook_url = self.settings.get_webhook_url(key)
            result.append({
                "key": key,
                "name": meta["name"],
                "name_cn": meta["name_cn"],
                "description": meta["description"],
                "min_bars": meta["min_bars"],
                "webhook_key": key,
                "webhook_url": webhook_url,
                "latest_result_count": len(self._result_cache.get(key, [])),
                "last_run_at": None,
            })
        return result

    def get_strategy_class(self, key: str) -> type[BaseStrategy] | None:
        return STRATEGY_REGISTRY.get(key)

    def run_strategy_async(self, key: str) -> str:
        task_id = uuid.uuid4().hex[:8]
        record = TaskRecord(task_id=task_id, strategy_key=key)
        self._task_store[task_id] = record
        self._executor.submit(self._run_strategy_task, task_id, key)
        return task_id

    def _run_strategy_task(self, task_id: str, key: str) -> None:
        record = self._task_store[task_id]
        record.status = TaskStatus.RUNNING
        record.started_at = datetime.now()
        try:
            cls = STRATEGY_REGISTRY[key]
            strategy = cls(engine=self.engine, settings=self.settings)
            results = strategy.run()
            record.results = results
            record.status = TaskStatus.DONE
            self._result_cache[key] = results
        except Exception as e:
            record.status = TaskStatus.ERROR
            record.error = str(e)
        finally:
            record.finished_at = datetime.now()

    def get_task_status(self, task_id: str) -> TaskRecord | None:
        return self._task_store.get(task_id)

    def get_cached_results(self, key: str) -> list[str]:
        return self._result_cache.get(key, [])

    # -- Data sync methods --

    def sync_data_async(self) -> str:
        task_id = uuid.uuid4().hex[:8]
        record = TaskRecord(task_id=task_id, strategy_key="__sync__")
        self._task_store[task_id] = record
        self._executor.submit(self._sync_data_task, task_id)
        return task_id

    def _sync_data_task(self, task_id: str) -> None:
        record = self._task_store[task_id]
        record.status = TaskStatus.RUNNING
        record.started_at = datetime.now()
        try:
            count = self.engine.sync_today_bulk()
            record.results = [f"synced:{count}"]
            record.status = TaskStatus.DONE
        except Exception as e:
            record.status = TaskStatus.ERROR
            record.error = str(e)
        finally:
            record.finished_at = datetime.now()

    def backfill_async(self) -> str:
        task_id = uuid.uuid4().hex[:8]
        record = TaskRecord(task_id=task_id, strategy_key="__backfill__")
        self._task_store[task_id] = record
        self._executor.submit(self._backfill_task, task_id)
        return task_id

    def _backfill_task(self, task_id: str) -> None:
        record = self._task_store[task_id]
        record.status = TaskStatus.RUNNING
        record.started_at = datetime.now()
        try:
            all_symbols = self.engine.get_all_symbols()
            self.engine.backfill(all_symbols)
            record.status = TaskStatus.DONE
        except Exception as e:
            record.status = TaskStatus.ERROR
            record.error = str(e)
        finally:
            record.finished_at = datetime.now()

    # -- Market analysis methods --

    def _get_analyzer(self) -> MarketAnalyzer:
        if self._market_analyzer is None:
            self._market_analyzer = MarketAnalyzer(self.settings)
        return self._market_analyzer

    def _get_stock_analyzer(self) -> StockAnalyzer:
        if self._stock_analyzer is None:
            self._stock_analyzer = StockAnalyzer(self.settings)
        return self._stock_analyzer

    def analyze_stock(self, symbol: str) -> dict:
        """同步分析个股，返回六层结构化决策报告（当日有效缓存）。

        A股T+1：日K收盘后技术指标/财报日内不变，分析结果当日有效。
        缓存 key 含日期，自然跨日失效；盘中实时价变化不影响决策逻辑。
        """
        import time
        from datetime import date
        today = date.today().isoformat()
        cached = self._stock_result_cache.get(symbol)
        if cached and cached[0].get("date") == today:
            return cached[1]
        result = self._get_stock_analyzer().analyze(symbol)
        self._stock_result_cache[symbol] = ({"date": today}, result)
        return result

    def analyze_portfolio(self, symbols: list[str]) -> dict:
        """批量分析多只股票，返回 {stocks, summary}（持仓体检）。

        - 预热东财全市场快照缓存（一次请求），避免并发刷新竞争。
        - ThreadPoolExecutor 并行分析（baostock 财报内部已加锁串行化）。
        - 聚合组合体检摘要：评分分布 / 行业集中度 / 加权估值 / 风险预警。
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ) -> dict:
        """交易决策中枢：多策略选股 → 质量过滤 → 共振定级 → 仓位分配。

        并行运行选定策略，汇总去重后跑个股分析，输出可执行买卖清单（10分钟缓存）。
        """
        import time
        now = time.time()
        # 缓存 key 包含策略+过滤参数，避免不同条件复用错误结果
        cache_key = f"{','.join(sorted(strategy_keys or []))}|{capital}|{min_score}|{','.join(sorted(exclude_markets or []))}|{exclude_st}"
        from datetime import date
        today = date.today().isoformat()
        cached = self._decision_cache.get(cache_key)
        if cached and cached[0].get("date") == today:
            return cached[1]
        if strategy_keys is None:
            strategy_keys = list(STRATEGY_REGISTRY.keys())

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Step1: 并行运行选定策略
        strategy_results: dict[str, list[str]] = {}
        def _run_strategy(key: str) -> tuple[str, list[str]]:
            # 优先用缓存结果（5min内跑过的）
            cached = self._result_cache.get(key, [])
            if cached:
                return key, cached
            cls = STRATEGY_REGISTRY.get(key)
            if not cls:
                return key, []
            try:
                strat = cls(engine=self.engine, settings=self.settings)
                results = strat.run()
                self._result_cache[key] = results
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
        # 大盘状态择时：复用已缓存的market报告，无缓存则实时分析
        def _market_fn():
            if self._market_report_cache:
                latest_date = max(self._market_report_cache.keys())
                return self._market_report_cache[latest_date]
            return self._get_analyzer().analyze()
        result = engine.generate(
            strategy_results=strategy_results,
            analyze_fn=self.analyze_stock,
            capital=capital,
            min_score=min_score,
            exclude_markets=exclude_markets,
            exclude_st=exclude_st,
            market_fn=_market_fn,
        )
        result["strategies_run"] = {
            k: len(v) for k, v in strategy_results.items()
        }
        self._decision_cache[cache_key] = ({"date": today}, result)
        logging.getLogger(__name__).info('决策缓存写入 key=' + cache_key[:40])
        logging.getLogger(__name__).info(
            f"决策生成完成：候选{result['pool_size']}只 → 买入{result['summary']['buy_count']}只"
        )
        return result

    def backtest_combos(self, combos: dict[str, list[str]] | None = None,
                         hold_days: list[int] | None = None) -> dict:
        """组合历史回测对比（向量化，采样500只，秒级返回）。"""
        if combos is None:
            combos = {
                "趋势突破": ["turtle", "ma_volume", "rps"],
                "低吸埋伏": ["pullback", "bottom"],
                "均衡全天候": ["rps", "turtle", "pullback", "bottom"],
            }
        hold_days = hold_days or [5, 10, 20]
        bt = ComboBacktester(self.engine, self.settings)
        return bt.run(combos, hold_days=hold_days)

    def backtest_resonance(self, hold_days: list[int] | None = None) -> dict:
        """共振度分档回测（验证多策略共振是否带来超额收益）。"""
        hold_days = hold_days or [5, 10, 20]
        bt = ComboBacktester(self.engine, self.settings)
        return bt.run_resonance(hold_days=hold_days)

    # ------------------------------------------------------------------
    # 持仓跟踪 PositionTracker
    # ------------------------------------------------------------------
    @property
    def positions(self) -> PositionTracker:
        """懒加载持仓跟踪器。"""
        if self._position_tracker is None:
            self._position_tracker = PositionTracker(self.engine, self.settings)
        return self._position_tracker

    def list_holdings(self, status: str = "open") -> list[dict]:
        return self.positions.list_holdings(status)

    def add_holding(self, data: dict) -> int:
        """从决策 buy_list 条目或手动录入新增持仓。"""
        return self.positions.add_holding(
            symbol=data["symbol"], name=data.get("name", ""),
            entry_price=float(data["entry_price"]), shares=int(data["shares"]),
            entry_date=data.get("entry_date"), stop_loss=float(data.get("stop_loss", 0)),
            target=float(data.get("target", 0)), grade=data.get("grade", ""),
            hit_strategies=data.get("hit_strategies", ""), notes=data.get("notes", ""),
        )

    def update_holding(self, hid: int, **fields) -> bool:
        return self.positions.update_holding(hid, **fields)

    def close_holding(self, hid: int, close_price: float, reason: str = "") -> bool:
        return self.positions.close_holding(hid, close_price, reason)

    def delete_holding(self, hid: int) -> bool:
        return self.positions.delete_holding(hid)

    def scan_positions(self, apply_stop_move: bool = False) -> dict:
        """扫描所有持仓，返回信号列表 + 组合摘要。"""
        signals = self.positions.scan_all(apply_stop_move=apply_stop_move)
        return {
            "signals": [self.positions.signal_to_dict(s) for s in signals],
            "summary": self.positions.summary(signals),
        }

    def import_decision_to_holdings(self, buy_list: list[dict]) -> dict:
        """把决策买入清单批量导入持仓表（跳过已持仓的）。"""
        added, skipped = 0, 0
        for r in buy_list:
            if r.get("shares", 0) <= 0 or r.get("price", 0) <= 0:
                skipped += 1
                continue
            try:
                self.positions.add_holding(
                    symbol=r["symbol"], name=r.get("name", ""),
                    entry_price=r["price"], shares=r["shares"],
                    stop_loss=r.get("stop_loss", 0), target=r.get("target", 0),
                    grade=r.get("grade", ""),
                    hit_strategies=",".join(r.get("hit_strategies", [])),
                )
                added += 1
            except Exception as e:
                logger.warning(f"导入持仓 {r.get('symbol')} 失败：{e!r}")
                skipped += 1
        logger.info(f"决策导入持仓：新增 {added} 只，跳过 {skipped} 只")
        return {"added": added, "skipped": skipped}

    def push_positions_feishu(self) -> dict:
        """推送持仓扫描报告到飞书。"""
        signals = self.positions.scan_all()
        summary = self.positions.summary(signals)
        notifier = FeishuNotifier(self.settings)
        ok = notifier.send_positions(
            [self.positions.signal_to_dict(s) for s in signals], summary,
        )
        return {"success": ok, "count": summary["count"]}

    def push_decision_feishu(self, decision: dict | None = None,
                             strategy_keys: list[str] | None = None,
                             capital: float = 100000.0, min_score: int = 50,
                             exclude_markets: list[str] | None = None,
                             exclude_st: bool = False) -> dict:
        """推送交易决策清单到飞书。无 decision 参数时自动生成。"""
        if decision is None:
            decision = self.generate_decision(
                strategy_keys=strategy_keys, capital=capital, min_score=min_score,
                exclude_markets=exclude_markets, exclude_st=exclude_st,
            )
        notifier = FeishuNotifier(self.settings)
        ok = notifier.send_decision(decision)
        return {"success": ok, "buy_count": decision.get("summary", {}).get("buy_count", 0)}

    def analyze_market_async(self, target_date: str | None = None) -> str:
        """异步生成大盘分析报告，结果缓存到内存。"""
        task_id = uuid.uuid4().hex[:8]
        record = TaskRecord(task_id=task_id, strategy_key="__market__")
        self._task_store[task_id] = record
        self._executor.submit(self._analyze_market_task, task_id, target_date)
        return task_id

    def _analyze_market_task(self, task_id: str, target_date: str | None) -> None:
        record = self._task_store[task_id]
        record.status = TaskStatus.RUNNING
        record.started_at = datetime.now()
        try:
            analyzer = self._get_analyzer()
            report = analyzer.analyze(target_date)
            self._market_report_cache[report["date"]] = report
            record.results = [report["date"]]
            record.status = TaskStatus.DONE
        except Exception as e:
            record.status = TaskStatus.ERROR
            record.error = str(e)
        finally:
            record.finished_at = datetime.now()

    def get_market_report(self, target_date: str | None = None) -> dict | None:
        """返回缓存的报告；target_date 为空时返回最新缓存的报告。"""
        if not self._market_report_cache:
            return None
        if target_date and target_date in self._market_report_cache:
            return self._market_report_cache[target_date]
        return list(self._market_report_cache.values())[-1]

    # ------------------------------------------------------------------
    # 信号评分回测
    # ------------------------------------------------------------------
    def backtest_async(self) -> str:
        """异步执行信号评分 IC 回测，结果缓存到内存。"""
        task_id = uuid.uuid4().hex[:8]
        record = TaskRecord(task_id=task_id, strategy_key="__backtest__")
        self._task_store[task_id] = record
        self._executor.submit(self._backtest_task, task_id)
        return task_id

    def _backtest_task(self, task_id: str) -> None:
        record = self._task_store[task_id]
        record.status = TaskStatus.RUNNING
        record.started_at = datetime.now()
        try:
            bt = SignalBacktester(self.settings)
            report = bt.run(min_days=120)
            self._backtest_cache = report
            record.results = [f"sample_days:{report.get('sample_days', 0)}"]
            record.status = TaskStatus.DONE
        except Exception as e:
            record.status = TaskStatus.ERROR
            record.error = str(e)
        finally:
            record.finished_at = datetime.now()

    def get_backtest_report(self) -> dict | None:
        """返回缓存的回测报告。"""
        return self._backtest_cache

    def refresh_industry_cache_async(self) -> str:
        """异步刷新行业/板块分类缓存：东财细分板块映射 + 证监会行业分类。"""
        task_id = uuid.uuid4().hex[:8]
        record = TaskRecord(task_id=task_id, strategy_key="__industry__")
        self._task_store[task_id] = record
        self._executor.submit(self._refresh_industry_task, task_id)
        return task_id

    def _refresh_industry_task(self, task_id: str) -> None:
        record = self._task_store[task_id]
        record.status = TaskStatus.RUNNING
        record.started_at = datetime.now()
        try:
            analyzer = self._get_analyzer()
            # 1. 东财细分板块映射（首选数据源，约 3~5 分钟）
            board_count = analyzer.refresh_board_cache()
            # 2. 证监会行业分类（回退数据源，约 30~45 秒）
            ind_count = analyzer.refresh_industry_cache()
            # 3. 流通市值缓存（板块市值加权用，约 10~15 秒）
            cap_count = analyzer.refresh_market_cap_cache()
            record.results = [
                f"board_stocks:{board_count}",
                f"industries:{ind_count}",
                f"market_caps:{cap_count}",
            ]
            record.status = TaskStatus.DONE
        except Exception as e:
            record.status = TaskStatus.ERROR
            record.error = str(e)
        finally:
            record.finished_at = datetime.now()

    # -- Stock data methods --

    def search_symbols(self, query: str, limit: int = 20) -> list[str]:
        with sqlite3.connect(self.engine.db_path, timeout=5) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM stock_daily WHERE symbol LIKE ? ORDER BY symbol LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [r[0] for r in rows]

    def get_stock_data(self, symbol: str, days: int = 120) -> list[dict]:
        df = self.engine.get_ohlcv(symbol)
        if df.empty:
            return []
        df = df.sort_values("date").tail(days)
        return df[["date", "open", "high", "low", "close", "volume", "turnover"]].to_dict("records")

    def get_stock_summary(self, symbol: str) -> dict | None:
        df = self.engine.get_ohlcv(symbol)
        if df.empty:
            return None
        df = df.sort_values("date")
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change_pct = round((latest["close"] - prev["close"]) / prev["close"] * 100, 2) if prev["close"] else 0
        return {
            "symbol": symbol,
            "latest_close": round(latest["close"], 2),
            "latest_date": latest["date"],
            "change_pct": change_pct,
            "total_rows": len(df),
            "date_range": [df.iloc[0]["date"], df.iloc[-1]["date"]],
            "xueqiu_code": _to_xueqiu_code(symbol),
        }

    # -- System methods --

    def get_system_info(self) -> dict:
        db_path = self.engine.db_path
        db_size_mb = round(Path(db_path).stat().st_size / 1024 / 1024, 2) if Path(db_path).exists() else 0
        with sqlite3.connect(db_path, timeout=5) as conn:
            total_rows = conn.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0]
            distinct_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM stock_daily").fetchone()[0]
            last_date = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0]
        return {
            "db_path": db_path,
            "db_size_mb": db_size_mb,
            "total_rows": total_rows,
            "distinct_symbols": distinct_symbols,
            "last_sync_date": last_date or "N/A",
        }

    def get_recent_logs(self, handler: RingBufferHandler, limit: int = 100) -> list[str]:
        items = list(handler.buffer)
        return items[-limit:]
