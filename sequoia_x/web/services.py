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
from sequoia_x.analysis.strategy_eval import StrategyEvaluator
from sequoia_x.analysis.factor import evaluate_factor_ic
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
from sequoia_x.strategy.multi_factor import MultiFactorStrategy


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
        MultiFactorStrategy,
    ]
}

STRATEGY_META: dict[str, dict] = {
    "multi_factor": {
        "name": "MultiFactor",
        "name_cn": "多因子选股",
        "description": "30因子IC加权合成综合分，选全市场Top50（数据驱动，非规则式）",
        "min_bars": 60,
        "category": "量化因子",
    },
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
        self._result_cache: dict[str, tuple[str, list[str]]] = {}  # key -> (data_date, symbols)
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
                "latest_result_count": len(self._result_cache.get(key, ("", []))[1]),
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
        return self._result_cache.get(key, ("", []))[1]

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
            # 数据更新后清缓存：决策/个股分析将基于新数据重算
            self.invalidate_caches()
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
            self.invalidate_caches()
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

    def invalidate_caches(self) -> None:
        """数据同步后清空决策/个股分析缓存，使下次请求基于新数据重算。

        缓存版本绑定 data_date，但 _data_date 有60s内存缓存需手动清除，
        且清空结果缓存可让正在等待的请求立即感知数据更新。
        """
        self._stock_result_cache.clear()
        self._decision_cache.clear()
        self._result_cache.clear()
        self._market_report_cache.clear()
        self._data_date_cache = None  # 清版本缓存，下次读最新data_date
        # 清共享全量K线缓存（数据更新后需重新加载）
        if hasattr(self.engine, "_all_daily_cache"):
            self.engine._all_daily_cache = None
            self.engine._daily_groups_cache = None
        logging.getLogger(__name__).info("数据同步完成，已清空分析/决策缓存（下次请求重算）")

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
        max_candidates: int | None = None,
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
        cache_key = f"{','.join(sorted(strategy_keys or []))}|{capital}|{min_score}|{','.join(sorted(exclude_markets or []))}|{exclude_st}|mc{mc}"
        data_date = self._data_date()
        cached = self._decision_cache.get(cache_key)
        if cached and cached[0].get("data_date") == data_date:
            return cached[1]
        if strategy_keys is None:
            strategy_keys = list(STRATEGY_REGISTRY.keys())

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Step1: 并行运行选定策略（data_date 来自上方缓存检查，闭包复用）
        # 每策略独立读K线+结果缓存；共享DF实测在3M行下groupby慢+内存复制开销，无净收益
        strategy_results: dict[str, list[str]] = {}

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
        self._decision_cache[cache_key] = ({"data_date": data_date}, result)
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

    def evaluate_strategies(self, hold_days: int = 20, sample_size: int = 500) -> dict:
        """策略评估：时间序列净值 + 全维度评分卡 + 基准对比。"""
        ev = StrategyEvaluator(self.engine, self.settings)
        return ev.evaluate(hold_days=hold_days, sample_size=sample_size)

    def get_strategy_weights(self) -> dict:
        """读取DB中的策略权重快照（前端展示当前权重+更新时间）。"""
        return self.engine.load_strategy_weights()

    def find_optimal_combos(self, hold_days: int = 20, sample_size: int = 500,
                            max_strategies: int = 5, top_n: int = 10) -> dict:
        """网格搜索最优策略组合（数据驱动，替代主观预设）。"""
        ev = StrategyEvaluator(self.engine, self.settings)
        return ev.find_optimal_combos(
            hold_days=hold_days, sample_size=sample_size,
            max_strategies=max_strategies, top_n=top_n,
        )

    def compare_combos(self, hold_days: int = 20, sample_size: int = 500) -> dict:
        """主观预设组合 vs 数据驱动最优组合 对比。"""
        ev = StrategyEvaluator(self.engine, self.settings)
        return ev.compare_combos(hold_days=hold_days, sample_size=sample_size)

    def evaluate_factors(self, hold_days: int = 20, sample_size: int = 500) -> dict:
        """因子IC评估：30个因子的预测力评估（Rank IC/ICIR/分层）。"""
        return evaluate_factor_ic(self.engine, hold_days=hold_days, sample_size=sample_size)

    def get_factor_weights(self) -> dict:
        """读取DB中的因子权重快照。"""
        return self.engine.load_factor_weights()

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

    def get_stock_data(self, symbol: str, days: int = 120) -> dict:
        """返回K线数据（后复权）+ 复权系数，前端可切换后复权/不复权视图。

        DB存后复权数据（运算正确口径），展示层需标注并提供不复权视图。
        复权系数 = 东财昨收真实价(f60) / DB最新日后复权收盘（同日纯系数）。
        """
        df = self.engine.get_ohlcv(symbol)
        if df.empty:
            return {"rows": [], "adjust_ratio": None, "real_latest": None, "price_type": "hfq", "real_available": False}
        df = df.sort_values("date").tail(days)
        hfq_latest = float(df.iloc[-1]["close"])
        adjust_ratio = None
        real_latest = None
        try:
            real_latest, prev_close = StockAnalyzer._fetch_price_quote(symbol)
            if prev_close and hfq_latest and prev_close > 0:
                adjust_ratio = prev_close / hfq_latest
        except Exception:
            pass
        rows = df[["date", "open", "high", "low", "close", "volume", "turnover"]].to_dict("records")
        return {
            "rows": rows,
            "adjust_ratio": round(adjust_ratio, 6) if adjust_ratio else None,
            "real_latest": round(real_latest, 2) if real_latest else None,
            "price_type": "hfq",
            "real_available": adjust_ratio is not None,
        }

    def get_stock_summary(self, symbol: str) -> dict | None:
        df = self.engine.get_ohlcv(symbol)
        if df.empty:
            return None
        df = df.sort_values("date")
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change_pct = round((latest["close"] - prev["close"]) / prev["close"] * 100, 2) if prev["close"] else 0
        # 真实价（东财昨收，与DB最新日同日），DB存后复权，展示需双口径
        real_price = None
        adjust_ratio = None
        try:
            real_latest, prev_close = StockAnalyzer._fetch_price_quote(symbol)
            if prev_close and latest["close"] and prev_close > 0:
                real_price = round(prev_close, 2)
                adjust_ratio = round(prev_close / latest["close"], 6)
        except Exception:
            pass
        return {
            "symbol": symbol,
            "latest_close": round(latest["close"], 2),      # 后复权
            "real_price": real_price,                         # 真实不复权
            "adjust_ratio": adjust_ratio,                     # 后复权→真实系数
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
