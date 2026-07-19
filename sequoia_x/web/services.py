"""Web 服务层门面：组合各业务域 mixin，桥接 engine/settings/strategies 到 Web API。

实际业务逻辑拆分到：
- :mod:`services_common`：TaskStatus/TaskRecord/RingBufferHandler 等共享类型
- :mod:`services_strategies` / :mod:`services_tasks` / ... / :mod:`services_paper`：
  按业务域对齐的 mixin 组件

本模块仅保留 ``WebServices`` 状态初始化，并通过多重继承组合所有能力。
对调用方接口完全不变：``request.app.state.services.xxx()`` 调用方式不变，
``TaskStatus`` / ``RingBufferHandler`` / ``STRATEGY_META`` 等 re-export 保留。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from sequoia_x.analysis.market import MarketAnalyzer
from sequoia_x.analysis.position import PositionTracker
from sequoia_x.analysis.stock_analysis import StockAnalyzer
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.strategy.registry import STRATEGY_META, STRATEGY_REGISTRY
from sequoia_x.web.services_analysis import AnalysisMixin
from sequoia_x.web.services_backtest import BacktestMixin
from sequoia_x.web.services_common import (
    RingBufferHandler,
    TaskRecord,
    TaskStatus,
    _to_xueqiu_code,
)
from sequoia_x.web.services_data import DataMixin
from sequoia_x.web.services_market import MarketMixin
from sequoia_x.web.services_paper import PaperMixin
from sequoia_x.web.services_positions import PositionMixin
from sequoia_x.web.services_stocks import StockMixin
from sequoia_x.web.services_strategies import StrategyMixin
from sequoia_x.web.services_system import SystemMixin
from sequoia_x.web.services_tasks import TaskMixin

# re-export：外部模块（api.py / pages.py / app.py / scheduler.py）依赖这些符号
__all__ = [
    "RingBufferHandler",
    "STRATEGY_META",
    "STRATEGY_REGISTRY",
    "TaskRecord",
    "TaskStatus",
    "WebServices",
    "_to_xueqiu_code",
]


class WebServices(
    StrategyMixin,
    TaskMixin,
    DataMixin,
    AnalysisMixin,
    BacktestMixin,
    PositionMixin,
    MarketMixin,
    StockMixin,
    SystemMixin,
    PaperMixin
):
    """Web 服务层：聚合数据/策略/回测/模拟盘/持仓等业务能力。

    各业务域逻辑由上方 mixin 组件提供，本类仅保留状态初始化，通过多重继承
    组合所有能力。接口与拆分前完全一致。
    """

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
        self._executor = ThreadPoolExecutor(max_workers=8)
        # ── 互斥锁：防止重复提交 + 并发决策去重 ──
        self._decision_lock = threading.Lock()   # generate_decision 串行化
        self._paper_lock = threading.Lock()       # paper_auto_run 全局互斥

    # -- Strategy methods --
