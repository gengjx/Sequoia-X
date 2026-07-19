"""大盘分析引擎门面：组合各板块 mixin，生成结构化盘面报告。

实际计算逻辑拆分到：
- :mod:`market_common`：共享常量、``MarketReport`` 数据结构、工具函数
- :mod:`market_breadth` / :mod:`market_indices` / ... / :mod:`market_risks`：
  按报告板块对齐的 mixin 组件

本模块仅保留主入口 ``analyze()`` 与日期处理，保持对调用方
（``MarketAnalyzer`` 类、``analyze()`` 方法）的接口完全不变。
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

from sequoia_x.analysis.market_breadth import BreadthMixin
from sequoia_x.analysis.market_cache import CacheMixin
from sequoia_x.analysis.market_common import MarketReport
from sequoia_x.analysis.market_fundamentals import FundamentalsMixin
from sequoia_x.analysis.market_history import HistoryMixin
from sequoia_x.analysis.market_indices import IndicesMixin
from sequoia_x.analysis.market_microstructure import MicrostructureMixin
from sequoia_x.analysis.market_news import NewsMixin
from sequoia_x.analysis.market_plan import PlanMixin
from sequoia_x.analysis.market_risks import RisksMixin
from sequoia_x.analysis.market_sectors import SectorsMixin
from sequoia_x.analysis.market_sentiment import SentimentMixin
from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class MarketAnalyzer(
    BreadthMixin,
    IndicesMixin,
    CacheMixin,
    SectorsMixin,
    SentimentMixin,
    NewsMixin,
    PlanMixin,
    HistoryMixin,
    FundamentalsMixin,
    MicrostructureMixin,
    RisksMixin
):
    """大盘分析器：聚合本地行情与 baostock 数据生成报告。

    各板块计算逻辑由上方 mixin 组件提供，本类仅保留主入口与日期处理，
    通过多重继承组合所有能力。接口与拆分前完全一致。
    """

    def __init__(self, settings: Settings) -> None:
        self.db_path: str = settings.db_path

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def analyze(self, target_date: str | None = None) -> dict:
        """生成指定日期（默认最新交易日）的大盘分析报告。"""
        latest, prev = self._latest_dates(target_date)
        self._ensure_stock_basic_cache()
        self._ensure_market_cap_cache()
        report = MarketReport(
            date=latest,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data_source=f"Sequoia-X 本地库 + baostock（{latest}）",
        )

        # 1. 盘面总览（本地库，瞬时）
        breadth = self._compute_breadth(latest, prev)
        turnover_stats = self._compute_turnover_stats(latest)
        report.overview = self._build_overview(breadth, turnover_stats)
        signal = report.overview["signal_score"]

        # 2. 指数结构（baostock，~5s）
        try:
            report.indices = self._fetch_indices(latest)
        except Exception as exc:
            logger.warning(f"指数数据获取失败：{exc}")
            report.indices = []

        # 3. 板块主线（本地库 + 缓存行业分类）
        try:
            report.sectors = self._compute_sectors(latest, prev)
        except Exception as exc:
            logger.warning(f"板块数据计算失败：{exc}")
            report.sectors = {"top": [], "bottom": [], "text": f"板块数据计算失败：{exc}"}

        # 4. 资金与情绪
        sentiment_breadth = self._compute_market_breadth(latest)
        report.sentiment = self._build_sentiment(
            breadth, report.indices, report.sectors, turnover_stats, sentiment_breadth
        )

        # 4.5 资金流向（主力资金 + 融资融券，网络数据源）
        try:
            report.fund_flow = self._build_fund_flow(latest)
        except Exception as exc:
            logger.warning(f"资金流向数据获取失败：{exc}")
            report.fund_flow = {"text": f"资金流向数据获取失败：{exc}"}

        # 5. 消息催化（实时新闻聚合 + 板块关联分析）
        report.news = self._build_news(report.sectors, latest)

        # 6. 明日交易计划
        report.plan = self._build_plan(signal, report.indices, report.sectors, turnover_stats)

        # 四B. 历史纵向对比
        try:
            report.history = self._compute_history(latest, prev)
        except Exception as exc:
            logger.warning(f"历史纵向对比失败：{exc}")

        # 五. 估值与宏观
        try:
            report.valuation = self._fetch_valuation()
        except Exception as exc:
            logger.warning(f"估值数据获取失败：{exc}")
        try:
            report.macro = self._fetch_macro()
        except Exception as exc:
            logger.warning(f"宏观数据获取失败：{exc}")

        # 六. 涨跌停结构 + 龙虎榜
        try:
            report.limit_structure = self._compute_limit_structure(latest, prev)
        except Exception as exc:
            logger.warning(f"涨停板结构分析失败：{exc}")
        try:
            report.dragon_tiger = self._fetch_dragon_tiger()
        except Exception as exc:
            logger.warning(f"龙虎榜获取失败：{exc}")

        # 7. 风险提示
        report.risks = self._build_risks(breadth, report.indices, report.sectors)

        return report.to_dict()

    # ------------------------------------------------------------------
    # 日期处理
    # ------------------------------------------------------------------
    def _latest_dates(self, target_date: str | None) -> tuple[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            if target_date:
                latest = target_date
            else:
                row = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()
                latest = row[0] if row and row[0] else date.today().strftime("%Y-%m-%d")
            prev_row = conn.execute(
                "SELECT MAX(date) FROM stock_daily WHERE date < ?", (latest,)
            ).fetchone()
            prev = prev_row[0] if prev_row and prev_row[0] else latest
        return latest, prev

    # ------------------------------------------------------------------
    # 1. 盘面总览
    # ------------------------------------------------------------------
