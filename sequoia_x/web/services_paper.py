"""Web 服务 - 模拟盘（WebServices mixin 组件）。

由 :class:`sequoia_x.web.services.WebServices` 多重继承组合，不单独实例化；
方法通过 ``self.engine`` / ``self.settings`` / ``self._task_store`` 等访问门面状态。
"""

from __future__ import annotations

import logging

from sequoia_x.analysis.paper_trade import PaperTradeEngine

logger = logging.getLogger(__name__)


class PaperMixin:
    """模拟盘（WebServices 的 mixin 组件）。"""

    @property
    def paper_engine(self) -> PaperTradeEngine:
        """懒加载模拟盘引擎。"""
        if not hasattr(self, "_paper_engine") or self._paper_engine is None:
            from sequoia_x.analysis.paper_trade import PaperTradeEngine
            self._paper_engine = PaperTradeEngine(self.settings)
        return self._paper_engine

    def paper_auto_buy(self, decision_result: dict) -> dict:
        """模拟盘自动买入。"""
        return self.paper_engine.auto_buy(decision_result)

    def paper_auto_sell(self, position_signals: list[dict]) -> dict:
        """模拟盘自动卖出（信号dict列表 → 引擎执行）。"""
        return self.paper_engine.auto_sell(position_signals)

    def paper_performance(self):
        """模拟盘绩效快照。"""
        return self.paper_engine.get_performance()

    def paper_record_nav(self) -> dict:
        """记录日度净值快照。"""
        return self.paper_engine.record_daily_nav()
