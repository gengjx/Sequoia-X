"""Web 服务 - 大盘分析（WebServices mixin 组件）。

由 :class:`sequoia_x.web.services.WebServices` 多重继承组合，不单独实例化；
方法通过 ``self.engine`` / ``self.settings`` / ``self._task_store`` 等访问门面状态。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from sequoia_x.web.services_common import (
    TaskRecord,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class MarketMixin:
    """大盘分析（WebServices 的 mixin 组件）。"""

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
