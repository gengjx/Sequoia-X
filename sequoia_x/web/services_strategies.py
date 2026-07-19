"""Web 服务 - 策略执行（WebServices mixin 组件）。

由 :class:`sequoia_x.web.services.WebServices` 多重继承组合，不单独实例化；
方法通过 ``self.engine`` / ``self.settings`` / ``self._task_store`` 等访问门面状态。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from sequoia_x.strategy.base import BaseStrategy
from sequoia_x.strategy.registry import (
    STRATEGY_META,
    STRATEGY_REGISTRY,
)
from sequoia_x.web.services_common import (
    TaskRecord,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class StrategyMixin:
    """策略执行（WebServices 的 mixin 组件）。"""

    def list_strategies(self) -> list[dict]:
        # 加载策略回测绩效（quality_score）用于排序
        scores: dict[str, int] = {}
        try:
            import sqlite3 as _sql
            with _sql.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT strategy_key, quality_score, annual_return FROM strategy_weights"
                ).fetchall()
                scores = {r[0]: r[1] for r in rows}
                annuals = {r[0]: r[2] for r in rows}
        except Exception:
            pass

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
                "quality_score": scores.get(key, 0),
                "annual_return": round(annuals.get(key, 0), 1),
                "role": meta.get("role", "active"),  # core/active/demoted/retired
            })
        # 排序：废弃排最后，其余按质量分降序
        result.sort(key=lambda x: (x.get("role") == "retired", -x["quality_score"]))
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
            self._result_cache[key] = (self._data_date(), results)
        except Exception as e:
            record.status = TaskStatus.ERROR
            record.error = str(e)
        finally:
            record.finished_at = datetime.now()

    def get_cached_results(self, key: str) -> list[str]:
        return self._result_cache.get(key, ("", []))[1]

    # ── 通用异步任务提交 ──
