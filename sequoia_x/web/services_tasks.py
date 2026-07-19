"""Web 服务 - 后台任务（WebServices mixin 组件）。

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


class TaskMixin:
    """后台任务（WebServices 的 mixin 组件）。"""

    def submit_task(self, task_name: str, fn: callable, *args, **kwargs) -> str:
        """提交任意长任务到后台线程池，立即返回 task_id。

        Args:
            task_name: 任务名称（如 "decision", "evaluate_factors"）
            fn: 要执行的函数
            *args, **kwargs: 传给 fn 的参数

        Returns:
            task_id（8位hex）
        """
        task_id = uuid.uuid4().hex[:8]
        record = TaskRecord(task_id=task_id, strategy_key=task_name)
        self._task_store[task_id] = record
        self._executor.submit(self._run_generic_task, task_id, task_name, fn, args, kwargs)
        return task_id

    def has_running_task(self, task_name: str) -> str | None:
        """检查指定名称的任务是否正在运行，返回 task_id 或 None。"""
        for tid, t in self._task_store.items():
            if t.strategy_key == task_name and t.status == TaskStatus.RUNNING:
                return tid
        return None

    def update_task_progress(self, task_name: str, progress: int, msg: str = "") -> None:
        """更新当前运行任务的进度（供后台函数调用）。"""
        for t in self._task_store.values():
            if t.strategy_key == task_name and t.status == TaskStatus.RUNNING:
                t.progress = progress
                if msg:
                    t.progress_msg = msg
                return

    def _run_generic_task(self, task_id: str, task_name: str,
                          fn: callable, args: tuple, kwargs: dict) -> None:
        record = self._task_store[task_id]
        record.status = TaskStatus.RUNNING
        record.started_at = datetime.now()
        import time as _time
        t0 = _time.time()
        try:
            record.progress = 10
            record.progress_msg = f"{task_name} 执行中..."
            result = fn(*args, **kwargs)
            record.result_data = result
            record.progress = 100
            record.progress_msg = "完成"
            record.status = TaskStatus.DONE
        except Exception as e:
            record.status = TaskStatus.ERROR
            record.error = str(e)
            record.progress_msg = f"失败: {e}"
        finally:
            record.finished_at = datetime.now()
            record.elapsed_sec = round(_time.time() - t0, 1)

    def list_tasks(self, limit: int = 20) -> list[dict]:
        """列出最近的任务记录。"""
        tasks = sorted(self._task_store.values(),
                       key=lambda t: t.started_at or datetime.min, reverse=True)[:limit]
        return [{
            "task_id": t.task_id, "name": t.strategy_key,
            "status": t.status.value, "progress": t.progress,
            "progress_msg": t.progress_msg, "elapsed": t.elapsed_sec,
            "started_at": t.started_at.isoformat() if t.started_at else None,
            "error": t.error,
        } for t in tasks]

    def get_task_status(self, task_id: str) -> TaskRecord | None:
        return self._task_store.get(task_id)

    def get_task_result(self, task_id: str) -> dict | None:
        """获取任务结果数据。"""
        t = self._task_store.get(task_id)
        if not t:
            return None
        return {
            "task_id": t.task_id, "name": t.strategy_key,
            "status": t.status.value, "progress": t.progress,
            "progress_msg": t.progress_msg, "elapsed": t.elapsed_sec,
            "result": t.result_data, "error": t.error,
        }

    # -- Data sync methods --
