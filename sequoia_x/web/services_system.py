"""Web 服务 - 系统监控（WebServices mixin 组件）。

由 :class:`sequoia_x.web.services.WebServices` 多重继承组合，不单独实例化；
方法通过 ``self.engine`` / ``self.settings`` / ``self._task_store`` 等访问门面状态。
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from sequoia_x.web.services_common import (
    RingBufferHandler,
)

logger = logging.getLogger(__name__)


class SystemMixin:
    """系统监控（WebServices 的 mixin 组件）。"""

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

    # ------------------------------------------------------------------
    # 模拟盘（Paper Trading）
    # ------------------------------------------------------------------
