"""Web 服务 - 数据同步（WebServices mixin 组件）。

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


class DataMixin:
    """数据同步（WebServices 的 mixin 组件）。"""

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
