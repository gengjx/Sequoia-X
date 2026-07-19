"""Web 服务共享层：任务记录、日志环缓冲、工具函数。

供各 services_*.py mixin 子模块与 services.py 门面共享，
独立于 WebServices 业务逻辑，避免循环导入。
"""

import collections
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

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
    strategy_key: str  # 通用任务名（不限于策略）
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    results: list[str] = field(default_factory=list)
    result_data: object = None  # 通用结果存储（任意JSON可序列化对象）
    error: str | None = None
    progress: int = 0  # 0-100
    progress_msg: str = ""  # 当前步骤描述
    elapsed_sec: float = 0.0


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
