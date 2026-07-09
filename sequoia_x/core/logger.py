"""日志模块：终端彩色输出 + 持久化文件（按天轮转）。

日志同时写入终端（RichHandler）和 data/logs/ 目录（TimedRotatingFileHandler），
每天一个文件，保留 30 天。重启不丢失。
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_FORMAT = "%(name)s - %(message)s"

_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "logs",
)


def get_logger(name: str) -> logging.Logger:
    """工厂函数：终端彩色 + 文件持久化双 handler。"""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # 终端彩色输出
    console = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        log_time_format="[%Y-%m-%d %H:%M:%S]",
    )
    console.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(console)

    # 文件持久化（按天轮转，保留30天）
    try:
        Path(_LOG_DIR).mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            os.path.join(_LOG_DIR, "sequoia.log"),
            when="midnight",
            backupCount=30,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception:
        pass  # 日志目录不可写时降级为纯终端输出

    return logger
