"""Sequoia-X V2 主程序入口。

运行模式：
  python main.py               # 日常模式：8进程增量补数据 + 跑策略 + 飞书推送（2~3分钟）
  python main.py --backfill    # 回填模式：baostock 拉全市场历史K线（首次/补数据用，约12分钟）
  python main.py web           # Web 仪表盘：可视化管理、配置飞书推送、浏览数据
"""

import argparse
import sys
from dotenv import load_dotenv
load_dotenv()

from datetime import date

import socket
socket.setdefaulttimeout(10.0)

from sequoia_x.core.config import get_settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.notify.feishu import FeishuNotifier
from sequoia_x.strategy.registry import STRATEGY_REGISTRY, STRATEGY_META, RETIRED_STRATEGY_KEYS


def _run_web(args: argparse.Namespace) -> None:
    """启动 Web 仪表盘。"""
    import uvicorn
    from sequoia_x.web.app import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequoia-X V2 选股系统")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="回填模式：通过 baostock 拉取全市场历史 K 线（约12分钟）",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["web"],
        help="web: 启动 Web 仪表盘",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Web 监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="Web 监听端口（默认 8000）")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    args = parser.parse_args()

    if args.command == "web":
        _run_web(args)
        return

    try:
        # 1. 初始化配置
        settings = get_settings()

        # 2. 初始化日志
        logger = get_logger(__name__)
        logger.info("Sequoia-X V2 启动")

        # 3. 初始化数据引擎
        engine = DataEngine(settings)

        if args.backfill:
            # ── 回填模式：单线程保守拉历史 K 线，自动多轮重跑 ──
            logger.info("进入回填模式...")
            all_symbols = engine.get_all_symbols()
            engine.backfill(all_symbols)
            logger.info("Sequoia-X V2 回填模式运行完成")
            return

        # ── 日常模式：单次 API 补今天 + 策略 + 推送 ──
        logger.info("开始拉取最新快照...")
        count = engine.sync_today_bulk()
        logger.info(f"快照同步完成，写入 {count} 只股票")

        # 4. 策略列表：从注册中心取所有非 retired 策略（含此前漏推的 core 策略）
        # 新增策略只需 @register_strategy 装饰，无需改动此处
        strategies = [
            cls(engine=engine, settings=settings)
            for key, cls in STRATEGY_REGISTRY.items()
            if key not in RETIRED_STRATEGY_KEYS
        ]
        logger.info(
            f"待推送策略 {len(strategies)} 个："
            f"{[k for k in STRATEGY_REGISTRY if k not in RETIRED_STRATEGY_KEYS]}"
        )

        notifier = FeishuNotifier(settings)

        # 5. 遍历策略，有结果则推送至对应机器人
        for strategy in strategies:
            strategy_name = type(strategy).__name__
            logger.info(f"执行策略：{strategy_name}")

            selected: list[str] = strategy.run()
            logger.info(f"{strategy_name} 选出 {len(selected)} 只股票")

            if selected:
                notifier.send(
                    symbols=selected,
                    strategy_name=strategy_name,
                    webhook_key=strategy.webhook_key,
                )
            else:
                logger.info(f"{strategy_name} 无选股结果，跳过推送")

    except Exception:
        try:
            _logger = get_logger(__name__)
            _logger.exception("主流程发生未捕获异常，程序终止")
        except Exception:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    logger.info("Sequoia-X V2 运行完成")


if __name__ == "__main__":
    main()
