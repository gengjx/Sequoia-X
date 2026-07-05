"""策略基类模块：定义所有选股策略的抽象接口。"""

from abc import ABC, abstractmethod

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine


class BaseStrategy(ABC):
    """选股策略抽象基类。

    所有具体策略必须继承此类并实现 run() 方法。

    Attributes:
        webhook_key: 策略对应的飞书 webhook 标识，用于路由到不同机器人。
            默认为 'default'，将使用 Settings.feishu_webhook_url。
            子类可覆盖此属性以路由到专属机器人，例如 'ma_volume'。
    """

    webhook_key: str = "default"

    def __init__(self, engine: DataEngine, settings: Settings) -> None:
        """
        初始化策略。

        Args:
            engine: DataEngine 实例，用于读取行情数据。
            settings: Settings 实例，用于读取配置。
        """
        self.engine = engine
        self.settings = settings
        self._shared_daily = None  # 共享K线分组dict（批量选股时注入，避免重复I/O）

    def set_shared_daily(self, groups: dict) -> None:
        """注入共享K线分组dict（{symbol: DataFrame}），run()中优先用它替代逐只get_ohlcv。

        必须传预分组dict而非原始DataFrame：逐只布尔过滤是O(n)全表扫描(5000只要1200s)，
        dict取片O(1)。groups由DataEngine.get_daily_groups()生成。
        """
        self._shared_daily = groups

    def get_daily(self, symbol: str):
        """获取单股K线：优先用共享分组dict O(1)取片，否则逐只查库。"""
        if self._shared_daily is not None:
            return self._shared_daily.get(symbol)
        return self.engine.get_ohlcv(symbol)

    @abstractmethod
    def run(self) -> list[str]:
        """
        执行选股逻辑，返回选中的股票代码列表。

        Returns:
            满足策略条件的股票代码列表，如 ['000001', '600519']。
            无选股结果时返回空列表。
        """
        ...
