"""策略模块：基类、注册中心与具体策略实现。

导入本包即触发所有策略通过 :func:`register_strategy` 自动注册到
:data:`sequoia_x.strategy.registry.STRATEGY_REGISTRY`。
新增策略：编写策略类 + ``@register_strategy("key")`` + 在此 import，
无需改动 ``main.py`` 或 ``web/services.py``。

注：各策略模块导入时即执行装饰器完成注册（副作用 import），
故下方策略类虽未直接引用，仍需保留以触发注册。
"""

from sequoia_x.strategy.base import BaseStrategy
from sequoia_x.strategy.bottom_volume import BottomVolumeStrategy  # noqa: F401
from sequoia_x.strategy.dragon_head import DragonHeadStrategy  # noqa: F401
from sequoia_x.strategy.high_tight_flag import HighTightFlagStrategy  # noqa: F401
from sequoia_x.strategy.lhb_follow import LhbFollowStrategy  # noqa: F401
from sequoia_x.strategy.limit_up_shakeout import LimitUpShakeoutStrategy  # noqa: F401
from sequoia_x.strategy.ma_volume import MaVolumeStrategy  # noqa: F401
from sequoia_x.strategy.multi_factor import MultiFactorStrategy  # noqa: F401
from sequoia_x.strategy.private_placement import PrivatePlacementStrategy  # noqa: F401
from sequoia_x.strategy.registry import (
    ACTIVE_STRATEGY_KEYS,
    ALL_ACTIVE_KEYS,
    CORE_STRATEGY_KEYS,
    RETIRED_STRATEGY_KEYS,
    STRATEGY_META,
    STRATEGY_REGISTRY,
    get_strategy_class,
    register_strategy,
)
from sequoia_x.strategy.rps_breakout import RpsBreakoutStrategy  # noqa: F401
from sequoia_x.strategy.sector_rotation import SectorRotationStrategy  # noqa: F401
from sequoia_x.strategy.shrink_pullback import ShrinkPullbackStrategy  # noqa: F401
from sequoia_x.strategy.turtle_trade import TurtleTradeStrategy  # noqa: F401
from sequoia_x.strategy.uptrend_limit_down import UptrendLimitDownStrategy  # noqa: F401
from sequoia_x.strategy.volume_extreme import VolumeExtremeStrategy  # noqa: F401

__all__ = [
    "ACTIVE_STRATEGY_KEYS",
    "ALL_ACTIVE_KEYS",
    "BaseStrategy",
    "CORE_STRATEGY_KEYS",
    "RETIRED_STRATEGY_KEYS",
    "STRATEGY_META",
    "STRATEGY_REGISTRY",
    "get_strategy_class",
    "register_strategy",
]
