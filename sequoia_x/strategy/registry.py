"""策略注册中心：装饰器自动注册 + 集中元数据。

通过 ``@register_strategy(key)`` 装饰具体策略类即完成注册，
策略 key（即飞书 webhook_key）在装饰器参数中**唯一定义**，
消除过去「类属性 webhook_key + STRATEGY_REGISTRY + main.py 硬编码列表」三处分散。

元数据（名称/描述/角色分层）集中在 :data:`STRATEGY_META` 维护，
其中 ``role`` 由全周期回测驱动，会随绩效迭代调整，故不放入策略类定义。
注册中心位于核心 ``strategy`` 层，供 ``main.py``、``web/services``、
``analysis/decision`` 等所有调用方共享，避免业务层反向依赖 Web 层。
"""

from __future__ import annotations

from sequoia_x.strategy.base import BaseStrategy

# ---------------------------------------------------------------------------
# 注册表（由 @register_strategy 装饰器在导入时填充）
# ---------------------------------------------------------------------------
STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}

# ---------------------------------------------------------------------------
# 元数据（人工/回测维护；role 会随绩效调整）
# ---------------------------------------------------------------------------
STRATEGY_META: dict[str, dict] = {
    "multi_factor": {
        "name": "MultiFactor",
        "name_cn": "多因子选股",
        "description": "30因子IC加权合成综合分，选全市场Top50（数据驱动，非规则式）",
        "role": "core",  # 核心策略：5.3年回测年化+22.6%，超额+6.4%，唯一穿越牛熊
        "min_bars": 60,
        "category": "量化因子",
    },
    "ma_volume": {
        "name": "MaVolume",
        "name_cn": "均线放量",
        "description": "5日均线上穿20日均线（金叉）且成交量放大1.5倍",
        "role": "demoted",  # 降权：年化-1.4%
        "min_bars": 20,
    },
    "turtle": {
        "name": "TurtleTrade",
        "name_cn": "海龟突破",
        "description": "20日新高突破 + 成交额过亿 + 阳线防诱多",
        "role": "retired",  # 已废弃：5.3年年化-17.1%，回撤-72%
        "min_bars": 21,
    },
    "flag": {
        "name": "HighTightFlag",
        "name_cn": "高位旗形",
        "description": "40日涨幅>60% + 10日窄幅震荡 + 缩量整理",
        "role": "active",  # 辅助策略：年化+7.0%
        "min_bars": 40,
    },
    "shakeout": {
        "name": "LimitUpShakeout",
        "name_cn": "涨停洗盘",
        "description": "昨日涨停 + 今日阴线放量 + 不破涨停支撑",
        "role": "retired",  # 已废弃：年化-20.8%，回撤-81%
        "min_bars": 5,
    },
    "limit_down": {
        "name": "UptrendLimitDown",
        "name_cn": "上升趋势跌停",
        "description": "MA20>MA60上升趋势 + 今日跌停 + 放量",
        "role": "retired",  # 已废弃：年化-33.2%，回撤-90%
        "min_bars": 60,
    },
    "rps": {
        "name": "RpsBreakout",
        "name_cn": "RPS相对强度",
        "description": "120日涨幅排名前10% + 接近120日新高",
        "role": "retired",  # 已废弃：年化-29.3%，回撤-91%
        "min_bars": 120,
    },
    "pullback": {
        "name": "ShrinkPullback",
        "name_cn": "缩量回踩",
        "description": "上升趋势回踩均线支撑 + 缩量企稳，右侧低吸买点",
        "role": "demoted",  # 降权：年化+1.9%，弱于基准
        "min_bars": 20,
    },
    "dragon": {
        "name": "DragonHead",
        "name_cn": "板块龙头",
        "description": "领涨板块内跑赢板块+成交过亿的强势龙头",
        "role": "retired",  # 已废弃：年化-27.4%，回撤-87%
        "min_bars": 2,
    },
    "bottom": {
        "name": "BottomVolume",
        "name_cn": "底部放量",
        "description": "超跌15%+异动放量3倍+下影线阳线，左侧反转信号",
        "role": "active",  # 辅助策略：年化+11.9%，超额-4.3%
        "min_bars": 20,
    },
    "volume_extreme": {
        "name": "VolumeExtreme",
        "name_cn": "地量见底",
        "description": "换手率创60日新低+价格企稳+非涨停，地量地价左侧反转",
        "role": "demoted",  # 降权：年化+1.2%
        "min_bars": 60,
        "category": "量能择时",
    },
    "lhb_follow": {
        "name": "LhbFollow",
        "name_cn": "龙虎榜跟买",
        "description": "龙虎榜机构净买入>5000万+趋势确认，聪明资金跟随",
        "min_bars": 60,
        "category": "事件驱动",
        "role": "supplementary",  # 信号级回测为负，但实时维度独立价值（龙虎榜真实数据）
    },
    "sector_rotation": {
        "name": "SectorRotation",
        "name_cn": "板块轮动",
        "description": "Top5强势板块×板块内龙头股，中期趋势跟随",
        "min_bars": 60,
        "category": "板块轮动",
        "role": "supplementary",  # 信号级回测为负，但实时板块轮动有独立价值
    },
    "private_placement": {
        "name": "PrivatePlacement",
        "name_cn": "定增公告",
        "description": "最近7天定向增发公告监控，事件驱动信息推送（非选股alpha）",
        "min_bars": 0,
        "category": "事件驱动",
        "role": "supplementary",  # 公告监控类，信息推送，不参与决策选股
    },
}

# ── 策略分层（基于5.3年完整牛熊周期回测）──
# core:         穿越牛熊有效，决策中枢默认核心
# active:       正收益辅助信号，可组合使用
# demoted:      弱于基准，默认不纳入决策（可在前端手动选择）
# supplementary: 信息/事件类，不参与选股 alpha 但有独立推送价值
# retired:      完整周期亏损，已废弃
CORE_STRATEGY_KEYS: list[str] = ["multi_factor"]
ACTIVE_STRATEGY_KEYS: list[str] = ["multi_factor", "bottom", "flag"]
ALL_ACTIVE_KEYS: list[str] = [
    k for k, v in STRATEGY_META.items() if v.get("role") in ("core", "active", "demoted")
]
RETIRED_STRATEGY_KEYS: list[str] = [
    k for k, v in STRATEGY_META.items() if v.get("role") == "retired"
]


def register_strategy(key: str):
    """装饰器：注册策略类并设定其 ``webhook_key``。

    Args:
        key: 策略唯一标识，同时作为飞书 webhook 路由 key。

    装饰后 ``cls.webhook_key`` 被设为 *key*（覆盖基类默认值 ``"default"``），
    策略类以 *key* 存入 :data:`STRATEGY_REGISTRY`。重复 key 抛 ``ValueError``。
    """

    def decorator(cls: type[BaseStrategy]) -> type[BaseStrategy]:
        if key in STRATEGY_REGISTRY:
            raise ValueError(
                f"策略 key 已存在: {key}（已被 {STRATEGY_REGISTRY[key].__name__} 注册）"
            )
        cls.webhook_key = key
        STRATEGY_REGISTRY[key] = cls
        STRATEGY_META.setdefault(key, {"key": key})
        return cls

    return decorator


def get_strategy_class(key: str) -> type[BaseStrategy] | None:
    """按 key 查询策略类，未注册返回 None。"""
    return STRATEGY_REGISTRY.get(key)


def all_strategy_keys() -> list[str]:
    """返回所有已注册策略 key。"""
    return list(STRATEGY_REGISTRY.keys())
