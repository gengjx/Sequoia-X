"""大盘分析共享层：报告数据结构、常量与工具函数。

本模块为各 market_*.py mixin 子模块与 market.py 门面提供共享符号，
独立于具体计算逻辑，避免 mixin 与门面之间的循环导入。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# 主要指数定义：key -> (baostock 代码, 中文名)
_INDEX_DEFS: list[tuple[str, str, str]] = [
    ("sh", "sh.000001", "上证指数"),
    ("sz", "sz.399001", "深证成指"),
    ("cyb", "sz.399006", "创业板指"),
    ("sz50", "sh.000016", "上证50"),
    ("hs300", "sh.000300", "沪深300"),
]

_INDUSTRY_TABLE = "stock_industry"
_BOARD_TABLE = "stock_board_em"
_STOCK_BASIC_TABLE = "stock_basic"
_MARKET_CAP_TABLE = "stock_market_cap"


def _clean_industry_name(raw: str) -> str:
    """清洗证监会行业名：去除前缀分类码（如 'C39计算机...' -> '计算机...'）。"""
    if not raw:
        return "未分类"
    return re.sub(r"^[A-Z]\d+\s*", "", raw).strip() or "未分类"


@dataclass
class MarketReport:
    """结构化盘面报告（可序列化为 dict）。"""

    date: str
    generated_at: str
    data_source: str
    overview: dict = field(default_factory=dict)
    indices: list[dict] = field(default_factory=list)
    sectors: dict = field(default_factory=dict)
    sentiment: dict = field(default_factory=dict)
    news: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    fund_flow: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    valuation: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)
    limit_structure: dict = field(default_factory=dict)
    dragon_tiger: dict = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "generated_at": self.generated_at,
            "data_source": self.data_source,
            "overview": self.overview,
            "indices": self.indices,
            "sectors": self.sectors,
            "sentiment": self.sentiment,
            "news": self.news,
            "plan": self.plan,
            "fund_flow": self.fund_flow,
            "history": self.history,
            "valuation": self.valuation,
            "macro": self.macro,
            "limit_structure": self.limit_structure,
            "dragon_tiger": self.dragon_tiger,
            "risks": self.risks,
        }
