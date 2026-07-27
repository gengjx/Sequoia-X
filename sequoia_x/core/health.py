"""因子健康度监控器。

追踪数据源加载量 + 因子 nan 覆盖率，将"静默失败"转为结构化告警。
用于 multi_factor.run() 选股后输出健康度摘要，供决策层/飞书日报展示。
"""

from __future__ import annotations

import threading

COVERAGE_WARN = 0.20

_SOURCE_LABELS = {
    "finance": "财报",
    "fund_flow": "资金流",
    "lhb": "龙虎榜",
    "north": "北向",
    "margin": "融资融券",
    "fund_hold": "基金持仓",
    "block": "大宗交易",
    "holder": "股东户数",
    "index": "沪深300",
}


class FactorHealth:
    """线程安全的因子健康度追踪器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sources: dict[str, tuple[int, int]] = {}
        self._factor_coverage: dict[str, float] = {}

    def record_source(self, name: str, loaded: int, total: int) -> None:
        with self._lock:
            self._sources[name] = (loaded, total)

    def record_factor_coverage(self, factor: str, valid_ratio: float) -> None:
        with self._lock:
            self._factor_coverage[factor] = valid_ratio

    def record_coverage_from_df(self, df, factors: list[str]) -> None:
        """从因子 DataFrame 计算各因子的有效（非 nan）覆盖率。"""
        if df is None or len(df) == 0:
            return
        with self._lock:
            for f in factors:
                if f in df.columns:
                    col = df[f]
                    ratio = float(col.notna().sum()) / len(col) if len(col) > 0 else 0.0
                    self._factor_coverage[f] = ratio

    @property
    def source_coverage(self) -> dict[str, float]:
        with self._lock:
            result = {}
            for name, (loaded, total) in self._sources.items():
                result[name] = loaded / total if total > 0 else 0.0
            return result

    @property
    def degraded_sources(self) -> list[str]:
        """覆盖率 < COVERAGE_WARN 的数据源列表。"""
        with self._lock:
            degraded = []
            for name, (loaded, total) in self._sources.items():
                ratio = loaded / total if total > 0 else 0.0
                if ratio < COVERAGE_WARN:
                    degraded.append(name)
            return degraded

    def warning_line(self) -> str:
        """格式化告警行；无降级时返回空串。"""
        degraded = self.degraded_sources
        if not degraded:
            return ""
        parts = []
        for name in degraded:
            loaded, total = self._sources.get(name, (0, 0))
            pct = loaded / total * 100 if total > 0 else 0
            label = _SOURCE_LABELS.get(name, name)
            parts.append(f"{label}{pct:.0f}%")
        return f"⚠️ 因子健康度：{' '.join(parts)} [{len(degraded)}源降级]"

    def to_summary(self) -> dict:
        """结构化摘要供 API/落库。"""
        with self._lock:
            sources = {
                name: {"loaded": l, "total": t, "coverage": l / t if t > 0 else 0.0}
                for name, (l, t) in self._sources.items()
            }
            factors = dict(self._factor_coverage)
        degraded = self.degraded_sources
        return {
            "sources": sources,
            "factor_coverage": factors,
            "degraded_sources": degraded,
            "is_degraded": len(degraded) > 0,
        }
