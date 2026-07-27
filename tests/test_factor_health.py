"""因子健康监控测试（FactorHealth + multi_factor 集成 + 决策降级标记）。

覆盖：
  - FactorHealth 纯函数单测（覆盖率/降级源/告警行/摘要）
  - multi_factor.run 集成（数据源降级时 last_health 正确记录）
  - 决策层降级标记（result["degraded"] 正确）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequoia_x.core.health import FactorHealth


# ── FactorHealth 纯函数单测 ──────────────────────────────────────

class TestFactorHealthUnit:
    """FactorHealth 监控器纯函数测试。"""

    def test_source_coverage_degraded(self):
        """覆盖率 <20% 的源被标记为降级。"""
        h = FactorHealth()
        h.record_source("fund_flow", 0, 500)
        h.record_source("finance", 480, 500)
        h.record_source("lhb", 50, 500)
        degraded = h.degraded_sources
        assert "fund_flow" in degraded
        assert "finance" not in degraded
        assert "lhb" in degraded

    def test_no_degradation_when_all_healthy(self):
        """所有源覆盖率 >=20% 时无降级。"""
        h = FactorHealth()
        h.record_source("finance", 400, 500)
        h.record_source("fund_flow", 200, 500)
        assert h.degraded_sources == []

    def test_warning_line_with_degradation(self):
        """有降级时告警行非空且包含源名和百分比。"""
        h = FactorHealth()
        h.record_source("fund_flow", 0, 500)
        h.record_source("north", 10, 500)
        line = h.warning_line()
        assert "⚠️" in line
        assert "资金流" in line
        assert "北向" in line
        assert "2源降级" in line

    def test_warning_line_empty_when_healthy(self):
        """无降级时告警行为空串。"""
        h = FactorHealth()
        h.record_source("finance", 450, 500)
        assert h.warning_line() == ""

    def test_warning_line_empty_when_no_sources(self):
        """未记录任何源时告警行为空串。"""
        h = FactorHealth()
        assert h.warning_line() == ""

    def test_to_summary_structure(self):
        """to_summary 返回结构化 dict。"""
        h = FactorHealth()
        h.record_source("finance", 400, 500)
        h.record_source("fund_flow", 0, 500)
        h.record_factor_coverage("roe", 0.95)
        h.record_factor_coverage("main_pct", 0.0)
        summary = h.to_summary()
        assert "sources" in summary
        assert "factor_coverage" in summary
        assert "degraded_sources" in summary
        assert "is_degraded" in summary
        assert summary["is_degraded"] is True
        assert summary["sources"]["finance"]["coverage"] == pytest.approx(0.8)
        assert summary["factor_coverage"]["roe"] == pytest.approx(0.95)

    def test_coverage_from_dataframe(self):
        """从 DataFrame 计算各因子有效覆盖率。"""
        h = FactorHealth()
        df = pd.DataFrame({
            "roe": [0.15, 0.20, np.nan, 0.10],
            "turnover": [1.5, 2.0, 3.0, np.nan],
            "main_pct": [np.nan, np.nan, np.nan, np.nan],
        })
        h.record_coverage_from_df(df, ["roe", "turnover", "main_pct"])
        summary = h.to_summary()
        assert summary["factor_coverage"]["roe"] == pytest.approx(0.75)
        assert summary["factor_coverage"]["turnover"] == pytest.approx(0.75)
        assert summary["factor_coverage"]["main_pct"] == pytest.approx(0.0)

    def test_zero_total_source(self):
        """total=0 的源覆盖率为 0（视为降级）。"""
        h = FactorHealth()
        h.record_source("north", 0, 0)
        assert "north" in h.degraded_sources

    def test_threshold_boundary(self):
        """恰好 20% 覆盖率不算降级（>=阈值）。"""
        h = FactorHealth()
        h.record_source("margin", 100, 500)
        assert "margin" not in h.degraded_sources

    def test_thread_safety(self):
        """多线程 record_source 不出错。"""
        import threading
        h = FactorHealth()
        def _worker():
            for i in range(100):
                h.record_source("test", i, 100)
        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert h.to_summary()["sources"]["test"]["total"] == 100


# ── multi_factor.run 集成测试 ────────────────────────────────────

class TestMultiFactorHealthIntegration:
    """multi_factor.run 的健康度集成测试。"""

    def test_run_sets_last_health(self, tmp_db):
        """run() 执行后 last_health 属性被设置。"""
        from sequoia_x.data.engine import DataEngine
        from sequoia_x.core.config import Settings
        from sequoia_x.strategy.multi_factor import MultiFactorStrategy

        settings = Settings(db_path=tmp_db)
        engine = DataEngine(settings)
        _seed_minimal_klines(engine)
        strat = MultiFactorStrategy(engine, settings)
        strat.run()
        assert strat.last_health is not None
        assert isinstance(strat.last_health, FactorHealth)

    def test_degraded_source_detected(self, tmp_db):
        """数据源缺失时 degraded_sources 包含该源。"""
        from sequoia_x.data.engine import DataEngine
        from sequoia_x.core.config import Settings
        from sequoia_x.strategy.multi_factor import MultiFactorStrategy

        settings = Settings(db_path=tmp_db)
        engine = DataEngine(settings)
        _seed_minimal_klines(engine)
        strat = MultiFactorStrategy(engine, settings)
        strat.run()
        health = strat.last_health
        summary = health.to_summary()
        assert summary["sources"]["fund_flow"]["loaded"] == 0
        assert "fund_flow" in summary["degraded_sources"]


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """创建临时 DB，先初始化 schema 再注入数据。"""
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.core.config import Settings
    db_path = str(tmp_path / "test.db")
    # DataEngine.__init__ 创建全部表 schema
    DataEngine(Settings(db_path=db_path))
    # 注入测试数据
    _seed_minimal_klines(db_path=db_path)
    return db_path


def _seed_minimal_klines(engine=None, db_path=None):
    """注入最小 stock_daily 数据供 multi_factor 运行。

    DataEngine.__init__ 已创建全部表 schema（含 stock_daily 13列），
    此函数只 INSERT 数据行，不创建表。
    """
    import sqlite3
    import pandas as pd
    if db_path is None and engine is not None:
        db_path = engine.db_path
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS stock_basic (symbol TEXT, name TEXT, ipo_date TEXT)")
        dates = pd.bdate_range("2024-01-01", periods=120)
        for sym in ("000001", "000002", "000003", "000004", "000005"):
            prices = np.cumprod(1 + np.random.randn(120) * 0.02) * 10
            for i, d in enumerate(dates):
                conn.execute(
                    "INSERT OR REPLACE INTO stock_daily "
                    "(symbol, date, open, high, low, close, volume, turnover, "
                    "turn, pct_chg, tradestatus, isst) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sym, d.strftime("%Y-%m-%d"), float(prices[i]*0.99),
                     float(prices[i]*1.01), float(prices[i]*0.98), float(prices[i]),
                     1000000.0, float(prices[i]*1000000), 2.5, 1.0, 1, 0),
                )
        for sym in ("000001", "000002", "000003", "000004", "000005"):
            conn.execute("INSERT OR REPLACE INTO stock_basic VALUES (?, ?, '2020-01-01')", (sym, "股票" + sym[-1]))
        conn.commit()
