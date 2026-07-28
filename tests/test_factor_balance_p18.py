"""P18 资产负债表+杜邦杠杆因子测试。

覆盖：
  - _AK_MAP 含资产负债率+权益乘数映射
  - compute_factors 含 debt_ratio + roe_leverage
  - FACTOR_META 含新因子定义
  - stock_finance 表含新列
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest


def _ohlcv(n: int) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes = [10.0 * (1.01 ** i) for i in range(n)]
    close = pd.Series(closes, index=dates)
    return pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": pd.Series([1_000_000] * n, index=dates),
        "turnover": pd.Series([c * 1_000_000 for c in closes], index=dates),
    }, index=dates)


def test_ak_map_has_balance_mappings():
    """_AK_MAP 含资产负债率+权益乘数映射。"""
    from sequoia_x.data.finance_sync import _AK_MAP

    assert ("常用指标", "资产负债率") in _AK_MAP
    assert ("财务风险", "权益乘数") in _AK_MAP
    assert _AK_MAP[("常用指标", "资产负债率")] == ("liability_to_asset", 100.0)
    assert _AK_MAP[("财务风险", "权益乘数")] == ("equity_multiplier", 1.0)


def test_factor_meta_has_new_factors():
    """FACTOR_META 含 debt_ratio + roe_leverage。"""
    from sequoia_x.analysis.factor import FACTOR_META

    assert "debt_ratio" in FACTOR_META
    assert "roe_leverage" in FACTOR_META
    assert FACTOR_META["debt_ratio"]["category"] == "偿债能力"
    assert FACTOR_META["roe_leverage"]["category"] == "盈利质量"


def test_compute_factors_with_balance_data():
    """finance dict 含负债率/权益乘数时因子正确取值。"""
    from sequoia_x.analysis.factor import compute_factors

    factors = compute_factors(
        _ohlcv(25),
        finance={"liability_to_asset": 0.65, "equity_multiplier": 2.8,
                 "roe": 0.12, "np_margin": 0.08},
    )
    assert factors["debt_ratio"] == pytest.approx(0.65)
    assert factors["roe_leverage"] == pytest.approx(2.8)


def test_compute_factors_without_finance():
    """finance 为 None 时 debt_ratio/roe_leverage 为 None（nan→None 转换）。"""
    from sequoia_x.analysis.factor import compute_factors

    factors = compute_factors(_ohlcv(25), finance=None)
    assert factors.get("debt_ratio") is None
    assert factors.get("roe_leverage") is None


def test_compute_factors_finance_missing_fields():
    """finance dict 有值但缺负债率字段时 debt_ratio 为 None。"""
    from sequoia_x.analysis.factor import compute_factors

    factors = compute_factors(
        _ohlcv(25),
        finance={"roe": 0.12, "np_margin": 0.08},
    )
    assert factors.get("debt_ratio") is None
    assert factors.get("roe_leverage") is None


def test_stock_finance_columns_exist(tmp_db):
    """stock_finance 建表 SQL 含新列（验证迁移DDL）。"""
    with sqlite3.connect(tmp_db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS stock_finance ("
            "symbol TEXT NOT NULL, stat_date TEXT NOT NULL, report_date TEXT,"
            "roe REAL, np_margin REAL, gp_margin REAL,"
            "liability_to_asset REAL, equity_multiplier REAL,"
            "PRIMARY KEY (symbol, stat_date))"
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(stock_finance)")}
    assert "liability_to_asset" in cols
    assert "equity_multiplier" in cols


@pytest.fixture
def tmp_db(tmp_path):
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    db_path = str(tmp_path / "test.db")
    DataEngine(Settings(db_path=db_path))
    return db_path
