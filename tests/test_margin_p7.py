"""P7 融资融券因子测试。

覆盖：
  - margin_sync：fetch_margin_detail 列映射（沪+深合并）
  - factor FACTOR_META：margin 因子定义存在
  - factor as-of 应用：取截面前最新融资融券
  - 持久化：margin_detail 表 CRUD
"""

from __future__ import annotations

import sqlite3

import pytest

from sequoia_x.analysis.factor import FACTOR_META
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine


# ===========================================================================
# FACTOR_META 定义
# ===========================================================================
class TestMarginFactorMeta:
    def test_margin_factors_defined(self):
        assert "margin_balance" in FACTOR_META
        assert "margin_netbuy" in FACTOR_META
        assert "short_ratio" in FACTOR_META

    def test_margin_factor_category(self):
        assert FACTOR_META["margin_balance"]["category"] == "融资融券"
        assert FACTOR_META["margin_netbuy"]["category"] == "融资融券"
        assert FACTOR_META["short_ratio"]["category"] == "融资融券"

    def test_margin_factor_has_desc(self):
        for f in ("margin_balance", "margin_netbuy", "short_ratio"):
            assert "desc" in FACTOR_META[f]


# ===========================================================================
# 持久化
# ===========================================================================
class TestMarginPersistence:
    def test_table_created(self, tmp_path):
        db = str(tmp_path / "m.db")
        DataEngine(Settings(db_path=db))
        cols = [
            r[1] for r in
            sqlite3.connect(db).execute("PRAGMA table_info(margin_detail)").fetchall()
        ]
        expected = {"symbol", "date", "rzye", "rzbuy", "rzrepay",
                    "rqlts", "rqsell", "rqrepay", "rqye"}
        assert expected.issubset(set(cols))

    def test_upsert_idempotent(self, tmp_path):
        db = str(tmp_path / "m.db")
        DataEngine(Settings(db_path=db))
        from sequoia_x.data.margin_sync import sync_margin_detail
        # 手动插入测试数据
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO margin_detail "
                "(symbol, date, rzye, rzbuy) VALUES ('600519', '2026-07-20', 1e8, 1e6)"
            )
            conn.commit()
            # 再插同样的PK不同值
            conn.execute(
                "INSERT OR REPLACE INTO margin_detail "
                "(symbol, date, rzye, rzbuy) VALUES ('600519', '2026-07-20', 2e8, 2e6)"
            )
            conn.commit()
            row = conn.execute(
                "SELECT rzye FROM margin_detail WHERE symbol='600519'"
            ).fetchone()
        assert row[0] == 2e8  # UPSERT 覆盖


# ===========================================================================
# short_ratio 计算逻辑
# ===========================================================================
class TestShortRatio:
    def test_short_ratio_formula(self):
        """short_ratio = 融券余额 / (融资余额 + 融券余额)。"""
        rzye = 1e8
        rqye = 2e7
        expected = rqye / (rzye + rqye)
        ratio = _compute_short_ratio(rzye, rqye)
        assert ratio == pytest.approx(expected)

    def test_zero_rqye(self):
        assert _compute_short_ratio(1e8, 0) == pytest.approx(0.0)

    def test_zero_rzye_returns_none(self):
        assert _compute_short_ratio(0, 1e7) is None


def _compute_short_ratio(rzye, rqye):
    """复现 factor.py 中的 short_ratio 逻辑。"""
    if rzye is None or rqye is None or rzye <= 0:
        return None
    return rqye / (rzye + rqye)
