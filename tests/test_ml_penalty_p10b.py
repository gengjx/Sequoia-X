"""P10b ML 因子尾部抑制测试：近期 IC 稳定性动态门控。

覆盖：
  - _ml_stability_penalty：连续软惩罚（强 IC→1.0、弱 IC→下限、线性中点、单调）
  - _recent_ml_ic_mean：run_date 级 IC 均值、as-of PIT 截断、样本不足返回 None
  - 注入点 as-of 惩罚：强周期满权重、衰退期降至下限、无历史不惩罚
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sequoia_x.analysis.factor import (
    ML_IC_FULL,
    ML_IC_WEAK,
    ML_PENALTY_FLOOR,
    _ml_stability_penalty,
    _recent_ml_ic_mean,
)


# ===========================================================================
# _ml_stability_penalty：连续软惩罚乘子
# ===========================================================================
class TestMlStabilityPenalty:
    def test_strong_ic_full_weight(self):
        """近期 IC ≥ ML_IC_FULL → 满权重 1.0。"""
        assert _ml_stability_penalty(ML_IC_FULL) == 1.0
        assert _ml_stability_penalty(0.10) == 1.0
        assert _ml_stability_penalty(0.20) == 1.0

    def test_weak_ic_floor(self):
        """近期 IC ≤ ML_IC_WEAK → 降权下限。"""
        assert _ml_stability_penalty(ML_IC_WEAK) == ML_PENALTY_FLOOR
        assert _ml_stability_penalty(0.0) == ML_PENALTY_FLOOR
        assert _ml_stability_penalty(-0.02) == ML_PENALTY_FLOOR  # 衰退负 IC 触底

    def test_midpoint_linear(self):
        """中间值线性插值：FULL/2 中点 → 0.5*(1+FLOOR)。"""
        mid_ic = (ML_IC_WEAK + ML_IC_FULL) / 2  # 0.0375
        expected = round(ML_PENALTY_FLOOR + (1.0 - ML_PENALTY_FLOOR) * 0.5, 4)
        assert _ml_stability_penalty(mid_ic) == pytest.approx(expected)

    def test_monotonic_increasing(self):
        """惩罚乘子随 IC 单调递增。"""
        ics = [-0.05, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, ML_IC_FULL, 0.10]
        penalties = [_ml_stability_penalty(ic) for ic in ics]
        assert penalties == sorted(penalties)

    def test_floor_never_broken(self):
        """惩罚不低于下限、不高于 1.0。"""
        for ic in [-0.5, 0.0, 0.005, 0.03, 0.05, 0.5, 1.0]:
            p = _ml_stability_penalty(ic)
            assert ML_PENALTY_FLOOR <= p <= 1.0


# ===========================================================================
# _recent_ml_ic_mean：run_date 级 IC 读取 + as-of 截断
# ===========================================================================
def _make_ml_scores_db(tmp_path: Path, run_dates_ic: list[tuple[str, float]]) -> str:
    """构造含 ml_scores 多 run_date IC 的临时 DB。每个 run_date 插一条汇总行。"""
    db = str(tmp_path / "ml_ic.db")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ml_scores "
            "(run_date TEXT, symbol TEXT, ml_score REAL, ic_mean REAL, icir REAL, "
            "t_stat REAL, model_version TEXT, PRIMARY KEY (run_date, symbol))"
        )
        for run_date, ic in run_dates_ic:
            conn.execute(
                "INSERT INTO ml_scores VALUES (?,?,?, ?,?,?,?)",
                (run_date, "000001", 0.5, ic, 1.0, 3.0, "test"),
            )
        conn.commit()
    return db


class TestRecentMlIcMean:
    def test_none_when_no_data(self, tmp_path):
        """无 ml_scores 数据 → None。"""
        db = str(tmp_path / "empty.db")
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS ml_scores "
                "(run_date TEXT, symbol TEXT, ml_score REAL, ic_mean REAL, icir REAL, "
                "t_stat REAL, model_version TEXT, PRIMARY KEY (run_date, symbol))"
            )
            conn.commit()
        assert _recent_ml_ic_mean(db, n=6) is None

    def test_none_when_insufficient_months(self, tmp_path):
        """可用 run_date < 3 → None（样本不足不惩罚）。"""
        db = _make_ml_scores_db(tmp_path, [
            ("2025-01-31", 0.10), ("2025-02-28", 0.08),
        ])
        assert _recent_ml_ic_mean(db, n=6) is None

    def test_live_takes_latest_n(self, tmp_path):
        """as_of=None 取全部最近 n 个 run_date 的 IC 均值。"""
        db = _make_ml_scores_db(tmp_path, [
            ("2025-01-31", 0.10), ("2025-02-28", 0.10), ("2025-03-31", 0.10),
            ("2025-04-30", 0.02), ("2025-05-31", 0.02), ("2025-06-30", 0.02),
        ])
        # n=6 → 全部 → mean=0.06
        assert _recent_ml_ic_mean(db, n=6) == pytest.approx(0.06, abs=1e-4)
        # n=3 → 最近 3 个 → mean=0.02
        assert _recent_ml_ic_mean(db, n=3) == pytest.approx(0.02, abs=1e-4)

    def test_asof_truncates_future(self, tmp_path):
        """as_of_date 非 None 时只取 ≤ as_of 的 run_date（PIT，无未来函数）。"""
        db = _make_ml_scores_db(tmp_path, [
            ("2024-06-30", 0.10), ("2024-09-30", 0.10), ("2024-12-31", 0.10),
            ("2025-01-31", 0.01), ("2025-02-28", 0.00), ("2025-03-31", -0.02),
            ("2025-06-30", 0.15),  # 衰退后复苏（未来）
        ])
        # as_of=2025-03-31 → ≤ as_of 的最近 6 个 = 2024-06..2025-03
        r = _recent_ml_ic_mean(db, as_of_date="2025-03-31", n=6)
        assert r == pytest.approx((0.10 + 0.10 + 0.10 + 0.01 + 0.00 - 0.02) / 6, abs=1e-4)
        # as_of=2025-04-01 → 同上 6 个（2025-06 被截断，未来函数防护）
        r_apr = _recent_ml_ic_mean(db, as_of_date="2025-04-01", n=6)
        assert r_apr == r  # 未来复苏月未纳入
        # 衰退期近期 IC 弱（< ML_IC_FULL → 触发降权）
        assert r < ML_IC_FULL

    def test_asof_before_all_returns_none(self, tmp_path):
        """as_of 早于所有 run_date → 无可用数据 → None。"""
        db = _make_ml_scores_db(tmp_path, [("2025-06-30", 0.10), ("2025-07-31", 0.10)])
        assert _recent_ml_ic_mean(db, as_of_date="2024-01-01", n=6) is None

    def test_handles_null_ic(self, tmp_path):
        """ic_mean 为 NULL 的行被跳过（不影响均值）。"""
        db = str(tmp_path / "null_ic.db")
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS ml_scores "
                "(run_date TEXT, symbol TEXT, ml_score REAL, ic_mean REAL, icir REAL, "
                "t_stat REAL, model_version TEXT, PRIMARY KEY (run_date, symbol))"
            )
            conn.execute("INSERT INTO ml_scores VALUES ('2025-01-31','000001',0.5,0.10,1,3,'t')")
            conn.execute("INSERT INTO ml_scores VALUES ('2025-02-28','000001',0.5,NULL,NULL,NULL,'t')")
            conn.execute("INSERT INTO ml_scores VALUES ('2025-03-31','000001',0.5,0.08,1,3,'t')")
            conn.execute("INSERT INTO ml_scores VALUES ('2025-04-30','000001',0.5,0.09,1,3,'t')")
            conn.commit()
        # NULL 行被 WHERE ic_mean IS NOT NULL 排除；3 个有效月均=(0.10+0.08+0.09)/3
        r = _recent_ml_ic_mean(db, n=6)
        assert r == pytest.approx(0.09, abs=1e-4)
