"""P17 策略工坊修复测试：bottom 放松出票 + flag 降级 + marginal_alpha 列保留。

覆盖：
  - bottom 实盘阈值放松后出票（旧阈值下 0 命中场景）
  - flag 退出 ACTIVE_STRATEGY_KEYS 且 role 降级
  - save_strategy_weights 列保留 UPSERT 不清零 marginal_alpha
  - 新插入 strategy_key 的 marginal_alpha 默认 0
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest


def _insert_bottom_bars(db_path: str, symbol: str = "000777") -> None:
    """插入一段满足"新宽松阈值"但不满足"旧严格阈值"的 K 线。

    设计：
      - 第5日 high=100，近20日 high_20=100，末日 close=87 → 跌幅13%
        （>12 新阈值通过；<15 旧阈值不通过）
      - 末日 volume=3,000,000，前4日各1,000,000 → vol_ma5=1.4M
        3M>1.4M*2=2.8M（新阈值通过）；3M<1.4M*3=4.2M（旧阈值不通过）
      - 末日 open=85 close=87 low=83.5 high=88 → body=2, 下影线=1.5
        1.5>2*0.5=1.0（新阈值通过）；1.5<2*2=4.0（旧阈值不通过）
      - turnover=87*3,000,000=261M > 50M（新/旧均通过）
    """
    dates = pd.bdate_range("2025-01-01", periods=21)
    base = [95.0, 96.0, 97.0, 98.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0]
    base += [94.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0, 87.5, 87.0, 86.5, 86.0]
    closes = base[:21]
    with sqlite3.connect(db_path) as conn:
        for i, d in enumerate(dates):
            c = closes[i]
            if i == len(dates) - 1:
                open_p, high_p, low_p, close_p = 85.0, 88.0, 83.5, 87.0
                vol = 3_000_000.0
            else:
                open_p, high_p, low_p = c * 0.995, c * 1.005, c * 0.99
                close_p = c
                vol = 1_000_000.0
            turnover = close_p * vol
            conn.execute(
                "INSERT OR REPLACE INTO stock_daily "
                "(symbol, date, open, high, low, close, volume, turnover, "
                "turn, pct_chg, tradestatus, isst) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (symbol, d.strftime("%Y-%m-%d"), open_p, high_p, low_p, close_p,
                 vol, turnover, 2.5, 1.0, 1, 0),
            )
        conn.commit()


def _load_daily(db_path: str, symbol: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            "SELECT date, open, high, low, close, volume, turnover "
            "FROM stock_daily WHERE symbol=? ORDER BY date",
            conn, params=(symbol,),
        )


def test_bottom_relaxed_picks_symbol(tmp_path):
    """宽松阈值下 bottom 应选出该 symbol。"""
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.strategy.bottom_volume import BottomVolumeStrategy

    db_path = str(tmp_path / "test.db")
    DataEngine(Settings(db_path=db_path))
    _insert_bottom_bars(db_path, "000777")

    engine = DataEngine(Settings(db_path=db_path))
    strat = BottomVolumeStrategy(engine=engine, settings=Settings(db_path=db_path))
    result = strat.run()

    assert "000777" in result, "宽松阈值下应选出 000777"


def test_bottom_strict_would_reject_same_data(tmp_path):
    """用旧严格阈值重算，确认同一数据在旧阈值下不命中（验证放松确实改变行为）。"""
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine

    db_path = str(tmp_path / "test.db")
    DataEngine(Settings(db_path=db_path))
    _insert_bottom_bars(db_path, "000777")

    df = _load_daily(db_path, "000777")
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["high_20"] = df["high"].rolling(20).max()
    last = df.iloc[-1]
    drawdown = (last["high_20"] - last["close"]) / last["high_20"] * 100
    body = abs(last["close"] - last["open"])
    lower_shadow = min(last["open"], last["close"]) - last["low"]

    assert not (drawdown > 15.0), "旧跌幅阈值15%应不通过（13%）"
    assert drawdown > 12.0, "新跌幅阈值12%应通过（13%）"
    assert not (last["volume"] > last["vol_ma5"] * 3), "旧放量3x应不通过"
    assert last["volume"] > last["vol_ma5"] * 2, "新放量2x应通过"
    assert not (body > 0 and lower_shadow > body * 2), "旧下影线2x应不通过"
    assert body > 0 and lower_shadow > body * 0.5, "新下影线0.5x应通过"


def test_flag_removed_from_active_keys():
    """flag 不再出现在 ACTIVE_STRATEGY_KEYS。"""
    from sequoia_x.strategy.registry import ACTIVE_STRATEGY_KEYS

    assert "flag" not in ACTIVE_STRATEGY_KEYS
    assert ACTIVE_STRATEGY_KEYS == ["multi_factor", "bottom"]


def test_flag_role_demoted():
    """flag 的 role 应为 demoted。"""
    from sequoia_x.strategy.registry import STRATEGY_META

    assert STRATEGY_META["flag"]["role"] == "demoted"


def test_marginal_alpha_preserved_on_resave(tmp_db):
    """save_strategy_weights 刷新质量分时不清零已有 marginal_alpha。"""
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine

    settings = Settings(db_path=tmp_db)
    engine = DataEngine(settings)
    engine.save_strategy_weights([
        {"strategy_key": "bottom", "quality_score": 53, "marginal_alpha": 24.5},
    ])
    weights = engine.load_strategy_weights()
    assert weights["bottom"]["marginal_alpha"] == pytest.approx(24.5)

    engine.save_strategy_weights([
        {"strategy_key": "bottom", "quality_score": 60},
    ])
    weights = engine.load_strategy_weights()
    assert weights["bottom"]["quality_score"] == 60, "质量分应刷新"
    assert weights["bottom"]["marginal_alpha"] == pytest.approx(24.5), "marginal_alpha 应保留"


def test_new_strategy_key_defaults_zero_marginal(tmp_db):
    """新插入的 strategy_key 若不带 marginal_alpha，默认为 0。"""
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine

    settings = Settings(db_path=tmp_db)
    engine = DataEngine(settings)
    engine.save_strategy_weights([
        {"strategy_key": "brand_new_strategy", "quality_score": 50},
    ])
    weights = engine.load_strategy_weights()
    assert weights["brand_new_strategy"]["marginal_alpha"] == 0


@pytest.fixture
def tmp_db(tmp_path):
    """创建临时 DB。"""
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.core.config import Settings
    db_path = str(tmp_path / "test.db")
    DataEngine(Settings(db_path=db_path))
    return db_path
