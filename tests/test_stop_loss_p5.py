"""P5 卖出系统升级测试：ATR/波动率自适应止损（入场+持仓全链路统一）。

覆盖：
  - calc_atr_stop 数学：True Range ATR、clamp[8%,15%]、数据不足回退
  - resolve_entry_stop：散乱 decision 止损 → ATR 统一口径
  - paper_replay._calc_atr_stop 委托共享函数（行为不变）
  - position.scan_one 持仓期 ATR 单向收紧（只收紧不放宽）
  - paper_trade.auto_buy 入场 ATR 止损覆盖（写库）
"""

from __future__ import annotations

import pandas as pd
import pytest

from sequoia_x.analysis.stop_loss import (
    DEFAULT_FALLBACK_PCT, calc_atr_stop, resolve_entry_stop,
)


# ---------------------------------------------------------------------------
# 辅助：构造 True Range ATR = target 的 OHLCV
# ---------------------------------------------------------------------------
def _ohlcv(atr_target: float, entry: float = 10.0, n: int = 25) -> pd.DataFrame:
    """生成 n 根 K 线，每根 H-L=atr_target、close 恒定 → True Range 均值=atr_target。

    H-L=atr_target 且与前一收盘无跳空 → TR=max(H-L, |H-PrevC|, |L-PrevC|)=atr_target。
    """
    half = atr_target / 2
    dates = pd.date_range("2025-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "date": dates,
        "high": entry + half,
        "low": entry - half,
        "close": entry,
    })


# ===========================================================================
# calc_atr_stop 数学
# ===========================================================================
class TestCalcAtrStop:
    ENTRY = 10.0

    def test_low_vol_clamped_to_min(self):
        """低波动(ATR=0.2)：2.5×0.2/10=5% < 8% 下限 → 封顶到 -8% → 9.2。"""
        stop = calc_atr_stop(_ohlcv(0.2), self.ENTRY)
        assert stop == pytest.approx(self.ENTRY * (1 - 0.08))

    def test_mid_vol_uses_raw_atr(self):
        """中波动(ATR=0.5)：2.5×0.5/10=12.5% ∈ [8%,15%] → 8.75。"""
        stop = calc_atr_stop(_ohlcv(0.5), self.ENTRY)
        assert stop == pytest.approx(self.ENTRY * (1 - 0.125))

    def test_high_vol_capped_at_max(self):
        """高波动(ATR=0.8)：2.5×0.8/10=20% > 15% 上限 → 封顶到 -15% → 8.5。"""
        stop = calc_atr_stop(_ohlcv(0.8), self.ENTRY)
        assert stop == pytest.approx(self.ENTRY * (1 - 0.15))

    def test_insufficient_data_fallback(self):
        """K线<21根 → 回退 12% 默认止损。"""
        stop = calc_atr_stop(_ohlcv(0.5, n=10), self.ENTRY)
        assert stop == pytest.approx(self.ENTRY * (1 - DEFAULT_FALLBACK_PCT))

    def test_empty_df_fallback(self):
        """空 DataFrame → 回退。"""
        assert calc_atr_stop(pd.DataFrame(), self.ENTRY) == pytest.approx(8.8)
        assert calc_atr_stop(None, self.ENTRY) == pytest.approx(8.8)

    def test_zero_atr_fallback(self):
        """H==L（ATR=0）→ 回退 12%。"""
        df = _ohlcv(0.0)
        assert calc_atr_stop(df, self.ENTRY) == pytest.approx(8.8)

    def test_as_of_date_excludes_future(self):
        """as_of_date 仅取 ≤ 该日期的 K 线（防未来函数）。"""
        df = _ohlcv(0.5, n=25)  # 2025-01-01 ~ 2025-02-03(B)
        full = calc_atr_stop(df, self.ENTRY, as_of_date="2025-12-31")  # 全部可用
        cut = calc_atr_stop(df, self.ENTRY, as_of_date="2025-01-15")   # <21根 → 回退
        assert full == pytest.approx(8.75)
        assert cut == pytest.approx(8.8)  # 1月15日只有 ~11 根

    def test_custom_params(self):
        """自定义 atr_mult/min/max 生效。"""
        stop = calc_atr_stop(_ohlcv(0.5), self.ENTRY, atr_mult=1.0,
                             min_pct=0.05, max_pct=0.30)
        # 1.0×0.5/10=5% ∈ [5%,30%] → 9.5
        assert stop == pytest.approx(9.5)

    def test_invalid_entry(self):
        """入场价<=0 → 返回 0（保护性）。"""
        assert calc_atr_stop(_ohlcv(0.5), 0.0) == 0.0


# ===========================================================================
# resolve_entry_stop：散乱止损 → ATR 统一口径
# ===========================================================================
class TestResolveEntryStop:
    def test_raw_zero_uses_atr(self):
        assert resolve_entry_stop(0.0, 10.0, 8.5) == 8.5

    def test_raw_none_uses_atr(self):
        assert resolve_entry_stop(None, 10.0, 8.5) == 8.5

    def test_wide_stop_uses_atr(self):
        """止损距离 >15%（过宽）→ 用 ATR。"""
        # raw=8.0 → 距离 20% > 15% → 用 atr 8.5
        assert resolve_entry_stop(8.0, 10.0, 8.5) == 8.5

    def test_normal_stop_kept(self):
        """止损距离 ≤15%（合理）→ 保留 decision 值。"""
        assert resolve_entry_stop(8.8, 10.0, 8.5) == 8.8  # 距离 12%

    def test_tight_stop_kept(self):
        """比 ATR 更紧的止损（如 -8%）→ 保留（不强制放宽）。"""
        assert resolve_entry_stop(9.2, 10.0, 8.5) == 9.2  # 距离 8%

    def test_boundary_exactly_15pct_kept(self):
        """距离恰好 15% → 不超过阈值，保留。"""
        assert resolve_entry_stop(8.5, 10.0, 8.0) == 8.5  # 距离 15%

    def test_boundary_just_over_15pct_uses_atr(self):
        """距离略超 15% → 用 ATR。"""
        assert resolve_entry_stop(8.49, 10.0, 8.8) == 8.8  # 距离 15.1%

    def test_negative_entry_falls_back_to_atr(self):
        assert resolve_entry_stop(9.0, 0.0, 8.5) == 8.5


# ===========================================================================
# paper_replay._calc_atr_stop 委托共享函数（行为不变）
# ===========================================================================
class TestReplayDelegation:
    def test_delegation_matches_shared(self):
        """回测 _calc_atr_stop == 共享 calc_atr_stop（有数据时）。"""
        from sequoia_x.analysis.paper_replay import PaperReplayEngine
        df = _ohlcv(0.5)
        groups = {"600519": df}
        delegated = PaperReplayEngine._calc_atr_stop(groups, "600519", "2025-12-31", 10.0)
        direct = calc_atr_stop(df, 10.0, as_of_date="2025-12-31")
        assert delegated == pytest.approx(direct)

    def test_no_symbol_uses_max_pct(self):
        """symbol 不在 groups → 用 max_pct(15%) 保守止损。"""
        from sequoia_x.analysis.paper_replay import PaperReplayEngine
        stop = PaperReplayEngine._calc_atr_stop({}, "000001", "2025-12-31", 10.0)
        assert stop == pytest.approx(10.0 * (1 - 0.15))

    def test_insufficient_uses_fallback(self):
        """数据不足 → 12% 回退。"""
        from sequoia_x.analysis.paper_replay import PaperReplayEngine
        groups = {"600519": _ohlcv(0.5, n=10)}
        stop = PaperReplayEngine._calc_atr_stop(groups, "600519", "2025-12-31", 10.0)
        assert stop == pytest.approx(8.8)


# ===========================================================================
# position.scan_one：持仓期 ATR 单向收紧
# ===========================================================================
class TestScanOneAtrTighten:
    @staticmethod
    def _make_db(tmp_path, entry, initial_stop, atr_target=0.5, n=30):
        """建临时 DB（stock_daily + portfolio_holding），返回 db_path。"""
        import sqlite3
        from sequoia_x.core.config import Settings
        from sequoia_x.data.engine import DataEngine

        db = str(tmp_path / "pos.db")
        DataEngine(Settings(db_path=db))  # 建全部表

        half = atr_target / 2
        dates = pd.date_range("2025-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
        rows = [(d, "TEST", entry + half, entry - half, entry, 1000,
                 atr_target * 100, 0.0, 1, 0) for d in dates]
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO stock_daily "
                "(date, symbol, high, low, close, volume, turn, pct_chg, tradestatus, isst) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)", rows,
            )
            conn.execute(
                "INSERT INTO portfolio_holding "
                "(symbol,name,entry_price,shares,entry_date,stop_loss,initial_stop,"
                "target,grade,hit_strategies,cost,status,notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("TEST", "测试", entry, 100, dates[-1], initial_stop, initial_stop,
                 0, "", "", entry * 100, "open", ""),
            )
            conn.commit()
        return db, dates

    def _tracker(self, db):
        from sequoia_x.core.config import Settings
        from sequoia_x.data.engine import DataEngine
        from sequoia_x.analysis.position import PositionTracker
        eng = DataEngine(Settings(db_path=db))
        return PositionTracker(eng, Settings(db_path=db))

    def test_atr_tightens_wide_stop(self, tmp_path):
        """原止损过宽(-18%) → ATR 收紧到 -12.5%（ATR=0.5,2.5×=12.5%）。"""
        db, _ = self._make_db(tmp_path, entry=10.0, initial_stop=8.2, atr_target=0.5)
        tr = self._tracker(db)
        h = tr.list_holdings("open")[0]
        sig = tr.scan_one(h)
        # ATR 止损 = 10*(1-0.125)=8.75 > 原 8.2 → 收紧到 8.75
        assert sig.new_stop == pytest.approx(8.75, abs=0.01)

    def test_atr_does_not_loosen(self, tmp_path):
        """原止损已紧(-8%) → ATR(8.75) 不会放宽，保持原值。"""
        db, _ = self._make_db(tmp_path, entry=10.0, initial_stop=9.2, atr_target=0.5)
        tr = self._tracker(db)
        h = tr.list_holdings("open")[0]
        sig = tr.scan_one(h)
        # ATR=8.75 < 9.2，不应放宽；硬止损未触发且 r_mult≈0 → new_stop 不应低于 9.2
        assert sig.new_stop <= 9.2 + 0.01
        assert sig.action != "止损清仓"

    def test_atr_unidirectional(self, tmp_path):
        """单向原则：ATR 算出的止损只允许上移。"""
        db, _ = self._make_db(tmp_path, entry=10.0, initial_stop=8.5, atr_target=0.2)
        tr = self._tracker(db)
        h = tr.list_holdings("open")[0]
        sig = tr.scan_one(h)
        # 低波动 ATR=0.2 → 止损=10*(1-0.08)=9.2 > 8.5 → 收紧到 9.2
        assert sig.new_stop >= 8.5 - 0.01  # 不下移
        assert sig.new_stop == pytest.approx(9.2, abs=0.01)


# ===========================================================================
# paper_trade.auto_buy：入场 ATR 止损覆盖（写库）
# ===========================================================================
class TestAutoBuyEntryStop:
    @staticmethod
    def _make_db(tmp_path, entry=10.0, atr_target=0.5, n=30):
        import sqlite3
        from sequoia_x.core.config import Settings
        from sequoia_x.data.engine import DataEngine

        db = str(tmp_path / "pt.db")
        DataEngine(Settings(db_path=db))
        half = atr_target / 2
        dates = pd.date_range("2025-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
        rows = [(d, "TEST", entry + half, entry - half, entry, 1000,
                 atr_target * 100, 0.0, 1, 0) for d in dates]
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO stock_daily "
                "(date, symbol, high, low, close, volume, turn, pct_chg, tradestatus, isst) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)", rows,
            )
            conn.commit()
        return db, dates[-1]

    def test_zero_stop_overridden_by_atr(self, tmp_path):
        """decision 传 stop_loss=0 → auto_buy 用 ATR 重算写入持仓。"""
        from sequoia_x.core.config import Settings
        from sequoia_x.analysis.paper_trade import PaperTradeEngine

        db, today = self._make_db(tmp_path, entry=10.0, atr_target=0.5)
        engine = PaperTradeEngine(Settings(db_path=db))
        atr_stop = engine._calc_entry_atr_stop("TEST", today, 10.0)
        # ATR=0.5 → 2.5×0.5/10=12.5% → 8.75
        assert atr_stop == pytest.approx(8.75, abs=0.01)

        result = engine.auto_buy({
            "market_state": {"state": "bull", "score": 70, "position_scale": 0.8},
            "buy_list": [{
                "symbol": "TEST", "name": "测试", "score": 80,
                "stop_loss": 0, "target": 0, "grade": "B",
                "hit_strategies": "", "action": "建仓",
            }],
        })
        assert len(result["bought"]) == 1
        import sqlite3
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT stop_loss, initial_stop FROM paper_holdings WHERE symbol='TEST'"
            ).fetchone()
        # 写入的入场止损应为 ATR 值（非 0）
        assert row[0] == pytest.approx(8.75, abs=0.01)
        assert row[1] == pytest.approx(8.75, abs=0.01)

    def test_wide_stop_overridden_by_atr(self, tmp_path):
        """decision 传过宽止损(距离>15%) → 用 ATR 重算。"""
        from sequoia_x.core.config import Settings
        from sequoia_x.analysis.paper_trade import PaperTradeEngine

        db, today = self._make_db(tmp_path, entry=10.0, atr_target=0.5)
        engine = PaperTradeEngine(Settings(db_path=db))
        result = engine.auto_buy({
            "market_state": {"state": "bull", "score": 70, "position_scale": 0.8},
            "buy_list": [{
                "symbol": "TEST", "name": "测试", "score": 80,
                "stop_loss": 7.5,   # 距离 25% > 15%
                "target": 0, "grade": "B",
                "hit_strategies": "", "action": "建仓",
            }],
        })
        assert len(result["bought"]) == 1
        import sqlite3
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT stop_loss FROM paper_holdings WHERE symbol='TEST'"
            ).fetchone()
        assert row[0] == pytest.approx(8.75, abs=0.01)  # ATR 值，非 7.5

    def test_normal_stop_kept(self, tmp_path):
        """decision 传合理止损(距离≤15%) → 保留 decision 值。"""
        from sequoia_x.core.config import Settings
        from sequoia_x.analysis.paper_trade import PaperTradeEngine

        db, today = self._make_db(tmp_path, entry=10.0, atr_target=0.5)
        engine = PaperTradeEngine(Settings(db_path=db))
        result = engine.auto_buy({
            "market_state": {"state": "bull", "score": 70, "position_scale": 0.8},
            "buy_list": [{
                "symbol": "TEST", "name": "测试", "score": 80,
                "stop_loss": 8.8,   # 距离 12% ≤ 15% → 保留
                "target": 0, "grade": "B",
                "hit_strategies": "", "action": "建仓",
            }],
        })
        assert len(result["bought"]) == 1
        import sqlite3
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT stop_loss FROM paper_holdings WHERE symbol='TEST'"
            ).fetchone()
        assert row[0] == pytest.approx(8.8, abs=0.01)  # 保留 decision 值
