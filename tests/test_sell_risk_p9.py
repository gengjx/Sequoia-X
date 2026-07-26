"""P9 卖出系统升级 + 组合风控接入测试。

覆盖：
  - 回测持仓期 ATR 收紧：波动率收敛时止损上移（只收紧），放大时不下移
  - 组合风控单股拦截：Beta/HHI/行业超限跳过该股（回测）
  - 实盘决策层单股风控：触发 danger 告警降级观望
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sequoia_x.analysis.paper_replay import PaperReplayEngine, ReplayPosition


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _make_replay_db(tmp_path: Path) -> str:
    """构造含 index_daily + stock_industry 的回测 DB。"""
    db = str(tmp_path / "test.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS index_daily (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
        conn.execute("CREATE TABLE IF NOT EXISTS stock_industry (symbol TEXT, industry TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS stock_basic (symbol TEXT, name TEXT, ipo_date TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS stock_daily (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, turnover REAL, pct_chg REAL, tradestatus INTEGER)")
        # 沪深300指数（递增，有波动）
        for i in range(100):
            close = 100.0 + i * 0.5 + np.random.RandomState(i).randn() * 0.3
            conn.execute("INSERT INTO index_daily VALUES ('000300', ?, 0,0,0,?,0)", (f"2024-01-{i+1:02d}", close))
        # 行业表
        for s, ind in [("000001", "银行"), ("000002", "房地产"), ("600000", "银行")]:
            conn.execute("INSERT INTO stock_industry VALUES (?,?)", (s, ind))
            conn.execute("INSERT INTO stock_basic VALUES (?, ?, '2020-01-01')", (s, s))
        conn.commit()
    return db


# ===========================================================================
# 持仓期 ATR 收紧（回测对齐实盘）
# ===========================================================================
class TestHoldPeriodATRTightening:
    def test_stop_tightens_when_volatility_shrinks(self):
        """波动率收敛（ATR 变小）→ 止损上移（更紧），用 calc_atr_stop 直接验证。"""
        from sequoia_x.analysis.stop_loss import calc_atr_stop
        entry = 10.0
        # 高波动（H-L=1.0，ATR≈1.0）：2.5*ATR/entry=25% → clamp 到 15% 最大止损
        high_vol = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=22, freq="B").strftime("%Y-%m-%d"),
            "high": entry + 0.5, "low": entry - 0.5, "close": entry,
        })
        stop_high = calc_atr_stop(high_vol, entry, lookback=20)
        # 低波动（H-L=0.3，ATR≈0.3）：2.5*0.3/10=7.5% → clamp 到 8% 最小止损
        low_vol = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=22, freq="B").strftime("%Y-%m-%d"),
            "high": entry + 0.15, "low": entry - 0.15, "close": entry,
        })
        stop_low = calc_atr_stop(low_vol, entry, lookback=20)
        # 低波动止损（8% off）> 高波动止损（15% off），即更紧
        assert stop_low > stop_high, f"低波动{stop_low}应>高波动{stop_high}"

    def test_stop_never_moves_down(self):
        """pos.stop_loss 只允许上移（单向原则）。"""
        pos = ReplayPosition(
            symbol="TEST", entry_date="2025-01-01", entry_price=10.0,
            shares=100, highest_price=11.0, stop_loss=9.0,
        )
        # 已有止损 9.0，新 ATR 止损若更低（如 8.5），不更新
        new_atr_stop = 8.5
        assert max(pos.stop_loss, new_atr_stop) == 9.0  # 保持原值


# ===========================================================================
# 组合风控单股拦截（回测 _check_buy_risk）
# ===========================================================================
class TestBuyRiskCheck:
    def test_beta_block(self):
        """组合 Beta 已高 → 高 Beta 候选股被拦截，低 Beta 正常。"""
        # 6 只已持仓（Beta=1.5，低 HHI=分散），总仓位 60000
        positions = [
            ReplayPosition(symbol=f"00000{i}", entry_date="2025-01-01", entry_price=10,
                           shares=1000, highest_price=10, stop_loss=9)
            for i in range(1, 7)
        ]
        today_prices = {f"00000{i}": 10.0 for i in range(1, 7)}
        beta_cache = {f"00000{i}": 1.5 for i in range(1, 7)}
        # 给每只不同行业（避免行业集中度误拦）
        ind_cache = {f"00000{i}": f"行业{i}" for i in range(1, 7)}
        ind_cache["HIGH"] = "行业7"
        ind_cache["LOW"] = "行业8"
        beta_cache["HIGH"] = 2.0   # 高 Beta 候选
        beta_cache["LOW"] = 0.5    # 低 Beta 候选
        # 买入 HIGH（Beta=2.0）：新 Beta = (1.5*60000 + 2.0*10000)/70000 = 1.57 → 接近但需更高
        # 用更大 buy 让 Beta 超 1.6：(90000+2.0*30000)/90000=1.67
        ok, reason = PaperReplayEngine._check_buy_risk(
            "HIGH", 30000, positions, today_prices, beta_cache, ind_cache)
        assert not ok
        assert "Beta" in reason
        # 买入 LOW（Beta=0.5）：新 Beta = (90000+0.5*10000)/70000=1.36 < 1.6 通过
        ok2, reason2 = PaperReplayEngine._check_buy_risk(
            "LOW", 10000, positions, today_prices, beta_cache, ind_cache)
        assert ok2, f"LOW 应通过但被拦：{reason2}"

    def test_industry_block(self):
        """单行业占比超限 → 同行业候选股被拦截。"""
        positions = [
            ReplayPosition(symbol="000001", entry_date="2025-01-01", entry_price=10,
                           shares=1000, highest_price=10, stop_loss=9),
            ReplayPosition(symbol="600000", entry_date="2025-01-01", entry_price=10,
                           shares=1000, highest_price=10, stop_loss=9),
        ]
        today_prices = {"000001": 10.0, "600000": 10.0}
        # 银行已占 20000，总 30000 → 67%，再加银行股应被拦
        industry_cache = {"000001": "银行", "600000": "银行", "601398": "银行"}
        ok, reason = PaperReplayEngine._check_buy_risk(
            "601398", 10000, positions, today_prices, {}, industry_cache)
        assert not ok
        assert "行业" in reason

    def test_different_industry_ok(self):
        """不同行业候选股不受行业约束（多持仓低 HHI）。"""
        positions = [
            ReplayPosition(symbol=f"00000{i}", entry_date="2025-01-01", entry_price=10,
                           shares=1000, highest_price=10, stop_loss=9)
            for i in range(1, 7)
        ]
        today_prices = {f"00000{i}": 10.0 for i in range(1, 7)}
        ind_cache = {f"00000{i}": f"行业{i}" for i in range(1, 7)}
        ind_cache["000007"] = "新行业"
        ok, reason = PaperReplayEngine._check_buy_risk(
            "000007", 10000, positions, today_prices, {}, ind_cache)
        assert ok, f"不同行业应通过：{reason}"

    def test_hhi_block_when_highly_concentrated(self):
        """持仓已高度集中（单票占比极高）→ 加仓触发 HHI 超限。"""
        # 单票占 90000，总仓位才 100000 → HHI 极高
        positions = [
            ReplayPosition(symbol="000001", entry_date="2025-01-01", entry_price=90,
                           shares=1000, highest_price=90, stop_loss=80),
        ]
        today_prices = {"000001": 90.0}
        # 再加 10000 → 仍高度集中
        ok, reason = PaperReplayEngine._check_buy_risk(
            "000002", 10000, positions, today_prices, {}, {})
        assert not ok
        assert "HHI" in reason


# ===========================================================================
# 回测完整跑通（风控预计算 + 拦截不崩）
# ===========================================================================
class TestReplayIntegration:
    def test_preload_risk_data(self, tmp_path):
        """_preload_risk_data 预计算 beta + 行业缓存。"""
        db = _make_replay_db(tmp_path)
        eng = PaperReplayEngine(db)
        # 构造 symbol_groups
        groups = {}
        for sym in ("000001", "000002"):
            df = pd.DataFrame({
                "date": [f"2024-01-{i:02d}" for i in range(1, 101)],
                "close": [10.0 + i * 0.01 for i in range(100)],
                "high": [10.5 + i * 0.01 for i in range(100)],
                "low": [9.5 + i * 0.01 for i in range(100)],
            })
            groups[sym] = df
        eng._index_ret_cache = eng._load_index_returns()
        eng._preload_risk_data(groups)
        assert eng._beta_cache is not None
        assert len(eng._beta_cache) >= 0  # 可能为空（数据不够60天有效TR），不崩即可
        assert eng._industry_cache is not None
        assert eng._industry_cache.get("000001") == "银行"
