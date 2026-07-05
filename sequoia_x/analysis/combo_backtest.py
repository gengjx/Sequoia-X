"""组合历史回测引擎：向量化解算策略信号 + 持有期收益对比。

核心思路：对每只股票预计算策略信号序列（布尔），统计信号触发后的持有期收益，
横向对比各组合的实战效果（平均收益/胜率/夏普/样本数）。

为控制耗时，采用采样：随机抽取 N 只股票 × 多个持有期窗口。
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine

logger = get_logger(__name__)

# 策略信号 → 计算函数名 映射
SIGNAL_FUNCS = {
    "ma_volume", "turtle", "pullback", "bottom", "rps",
}


def _compute_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    """计算所有策略的逐日信号序列（布尔，True=当天触发）。

    复用 StockAnalyzer._detect_strategy_hits 的核心逻辑，但面向整个序列向量化。
    """
    if len(df) < 20:
        return {}
    close, high, low, volume, turnover = (
        df["close"], df["high"], df["low"], df["volume"], df["turnover"]
    )
    signals: dict[str, pd.Series] = {}

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    vol_ma20 = volume.rolling(20).mean()
    vol_ma5 = volume.rolling(5).mean()
    high_20 = high.shift(1).rolling(20).max()
    high_20b = high.rolling(20).max()

    # 均线放量：金叉 + 放量
    cross = (ma5.shift(1) < ma20.shift(1)) & (ma5 > ma20)
    signals["ma_volume"] = cross & (volume > vol_ma20 * 1.5)

    # 海龟突破：20日新高 + 阳线 + 成交过亿
    breakout = close > high_20
    is_yang = close > df["open"]
    signals["turtle"] = breakout & (turnover > 1e8) & is_yang

    # 缩量回踩：多头排列 + 回踩MA5 + 缩量
    uptrend = (ma5 > ma10) & (ma10 > ma20)
    near_ma5 = (close - ma5).abs() / ma5 <= 0.015
    recent_vol = volume.rolling(3).mean()
    shrink = recent_vol < vol_ma20 * 0.7
    signals["pullback"] = uptrend & near_ma5 & shrink

    # 底部放量：超跌 + 放量3倍 + 下影阳线
    drawdown = (high_20b - close) / high_20b * 100
    body = (close - df["open"]).abs()
    lower_shadow = df[["open", "close"]].min(axis=1) - low
    signals["bottom"] = (
        (drawdown > 15) & (volume > vol_ma5 * 3) & (close > df["open"])
        & (body > 0) & (lower_shadow > body * 2)
    )

    # RPS强势：120日涨幅>50% + 接近新高
    if len(df) >= 120:
        ret_120 = (close / close.shift(120) - 1) * 100
        high_120 = high.rolling(120).max()
        signals["rps"] = (ret_120 > 50) & (close > high_120 * 0.9)
    else:
        signals["rps"] = pd.Series(False, index=df.index)

    return signals


class ComboBacktester:
    """组合历史回测器。"""

    def __init__(self, engine: DataEngine, settings: Settings) -> None:
        self.engine = engine
        self.settings = settings

    def run(
        self, combos: dict[str, list[str]], hold_days: list[int] | None = None,
        sample_size: int = 500, seed: int = 42,
    ) -> dict:
        """回测多个组合，返回横向对比报告。"""
        hold_days = hold_days or [5, 10, 20]
        strategy_returns = self._collect_returns(hold_days, sample_size, seed)
        processed = strategy_returns.pop("__processed__", 0)

        results = []
        for combo_name, skeys in combos.items():
            valid_keys = [k for k in skeys if k in SIGNAL_FUNCS]
            for h in hold_days:
                all_rets: list[float] = []
                for k in valid_keys:
                    all_rets.extend(strategy_returns.get(k, {}).get(h, []))
                results.append({
                    "combo": combo_name, "hold_days": h,
                    **self._stats(all_rets),
                })
        return {"combos": results, "sample_size": processed}

    def run_resonance(self, hold_days: list[int] | None = None,
                      sample_size: int = 500, seed: int = 42) -> dict:
        """共振度分档回测：统计不同共振度（1/2/3+）下的持有期收益。

        核心验证：多策略共振是否真能带来超额收益（决策中枢定级矩阵的基础假设）。
        """
        hold_days = hold_days or [5, 10, 20]
        symbols = self.engine.get_local_symbols()
        if sample_size and len(symbols) > sample_size:
            rng = random.Random(seed)
            symbols = rng.sample(symbols, sample_size)
        logger.info(f"共振回测：采样 {len(symbols)} 只股票")

        # {共振度档位: {hold: [收益率]}}
        bands = {"1": {h: [] for h in hold_days},
                 "2": {h: [] for h in hold_days},
                 "3+": {h: [] for h in hold_days}}
        processed = 0

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < 60:
                    continue
                df = df.reset_index(drop=True)
                signals = _compute_signals(df)
                if not signals:
                    continue
                # 计算每日共振度（同时命中几个策略）
                sig_df = pd.DataFrame({k: v.fillna(False) for k, v in signals.items()})
                resonance_count = sig_df.sum(axis=1)  # 每日共振数
                for h in hold_days:
                    for idx in resonance_count.index:
                        rc = resonance_count.iloc[idx]
                        if rc == 0 or idx >= len(df) - h - 1:
                            continue
                        entry = df["close"].iloc[idx]
                        exit_p = df["close"].iloc[idx + h]
                        if entry <= 0:
                            continue
                        ret = exit_p / entry - 1
                        band = "3+" if rc >= 3 else str(rc)
                        if band in bands:
                            bands[band][h].append(ret)
                processed += 1
            except Exception:
                continue

        logger.info(f"共振回测：处理 {processed}/{len(symbols)} 只")
        results = []
        for band in ["1", "2", "3+"]:
            for h in hold_days:
                results.append({"resonance": band, "hold_days": h,
                                **self._stats(bands[band][h])})
        return {"resonance": results, "sample_size": processed}

    def _collect_returns(self, hold_days: list[int], sample_size: int,
                         seed: int) -> dict:
        """采集每个策略的触发点收益（共享数据采集循环）。"""
        symbols = self.engine.get_local_symbols()
        if sample_size and len(symbols) > sample_size:
            rng = random.Random(seed)
            symbols = rng.sample(symbols, sample_size)
        logger.info(f"组合回测：采样 {len(symbols)} 只股票，持有期 {hold_days}")

        strategy_returns: dict = {skey: {h: [] for h in hold_days} for skey in SIGNAL_FUNCS}
        processed = 0
        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < 60:
                    continue
                df = df.reset_index(drop=True)
                signals = _compute_signals(df)
                if not signals:
                    continue
                for skey, sig in signals.items():
                    for h in hold_days:
                        rets = self._forward_returns(df, sig, h)
                        strategy_returns[skey][h].extend(rets)
                processed += 1
            except Exception:
                continue
        logger.info(f"组合回测：处理 {processed}/{len(symbols)} 只")
        strategy_returns["__processed__"] = processed
        return strategy_returns

    @staticmethod
    def _forward_returns(df: pd.DataFrame, signal: pd.Series, hold: int) -> list[float]:
        """计算信号触发后的持有期前向收益。"""
        close = df["close"]
        rets: list[float] = []
        sig = signal.fillna(False)
        max_i = len(df) - hold - 1
        # 遍历触发点，计算持有期收益
        for i in sig.index[sig]:
            if i >= max_i:
                continue
            entry = close.iloc[i]
            exit_p = close.iloc[i + hold]
            if entry > 0:
                rets.append(exit_p / entry - 1)
        return rets

    @staticmethod
    def _stats(returns: list[float]) -> dict:
        if not returns:
            return {"avg_return": 0, "win_rate": 0, "count": 0, "sharpe": 0}
        arr = np.array(returns)
        avg = float(arr.mean()) * 100
        win = float((arr > 0).mean()) * 100
        std = float(arr.std())
        sharpe = float(avg / std) if std > 0 else 0
        return {
            "avg_return": round(avg, 2),
            "win_rate": round(win, 1),
            "count": len(arr),
            "sharpe": round(sharpe, 2),
        }
