"""策略评估引擎：时间序列净值 + 全维度评分卡 + 新旧策略对比。

核心机制（对标聚宽/米筐/优矿）：
  把策略信号 → 月度等权组合净值曲线，计算专业级评价指标。
  不再只是横截面均值统计，而是还原策略的真实时间序列表现。

评价维度（5维）：
  1. 收益力：年化收益、月均净收益
  2. 抗风险：最大回撤（最坏情况）
  3. 风险调整：夏普比率、卡玛比率（年化/最大回撤）
  4. 交易质量：月胜率、盈亏比（赚的月平均赚多少/亏的月平均亏多少）
  5. 超额收益：跑赢全市场等权基准的幅度（真alpha）

成本模型：继承 combo_backtest.CostModel，A股往返0.192%/笔。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sequoia_x.analysis.combo_backtest import (
    CostModel,
    DEFAULT_COST,
    SIGNAL_FUNCS,
    STRATEGY_LABELS,
    _compute_signals,
)
from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine

logger = get_logger(__name__)

# 回测默认持有期（月度，对齐A股T+1）
DEFAULT_HOLD = 20


@dataclass
class StrategyMetrics:
    """单策略全维度评价。"""
    key: str
    label: str
    annual_return: float       # 年化收益%
    avg_net_return: float      # 月均净收益%
    max_drawdown: float        # 最大回撤%（负数）
    sharpe: float              # 月频夏普
    calmar: float              # 卡玛比率
    win_rate: float            # 月胜率%
    profit_loss_ratio: float   # 盈亏比
    alpha: float               # 超额收益%（年化-基准年化）
    sample_trades: int         # 样本数（信号触发总次数）
    monthly_returns: list      # 月度收益序列
    curve: list                # 累计净值曲线

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label,
            "annual_return": self.annual_return,
            "avg_net_return": self.avg_net_return,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe, "calmar": self.calmar,
            "win_rate": self.win_rate,
            "profit_loss_ratio": self.profit_loss_ratio,
            "alpha": self.alpha, "sample_trades": self.sample_trades,
            "monthly_returns": self.monthly_returns,
            "curve": self.curve,
        }


class StrategyEvaluator:
    """策略评估器：生成时间序列净值曲线与全维度评分卡。"""

    def __init__(self, engine: DataEngine, settings: Settings,
                 cost: CostModel | None = None) -> None:
        self.engine = engine
        self.settings = settings
        self.cost = cost or DEFAULT_COST

    def evaluate(
        self, hold_days: int = DEFAULT_HOLD, sample_size: int = 500,
        seed: int = 42,
    ) -> dict:
        """评估全部策略，返回评分卡 + 净值曲线 + 基准。

        Returns:
            {
                "strategies": [StrategyMetrics.to_dict(), ...],
                "benchmark": {...},
                "months": [...],  # x轴月份
                "hold_days": 20,
                "sample_size": N,
                "round_trip_cost_pct": 0.192,
            }
        """
        rtc = self.cost.round_trip_cost()
        collected = self._collect(hold_days, sample_size, seed)
        months = collected["months"]
        strat_monthly = collected["strategy_monthly"]  # {skey: {month: [returns]}}
        bench_monthly = collected["benchmark_monthly"]  # {month: [returns]}
        trade_counts = collected["trade_counts"]

        # 基准净值曲线
        bench_ret_series = self._monthly_mean(bench_monthly, months)
        bench_curve = self._cumulative_curve(bench_ret_series)
        bench_annual = self._annualized(bench_curve)
        bench_dd = self._max_drawdown(bench_curve)

        # 各策略评分卡
        strategies = []
        MIN_TRADES = 30  # 最小样本数，低于此值不可靠
        for skey in SIGNAL_FUNCS:
            monthly = strat_monthly.get(skey, {})
            ret_series = self._monthly_mean(monthly, months)
            n_trades = trade_counts.get(skey, 0)
            if n_trades < MIN_TRADES:
                continue  # 样本不足，跳过（不可靠）
            curve = self._cumulative_curve(ret_series)
            annual = self._annualized(curve)
            dd = self._max_drawdown(curve)
            sharpe = self._sharpe(ret_series)
            calmar = self._calmar(annual, dd)
            win_rate = self._win_rate(ret_series)
            pl_ratio = self._profit_loss_ratio(ret_series)
            alpha = annual - bench_annual

            strategies.append(StrategyMetrics(
                key=skey, label=STRATEGY_LABELS.get(skey, skey),
                annual_return=round(annual, 2),
                avg_net_return=round(float(np.mean(ret_series)) * 100, 2),
                max_drawdown=round(dd, 2),
                sharpe=round(sharpe, 2),
                calmar=round(calmar, 2),
                win_rate=round(win_rate, 1),
                profit_loss_ratio=round(pl_ratio, 2),
                alpha=round(alpha, 2),
                sample_trades=trade_counts.get(skey, 0),
                monthly_returns=[round(r * 100, 2) for r in ret_series],
                curve=[round(v, 4) for v in curve],
            ).to_dict())

        # 按夏普降序
        strategies.sort(key=lambda x: x["sharpe"], reverse=True)

        return {
            "strategies": strategies,
            "benchmark": {
                "annual_return": round(bench_annual, 2),
                "max_drawdown": round(bench_dd, 2),
                "curve": [round(v, 4) for v in bench_curve],
            },
            "months": months,
            "hold_days": hold_days,
            "sample_size": collected["processed"],
            "round_trip_cost_pct": round(rtc * 100, 3),
        }

    def _collect(self, hold_days: int, sample_size: int, seed: int) -> dict:
        """采集所有股票的信号触发收益 + 全市场基准收益，按月聚合。

        Returns:
            strategy_monthly: {skey: {month: [net_returns]}}
            benchmark_monthly: {month: [net_returns]}  (所有股票所有交易日)
            months: 有序列表
        """
        symbols = self.engine.get_local_symbols()
        if sample_size and len(symbols) > sample_size:
            rng = random.Random(seed)
            symbols = rng.sample(symbols, sample_size)
        logger.info(f"策略评估：采样 {len(symbols)} 只，持有期 {hold_days} 天")

        rtc = self.cost.round_trip_cost()
        strat_monthly: dict[str, dict[str, list[float]]] = {
            s: {} for s in SIGNAL_FUNCS
        }
        bench_monthly: dict[str, list[float]] = {}
        trade_counts: dict[str, int] = {s: 0 for s in SIGNAL_FUNCS}
        all_months: set[str] = set()
        processed = 0

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < hold_days + 60:
                    continue
                df = df.reset_index(drop=True)
                close = df["close"]
                dates = df["date"].astype(str) if "date" in df.columns else None
                if dates is None:
                    continue

                # 前向收益（每个交易日买入持N天的收益）
                fwd = (close.shift(-hold_days) / close - 1).values
                # 有效行：前向收益非NaN 且 收盘价>0
                valid_mask = ~(np.isnan(fwd)) & (close.values > 0)
                dts = dates.values
                months_arr = np.array([str(d)[:7] for d in dts])

                # 基准：全市场每个有效交易日的净收益
                for i in range(len(fwd)):
                    if not valid_mask[i]:
                        continue
                    m = months_arr[i]
                    all_months.add(m)
                    bench_monthly.setdefault(m, []).append(fwd[i] - rtc)

                # 策略信号
                signals = _compute_signals(df)
                for skey, sig in signals.items():
                    sig_vals = sig.fillna(False).values
                    for i in range(len(sig_vals)):
                        if sig_vals[i] and valid_mask[i]:
                            m = months_arr[i]
                            strat_monthly[skey].setdefault(m, []).append(fwd[i] - rtc)
                            trade_counts[skey] += 1
                processed += 1
            except Exception:
                continue

        logger.info(f"策略评估：处理 {processed}/{len(symbols)} 只")
        months_sorted = sorted(all_months)
        return {
            "strategy_monthly": strat_monthly,
            "benchmark_monthly": bench_monthly,
            "months": months_sorted,
            "trade_counts": trade_counts,
            "processed": processed,
        }

    # ------------------------------------------------------------------
    # 指标计算（月频）
    # ------------------------------------------------------------------
    @staticmethod
    def _monthly_mean(monthly: dict, months: list[str]) -> list[float]:
        """把 {month: [returns]} 展开为月度平均收益序列（对齐months）。"""
        result = []
        for m in months:
            rets = monthly.get(m, [])
            result.append(float(np.mean(rets)) if rets else 0.0)
        return result

    @staticmethod
    def _cumulative_curve(monthly_returns: list[float]) -> list[float]:
        """月度收益序列 → 累计净值曲线（起点1.0）。"""
        curve = [1.0]
        for r in monthly_returns:
            curve.append(curve[-1] * (1 + r))
        return curve[1:]  # 去掉起点，对齐月份

    @staticmethod
    def _annualized(curve: list[float]) -> float:
        """累计净值 → 年化收益%。"""
        if len(curve) < 2 or curve[-1] <= 0:
            return 0.0
        n_months = len(curve)
        return (curve[-1] ** (12 / n_months) - 1) * 100

    @staticmethod
    def _max_drawdown(curve: list[float]) -> float:
        """累计净值 → 最大回撤%（负数）。"""
        peak = curve[0]
        max_dd = 0.0
        for v in curve:
            if v > peak:
                peak = v
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
        return max_dd * 100

    @staticmethod
    def _sharpe(monthly_returns: list[float]) -> float:
        """月频夏普（年化）。"""
        arr = np.array(monthly_returns)
        if len(arr) < 3:
            return 0.0
        std = float(arr.std())
        if std == 0:
            return 0.0
        return float(arr.mean() / std * np.sqrt(12))

    @staticmethod
    def _calmar(annual_return: float, max_drawdown: float) -> float:
        """卡玛比率 = 年化收益 / |最大回撤|。"""
        if max_drawdown == 0:
            return 0.0
        return annual_return / abs(max_drawdown)

    @staticmethod
    def _win_rate(monthly_returns: list[float]) -> float:
        """月胜率%。"""
        arr = np.array(monthly_returns)
        if len(arr) == 0:
            return 0.0
        return float((arr > 0).sum() / len(arr) * 100)

    @staticmethod
    def _profit_loss_ratio(monthly_returns: list[float]) -> float:
        """盈亏比 = 盈利月平均收益 / 亏损月平均|收益|。"""
        arr = np.array(monthly_returns)
        gains = arr[arr > 0]
        losses = arr[arr < 0]
        if len(losses) == 0 or float(np.mean(losses)) == 0:
            return 0.0
        return float(np.mean(gains) / abs(np.mean(losses)))
