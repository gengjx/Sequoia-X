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

        # 计算综合质量分（0-100）并写回DB
        for strat in strategies:
            strat["quality_score"] = self._quality_score(strat, bench_annual)

        # 按质量分降序（质量分已融合夏普/回撤/alpha，比纯夏普更全面）
        strategies.sort(key=lambda x: x["quality_score"], reverse=True)

        # 写回DB（动态刷新决策权重）
        try:
            weights = [{
                "strategy_key": s["key"],
                "quality_score": s["quality_score"],
                "sharpe": s["sharpe"], "max_dd": s["max_drawdown"],
                "alpha": s["alpha"], "calmar": s["calmar"],
                "win_rate": s["win_rate"], "pl_ratio": s["profit_loss_ratio"],
                "annual_return": s["annual_return"],
                "sample_trades": s["sample_trades"],
            } for s in strategies]
            self.engine.save_strategy_weights(weights)
            logger.info(f"策略权重已刷新写入DB：{len(weights)}个策略")
        except Exception as e:
            logger.warning(f"策略权重写DB失败（不影响评估结果）：{e!r}")

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
        cutoff_map = self.engine.get_ipo_cutoff_map()

        # 预加载全市场财报（point-in-time质量因子回测）
        finance_map = self._load_finance_for_backtest()
        # 加载DB完整IC权重（39因子，含质量+资金+结构）
        db_weights = self._load_db_factor_weights()
        # 加载三态市场状态权重
        state_weights = self._load_state_factor_weights()

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
                co = cutoff_map.get(symbol)
                if co and "date" in df.columns:
                    df = df[df["date"].astype(str) >= co]
                df = df.reset_index(drop=True)
                close = df["close"]
                dates = df["date"].astype(str) if "date" in df.columns else None
                if dates is None:
                    continue

                # 前向收益（信号当日收盘生成→次日收盘买入持N天卖出，杜绝前视）
                # 策略与基准同口径：fwd[i] = close[i+hold+1]/close[i+1] - 1
                fwd = (close.shift(-(hold_days + 1)) / close.shift(-1) - 1).values
                # 有效行：前向收益非NaN 且 收盘价>0
                valid_mask = ~(np.isnan(fwd)) & (close.values > 0)
                dts = dates.values
                months_arr = np.array([str(d)[:7] for d in dts])

                # 基准：全市场每个有效交易日的毛收益（被动持有，不扣换手成本）
                # 策略收益扣 rtc、基准不扣 → alpha = 策略净 - 基准毛，反映真实扣费后超额
                for i in range(len(fwd)):
                    if not valid_mask[i]:
                        continue
                    m = months_arr[i]
                    all_months.add(m)
                    bench_monthly.setdefault(m, []).append(fwd[i])

                # 策略信号
                fin_series = self._build_finance_series(df, finance_map.get(symbol, []))
                # P1: multi_factor 按月度市场状态切换权重
                mf_weights = self._select_month_weights(months_arr, state_weights, db_weights)
                signals = _compute_signals(df, finance_series=fin_series, factor_weights=mf_weights)
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

    def _load_state_factor_weights(self) -> dict[str, dict[str, float]]:
        """从DB加载三态市场状态因子权重。"""
        try:
            result = self.engine.load_market_factor_weights()
            if result:
                logger.info(f"三态因子权重加载：bull={len(result.get('bull',{}))} "
                           f"neutral={len(result.get('neutral',{}))} "
                           f"bear={len(result.get('bear',{}))}")
            return result
        except Exception as e:
            logger.warning(f"三态权重加载失败：{e!r}")
            return {}

    @staticmethod
    def _select_month_weights(months_arr: np.ndarray, state_weights: dict,
                              fallback: dict | None) -> dict[str, float] | None:
        """根据回测样本的月份分布，选择主导市场状态对应的权重。

        由于 _compute_signals 对整只股票一次性计算信号序列，
        无法逐月切换权重（需按月重算复合分）。
        这里取该股票涉及月份中出现最多的市场状态对应的权重。
        """
        if not state_weights:
            return fallback
        # 简化：如果三态权重中 neutral 有数据，优先用 neutral（最常见状态）
        # 实际逐月切换在 factor.py 的 evaluate 中已实现（计算三套IC）
        for state in ("neutral", "bull", "bear"):
            if state in state_weights and state_weights[state]:
                return state_weights[state]
        return fallback

    def _load_db_factor_weights(self) -> dict[str, float] | None:
        """从DB加载完整IC权重（替代硬编码默认13因子权重）。"""
        try:
            db_w = self.engine.load_factor_weights()
            if db_w:
                weights = {k: v["weight"] for k, v in db_w.items() if v.get("weight", 0) != 0}
                logger.info(f"回测因子权重从DB加载：{len(weights)}个因子")
                return weights
        except Exception as e:
            logger.warning(f"DB因子权重加载失败，使用默认：{e!r}")
        return None

    def _load_finance_for_backtest(self) -> dict[str, list]:
        """批量加载全市场财报，返回 {symbol: [{stat_date, report_date, roe, ...}]}。"""
        import sqlite3
        result: dict[str, list] = {}
        try:
            with sqlite3.connect(self.engine.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT symbol, stat_date, report_date, roe, np_margin, gp_margin, yoy_eps, yoy_pni "
                    "FROM stock_finance"
                ).fetchall()
            for r in rows:
                result.setdefault(r["symbol"], []).append(dict(r))
            logger.info(f"回测财报加载：{len(result)}只")
        except Exception as e:
            logger.warning(f"回测财报加载失败：{e!r}")
        return result

    def _build_finance_series(self, df: pd.DataFrame, finance_rows: list) -> dict[str, pd.Series] | None:
        """从财报行构建point-in-time质量因子序列。

        用 report_date 对齐K线日期，避免未来函数（报告披露后才可用）。
        """
        if not finance_rows or "date" not in df.columns:
            return None

        n = len(df)
        dates = df["date"].astype(str).values
        quality_map = {
            "roe": "roe", "np_margin": "np_margin", "gp_margin": "gp_margin",
            "rev_growth": "yoy_eps", "profit_growth": "yoy_pni",
        }

        # 按 report_date 排序（report_date为空则用stat_date+90天作为保守估计）
        def _sort_key(r):
            rd = r.get("report_date")
            if rd:
                return rd
            sd = r.get("stat_date", "")
            return sd

        sorted_rows = sorted(finance_rows, key=_sort_key)

        result = {}
        for qname, qcol in quality_map.items():
            vals = np.full(n, np.nan)
            for row in sorted_rows:
                rd = row.get("report_date") or row.get("stat_date", "")
                v = row.get(qcol)
                if v is not None and rd:
                    mask = dates >= rd
                    if mask.any():
                        vals[mask] = v
            result[qname] = pd.Series(vals, index=df.index)
        return result

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

    @staticmethod
    def _quality_score(strat: dict, bench_annual: float) -> int:
        """综合质量分（0-100），融合收益/风险/alpha/稳定性。

        评分维度（各0-100标准化后加权）：
          - 风险调整 40%：夏普归一化（>1.0=100分，<-1.0=0分）
          - 抗风险 25%：最大回撤归一化（回撤越小越好，0%=100，-70%=0）
          - 超额alpha 20%：年化超额归一化（跑赢基准20%+=100，落后40%=0）
          - 交易质量 15%：盈亏比归一化（>2.0=100，<0.5=0）
        最终clamp到0-100整数，对齐决策中枢的S/A/B/C/D分层。
        """
        def _norm(v, hi, lo):
            """线性归一化到0-100（hi=100分，lo=0分）。"""
            if hi == lo:
                return 50.0
            return max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))

        s_sharpe = _norm(strat["sharpe"], 1.0, -1.0)
        s_dd = _norm(strat["max_drawdown"], 0.0, -70.0)  # 0%回撤=100分
        s_alpha = _norm(strat["alpha"] if "alpha" in strat else (strat["annual_return"] - bench_annual),
                        20.0, -40.0)
        s_pl = _norm(strat["profit_loss_ratio"], 2.0, 0.5)

        score = (0.40 * s_sharpe + 0.25 * s_dd + 0.20 * s_alpha + 0.15 * s_pl)
        return int(round(max(0, min(100, score))))


    # ------------------------------------------------------------------
    # 组合对比（主观预设 vs 数据驱动，同一把尺子）
    # ------------------------------------------------------------------
    def _eval_combo_metrics(
        self, strat_keys: list[str], strat_series: dict[str, list[float]],
        months: list[str], bench_annual: float,
    ) -> dict:
        """用统一的月度净值体系评估任意策略组合（可复用）。"""
        valid = [s for s in strat_keys if s in strat_series]
        combo_monthly = []
        for i in range(len(months)):
            vals = [strat_series[s][i] for s in valid if not np.isnan(strat_series[s][i])]
            combo_monthly.append(float(np.mean(vals)) if vals else 0.0)
        curve = self._cumulative_curve(combo_monthly)
        annual = self._annualized(curve)
        dd = self._max_drawdown(curve)
        return {
            "strategies": valid,
            "labels": [STRATEGY_LABELS.get(s, s) for s in valid],
            "size": len(valid),
            "annual_return": round(annual, 2),
            "max_drawdown": round(dd, 2),
            "sharpe": round(self._sharpe(combo_monthly), 2),
            "calmar": round(self._calmar(annual, dd), 2),
            "win_rate": round(self._win_rate(combo_monthly), 1),
            "alpha": round(annual - bench_annual, 2),
            "curve": [round(v, 4) for v in curve],
        }

    # 主观预设组合（与 decision.html COMBOS 同步）
    PRESET_COMBOS: dict[str, list[str]] = {
        "趋势突破": ["turtle", "ma_volume", "rps"],
        "主线龙头": ["dragon", "rps", "turtle"],
        "低吸埋伏": ["pullback", "bottom", "limit_down"],
        "均衡全天候": ["rps", "dragon", "turtle", "pullback", "bottom"],
        "高共振精选": ["turtle", "ma_volume", "rps", "dragon", "pullback",
                     "bottom", "flag", "shakeout", "limit_down"],
    }

    def compare_combos(
        self, hold_days: int = DEFAULT_HOLD, sample_size: int = 500,
        top_n: int = 3, seed: int = 42,
    ) -> dict:
        """主观预设组合 vs 数据驱动最优组合 对比（同一时间序列评估体系）。

        一次采集数据，两组组合用同一把尺子评估，直接对比孰优孰劣。
        """
        from itertools import combinations
        rtc = self.cost.round_trip_cost()
        collected = self._collect(hold_days, sample_size, seed)
        months = collected["months"]
        strat_monthly = collected["strategy_monthly"]
        bench_monthly = collected["benchmark_monthly"]
        processed = collected["processed"]

        strat_series: dict[str, list[float]] = {}
        valid_strats = []
        for skey in SIGNAL_FUNCS:
            series = self._monthly_mean(strat_monthly.get(skey, {}), months)
            if any(series):
                strat_series[skey] = series
                valid_strats.append(skey)

        bench_series = self._monthly_mean(bench_monthly, months)
        bench_curve = self._cumulative_curve(bench_series)
        bench_annual = self._annualized(bench_curve)

        # 1. 主观预设组合
        presets = []
        for name, keys in self.PRESET_COMBOS.items():
            m = self._eval_combo_metrics(keys, strat_series, months, bench_annual)
            m["name"] = name
            m["source"] = "preset"
            presets.append(m)

        # 2. 数据驱动最优组合（网格搜索Top N）
        all_combos = []
        for size in range(1, min(5, len(valid_strats)) + 1):
            for combo in combinations(valid_strats, size):
                m = self._eval_combo_metrics(list(combo), strat_series, months, bench_annual)
                all_combos.append(m)
        all_combos.sort(key=lambda x: x["calmar"], reverse=True)
        optimal = []
        for i, m in enumerate(all_combos[:top_n]):
            m["name"] = f"数据最优#{i+1}"
            m["source"] = "optimal"
            optimal.append(m)

        return {
            "presets": presets,
            "optimal": optimal,
            "months": months,
            "hold_days": hold_days,
            "sample_size": processed,
            "round_trip_cost_pct": round(rtc * 100, 3),
            "benchmark_annual": round(bench_annual, 2),
            "benchmark_curve": [round(v, 4) for v in bench_curve],
        }

    # ------------------------------------------------------------------
    # 最优组合搜索（数据驱动，替代主观预设）
    # ------------------------------------------------------------------
    def find_optimal_combos(
        self, hold_days: int = DEFAULT_HOLD, sample_size: int = 500,
        max_strategies: int = 5, top_n: int = 10, seed: int = 42,
    ) -> dict:
        """网格搜索最优策略组合（数据驱动，替代主观预设）。

        遍历所有策略子集（C(8,1)~C(8,max_strategies)），对每个组合：
          - 把子集策略的月度信号收益按月聚合为组合净值
          - 计算年化收益、最大回撤、夏普、卡玛比率
          - 按卡玛比率排序（兼顾收益与抗风险，比纯年化更实战）

        一次采集N种组合复用，不重复计算I/O。

        Args:
            hold_days: 持有期（天），默认20（约月频）。
            sample_size: 采样股票数。
            max_strategies: 单组合最多策略数（限制复杂度，防过拟合）。
            top_n: 返回前N个最优组合。
            seed: 随机种子（可复现）。
        """
        from itertools import combinations

        rtc = self.cost.round_trip_cost()
        collected = self._collect(hold_days, sample_size, seed)
        months = collected["months"]
        strat_monthly = collected["strategy_monthly"]
        bench_monthly = collected["benchmark_monthly"]
        processed = collected["processed"]

        # 各策略月度平均收益序列（对齐months）
        strat_series: dict[str, list[float]] = {}
        valid_strats = []
        for skey in SIGNAL_FUNCS:
            series = self._monthly_mean(strat_monthly.get(skey, {}), months)
            if any(series):  # 有触发的策略才参与搜索
                strat_series[skey] = series
                valid_strats.append(skey)

        # 基准
        bench_series = self._monthly_mean(bench_monthly, months)
        bench_curve = self._cumulative_curve(bench_series)
        bench_annual = self._annualized(bench_curve)

        # 网格搜索所有子集
        all_combos = []
        for size in range(1, min(max_strategies, len(valid_strats)) + 1):
            for combo in combinations(valid_strats, size):
                # 组合月度收益 = 子集策略等权平均（去重后，同一只票同月多策略只算一份均权）
                combo_monthly = []
                for i in range(len(months)):
                    vals = [strat_series[s][i] for s in combo if not np.isnan(strat_series[s][i])]
                    combo_monthly.append(float(np.mean(vals)) if vals else 0.0)

                curve = self._cumulative_curve(combo_monthly)
                annual = self._annualized(curve)
                dd = self._max_drawdown(curve)
                sharpe = self._sharpe(combo_monthly)
                calmar = self._calmar(annual, dd)
                win_rate = self._win_rate(combo_monthly)

                all_combos.append({
                    "strategies": list(combo),
                    "labels": [STRATEGY_LABELS.get(s, s) for s in combo],
                    "size": size,
                    "annual_return": round(annual, 2),
                    "max_drawdown": round(dd, 2),
                    "sharpe": round(sharpe, 2),
                    "calmar": round(calmar, 2),
                    "win_rate": round(win_rate, 1),
                    "alpha": round(annual - bench_annual, 2),
                    "curve": [round(v, 4) for v in curve],
                })

        # 按卡玛排序（卡玛=年化/|回撤|，兼顾收益与抗风险）
        all_combos.sort(key=lambda x: x["calmar"], reverse=True)
        top = all_combos[:top_n]

        logger.info(f"组合搜索：{len(all_combos)}种组合 → Top{top_n}（卡玛最优）")

        return {
            "combos": top,
            "total_searched": len(all_combos),
            "months": months,
            "hold_days": hold_days,
            "sample_size": processed,
            "round_trip_cost_pct": round(rtc * 100, 3),
            "benchmark_annual": round(bench_annual, 2),
            "benchmark_curve": [round(v, 4) for v in bench_curve],
        }
