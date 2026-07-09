"""组合历史回测引擎：向量化解算策略信号 + 持有期收益对比（含A股交易成本）。

核心思路：对每只股票预计算策略信号序列（布尔），统计信号触发后的持有期收益，
横向对比各组合的实战效果（净收益/毛收益/成本侵蚀/胜率/夏普/样本数）。

成本模型（A股标准，往返双边口径）：
  往返成本 = 佣金(万2.5单边×2) + 印花税(万5卖出单边) + 过户费(万0.1双边) + 滑点(2bp双边)
  约 0.192%。短线持有期(5日)成本侵蚀显著，裸价收益会严重高估策略表现。

因子评价（Rank IC）：
  - 共振度因子IC：把"多策略共振数"作为连续因子，算与持有期收益的Spearman秩相关，
    验证决策中枢"共振→超额收益"的核心假设（IC>0.02且单调递增方为有效）。
  - 策略独立IC：每个策略信号(布尔)与持有期收益的秩相关，衡量该策略相对其他策略
    的选股预测力（IC>0表示选出的股票优于策略池平均）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine

logger = get_logger(__name__)

# 策略信号 → 计算函数名 映射
SIGNAL_FUNCS = {
    "ma_volume", "turtle", "pullback", "bottom", "rps",
    "flag", "shakeout", "limit_down", "multi_factor",
}

# 策略显示名
STRATEGY_LABELS = {
    "ma_volume": "均线放量", "turtle": "海龟突破", "pullback": "缩量回踩",
    "bottom": "底部放量", "rps": "RPS强势",
    "flag": "高位旗形", "shakeout": "涨停洗盘", "limit_down": "上升趋势跌停",
    "multi_factor": "多因子选股",
}


@dataclass(frozen=True)
class CostModel:
    """A股交易成本模型。

    费率按单边定义，往返（买+卖）自动×2（佣金/过户费/滑点）；
    印花税仅卖出单边（2023.8.28减半至万5）。
    默认券商主流费率，可经 UI 调整。

    Attributes:
        commission_rate: 佣金率，默认万2.5（双边）。
        stamp_duty_rate: 印花税率，默认万5（仅卖出，2023.8.28起）。
        transfer_fee_rate: 过户费率，默认万0.1（双边，沪市）。
        slippage_rate: 滑点率，默认2bp（双边，保守估计）。
    """

    commission_rate: float = 0.00025
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_rate: float = 0.0002

    def round_trip_cost(self) -> float:
        """单次往返（买+卖）总成本率。"""
        return (
            self.commission_rate * 2
            + self.stamp_duty_rate
            + self.transfer_fee_rate * 2
            + self.slippage_rate * 2
        )


DEFAULT_COST = CostModel()


def _ic_assessment(ic_mean: float) -> str:
    """单值 IC 有效性评级（panel 整体秩相关口径）。"""
    if ic_mean >= 0.05:
        return "强正向"
    if ic_mean >= 0.02:
        return "有效正向"
    if ic_mean <= -0.05:
        return "强负向"
    if ic_mean <= -0.02:
        return "有效负向"
    return "区分力不足"

def _compute_signals(df: pd.DataFrame, finance_series: dict[str, pd.Series] | None = None, factor_weights: dict[str, float] | None = None) -> dict[str, pd.Series]:
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

    # 高位旗形：40日强动量(涨幅>60%) + 10日收敛(振幅<15%) + 高位抗跌 + 缩量
    if len(df) >= 40:
        high40 = high.rolling(40).max()
        low40 = low.rolling(40).min()
        high10 = high.rolling(10).max()
        low10 = low.rolling(10).min()
        momentum = high40 / low40.replace(0, np.nan) > 1.6
        consolidation = high10 / low10.replace(0, np.nan) < 1.15
        high_level = low10 >= high40 * 0.8
        shrink_flag = volume < vol_ma20 * 0.6
        signals["flag"] = momentum & consolidation & high_level & shrink_flag
    else:
        signals["flag"] = pd.Series(False, index=df.index)

    # 涨停洗盘：昨日涨停(>=前日*1.095) + 今日收阴 + 放量2倍 + 支撑不破
    prev_close = close.shift(1)
    prev2_close = close.shift(2)
    limit_up_y = prev_close >= prev2_close * 1.095
    bearish = close < df["open"]
    vol_surge_shake = volume > volume.shift(1) * 2.0
    support = low >= prev_close
    signals["shakeout"] = limit_up_y & bearish & vol_surge_shake & support

    # 上升趋势跌停：昨日MA20>MA60 + 今日跌停(<=昨日*0.905) + 放量2倍
    if len(df) >= 60:
        ma60 = close.rolling(60).mean()
        uptrend_ld = ma20.shift(1) > ma60.shift(1)
        limit_down = close <= prev_close * 0.905
        vol_surge_ld = volume > vol_ma20 * 2.0
        signals["limit_down"] = uptrend_ld & limit_down & vol_surge_ld
    else:
        signals["limit_down"] = pd.Series(False, index=df.index)

    # 多因子选股：综合因子分Top20%分位 = 触发信号
    try:
        from sequoia_x.analysis.factor import compute_composite_score
        composite = compute_composite_score(df, weights=factor_weights, finance_series=finance_series)
        # 滚动分位：综合分进入自身历史前20%时触发
        threshold = composite.rolling(120, min_periods=60).quantile(0.8)
        signals["multi_factor"] = composite >= threshold
    except Exception:
        signals["multi_factor"] = pd.Series(False, index=df.index)

    return signals


class ComboBacktester:
    """组合历史回测器（含A股交易成本）。

    Attributes:
        engine: DataEngine 实例。
        settings: Settings 实例。
        cost: CostModel，交易成本模型，默认 A股主流费率。
    """

    def __init__(self, engine: DataEngine, settings: Settings, cost: CostModel | None = None) -> None:
        self.engine = engine
        self.settings = settings
        self.cost = cost or DEFAULT_COST

    def run(
        self, combos: dict[str, list[str]], hold_days: list[int] | None = None,
        sample_size: int = 500, seed: int = 42,
    ) -> dict:
        """回测多个组合，返回横向对比报告（含净/毛收益与策略IC）。"""
        hold_days = hold_days or [5, 10, 20]
        collected = self._collect_returns(hold_days, sample_size, seed)
        strategy_returns = collected["returns"]
        processed = collected["processed"]

        results = []
        for combo_name, skeys in combos.items():
            valid_keys = [k for k in skeys if k in SIGNAL_FUNCS]
            for h in hold_days:
                nets: list[float] = []
                gross: list[float] = []
                for k in valid_keys:
                    pair = strategy_returns.get(k, {}).get(h, ([], []))
                    nets.extend(pair[0])
                    gross.extend(pair[1])
                results.append({
                    "combo": combo_name, "hold_days": h,
                    **self._stats(nets, gross, h),
                })

        # 策略独立 IC（panel 整体秩相关：该策略触发 vs 其他策略触发的收益差异）
        strategy_ic = self._strategy_panel_ic(strategy_returns, hold_days)

        return {
            "combos": results, "sample_size": processed,
            "strategy_ic": strategy_ic,
            "round_trip_cost_pct": round(self.cost.round_trip_cost() * 100, 3),
        }

    def run_resonance(self, hold_days: list[int] | None = None,
                      sample_size: int = 500, seed: int = 42) -> dict:
        """共振度分档回测：统计不同共振度（1/2/3+）下的持有期收益。

        核心验证：多策略共振是否真能带来超额收益（决策中枢定级矩阵的基础假设）。
        同时计算共振度因子 Rank IC（验证"共振度→收益"的单调有效性）。
        """
        hold_days = hold_days or [5, 10, 20]
        symbols = self.engine.get_local_symbols()
        if sample_size and len(symbols) > sample_size:
            rng = random.Random(seed)
            symbols = rng.sample(symbols, sample_size)
        logger.info(f"共振回测：采样 {len(symbols)} 只股票")
        # 加载因子权重 + 财报（与 _collect_returns 同口径）
        factor_weights = self._load_factor_weights()
        finance_map = self._load_finance_map()
        cutoff_map = self.engine.get_ipo_cutoff_map()

        bands = {"1": {h: ([], []) for h in hold_days},
                 "2": {h: ([], []) for h in hold_days},
                 "3+": {h: ([], []) for h in hold_days}}
        # 共振因子 IC 采样：(共振度, 净收益)，按持有期分桶
        factor_samples: dict[int, list[tuple[int, float]]] = {h: [] for h in hold_days}
        processed = 0
        rtc = self.cost.round_trip_cost()

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < 60:
                    continue
                co = cutoff_map.get(symbol)
                if co and "date" in df.columns:
                    df = df[df["date"].astype(str) >= co]
                df = df.reset_index(drop=True)
                fin_series = self._build_finance_series(df, finance_map.get(symbol, []))
                signals = _compute_signals(df, finance_series=fin_series, factor_weights=factor_weights)
                if not signals:
                    continue
                sig_df = pd.DataFrame({k: v.fillna(False) for k, v in signals.items()})
                resonance_count = sig_df.sum(axis=1)
                close = df["close"]
                for h in hold_days:
                    for idx in resonance_count.index:
                        rc = int(resonance_count.iloc[idx])
                        if rc == 0 or idx + 1 + h >= len(df):
                            continue
                        entry = close.iloc[idx + 1]
                        exit_p = close.iloc[idx + 1 + h]
                        if entry <= 0:
                            continue
                        gross = exit_p / entry - 1
                        net = gross - rtc
                        band = "3+" if rc >= 3 else str(rc)
                        if band in bands:
                            bands[band][h][0].append(net)
                            bands[band][h][1].append(gross)
                        factor_samples[h].append((rc, net))
                processed += 1
            except Exception:
                continue

        logger.info(f"共振回测：处理 {processed}/{len(symbols)} 只")
        results = []
        for band in ["1", "2", "3+"]:
            for h in hold_days:
                nets, gross = bands[band][h]
                results.append({"resonance": band, "hold_days": h,
                                **self._stats(nets, gross, h)})

        # 共振度因子 IC（每个持有期一组）
        resonance_ic = []
        for h in hold_days:
            samples = factor_samples[h]
            ic = self._panel_ic([s[0] for s in samples], [s[1] for s in samples])
            ic["hold_days"] = h
            resonance_ic.append(ic)

        # 单调性检验：10日持有期下各档净收益是否随共振度递增
        main = {r["resonance"]: r for r in results if r["hold_days"] == 10}
        order = ["1", "2", "3+"]
        rets_10 = [main[b]["avg_return"] for b in order if b in main]
        monotonic = (
            len(rets_10) >= 2 and
            all(rets_10[i] <= rets_10[i + 1] for i in range(len(rets_10) - 1))
        )

        return {
            "resonance": results, "sample_size": processed,
            "resonance_ic": resonance_ic,
            "monotonic_10d": monotonic,
            "round_trip_cost_pct": round(rtc * 100, 3),
        }

    def _collect_returns(self, hold_days: list[int], sample_size: int,
                         seed: int) -> dict:
        """采集每个策略的触发点收益（共享数据采集循环），含净/毛双口径。"""
        # 加载 DB 因子权重 + 财报数据（修复：combo_backtest 之前从未传入因子权重）
        factor_weights = self._load_factor_weights()
        finance_map = self._load_finance_map()
        symbols = self.engine.get_local_symbols()
        if sample_size and len(symbols) > sample_size:
            rng = random.Random(seed)
            symbols = rng.sample(symbols, sample_size)
        logger.info(f"组合回测：采样 {len(symbols)} 只股票，持有期 {hold_days}")
        cutoff_map = self.engine.get_ipo_cutoff_map()

        strategy_returns: dict = {
            skey: {h: ([], []) for h in hold_days} for skey in SIGNAL_FUNCS
        }
        processed = 0
        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < 60:
                    continue
                co = cutoff_map.get(symbol)
                if co and "date" in df.columns:
                    df = df[df["date"].astype(str) >= co]
                df = df.reset_index(drop=True)
                fin_series = self._build_finance_series(df, finance_map.get(symbol, []))
                signals = _compute_signals(df, finance_series=fin_series, factor_weights=factor_weights)
                if not signals:
                    continue
                for skey, sig in signals.items():
                    for h in hold_days:
                        nets, gross = self._forward_returns(df, sig, h)
                        strategy_returns[skey][h][0].extend(nets)
                        strategy_returns[skey][h][1].extend(gross)
                processed += 1
            except Exception:
                continue
        logger.info(f"组合回测：处理 {processed}/{len(symbols)} 只")
        return {"returns": strategy_returns, "processed": processed}

    def _load_factor_weights(self) -> dict[str, float] | None:
        """从 DB 加载因子 IC 权重（带符号加权）。"""
        try:
            db_w = self.engine.load_factor_weights()
            if db_w:
                weights = {k: v["weight"] for k, v in db_w.items() if v.get("weight", 0) != 0}
                logger.info(f"组合回测因子权重：{len(weights)} 个因子")
                return weights
        except Exception as e:
            logger.warning(f"因子权重加载失败：{e!r}")
        return None

    def _load_finance_map(self) -> dict[str, list]:
        """批量加载全市场财报。"""
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
            logger.info(f"组合回测财报加载：{len(result)} 只")
        except Exception as e:
            logger.warning(f"财报加载失败：{e!r}")
        return result

    @staticmethod
    def _build_finance_series(df: pd.DataFrame, finance_rows: list) -> dict[str, pd.Series] | None:
        """从财报行构建 point-in-time 质量因子序列。"""
        if not finance_rows or "date" not in df.columns:
            return None
        n = len(df)
        dates = df["date"].astype(str).values
        # 报告日期 → 对齐到K线日期（报告披露后才可用，防未来函数）
        roe_s = pd.Series(np.nan, index=df.index)
        np_margin_s = pd.Series(np.nan, index=df.index)
        gp_margin_s = pd.Series(np.nan, index=df.index)
        rev_growth_s = pd.Series(np.nan, index=df.index)
        profit_growth_s = pd.Series(np.nan, index=df.index)
        for row in sorted(finance_rows, key=lambda x: x.get("stat_date") or ""):
            rd = (row.get("report_date") or row.get("stat_date") or "")[:10]
            if not rd:
                continue
            # 找到 >= 报告日期的第一个K线日
            mask = dates >= rd
            if mask.any():
                idx = np.argmax(mask)
                if row.get("roe") is not None:
                    roe_s.iloc[idx:] = float(row["roe"])
                if row.get("np_margin") is not None:
                    np_margin_s.iloc[idx:] = float(row["np_margin"])
                if row.get("gp_margin") is not None:
                    gp_margin_s.iloc[idx:] = float(row["gp_margin"])
                if row.get("yoy_eps") is not None:
                    rev_growth_s.iloc[idx:] = float(row["yoy_eps"])
                if row.get("yoy_pni") is not None:
                    profit_growth_s.iloc[idx:] = float(row["yoy_pni"])
        result = {}
        for name, s in [("roe", roe_s), ("np_margin", np_margin_s), ("gp_margin", gp_margin_s),
                         ("rev_growth", rev_growth_s), ("profit_growth", profit_growth_s)]:
            if s.notna().any():
                result[name] = s
        return result if result else None

    @staticmethod
    def _forward_returns(df: pd.DataFrame, signal: pd.Series, hold: int) -> tuple[list[float], list[float]]:
        """计算信号触发后的持有期收益（净/毛双口径）。

        Returns:
            (净收益列表, 毛收益列表)。净收益 = 毛收益 - 往返成本率。
        """
        close = df["close"]
        net_rets: list[float] = []
        gross_rets: list[float] = []
        sig = signal.fillna(False)
        rtc = DEFAULT_COST.round_trip_cost()
        # 信号第i日收盘生成 → 最早i+1日成交（次日收盘买入→i+1+hold收盘卖出），杜绝前视
        for i in sig.index[sig]:
            if i + 1 + hold >= len(df):
                continue
            entry = close.iloc[i + 1]
            exit_p = close.iloc[i + 1 + hold]
            if entry > 0:
                gross = exit_p / entry - 1
                gross_rets.append(gross)
                net_rets.append(gross - rtc)
        return net_rets, gross_rets

    @staticmethod
    def _stats(net_returns: list[float], gross_returns: list[float], hold: int = 10) -> dict:
        """统计净/毛收益、成本侵蚀、胜率、夏普（年化）。

        hold: 持有期天数，夏普年化系数 = √(252/hold)。
        夏普用收益小数/标准差小数（同口径），不再×100导致单位错配。
        """
        if not net_returns:
            return {"avg_return": 0, "gross_return": 0, "cost_drag": 0,
                    "win_rate": 0, "count": 0, "sharpe": 0}
        n = np.array(net_returns)
        g = np.array(gross_returns) if gross_returns else n
        avg = float(n.mean()) * 100          # 净收益
        gavg = float(g.mean()) * 100         # 毛收益
        win = float((n > 0).mean()) * 100
        std = float(n.std())
        sharpe = float(n.mean() / std * np.sqrt(252 / hold)) if std > 0 else 0
        return {
            "avg_return": round(avg, 2),       # 净收益（扣成本后）
            "gross_return": round(gavg, 2),    # 毛收益（裸价）
            "cost_drag": round(gavg - avg, 2), # 成本侵蚀
            "win_rate": round(win, 1),
            "count": len(n),
            "sharpe": round(sharpe, 2),
        }

    @staticmethod
    def _panel_ic(factor: list, target: list) -> dict:
        """整体 Spearman 秩相关 IC（panel 口径，所有样本合并）。"""
        f = pd.Series(factor)
        t = pd.Series(target)
        valid = pd.concat([f, t], axis=1).dropna()
        if len(valid) < 30:
            return {"ic_mean": 0.0, "n": int(len(valid)), "assessment": "样本不足"}
        ic = float(valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank()))
        return {
            "ic_mean": round(ic, 4),
            "n": int(len(valid)),
            "assessment": _ic_assessment(ic),
        }

    def _strategy_panel_ic(self, strategy_returns: dict, hold_days: list[int]) -> list[dict]:
        """每个策略的独立 IC：该策略触发样本 vs 全策略池样本的收益秩相关。

        构造方式：对每个持有期，把所有策略触发点合并，标记"是否本策略触发"(0/1)，
        与净收益做 Spearman 秩相关。IC>0 表示该策略选出的股票优于策略池平均。
        """
        out = []
        for h in hold_days:
            for skey in SIGNAL_FUNCS:
                nets = strategy_returns.get(skey, {}).get(h, ([], []))[0]
                if len(nets) < 30:
                    continue
                # 构造：本策略触发=1，其余策略触发=0，与净收益做秩相关
                others = [r for sk in SIGNAL_FUNCS if sk != skey
                          for r in strategy_returns.get(sk, {}).get(h, ([], []))[0]]
                factor = [1] * len(nets) + [0] * len(others)
                target = nets + others
                ic = self._panel_ic(factor, target)
                out.append({
                    "strategy": skey,
                    "label": STRATEGY_LABELS.get(skey, skey),
                    "hold_days": h,
                    **ic,
                })
        return out
