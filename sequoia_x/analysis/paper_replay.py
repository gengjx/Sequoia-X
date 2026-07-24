"""模拟盘历史回放引擎：用完整日K数据逐日重放决策链路。

核心思路：
  对回测区间内每个交易日，用"截至该日的数据"（杜绝前视）执行：
    1. 多因子综合分选股 → Top候选池
    2. 市场状态自适应仓位
    3. 买入（等权分配，受仓位上限约束）
    4. 止损/止盈检查 → 自动卖出
    5. 记录净值

与strategy_eval信号级回测的区别：
  - strategy_eval: 每个信号独立计算前向收益，不模拟持仓管理
  - paper_replay: 完整模拟资金管理、仓位限制、止损执行、复利效应

输出：净值曲线 + 交易明细 + 绩效指标（年化/回撤/夏普/胜率）
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sequoia_x.analysis.factor import compute_factors, cross_section_rank
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# ── 回放参数（对齐 paper_trade.py 实盘模拟盘参数）──
HOLD_DAYS = 20             # 默认持有期（参数扫描最优：20天调仓）
MAX_POSITION_PCT = 0.15     # 单票最大仓位15%
MAX_TOTAL_PCT = 0.80        # 总仓位上限80%
TOP_N = 20                  # 每日选出的候选股票数
STOP_LOSS_PCT = -12.0       # 止损线-12%（扫描最优：放宽避免假止损）
TAKE_PROFIT_PCT = 30.0      # 止盈线+30%（扫描最优：让赢家跑更远）
TRAILING_STOP_PCT = -10.0   # 移动止损（从最高点回落10%，放宽减少假止损）
RTC = 0.00192               # 往返交易成本0.192%（印花税+佣金+滑点）
SAMPLE_SYMBOLS = 500        # 采样股票数（全市场5183只太慢）
MAX_HOLDINGS = 8            # 最大同时持仓数
REBALANCE_INTERVAL = 20     # 每20个交易日重新选股调仓（扫描最优：降低交易频率）


@dataclass
class ReplayPosition:
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    highest_price: float
    stop_loss: float


@dataclass
class ReplayTrade:
    symbol: str
    side: str          # buy / sell
    date: str
    price: float
    shares: int
    amount: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_days: int = 0
    reason: str = ""


class PaperReplayEngine:
    """模拟盘历史回放引擎。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def replay(
        self,
        start_date: str = "",
        end_date: str = "",
        initial_capital: float = 100000.0,
        sample_size: int = SAMPLE_SYMBOLS,
        seed: int = 42,
        progress_callback=None,
    ) -> dict:
        """执行历史回放。

        Args:
            start_date: 回放起始日（空=自动取最近250交易日）
            end_date: 回放结束日（空=最新数据日）
            initial_capital: 初始资金
            sample_size: 采样股票数
            seed: 随机种子
            progress_callback: 进度回调 fn(current, total, msg)

        Returns:
            {
                "nav_curve": [{date, nav, cash, market_value, daily_return, holdings_count}],
                "trades": [ReplayTrade],
                "metrics": {annual_return, max_drawdown, sharpe, win_rate, ...},
                "benchmark": [{date, nav}],
                "market_states": [{date, state}],
            }
        """
        # ── Step 1: 加载数据 ──
        logger.info(f"回放引擎：加载数据（db={self.db_path}）")
        all_daily = self._load_all_daily()
        finance_map = self._load_finance_map()
        state_weights = self._load_state_weights()
        st_set = self._load_st_set()

        all_dates = sorted(all_daily["date"].unique())
        if not start_date:
            # 默认取最近250个交易日
            start_date = all_dates[-250] if len(all_dates) > 250 else all_dates[0]
        if not end_date:
            end_date = all_dates[-1]

        replay_dates = [d for d in all_dates if start_date <= d <= end_date]
        logger.info(f"回放区间：{replay_dates[0]} ~ {replay_dates[-1]}（{len(replay_dates)}个交易日）")

        # 采样股票
        all_symbols = sorted(all_daily["symbol"].unique())
        all_symbols = [s for s in all_symbols if s not in st_set]
        if sample_size and len(all_symbols) > sample_size:
            import random
            rng = random.Random(seed)
            sampled = rng.sample(all_symbols, sample_size)
        else:
            sampled = all_symbols

        # 预分组：{symbol: DataFrame}，截断到 end_date（杜绝前视）
        daily_filtered = all_daily[all_daily["date"] <= end_date]
        symbol_groups = {sym: g for sym, g in daily_filtered.groupby("symbol", sort=False)
                         if sym in set(sampled)}

        logger.info(f"回放采样：{len(symbol_groups)}只股票")

        # ── Step 2: 预计算每个交易日的截面因子 ──
        # 为了性能，只在天数足够时计算因子
        cutoff_map = self._load_ipo_cutoff()

        # ── Step 3: 逐日回放 ──
        cash = initial_capital
        positions: list[ReplayPosition] = []
        trades: list[ReplayTrade] = []
        nav_curve = []
        benchmark_curve = []
        market_states = []
        last_rebalance_idx = -REBALANCE_INTERVAL

        portfolio_high = initial_capital
        max_dd = 0.0

        total_days = len(replay_dates)
        for day_idx, today in enumerate(replay_dates):
            if progress_callback and day_idx % 10 == 0:
                progress_callback(day_idx, total_days, f"回放进度 {day_idx}/{total_days}")

            # 获取当日全市场收盘价
            today_data = all_daily[all_daily["date"] == today]
            today_prices = dict(zip(today_data["symbol"], today_data["close"]))

            # ── 持仓止损/止盈检查 ──
            new_positions = []
            for pos in positions:
                price = today_prices.get(pos.symbol)
                if price is None or price <= 0:
                    new_positions.append(pos)
                    continue

                pos.highest_price = max(pos.highest_price, price)
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                trailing_stop = pos.highest_price * (1 + TRAILING_STOP_PCT / 100)
                effective_stop = max(pos.stop_loss, trailing_stop)

                # 止损/止盈/移动止损触发
                days_held = day_idx - replay_dates.index(pos.entry_date) if pos.entry_date in replay_dates else HOLD_DAYS
                should_sell = False
                reason = ""

                if price <= effective_stop and pnl_pct < 0:
                    should_sell = True
                    reason = f"止损{pnl_pct:.1f}%"
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    should_sell = True
                    reason = f"止盈{pnl_pct:.1f}%"
                elif days_held >= 60:  # 超时卖出（60天）
                    should_sell = True
                    reason = f"超时{days_held}天 pnl={pnl_pct:.1f}%"

                if should_sell:
                    amount = price * pos.shares
                    cost = amount * RTC
                    pnl = (price - pos.entry_price) * pos.shares - cost
                    cash += amount - cost
                    trades.append(ReplayTrade(
                        symbol=pos.symbol, side="sell", date=today,
                        price=price, shares=pos.shares, amount=amount,
                        pnl=pnl, pnl_pct=pnl_pct, hold_days=days_held, reason=reason,
                    ))
                else:
                    new_positions.append(pos)

            positions = new_positions

            # ── 调仓：每 REBALANCE_INTERVAL 天重新选股 ──
            if day_idx - last_rebalance_idx >= REBALANCE_INTERVAL:
                last_rebalance_idx = day_idx

                # 市场状态检测
                market_state = self._detect_market_state_at(symbol_groups, today)
                market_states.append({"date": today, "state": market_state})

                adaptive = {"bull": (1.0, 5), "neutral": (0.9, 7), "bear": (0.5, 10)}
                position_scale, min_score = adaptive.get(market_state, (0.9, 7))

                # 极端弱市空仓
                if market_state == "bear":
                    median_20d = self._get_market_median_return(symbol_groups, today, 20)
                    if median_20d < -0.05:
                        # 清仓
                        for pos in positions:
                            price = today_prices.get(pos.symbol, pos.entry_price)
                            if price > 0:
                                amount = price * pos.shares
                                pnl = (price - pos.entry_price) * pos.shares
                                cash += amount * (1 - RTC)
                                trades.append(ReplayTrade(
                                    symbol=pos.symbol, side="sell", date=today,
                                    price=price, shares=pos.shares, amount=amount,
                                    pnl=pnl, pnl_pct=(price/pos.entry_price-1)*100,
                                    hold_days=day_idx - replay_dates.index(pos.entry_date) if pos.entry_date in replay_dates else 0,
                                    reason="极端弱市清仓",
                                ))
                        positions = []
                else:
                    # 计算因子截面 → 选股
                    candidates = self._select_top(
                        symbol_groups, today, state_weights, finance_map, cutoff_map, st_set
                    )

                    # 持仓不重复买
                    held_symbols = {p.symbol for p in positions}
                    candidates = [c for c in candidates if c not in held_symbols]

                    # 仓位计算
                    holdings_value = sum(
                        today_prices.get(p.symbol, p.entry_price) * p.shares
                        for p in positions
                    )
                    max_total = initial_capital * MAX_TOTAL_PCT * position_scale
                    available = min(cash, max_total - holdings_value)

                    # 买入
                    slots = MAX_HOLDINGS - len(positions)
                    for sym in candidates[:slots]:
                        price = today_prices.get(sym)
                        if price is None or price <= 0:
                            continue
                        # 等权分配
                        target_amount = min(
                            initial_capital * MAX_POSITION_PCT,
                            available / max(slots, 1),
                        )
                        if target_amount < price * 100:  # 买不起1手
                            continue
                        shares = int(target_amount / price / 100) * 100  # 整手
                        if shares <= 0:
                            continue
                        cost = price * shares * RTC
                        total_cost = price * shares + cost
                        if total_cost > cash:
                            continue

                        cash -= total_cost
                        # ATR动态止损：根据股票自身波动率计算（替代固定-12%）
                        stop = self._calc_atr_stop(symbol_groups, sym, today, price)
                        positions.append(ReplayPosition(
                            symbol=sym, entry_date=today, entry_price=price,
                            shares=shares, highest_price=price, stop_loss=stop,
                        ))
                        trades.append(ReplayTrade(
                            symbol=sym, side="buy", date=today,
                            price=price, shares=shares, amount=price*shares,
                            reason=f"多因子选股（{market_state}）",
                        ))

            # ── 记录净值 ──
            market_value = sum(
                today_prices.get(p.symbol, p.entry_price) * p.shares
                for p in positions
            )
            total_assets = cash + market_value
            prev_nav = nav_curve[-1]["nav"] if nav_curve else 1.0
            daily_ret = (total_assets / (prev_nav * initial_capital) - 1) * 100 if prev_nav > 0 else 0
            nav_curve.append({
                "date": today,
                "nav": round(total_assets / initial_capital, 4),
                "cash": round(cash, 2),
                "market_value": round(market_value, 2),
                "daily_return": round(daily_ret, 3),
                "holdings_count": len(positions),
            })

            portfolio_high = max(portfolio_high, total_assets)
            dd = (total_assets - portfolio_high) / portfolio_high * 100
            if dd < max_dd:
                max_dd = dd

            # 基准（等权全市场）
            bench_rets = []
            for sym, g in symbol_groups.items():
                gtd = g[g["date"] == today]
                if len(gtd) > 0:
                    bench_rets.append(gtd["pct_chg"].values[0] if gtd["pct_chg"].values[0] == gtd["pct_chg"].values[0] else 0)
            bench_daily = float(np.mean(bench_rets)) if bench_rets else 0
            prev_bench = benchmark_curve[-1]["nav"] if benchmark_curve else 1.0
            benchmark_curve.append({
                "date": today,
                "nav": round(prev_bench * (1 + bench_daily / 100), 4),
            })

        # ── Step 4: 计算绩效 ──
        metrics = self._calc_metrics(nav_curve, benchmark_curve, trades, initial_capital)

        return {
            "nav_curve": nav_curve,
            "benchmark": benchmark_curve,
            "trades": [self._trade_to_dict(t) for t in trades],
            "metrics": metrics,
            "market_states": market_states,
            "config": {
                "start_date": replay_dates[0],
                "end_date": replay_dates[-1],
                "trading_days": len(replay_dates),
                "initial_capital": initial_capital,
                "sample_size": len(symbol_groups),
                "top_n": TOP_N,
                "hold_days": HOLD_DAYS,
                "stop_loss": STOP_LOSS_PCT,
                "take_profit": TAKE_PROFIT_PCT,
                "trailing_stop": TRAILING_STOP_PCT,
                "max_position_pct": MAX_POSITION_PCT,
                "max_holdings": MAX_HOLDINGS,
                "rebalance_interval": REBALANCE_INTERVAL,
            },
        }

    def _select_top(
        self, symbol_groups: dict, today: str,
        state_weights: dict, finance_map: dict, cutoff_map: dict, st_set: set,
    ) -> list[str]:
        """多因子选股：计算截面因子 → 趋势确认过滤 → 三态IC加权 → Top N。

        核心改进（vs 纯IC加权选股）：
          纯反转因子选出"跌多了的股票"→ 77%继续跌触发止损。
          加入趋势确认过滤：只在趋势企稳的股票中应用因子选股。
        """
        market_state = self._detect_market_state_at(symbol_groups, today)
        active_weights = state_weights.get(market_state) or state_weights.get("neutral") or state_weights.get("bear") or state_weights.get("bull") or {}

        if not active_weights:
            return []

        rows = []
        # ── 趋势确认过滤数据 ──
        trend_info: dict[str, dict] = {}
        for sym, g in symbol_groups.items():
            if sym in st_set:
                continue
            g_cut = g[g["date"] <= today]
            if len(g_cut) < 60:
                continue
            co = cutoff_map.get(sym)
            if co:
                g_cut = g_cut[g_cut["date"] >= co]
            if len(g_cut) < 60:
                continue

            close = g_cut["close"]
            if len(close) < 60:
                continue

            try:
                factors = compute_factors(g_cut, finance=finance_map.get(sym))
                if not factors:
                    continue
                factors["symbol"] = sym
                rows.append(factors)

                # 趋势确认指标
                ma20 = close.iloc[-20:].mean()
                ma60 = close.iloc[-60:].mean()
                ret_20d = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0
                ret_5d = close.iloc[-1] / close.iloc[-6] - 1 if len(close) >= 6 else 0

                trend_info[sym] = {
                    "above_ma20": close.iloc[-1] > ma20,
                    "above_ma60": close.iloc[-1] > ma60,
                    "ma20_above_ma60": ma20 > ma60,  # 中期上升趋势
                    "ret_20d": ret_20d,
                    "ret_5d": ret_5d,
                    # 相对强度：5日反弹力度（反转因子选出的票需要确认企稳）
                    "stabilized": ret_5d > -0.03,  # 5日跌幅<3%视为企稳
                }
            except Exception:
                continue

        if len(rows) < 30:
            return []

        df_factors = pd.DataFrame(rows).set_index("symbol")

        # ── 趋势确认过滤：3层渐进式 ──
        # Layer 1: 价格在MA60之上（中期不破位）
        confirmed = {sym for sym, t in trend_info.items()
                     if t["above_ma60"] and t["stabilized"]}
        if len(confirmed) < 20:
            # 候选不足，放宽到MA20之上
            confirmed = {sym for sym, t in trend_info.items()
                         if t["above_ma20"] and t["stabilized"]}
        if len(confirmed) < 20:
            # 仍然不足，去掉stabilized条件
            confirmed = {sym for sym, t in trend_info.items() if t["above_ma20"]}

        df_factors = df_factors[df_factors.index.isin(confirmed)]
        if len(df_factors) < 15:
            return []

        valid_factors = [f for f in active_weights if f in df_factors.columns]
        if not valid_factors:
            return []

        df_rank = cross_section_rank(df_factors[valid_factors])
        total_w = sum(abs(active_weights[f]) for f in valid_factors)
        if total_w == 0:
            return []

        df_rank["composite"] = sum(
            df_rank[f].fillna(0) * active_weights[f] for f in valid_factors
        ) / total_w

        # ── 动量修正加分：趋势确认股中，短期反弹力度强的优先 ──
        # 在趋势向上的股票中，反转因子选出回调到位的，加分给已确认反弹的
        for sym in df_rank.index:
            t = trend_info.get(sym, {})
            ret5 = t.get("ret_5d", 0)
            ret20 = t.get("ret_20d", 0)
            # 5日正收益+20日不暴跌 = 确认企稳反弹
            if ret5 > 0 and ret20 > -0.05:
                df_rank.loc[sym, "composite"] += 0.08
            elif ret5 > 0:
                df_rank.loc[sym, "composite"] += 0.04
            # 20日强势股额外加分（趋势跟随）
            if ret20 > 0.05:
                df_rank.loc[sym, "composite"] += 0.03

        top = df_rank.nlargest(TOP_N, "composite")
        return top.index.tolist()

    @staticmethod
    def _detect_market_state_at(symbol_groups: dict, today: str) -> str:
        """检测某日的市场状态（动量 + 趋势双层）。"""
        rets = []
        above_ma20 = 0
        total_ma = 0
        for sym, g in symbol_groups.items():
            g_cut = g[g["date"] <= today]
            if len(g_cut) < 22 or "close" not in g_cut.columns:
                continue
            last = float(g_cut["close"].iloc[-1])
            ma20 = float(g_cut["close"].iloc[-20:].mean())
            if ma20 > 0:
                total_ma += 1
                if last > ma20:
                    above_ma20 += 1
            r = g_cut["close"].iloc[-1] / g_cut["close"].iloc[-21] - 1
            if r == r:
                rets.append(float(r))
        if len(rets) < 50:
            return "neutral"
        median_ret = float(np.median(rets))
        # 趋势叠加层：MA20上方占比<45%强制bear（防止下跌途中反弹日误判）
        pct_above = above_ma20 / total_ma * 100 if total_ma else 50.0
        if median_ret > 0.03:
            state = "bull"
        elif median_ret < -0.03:
            state = "bear"
        else:
            state = "neutral"
        if pct_above < 45.0 and state != "bear":
            state = "neutral" if pct_above >= 40.0 else "bear"
        return state

    @staticmethod
    def _calc_atr_stop(symbol_groups: dict, symbol: str, today: str, entry_price: float,
                       atr_mult: float = 2.5, min_pct: float = 0.08, max_pct: float = 0.15) -> float:
        """ATR动态止损：根据股票自身波动率计算止损价。

        ATR高的股票（创业板/活跃股）止损宽，ATR低的股票（大盘/慢牛股）止损窄。
        范围限制：8%~15%（防止假止损）。

        Args:
            atr_mult: ATR乘数（2.5=2.5倍ATR，适配A股波动）
            min_pct: 最小止损幅度8%（A股日内波动常3-5%）
            max_pct: 最大止损幅度15%
        """
        g = symbol_groups.get(symbol)
        if g is None:
            return entry_price * (1 - max_pct)
        g_cut = g[g["date"] <= today]
        if len(g_cut) < 21:
            return entry_price * (1 - 0.12)  # 不足20天用默认12%

        high = g_cut["high"].iloc[-20:]
        low = g_cut["low"].iloc[-20:]
        close = g_cut["close"].iloc[-20:]
        prev_close = g_cut["close"].iloc[-21] if len(g_cut) > 20 else close.iloc[0]

        # True Range = max(H-L, |H-PrevC|, |L-PrevC|)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = float(tr.mean())
        if atr <= 0:
            return entry_price * (1 - 0.12)

        stop_pct = min(max(atr * atr_mult / entry_price, min_pct), max_pct)
        return entry_price * (1 - stop_pct)

    @staticmethod
    def _get_market_median_return(symbol_groups: dict, today: str, lookback: int) -> float:
        rets = []
        for sym, g in symbol_groups.items():
            g_cut = g[g["date"] <= today]
            if len(g_cut) < lookback + 1:
                continue
            r = g_cut["close"].iloc[-1] / g_cut["close"].iloc[-(lookback + 1)] - 1
            if r == r:
                rets.append(float(r))
        return float(np.median(rets)) if len(rets) >= 30 else 0.0

    def _calc_metrics(
        self, nav_curve: list, benchmark: list, trades: list, initial: float,
    ) -> dict:
        if len(nav_curve) < 2:
            return {}

        navs = [d["nav"] for d in nav_curve]
        final_nav = navs[-1]
        n_days = len(nav_curve)

        # 年化
        years = n_days / 252
        annual = (final_nav ** (1 / years) - 1) * 100 if years > 0 else 0

        # 最大回撤
        peak = navs[0]
        max_dd = 0
        for v in navs:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd

        # 日收益率序列 → 夏普
        daily_rets = [d["daily_return"] / 100 for d in nav_curve]
        arr = np.array(daily_rets)
        std = float(arr.std())
        sharpe = float(arr.mean() / std * math.sqrt(252)) if std > 0 else 0

        # 交易统计
        sell_trades = [t for t in trades if t.side == "sell"]
        wins = [t for t in sell_trades if t.pnl > 0]
        win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0

        total_pnl = sum(t.pnl for t in sell_trades)
        avg_pnl_pct = float(np.mean([t.pnl_pct for t in sell_trades])) if sell_trades else 0

        # 基准
        bench_final = benchmark[-1]["nav"] if benchmark else 1.0
        bench_annual = (bench_final ** (1 / years) - 1) * 100 if years > 0 else 0

        # Calmar
        calmar = annual / abs(max_dd) if max_dd != 0 else 0

        # 平均持仓天数
        avg_hold = float(np.mean([t.hold_days for t in sell_trades])) if sell_trades else 0

        # 月度胜率
        monthly_rets = {}
        for d in nav_curve:
            m = d["date"][:7]
            monthly_rets.setdefault(m, []).append(d["daily_return"] / 100)
        month_means = [float(np.mean(v)) for v in monthly_rets.values()]
        month_win = sum(1 for r in month_means if r > 0) / len(month_means) * 100 if month_means else 0

        return {
            "final_nav": round(final_nav, 4),
            "total_return": round((final_nav - 1) * 100, 2),
            "annual_return": round(annual, 2),
            "max_drawdown": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
            "calmar": round(calmar, 2),
            "win_rate": round(win_rate, 1),
            "total_trades": len(sell_trades),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_pct": round(avg_pnl_pct, 2),
            "avg_hold_days": round(avg_hold, 1),
            "benchmark_annual": round(bench_annual, 2),
            "benchmark_total": round((bench_final - 1) * 100, 2),
            "alpha": round(annual - bench_annual, 2),
            "monthly_win_rate": round(month_win, 1),
        }

    # ── 数据加载 ──

    def _load_all_daily(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql("SELECT * FROM stock_daily", conn)

    def _load_finance_map(self) -> dict[str, dict]:
        result = {}
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT symbol, roe, np_margin, gp_margin, yoy_eps, yoy_pni "
                "FROM stock_finance ORDER BY stat_date DESC"
            ).fetchall()
            seen = set()
            for r in rows:
                if r[0] in seen:
                    continue
                seen.add(r[0])
                result[r[0]] = {"roe": r[1], "np_margin": r[2], "gp_margin": r[3],
                                "rev_growth": r[4], "profit_growth": r[5]}
        return result

    def _load_state_weights(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {"bull": {}, "neutral": {}, "bear": {}}
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT market_state, factor_name, weight FROM market_factor_weights"
                ).fetchall()
            for state, fname, wt in rows:
                if wt and wt != 0:
                    result.setdefault(state, {})[fname] = wt
        except Exception:
            pass
        return result

    def _load_st_set(self) -> set:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT symbol FROM stock_basic "
                "WHERE name LIKE 'ST%' OR name LIKE '%*ST%' OR name LIKE '%退%'"
            ).fetchall()
            return {r[0] for r in rows}

    def _load_ipo_cutoff(self) -> dict:
        result = {}
        with sqlite3.connect(self.db_path) as conn:
            try:
                rows = conn.execute("SELECT symbol, ipo_date FROM stock_basic").fetchall()
                for sym, ipo in rows:
                    if ipo:
                        # 上市后60个交易日才纳入
                        result[sym] = ipo
            except Exception:
                pass
        return result

    @staticmethod
    def _trade_to_dict(t: ReplayTrade) -> dict:
        return {
            "symbol": t.symbol, "side": t.side, "date": t.date,
            "price": round(t.price, 2), "shares": t.shares,
            "amount": round(t.amount, 2), "pnl": round(t.pnl, 2),
            "pnl_pct": round(t.pnl_pct, 2), "hold_days": t.hold_days,
            "reason": t.reason,
        }

    # ------------------------------------------------------------------
    # 参数扫描：选股预计算 + 多参数组合快速模拟
    # ------------------------------------------------------------------

    def sweep(
        self,
        start_date: str = "",
        end_date: str = "",
        initial_capital: float = 100000.0,
        sample_size: int = 300,
        seed: int = 42,
        progress_callback=None,
    ) -> dict:
        """参数扫描：预计算选股截面一次，遍历止损/止盈/调仓/仓位组合。

        核心优化：选股结果与止损/止盈/调仓频率无关，
        所以只需一次全量因子计算，然后快速遍历参数组合。

        Returns:
            {
                "configs": [{stop_loss, take_profit, trailing_stop, rebalance, ...}, ...],
                "results": [{config_id, annual, max_dd, sharpe, win_rate, ...}, ...],
                "best": 最优配置,
                "baseline": 当前配置(对照),
            }
        """
        logger.info("参数扫描：加载数据...")
        all_daily = self._load_all_daily()
        finance_map = self._load_finance_map()
        state_weights = self._load_state_weights()
        st_set = self._load_st_set()
        cutoff_map = self._load_ipo_cutoff()

        all_dates = sorted(all_daily["date"].unique())
        if not start_date:
            start_date = all_dates[-250]
        if not end_date:
            end_date = all_dates[-1]
        replay_dates = [d for d in all_dates if start_date <= d <= end_date]

        all_symbols = sorted(all_daily["symbol"].unique())
        all_symbols = [s for s in all_symbols if s not in st_set]
        import random
        rng = random.Random(seed)
        if sample_size and len(all_symbols) > sample_size:
            sampled = rng.sample(all_symbols, sample_size)
        else:
            sampled = all_symbols

        daily_filtered = all_daily[all_daily["date"] <= end_date]
        symbol_groups = {sym: g for sym, g in daily_filtered.groupby("symbol", sort=False)
                         if sym in set(sampled)}

        # 价格字典：{date: {symbol: close}}
        price_lookup = {}
        pct_lookup = {}
        for _, row in all_daily.iterrows():
            d = row["date"]
            if d not in price_lookup:
                price_lookup[d] = {}
                pct_lookup[d] = {}
            price_lookup[d][row["symbol"]] = row["close"]
            pct_val = row.get("pct_chg")
            pct_lookup[d][row["symbol"]] = pct_val if pct_val == pct_val else 0.0

        # ── 预计算每个调仓日的选股结果 ──
        # 选股频率取最细粒度（3天），更粗的频率取子集
        finest_interval = 3
        logger.info(f"参数扫描：预计算选股截面（{len(replay_dates)}天，间隔{finest_interval}天）")

        daily_selections: dict[str, list[str]] = {}  # {date: [top symbols]}
        daily_market_state: dict[str, str] = {}
        daily_median_20d: dict[str, float] = {}

        for day_idx, today in enumerate(replay_dates):
            if day_idx % finest_interval != 0:
                continue
            if progress_callback:
                pct_done = int(day_idx / len(replay_dates) * 50)
                progress_callback(pct_done, 100, f"预计算选股 {day_idx}/{len(replay_dates)}")

            market_state = self._detect_market_state_at(symbol_groups, today)
            daily_market_state[today] = market_state
            daily_median_20d[today] = self._get_market_median_return(symbol_groups, today, 20)

            candidates = self._select_top(
                symbol_groups, today, state_weights, finance_map, cutoff_map, st_set
            )
            if candidates:
                daily_selections[today] = candidates

        logger.info(f"选股预计算完成：{len(daily_selections)}个截面")

        # ── 定义参数组合 ──
        param_grid = [
            # (stop_loss, take_profit, trailing_stop, rebalance_interval, max_holdings, max_total_pct)
            (-5.0,  20.0, -3.0,  5, 8, 0.80),   # 当前基线
            (-8.0,  20.0, -5.0,  5, 8, 0.80),   # 当前实际参数
            (-12.0, 30.0, -8.0,  5, 8, 0.80),   # 放宽止损止盈
            (-15.0, 40.0, -10.0, 5, 8, 0.80),   # 更宽松
            (-12.0, 0,    -8.0,  5, 8, 0.80),   # 只止损不主动止盈
            (-8.0,  20.0, -5.0, 10, 8, 0.80),   # 低频调仓
            (-12.0, 30.0, -8.0, 10, 8, 0.80),   # 放宽+低频
            (-12.0, 30.0, -8.0, 20, 8, 0.80),   # 月频调仓
            (-15.0, 0,    -10.0, 10, 8, 0.95),  # 宽止损+不止盈+高仓位
            (-12.0, 30.0, -8.0, 10, 5, 0.80),   # 集中持仓5只
            (-12.0, 30.0, -8.0, 10, 12, 0.80),  # 分散持仓12只
            (-15.0, 0,    -8.0, 20, 8, 0.95),   # 买入持有风格
        ]

        results = []
        total_configs = len(param_grid)

        for cfg_idx, (sl, tp, ts, rebal, max_h, max_tp) in enumerate(param_grid):
            if progress_callback:
                progress_callback(50 + int(cfg_idx / total_configs * 50), 100,
                                  f"模拟参数组合 {cfg_idx+1}/{total_configs}")

            metrics = self._simulate_portfolio(
                replay_dates, daily_selections, daily_market_state, daily_median_20d,
                price_lookup, pct_lookup,
                initial_capital, sl, tp, ts, rebal, max_h, max_tp,
            )
            results.append({
                "config_id": cfg_idx,
                "stop_loss": sl,
                "take_profit": tp if tp > 0 else None,
                "trailing_stop": ts,
                "rebalance_interval": rebal,
                "max_holdings": max_h,
                "max_total_pct": max_tp,
                **metrics,
            })

        # 排序：按夏普降序
        results.sort(key=lambda x: x.get("sharpe", -999), reverse=True)

        # 基线对照
        baseline = next((r for r in results if r["config_id"] == 1), results[0])
        best = results[0]

        return {
            "configs": results,
            "best": best,
            "baseline": baseline,
            "period": {"start": replay_dates[0], "end": replay_dates[-1], "days": len(replay_dates)},
            "sample_size": len(symbol_groups),
            "selections_computed": len(daily_selections),
        }

    def _simulate_portfolio(
        self, replay_dates: list, daily_selections: dict,
        daily_market_state: dict, daily_median_20d: dict,
        price_lookup: dict, pct_lookup: dict,
        initial_capital: float, stop_loss: float, take_profit: float,
        trailing_stop: float, rebal_interval: int,
        max_holdings: int, max_total_pct: float,
    ) -> dict:
        """快速组合模拟（选股已预计算，只做持仓管理）。"""

        @dataclass
        class Pos:
            symbol: str
            entry_date: str
            entry_date_idx: int
            entry_price: float
            shares: int
            highest: float

        cash = initial_capital
        positions: list[Pos] = []
        nav_curve = []
        trades_count = 0
        sell_trades = []

        portfolio_high = initial_capital

        for day_idx, today in enumerate(replay_dates):
            prices = price_lookup.get(today, {})

            # 止损/止盈检查
            new_positions = []
            for pos in positions:
                price = prices.get(pos.symbol)
                if price is None or price <= 0:
                    new_positions.append(pos)
                    continue

                pos.highest = max(pos.highest, price)
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                eff_stop = max(pos.entry_price * (1 + stop_loss / 100),
                               pos.highest * (1 + trailing_stop / 100))
                days_held = day_idx - pos.entry_date_idx

                should_sell = False
                reason = ""
                if price <= eff_stop and pnl_pct < 0:
                    should_sell = True
                    reason = f"止损{pnl_pct:.1f}%"
                elif take_profit > 0 and pnl_pct >= take_profit:
                    should_sell = True
                    reason = f"止盈{pnl_pct:.1f}%"
                elif days_held >= 60:
                    should_sell = True
                    reason = f"超时{pnl_pct:.1f}%"

                if should_sell:
                    amount = price * pos.shares
                    cost = amount * RTC
                    pnl = (price - pos.entry_price) * pos.shares - cost
                    cash += amount - cost
                    trades_count += 1
                    sell_trades.append({"pnl": pnl, "pnl_pct": pnl_pct, "hold_days": days_held})
                else:
                    new_positions.append(pos)
            positions = new_positions

            # 调仓
            if day_idx % rebal_interval == 0 and today in daily_selections:
                market_state = daily_market_state.get(today, "neutral")
                adaptive_scale = {"bull": 1.0, "neutral": 0.9, "bear": 0.5}
                position_scale = adaptive_scale.get(market_state, 0.7)

                # 极端弱市：不买新仓，不清仓（避免踏空反弹）
                median_20d = daily_median_20d.get(today, 0)
                if market_state == "bear" and median_20d < -0.05:
                    pass  # 跳过买入，让止损自然退出
                else:
                    candidates = daily_selections.get(today, [])
                    held = {p.symbol for p in positions}
                    candidates = [c for c in candidates if c not in held]

                    holdings_val = sum(prices.get(p.symbol, p.entry_price) * p.shares for p in positions)
                    max_total = initial_capital * max_total_pct * position_scale
                    available = min(cash, max_total - holdings_val)
                    slots = max_holdings - len(positions)

                    for sym in candidates[:slots]:
                        price = prices.get(sym)
                        if price is None or price <= 0:
                            continue
                        target = min(initial_capital * 0.15, available / max(slots, 1))
                        if target < price * 100:
                            continue
                        shares = int(target / price / 100) * 100
                        if shares <= 0:
                            continue
                        total_cost = price * shares * (1 + RTC)
                        if total_cost > cash:
                            continue
                        cash -= total_cost
                        positions.append(Pos(sym, today, day_idx, price, shares, price))

            # 净值
            mv = sum(prices.get(p.symbol, p.entry_price) * p.shares for p in positions)
            total = cash + mv
            portfolio_high = max(portfolio_high, total)
            nav_curve.append(total)

        # 绩效计算
        if len(nav_curve) < 2:
            return {}

        final = nav_curve[-1] / initial_capital
        years = len(nav_curve) / 252
        annual = (final ** (1 / years) - 1) * 100 if years > 0 else 0

        peak = nav_curve[0]
        max_dd = 0
        for v in nav_curve:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd

        nav_rets = np.diff(nav_curve) / np.array(nav_curve[:-1])
        std = float(nav_rets.std()) if len(nav_rets) > 1 else 0
        sharpe = float(nav_rets.mean() / std * math.sqrt(252)) if std > 0 else 0

        win_rate = (sum(1 for t in sell_trades if t["pnl"] > 0) / len(sell_trades) * 100) if sell_trades else 0
        calmar = annual / abs(max_dd) if max_dd != 0 else 0

        return {
            "final_nav": round(final, 4),
            "total_return": round((final - 1) * 100, 2),
            "annual_return": round(annual, 2),
            "max_drawdown": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
            "calmar": round(calmar, 2),
            "win_rate": round(win_rate, 1),
            "total_trades": trades_count,
            "avg_hold_days": round(float(np.mean([t["hold_days"] for t in sell_trades])), 1) if sell_trades else 0,
        }

    # ------------------------------------------------------------------
    # 闭环2：回放交易明细反馈因子权重
    # ------------------------------------------------------------------

    def feedback_factor_weights(self, trades: list[dict]) -> list[dict]:
        """闭环2：回放交易明细 → 因子权重反馈。

        分析回放中止损vs止盈的交易，按因子类别统计胜率，
        止损率高的因子类别降低权重，止盈率高的提升权重。

        Args:
            trades: 回放返回的交易明细列表

        Returns:
            调整记录列表
        """
        import sqlite3 as _sq
        import time as _time

        sells = [t for t in trades if t.get("side") == "sell"]
        if len(sells) < 10:
            logger.info("回放反馈：交易样本不足(<10)，跳过")
            return []

        # 按卖出原因分类
        stop_loss_trades = [t for t in sells if "止损" in t.get("reason", "")]
        take_profit_trades = [t for t in sells if "止盈" in t.get("reason", "")]
        timeout_trades = [t for t in sells if "超时" in t.get("reason", "")]

        overall_stop_rate = len(stop_loss_trades) / len(sells) if sells else 0
        logger.info(
            f"回放反馈：{len(sells)}笔卖出，止损{len(stop_loss_trades)}笔"
            f"({overall_stop_rate*100:.0f}%)，止盈{len(take_profit_trades)}笔"
        )

        # 止损率过高 → 说明当前因子组合选出的票路径风险大
        # 策略：整体降低动量/反转因子权重（路径风险最高的类别），提升质量因子权重
        adjustments = []
        now = _time.strftime("%Y-%m-%d %H:%M:%S")

        # 根据止损率决定调整幅度
        if overall_stop_rate > 0.6:
            # 止损率>60%：动量/反转/流动性因子降权，质量/估值提权
            factor_adjustments = {
                # 路径风险高的因子降权
                "动量": 0.85, "反转": 0.90, "流动性": 0.90,
                # 路径风险低的因子提权
                "质量": 1.08, "估值": 1.08, "波动": 1.05,
            }
            severity = "高止损率(>{:.0f}%)".format(overall_stop_rate * 100)
        elif overall_stop_rate > 0.4:
            # 止损率40-60%：轻微调整
            factor_adjustments = {
                "动量": 0.93, "反转": 0.93, "流动性": 0.96,
                "质量": 1.08, "估值": 1.08,
            }
            severity = "中等止损率({:.0f}%)".format(overall_stop_rate * 100)
        else:
            # 止损率<40%：表现良好，轻微提权动量（路径风险已可控）
            factor_adjustments = {
                "动量": 1.05, "反转": 1.03,
                "质量": 1.03, "估值": 1.03,
            }
            severity = "低止损率({:.0f}%)".format(overall_stop_rate * 100)

        try:
            with _sq.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT market_state, factor_name, category, weight "
                    "FROM market_factor_weights WHERE weight != 0"
                ).fetchall()

                updates = []
                for state, fname, category, old_wt in rows:
                    multiplier = factor_adjustments.get(category, 1.0)
                    if multiplier == 1.0:
                        continue

                    new_wt = old_wt * multiplier
                    # 限制单次调整幅度 ±5%（收紧防destabilize）
                    if abs(new_wt) > abs(old_wt) * 1.05:
                        new_wt = old_wt * 1.05
                    elif abs(new_wt) < abs(old_wt) * 0.95:
                        new_wt = old_wt * 0.95

                    updates.append((new_wt, now, state, fname))
                    adjustments.append({
                        "factor": fname, "state": state, "category": category,
                        "old_weight": round(old_wt, 4),
                        "new_weight": round(new_wt, 4),
                        "multiplier": multiplier,
                        "reason": f"{severity}→{category}类{('降权' if multiplier < 1 else '提权')}{abs(multiplier-1)*100:.0f}%",
                    })

                if updates:
                    conn.executemany(
                        "UPDATE market_factor_weights SET weight=?, updated_at=? "
                        "WHERE market_state=? AND factor_name=?",
                        updates
                    )
                    logger.info(f"回放反馈：基于止损率{overall_stop_rate*100:.0f}%调整{len(updates)}个因子权重")
        except Exception as e:
            logger.warning(f"回放权重反馈失败：{e!r}")

        return adjustments
