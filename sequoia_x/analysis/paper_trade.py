"""模拟盘引擎：10万本金全自动闭环（选股→买入→跟踪→卖出→绩效）。

核心流程：
  1. auto_buy(decision_result) — 决策跑完后自动买入buy_list
  2. auto_sell(position_signals) — 持仓扫描后自动执行止损/止盈/减仓信号
  3. get_performance() — 绩效统计（收益率/胜率/盈亏比/回撤）

风控规则：
  - 单票仓位 ≤ 15%（避免集中）
  - 总仓位 ≤ 80%（留20%现金）
  - 市场bear评分<10时空仓不买
  - 模拟成交价用当日收盘价（后复权→真实价转换）
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from sequoia_x.analysis.stop_loss import (
    DEFAULT_FALLBACK_PCT, calc_atr_stop, resolve_entry_stop,
)
from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# ── 风控参数 ──
MAX_POSITION_PCT = 0.15      # 单票最大仓位15%
MAX_TOTAL_PCT = 0.80         # 总仓位上限80%
MIN_BUY_SCORE = 50           # 最低买入评分
BEAR_NO_BUY_SCORE = 15       # 市场评分<此值时空仓不买
# ── 风控熔断参数 ──
MAX_DRAWDOWN_CIRCUIT = 0.15   # 最大回撤>15%触发熔断（暂停开新仓）
MAX_CONSEC_LOSS = 3           # 连续亏损≥3笔降仓
CONSEC_LOSS_SCALE = 0.5       # 连亏时仓位缩放比例
MAX_DAILY_LOSS_PCT = 3.0      # 单日亏损>3%停止开仓
# ── 板块集中度限制 ──
MAX_BOARD_PCT = {
    "gem": 0.30,      # 创业板(30x)：±20%涨跌停，高波动，上限30%
    "star": 0.30,     # 科创板(688)：±20%涨跌停，高波动，上限30%
    "main": 0.60,     # 主板(60x/00x)：±10%，流动性好，上限60%
}
# ── 市场状态自适应风控 ──
# bear市场：提高评分门槛、降低总仓位、排除高波动板块
MARKET_ADAPTIVE = {
    "bull":    {"min_score": 50, "position_scale": 1.0, "exclude_high_volatility": False, "label": "牛市正常买入"},
    "neutral": {"min_score": 55, "position_scale": 0.7, "exclude_high_volatility": False, "label": "中性谨慎建仓"},
    "bear":    {"min_score": 65, "position_scale": 0.4, "exclude_high_volatility": True,  "label": "熊市仅主板+低仓位"},
}
BEAR_NO_BUY_SCORE = 30  # 极端弱市(评分<30)空仓不买
# ── 仓位管理参数 ──
BASE_RISK_PCT = 0.02          # 单笔风险占总资产2%（风险预算）
MAX_RISK_PCT = 0.04           # 单笔最大风险4%
SCORE_POWER = 1.5             # 评分→仓位幂次（评分越高仓位越大）
# ── 止盈参数 ──
PARTIAL_TP_PCT = 15.0         # 浮盈>15%减仓一半（扫描最优：放宽止盈）
FULL_TP_PCT = 30.0            # 浮盈>30%全部止盈（扫描最优：让赢家跑更远）
TRAILING_START_PCT = 12.0     # 浮盈>12%启动移动止盈（扫描最优）
TRAILING_PULLBACK = 8.0       # 移动止盈回撤8%（扫描最优：避免假止损）
MAX_HOLD_DAYS = 60            # 持仓超60天强制平仓（扫描最优：降低换手）

# ── 交易成本参数（实盘口径）──
BUY_COMMISSION_RATE = 0.00025   # 买入佣金 0.025%（券商普遍费率，含规费）
SELL_COMMISSION_RATE = 0.00025  # 卖出佣金 0.025%
STAMP_DUTY_RATE = 0.0005        # 印花税 0.05%（卖出单边征收）
SLIPPAGE_RATE = 0.001           # 滑点 0.1%（突破买入成交价偏高 / 卖出偏低）


def apply_trading_costs(side: str, price: float, shares: int) -> dict:
    """计算实盘交易成本，返回成交价与各项费用。

    买方成交价偏高（滑点向不利方向）、卖方成交价偏低，佣金按成交额计，
    印花税仅卖出征收。该函数供模拟盘与回放引擎共用，保证口径一致。

    Args:
        side: 'buy' 或 'sell'
        price: 名义成交价（决策价/收盘价）
        shares: 成交股数

    Returns:
        {fill_price, amount, commission, stamp_duty, net_cash}
        buy: net_cash 为现金流出（正数）；sell: net_cash 为现金净流入（正数）
    """
    if side == "buy":
        fill_price = round(price * (1 + SLIPPAGE_RATE), 3)
        amount = fill_price * shares
        commission = amount * BUY_COMMISSION_RATE
        net_cash = amount + commission  # 现金流出
        return {"fill_price": fill_price, "amount": amount,
                "commission": commission, "stamp_duty": 0.0, "net_cash": net_cash}
    else:
        fill_price = round(price * (1 - SLIPPAGE_RATE), 3)
        amount = fill_price * shares
        commission = amount * SELL_COMMISSION_RATE
        stamp_duty = amount * STAMP_DUTY_RATE
        net_cash = amount - commission - stamp_duty  # 现金净流入
        return {"fill_price": fill_price, "amount": amount,
                "commission": commission, "stamp_duty": stamp_duty, "net_cash": net_cash}


@dataclass


class PaperPerformance:
    """模拟盘绩效快照（专业量化指标）。"""
    initial_capital: float = 0
    cash: float = 0
    market_value: float = 0        # 持仓市值
    total_assets: float = 0        # 总资产 = 现金 + 持仓市值
    total_return_pct: float = 0    # 累计收益率
    total_trades: int = 0          # 总卖出笔数（已完成交易）
    win_trades: int = 0            # 盈利笔数
    loss_trades: int = 0           # 亏损笔数
    win_rate: float = 0            # 胜率
    avg_win: float = 0             # 平均盈利%
    avg_loss: float = 0            # 平均亏损%
    profit_factor: float = 0       # 盈亏比（总盈利/总亏损绝对值）
    total_pnl: float = 0           # 已实现盈亏
    max_drawdown_pct: float = 0    # 最大回撤
    holding_count: int = 0         # 当前持仓数
    best_trade_pct: float = 0      # 单笔最大盈利%
    worst_trade_pct: float = 0     # 单笔最大亏损%
    avg_hold_days: float = 0       # 平均持仓天数
    # ── 专业指标（需日度NAV数据支撑）──
    sharpe_ratio: float = 0        # 夏普比率（年化）
    sortino_ratio: float = 0       # 索提诺比率（年化）
    calmar_ratio: float = 0        # 卡玛比率（年化收益/最大回撤）
    annual_return: float = 0       # 年化收益率%
    alpha: float = 0               # 超额收益（相对沪深300）
    benchmark_return: float = 0    # 沪深300同期收益率%
    consec_loss: int = 0           # 当前连续亏损笔数
    risk_circuit: str = "正常"     # 风控状态



def _classify_board(symbol: str) -> str:
    """根据股票代码判断板块。"""
    sym = str(symbol).strip()
    if sym.startswith("30"):
        return "gem"
    if sym.startswith("688"):
        return "star"
    return "main"


class PaperTradeEngine:
    """模拟盘交易引擎。"""

    def __init__(self, settings: Settings) -> None:
        self.db_path = settings.db_path
        self.account_id = 1  # 默认账户
        self._ensure_account()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_account(self) -> None:
        """确保默认账户存在（首次运行自动创建10万本金账户）。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM paper_account WHERE id=1"
            ).fetchone()
            if not row:
                now = datetime.now().isoformat()
                conn.execute(
                    "INSERT INTO paper_account (id, name, initial_capital, cash, created_at, updated_at) "
                    "VALUES (1, 'default', 100000, 100000, ?, ?)",
                    (now, now),
                )
                conn.commit()
                logger.info("模拟盘账户已创建：本金10万")

    # ════════════════════════════════════════
    # 账户查询
    # ════════════════════════════════════════

    def get_account(self) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM paper_account WHERE id=1"
            ).fetchone()
            return dict(row) if row else {}

    def get_holdings(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_holdings WHERE account_id=1 ORDER BY entry_date DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_trades(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_trades WHERE account_id=1 "
                "ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ════════════════════════════════════════
    # 自动买入
    # ════════════════════════════════════════

    def auto_buy(self, decision_result: dict) -> dict:
        """根据决策结果自动买入。

        业务逻辑：
          1. 市场bear评分<BEAR_NO_BUY_SCORE → 空仓不买
          2. 按buy_list评分排序，逐只评估仓位
          3. 单票仓位=min(评分对应仓位, 单票上限, 可用资金)
          4. 总仓位超80%停止买入
          5. 已持有的票跳过（不加仓，除非显式允许）

        Args:
            decision_result: generate_decision返回的完整结果

        Returns:
            {bought: [...], skipped: [...], reason: str}
        """
        account = self.get_account()
        cash = account.get("cash", 0)
        initial = account.get("initial_capital", 100000)

        # 市场状态检查
        market = decision_result.get("market_state", {})
        market_score = market.get("score", 50)
        market_state = market.get("state", "neutral")
        # ── 自适应风控参数 ──
        adaptive = MARKET_ADAPTIVE.get(market_state, MARKET_ADAPTIVE["neutral"])
        min_score_eff = max(MIN_BUY_SCORE, adaptive["min_score"])
        position_scale_eff = adaptive["position_scale"]
        exclude_hv = adaptive["exclude_high_volatility"]
        logger.info(f"模拟盘自适应风控：市场={market_state}({market_score}分) → "
                     f"门槛{min_score_eff} 仓位×{position_scale_eff} "
                     f"{'排除高波动板块' if exclude_hv else '不排除'} ({adaptive['label']})")
        position_scale = market.get("position_scale", 0.8)

        if market_state == "bear" and market_score < BEAR_NO_BUY_SCORE:
            logger.info(f"模拟盘：市场极弱(评分{market_score}<{BEAR_NO_BUY_SCORE})，空仓不买")
            return {"bought": [], "skipped": [], "reason": f"市场极弱(评分{market_score})，空仓等待"}

        # ── 风控熔断检查 ──
        circuit = self._check_risk_circuit(initial)
        if circuit["halt"]:
            logger.warning(f"模拟盘风控熔断：{circuit['reason']}，暂停开新仓")
            return {"bought": [], "skipped": [], "reason": circuit["reason"]}

        # ── 组合级风控检查（Beta/VaR/回撤/集中度）──
        try:
            from sequoia_x.analysis.portfolio_risk import PortfolioRiskMonitor
            risk_monitor = PortfolioRiskMonitor(self.db_path)
            risk_report = risk_monitor.analyze(cash=cash)
            if risk_report.risk_score < 40:
                danger_alerts = [a for a in risk_report.alerts if a.level == "danger"]
                if danger_alerts:
                    reasons = "; ".join(a.message for a in danger_alerts[:2])
                    logger.warning(f"模拟盘组合风控拦截：风险评分{risk_report.risk_score}，{reasons}")
                    return {"bought": [], "skipped": [], "reason": f"组合风控拦截({risk_report.risk_score}分)：{reasons}"}
        except Exception as e:
            logger.warning(f"组合风控检查跳过：{e!r}")

        # 当前持仓总市值
        holdings = self.get_holdings()
        holding_symbols = {h["symbol"] for h in holdings}
        holding_value = self._calc_holding_value(holdings)

        buy_list = decision_result.get("buy_list", [])
        if not buy_list:
            return {"bought": [], "skipped": [], "reason": "决策无买入清单"}

        # 按评分排序（高分优先）
        # 排序：评分→板块偏好(主板优先)→低价优先(买得起整手)
        def _sort_key(x):
            sym = x.get("symbol", "")
            score = x.get("score", 0)
            board = _classify_board(sym)
            # 主板+0.5分加权、创业板/科创板不加权，确保同评分主板优先
            board_bonus = 0.5 if board == "main" else 0
            return -(score + board_bonus)
        buy_sorted = sorted(buy_list, key=_sort_key)

        bought = []
        skipped = []
        today = datetime.now().strftime("%Y-%m-%d")
        max_total = initial * MAX_TOTAL_PCT * position_scale_eff  # 自适应缩放

        for item in buy_sorted:
            sym = item.get("symbol", "")
            score = item.get("score", 0)
            if score < min_score_eff:  # 自适应门槛
                skipped.append({"symbol": sym, "reason": f"评分{score}<{MIN_BUY_SCORE}"})
                continue

            # 已持有跳过
            if sym in holding_symbols:
                skipped.append({"symbol": sym, "reason": "已持有"})
                continue

            # bear市场排除高波动板块（创业板/科创板）
            board = _classify_board(sym)
            if exclude_hv and board in ("gem", "star"):
                skipped.append({"symbol": sym, "reason": f"熊市排除高波动板块({board})"})
                continue

            # 板块集中度风控：限制单一板块持仓占比
            board_pct_limit = MAX_BOARD_PCT.get(board, 0.30)
            board_value = sum(
                h2["shares"] * h2["entry_price"]
                for h2 in holdings
                if _classify_board(h2.get("symbol", "")) == board
            )
            board_pct_now = board_value / initial if initial else 0
            if board_pct_now >= board_pct_limit:
                skipped.append({"symbol": sym, "reason": f"板块({board})仓位{board_pct_now:.0%}≥上限{board_pct_limit:.0%}"})
                continue

            # 仓位计算：风险预算（信号置信度×波动率调整）
            position_pct = self._calc_risk_position(
                score=score, symbol=sym, stop_loss=item.get("stop_loss", 0),
                entry_price_hint=self._get_close_price(sym, today),
                position_scale=position_scale, consec_loss=circuit.get("consec_loss", 0),
            )
            target_amount = initial * position_pct

            if cash < target_amount * 0.5:
                skipped.append({"symbol": sym, "reason": f"现金不足(需{target_amount:.0f}/有{cash:.0f})"})
                continue

            # 总仓位约束
            if holding_value + target_amount > max_total:
                skipped.append({"symbol": sym, "reason": "总仓位将超80%"})
                continue

            # 获取成交价（当日收盘价，后复权→真实价）
            price = self._get_close_price(sym, today)
            if not price or price <= 0:
                skipped.append({"symbol": sym, "reason": "无成交价"})
                continue

            # 实战成交假设：涨停封板/停牌的票买不进（与决策层 _filter_limit_up 口径一致）
            tradable, not_reason = self._check_tradable(sym, today)
            if not tradable:
                skipped.append({"symbol": sym, "reason": not_reason})
                continue

            # 高价股补救：目标仓位不足1手但1手在单票上限内 → 最低建仓1手
            one_lot_cost = price * 100
            max_per_lot = initial * MAX_POSITION_PCT
            if target_amount < one_lot_cost <= max_per_lot:
                target_amount = one_lot_cost

            # 计算股数（整手100股）
            shares = int(target_amount / price / 100) * 100
            if shares <= 0:
                skipped.append({"symbol": sym, "reason": f"金额{target_amount:.0f}不足买1手@{price:.2f}"})
                continue

            amount = shares * price
            if amount > cash:
                shares = int(cash / price / 100) * 100
                amount = shares * price
                if shares <= 0:
                    skipped.append({"symbol": sym, "reason": "现金不足1手"})
                    continue

            # ── P5：入场 ATR 自适应止损统一 ──
            # decision/stock_analysis 传来的 stop_loss 散乱（-4%~-20% 且无封顶），
            # 用 2.5×ATR（封顶 8%~15%）统一口径：为空/过宽(>15%)时用 ATR 重算。
            raw_stop = item.get("stop_loss", 0) or 0.0
            atr_stop = self._calc_entry_atr_stop(sym, today, price)
            entry_stop = resolve_entry_stop(raw_stop, price, atr_stop)
            if entry_stop != raw_stop:
                logger.info(
                    f"模拟盘入场止损统一 {sym}：原{raw_stop}→ATR{entry_stop:.2f}"
                    f"(距{((price - entry_stop) / price * 100):.1f}%)"
                )

            # 执行买入
            actual_cash_out = self._execute_buy(
                symbol=sym, name=item.get("name", ""),
                price=price, shares=shares, amount=amount,
                date=today, stop_loss=entry_stop,
                target=item.get("target", 0), grade=item.get("grade", ""),
                hit_strategies=item.get("hit_strategies", ""),
                reason=f"决策评分{score} {item.get('action','')}",
            )
            cash -= actual_cash_out
            holding_value += actual_cash_out
            holding_symbols.add(sym)
            bought.append({
                "symbol": sym, "name": item.get("name", ""),
                "price": price, "shares": shares, "amount": round(amount, 2),
                "score": score,
            })
            logger.info(f"模拟买入：{sym} {shares}股@{price:.2f}={amount:.0f}元")

        return {
            "bought": bought,
            "skipped": skipped,
            "reason": f"买入{len(bought)}只，跳过{len(skipped)}只",
        }

    def _execute_buy(self, symbol: str, name: str, price: float, shares: int,
                     amount: float, date: str, stop_loss: float, target: float,
                     grade: str, hit_strategies, reason: str) -> float:
        """执行买入：扣现金（含成本）+ 写持仓 + 写交易记录。

        成本模型：成交价含滑点（买方偏高），另收佣金。entry_price 记实际成交价，
        使止损/盈亏计算基于真实成本。返回实际现金流出额（含佣金）。
        """
        # hit_strategies 可能为 list，SQLite 不支持绑定 list，需转为字符串
        if isinstance(hit_strategies, (list, tuple)):
            hit_strategies = ", ".join(str(s) for s in hit_strategies)
        elif hit_strategies is None:
            hit_strategies = ""
        cost = apply_trading_costs("buy", price, shares)
        fill_price = cost["fill_price"]
        net_cash = cost["net_cash"]  # 现金流出（成交额 + 佣金）
        now = datetime.now().isoformat()
        with self._conn() as conn:
            # 扣现金（成交额 + 佣金）
            conn.execute(
                "UPDATE paper_account SET cash=cash-?, updated_at=? WHERE id=1",
                (net_cash, now),
            )
            # 写持仓
            conn.execute(
                "INSERT OR REPLACE INTO paper_holdings "
                "(account_id, symbol, name, entry_price, shares, entry_date, "
                "stop_loss, initial_stop, target, grade, hit_strategies, cost) "
                "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, name, fill_price, shares, date,
                 stop_loss, stop_loss, target, grade, hit_strategies, net_cash),
            )
            # 写交易记录
            conn.execute(
                "INSERT INTO paper_trades "
                "(account_id, symbol, name, side, price, shares, amount, date, reason, created_at) "
                "VALUES (1, ?, ?, 'buy', ?, ?, ?, ?, ?, ?)",
                (symbol, name, fill_price, shares, cost["amount"], date, reason, now),
            )
            conn.commit()
        return net_cash

    # ════════════════════════════════════════
    # 自动卖出
    # ════════════════════════════════════════

    def auto_sell(self, position_signals: list[dict]) -> dict:
        """根据持仓扫描信号自动执行卖出。

        业务逻辑：
          - 止损清仓/止盈 → 全部卖出
          - 减仓半仓 → 卖出一半
          - 移动止损 → 只更新止损价，不卖出
          - 持有/逻辑减弱/相对走弱/放量滞涨/持仓低效 → 不卖，仅记录

        Args:
            position_signals: scan_all返回的信号列表（dict格式）

        Returns:
            {sold: [...], updated: [...], reason: str}
        """
        sold = []
        updated = []
        today = datetime.now().strftime("%Y-%m-%d")

        sell_actions = {"止损清仓", "止盈", "减仓/清仓"}
        # 弱信号也触发减仓半仓：主动调仓换股
        half_actions = {"减仓半仓", "相对走弱", "逻辑减弱", "放量滞涨", "持仓低效"}

        # ── 增强卖出信号：遍历当前持仓，检查止盈/移动止损/到期 ──
        enhanced_signals = self._generate_enhanced_sells(position_signals, today)
        position_signals = enhanced_signals

        for sig in position_signals:
            sym = sig.get("symbol", "")
            action = sig.get("action", "持有")

            if action in sell_actions:
                price = self._get_close_price(sym, today)
                if not price:
                    continue
                result = self._execute_sell(sym, price, sig.get("shares", 0), today, action, sig.get("reasons", []))
                if result:
                    sold.append(result)
            elif action in half_actions:
                price = self._get_close_price(sym, today)
                if not price:
                    continue
                # 减仓半仓：卖出一半，向下取整到100的倍数
                half = (sig.get("shares", 0) // 2 // 100) * 100
                if half <= 0:
                    continue
                result = self._execute_sell(sym, price, half, today, action, sig.get("reasons", []), partial=True)
                if result:
                    sold.append(result)
            elif action == "移动止损" and sig.get("new_stop", 0) > 0:
                # 只更新止损价
                self._update_stop(sym, sig["new_stop"])
                updated.append({"symbol": sym, "new_stop": sig["new_stop"]})

        return {
            "sold": sold,
            "updated": updated,
            "reason": f"卖出{len(sold)}笔，更新止损{len(updated)}只",
        }

    def auto_sell_intraday(self, sell_signals: list[dict]) -> dict:
        """盘中实时卖出：用信号中的实时价格执行。

        与 auto_sell 的区别：
          - 用实时快照价（sell_signals[].realtime_price）而非昨日收盘价
          - 不重新生成增强信号（盘中扫描已包含）
          - 立即执行，不等待盘后闭环
        """
        sold = []
        updated = []
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%H:%M:%S")

        sell_actions = {"止损清仓", "止盈", "减仓/清仓"}
        # 弱信号盘中也触发减仓半仓
        half_actions = {"减仓半仓", "相对走弱", "逻辑减弱", "放量滞涨", "持仓低效"}

        for sig in sell_signals:
            sym = sig.get("symbol", "")
            action = sig.get("action", "")
            realtime_price = sig.get("realtime_price", 0)

            if not sym or not realtime_price or realtime_price <= 0:
                continue

            # 止损清仓/止盈 → 全部卖出
            if action in sell_actions:
                result = self._execute_sell(
                    sym, realtime_price, sig.get("shares", 0),
                    today, action, sig.get("reasons", [])
                )
                if result:
                    result["timestamp"] = timestamp
                    sold.append(result)
                    logger.info(
                        f"盘中实时卖出：{sym} {result['shares']}股@{realtime_price:.2f}"
                        f"={result['amount']:.0f}元 盈亏{result['pnl']:+.0f}"
                        f"({result['pnl_pct']:+.1f}%) [{timestamp}]"
                    )

            # 减仓半仓 → 卖一半
            elif action in half_actions:
                # 减仓半仓：向下取整到100的倍数
                half_shares = (sig.get("shares", 0) // 2 // 100) * 100
                if half_shares > 0:
                    result = self._execute_sell(
                        sym, realtime_price, half_shares,
                        today, action, sig.get("reasons", [])
                    )
                    if result:
                        result["timestamp"] = timestamp
                        sold.append(result)

            # 移动止损 → 只更新止损价
            elif action == "移动止损" and sig.get("new_stop", 0) > 0:
                self._update_stop(sym, sig["new_stop"])
                updated.append({"symbol": sym, "new_stop": sig["new_stop"]})

        if sold:
            logger.warning(f"盘中实时卖出完成：{len(sold)}笔")
        return {
            "sold": sold,
            "updated": updated,
            "reason": f"盘中卖出{len(sold)}笔，更新止损{len(updated)}只",
        }

    def _execute_sell(self, symbol: str, price: float, shares: int,
                      date: str, reason: str, reasons: list, partial: bool = False) -> dict | None:
        """执行卖出：加现金 + 删/改持仓 + 写交易记录（含盈亏）。"""
        with self._conn() as conn:
            h = conn.execute(
                "SELECT * FROM paper_holdings WHERE account_id=1 AND symbol=?", (symbol,)
            ).fetchone()
            if not h:
                return None
            h = dict(h)

            # 整手处理：卖出股数向下取整到100的倍数（A股最小交易单位1手=100股）
            sell_shares = min(shares, h["shares"])
            sell_shares = (sell_shares // 100) * 100
            if sell_shares <= 0:
                return None

            entry_price = h["entry_price"]
            # 成本模型：滑点（卖方成交价偏低）+ 佣金 + 印花税
            cost = apply_trading_costs("sell", price, sell_shares)
            fill_price = cost["fill_price"]
            amount = cost["amount"]            # 滑点后成交额
            net_cash = cost["net_cash"]        # 实际现金净流入（扣佣金+印花税）
            # 已实现盈亏 = 净流入 - 成本基础（entry_price 已含买入滑点）
            cost_basis = entry_price * sell_shares
            pnl = net_cash - cost_basis if cost_basis else 0
            pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0
            hold_days = (datetime.strptime(date, "%Y-%m-%d") -
                         datetime.strptime(h["entry_date"], "%Y-%m-%d")).days

            now = datetime.now().isoformat()
            reason_str = reason + (" | " + "; ".join(reasons[:2]) if reasons else "")

            # 加现金（净流入，已扣佣金+印花税）
            conn.execute(
                "UPDATE paper_account SET cash=cash+?, updated_at=? WHERE id=1",
                (net_cash, now),
            )
            # 更新/删除持仓
            remaining = h["shares"] - sell_shares
            if remaining <= 0:
                conn.execute(
                    "DELETE FROM paper_holdings WHERE account_id=1 AND symbol=?", (symbol,)
                )
            else:
                conn.execute(
                    "UPDATE paper_holdings SET shares=?, cost=? WHERE account_id=1 AND symbol=?",
                    (remaining, h["cost"] * remaining / h["shares"], symbol),
                )
            # 写交易记录
            conn.execute(
                "INSERT INTO paper_trades "
                "(account_id, symbol, name, side, price, shares, amount, date, reason, "
                "pnl, pnl_pct, hold_days, entry_price, created_at) "
                "VALUES (1, ?, ?, 'sell', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, h["name"], fill_price, sell_shares, amount, date, reason_str,
                 round(pnl, 2), round(pnl_pct, 2), hold_days, entry_price, now),
            )
            conn.commit()

            logger.info(
                f"模拟卖出：{symbol} {sell_shares}股@{fill_price:.2f}={amount:.0f}元(费{amount-net_cash:.0f}) "
                f"盈亏{pnl:+.0f}({pnl_pct:+.1f}%) 持{hold_days}天"
            )
            return {
                "symbol": symbol, "name": h["name"],
                "price": price, "shares": sell_shares, "amount": round(amount, 2),
                "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
                "hold_days": hold_days, "reason": reason,
                "partial": partial,
            }

    def _update_stop(self, symbol: str, new_stop: float) -> None:
        """更新止损价。"""
        with self._conn() as conn:
            conn.execute(
                "UPDATE paper_holdings SET stop_loss=? WHERE account_id=1 AND symbol=?",
                (new_stop, symbol),
            )
            conn.commit()
    def _generate_enhanced_sells(self, base_signals: list[dict], today: str) -> list[dict]:
        """生成增强卖出信号：分批止盈 + 移动止盈 + 到期强平。

        遍历当前持仓，叠加在基础信号之上。
        """
        holdings = self.get_holdings()
        sig_map = {s["symbol"]: s for s in base_signals}
        result = list(base_signals)
        sell_actions = {"止损清仓", "止盈", "减仓/清仓"}

        for h in holdings:
            sym = h["symbol"]
            entry = h["entry_price"]
            current = self._get_close_price(sym, today)
            if not current or current <= 0:
                continue
            pnl_pct = (current / entry - 1) * 100 if entry > 0 else 0
            hold_days = (datetime.strptime(today, "%Y-%m-%d") -
                         datetime.strptime(h["entry_date"], "%Y-%m-%d")).days

            base = sig_map.get(sym, {"symbol": sym, "action": "持有",
                                     "reasons": [], "shares": h["shares"]})

            # 到期强平（>MAX_HOLD_DAYS，除非已有止损信号）
            if hold_days >= MAX_HOLD_DAYS and base.get("action") == "持有":
                result.append({
                    "symbol": sym, "action": "止盈",
                    "reasons": [f"持仓{hold_days}天到期强平"],
                    "shares": h["shares"],
                })
                continue

            # 分批止盈：浮盈>PARTIAL_TP_PCT 且未减过仓
            if pnl_pct >= FULL_TP_PCT:
                if base.get("action") not in sell_actions:
                    result.append({
                        "symbol": sym, "action": "止盈",
                        "reasons": [f"浮盈{pnl_pct:.1f}%≥{FULL_TP_PCT}%·全部止盈"],
                        "shares": h["shares"],
                    })
            elif pnl_pct >= PARTIAL_TP_PCT:
                if base.get("action") not in sell_actions and base.get("action") not in {"减仓半仓"}:
                    result.append({
                        "symbol": sym, "action": "减仓半仓",
                        "reasons": [f"浮盈{pnl_pct:.1f}%≥{PARTIAL_TP_PCT}%·减仓一半"],
                        "shares": h["shares"] // 2,
                    })

            # 移动止盈：浮盈>TRAILING_START_PCT，止损上移
            if pnl_pct >= TRAILING_START_PCT:
                new_stop = current * (1 - TRAILING_PULLBACK / 100)
                old_stop = h.get("stop_loss", 0)
                if new_stop > old_stop:
                    base["action"] = "移动止损"
                    base["new_stop"] = round(new_stop, 2)
                    if base not in result:
                        result.append(base)

        return result


    # ════════════════════════════════════════
    # 绩效统计
    # ════════════════════════════════════════

    def get_performance(self) -> PaperPerformance:
        """计算当前绩效快照。"""
        account = self.get_account()
        holdings = self.get_holdings()
        cash = account.get("cash", 0)
        initial = account.get("initial_capital", 100000)

        # 持仓市值（批量获取收盘价，避免逐只串行东财请求）
        sym_list = [h["symbol"] for h in holdings]
        close_prices = self._get_close_prices_batch(sym_list)
        market_value = 0
        for h in holdings:
            price = close_prices.get(h["symbol"])
            if price:
                market_value += price * h["shares"]

        total_assets = cash + market_value
        total_return = (total_assets / initial - 1) * 100 if initial else 0

        # 已完成交易统计
        with self._conn() as conn:
            trades = conn.execute(
                "SELECT * FROM paper_trades WHERE account_id=1 AND side='sell' "
                "ORDER BY created_at"
            ).fetchall()

        perf = PaperPerformance(
            initial_capital=initial,
            cash=round(cash, 2),
            market_value=round(market_value, 2),
            total_assets=round(total_assets, 2),
            total_return_pct=round(total_return, 2),
            holding_count=len(holdings),
            total_trades=len(trades),
        )

        if trades:
            pnls = [t["pnl"] for t in trades]
            pnl_pcts = [t["pnl_pct"] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            perf.total_pnl = round(sum(pnls), 2)
            perf.win_trades = len(wins)
            perf.loss_trades = len(losses)
            perf.win_rate = round(len(wins) / len(trades) * 100, 1) if trades else 0
            perf.avg_win = round(sum(p for p in pnl_pcts if p > 0) / len(wins), 2) if wins else 0
            perf.avg_loss = round(sum(p for p in pnl_pcts if p < 0) / len(losses), 2) if losses else 0
            perf.profit_factor = round(
                sum(wins) / abs(sum(losses)), 2
            ) if losses and sum(losses) != 0 else (999.0 if wins else 0)
            perf.best_trade_pct = max(pnl_pcts) if pnl_pcts else 0
            perf.worst_trade_pct = min(pnl_pcts) if pnl_pcts else 0
            perf.avg_hold_days = round(
                sum(t["hold_days"] for t in trades) / len(trades), 1
            ) if trades else 0

            # 最大回撤（按交易时序累计资产）
            equity = [initial]
            for t in trades:
                equity.append(equity[-1] + t["pnl"])
            peak = equity[0]
            max_dd = 0
            for e in equity:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak * 100 if peak else 0
                if dd > max_dd:
                    max_dd = dd
            perf.max_drawdown_pct = round(max_dd, 2)

        # ── 连续亏损统计 ──
        consec = 0
        for t in reversed(trades):
            if t["pnl"] < 0:
                consec += 1
            else:
                break
        perf.consec_loss = consec
        if consec >= MAX_CONSEC_LOSS:
            perf.risk_circuit = f"连亏{consec}笔·降仓"

        # ── 专业指标：基于日度NAV ──
        with self._conn() as conn:
            nav_rows = conn.execute(
                "SELECT daily_return, cum_return, benchmark_return, benchmark_cum, date "
                "FROM paper_nav ORDER BY date"
            ).fetchall()

        if len(nav_rows) >= 2:
            import math
            daily_rets = [r["daily_return"] / 100 for r in nav_rows]
            bench_rets = [r["benchmark_return"] / 100 for r in nav_rows]
            trading_days = len(nav_rows)

            mean_ret = sum(daily_rets) / trading_days
            # 夏普比率（无风险利率取2%/年 ≈ 0.008%/日）
            rf_daily = 0.02 / 252
            excess = [r - rf_daily for r in daily_rets]
            std_ret = (sum((r - mean_ret) ** 2 for r in daily_rets) / (trading_days - 1)) ** 0.5
            if std_ret > 0:
                perf.sharpe_ratio = round(sum(excess) / trading_days / std_ret * math.sqrt(252), 2)

            # 索提诺比率（仅用下行标准差）
            downside = [min(r, 0) for r in daily_rets]
            ds_std = (sum(d ** 2 for d in downside) / trading_days) ** 0.5
            if ds_std > 0:
                perf.sortino_ratio = round(mean_ret / ds_std * math.sqrt(252), 2)

            # 年化收益率
            total_cum = nav_rows[-1]["cum_return"] / 100
            years = trading_days / 252
            if years > 0 and (1 + total_cum) > 0:
                perf.annual_return = round(((1 + total_cum) ** (1 / years) - 1) * 100, 2)

            # 卡玛比率
            if perf.max_drawdown_pct > 0:
                perf.calmar_ratio = round(abs(perf.annual_return / perf.max_drawdown_pct), 2)

            # Alpha（超额收益 vs 沪深300）
            bench_cum = nav_rows[-1]["benchmark_cum"] if nav_rows else 0
            perf.benchmark_return = round(bench_cum, 2)
            perf.alpha = round(nav_rows[-1]["cum_return"] - bench_cum, 2)

            # 最大回撤熔断检查
            if perf.max_drawdown_pct >= MAX_DRAWDOWN_CIRCUIT * 100:
                perf.risk_circuit = f"回撤{perf.max_drawdown_pct:.1f}%·熔断"

        return perf

    # ════════════════════════════════════════
    # 辅助方法
    # ════════════════════════════════════════

    def _check_tradable(self, symbol: str, date_str: str | None = None) -> tuple[bool, str]:
        """检查个股当日是否可买入（停牌/涨停封板买不进）。

        Args:
            symbol: 股票代码
            date_str: 日期（空=最新K线日）

        Returns:
            (tradable, reason) — True=可买入空原因；False=不可买入+原因
        """
        try:
            with self._conn() as conn:
                if date_str:
                    row = conn.execute(
                        "SELECT tradestatus, pct_chg FROM stock_daily "
                        "WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
                        (symbol, date_str),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT tradestatus, pct_chg FROM stock_daily "
                        "WHERE symbol=? ORDER BY date DESC LIMIT 1", (symbol,)
                    ).fetchone()
            if not row:
                return False, "当日无K线数据，保守视为不可交易"
            tradestatus = row["tradestatus"] if "tradestatus" in row.keys() else 1
            if tradestatus == 0:
                return False, "今日停牌"
            pct_chg = row["pct_chg"] if "pct_chg" in row.keys() and row["pct_chg"] is not None else 0
            # 板块涨停阈值（复用 decision._filter_limit_up 逻辑）
            threshold = 28.5 if symbol.startswith(("8", "4", "92")) else (
                19.0 if symbol.startswith(("300", "301", "688", "689")) else 9.5)
            if pct_chg >= threshold:
                return False, f"涨停封板({pct_chg:+.1f}%)买不进"
            return True, ""
        except Exception as e:
            logger.debug(f"可交易性检查失败 {symbol}：{e!r}")
            return True, ""  # 查询失败不阻断买入（保守不假阴性）

    def _calc_entry_atr_stop(self, symbol: str, date_str: str, entry_price: float) -> float:
        """计算入场 ATR 止损价（20日 True Range，2.5×ATR，封顶 8%~15%）。

        从 stock_daily 取该股 ≤ date_str 的最近 25 根 K 线，委托共享 calc_atr_stop，
        口径与回测 _calc_atr_stop 完全一致。数据不足/异常时回退到 12% 止损，
        不阻断买入流程。
        """
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT date, high, low, close FROM stock_daily "
                    "WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 25",
                    (symbol, date_str),
                ).fetchall()
            if not rows:
                return round(entry_price * (1 - DEFAULT_FALLBACK_PCT), 3)
            df = pd.DataFrame([dict(r) for r in rows]).sort_values("date")
            return round(calc_atr_stop(df, entry_price, as_of_date=date_str), 3)
        except Exception as e:
            logger.debug(f"ATR 入场止损计算失败 {symbol}：{e!r}")
            return round(entry_price * (1 - DEFAULT_FALLBACK_PCT), 3)

    def _get_close_price(self, symbol: str, date_str: str | None = None) -> float | None:
        """获取收盘价：优先日K最新价，日K滞后时fallback东财实时价。

        paper_holdings的entry_price是东财实时价（真实价），因此卖出时也
        需要用真实价而非后复权价。当日K数据未更新到目标日期时，从东财
        push2接口获取实时价格，避免卖出价=买入价导致pnl恒为0。
        """
        from datetime import date as _date

        # Step 1: 从DB获取最新日K日期和收盘价
        with self._conn() as conn:
            if date_str:
                row = conn.execute(
                    "SELECT date, close FROM stock_daily WHERE symbol=? AND date<=? "
                    "ORDER BY date DESC LIMIT 1", (symbol, date_str)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT date, close FROM stock_daily WHERE symbol=? "
                    "ORDER BY date DESC LIMIT 1", (symbol,)
                ).fetchone()

        target = date_str or _date.today().strftime("%Y-%m-%d")

        if row:
            db_date = row["date"]
            # 日K数据已是目标日期 → 直接用
            if db_date >= target:
                return round(row["close"], 3)
            # 日K滞后 → fallback东财实时价
            rt = self._fetch_realtime_price(symbol)
            if rt and rt > 0:
                return round(rt, 3)
            # 东财也失败 → 用DB最近的（至少不为None）
            return round(row["close"], 3)
        return None

    def _fetch_realtime_price(self, symbol: str) -> float | None:
        """从东财push2接口获取单只股票实时价格。"""
        import requests
        market = "1" if symbol.startswith(("6", "9")) else "0"
        secid = f"{market}.{symbol}"
        try:
            resp = requests.get(
                "http://push2delay.eastmoney.com/api/qt/stock/get",
                params={"secid": secid, "fields": "f43"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5,
            )
            data = resp.json().get("data", {})
            price = data.get("f43", 0)
            if price and price > 0:
                return float(price) / 100
        except Exception:
            pass
        return None

    def _fetch_realtime_prices_batch(self, symbols: list[str]) -> dict[str, float]:
        """并发获取多只股票实时价格（绕过东财 0.4s 限流间隔的串行延迟）。

        东财 stock/get 单只接口返回正确价格（f43/100），但限流间隔 0.4s
        导致串行 N 只需 N×0.4s。用线程池并发请求，总耗时≈单次请求时间。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not symbols:
            return {}
        prices: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
            futures = {
                pool.submit(self._fetch_realtime_price, sym): sym
                for sym in symbols
            }
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    p = fut.result()
                    if p and p > 0:
                        prices[sym] = p
                except Exception:
                    pass
        return prices

    def _get_close_prices_batch(self, symbols: list[str]) -> dict[str, float]:
        """批量获取多只股票收盘价（DB 优先，滞后时批量东财实时）。

        性能：N 只持仓 = 1 次 DB 查询 + 最多 1 次东财批量请求，
        替代原来 N 次串行 _get_close_price（每只含独立东财请求）。
        """
        from datetime import date as _date

        if not symbols:
            return {}
        today = _date.today().strftime("%Y-%m-%d")
        prices: dict[str, float] = {}
        stale_syms: list[str] = []

        # 一次 DB 查询取全部持仓的最新日K
        placeholders = ",".join("?" * len(symbols))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT symbol, date, close FROM stock_daily "
                f"WHERE symbol IN ({placeholders}) "
                f"AND date IN (SELECT MAX(date) FROM stock_daily WHERE symbol IN ({placeholders}))",
                (*symbols, *symbols),
            ).fetchall()

        for r in rows:
            sym, d, close = r["symbol"], r["date"], r["close"]
            if d >= today:
                prices[sym] = round(close, 3)
            else:
                stale_syms.append(sym)

        # 日K滞后的持仓 → 批量东财实时（1次请求）
        if stale_syms:
            rt = self._fetch_realtime_prices_batch(stale_syms)
            for sym in stale_syms:
                if sym in rt:
                    prices[sym] = rt[sym]
                elif sym in dict((r["symbol"], r["close"]) for r in rows if r["symbol"] == sym):
                    # 东财失败 → fallback DB 最近
                    db_close = next((r["close"] for r in rows if r["symbol"] == sym), None)
                    if db_close:
                        prices[sym] = round(db_close, 3)
        return prices

    def _calc_holding_value(self, holdings: list[dict]) -> float:
        """计算持仓总市值。"""
        total = 0
        for h in holdings:
            price = self._get_close_price(h["symbol"])
            if price:
                total += price * h["shares"]
        return total

    def reset(self) -> None:
        """重置模拟盘（清空所有持仓和交易，恢复10万本金）。"""
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute("DELETE FROM paper_holdings WHERE account_id=1")
            conn.execute("DELETE FROM paper_trades WHERE account_id=1")
            conn.execute(
                "UPDATE paper_account SET cash=initial_capital, updated_at=? WHERE id=1",
                (now,),
            )
            conn.commit()
        logger.info("模拟盘已重置：清空持仓+交易，恢复本金")

    # ════════════════════════════════════════
    # 风险预算仓位管理
    # ════════════════════════════════════════

    def _calc_risk_position(self, score: float, symbol: str, stop_loss: float,
                           entry_price_hint: float | None, position_scale: float,
                           consec_loss: int = 0) -> float:
        """风险预算仓位计算。

        方法论：
          1. 信号置信度：score^SCORE_POWER 归一化 → 基础仓位比例
          2. 波动率调整：有止损位时用固定风险金额/止损距离反推仓位
          3. 连亏缩放：连续亏损时降仓
          4. 市场状态缩放

        Returns:
            position_pct: 目标仓位占总资产比例（0~MAX_POSITION_PCT）
        """
        # 信号置信度 → 基础仓位（评分50=5%，80=12%，100=15%）
        conf = max(0, min(score, 100)) / 100
        base_pct = MAX_POSITION_PCT * (conf ** SCORE_POWER)

        # 波动率调整：有止损位时按风险预算
        if stop_loss > 0 and entry_price_hint and entry_price_hint > 0:
            stop_distance = abs(entry_price_hint - stop_loss) / entry_price_hint
            if stop_distance > 0.01:  # 止损距离>1%才用
                risk_pct = min(BASE_RISK_PCT + conf * (MAX_RISK_PCT - BASE_RISK_PCT), MAX_RISK_PCT)
                risk_based_pct = risk_pct / stop_distance
                base_pct = min(base_pct, risk_based_pct, MAX_POSITION_PCT)

        # 连亏缩放
        if consec_loss >= MAX_CONSEC_LOSS:
            base_pct *= CONSEC_LOSS_SCALE

        # 市场状态缩放
        base_pct *= position_scale

        return round(min(base_pct, MAX_POSITION_PCT), 4)

    def _check_risk_circuit(self, initial: float) -> dict:
        """风控熔断检查。

        Returns:
            {halt: bool, reason: str, consec_loss: int}
        """
        # 连续亏损
        with self._conn() as conn:
            recent_sells = conn.execute(
                "SELECT pnl FROM paper_trades WHERE account_id=1 AND side='sell' "
                "ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
        consec = 0
        for s in recent_sells:
            if s["pnl"] < 0:
                consec += 1
            else:
                break

        # 最大回撤检查（基于NAV）
        with self._conn() as conn:
            nav_rows = conn.execute(
                "SELECT total_assets FROM paper_nav ORDER BY date"
            ).fetchall()
        if nav_rows:
            assets = [r["total_assets"] for r in nav_rows]
            current = assets[-1]
            peak = max(assets)
            if peak > 0:
                dd = (peak - current) / peak
                if dd >= MAX_DRAWDOWN_CIRCUIT:
                    return {"halt": True, "reason": f"最大回撤{dd*100:.1f}%≥{MAX_DRAWDOWN_CIRCUIT*100:.0f}%·熔断",
                            "consec_loss": consec}

        # 连续亏损≥5笔熔断
        if consec >= MAX_CONSEC_LOSS + 2:
            return {"halt": True, "reason": f"连续亏损{consec}笔·熔断",
                    "consec_loss": consec}

        return {"halt": False, "reason": "", "consec_loss": consec}

    # ════════════════════════════════════════
    # 日度净值记录
    # ════════════════════════════════════════

    def record_daily_nav(self) -> dict:
        """记录当日净值快照（每日盘后调用一次）。

        记录总资产、现金、持仓市值、收益率，以及沪深300基准对比。
        """
        account = self.get_account()
        holdings = self.get_holdings()
        cash = account.get("cash", 0)
        initial = account.get("initial_capital", 100000)
        today = datetime.now().strftime("%Y-%m-%d")

        market_value = 0
        for h in holdings:
            price = self._get_close_price(h["symbol"], today)
            if price:
                market_value += price * h["shares"]

        total_assets = cash + market_value
        cum_return = (total_assets / initial - 1) * 100 if initial else 0

        # 昨日资产 → 今日收益率
        with self._conn() as conn:
            prev = conn.execute(
                "SELECT total_assets FROM paper_nav ORDER BY date DESC LIMIT 1"
            ).fetchone()
        prev_assets = prev["total_assets"] if prev else initial
        daily_return = ((total_assets / prev_assets) - 1) * 100 if prev_assets else 0

        # 基准
        bench_daily, bench_cum = self._get_benchmark_return(today)

        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO paper_nav "
                "(date, total_assets, cash, market_value, daily_return, cum_return, "
                "benchmark_return, benchmark_cum, holding_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (today, round(total_assets, 2), round(cash, 2), round(market_value, 2),
                 round(daily_return, 4), round(cum_return, 4),
                 round(bench_daily, 4), round(bench_cum, 4), len(holdings))
            )
            conn.commit()

        logger.info(f"NAV记录 {today}: 资产{total_assets:,.0f} 日收益{daily_return:+.2f}% 基准{bench_daily:+.2f}%")
        return {"date": today, "total_assets": total_assets, "daily_return": daily_return,
                "benchmark_daily": bench_daily}

    def get_nav_history(self, days: int = 90) -> list[dict]:
        """获取最近N天净值曲线。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT date, total_assets, daily_return, cum_return, "
                "benchmark_return, benchmark_cum, holding_count "
                "FROM paper_nav ORDER BY date DESC LIMIT ?", (days,)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def _get_benchmark_return(self, today: str) -> tuple[float, float]:
        """获取沪深300当日收益率和累计收益率（相对模拟盘起始日）。

        Returns:
            (daily_return_pct, cum_return_pct_since_paper_start)
        """
        try:
            import akshare as ak
            # 取最近30天沪深300日K
            df = ak.stock_zh_index_daily(symbol="sh000300")
            if df is None or df.empty:
                return 0.0, 0.0
            df = df.sort_values("date")
            df["date"] = df["date"].astype(str)

            # 当日涨跌幅
            today_row = df[df["date"] == today]
            if today_row.empty:
                # 取最近一条
                today_row = df.tail(1)
            if len(df) >= 2:
                prev_close = df.iloc[-2]["close"]
                today_close = today_row.iloc[0]["close"]
                daily = (today_close / prev_close - 1) * 100 if prev_close else 0
            else:
                daily = 0

            # 累计：从模拟盘创建日到今天的涨幅
            account = self.get_account()
            start_date = account.get("created_at", "")[:10]
            start_row = df[df["date"] >= start_date]
            if not start_row.empty:
                start_close = start_row.iloc[0]["close"]
                cum = (today_row.iloc[0]["close"] / start_close - 1) * 100 if start_close else 0
            else:
                cum = 0

            return round(daily, 2), round(cum, 2)
        except Exception as e:
            logger.debug(f"基准数据获取失败: {e}")
            return 0.0, 0.0
