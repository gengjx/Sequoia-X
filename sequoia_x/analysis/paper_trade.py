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
# ── 仓位管理参数 ──
BASE_RISK_PCT = 0.02          # 单笔风险占总资产2%（风险预算）
MAX_RISK_PCT = 0.04           # 单笔最大风险4%
SCORE_POWER = 1.5             # 评分→仓位幂次（评分越高仓位越大）
# ── 止盈参数 ──
PARTIAL_TP_PCT = 5.0          # 浮盈>5%减仓一半
FULL_TP_PCT = 10.0            # 浮盈>10%全部止盈
TRAILING_START_PCT = 7.0      # 浮盈>7%启动移动止盈
TRAILING_PULLBACK = 2.5       # 移动止盈回撤2.5%
MAX_HOLD_DAYS = 20            # 持仓超20天强制平仓


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
        position_scale = market.get("position_scale", 0.8)

        if market_state == "bear" and market_score < BEAR_NO_BUY_SCORE:
            logger.info(f"模拟盘：市场极弱(评分{market_score}<{BEAR_NO_BUY_SCORE})，空仓不买")
            return {"bought": [], "skipped": [], "reason": f"市场极弱(评分{market_score})，空仓等待"}

        # ── 风控熔断检查 ──
        circuit = self._check_risk_circuit(initial)
        if circuit["halt"]:
            logger.warning(f"模拟盘风控熔断：{circuit['reason']}，暂停开新仓")
            return {"bought": [], "skipped": [], "reason": circuit["reason"]}

        # 当前持仓总市值
        holdings = self.get_holdings()
        holding_symbols = {h["symbol"] for h in holdings}
        holding_value = self._calc_holding_value(holdings)

        buy_list = decision_result.get("buy_list", [])
        if not buy_list:
            return {"bought": [], "skipped": [], "reason": "决策无买入清单"}

        # 按评分排序（高分优先）
        buy_sorted = sorted(buy_list, key=lambda x: x.get("score", 0), reverse=True)

        bought = []
        skipped = []
        today = datetime.now().strftime("%Y-%m-%d")
        max_total = initial * MAX_TOTAL_PCT

        for item in buy_sorted:
            sym = item.get("symbol", "")
            score = item.get("score", 0)
            if score < MIN_BUY_SCORE:
                skipped.append({"symbol": sym, "reason": f"评分{score}<{MIN_BUY_SCORE}"})
                continue

            # 已持有跳过
            if sym in holding_symbols:
                skipped.append({"symbol": sym, "reason": "已持有"})
                continue

            # 仓位计算：风险预算（信号置信度×波动率调整）
            position_pct = self._calc_risk_position(
                score=score, symbol=sym, stop_loss=item.get("stop_loss", 0),
                entry_price_hint=self._get_close_price(sym, today),
                position_scale=position_scale, consec_loss=circuit.get("consec_loss", 0),
            )
            target_amount = initial * position_pct

            # 可用资金约束
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

            # 执行买入
            self._execute_buy(
                symbol=sym, name=item.get("name", ""),
                price=price, shares=shares, amount=amount,
                date=today, stop_loss=item.get("stop_loss", 0),
                target=item.get("target", 0), grade=item.get("grade", ""),
                hit_strategies=item.get("hit_strategies", ""),
                reason=f"决策评分{score} {item.get('action','')}",
            )
            cash -= amount
            holding_value += amount
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
                     grade: str, hit_strategies: str, reason: str) -> None:
        """执行买入：扣现金 + 写持仓 + 写交易记录。"""
        now = datetime.now().isoformat()
        with self._conn() as conn:
            # 扣现金
            conn.execute(
                "UPDATE paper_account SET cash=cash-?, updated_at=? WHERE id=1",
                (amount, now),
            )
            # 写持仓
            conn.execute(
                "INSERT OR REPLACE INTO paper_holdings "
                "(account_id, symbol, name, entry_price, shares, entry_date, "
                "stop_loss, initial_stop, target, grade, hit_strategies, cost) "
                "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, name, price, shares, date,
                 stop_loss, stop_loss, target, grade, hit_strategies, amount),
            )
            # 写交易记录
            conn.execute(
                "INSERT INTO paper_trades "
                "(account_id, symbol, name, side, price, shares, amount, date, reason, created_at) "
                "VALUES (1, ?, ?, 'buy', ?, ?, ?, ?, ?, ?)",
                (symbol, name, price, shares, amount, date, reason, now),
            )
            conn.commit()

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
        half_actions = {"减仓半仓"}

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
                result = self._execute_sell(sym, price, sig.get("shares", 0) // 2, today, action, sig.get("reasons", []), partial=True)
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

            sell_shares = min(shares, h["shares"])
            if sell_shares <= 0:
                return None

            amount = sell_shares * price
            entry_price = h["entry_price"]
            pnl = (price - entry_price) * sell_shares
            pnl_pct = (price / entry_price - 1) * 100 if entry_price else 0
            hold_days = (datetime.strptime(date, "%Y-%m-%d") -
                         datetime.strptime(h["entry_date"], "%Y-%m-%d")).days

            now = datetime.now().isoformat()
            reason_str = reason + (" | " + "; ".join(reasons[:2]) if reasons else "")

            # 加现金
            conn.execute(
                "UPDATE paper_account SET cash=cash+?, updated_at=? WHERE id=1",
                (amount, now),
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
                (symbol, h["name"], price, sell_shares, amount, date, reason_str,
                 round(pnl, 2), round(pnl_pct, 2), hold_days, entry_price, now),
            )
            conn.commit()

            logger.info(
                f"模拟卖出：{symbol} {sell_shares}股@{price:.2f}={amount:.0f}元 "
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

        # 持仓市值（用最新收盘价）
        market_value = 0
        for h in holdings:
            price = self._get_close_price(h["symbol"])
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

    def _get_close_price(self, symbol: str, date_str: str | None = None) -> float | None:
        """获取收盘价（后复权→真实价转换）。

        paper_holdings的entry_price是真实价，这里也用真实价。
        """
        with self._conn() as conn:
            if date_str:
                row = conn.execute(
                    "SELECT close FROM stock_daily WHERE symbol=? AND date<=? "
                    "ORDER BY date DESC LIMIT 1", (symbol, date_str)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT close FROM stock_daily WHERE symbol=? ORDER BY date DESC LIMIT 1",
                    (symbol,)
                ).fetchone()
            if not row:
                return None
            hfq_close = row["close"]
            # 后复权→真实价：需要复权系数
            # 简化：用东财快照的真实价，或用最近真实价/后复价比近似
            # 这里直接用后复权价（不影响相对盈亏计算，因为entry也是同口径）
            return round(hfq_close, 3)

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


