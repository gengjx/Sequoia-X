"""持仓跟踪 + 移动止损引擎：把"选股工具"升级为"交易系统"。

核心能力：
  - 持仓扫描：逐只读取最新行情 + 均线，判定止损/减仓/止盈/移动止损信号
  - 移动止损规则（A股趋势跟随实战）：
      1. 硬止损：现价 ≤ stop_loss → 清仓止损
      2. 保本移：浮盈 ≥ 1R（风险空间翻倍）→ 止损上移到成本价
      3. MA防守：跌破MA10 → 减仓半仓信号；跌破MA20 → 清仓信号
      4. 止盈：现价 ≥ target → 止盈信号
      5. 移盈：浮盈扩大 → 止损线进一步上移锁利（trailing）

  R = 入场风险空间 = entry_price - initial_stop
  盈亏比 = 浮盈 / R（即赚了几个R）
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine

logger = get_logger(__name__)


@dataclass
class HoldingSignal:
    """单只持仓的扫描结果。"""
    id: int
    symbol: str
    name: str
    entry_price: float
    shares: int
    entry_date: str
    stop_loss: float        # 当前止损价
    initial_stop: float
    target: float
    grade: str
    cost: float
    # 扫描后动态计算
    price: float = 0.0          # 最新价
    pnl: float = 0.0            # 浮动盈亏（元）
    pnl_pct: float = 0.0        # 浮动盈亏%
    r_multiple: float = 0.0     # 盈亏比（赚了几个R）
    ma10: float = 0.0
    ma20: float = 0.0
    action: str = "持有"        # 持有/减仓/清仓/止损/止盈/移动止损
    new_stop: float = 0.0       # 建议新止损价（移动止损后）
    signal_level: str = "info"  # info/warn/danger/success
    reasons: list[str] = field(default_factory=list)


class PositionTracker:
    """持仓跟踪与移动止损扫描器。"""

    def __init__(self, engine: DataEngine, settings: Settings) -> None:
        self.engine = engine
        self.settings = settings
        self.db_path = settings.db_path

    # ------------------------------------------------------------------
    # 持仓 CRUD（直接操作 portfolio_holding 表）
    # ------------------------------------------------------------------
    def list_holdings(self, status: str = "open") -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM portfolio_holding WHERE status=? ORDER BY entry_date DESC, id DESC",
                (status,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_holding(self, hid: int) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute("SELECT * FROM portfolio_holding WHERE id=?", (hid,)).fetchone()
        return dict(r) if r else None

    def add_holding(
        self, symbol: str, name: str, entry_price: float, shares: int,
        entry_date: str | None = None, stop_loss: float = 0, target: float = 0,
        grade: str = "", hit_strategies: str = "", notes: str = "",
    ) -> int:
        """新增持仓。同一只股票若已有 open 持仓则更新（加仓），否则新建。"""
        entry_date = entry_date or date.today().strftime("%Y-%m-%d")
        cost = round(entry_price * shares, 2)
        initial_stop = stop_loss if stop_loss > 0 else round(entry_price * 0.93, 2)
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT id, shares, cost, stop_loss FROM portfolio_holding WHERE symbol=? AND status='open'",
                (symbol,),
            ).fetchone()
            if existing:
                # 加仓：加权平均成本，止损取更紧的
                eid, old_shares, old_cost, old_stop = existing
                total_shares = old_shares + shares
                avg_price = round((old_cost + cost) / total_shares, 3)
                new_stop = max(old_stop, stop_loss) if stop_loss > 0 else old_stop
                conn.execute(
                    "UPDATE portfolio_holding SET entry_price=?, shares=?, cost=?, stop_loss=?, "
                    "initial_stop=MIN(initial_stop,?) WHERE id=?",
                    (avg_price, total_shares, old_cost + cost, new_stop, initial_stop, eid),
                )
                conn.commit()
                logger.info(f"加仓 {symbol}：{old_shares}+{shares}={total_shares}股 成本{avg_price}")
                return eid
            cur = conn.execute(
                "INSERT INTO portfolio_holding "
                "(symbol,name,entry_price,shares,entry_date,stop_loss,initial_stop,target,"
                "grade,hit_strategies,cost,status,notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (symbol, name, entry_price, shares, entry_date, stop_loss, initial_stop,
                 target, grade, hit_strategies, cost, "open", notes),
            )
            conn.commit()
            hid = cur.lastrowid
            logger.info(f"新建持仓 {symbol} {name}：{shares}股@{entry_price} 止损{stop_loss} 目标{target}")
            return hid

    def update_holding(self, hid: int, **fields) -> bool:
        """更新持仓字段（stop_loss/target/shares/notes 等）。"""
        allowed = {"entry_price", "shares", "stop_loss", "target", "notes", "name", "entry_date"}
        sets = [f"{k}=?" for k in fields if k in allowed]
        vals = [fields[k] for k in fields if k in allowed]
        if not sets:
            return False
        if "shares" in fields and "entry_price" in fields:
            sets.append("cost=?")
            vals.append(round(fields["entry_price"] * fields["shares"], 2))
        vals.append(hid)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE portfolio_holding SET {', '.join(sets)} WHERE id=?", vals)
            conn.commit()
        return True

    def close_holding(self, hid: int, close_price: float, reason: str = "") -> bool:
        """平仓持仓（status→closed）。"""
        today = date.today().strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE portfolio_holding SET status='closed', closed_price=?, closed_date=?, "
                "close_reason=? WHERE id=?",
                (close_price, today, reason, hid),
            )
            conn.commit()
        logger.info(f"平仓持仓 id={hid} @{close_price} 原因={reason}")
        return True

    def delete_holding(self, hid: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM portfolio_holding WHERE id=?", (hid,))
            conn.commit()
        return True

    # ------------------------------------------------------------------
    # 行情获取（复用 K线 + 均线）
    # ------------------------------------------------------------------
    def _get_quote_and_ma(self, symbol: str) -> dict:
        """获取最新价 + MA10/MA20 + ATR（用本地K线，离线可用）。"""
        df = self.engine.get_ohlcv(symbol)
        if df is None or len(df) < 5:
            return {}
        df = df.sort_values("date")
        close = df["close"].astype(float)
        price = round(float(close.iloc[-1]), 3)
        ma10 = round(float(close.rolling(10).mean().iloc[-1]), 3) if len(close) >= 10 else 0
        ma20 = round(float(close.rolling(20).mean().iloc[-1]), 3) if len(close) >= 20 else 0
        # ATR（14）
        atr = 0.0
        if len(df) >= 15:
            hl = df["high"].astype(float) - df["low"].astype(float)
            atr = round(float(hl.rolling(14).mean().iloc[-1]), 3)
        last_date = str(df["date"].iloc[-1])
        return {"price": price, "ma10": ma10, "ma20": ma20, "atr": atr, "kline_date": last_date}

    @staticmethod
    def _fetch_real_price(symbol: str) -> float | None:
        """从东财延迟行情获取真实（不复权）最新价，用于把后复权MA/价格转回真实价空间。

        持仓跟踪的价格必须用真实价：用户录入的买入价/止损/目标都是真实价，
        若直接与后复权K线收盘价比较会导致信号误判（如中天科技后复权342 vs 真实50）。
        """
        import requests
        prefix = "1" if symbol.startswith(("6", "5", "9")) else "0"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        try:
            r = requests.get(
                "https://push2delay.eastmoney.com/api/qt/stock/get",
                params={"secid": f"{prefix}.{symbol}", "fields": "f43"},
                headers=headers, timeout=5,
            )
            raw = r.json().get("data", {}).get("f43")
            if raw:
                return raw / 100.0
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # 核心：单只持仓信号扫描
    # ------------------------------------------------------------------
    def scan_one(self, h: dict) -> HoldingSignal:
        """对单只持仓执行移动止损规则，返回信号。"""
        sig = HoldingSignal(
            id=h["id"], symbol=h["symbol"], name=h.get("name", ""),
            entry_price=h["entry_price"], shares=h["shares"],
            entry_date=h["entry_date"], stop_loss=h.get("stop_loss", 0),
            initial_stop=h.get("initial_stop", 0) or h.get("stop_loss", 0),
            target=h.get("target", 0), grade=h.get("grade", ""),
            cost=h.get("cost", 0),
        )
        q = self._get_quote_and_ma(h["symbol"])
        if not q:
            sig.reasons.append("无行情数据")
            sig.action = "数据缺失"
            sig.signal_level = "warn"
            return sig

        # 后复权K线价 → 真实价转换（entry/stop/target 均为真实价，须统一空间）
        hfq_price = q["price"]
        real_price = self._fetch_real_price(h["symbol"])
        price_source = "hfq"
        if real_price and hfq_price and real_price > 0:
            ratio = real_price / hfq_price
            price = round(real_price, 3)
            ma10 = round(q["ma10"] * ratio, 3) if q["ma10"] else 0
            ma20 = round(q["ma20"] * ratio, 3) if q["ma20"] else 0
            atr = round((q["atr"] or h["entry_price"] * 0.03) * ratio, 3)
            price_source = "real"
        else:
            price = hfq_price
            ma10, ma20 = q["ma10"], q["ma20"]
            atr = q["atr"] or h["entry_price"] * 0.03
        entry = h["entry_price"]
        risk = entry - sig.initial_stop
        if risk <= 0:
            risk = entry * 0.05  # 兜底
        r_mult = round((price - entry) / risk, 2)  # 盈亏比（R倍数）

        sig.price = price
        sig.price_source = price_source
        sig.pnl = round((price - entry) * h["shares"], 0)
        sig.pnl_pct = round((price / entry - 1) * 100, 1)
        sig.r_multiple = r_mult
        sig.ma10 = ma10
        sig.ma20 = ma20
        new_stop = h["stop_loss"]

        # ── 规则1：硬止损 ──
        if h["stop_loss"] > 0 and price <= h["stop_loss"]:
            sig.action = "止损清仓"
            sig.signal_level = "danger"
            sig.reasons.append(f"现价{price} ≤ 止损线{h['stop_loss']}，跌破硬止损")
            return sig

        # ── 规则2：止盈 ──
        if h["target"] > 0 and price >= h["target"]:
            sig.action = "止盈"
            sig.signal_level = "success"
            sig.reasons.append(f"现价{price} ≥ 目标{h['target']}，到达止盈位")
            return sig

        # ── 规则3：MA防守（跌破MA20清仓）──
        if ma20 > 0 and price < ma20:
            sig.action = "减仓/清仓"
            sig.signal_level = "danger"
            sig.reasons.append(f"现价{price}跌破MA20({ma20})，趋势破位")
            return sig

        # ── 规则4：MA减仓（跌破MA10半仓）──
        if ma10 > 0 and price < ma10:
            sig.action = "减仓半仓"
            sig.signal_level = "warn"
            sig.reasons.append(f"现价{price}跌破MA10({ma10})，短线走弱")

        # ── 规则5：移动止损（盈利后上移锁利）──
        # 保本移：盈利达1R，止损上移到成本价
        if r_mult >= 1.0 and new_stop < entry:
            new_stop = entry
            sig.reasons.append(f"盈利{r_mult}R≥1R，止损上移保本至{entry}")
        # 移盈：盈利达2R，止损上移到 entry+1R（锁1R利润）
        if r_mult >= 2.0:
            lock_stop = round(entry + risk, 2)
            if lock_stop > new_stop:
                new_stop = lock_stop
                sig.reasons.append(f"盈利{r_mult}R≥2R，止损上移锁利至{lock_stop}(+1R)")
        # 移盈：盈利达3R，止损上移到 entry+2R
        if r_mult >= 3.0:
            lock_stop = round(entry + 2 * risk, 2)
            if lock_stop > new_stop:
                new_stop = lock_stop
                sig.reasons.append(f"盈利{r_mult}R≥3R，止损上移至{lock_stop}(+2R)")

        sig.new_stop = round(new_stop, 2)
        if new_stop > h["stop_loss"]:
            sig.action = "移动止损" if sig.action == "持有" else sig.action
            sig.signal_level = "success" if sig.signal_level == "info" else sig.signal_level

        if not sig.reasons:
            sig.reasons.append("趋势正常，继续持有")
        return sig

    def scan_all(self, apply_stop_move: bool = False) -> list[HoldingSignal]:
        """扫描所有 open 持仓。

        Args:
            apply_stop_move: True 则将移动止损后的 new_stop 写回数据库
        """
        holdings = self.list_holdings("open")
        signals = []
        for h in holdings:
            sig = self.scan_one(h)
            if apply_stop_move and sig.new_stop > h.get("stop_loss", 0):
                self.update_holding(h["id"], stop_loss=sig.new_stop)
            signals.append(sig)
        # 按信号危险等级排序：danger > warn > success > info
        order = {"danger": 0, "warn": 1, "success": 2, "info": 3}
        signals.sort(key=lambda s: (order.get(s.signal_level, 9), -s.pnl))
        return signals

    # ------------------------------------------------------------------
    # 持仓组合摘要
    # ------------------------------------------------------------------
    def summary(self, signals: list[HoldingSignal] | None = None) -> dict:
        if signals is None:
            signals = self.scan_all()
        total_cost = sum(s.cost for s in signals)
        total_value = sum(round(s.price * s.shares, 0) for s in signals if s.price)
        total_pnl = round(total_value - total_cost, 0)
        total_pnl_pct = round(total_pnl / total_cost * 100, 1) if total_cost else 0
        danger = sum(1 for s in signals if s.signal_level == "danger")
        warn = sum(1 for s in signals if s.signal_level == "warn")
        # 个股浮亏统计
        winners = sum(1 for s in signals if s.pnl > 0)
        win_rate = round(winners / len(signals) * 100, 1) if signals else 0
        return {
            "count": len(signals),
            "total_cost": round(total_cost, 0),
            "total_value": round(total_value, 0),
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "danger_count": danger,
            "warn_count": warn,
            "win_rate": win_rate,
            "winners": winners,
        }

    # ------------------------------------------------------------------
    # 信号转 dict（供 API/前端）
    # ------------------------------------------------------------------
    @staticmethod
    def signal_to_dict(s: HoldingSignal) -> dict:
        return {
            "id": s.id, "symbol": s.symbol, "name": s.name,
            "entry_price": s.entry_price, "shares": s.shares, "entry_date": s.entry_date,
            "stop_loss": s.stop_loss, "initial_stop": s.initial_stop, "target": s.target,
            "grade": s.grade, "cost": s.cost,
            "price": s.price, "pnl": s.pnl, "pnl_pct": s.pnl_pct, "r_multiple": s.r_multiple,
            "ma10": s.ma10, "ma20": s.ma20,
            "action": s.action, "new_stop": s.new_stop,
            "signal_level": s.signal_level, "reasons": s.reasons,
            "price_source": getattr(s, "price_source", "hfq"),
        }

    def apply_signal_action(self, hid: int, action: str, price: float | None = None,
                            new_stop: float | None = None) -> bool:
        """执行信号动作（前端按钮触发）。"""
        h = self.get_holding(hid)
        if not h:
            return False
        if action in ("止损清仓", "减仓/清仓", "止盈"):
            self.close_holding(hid, price or h["entry_price"], action)
        elif action == "移动止损" and new_stop and new_stop > h.get("stop_loss", 0):
            self.update_holding(hid, stop_loss=new_stop)
        return True
