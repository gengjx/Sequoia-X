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
import pandas as pd
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
    def _get_quote_and_ma(self, symbol: str, realtime_price: float | None = None) -> dict:
        """获取最新价 + MA10/MA20 + ATR（用本地K线，离线可用）。

        Args:
            realtime_price: 盘中实时真实价。传入时估算实时MA：
                MA10 ≈ (前9日后复权close和 + 实时价/复权系数) / 10
                让盘中价格波动即时反映到MA防守线上。
        """
        df = self.engine.get_ohlcv(symbol)
        if df is None or len(df) < 5:
            return {}
        df = df.sort_values("date")
        close = df["close"].astype(float)
        price = round(float(close.iloc[-1]), 3)
        # 日内最低价（止损按最坏情形触发，防盘中穿仓后收盘反弹漏检）
        low = round(float(df["low"].astype(float).iloc[-1]), 3) if "low" in df.columns else price
        # 实时MA估算：把今日实时价替换最后一根K线的收盘
        ma_source = close.copy()
        if realtime_price and realtime_price > 0 and len(close) >= 2:
            # 用复权系数把实时真实价转为后复权口径
            ratio = price / realtime_price if price > 0 else 1.0
            if 0.01 < ratio < 100.0:
                ma_source.iloc[-1] = realtime_price * ratio
        ma10 = round(float(ma_source.rolling(10).mean().iloc[-1]), 3) if len(ma_source) >= 10 else 0
        ma20 = round(float(ma_source.rolling(20).mean().iloc[-1]), 3) if len(ma_source) >= 20 else 0
        # ATR（14）
        atr = 0.0
        if len(df) >= 15:
            hl = df["high"].astype(float) - df["low"].astype(float)
            atr = round(float(hl.rolling(14).mean().iloc[-1]), 3)
        last_date = str(df["date"].iloc[-1])
        # 换手率（供量能验证规则）
        turn = 0.0
        turn_ma20 = 0.0
        if "turn" in df.columns:
            t = df["turn"].astype(float)
            turn = round(float(t.iloc[-1]), 3) if pd.notna(t.iloc[-1]) else 0.0
            turn_ma20 = round(float(t.rolling(20).mean().iloc[-1]), 3) if len(t) >= 20 and pd.notna(t.rolling(20).mean().iloc[-1]) else 0.0
        # pct_chg（供相对强度规则）
        pct_chg = 0.0
        if "pct_chg" in df.columns:
            pct_chg = float(df["pct_chg"].astype(float).iloc[-1]) if pd.notna(df["pct_chg"].astype(float).iloc[-1]) else 0.0
        return {"price": price, "ma10": ma10, "ma20": ma20, "atr": atr, "kline_date": last_date,
                "turn": turn, "turn_ma20": turn_ma20, "pct_chg": pct_chg, "low": low}

    @staticmethod
    def _fetch_quote(symbol: str) -> tuple[float | None, float | None]:
        """复用 StockAnalyzer 的行情接口，返回 (最新价, 昨收)。"""
        from sequoia_x.analysis.stock_analysis import StockAnalyzer
        return StockAnalyzer._fetch_price_quote(symbol)

    def _fetch_real_price(symbol: str) -> float | None:
        """从东财延迟行情获取真实（不复权）最新价。"""
        info = PositionTracker._fetch_quote(symbol)
        return info[0] if info else None

    # ------------------------------------------------------------------
    # 核心：单只持仓信号扫描
    # ------------------------------------------------------------------
    def scan_one(self, h: dict, realtime_price: float | None = None) -> HoldingSignal:
        """对单只持仓执行移动止损规则，返回信号。

        Args:
            realtime_price: 盘中实时真实价（来自东财快照），None则用日K收盘价。
        """
        sig = HoldingSignal(
            id=h["id"], symbol=h["symbol"], name=h.get("name", ""),
            entry_price=h["entry_price"], shares=h["shares"],
            entry_date=h["entry_date"], stop_loss=h.get("stop_loss", 0),
            initial_stop=h.get("initial_stop", 0) or h.get("stop_loss", 0),
            target=h.get("target", 0), grade=h.get("grade", ""),
            cost=h.get("cost", 0),
        )
        q = self._get_quote_and_ma(h["symbol"], realtime_price=realtime_price)
        if not q:
            sig.reasons.append("无行情数据")
            sig.action = "数据缺失"
            sig.signal_level = "warn"
            return sig

        # 前复权数据：DB收盘价≈真实交易价，entry/stop/target 同为真实价，无需复权转换
        latest_price, prev_close = self._fetch_quote(h["symbol"])
        price_source = "db"
        if latest_price and latest_price > 0:
            price = round(latest_price, 3)  # 展示用实时价
            price_source = "real"
        else:
            price = q["price"]  # fallback 用DB收盘价
        ma10 = q["ma10"] or 0
        ma20 = q["ma20"] or 0
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
        # 用日内最低价判定（防盘中穿仓后收盘反弹漏检，如300821止损24.97却持有到21.92）
        day_low = q.get("low", price)
        if h["stop_loss"] > 0 and day_low <= h["stop_loss"]:
            sig.action = "止损清仓"
            sig.signal_level = "danger"
            # 执行价取止损线（保守：止损单在该价位成交，而非可能反弹的收盘价）
            sig.price = h["stop_loss"]
            sig.pnl = round((h["stop_loss"] - entry) * h["shares"], 0)
            sig.pnl_pct = round((h["stop_loss"] / entry - 1) * 100, 1)
            sig.reasons.append(f"日内低点{day_low} ≤ 止损线{h['stop_loss']}，触发硬止损@{h['stop_loss']}")
            return sig

        # ── 规则2：止盈 ──
        if h["target"] > 0 and price >= h["target"]:
            sig.action = "止盈"
            sig.signal_level = "success"
            sig.reasons.append(f"现价{price} ≥ 目标{h['target']}，到达止盈位")
            return sig

        # ── 规则3：MA防守（跌破MA20清仓）──
        # 盈利中(r_mult≥1)：优先移动止损锁利，MA20破位降级为减仓，不直接清仓
        # 浮亏或微利：MA20破位即清仓（趋势反转风控）
        if ma20 > 0 and price < ma20:
            if r_mult >= 1.0:
                # 盈利单回踩：仅减仓，继续评估移动止损（不 return）
                sig.action = "减仓半仓"
                sig.signal_level = "warn"
                sig.reasons.append(f"盈利中回踩MA20({ma20})，减仓防守待移动止损确认")
            else:
                sig.action = "减仓/清仓"
                sig.signal_level = "danger"
                sig.reasons.append(f"现价{price}跌破MA20({ma20})，趋势破位")
                return sig

        # ── 规则4：MA减仓（跌破MA10半仓）──
        elif ma10 > 0 and price < ma10:
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

        # ── 规则5b：ATR 自适应收紧（P5，只收紧不放宽）──
        # 用当前 14 日 ATR 重算理论止损 entry*(1-clamp(2.5*atr/entry,8%,15%))。
        # 波动率收敛时上移止损锁利/降险；波动率放大时维持原止损（不让亏损空间
        # 扩大）。与 R 倍数移动止损叠加取更高值，遵循移动止损单向原则。
        if atr > 0 and entry > 0:
            atr_stop = round(entry * (1 - max(0.08, min(0.15, 2.5 * atr / entry))), 2)
            if atr_stop > new_stop:
                new_stop = atr_stop
                sig.reasons.append(
                    f"ATR自适应收紧：2.5×ATR={2.5 * atr:.2f}，止损上移至{atr_stop}"
                )

        sig.new_stop = round(new_stop, 2)
        if new_stop > h["stop_loss"]:
            sig.action = "移动止损" if sig.action == "持有" else sig.action
            sig.signal_level = "success" if sig.signal_level == "info" else sig.signal_level

        # ════════ 第二层出场规则（立体风控，数据驱动）════════
        symbol = h["symbol"]
        # 仅在硬止损/止盈未触发时执行（不覆盖"持有"和"移动止损"的乐观判断）

        # ── 规则6：逻辑证伪（入场策略不再命中）──
        hit_strats = (h.get("hit_strategies") or "").strip()
        if hit_strats:
            still_active = self._check_thesis(symbol, hit_strats)
            if not still_active:
                if sig.action == "持有":
                    sig.action = "逻辑减弱"
                    sig.signal_level = "warn"
                sig.reasons.append(f"入场策略[{hit_strats}]今日不再命中，买入逻辑减弱")

        # ── 规则7：相对强度（近5日跑输大盘3%+）──
        rs = self._check_relative_strength(symbol)
        if rs is not None and rs <= -3.0:
            if sig.action == "持有":
                sig.action = "相对走弱"
                sig.signal_level = "warn"
            sig.reasons.append(f"近5日跑输大盘{abs(rs):.1f}%，相对强度下降")

        # ── 规则8：量能异动（高位放量滞涨/疑似出货）──
        if q.get("turn", 0) > 0 and q.get("turn_ma20", 0) > 0:
            turn_ratio = q["turn"] / q["turn_ma20"] if q["turn_ma20"] else 0
            if turn_ratio >= 2.0 and r_mult >= 0.5 and q.get("pct_chg", 0) < 1.0:
                if sig.action == "持有":
                    sig.action = "放量滞涨"
                    sig.signal_level = "warn"
                sig.reasons.append(
                    f"换手{q['turn']:.1f}%是20日均{q['turn_ma20']:.1f}%的{turn_ratio:.1f}倍，"
                    f"但价格仅涨{q.get('pct_chg',0):.1f}%，疑似高位出货"
                )

        # ── 规则9：时间止损（持仓低效）──
        try:
            from datetime import datetime as _dt
            entry_dt = _dt.strptime(h["entry_date"], "%Y-%m-%d")
            hold_days = (_dt.now() - entry_dt).days
            if hold_days >= 20 and r_mult < 0.5 and sig.action == "持有":
                sig.action = "持仓低效"
                sig.signal_level = "warn"
                sig.reasons.append(
                    f"持仓{hold_days}天R倍数仅{r_mult}，走势迟缓占用资金，考虑换股"
                )
        except (ValueError, TypeError):
            pass

        if not sig.reasons:
            sig.reasons.append("趋势正常，继续持有")
        return sig

    def _check_thesis(self, symbol: str, hit_strategies: str) -> bool:
        """逻辑证伪检查：入场时命中的策略是否还选中该股。

        重跑轻量策略（不依赖财报/全市场排名的纯量价策略），如果全部不再命中，
        说明买入的技术逻辑已失效。

        Args:
            symbol: 股票代码
            hit_strategies: 入场时命中的策略key，逗号分隔

        Returns:
            True=至少一个策略仍命中（逻辑仍在），False=全部失效（逻辑证伪）
        """
        try:
            from sequoia_x.strategy.turtle_trade import TurtleTradeStrategy
            from sequoia_x.strategy.ma_volume import MaVolumeStrategy
            from sequoia_x.strategy.shrink_pullback import ShrinkPullbackStrategy
            from sequoia_x.strategy.rps_breakout import RpsBreakoutStrategy

            # 量价策略的快速重跑（只看该股，不做全市场排名）
            df = self.engine.get_ohlcv(symbol)
            if df is None or len(df) < 25:
                return True  # 数据不足无法判断，保守认为逻辑仍在

            df = df.sort_values("date").reset_index(drop=True)
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            volume = df["volume"].astype(float)
            turnover = df["turnover"].astype(float) if "turnover" in df.columns else volume * close

            strats = [s.strip() for s in hit_strategies.split(",") if s.strip()]

            # 逐策略检查核心条件
            for s in strats:
                if s == "turtle":
                    # 海龟：20日新高 + 成交过亿
                    if len(close) >= 21 and high.iloc[-1] >= high.iloc[-21:-1].max() and turnover.iloc[-1] > 1e8:
                        return True
                elif s == "ma_volume":
                    # 均线放量：MA5>MA20 + 量比>1.5
                    if len(close) >= 21:
                        ma5 = close.rolling(5).mean().iloc[-1]
                        ma20 = close.rolling(20).mean().iloc[-1]
                        vol_ma5 = volume.rolling(5).mean().iloc[-1]
                        if ma5 > ma20 and volume.iloc[-1] > vol_ma5 * 1.5:
                            return True
                elif s == "pullback":
                    # 缩量回踩：MA20上行 + 近3日缩量
                    if len(close) >= 21:
                        ma20 = close.rolling(20).mean()
                        if ma20.iloc[-1] > ma20.iloc[-5] and turnover.iloc[-1] < turnover.iloc[-20:].mean():
                            return True
                elif s == "rps":
                    # RPS需要全市场排名，单股无法判断，保守认为仍在
                    return True
                # 其他策略(multi_factor/dragon等)需全市场数据，保守不证伪
                elif s in ("multi_factor", "dragon", "flag", "shakeout", "limit_down", "bottom"):
                    return True
            return False  # 所有可验证的策略都不再命中
        except Exception:
            return True  # 出错保守处理

    def _check_relative_strength(self, symbol: str) -> float | None:
        """相对强度：个股近5日涨幅 vs 大盘近5日涨幅的差值。

        正值=跑赢大盘，负值=跑输大盘。
        返回None表示数据不足。
        """
        try:
            import sqlite3
            with sqlite3.connect(self.engine.db_path) as conn:
                # 个股近5日涨幅
                rows = conn.execute(
                    "SELECT date, close FROM stock_daily WHERE symbol=? ORDER BY date DESC LIMIT 6",
                    (symbol,)
                ).fetchall()
                if len(rows) < 6:
                    return None
                stock_ret = (rows[0][1] / rows[5][1] - 1) * 100 if rows[5][1] else None
                if stock_ret is None:
                    return None
                # 大盘近5日涨幅（用全市场等权平均近似）
                rows2 = conn.execute(
                    "SELECT AVG(pct_chg) FROM ("
                    "  SELECT symbol, pct_chg FROM stock_daily WHERE date IN ("
                    "    SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT 6"
                    "  ) AND pct_chg IS NOT NULL)"
                ).fetchone()
                market_ret = rows2[0] * 5 if rows2 and rows2[0] else 0  # 日均×5天近似
                return round(stock_ret - market_ret, 1)
        except Exception:
            return None

    def scan_all(self, apply_stop_move: bool = False,
                 realtime_map: dict[str, float] | None = None) -> list[HoldingSignal]:
        """扫描所有 open 持仓。

        Args:
            apply_stop_move: True 则将移动止损后的 new_stop 写回数据库
            realtime_map: {symbol: 实时真实价}，传入后用实时价估算MA（盘中盯盘模式）
        """
        holdings = self.list_holdings("open")
        signals = []
        for h in holdings:
            rt = realtime_map.get(h["symbol"]) if realtime_map else None
            sig = self.scan_one(h, realtime_price=rt)
            if apply_stop_move and sig.new_stop > h.get("stop_loss", 0):
                self.update_holding(h["id"], stop_loss=sig.new_stop)
            signals.append(sig)
        # 按信号危险等级排序：danger > warn > success > info
        order = {"danger": 0, "warn": 1, "success": 2, "info": 3}
        signals.sort(key=lambda s: (order.get(s.signal_level, 9), -s.pnl))
        return signals

    def scan_intraday(self, notifier=None) -> list[HoldingSignal]:
        """盘中实时盯盘：批量快照 + 实时MA估算 + 信号推送。

        与 scan_all 的区别：
          - 用东财批量快照拿实时价（毫秒级，不用逐只请求）
          - 实时MA估算让MA防守线随盘中价格波动
          - 仅推送新信号（同票同信号30分钟内不重复推送）

        Args:
            notifier: FeishuNotifier，传入则推送新信号

        Returns:
            所有持仓的信号列表
        """
        import time as _time
        holdings = self.list_holdings("open")
        if not holdings:
            return []

        symbols = [h["symbol"] for h in holdings]
        spot_map = self._fetch_spot_batch(symbols)

        # 构建 realtime_map
        realtime_map: dict[str, float] = {}
        for sym in symbols:
            spot = spot_map.get(sym, {})
            price = spot.get("price", 0)
            if price > 0:
                realtime_map[sym] = price

        signals = self.scan_all(apply_stop_move=True, realtime_map=realtime_map)

        # 推送新信号（节流：同票同action 30分钟内不重复）
        if notifier:
            throttle_key = f"position_throttle_{_time.strftime('%Y%m%d')}"
            if not hasattr(self, throttle_key):
                setattr(self, throttle_key, {})
            throttle: dict[tuple[str, str], float] = getattr(self, throttle_key)
            now_ts = _time.time()
            for sig in signals:
                if sig.signal_level not in ("danger", "warn", "success"):
                    continue
                if sig.action in ("持有", "数据缺失", "逻辑减弱", "相对走弱", "持仓低效"):
                    continue  # 低优先级信号盘中不推送
                key = (sig.symbol, sig.action)
                if now_ts - throttle.get(key, 0) < 1800:  # 30分钟节流
                    continue
                self._push_position_signal(notifier, sig)
                throttle[key] = now_ts

        return signals

    def _fetch_spot_batch(self, symbols: list[str]) -> dict[str, dict]:
        """批量拉实时快照（复用东财clist接口）。"""
        import requests

        if not symbols:
            return {}
        sym_set = set(symbols)
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        result: dict[str, dict] = {}
        try:
            r = requests.get(
                "https://push2delay.eastmoney.com/api/qt/clist/get",
                params={"pn": 1, "pz": 200, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                        "fs": "m:0+t:6+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2",
                        "fields": "f12,f14,f2,f3,f60"},
                headers=headers, timeout=8,
            )
            for item in r.json().get("data", {}).get("diff", []) or []:
                sym = item.get("f12", "")
                if sym in sym_set:
                    try:
                        price = float(item.get("f2", 0))
                    except (TypeError, ValueError):
                        price = 0
                    result[sym] = {"price": price, "name": item.get("f14", "")}
        except Exception as e:
            logger.warning(f"持仓快照拉取失败：{e!r}")
        return result

    def _push_position_signal(self, notifier, sig: HoldingSignal) -> None:
        """持仓信号飞书推送。"""
        emoji = {"danger": "🔴", "warn": "🟡", "success": "🟢", "info": "⚪"}.get(sig.signal_level, "⚪")
        title = f"Sequoia-X | {emoji} 持仓信号 {sig.name}({sig.symbol}) {sig.action}"
        reasons_text = "\n".join(f"▶ {r}" for r in sig.reasons)
        content = (
            f"{emoji} **{sig.name}** `{sig.symbol}`\n"
            f"▶ 现价 {sig.price:.2f}（浮盈{sig.pnl_pct:+.1f}%，{sig.r_multiple:+.1f}R）\n"
            f"▶ 操作建议：**{sig.action}**\n"
            f"{reasons_text}\n"
            f"▶ 止损线 {sig.stop_loss:.2f} → 新止损 {sig.new_stop or sig.stop_loss:.2f}"
        )
        try:
            notifier.send_text(title=title, content=content, webhook_key="position")
        except Exception as e:
            logger.warning(f"持仓信号推送失败 {sig.symbol}: {e!r}")

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
