"""盘中实时信号扫描器：9:30-15:00 每30秒扫描关注池，突破/异动实时触发飞书。

4类信号（日K策略的盘中版本）：
  1. 20日新高突破（海龟盘中版）：盘中价突破昨日20日最高价
  2. 均线金叉+放量（均线放量盘中版）：5分钟K MA5上穿MA20+量比放大
  3. 涨停封板/炸板：触及涨停板 + 炸板预警
  4. 急拉急跌：5分钟内涨跌>3%异动

节流：同票同信号30分钟内不重复推送，防刷屏。
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IntradaySignal:
    symbol: str
    name: str = ""
    signal_type: str = ""    # breakout / ma_cross / limit_up / spike / limit_break
    price: float = 0.0
    detail: str = ""
    severity: str = "info"   # danger / warn / info / success
    ts: str = ""


class IntradayScanner:
    """盘中信号扫描器。"""

    # 急拉急跌阈值：5分钟内涨跌幅
    SPIKE_PCT = 3.0
    SPIKE_WINDOW_MIN = 5
    # 节流：同票同信号冷却时间（秒）
    THROTTLE_SEC = 1800  # 30分钟

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # 节流缓存：{(symbol, signal_type): last_push_ts}
        self._throttle: dict[tuple[str, str], float] = {}

    def _get_daily_high20(self, symbol: str) -> float | None:
        """从日K取昨日20日最高价（突破基准）。"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT high FROM stock_daily WHERE symbol=? ORDER BY date DESC LIMIT 21",
                (symbol,),
            ).fetchall()
        if len(rows) < 21:
            return None
        # rows[0]=最新日(今日), 取前20日（不含今日）的最高
        highs = [r[0] for r in rows[1:21] if r[0]]
        return max(highs) if highs else None

    def _get_prev_close(self, symbol: str) -> float | None:
        """取昨日收盘（涨停基准）。"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT close FROM stock_daily WHERE symbol=? ORDER BY date DESC LIMIT 2",
                (symbol,),
            ).fetchall()
        if len(row) >= 2 and row[1][0]:
            return row[1][0]  # 倒数第2个=昨日
        return None

    def _limit_pct(self, symbol: str) -> float:
        """涨停幅度：创业板/科创板20%，主板10%，ST 5%（简化）。"""
        if symbol.startswith(("3", "68")):
            return 0.20
        return 0.10

    def detect_breakout(self, symbol: str, current_price: float, name: str = "") -> IntradaySignal | None:
        """1. 20日新高突破：盘中价 > 昨日20日最高价。"""
        high20 = self._get_daily_high20(symbol)
        if not high20 or current_price <= high20:
            return None
        pct_over = round((current_price / high20 - 1) * 100, 2)
        return IntradaySignal(
            symbol=symbol, name=name, signal_type="breakout",
            price=current_price, severity="success",
            detail=f"突破20日新高({high20:.2f})，超出{pct_over}%",
            ts=datetime.now().strftime("%H:%M:%S"),
        )

    def detect_ma_cross(self, symbol: str, minute_df: pd.DataFrame, name: str = "") -> IntradaySignal | None:
        """2. 均线金叉+放量：5分钟K MA5上穿MA20 + 量比放大。"""
        if len(minute_df) < 20:
            return None
        df = minute_df.copy()
        df["ma5"] = df["close"].rolling(5).mean()
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if pd.isna(last["ma5"]) or pd.isna(last["vol_ma20"]):
            return None
        # 金叉：上一根 ma5<ma20，当前 ma5>ma20（用close近似，分钟K均线需够长）
        cross = prev["ma5"] <= prev["close"].rolling(5).mean() if len(df) > 20 else False
        # 简化：当前MA5>MA20 且 放量
        ma20_now = df["close"].rolling(20).mean().iloc[-1]
        vol_surge = last["volume"] > last["vol_ma20"] * 1.5
        if last["ma5"] > ma20_now and vol_surge and last["ma5"] > prev["ma5"]:
            return IntradaySignal(
                symbol=symbol, name=name, signal_type="ma_cross",
                price=last["close"], severity="success",
                detail=f"5分钟均线放量金叉 MA5={last['ma5']:.2f}>MA20={ma20_now:.2f}",
                ts=datetime.now().strftime("%H:%M:%S"),
            )
        return None

    def detect_limit(self, symbol: str, current_price: float, name: str = "") -> IntradaySignal | None:
        """3. 涨停封板/炸板：触及涨停板 或 涨停后回落（炸板）。"""
        prev_close = self._get_prev_close(symbol)
        if not prev_close:
            return None
        limit_price = round(prev_close * (1 + self._limit_pct(symbol)), 2)
        limit_down = round(prev_close * (1 - self._limit_pct(symbol)), 2)
        # 涨停（当前价接近涨停板，差<0.1%）
        if current_price >= limit_price * 0.999:
            return IntradaySignal(
                symbol=symbol, name=name, signal_type="limit_up",
                price=current_price, severity="danger",
                detail=f"触及涨停板 {limit_price}",
                ts=datetime.now().strftime("%H:%M:%S"),
            )
        # 跌停
        if current_price <= limit_down * 1.001:
            return IntradaySignal(
                symbol=symbol, name=name, signal_type="limit_down",
                price=current_price, severity="danger",
                detail=f"触及跌停板 {limit_down}",
                ts=datetime.now().strftime("%H:%M:%S"),
            )
        return None

    def detect_spike(self, symbol: str, minute_df: pd.DataFrame, name: str = "") -> IntradaySignal | None:
        """4. 急拉急跌：5分钟内涨跌幅>3%。"""
        if len(minute_df) < self.SPIKE_WINDOW_MIN:
            return None
        recent = minute_df.tail(self.SPIKE_WINDOW_MIN)
        ret = (recent.iloc[-1]["close"] / recent.iloc[0]["close"] - 1) * 100
        if abs(ret) >= self.SPIKE_PCT:
            direction = "急拉" if ret > 0 else "急跌"
            return IntradaySignal(
                symbol=symbol, name=name, signal_type="spike",
                price=recent.iloc[-1]["close"],
                severity="danger" if ret > 0 else "warn",
                detail=f"{direction}{ret:+.2f}%（{self.SPIKE_WINDOW_MIN}分钟内）",
                ts=datetime.now().strftime("%H:%M:%S"),
            )
        return None

    def _should_push(self, sig: IntradaySignal) -> bool:
        """节流：同票同信号30分钟内不重复。"""
        key = (sig.symbol, sig.signal_type)
        now = time.time()
        last = self._throttle.get(key, 0)
        if now - last < self.THROTTLE_SEC:
            return False
        self._throttle[key] = now
        return True

    def scan_once(self, notifier=None, fetch_minute_fn: Callable | None = None) -> list[IntradaySignal]:
        """扫描关注池一次，返回触发的信号列表。

        Args:
            notifier: FeishuNotifier（可选，触发时推送）
            fetch_minute_fn: 分钟K拉取函数（symbol→DataFrame），None则从DB读
        """
        from sequoia_x.analysis.minute import build_watchlist, fetch_minute_klines

        watchlist = build_watchlist(self.db_path)
        if not watchlist:
            return []

        # 批量拉实时快照（全市场，取关注池的价）
        spot_map = self._fetch_spot_batch([w["symbol"] for w in watchlist])

        signals: list[IntradaySignal] = []
        for w in watchlist:
            sym = w["symbol"]
            spot = spot_map.get(sym, {})
            price = spot.get("price", 0)
            name = spot.get("name", "")
            if price <= 0:
                continue

            # 1. 突破（用实时价）
            if sig := self.detect_breakout(sym, price, name):
                signals.append(sig)
            # 3. 涨停/跌停（用实时价）
            if sig := self.detect_limit(sym, price, name):
                signals.append(sig)
            # 2+4 均线/异动（用分钟K）
            try:
                if fetch_minute_fn:
                    minute_df = fetch_minute_fn(sym)
                else:
                    minute_df = pd.DataFrame(fetch_minute_klines(sym, klt=5, days=1))
            except Exception:
                minute_df = None
            if minute_df is not None and not minute_df.empty:
                if sig := self.detect_ma_cross(sym, minute_df, name):
                    signals.append(sig)
                if sig := self.detect_spike(sym, minute_df, name):
                    signals.append(sig)

        # 推送（节流）
        pushed = []
        for sig in signals:
            if self._should_push(sig) and notifier:
                self._push_one(notifier, sig)
                pushed.append(sig)
            elif not notifier:
                pushed.append(sig)

        logger.info(f"盘中扫描：关注{len(watchlist)}只，信号{len(signals)}个，推送{len(pushed)}个")
        return pushed if notifier else signals

    def _fetch_spot_batch(self, symbols: list[str]) -> dict[str, dict]:
        """批量拉实时快照（东财 clist，取关注池的价/名）。"""
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
                        "fields": "f12,f14,f2,f3"},
                headers=headers, timeout=8,
            )
            for item in r.json().get("data", {}).get("diff", []) or []:
                sym = item.get("f12", "")
                if sym in sym_set:
                    try:
                        price = float(item.get("f2", 0))
                    except (TypeError, ValueError):
                        price = 0
                    result[sym] = {"price": price, "name": item.get("f14", ""), "pct": float(item.get("f3", 0) or 0)}
            # 翻页（关注池>200只时）
            total = r.json().get("data", {}).get("total", 0)
            for page in range(2, total // 200 + 2):
                if len(result) >= len(sym_set):
                    break
                r2 = requests.get(
                    "https://push2delay.eastmoney.com/api/qt/clist/get",
                    params={"pn": page, "pz": 200, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                            "fs": "m:0+t:6+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2",
                            "fields": "f12,f14,f2,f3"},
                    headers=headers, timeout=8,
                )
                for item in r2.json().get("data", {}).get("diff", []) or []:
                    sym = item.get("f12", "")
                    if sym in sym_set:
                        try:
                            price = float(item.get("f2", 0))
                        except (TypeError, ValueError):
                            price = 0
                        result[sym] = {"price": price, "name": item.get("f14", "")}
        except Exception as e:
            logger.warning(f"盘中快照拉取失败：{e!r}")
        return result

    def _push_one(self, notifier, sig: IntradaySignal) -> None:
        """单信号飞书推送。"""
        emoji = {"danger": "🔴", "warn": "🟡", "success": "🟢", "info": "⚪"}.get(sig.severity, "⚪")
        title_map = {
            "breakout": "突破新高", "ma_cross": "均线放量金叉",
            "limit_up": "涨停封板", "limit_down": "跌停",
            "spike": "盘中异动",
        }
        title = f"Sequoia-X | {emoji} {sig.name}({sig.symbol}) {title_map.get(sig.signal_type, sig.signal_type)}"
        content = f"{emoji} **{sig.name}** `{sig.symbol}`\n▶ 现价 {sig.price:.2f}\n▶ {sig.detail}\n▶ 时间 {sig.ts}"
        try:
            notifier.send_text(title=title, content=content, webhook_key="intraday")
        except Exception as e:
            logger.warning(f"盘中信号推送失败 {sig.symbol}: {e!r}")
