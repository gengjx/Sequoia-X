"""个股分析引擎：单只股票深度研判，输出结构化买卖决策报告。

分析四层：
  1. 技术面（趋势/均线/MACD/RSI/KDJ/布林带/ATR/量价）
  2. 相对强度（RPS 120/60/20 + 年内位置 + 板块内排名）
  3. 市场环境联动（大盘评分 + 板块强度）
  4. 策略命中汇总（9 策略逐一检测）
合成：综合评分 → 买卖建议（入场/止损/目标/仓位）+ 结构化风险提示
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

_BOARD_TABLE = "stock_board_em"
_MARKET_CAP_TABLE = "stock_market_cap"
_STOCK_BASIC_TABLE = "stock_basic"
_FINANCE_TABLE = "stock_finance"
_BAOSTOCK_LOCK = threading.Lock()
_BAOSTOCK_SESSION = {"count": 0, "alive": False}  # 引用计数，避免反复login/logout


def _baostock_acquire() -> None:
    """获取 baostock 会话（引用计数管理）。

    baostock 全局 socket 非线程安全，且 login 约1-2s。
    决策批量分析40只票时，若每只都login+logout要40次握手，
    改为首次login、末次logout，中间复用同一会话。
    """
    import baostock as bs
    with _BAOSTOCK_LOCK:
        if not _BAOSTOCK_SESSION["alive"]:
            bs.login()
            _BAOSTOCK_SESSION["alive"] = True
            _BAOSTOCK_SESSION["count"] = 1
        else:
            _BAOSTOCK_SESSION["count"] += 1


def _baostock_release() -> None:
    """释放 baostock 会话（引用计数归零才真正logout）。"""
    import baostock as bs
    with _BAOSTOCK_LOCK:
        if not _BAOSTOCK_SESSION["alive"]:
            return
        _BAOSTOCK_SESSION["count"] -= 1
        if _BAOSTOCK_SESSION["count"] <= 0:
            try:
                bs.logout()
            except Exception:
                pass
            _BAOSTOCK_SESSION["alive"] = False
            _BAOSTOCK_SESSION["count"] = 0


@dataclass
class StockReport:
    """个股分析报告（可序列化为 dict）。"""
    symbol: str
    name: str = ""
    date: str = ""
    price: float = 0.0
    technical: dict = field(default_factory=dict)
    relative_strength: dict = field(default_factory=dict)
    market_context: dict = field(default_factory=dict)
    strategy_hits: list[dict] = field(default_factory=list)
    fundamental: dict = field(default_factory=dict)
    capital: dict = field(default_factory=dict)
    recommendation: dict = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "name": self.name, "date": self.date, "price": self.price,
            "technical": self.technical, "relative_strength": self.relative_strength,
            "market_context": self.market_context, "strategy_hits": self.strategy_hits,
            "fundamental": self.fundamental, "capital": self.capital,
            "recommendation": self.recommendation, "risks": self.risks, "summary": self.summary,
        }


class StockAnalyzer:
    """个股分析器：基于本地行情库 + 全市场对比，秒级生成决策报告。"""

    def __init__(self, settings: Settings) -> None:
        self.db_path: str = settings.db_path
        self._cache_all: pd.DataFrame | None = None
        self._cache_quote: dict | None = None
        self._cache_quote_ts: float = 0.0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def analyze(self, symbol: str) -> dict:
        """分析指定股票，返回结构化报告。"""
        symbol = symbol.strip()
        report = StockReport(symbol=symbol)

        # 股票名称
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                f"SELECT name FROM {_STOCK_BASIC_TABLE} WHERE symbol=?", (symbol,)
            ).fetchone()
            report.name = row[0] if row else symbol

        df = self._load_ohlcv(symbol)
        if len(df) < 20:
            report.summary = f"数据不足（仅 {len(df)} 根K线），无法分析"
            return report.to_dict()

        report.date = str(df.iloc[-1]["date"])
        report.price = round(float(df.iloc[-1]["close"]), 2)

        # 六层分析：技术 + 相对强度 + 市场环境 + 策略命中 + 基本面 + 资金面
        report.technical = self._analyze_technical(df)
        report.relative_strength = self._analyze_relative_strength(symbol, df)
        report.market_context = self._analyze_market_context(symbol)
        report.strategy_hits = self._detect_strategy_hits(df, symbol)
        report.fundamental = self._analyze_fundamental(symbol)
        report.capital = self._analyze_capital(symbol)
        report.recommendation = self._build_recommendation(report)
        report.risks = self._detect_risks(report)
        report.summary = self._build_summary(report)

        # 展示层价格转换：后复权 → 真实市价（技术指标内部不受影响）
        # 复权系数必须同日口径：用 昨收(f60,与DB最新日同日) / DB后复权收盘
        # 避免旧bug: 用今日实时价(f43)/昨日复权价 混入当日涨跌幅导致价格失真
        result = report.to_dict()
        latest_price, prev_close = self._fetch_price_quote(symbol)
        if prev_close and report.price > 0:
            ratio = prev_close / report.price  # 纯复权系数（同日）
            if abs(ratio - 1.0) > 0.005:  # 有除权才转换
                result = self._convert_prices(result, ratio)
                # 展示价用实时最新价（盘中看盘需要），技术位用复权系数转换
                display = latest_price if latest_price else prev_close
                result["price"] = round(display, 2)
                result["price_source"] = "real"
            else:
                result["price_source"] = "hfq(无除权)"
        else:
            result["price_source"] = "hfq(真实价获取失败)"

        logger.info(f"个股分析完成：{symbol} {report.name}，评分 {report.recommendation.get('score', 0)}")
        return result

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    @staticmethod
    def _fetch_real_price(symbol: str) -> float | None:
        """从东财延迟行情查最新真实（不复权）价，用于展示层。"""
        info = StockAnalyzer._fetch_price_quote(symbol)
        return info[0] if info else None

    @staticmethod
    def _fetch_price_quote(symbol: str) -> tuple[float | None, float | None]:
        """查东财延迟行情，返回 (最新价f43, 昨收f60)，均为不复权真实价。

        f43=最新成交价（盘中实时/收盘价），f60=昨收（与DB后复权最新日同日）。
        复权系数必须用 f60/DB后复权收盘（同日口径），避免混入当日涨跌幅。
        """
        import requests

        prefix = "1" if symbol.startswith(("6", "5", "9")) else "0"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        try:
            r = requests.get(
                "https://push2delay.eastmoney.com/api/qt/stock/get",
                params={"secid": f"{prefix}.{symbol}", "fields": "f43,f60"},
                headers=headers, timeout=5,
            )
            d = r.json().get("data", {})
            latest = d.get("f43")
            prev = d.get("f60")
            latest = latest / 100.0 if latest else None
            prev = prev / 100.0 if prev else None
            return (latest, prev)
        except Exception:
            return (None, None)

    def _load_ohlcv(self, symbol: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM stock_daily WHERE symbol=? ORDER BY date", conn, params=(symbol,)
            )
        for c in ["open", "high", "low", "close", "volume", "turnover"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _load_all_latest(self, symbol: str | None = None) -> pd.DataFrame:
        """加载全市场最新两日数据（用于 RPS / 板块排名），带缓存。

        若指定 symbol，则确保该 symbol 的最新日期参与计算（避免数据同步不完整时
        个股最新日与全市场最新日不一致导致 RPS 失效）。
        """
        cache_key = symbol or "__global__"
        if self._cache_all is not None and getattr(self, "_cache_key", None) == cache_key:
            return self._cache_all
        with sqlite3.connect(self.db_path) as conn:
            mkt_dates = [r[0] for r in conn.execute(
                "SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT 2"
            ).fetchall()]
            if symbol:
                sym_dates = [r[0] for r in conn.execute(
                    "SELECT DISTINCT date FROM stock_daily WHERE symbol=? ORDER BY date DESC LIMIT 2",
                    (symbol,),
                ).fetchall()]
                # 若个股最新日与全市场不一致（数据同步未覆盖），用个股自己的最新两日
                # 保证 RPS 基于个股实际有的交易日，避免 None
                if sym_dates and mkt_dates and sym_dates[0] != mkt_dates[0]:
                    dates = sym_dates
                else:
                    dates = sorted(set(mkt_dates + sym_dates), reverse=True)[:2]
            else:
                dates = mkt_dates
            if len(dates) < 2:
                self._cache_all = pd.DataFrame()
                self._cache_key = cache_key
                return self._cache_all
            latest, prev = dates[0], dates[1]
            df = pd.read_sql_query(
                "SELECT symbol, date, close, high, low, volume, turnover FROM stock_daily "
                "WHERE date IN (?, ?)", conn, params=(latest, prev)
            )
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        self._cache_all = df
        self._cache_key = cache_key
        self._cache_dates = (latest, prev)
        return df

    # ------------------------------------------------------------------
    # 1. 技术面分析
    # ------------------------------------------------------------------
    def _analyze_technical(self, df: pd.DataFrame) -> dict:
        """技术面全指标（pandas 向量化）。"""
        d = df.copy()
        close, high, low = d["close"], d["high"], d["low"]
        volume = d["volume"]

        # 均线
        for w in [5, 10, 20, 60]:
            d[f"ma{w}"] = close.rolling(w).mean()
        d["ma5_vol"] = volume.rolling(5).mean()
        d["ma20_vol"] = volume.rolling(20).mean()

        last = d.iloc[-1]
        prev = d.iloc[-2]

        # 趋势排列
        ma_vals = {w: last.get(f"ma{w}") for w in [5, 10, 20, 60]}
        ma_valid = all(pd.notna(v) for v in ma_vals.values())
        if ma_valid and ma_vals[5] > ma_vals[10] > ma_vals[20] > ma_vals[60]:
            arrangement = "强势多头（MA5>10>20>60）"
            trend_dir = 1.0
        elif ma_valid and ma_vals[5] > ma_vals[10] > ma_vals[20]:
            arrangement = "多头排列（MA5>10>20）"
            trend_dir = 0.75
        elif ma_valid and ma_vals[5] < ma_vals[10] < ma_vals[20] < ma_vals[60]:
            arrangement = "强势空头（MA5<10<20<60）"
            trend_dir = -1.0
        elif ma_valid and ma_vals[5] < ma_vals[10] < ma_vals[20]:
            arrangement = "空头排列（MA5<10<20）"
            trend_dir = -0.75
        else:
            arrangement = "均线缠绕，趋势不明"
            trend_dir = 0.0

        # 乖离率
        bias = {}
        for w in [5, 10, 20, 60]:
            v = last.get(f"ma{w}")
            bias[f"bias_ma{w}"] = round((last["close"] - v) / v * 100, 2) if pd.notna(v) else None

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_bar = (dif - dea) * 2
        macd_cross = "金叉" if (prev["close"] is not None and dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2]) else \
                     ("死叉" if dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2] else ("多头" if dif.iloc[-1] > dea.iloc[-1] else "空头"))

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi14 = (100 - 100 / (1 + rs)).iloc[-1] if len(close) >= 15 else 50.0

        # KDJ
        low_9 = low.rolling(9).min()
        high_9 = high.rolling(9).max()
        rsv = (close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        dj = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * dj

        # 布林带
        boll_mid = close.rolling(20).mean()
        boll_std = close.rolling(20).std()
        boll_upper = boll_mid + 2 * boll_std
        boll_lower = boll_mid - 2 * boll_std
        boll_pos = (last["close"] - boll_lower.iloc[-1]) / (boll_upper.iloc[-1] - boll_lower.iloc[-1]) if boll_upper.iloc[-1] != boll_lower.iloc[-1] else 0.5

        # ATR（止损宽度用）
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1] if len(close) >= 15 else (high - low).tail(14).mean()

        # 量价
        vol_ratio = last["volume"] / last["ma5_vol"] if last["ma5_vol"] > 0 else 1.0
        vol_trend = "放量" if vol_ratio > 1.5 else ("缩量" if vol_ratio < 0.7 else "正常")

        # 量价背离（价涨量缩 / 价跌量增）
        price_up = last["close"] > prev["close"]
        vol_down = last["volume"] < prev["volume"]
        divergence = "量价背离（价涨量缩）" if price_up and vol_down else \
                     ("量价共振（放量上涨）" if price_up and not vol_down else "")

        # K线形态
        body = last["close"] - last["open"]
        body_pct = abs(body) / last["open"] * 100 if last["open"] > 0 else 0
        upper_shadow = last["high"] - max(last["open"], last["close"])
        lower_shadow = min(last["open"], last["close"]) - last["low"]
        if body > 0 and body_pct > 3:
            pattern = "大阳线"
        elif body < 0 and body_pct > 3:
            pattern = "大阴线"
        elif lower_shadow > abs(body) * 2 and body >= 0:
            pattern = "锤头线（下影线长，买方承接）"
        elif upper_shadow > abs(body) * 2:
            pattern = "射击之星（上影线长，卖方压力）"
        elif body_pct < 0.5:
            pattern = "十字星（方向待选择）"
        else:
            pattern = f"{'小阳' if body > 0 else '小阴'}线"

        # 支撑压力位
        support = round(float(low.tail(20).min()), 2)
        resistance = round(float(high.tail(20).max()), 2)
        if pd.notna(last.get("ma20")):
            support = min(support, round(float(last["ma20"]), 2))

        # 趋势强度评分（0-100）
        strength = 50.0
        if trend_dir >= 0.75:
            strength += 20
        elif trend_dir >= 0.5:
            strength += 10
        elif trend_dir <= -0.75:
            strength -= 20
        elif trend_dir <= -0.5:
            strength -= 10
        if dif.iloc[-1] > dea.iloc[-1]:
            strength += 10
        else:
            strength -= 5
        if rsi14 and 50 < rsi14 < 70:
            strength += 10
        elif rsi14 and rsi14 >= 80:
            strength -= 10
        elif rsi14 and rsi14 <= 30:
            strength += 5
        if vol_ratio > 1.5 and price_up:
            strength += 10
        strength = max(0, min(100, strength))

        return {
            "arrangement": arrangement,
            "trend_direction": trend_dir,
            "trend_strength": round(strength, 0),
            "ma": {f"ma{w}": round(float(v), 2) for w, v in ma_vals.items() if pd.notna(v)},
            "bias": bias,
            "macd": {
                "dif": round(float(dif.iloc[-1]), 3),
                "dea": round(float(dea.iloc[-1]), 3),
                "bar": round(float(macd_bar.iloc[-1]), 3),
                "signal": macd_cross,
            },
            "rsi": round(float(rsi14), 1) if pd.notna(rsi14) else 50,
            "rsi_signal": "超买" if rsi14 > 70 else ("超卖" if rsi14 < 30 else "中性"),
            "kdj": {
                "k": round(float(k.iloc[-1]), 1) if pd.notna(k.iloc[-1]) else 50,
                "d": round(float(dj.iloc[-1]), 1) if pd.notna(dj.iloc[-1]) else 50,
                "j": round(float(j.iloc[-1]), 1) if pd.notna(j.iloc[-1]) else 50,
            },
            "boll": {
                "upper": round(float(boll_upper.iloc[-1]), 2) if pd.notna(boll_upper.iloc[-1]) else None,
                "mid": round(float(boll_mid.iloc[-1]), 2) if pd.notna(boll_mid.iloc[-1]) else None,
                "lower": round(float(boll_lower.iloc[-1]), 2) if pd.notna(boll_lower.iloc[-1]) else None,
                "position": round(float(boll_pos), 2) if pd.notna(boll_pos) else 0.5,
            },
            "atr": round(float(atr14), 2) if pd.notna(atr14) else None,
            "atr_pct": round(float(atr14 / last["close"] * 100), 2) if pd.notna(atr14) and last["close"] else None,
            "volume_ratio": round(float(vol_ratio), 2),
            "volume_trend": vol_trend,
            "divergence": divergence,
            "pattern": pattern,
            "support": support,
            "resistance": resistance,
        }

    # ------------------------------------------------------------------
    # 2. 相对强度
    # ------------------------------------------------------------------
    def _analyze_relative_strength(self, symbol: str, df: pd.DataFrame) -> dict:
        """RPS 相对强度 + 年内位置 + 板块内排名。"""
        result: dict = {}

        # RPS：全市场对比
        all_df = self._load_all_latest(symbol)
        latest, prev = getattr(self, "_cache_dates", (None, None))
        if len(all_df) >= 100 and latest and prev:
            piv = all_df[all_df["date"].isin([latest, prev])].pivot(index="symbol", columns="date", values="close")
            if latest in piv.columns and prev in piv.columns:
                chg = (piv[latest] - piv[prev]) / piv[prev] * 100
                # 用 pct_chg 校正：除权日后复权自算会失真，改用 DB pct_chg 字段
                import sqlite3 as _sq
                with _sq.connect(self.db_path) as _c:
                    _chg_rows = _c.execute(
                        "SELECT symbol, pct_chg FROM stock_daily WHERE date=? AND pct_chg IS NOT NULL",
                        (latest,),
                    ).fetchall()
                if _chg_rows:
                    _chg_map = {s: v for s, v in _chg_rows}
                    chg = chg.copy()
                    for _idx in chg.index:
                        if _idx in _chg_map:
                            chg[_idx] = _chg_map[_idx]
                if symbol in chg.index and pd.notna(chg[symbol]):
                    rank = (chg > chg[symbol]).sum() + 1
                    result["today_chg"] = round(float(chg[symbol]), 2)
                    result["today_rank"] = int(rank)
                    result["today_total"] = int(len(chg))
                    result["today_percentile"] = round(float((1 - rank / len(chg)) * 100), 1)

        # RPS 多周期（120/60/20 日涨幅排名）
        for period in [120, 60, 20]:
            if len(df) >= period:
                ret = (df.iloc[-1]["close"] / df.iloc[-period]["close"] - 1) * 100
                # 全市场对比需查 DB，但为性能只做粗略百分位（用当日涨跌分位近似）
                result[f"rps_{period}_ret"] = round(float(ret), 2)

        # 年内位置（52周高低点）
        if len(df) >= 20:
            window = df.tail(min(len(df), 250))
            year_high = window["high"].max()
            year_low = window["low"].min()
            pos = (df.iloc[-1]["close"] - year_low) / (year_high - year_low) * 100 if year_high != year_low else 50
            result["year_high"] = round(float(year_high), 2)
            result["year_low"] = round(float(year_low), 2)
            result["year_position"] = round(float(pos), 1)
            result["dist_to_high"] = round(float((year_high - df.iloc[-1]["close"]) / df.iloc[-1]["close"] * 100), 1)
            result["dist_to_low"] = round(float((df.iloc[-1]["close"] - year_low) / df.iloc[-1]["close"] * 100), 1)

        # 板块内排名
        result.update(self._board_rank(symbol))

        return result

    def _board_rank(self, symbol: str) -> dict:
        """个股在所在板块内的涨幅排名。"""
        latest, prev = getattr(self, "_cache_dates", (None, None))
        if not latest or not prev:
            return {}
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                f"SELECT board FROM {_BOARD_TABLE} WHERE symbol=?", (symbol,)
            ).fetchone()
            if not row:
                return {"board": "未分类"}
            board = row[0]
            rows = conn.execute("""
                SELECT t.symbol, COALESCE(t.pct_chg, (t.close-p.close)/p.close*100) AS chg
                FROM (SELECT symbol,close,pct_chg FROM stock_daily WHERE date=?) t
                JOIN (SELECT symbol,close FROM stock_daily WHERE date=?) p USING(symbol)
                JOIN stock_board_em bd ON bd.symbol=t.symbol WHERE bd.board=?
            """, (latest, prev, board)).fetchall()
            if not rows:
                return {"board": board}
            sorted_rows = sorted(rows, key=lambda x: x[1], reverse=True)
            rank = next((i + 1 for i, (s, _) in enumerate(sorted_rows) if s == symbol), len(sorted_rows))
            board_avg = sum(r[1] for r in rows) / len(rows)
            my_chg = next((c for s, c in rows if s == symbol), 0)
            return {
                "board": board,
                "board_rank": rank,
                "board_total": len(rows),
                "board_avg_chg": round(float(board_avg), 2),
                "vs_board": round(float(my_chg - board_avg), 2),
            }

    # ------------------------------------------------------------------
    # 3. 市场环境联动
    # ------------------------------------------------------------------
    def _analyze_market_context(self, symbol: str) -> dict:
        """大盘状态 + 板块强度（复用 breadth 计算）。"""
        result: dict = {}
        latest, prev = getattr(self, "_cache_dates", (None, None))
        if not latest or not prev:
            return result
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT
                    SUM(CASE WHEN t.close>p.close THEN 1 ELSE 0 END),
                    SUM(CASE WHEN t.close<p.close THEN 1 ELSE 0 END),
                    COUNT(*), SUM(t.turnover)
                FROM (SELECT symbol,close,turnover FROM stock_daily WHERE date=?) t
                JOIN (SELECT symbol,close FROM stock_daily WHERE date=?) p USING(symbol)
            """, (latest, prev)).fetchone()
            up, down, total, turnover = row
            up, down, total = up or 0, down or 0, total or 0
            decided = up + down
            up_ratio = up / decided * 100 if decided else 50
            turnover_yi = (turnover or 0) / 1e8

            # 大盘评分（复用 signal_score 逻辑）
            breadth_score = round(up_ratio)
            avg_chg_row = conn.execute("""
                SELECT AVG(COALESCE(t.pct_chg, (t.close-p.close)/p.close*100)) FROM
                (SELECT symbol,close,pct_chg FROM stock_daily WHERE date=?) t
                JOIN (SELECT symbol,close FROM stock_daily WHERE date=?) p USING(symbol)
            """, (latest, prev)).fetchone()
            avg_chg = avg_chg_row[0] or 0
            change_score = max(0.0, min(100.0, (avg_chg + 5) / 10 * 100))

            # 涨跌停
            lim = conn.execute("""
                SELECT
                    SUM(CASE WHEN age>5 AND chg>=th THEN 1 ELSE 0 END),
                    SUM(CASE WHEN age>5 AND chg<=-th THEN 1 ELSE 0 END)
                FROM (
                    SELECT c.symbol,COALESCE(c.pct_chg,(c.close-p.close)/p.close*100) AS chg,
                        CASE WHEN c.symbol LIKE '30%' OR c.symbol LIKE '68%' THEN 19.5
                             WHEN COALESCE(b.name,'') LIKE '%ST%' THEN 4.6 ELSE 9.7 END AS th,
                        julianday(?) - julianday(COALESCE(b.ipo_date,'2000-01-01')) AS age
                    FROM (SELECT symbol,close,pct_chg FROM stock_daily WHERE date=?) c
                    JOIN (SELECT symbol,close FROM stock_daily WHERE date=?) p USING(symbol)
                    LEFT JOIN stock_basic b ON b.symbol=c.symbol
                )
            """, (latest, latest, prev)).fetchone()
            lu, ld = lim[0] or 0, lim[1] or 0
            lim_total = lu + ld
            limit_score = (lu / lim_total * 100) if lim_total else 50

            market_score = round(0.45 * breadth_score + 0.30 * change_score + 0.25 * limit_score)
            if market_score >= 55:
                market_state = "偏暖"
            elif market_score < 45:
                market_state = "偏冷"
            else:
                market_state = "中性"

            # 该股所在板块是否领涨 TOP10
            brd_row = conn.execute(f"SELECT board FROM {_BOARD_TABLE} WHERE symbol=?", (symbol,)).fetchone()
            board_hot = None
            if brd_row:
                boards = conn.execute(f"""
                    WITH chg AS (SELECT bd.board AS b,COALESCE(t.pct_chg,(t.close-p.close)/p.close*100) AS pct,
                        COALESCE(mc.circ_mv,t.turnover,1) AS w FROM
                        (SELECT symbol,close,turnover,pct_chg FROM stock_daily WHERE date=?) t
                        JOIN (SELECT symbol,close FROM stock_daily WHERE date=?) p USING(symbol)
                        JOIN {_BOARD_TABLE} bd ON bd.symbol=t.symbol
                        LEFT JOIN {_MARKET_CAP_TABLE} mc ON mc.symbol=t.symbol)
                    SELECT b, ROUND(SUM(pct*w)/SUM(w),2) FROM chg GROUP BY b HAVING COUNT(*)>=5 ORDER BY 2 DESC LIMIT 10
                """, (latest, prev)).fetchall()
                top_names = [b[0] for b in boards]
                board_chg = next((b[1] for b in boards if b[0] == brd_row[0]), None)
                board_hot = {
                    "is_top10": brd_row[0] in top_names,
                    "board_chg": board_chg,
                    "rank": next((i + 1 for i, b in enumerate(top_names) if b == brd_row[0]), None),
                    "name": brd_row[0],
                }

        result = {
            "market_score": market_score,
            "market_state": market_state,
            "up_ratio": round(up_ratio, 1),
            "avg_chg": round(avg_chg, 2),
            "turnover_yi": round(turnover_yi, 0),
            "limit_up": lu, "limit_down": ld,
        }
        if board_hot:
            result["board_hot"] = board_hot
        return result

    # ------------------------------------------------------------------
    # 4. 策略命中检测
    # ------------------------------------------------------------------
    def _detect_strategy_hits(self, df: pd.DataFrame, symbol: str) -> list[dict]:
        """检测该股当前命中了哪些策略（轻量版，复用策略核心逻辑）。"""
        hits: list[dict] = []
        if len(df) < 20:
            return hits
        last = df.iloc[-1]
        prev = df.iloc[-2]

        for w in [5, 10, 20]:
            if len(df) < w:
                continue
        ma5 = df["close"].rolling(5).mean()
        ma20 = df["close"].rolling(20).mean()
        vol_ma20 = df["volume"].rolling(20).mean()
        high_20 = df["high"].shift(1).rolling(20).max()

        # 均线金叉
        if len(df) >= 20 and pd.notna(ma5.iloc[-1]) and pd.notna(ma20.iloc[-1]):
            if ma5.iloc[-2] < ma20.iloc[-2] and ma5.iloc[-1] > ma20.iloc[-1]:
                if last["volume"] > vol_ma20.iloc[-1] * 1.5:
                    hits.append({"strategy": "均线放量", "key": "ma_volume", "signal": "金叉+放量"})

        # 海龟突破
        if len(df) >= 21 and pd.notna(high_20.iloc[-1]):
            if last["close"] > high_20.iloc[-1] and last["turnover"] > 1e8 and last["close"] > last["open"]:
                hits.append({"strategy": "海龟突破", "key": "turtle", "signal": "20日新高突破"})

        # 缩量回踩
        if len(df) >= 20 and pd.notna(ma5.iloc[-1]):
            ma10 = df["close"].rolling(10).mean()
            if pd.notna(ma10.iloc[-1]) and ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
                if abs(last["close"] - ma5.iloc[-1]) / ma5.iloc[-1] <= 0.015:
                    recent_vol = df["volume"].tail(3).mean()
                    if recent_vol < vol_ma20.iloc[-1] * 0.7 and last["low"] >= ma10.iloc[-1]:
                        hits.append({"strategy": "缩量回踩", "key": "pullback", "signal": "回踩MA5+缩量企稳"})

        # 底部放量
        if len(df) >= 20:
            high_20b = df["high"].rolling(20).max()
            vol_ma5 = df["volume"].rolling(5).mean()
            if pd.notna(high_20b.iloc[-1]) and pd.notna(vol_ma5.iloc[-1]):
                drawdown = (high_20b.iloc[-1] - last["close"]) / high_20b.iloc[-1] * 100
                body = abs(last["close"] - last["open"])
                lower_shadow = min(last["open"], last["close"]) - last["low"]
                if drawdown > 15 and last["volume"] > vol_ma5.iloc[-1] * 3 and last["close"] > last["open"] and body > 0 and lower_shadow > body * 2:
                    hits.append({"strategy": "底部放量", "key": "bottom", "signal": "超跌反转信号"})

        # RPS 强势
        if len(df) >= 120:
            ret_120 = (last["close"] / df.iloc[-120]["close"] - 1) * 100
            high_120 = df["high"].tail(120).max()
            if ret_120 > 50 and last["close"] > high_120 * 0.9:
                hits.append({"strategy": "RPS强势", "key": "rps", "signal": "120日强势+接近新高"})

        return hits

    # ------------------------------------------------------------------
    # 5. 综合评分与买卖建议
    # ------------------------------------------------------------------
    def _build_recommendation(self, report: StockReport) -> dict:
        """综合四层分析生成买卖建议。"""
        tech = report.technical
        rs = report.relative_strength
        mc = report.market_context

        # 子评分（0-100）
        tech_score = tech.get("trend_strength", 50)

        # 相对强度：多周期 RPS（120/60/20日涨幅）映射到0-100，反映中长期强度而非单日暴涨
        # 120日:满分门槛+80%，60日:+50%，20日:+25%（逐级递减权重，重长周期）
        rs_score = 50.0
        r120 = rs.get("rps_120_ret")
        r60 = rs.get("rps_60_ret")
        r20 = rs.get("rps_20_ret")
        components = []
        for ret, ceiling, w in [(r120, 80, 0.5), (r60, 50, 0.3), (r20, 25, 0.2)]:
            if ret is not None:
                components.append(min(100, max(0, ret / ceiling * 100)) * w)
        if components:
            rs_score = round(sum(components))

        market_score = mc.get("market_score", 50)

        vol_score = 50
        vr = tech.get("volume_ratio", 1.0)
        if vr > 1.5 and tech.get("divergence", "").startswith("量价共振"):
            vol_score = 80
        elif vr < 0.7:
            vol_score = 35
        elif tech.get("divergence", "").startswith("量价背离"):
            vol_score = 30

        # V2 新增：基本面 + 资金面子评分
        fund = report.fundamental
        cap = report.capital
        fund_score = fund.get("score", 50)
        capital_score = cap.get("score", 50)

        # 六层加权（技术25 + 相对15 + 市场15 + 量价10 + 基本面20 + 资金面15）
        total = round(
            0.25 * tech_score + 0.15 * rs_score + 0.15 * market_score
            + 0.10 * vol_score + 0.20 * fund_score + 0.15 * capital_score
        )

        # 策略命中加成
        hits = report.strategy_hits
        if hits:
            total = min(100, total + min(len(hits) * 3, 10))

        # 大盘逆风降权
        if market_score < 40:
            total = max(0, total - 10)

        # 分档建议
        if total >= 80:
            action, position, detail = "强烈买入", "15-20%", "多维信号共振，强势主线，可重仓参与"
        elif total >= 65:
            action, position, detail = "买入", "8-12%", "趋势确认，逢低分批介入"
        elif total >= 50:
            action, position, detail = "观望", "0%", "信号中性，建议等待趋势确认或回踩支撑"
        elif total >= 35:
            action, position, detail = "减仓", "<5%", "趋势走弱，降低敞口"
        else:
            action, position, detail = "回避", "0%", "多空齐弱，远离"

        # 买卖点位（ATR 自适应）
        price = report.price
        atr = tech.get("atr") or price * 0.03
        atr_pct = tech.get("atr_pct") or 3.0
        ma5 = tech.get("ma", {}).get("ma5", price)
        support = tech.get("support", price)
        ma10 = tech.get("ma", {}).get("ma10", price)

        entry_ideal = round(min(ma5, support), 2)
        entry_secondary = round(min(ma10, price * 0.98), 2)
        stop_loss = round(price - 2 * atr, 2)
        risk_per_share = price - stop_loss
        target = round(price + 3 * risk_per_share, 2) if risk_per_share > 0 else None

        # 仓位 sizing（假设总资金，单笔风险 1.5%）
        capital = 100000  # 默认10万
        risk_amount = capital * 0.015
        shares = int(risk_amount / risk_per_share / 100) * 100 if risk_per_share > 0 else 0

        return {
            "score": total,
            "action": action,
            "position": position,
            "detail": detail,
            "entry_ideal": entry_ideal,
            "entry_secondary": entry_secondary,
            "stop_loss": stop_loss,
            "target": target,
            "risk_reward_ratio": "1:3",
            "suggested_shares": shares,
            "suggested_capital": round(shares * price, 0),
            "atr_pct": atr_pct,
            "sub_scores": {
                "technical": round(tech_score),
                "relative_strength": round(rs_score),
                "market": round(market_score),
                "volume": round(vol_score),
                "fundamental": round(fund_score),
                "capital": round(capital_score),
            },
        }

    # ------------------------------------------------------------------
    # 5. 基本面分析（估值 + 财务质量 + 成长性）
    # ------------------------------------------------------------------
    def _analyze_fundamental(self, symbol: str) -> dict:
        """基本面：估值（东财实时 PE/PB）+ 财务质量（baostock 季报）+ 成长性。"""
        quote = self._fetch_quote(symbol)
        finance = self._ensure_finance_cache(symbol)  # 最近4季（倒序）

        pe = quote.get("pe")
        pb = quote.get("pb")
        fin_latest = finance[0] if finance else {}
        fin_prev = finance[1] if len(finance) > 1 else {}
        # TTM：财务质量优先取年报数据（全年值，比单季稳定，避免 Q1/三季报 ROE 偏低）
        annual = next((r for r in finance if (r.get("stat_date") or "").endswith("12-31")), None)
        fin_quality = annual or fin_latest
        roe = self._sf(fin_quality.get("roe"))
        np_margin = self._sf(fin_quality.get("np_margin"))
        gp_margin = self._sf(fin_quality.get("gp_margin"))
        yoy_ni_now = self._sf(fin_latest.get("yoy_ni"))
        yoy_ni_prev = self._sf(fin_prev.get("yoy_ni"))

        # 行业分位估值（优先）：PE/PB 在同行业内排名，跨行业可比性更强
        pe_pct = self._industry_percentile(symbol, "pe")
        pb_pct = self._industry_percentile(symbol, "pb")
        pe_score = self._score_from_percentile(pe_pct) if pe_pct is not None else self._pe_score(pe)
        pb_score = self._score_from_percentile(pb_pct) if pb_pct is not None else self._pb_score(pb)
        val_score = 0.6 * pe_score + 0.4 * pb_score
        quality_score = self._quality_score(roe, np_margin)
        growth_score = self._growth_score(yoy_ni_now, yoy_ni_prev)
        score = round(0.40 * val_score + 0.30 * quality_score + 0.30 * growth_score)

        return {
            "pe": pe, "pb": pb,
            "market_cap": quote.get("market_cap"),
            "float_cap": quote.get("float_cap"),
            "industry": quote.get("industry"),
            "pe_industry_pct": pe_pct,
            "pb_industry_pct": pb_pct,
            "roe": round(roe * 100, 1) if roe is not None else None,
            "np_margin": round(np_margin * 100, 1) if np_margin is not None else None,
            "gp_margin": round(gp_margin * 100, 1) if gp_margin is not None else None,
            "yoy_ni": round(yoy_ni_now * 100, 1) if yoy_ni_now is not None else None,
            "net_profit": self._sf(fin_quality.get("net_profit")),
            "eps_ttm": self._sf(fin_latest.get("eps_ttm")),
            "revenue": self._sf(fin_quality.get("revenue")),
            "finance_basis": "年报(TTM)" if annual else "最新单季",
            "stat_date": fin_quality.get("stat_date"),
            "score": score,
            "sub_scores": {
                "valuation": round(val_score),
                "quality": round(quality_score),
                "growth": round(growth_score),
            },
        }

    # ------------------------------------------------------------------
    # 6. 资金面分析（主力资金 + 龙虎榜）
    # ------------------------------------------------------------------
    def _analyze_capital(self, symbol: str) -> dict:
        """资金面：主力净流入占比/净额（东财）+ 龙虎榜（近5日）。"""
        quote = self._fetch_quote(symbol)

        def _safe_float(v):
            """东财返回'-'等非数字字符串时返回None，避免类型错误。"""
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        main_amount = _safe_float(quote.get("main_amount"))
        float_cap = _safe_float(quote.get("float_cap"))
        main_pct = _safe_float(quote.get("main_pct"))
        if main_pct is None and main_amount is not None and float_cap and float_cap > 0:
            main_pct = main_amount / float_cap * 100

        # 主力净流入占比评分
        if main_pct is None:
            flow_score = 50.0
        elif main_pct >= 5:
            flow_score = 90
        elif main_pct >= 2:
            flow_score = 78
        elif main_pct >= 0:
            flow_score = 60
        elif main_pct >= -2:
            flow_score = 42
        else:
            flow_score = 28

        # 净额强度（净额 / 流通市值，相对量级）
        amount_ratio = None
        if main_amount is not None and float_cap and float_cap > 0:
            amount_ratio = main_amount / float_cap * 100
        strength_score = 50.0
        if amount_ratio is not None:
            if amount_ratio >= 3:
                strength_score = 88
            elif amount_ratio >= 1:
                strength_score = 70
            elif amount_ratio >= 0:
                strength_score = 55
            elif amount_ratio >= -1:
                strength_score = 40
            else:
                strength_score = 28

        # 龙虎榜（近5日）
        lhb = self._fetch_lhb(symbol)
        lhb_score = 50.0
        if lhb:
            lhb_score = 65
            if lhb.get("net_buy", 0) > 0:
                lhb_score = 80

        score = round(0.40 * flow_score + 0.30 * strength_score + 0.30 * lhb_score)
        return {
            "score": score,
            "main_pct": round(main_pct, 2) if main_pct is not None else None,
            "main_amount": main_amount,
            "amount_ratio": round(amount_ratio, 3) if amount_ratio is not None else None,
            "lhb": lhb,
            "sub_scores": {
                "flow": round(flow_score),
                "strength": round(strength_score),
                "lhb": round(lhb_score),
            },
        }

    # ── 数据采集 ──

    def _refresh_quote_cache(self) -> None:
        """东财 push2delay 全市场快照（PE/PB/市值/板块/主力资金），5 分钟缓存。

        push2delay 每页限 100 条，8 线程并发拉全市场 ~56 页（约 2-3s）。
        """
        import time
        import requests
        from concurrent.futures import ThreadPoolExecutor
        now = time.time()
        if self._cache_quote is not None and now - self._cache_quote_ts < 300:
            return
        url = "https://push2delay.eastmoney.com/api/qt/clist/get"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        fs = "m:1 t:2,m:0 t:6,m:0 t:80,m:1 t:23"
        fields = "f12,f14,f2,f9,f23,f100,f20,f21,f62"

        def _page(pn: int) -> list:
            try:
                r = requests.get(url, params={
                    "pn": pn, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                    "fid": "f12", "fs": fs, "fields": fields,
                }, headers=headers, timeout=6)
                return (r.json().get("data") or {}).get("diff") or []
            except Exception:
                return []

        try:
            r0 = requests.get(url, params={
                "pn": 1, "pz": 1, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                "fid": "f12", "fs": fs, "fields": "f12",
            }, headers=headers, timeout=6)
            total = (r0.json().get("data") or {}).get("total") or 5500
            pages = max(1, (total + 99) // 100)
            with ThreadPoolExecutor(max_workers=8) as ex:
                results = list(ex.map(_page, range(1, pages + 1)))
            diff = [it for page in results for it in page]
            cache: dict[str, dict] = {}
            for it in diff:
                sym = str(it.get("f12") or "")
                if not sym:
                    continue
                cache[sym] = {
                    "name": it.get("f14"),
                    "price": it.get("f2"),
                    "pe": it.get("f9"),
                    "pb": it.get("f23"),
                    "industry": it.get("f100"),
                    "market_cap": it.get("f20"),
                    "float_cap": it.get("f21"),
                    "main_amount": it.get("f62"),
                }
            self._cache_quote = cache
            self._cache_quote_ts = time.time()
            logger.info(f"东财快照缓存刷新：{len(cache)} 只（{pages} 页）")
        except Exception as e:
            logger.warning(f"东财快照缓存失败：{e!r}")
            self._cache_quote = self._cache_quote or {}

    def _fetch_quote(self, symbol: str) -> dict:
        """从全市场快照缓存取单股；缓存未命中则刷新。"""
        if self._cache_quote is None:
            self._refresh_quote_cache()
        return (self._cache_quote or {}).get(symbol, {})

    def _ensure_finance_cache(self, symbol: str) -> list[dict]:
        """确保 stock_finance 表有该股最近 4 季财报；返回倒序列表。

        baostock 季报单股约 10s，落库后读秒级；季报更新频率低，按 (symbol,stat_date) 判重。
        """
        cols = [
            "symbol", "stat_date", "report_date", "roe", "np_margin", "gp_margin",
            "net_profit", "eps_ttm", "revenue", "yoy_equity", "yoy_asset", "yoy_ni",
            "yoy_eps", "yoy_pni", "nr_turn", "inv_turn", "asset_turn",
        ]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_FINANCE_TABLE} ("
                "symbol TEXT NOT NULL, stat_date TEXT NOT NULL, report_date TEXT,"
                "roe REAL, np_margin REAL, gp_margin REAL, net_profit REAL, eps_ttm REAL, revenue REAL,"
                "yoy_equity REAL, yoy_asset REAL, yoy_ni REAL, yoy_eps REAL, yoy_pni REAL,"
                "nr_turn REAL, inv_turn REAL, asset_turn REAL,"
                "PRIMARY KEY (symbol, stat_date))"
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_fin_sym ON {_FINANCE_TABLE}(symbol)")
            rows = conn.execute(
                f"SELECT {','.join(cols)} FROM {_FINANCE_TABLE} WHERE symbol=? ORDER BY stat_date DESC LIMIT 4",
                (symbol,),
            ).fetchall()
        if len(rows) >= 4:
            return [dict(zip(cols, r)) for r in rows]

        # 库中不足 4 季，从 baostock 拉取（最近 6 个候选季度，取有效者）。
        # 引用计数管理 baostock 会话：批量分析时复用同一 login，避免反复握手。
        import baostock as bs
        bs_code = self._to_baostock_code(symbol)
        fetched: list[dict] = []
        _baostock_acquire()
        try:
            for year, q in self._recent_quarters(6):
                rec = {"symbol": symbol, "stat_date": None}
                try:
                    rp = bs.query_profit_data(code=bs_code, year=year, quarter=q)
                    if rp.error_code != "0":
                        continue
                    p = None
                    while rp.next():
                        p = rp.get_row_data()
                    if not p:
                        continue
                    rec["report_date"], rec["stat_date"] = p[1], p[2]
                    rec["roe"] = self._sf(p[3])
                    rec["np_margin"] = self._sf(p[4])
                    rec["gp_margin"] = self._sf(p[5])
                    rec["net_profit"] = self._sf(p[6])
                    rec["eps_ttm"] = self._sf(p[7])
                    rec["revenue"] = self._sf(p[8])

                    rg = bs.query_growth_data(code=bs_code, year=year, quarter=q)
                    while rg.next():
                        g = rg.get_row_data()
                        rec["yoy_equity"] = self._sf(g[3])
                        rec["yoy_asset"] = self._sf(g[4])
                        rec["yoy_ni"] = self._sf(g[5])
                        rec["yoy_eps"] = self._sf(g[6])
                        rec["yoy_pni"] = self._sf(g[7])

                    ro = bs.query_operation_data(code=bs_code, year=year, quarter=q)
                    while ro.next():
                        o = ro.get_row_data()
                        rec["nr_turn"] = self._sf(o[3])
                        rec["inv_turn"] = self._sf(o[5])
                        rec["asset_turn"] = self._sf(o[8])
                    fetched.append(rec)
                except Exception as e:
                    logger.debug(f"财报采集季度 {year}Q{q} 失败：{e!r}")
                    continue
        except Exception as e:
            logger.warning(f"baostock 财报采集异常：{e!r}")
        finally:
            _baostock_release()

        if fetched:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    f"INSERT OR REPLACE INTO {_FINANCE_TABLE} ({','.join(cols)}) "
                    f"VALUES ({','.join('?' * len(cols))})",
                    [tuple(rec.get(c) for c in cols) for rec in fetched],
                )
            logger.info(f"财报入库 {symbol}：{len(fetched)} 季")
        return fetched[:4]

    def batch_prefetch_finance(self, symbols: list[str]) -> dict:
        """批量预采财报：一次 login 集中采完所有缺财报的票，再 logout。

        决策批量分析前的预热步骤。避免分析时逐只串行采集（每只3个接口×6季度），
        统一 login 后顺序采集，省去重复握手开销。
        返回 {missing: 缺财报数量, fetched: 实际采集数量, skipped: 已有数量}。
        """
        if not symbols:
            return {"missing": 0, "fetched": 0, "skipped": 0}
        # 查哪些票缺财报
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_FINANCE_TABLE} ("
                "symbol TEXT NOT NULL, stat_date TEXT NOT NULL, report_date TEXT,"
                "roe REAL, np_margin REAL, gp_margin REAL, net_profit REAL, eps_ttm REAL, revenue REAL,"
                "yoy_equity REAL, yoy_asset REAL, yoy_ni REAL, yoy_eps REAL, yoy_pni REAL,"
                "nr_turn REAL, inv_turn REAL, asset_turn REAL,"
                "PRIMARY KEY (symbol, stat_date))"
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_fin_sym ON {_FINANCE_TABLE}(symbol)")
            placeholders = ",".join("?" * len(symbols))
            rows = conn.execute(
                f"SELECT symbol, COUNT(*) FROM {_FINANCE_TABLE} "
                f"WHERE symbol IN ({placeholders}) GROUP BY symbol",
                symbols,
            ).fetchall()
        have = {r[0]: r[1] for r in rows}
        missing = [s for s in symbols if have.get(s, 0) < 4]
        if not missing:
            logger.info(f"财报预采：{len(symbols)}只全部已有，无需采集")
            return {"missing": 0, "fetched": 0, "skipped": len(symbols)}

        logger.info(f"财报批量预采：{len(missing)}/{len(symbols)} 只缺财报，集中采集...")
        import time, socket
        t0 = time.time()

        # baostock连通性预检（3秒socket超时，避免服务器宕机时卡死决策线程）
        try:
            sock = socket.create_connection(("114.94.20.73", 10030), timeout=3)
            sock.close()
        except Exception:
            logger.warning(
                f"baostock不可达(3s超时)，跳过财报预采{len(missing)}只，"
                f"已有缓存{len(symbols) - len(missing)}只照常使用"
            )
            return {"missing": len(missing), "fetched": 0, "skipped": len(symbols) - len(missing)}

        import baostock as bs
        _baostock_acquire()
        fetched_total = 0
        try:
            for i, sym in enumerate(missing):
                try:
                    result = self._fetch_finance_from_baostock(sym)
                    fetched_total += len(result)
                except Exception as e:
                    logger.debug(f"财报预采 {sym} 失败：{e!r}")
                if (i + 1) % 10 == 0:
                    logger.info(f"财报预采进度：{i + 1}/{len(missing)}")
        finally:
            _baostock_release()
        cost = time.time() - t0
        logger.info(
            f"财报批量预采完成：{fetched_total}季 / {len(missing)}只，耗时{cost:.1f}s"
            f"（平均{cost / max(len(missing), 1):.1f}s/只）"
        )
        return {"missing": len(missing), "fetched": fetched_total, "skipped": len(symbols) - len(missing)}

    def _fetch_finance_from_baostock(self, symbol: str) -> list[dict]:
        """从 baostock 采集单股财报并入库（假设已在外层 acquire 了 baostock 会话）。"""
        cols = [
            "symbol", "stat_date", "report_date", "roe", "np_margin", "gp_margin",
            "net_profit", "eps_ttm", "revenue", "yoy_equity", "yoy_asset", "yoy_ni",
            "yoy_eps", "yoy_pni", "nr_turn", "inv_turn", "asset_turn",
        ]
        import baostock as bs
        bs_code = self._to_baostock_code(symbol)
        fetched: list[dict] = []
        for year, q in self._recent_quarters(6):
            rec = {"symbol": symbol, "stat_date": None}
            try:
                rp = bs.query_profit_data(code=bs_code, year=year, quarter=q)
                if rp.error_code != "0":
                    continue
                p = None
                while rp.next():
                    p = rp.get_row_data()
                if not p:
                    continue
                rec["report_date"], rec["stat_date"] = p[1], p[2]
                rec["roe"] = self._sf(p[3])
                rec["np_margin"] = self._sf(p[4])
                rec["gp_margin"] = self._sf(p[5])
                rec["net_profit"] = self._sf(p[6])
                rec["eps_ttm"] = self._sf(p[7])
                rec["revenue"] = self._sf(p[8])
                rg = bs.query_growth_data(code=bs_code, year=year, quarter=q)
                while rg.next():
                    g = rg.get_row_data()
                    rec["yoy_equity"] = self._sf(g[3])
                    rec["yoy_asset"] = self._sf(g[4])
                    rec["yoy_ni"] = self._sf(g[5])
                    rec["yoy_eps"] = self._sf(g[6])
                    rec["yoy_pni"] = self._sf(g[7])
                ro = bs.query_operation_data(code=bs_code, year=year, quarter=q)
                while ro.next():
                    o = ro.get_row_data()
                    rec["nr_turn"] = self._sf(o[3])
                    rec["inv_turn"] = self._sf(o[5])
                    rec["asset_turn"] = self._sf(o[8])
                fetched.append(rec)
            except Exception:
                continue
        if fetched:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    f"INSERT OR REPLACE INTO {_FINANCE_TABLE} ({','.join(cols)}) "
                    f"VALUES ({','.join('?' * len(cols))})",
                    [tuple(rec.get(c) for c in cols) for rec in fetched],
                )
        return fetched

    def _fetch_lhb(self, symbol: str) -> dict | None:
        """龙虎榜：查近 7 日该股是否上榜，返回上榜日/净买额/解读。
        优先查本地 DB（毫秒级），无数据时 fallback akshare。
        """
        result = self._fetch_lhb_local(symbol)
        if result is not None:
            return result
        return self._fetch_lhb_remote(symbol)

    def _fetch_lhb_local(self, symbol: str) -> dict | None:
        """从本地 lhb_detail 查个股龙虎榜（毫秒级，无网络依赖）。"""
        import sqlite3
        from sequoia_x.core.config import Settings
        db_path = Settings().db_path
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """SELECT date, net_buy, close, pct_chg, reason, interp
                       FROM lhb_detail WHERE symbol=?
                       ORDER BY date DESC LIMIT 1""",
                    (symbol,),
                ).fetchone()
        except Exception as e:
            logger.debug(f"本地龙虎榜查询失败（表可能未创建）：{e!r}")
            return None
        if not row:
            return None
        return {
            "date": str(row["date"]),
            "reason": str(row["interp"] or row["reason"] or ""),
            "net_buy": self._sf(row["net_buy"]),
            "close": self._sf(row["close"]),
            "chg": self._sf(row["pct_chg"]),
        }

    def _fetch_lhb_remote(self, symbol: str) -> dict | None:
        """fallback：实时拉 akshare 龙虎榜（网络慢）。"""
        import akshare as ak
        from datetime import date, timedelta
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=7)).strftime("%Y%m%d")
        try:
            df = ak.stock_lhb_detail_em(start_date=start, end_date=end)
        except Exception as e:
            logger.debug(f"龙虎榜查询失败：{e!r}")
            return None
        if df is None or df.empty:
            return None
        row = df[df["代码"].astype(str) == symbol]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "date": str(r.get("上榜日", "")),
            "reason": str(r.get("解读", "")),
            "net_buy": self._sf(r.get("龙虎榜净买额")),
            "close": self._sf(r.get("收盘价")),
            "chg": self._sf(r.get("涨跌幅")),
        }

    # ── 评分辅助 ──

    @staticmethod
    def _to_baostock_code(symbol: str) -> str:
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        return f"{prefix}.{symbol}"

    @staticmethod
    def _sf(v):
        """安全转 float：None/空/'-' 返回 None。"""
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _recent_quarters(n: int = 4) -> list[tuple[int, int]]:
        """根据当前日期推算最近 n 个已披露季度（时间倒序）。

        A 股披露节奏：Q1(一季报)4/30、Q2(中报)8/31、Q3(三季报)10/31、Q4(年报)次年4/30。
        """
        from datetime import date
        today = date.today()
        y, m = today.year, today.month
        if m <= 4:
            qs = [(y - 1, q) for q in (4, 3, 2, 1)]
        elif m <= 8:
            qs = [(y, 1), (y - 1, 4), (y - 1, 3), (y - 1, 2)]
        elif m <= 10:
            qs = [(y, 2), (y, 1), (y - 1, 4), (y - 1, 3)]
        else:
            qs = [(y, 3), (y, 2), (y, 1), (y - 1, 4)]
        return qs[:n]

    def _industry_percentile(self, symbol: str, field: str) -> float | None:
        """计算该股指标在同行业内的分位（0-100，越低越便宜）。

        基于全市场快照缓存（含 PE/PB + 行业），仅统计正值样本。
        样本不足 5 只或行业缺失时返回 None，回退到绝对档位评分。
        """
        cache = self._cache_quote
        if not cache:
            return None
        my = cache.get(symbol, {})
        industry = my.get("industry")
        if not industry or industry == "-":
            return None
        my_val = self._sf(my.get(field))
        if my_val is None or my_val <= 0:
            return None
        peers = []
        for v in cache.values():
            if v.get("industry") != industry:
                continue
            pv = self._sf(v.get(field))
            if pv is not None and pv > 0:
                peers.append(pv)
        if len(peers) < 5:
            return None
        peers.sort()
        rank = sum(1 for x in peers if x <= my_val)
        return round(rank / len(peers) * 100, 1)

    @staticmethod
    def _score_from_percentile(pct: float) -> float:
        """行业分位评分（0-100）：分位越低（行业内越便宜）得分越高。

        极低分位（行业内最便宜）给高分，但保留合理区间，避免"便宜陷阱"。
        """
        if pct is None:
            return 50.0
        if pct <= 15:
            return 88.0   # 行业内显著低估
        if pct <= 30:
            return 80.0   # 偏低估
        if pct <= 50:
            return 70.0   # 中性偏低
        if pct <= 70:
            return 58.0   # 中性偏高
        if pct <= 85:
            return 44.0   # 偏贵
        return 32.0       # 行业内显著高估

    @staticmethod
    def _pe_score(pe) -> float:
        """PE 估值评分（0-100）：适中最高，两端递减，亏损重扣（行业分位缺失时的回退）。"""
        pe = StockAnalyzer._sf(pe)
        if pe is None:
            return 50.0
        if pe <= 0:
            return 18.0
        if pe <= 15:
            return 88.0
        if pe <= 25:
            return 78.0
        if pe <= 40:
            return 62.0
        if pe <= 60:
            return 46.0
        if pe <= 100:
            return 34.0
        return 26.0

    @staticmethod
    def _pb_score(pb) -> float:
        """PB 估值评分（0-100）。"""
        pb = StockAnalyzer._sf(pb)
        if pb is None:
            return 50.0
        if pb <= 0:
            return 20.0
        if pb <= 1:
            return 82.0
        if pb <= 2:
            return 78.0
        if pb <= 4:
            return 64.0
        if pb <= 8:
            return 46.0
        return 30.0

    @staticmethod
    def _quality_score(roe, np_margin) -> float:
        """财务质量评分：ROE（主导）+ 净利率（辅助）。"""
        s = 50.0
        if roe is not None:
            if roe >= 0.20:
                s = 92
            elif roe >= 0.15:
                s = 82
            elif roe >= 0.10:
                s = 70
            elif roe >= 0.05:
                s = 56
            elif roe >= 0:
                s = 42
            else:
                s = 22
        if np_margin is not None:
            if np_margin >= 0.20:
                s = min(95, s + 6)
            elif np_margin >= 0.10:
                s = min(90, s + 3)
            elif np_margin < 0:
                s = max(15, s - 8)
        return s

    @staticmethod
    def _growth_score(yoy_now, yoy_prev) -> float:
        """成长性评分：净利同比，加速加分，连续负重扣。"""
        if yoy_now is None:
            return 50.0
        if yoy_now >= 0.50:
            s = 92
        elif yoy_now >= 0.30:
            s = 82
        elif yoy_now >= 0.15:
            s = 72
        elif yoy_now >= 0:
            s = 60
        elif yoy_now >= -0.15:
            s = 42
        else:
            s = 28
        if yoy_prev is not None and yoy_now > yoy_prev + 0.05:
            s = min(98, s + 5)
        elif yoy_prev is not None and yoy_now < yoy_prev - 0.10:
            s = max(15, s - 6)
        return s

    # ------------------------------------------------------------------
    # 7. 风险提示
    # ------------------------------------------------------------------
    def _detect_risks(self, report: StockReport) -> list[str]:
        """基于数据的具体风险提示。"""
        risks: list[str] = []
        tech = report.technical
        rs = report.relative_strength
        mc = report.market_context
        bias = tech.get("bias", {})

        # 乖离率过高
        bias_ma20 = bias.get("bias_ma20")
        if bias_ma20 and bias_ma20 > 10:
            risks.append(f"距MA20乖离率 {bias_ma20}%，短期超买，追高风险较大")
        bias_ma5 = bias.get("bias_ma5")
        if bias_ma5 and bias_ma5 > 8:
            risks.append(f"距MA5乖离率 {bias_ma5}%，短线获利盘较多")

        # 量价背离
        if tech.get("divergence", "").startswith("量价背离"):
            risks.append("量价背离：价涨量缩，上涨动能不足，警惕回调")

        # RSI 超买
        if tech.get("rsi", 50) > 80:
            risks.append(f"RSI={tech['rsi']}，严重超买")
        elif tech.get("rsi", 50) < 20:
            risks.append(f"RSI={tech['rsi']}，严重超卖（可能反弹但趋势未确认）")

        # 布林带上轨
        boll_pos = tech.get("boll", {}).get("position", 0.5)
        if boll_pos > 1.0:
            risks.append("股价突破布林带上轨，短期过热")

        # 板块退潮
        board_hot = mc.get("board_hot", {})
        if board_hot.get("is_top10") is False:
            risks.append(f"所在板块「{board_hot.get('name', '')}」非领涨板块，板块支撑弱")

        # 大盘逆风
        if mc.get("market_score", 50) < 40:
            risks.append(f"大盘评分 {mc['market_score']}（偏冷），系统性风险，建议降低仓位")

        # 年内高位
        if rs.get("year_position", 50) > 90:
            risks.append(f"股价处于年内 {rs['year_position']}% 分位（接近高点），上行空间有限")

        # A股规则提示
        name = report.name or ""
        if "ST" in name.upper():
            risks.append("ST/*ST股票，日涨跌幅限制5%，退市风险高")

        # 涨跌停状态
        latest, prev = getattr(self, "_cache_dates", (None, None))
        if latest and prev:
            chg = rs.get("today_chg", 0)
            if chg >= 9.7:
                risks.append("当日接近/触及涨停，封板状态可能无法买入")
            elif chg <= -9.7:
                risks.append("当日接近/触及跌停，流动性风险")

        # V2 基本面风险
        fund = report.fundamental
        pe_f = fund.get("pe")
        if pe_f is not None and pe_f > 80:
            risks.append(f"PE={pe_f:.0f} 偏高，估值存在泡沫风险")
        yoy_ni = fund.get("yoy_ni")
        if yoy_ni is not None and yoy_ni < -10:
            risks.append(f"净利同比 {yoy_ni:.1f}%，基本面恶化、业绩承压")
        roe_f = fund.get("roe")
        if roe_f is not None and roe_f < 3:
            risks.append(f"ROE 仅 {roe_f:.1f}%，盈利能力偏弱")

        # V2 资金面风险
        cap = report.capital
        main_pct = cap.get("main_pct")
        if main_pct is not None and main_pct < -2:
            risks.append(f"主力资金净流出 {main_pct:.2f}%，资金面承压")
        lhb = cap.get("lhb")
        if lhb and lhb.get("net_buy", 0) < 0:
            risks.append(f"龙虎榜净卖出约 {lhb['net_buy']:.0f} 万元，机构/游资撤离")

        if not risks:
            risks.append("暂无明显风险信号，关注量能变化与趋势延续性")
        return risks

    # ------------------------------------------------------------------
    # 总结文案
    # ------------------------------------------------------------------
    @staticmethod
    def _convert_prices(result: dict, ratio: float) -> dict:
        """将报告内所有展示价格从后复权转换为真实市价（×ratio）。技术指标比值/百分比不变。"""
        def _p(v):
            return round(v * ratio, 2) if isinstance(v, (int, float)) and v else v

        r = result
        # 技术面：均线、支撑压力、布林带（ATR 不转，保留为后复权波动率）
        tech = r.get("technical", {})
        if tech.get("ma"):
            tech["ma"] = {k: _p(v) for k, v in tech["ma"].items()}
        tech["support"] = _p(tech.get("support"))
        tech["resistance"] = _p(tech.get("resistance"))
        if tech.get("boll"):
            for k in ("upper", "mid", "lower"):
                tech["boll"][k] = _p(tech["boll"].get(k))
        tech["atr"] = _p(tech.get("atr"))

        # 相对强度：年内高低点
        rs = r.get("relative_strength", {})
        rs["year_high"] = _p(rs.get("year_high"))
        rs["year_low"] = _p(rs.get("year_low"))

        # 买卖点位
        rec = r.get("recommendation", {})
        for k in ("entry_ideal", "entry_secondary", "stop_loss", "target"):
            rec[k] = _p(rec.get(k))
        rec["suggested_capital"] = round(rec.get("suggested_shares", 0) * _p(r.get("price", 0)), 0)

        # 当前价也转换（在重建文案之前，确保文案用真实价）
        r["price"] = _p(r.get("price", 0))

        # 重建总结文案（使用转换后的真实价格）
        from sequoia_x.analysis.stock_analysis import StockAnalyzer as _SA
        tmp = StockReport(
            symbol=r["symbol"], name=r["name"], date=r["date"],
            price=r.get("price", 0),
            technical=r.get("technical", {}),
            fundamental=r.get("fundamental", {}),
            capital=r.get("capital", {}),
            recommendation=r.get("recommendation", {}),
            strategy_hits=r.get("strategy_hits", []),
        )
        r["summary"] = _SA._build_summary(tmp)
        return r

    @staticmethod
    def _build_summary(report: StockReport) -> str:
        rec = report.recommendation
        tech = report.technical
        parts = [
            f"{report.name}（{report.symbol}）当前价 {report.price}，",
            f"技术面{tech.get('arrangement', '')}，趋势强度 {tech.get('trend_strength', 50)}/100。",
            f"综合评分 {rec.get('score', 0)}/100，建议【{rec.get('action', '观望')}】。",
        ]
        if rec.get("action") in ("买入", "强烈买入"):
            parts.append(
                f"理想买点 {rec.get('entry_ideal')}，止损 {rec.get('stop_loss')}，"
                f"目标 {rec.get('target')}，仓位 {rec.get('position')}。"
            )
        hits = report.strategy_hits
        if hits:
            parts.append(f"命中策略：{ '、'.join(h['strategy'] for h in hits)}。")
        # V2 资金/基本面摘要
        bits = []
        fund = report.fundamental
        if fund.get("pe") is not None:
            bits.append(f"PE{fund['pe']:.0f}")
        if fund.get("roe") is not None:
            bits.append(f"ROE{fund.get('roe'):.1f}%")
        cap = report.capital
        mp = cap.get("main_pct")
        if mp is not None:
            bits.append(f"主力{'净流入' if mp >= 0 else '净流出'}{abs(mp):.1f}%")
        if bits:
            parts.append("资金/基本面：" + "、".join(bits) + "。")
        return "".join(parts)
