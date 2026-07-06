"""大盘分析引擎：基于本地行情库 + baostock 指数/行业数据，生成结构化盘面报告。

报告七大板块：
  1. 盘面总览（涨跌家数、涨跌停、成交额、盘面信号评分）
  2. 指数结构（上证、深证、创业板、上证50、沪深300 + 支撑/压力位）
  3. 板块主线（证监会行业分类涨跌幅排行）
  4. 资金与情绪
  5. 消息催化
  6. 明日交易计划
  7. 风险提示
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# 主要指数定义：key -> (baostock 代码, 中文名)
_INDEX_DEFS: list[tuple[str, str, str]] = [
    ("sh", "sh.000001", "上证指数"),
    ("sz", "sz.399001", "深证成指"),
    ("cyb", "sz.399006", "创业板指"),
    ("sz50", "sh.000016", "上证50"),
    ("hs300", "sh.000300", "沪深300"),
]

_INDUSTRY_TABLE = "stock_industry"
_BOARD_TABLE = "stock_board_em"
_STOCK_BASIC_TABLE = "stock_basic"
_MARKET_CAP_TABLE = "stock_market_cap"


def _clean_industry_name(raw: str) -> str:
    """清洗证监会行业名：去除前缀分类码（如 'C39计算机...' -> '计算机...'）。"""
    if not raw:
        return "未分类"
    return re.sub(r"^[A-Z]\d+\s*", "", raw).strip() or "未分类"


@dataclass
class MarketReport:
    """结构化盘面报告（可序列化为 dict）。"""

    date: str
    generated_at: str
    data_source: str
    overview: dict = field(default_factory=dict)
    indices: list[dict] = field(default_factory=list)
    sectors: dict = field(default_factory=dict)
    sentiment: dict = field(default_factory=dict)
    news: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    fund_flow: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    valuation: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)
    limit_structure: dict = field(default_factory=dict)
    dragon_tiger: dict = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "generated_at": self.generated_at,
            "data_source": self.data_source,
            "overview": self.overview,
            "indices": self.indices,
            "sectors": self.sectors,
            "sentiment": self.sentiment,
            "news": self.news,
            "plan": self.plan,
            "fund_flow": self.fund_flow,
            "history": self.history,
            "valuation": self.valuation,
            "macro": self.macro,
            "limit_structure": self.limit_structure,
            "dragon_tiger": self.dragon_tiger,
            "risks": self.risks,
        }


class MarketAnalyzer:
    """大盘分析器：聚合本地行情与 baostock 数据生成报告。"""

    def __init__(self, settings: Settings) -> None:
        self.db_path: str = settings.db_path

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def analyze(self, target_date: str | None = None) -> dict:
        """生成指定日期（默认最新交易日）的大盘分析报告。"""
        latest, prev = self._latest_dates(target_date)
        self._ensure_stock_basic_cache()
        self._ensure_market_cap_cache()
        report = MarketReport(
            date=latest,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data_source=f"Sequoia-X 本地库 + baostock（{latest}）",
        )

        # 1. 盘面总览（本地库，瞬时）
        breadth = self._compute_breadth(latest, prev)
        turnover_stats = self._compute_turnover_stats(latest)
        report.overview = self._build_overview(breadth, turnover_stats)
        signal = report.overview["signal_score"]

        # 2. 指数结构（baostock，~5s）
        try:
            report.indices = self._fetch_indices(latest)
        except Exception as exc:
            logger.warning(f"指数数据获取失败：{exc}")
            report.indices = []

        # 3. 板块主线（本地库 + 缓存行业分类）
        try:
            report.sectors = self._compute_sectors(latest, prev)
        except Exception as exc:
            logger.warning(f"板块数据计算失败：{exc}")
            report.sectors = {"top": [], "bottom": [], "text": f"板块数据计算失败：{exc}"}

        # 4. 资金与情绪
        sentiment_breadth = self._compute_market_breadth(latest)
        report.sentiment = self._build_sentiment(
            breadth, report.indices, report.sectors, turnover_stats, sentiment_breadth
        )

        # 4.5 资金流向（主力资金 + 融资融券，网络数据源）
        try:
            report.fund_flow = self._build_fund_flow(latest)
        except Exception as exc:
            logger.warning(f"资金流向数据获取失败：{exc}")
            report.fund_flow = {"text": f"资金流向数据获取失败：{exc}"}

        # 5. 消息催化（实时新闻聚合 + 板块关联分析）
        report.news = self._build_news(report.sectors, latest)

        # 6. 明日交易计划
        report.plan = self._build_plan(signal, report.indices, report.sectors, turnover_stats)

        # 四B. 历史纵向对比
        try:
            report.history = self._compute_history(latest, prev)
        except Exception as exc:
            logger.warning(f"历史纵向对比失败：{exc}")

        # 五. 估值与宏观
        try:
            report.valuation = self._fetch_valuation()
        except Exception as exc:
            logger.warning(f"估值数据获取失败：{exc}")
        try:
            report.macro = self._fetch_macro()
        except Exception as exc:
            logger.warning(f"宏观数据获取失败：{exc}")

        # 六. 涨跌停结构 + 龙虎榜
        try:
            report.limit_structure = self._compute_limit_structure(latest, prev)
        except Exception as exc:
            logger.warning(f"涨停板结构分析失败：{exc}")
        try:
            report.dragon_tiger = self._fetch_dragon_tiger()
        except Exception as exc:
            logger.warning(f"龙虎榜获取失败：{exc}")

        # 7. 风险提示
        report.risks = self._build_risks(breadth, report.indices, report.sectors)

        return report.to_dict()

    # ------------------------------------------------------------------
    # 日期处理
    # ------------------------------------------------------------------
    def _latest_dates(self, target_date: str | None) -> tuple[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            if target_date:
                latest = target_date
            else:
                row = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()
                latest = row[0] if row and row[0] else date.today().strftime("%Y-%m-%d")
            prev_row = conn.execute(
                "SELECT MAX(date) FROM stock_daily WHERE date < ?", (latest,)
            ).fetchone()
            prev = prev_row[0] if prev_row and prev_row[0] else latest
        return latest, prev

    # ------------------------------------------------------------------
    # 1. 盘面总览
    # ------------------------------------------------------------------
    def _compute_breadth(self, latest: str, prev: str) -> dict:
        sql = """
        WITH t AS (SELECT symbol, close, turnover FROM stock_daily WHERE date = ?),
             p AS (SELECT symbol, close FROM stock_daily WHERE date = ?)
        SELECT
            SUM(CASE WHEN t.close > p.close THEN 1 ELSE 0 END) AS up,
            SUM(CASE WHEN t.close < p.close THEN 1 ELSE 0 END) AS down,
            SUM(CASE WHEN t.close = p.close THEN 1 ELSE 0 END) AS flat,
            COUNT(*) AS total,
            SUM(t.turnover) AS turnover,
            AVG((t.close - p.close) / p.close * 100) AS avg_chg
        FROM t JOIN p ON t.symbol = p.symbol
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(sql, (latest, prev)).fetchone()
        up, down, flat, total, turnover, avg_chg = row

        # 涨跌停：封板(close 涨幅达标)、触及(high/low 涨幅达标)、炸板(触及未封)
        # 后复权下涨跌幅比例不变；按板块/ST 精确限幅，新股(上市≤5日)排除
        lim_sql = """
        WITH c AS (
            SELECT t.symbol,
                   (t.close - p.close) / p.close * 100 AS chg,
                   (t.high  - p.close) / p.close * 100 AS hi,
                   (t.low   - p.close) / p.close * 100 AS lo
            FROM (SELECT symbol, close, high, low FROM stock_daily WHERE date = ?) t
            JOIN (SELECT symbol, close FROM stock_daily WHERE date = ?) p USING (symbol)
        ),
        cl AS (
            SELECT c.*,
                   CASE
                       WHEN c.symbol LIKE '30%' OR c.symbol LIKE '68%' THEN 19.5
                       WHEN c.symbol LIKE '8%' OR c.symbol LIKE '4%' THEN 29.0
                       WHEN COALESCE(b.name, '') LIKE '%ST%' THEN 4.6
                       ELSE 9.7
                   END AS th,
                   julianday(?) - julianday(COALESCE(b.ipo_date, '2000-01-01')) AS age
            FROM c
            LEFT JOIN stock_basic b ON b.symbol = c.symbol
        )
        SELECT
            SUM(CASE WHEN age > 5 AND chg >=  th THEN 1 ELSE 0 END) AS lu,
            SUM(CASE WHEN age > 5 AND hi  >=  th THEN 1 ELSE 0 END) AS touched_up,
            SUM(CASE WHEN age > 5 AND chg <= -th THEN 1 ELSE 0 END) AS ld,
            SUM(CASE WHEN age > 5 AND lo  <= -th THEN 1 ELSE 0 END) AS touched_dn
        FROM cl
        """
        with sqlite3.connect(self.db_path) as conn:
            lu, touched_up, ld, touched_dn = conn.execute(
                lim_sql, (latest, prev, latest)
            ).fetchone()
        lu, touched_up = lu or 0, touched_up or 0
        ld, touched_dn = ld or 0, touched_dn or 0
        broken_up = touched_up - lu
        broken_rate = round(broken_up / touched_up * 100, 1) if touched_up else 0.0

        turnover_yi = round((turnover or 0) / 1e8, 1)
        decided = (up or 0) + (down or 0)
        up_ratio = round((up or 0) / decided * 100, 1) if decided else 0.0

        return {
            "up": up or 0,
            "down": down or 0,
            "flat": flat or 0,
            "total": total or 0,
            "up_ratio": up_ratio,
            "limit_up": lu,
            "touched_up": touched_up,
            "broken_up": broken_up,
            "broken_rate": broken_rate,
            "limit_down": ld,
            "touched_down": touched_dn,
            "limit_diff": lu - ld,
            "turnover_yi": turnover_yi,
            "avg_change": round(avg_chg or 0, 2),
            "turnover_label": self._turnover_label(turnover_yi),
        }

    @staticmethod
    def _turnover_label(turnover_yi: float) -> str:
        if turnover_yi >= 20000:
            return "极高活跃度"
        if turnover_yi >= 12000:
            return "高活跃度"
        if turnover_yi >= 8000:
            return "中等活跃度"
        return "缩量"

    def _compute_turnover_stats(self, latest: str) -> dict:
        """计算近期成交额均值（近5日、近20日，均不含当日），用于纵向放量/缩量对比。"""
        sql = """
        SELECT date, SUM(turnover) AS turnover
        FROM stock_daily
        WHERE date < ? AND date IN (SELECT DISTINCT date FROM stock_daily)
        GROUP BY date
        ORDER BY date DESC
        LIMIT 20
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, (latest,)).fetchall()
        recent = [r[1] / 1e8 for r in rows]  # 转为亿元
        avg5 = round(sum(recent[:5]) / len(recent[:5]), 0) if len(recent) >= 5 else (round(sum(recent) / len(recent), 0) if recent else 0)
        avg20 = round(sum(recent) / len(recent), 0) if recent else 0
        return {"avg5": avg5, "avg20": avg20, "days": len(recent)}

    @staticmethod
    def _signal_score(b: dict) -> tuple[int, str, str]:
        """计算盘面信号评分（0-100）及建议，返回 (score, label, advice)。"""
        breadth_score = b["up_ratio"]
        change_score = max(0.0, min(100.0, (b["avg_change"] + 5) / 10 * 100))
        lim = b["limit_up"] + b["limit_down"]
        limit_score = (b["limit_up"] / lim * 100) if lim else 50.0
        score = round(0.45 * breadth_score + 0.30 * change_score + 0.25 * limit_score)

        if score >= 70:
            label, advice = "强势，积极进攻", "风险偏好高，可重仓主线，但仍需保持仓位纪律。"
        elif score >= 55:
            label, advice = "偏暖，可进攻", "风险偏好尚可，关注主线延续与仓位纪律。"
        elif score >= 45:
            label, advice = "中性，均衡配置", "多空均衡，建议均衡配置，控制单一主线仓位。"
        elif score >= 35:
            label, advice = "偏冷，谨慎防守", "赚钱效应转弱，降低仓位、收缩至防御性方向。"
        else:
            label, advice = "弱势，严控仓位", "市场避险情绪浓厚，严控仓位、以观望为主。"
        return score, label, advice

    def _build_overview(self, b: dict, turnover_stats: dict | None = None) -> dict:
        score, label, advice = self._signal_score(b)
        decided = b["up"] + b["down"]
        ratio_pct = round(b["up"] / decided * 100, 1) if decided else 0
        tone = "整体偏暖" if score >= 55 else ("整体偏冷" if score < 45 else "多空均衡")
        # 成交额纵向对比
        vol_note = ""
        if turnover_stats and turnover_stats["avg20"] > 0:
            avg20 = turnover_stats["avg20"]
            cur = b["turnover_yi"]
            pct_vs = (cur - avg20) / avg20 * 100
            if pct_vs >= 15:
                vol_note = f"（较近{turnover_stats['days']}日均值 {avg20:.0f} 亿显著放量 {pct_vs:+.0f}%，增量资金入场信号明显）"
            elif pct_vs >= 5:
                vol_note = f"（较近{turnover_stats['days']}日均值 {avg20:.0f} 亿放量 {pct_vs:+.0f}%）"
            elif pct_vs <= -15:
                vol_note = f"（较近{turnover_stats['days']}日均值 {avg20:.0f} 亿明显缩量 {pct_vs:+.0f}%）"
            elif pct_vs <= -5:
                vol_note = f"（较近{turnover_stats['days']}日均值 {avg20:.0f} 亿小幅缩量 {pct_vs:+.0f}%）"
            else:
                vol_note = f"（与近{turnover_stats['days']}日均值 {avg20:.0f} 亿基本持平）"
        text = (
            f"今日市场{tone}，全市场 {b['total']} 只个股中上涨 {b['up']} 家、下跌 {b['down']} 家，"
            f"上涨占比约 {ratio_pct}%。两市成交额 {b['turnover_yi']:.0f} 亿元{vol_note}，"
            f"涨停 {b['limit_up']} 家、跌停 {b['limit_down']} 家（涨跌停差 {b['limit_diff']:+d}），"
            f"全市场平均涨跌幅 {b['avg_change']:+.2f}%。"
        )
        return {
            "text": text,
            "signal_score": score,
            "signal_label": label,
            "signal_advice": advice,
            "signal_basis": (
                f"上涨家数占比 {ratio_pct}%，{'市场分化' if 40 <= ratio_pct <= 65 else ''}；"
                f"全市场平均涨跌幅 {b['avg_change']:+.2f}%；涨跌停差 {b['limit_diff']:+d}"
            ).strip(),
            "metrics": b,
        }

    # ------------------------------------------------------------------
    # 2. 指数结构（baostock）
    # ------------------------------------------------------------------
    def _fetch_indices(self, latest: str) -> list[dict]:
        import baostock as bs
        from datetime import date as _date, timedelta as _td

        # 仅拉取近 ~45 个交易日（约 65 自然日），足够计算涨跌幅与 20 日支撑/压力
        start_date = (_date.fromisoformat(latest) - _td(days=65)).strftime("%Y-%m-%d")

        bs.login()
        results: list[dict] = []
        try:
            for key, bs_code, name in _INDEX_DEFS:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,volume,amount",
                    start_date=start_date,
                    end_date=latest,
                    frequency="d",
                )
                if rs.error_code != "0":
                    continue
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if len(rows) < 2:
                    continue
                df = pd.DataFrame(rows, columns=rs.fields)
                for col in ["open", "high", "low", "close", "amount"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"]).reset_index(drop=True)
                if len(df) < 2:
                    continue
                cur = df.iloc[-1]
                prv = df.iloc[-2]
                change_pct = round((cur["close"] - prv["close"]) / prv["close"] * 100, 2)
                amplitude = round((cur["high"] - cur["low"]) / prv["close"] * 100, 2)
                # 支撑/压力：近 5 日高低点（贴近实战的短期关键位）
                window = df.tail(5)
                support = round(float(window["low"].min()), 2)
                resistance = round(float(window["high"].max()), 2)
                results.append({
                    "key": key,
                    "name": name,
                    "latest": round(float(cur["close"]), 2),
                    "change_pct": change_pct,
                    "open": round(float(cur["open"]), 2),
                    "high": round(float(cur["high"]), 2),
                    "low": round(float(cur["low"]), 2),
                    "amplitude": amplitude,
                    "turnover_yi": round(float(cur["amount"]) / 1e8, 0),
                    "support": support,
                    "resistance": resistance,
                })
        finally:
            bs.logout()
        return results

    # ------------------------------------------------------------------
    # 3. 板块主线
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 股票元数据缓存（名称 / 上市日 → 涨跌停限幅与新股过滤）
    # ------------------------------------------------------------------
    def _ensure_stock_basic_cache(self) -> None:
        """确保 stock_basic 缓存表存在；不存在则从 baostock 拉取股票名称与上市日。"""
        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (_STOCK_BASIC_TABLE,)
            ).fetchone()
            if exists:
                return

        logger.info("首次构建股票元数据缓存（baostock，约 10~20s）...")
        import baostock as bs

        bs.login()
        rows: list[tuple[str, str, str]] = []
        try:
            rs = bs.query_stock_basic(code_name="", code="")
            while rs.next():
                data = rs.get_row_data()
                if data[4] == "1":  # type == "1"：股票（含已退市，保留以覆盖历史回放）
                    symbol = data[0].split(".")[-1]
                    rows.append((symbol, data[1], data[2]))  # symbol, name, ipo_date
        finally:
            bs.logout()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_STOCK_BASIC_TABLE} ("
                "symbol TEXT PRIMARY KEY, name TEXT, ipo_date TEXT)"
            )
            conn.executemany(
                f"INSERT OR REPLACE INTO {_STOCK_BASIC_TABLE} (symbol, name, ipo_date) VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()
        logger.info(f"股票元数据缓存完成，共 {len(rows)} 条")

    def refresh_stock_basic_cache(self) -> int:
        """强制刷新股票元数据缓存，返回写入条数。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {_STOCK_BASIC_TABLE}")
            conn.commit()
        self._ensure_stock_basic_cache()
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {_STOCK_BASIC_TABLE}").fetchone()[0]

    # ------------------------------------------------------------------
    # 流通市值缓存（板块市值加权用；缓变，周期刷新）
    # ------------------------------------------------------------------
    def _ensure_market_cap_cache(self) -> None:
        """确保流通市值缓存表存在；不存在则从东财延迟行情拉取全市场快照。"""
        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (_MARKET_CAP_TABLE,)
            ).fetchone()
            if exists:
                return
        logger.info("首次构建流通市值缓存（东财延迟行情，约 5~10s）...")
        rows = self._fetch_market_cap_all()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_MARKET_CAP_TABLE} ("
                "symbol TEXT PRIMARY KEY, circ_mv REAL)"
            )
            conn.executemany(
                f"INSERT OR REPLACE INTO {_MARKET_CAP_TABLE} (symbol, circ_mv) VALUES (?, ?)", rows
            )
            conn.commit()
        logger.info(f"流通市值缓存完成，共 {len(rows)} 条")

    def refresh_market_cap_cache(self) -> int:
        """强制刷新流通市值缓存，返回写入条数。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {_MARKET_CAP_TABLE}")
            conn.commit()
        self._ensure_market_cap_cache()
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {_MARKET_CAP_TABLE}").fetchone()[0]

    @staticmethod
    def _fetch_market_cap_all() -> list[tuple[str, float]]:
        """从东财 push2delay 拉取全市场流通市值（单位：元，与 turnover 同口径）。

        push2delay 强制每页上限 100 条（total≈5800），故按页全量分页拉取；
        端点为延迟行情，盘后稳定可用。字段：f12=代码, f21=流通市值。
        """
        import requests

        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        base_url = "https://push2delay.eastmoney.com/api/qt/clist/get"
        fs = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
        page_size = 100
        rows: list[tuple[str, float]] = []
        page = 1
        while True:
            params = {
                "pn": page, "pz": page_size, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                "fid": "f3", "fs": fs, "fields": "f12,f21",
            }
            try:
                r = requests.get(base_url, params=params, headers=headers, timeout=10)
                data = r.json().get("data") or {}
                diff = data.get("diff", []) or []
                total = data.get("total", 0)
            except Exception:
                diff = []
                total = 0
            if not diff:
                break
            for b in diff:
                sym, mv = str(b.get("f12", "")), b.get("f21")
                if sym and mv is not None and mv != "-":
                    rows.append((sym, float(mv)))
            if total and len(rows) >= total:
                break
            page += 1
            time.sleep(0.1)
        return rows

    def _ensure_industry_cache(self) -> None:
        """确保 stock_industry 缓存表存在；不存在则从 baostock 拉取并写入。"""
        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (_INDUSTRY_TABLE,)
            ).fetchone()
            if exists:
                return

        logger.info("首次构建行业分类缓存（baostock，约 30~45s）...")
        import baostock as bs

        bs.login()
        rows: list[tuple[str, str]] = []
        try:
            rs = bs.query_stock_industry()
            while rs.next():
                data = rs.get_row_data()
                code = data[1]          # sh.600000
                industry = data[3]      # C39计算机...
                symbol = code.split(".")[-1]
                rows.append((symbol, _clean_industry_name(industry)))
        finally:
            bs.logout()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_INDUSTRY_TABLE} ("
                "symbol TEXT PRIMARY KEY, industry TEXT)"
            )
            conn.executemany(
                f"INSERT OR REPLACE INTO {_INDUSTRY_TABLE} (symbol, industry) VALUES (?, ?)", rows
            )
            conn.commit()
        logger.info(f"行业分类缓存完成，共 {len(rows)} 条")

    def refresh_industry_cache(self) -> int:
        """强制刷新行业分类缓存，返回写入条数。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {_INDUSTRY_TABLE}")
            conn.commit()
        self._ensure_industry_cache()
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {_INDUSTRY_TABLE}").fetchone()[0]

    # ------------------------------------------------------------------
    # 东财细分行业板块缓存（个股→最细板块映射，本地计算涨跌幅）
    # ------------------------------------------------------------------
    def _has_board_cache(self) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {_BOARD_TABLE}").fetchone()
            return row[0] > 0

    def refresh_board_cache(self) -> int:
        """从东方财富拉取全部行业板块成分股，构建「个股→最细板块」映射并持久化。

        东财板块存在层级（如「半导体」含「模拟芯片设计」），处理策略：按成员数
        升序遍历，每只股票首次出现即归属（最细板块优先），粗板块自动仅保留无细分的个股。
        一次构建约 3~5 分钟，之后板块涨跌幅全部从本地 stock_daily 计算，零网络依赖。
        """
        import requests

        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        base_url = "https://push2delay.eastmoney.com/api/qt/clist/get"

        # 1. 拉取全部行业板块（代码、名称、成员数）
        boards: list[tuple[str, str, int]] = []
        for page in range(1, 8):
            params = {
                "pn": page, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                "fid": "f3", "fs": "m:90 t:2", "fields": "f12,f14,f104,f105",
            }
            try:
                r = requests.get(base_url, params=params, headers=headers, timeout=10)
                diff = (r.json().get("data") or {}).get("diff", []) or []
            except Exception:
                diff = []
            if not diff:
                break
            for b in diff:
                count = int(b["f104"]) + int(b["f105"])
                boards.append((b["f12"], str(b["f14"]), count))
            if len(diff) < 100:
                break
            time.sleep(0.3)

        if not boards:
            raise RuntimeError("无法获取板块列表（东财接口不可用）")

        # 2. 按成员数升序（最细板块优先），逐板块取成分股并分配
        boards.sort(key=lambda x: x[2])
        stock_board: dict[str, str] = {}
        for idx, (code, name, _) in enumerate(boards):
            for pn in range(1, 4):  # 每板块最多 3 页（300 只），覆盖绝大多数
                params = {
                    "pn": pn, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                    "fid": "f3", "fs": f"b:{code}", "fields": "f12",
                }
                try:
                    r = requests.get(base_url, params=params, headers=headers, timeout=10)
                    diff = (r.json().get("data") or {}).get("diff", []) or []
                except Exception:
                    diff = []
                if not diff:
                    break
                for b in diff:
                    sym = str(b["f12"])
                    if sym not in stock_board:
                        stock_board[sym] = name
                if len(diff) < 100:
                    break
                time.sleep(0.15)
            time.sleep(0.15)
            if (idx + 1) % 50 == 0:
                logger.info(f"板块映射构建中... {idx + 1}/{len(boards)} 板块，已映射 {len(stock_board)} 只")

        # 3. 持久化到 stock_board_em 表
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {_BOARD_TABLE}")
            conn.execute(
                f"CREATE TABLE {_BOARD_TABLE} (symbol TEXT PRIMARY KEY, board TEXT)"
            )
            conn.executemany(
                f"INSERT OR REPLACE INTO {_BOARD_TABLE} (symbol, board) VALUES (?, ?)",
                list(stock_board.items()),
            )
            conn.commit()
        logger.info(f"东财板块映射缓存完成：{len(stock_board)} 只股票 → {len(set(stock_board.values()))} 个板块")
        return len(stock_board)

    def _fetch_sectors_em(self) -> list[dict]:
        """从东方财富拉取行业板块实时涨跌幅（~496 个细分行业，与市面盘面报告口径一致）。

        数据源：91.push2.eastmoney.com （push2 主域名在部分网络被限，91 子域可用）。
        """
        import requests

        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        base_params = {
            "po": 1, "np": 1, "fltt": 2, "invt": 2,
            "fid": "f3", "fs": "m:90 t:2",
            "fields": "f12,f14,f3,f104,f105,f6",  # 代码,名称,涨跌幅,涨家数,跌家数,成交额
        }
        # push2delay 为延迟行情服务（~15min），基础设施独立于 push2 主域，不易被限流；
        # 对盘后分析无影响。91.push2 为实时行情备选。
        endpoints = [
            "https://push2delay.eastmoney.com/api/qt/clist/get",
            "https://91.push2.eastmoney.com/api/qt/clist/get",
        ]
        # 东财 API 每页上限 100 条，需分页拉取全部 ~496 个行业板块
        page_size = 100
        all_boards: list[dict] = []
        for attempt in range(3):
            all_boards = []
            for url in endpoints:
                try:
                    for page in range(1, 8):  # 最多 7 页 = 700 条，覆盖全部
                        params = {**base_params, "pn": page, "pz": page_size}
                        r = requests.get(url, params=params, headers=headers, timeout=10)
                        data = r.json()
                        diff = (data.get("data") or {}).get("diff", []) or []
                        if not diff:
                            break
                        for b in diff:
                            all_boards.append({
                                "name": str(b["f14"]),
                                "change_pct": round(float(b["f3"]), 2),
                                "count": int(b["f104"]) + int(b["f105"]),
                                "turnover_yi": round(float(b["f6"]) / 1e8, 0),
                            })
                        if len(diff) < page_size:
                            break  # 最后一页不足 100，说明已取完
                    if all_boards:
                        break  # 某个端点成功即跳出端点循环
                except Exception:
                    continue
            if all_boards:
                return all_boards
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        return []

    def _compute_sectors(self, latest: str, prev: str) -> dict:
        """板块主线（三级回退）：
        1. 东财实时板块接口 push2delay（首选，涨跌幅为官方结算值，与市面报告一致）
        2. 本地东财细分板块映射 stock_board_em（网络不可用时，基于本地 close 计算）
        3. 本地证监会行业分类 stock_industry（最终回退）
        """
        # 1. 东财实时接口（官方涨跌幅，与样例报告口径一致）
        try:
            boards = self._fetch_sectors_em()
            if boards:
                boards.sort(key=lambda x: x["change_pct"], reverse=True)
                top = boards[:8]
                bottom = list(reversed(boards[-8:]))
                themes = self._detect_themes(top[:5])
                catalyst = self._catalyst_note(themes)
                text = self._sector_text(top[:5], bottom[:5], themes, catalyst)
                logger.info(f"板块数据来源：东方财富实时接口（{len(boards)} 个细分行业）")
                return {"top": top, "bottom": bottom, "text": text}
            logger.warning("东财实时板块数据为空，回退本地板块缓存")
        except Exception as exc:
            logger.warning(f"东财实时板块数据异常，回退本地计算：{exc}")

        # 2. 本地东财细分板块缓存
        if self._has_board_cache():
            try:
                result = self._compute_sectors_local_em(latest, prev)
                logger.info("板块数据来源：本地东财细分板块映射（stock_board_em）")
                return result
            except Exception as exc:
                logger.warning(f"本地板块计算异常，回退证监会行业分类：{exc}")

        # 3. 本地证监会行业
        return self._compute_sectors_local(latest, prev)

    def _compute_sectors_local_em(self, latest: str, prev: str) -> dict:
        """基于本地 stock_board_em 映射 + stock_daily 计算板块涨跌幅（流通市值加权）。"""
        sql = f"""
        WITH chg AS (
            SELECT bd.board AS sector,
                   (t.close - p.close) / p.close * 100 AS pct,
                   COALESCE(mc.circ_mv, t.turnover, 1) AS w
            FROM (SELECT symbol, close, turnover FROM stock_daily WHERE date = ?) t
            JOIN (SELECT symbol, close FROM stock_daily WHERE date = ?) p USING (symbol)
            JOIN {_BOARD_TABLE} bd ON bd.symbol = t.symbol
            LEFT JOIN {_MARKET_CAP_TABLE} mc ON mc.symbol = t.symbol
        )
        SELECT sector,
               COUNT(*) AS cnt,
               ROUND(SUM(pct * w) / SUM(w), 2) AS avg_chg
        FROM chg
        GROUP BY sector
        HAVING cnt >= 1
        ORDER BY avg_chg DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, (latest, prev)).fetchall()

        top = [{"name": r[0], "change_pct": r[2], "count": r[1]} for r in rows[:8]]
        bottom = [{"name": r[0], "change_pct": r[2], "count": r[1]} for r in rows[-8:][::-1]]
        themes = self._detect_themes(top[:5])
        catalyst = self._catalyst_note(themes)
        text = self._sector_text(top[:5], bottom[:5], themes, catalyst)
        return {"top": top, "bottom": bottom, "text": text}

    def _compute_sectors_local(self, latest: str, prev: str) -> dict:
        """回退方案：基于本地行情库 + 证监会行业分类（baostock），流通市值加权。"""
        self._ensure_industry_cache()
        sql = f"""
        WITH c AS (
            SELECT ind.industry AS sector,
                   (t.close - p.close) / p.close * 100 AS chg,
                   COALESCE(mc.circ_mv, t.turnover, 1) AS w
            FROM (SELECT symbol, close, turnover FROM stock_daily WHERE date = ?) t
            JOIN (SELECT symbol, close FROM stock_daily WHERE date = ?) p USING (symbol)
            JOIN {_INDUSTRY_TABLE} ind ON ind.symbol = t.symbol
            LEFT JOIN {_MARKET_CAP_TABLE} mc ON mc.symbol = t.symbol
        )
        SELECT sector,
               COUNT(*) AS cnt,
               ROUND(SUM(chg * w) / SUM(w), 2) AS avg_chg
        FROM c
        WHERE sector != '未分类'
        GROUP BY sector
        HAVING cnt >= 5
        ORDER BY avg_chg DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, (latest, prev)).fetchall()

        top = [{"name": r[0], "change_pct": r[2], "count": r[1]} for r in rows[:8]]
        bottom = [{"name": r[0], "change_pct": r[2], "count": r[1]} for r in rows[-8:][::-1]]
        themes = self._detect_themes(top[:5])
        text = self._sector_text(top[:5], bottom[:5], themes)
        return {"top": top, "bottom": bottom, "text": text}

    # 主题关键词 → 产业链名称（用于识别领涨板块的共同主线）
    _THEME_KEYWORDS: list[tuple[str, str]] = [
        ("半导体", "半导体产业链"),
        ("芯片", "半导体产业链"),
        ("集成电路", "半导体产业链"),
        ("分立器件", "半导体产业链"),
        ("光电子", "消费电子链"),
        ("光学", "消费电子链"),
        ("消费电子", "消费电子链"),
        ("面板", "消费电子链"),
        ("软件", "信创与软件"),
        ("计算机", "信创与软件"),
        ("人工智能", "AI 算力"),
        ("算力", "AI 算力"),
        ("煤炭", "周期资源品"),
        ("有色", "周期资源品"),
        ("钢铁", "周期资源品"),
        ("矿业", "周期资源品"),
        ("钼", "周期资源品"),
        ("锂", "周期资源品"),
        ("银行", "大金融"),
        ("证券", "大金融"),
        ("保险", "大金融"),
        ("医药", "医药生物"),
        ("医疗", "医药生物"),
        ("生物", "医药生物"),
        ("白酒", "大消费"),
        ("食品", "大消费"),
        ("零售", "大消费"),
        ("农业", "农业"),
        ("种", "农业"),
    ]

    # 产业链主题 → 事件催化语义（用于生成主线研判文本）
    _THEME_CATALYST: dict[str, str] = {
        "半导体产业链": "国产替代预期与龙头带动效应，形成强主线",
        "消费电子链": "消费电子复苏传闻与产业景气向上",
        "信创与软件": "自主可控政策预期与数字化转型需求",
        "AI 算力": "AI 算力需求扩张，光模块/服务器产业链景气度高企",
        "周期资源品": "全球需求走弱预期与库存压力，资金避险出逃",
        "大金融": "经济基本面偏弱压制估值，蓝筹板块表现疲弱",
        "大消费": "内需复苏不及预期，消费板块承压",
        "农业": "周期与农业板块，资金流出迹象明显",
    }

    def _detect_themes(self, sectors: list[dict]) -> list[str]:
        """根据板块名关键词推断所属产业链主题。"""
        themes: list[str] = []
        for s in sectors:
            name = s["name"]
            for kw, theme in self._THEME_KEYWORDS:
                if kw in name:
                    if theme not in themes:
                        themes.append(theme)
                    break
        return themes

    def _catalyst_note(self, themes: list[str]) -> str:
        """根据产业链主题生成事件催化语义。"""
        if not themes:
            return ""
        catalysts = []
        for theme in themes:
            note = self._THEME_CATALYST.get(theme)
            if note:
                catalysts.append(note)
                break  # 取最强主线的一条催化
        if not catalysts:
            return "具备龙头带动效应，形成强主线。"
        return catalysts[0] + "。"

    @staticmethod
    def _sector_text(top: list[dict], bottom: list[dict],
                     themes: list[str] | None = None, catalyst: str = "") -> str:
        if not top:
            return "暂无板块数据。"
        lead = "、".join(f"{s['name']}（{s['change_pct']:+.2f}%）" for s in top[:3])
        lag = "、".join(f"{s['name']}（{s['change_pct']:+.2f}%）" for s in bottom[:3]) if bottom else "无"
        # 主线研判：产业链归属 + 事件催化
        theme_note = ""
        if themes:
            if len(themes) <= 2:
                theme_note = f"三者均属于{'及'.join(themes)}，"
            else:
                theme_note = f"涉及{'、'.join(themes)}等多条主线，资金聚焦明显。"
        if catalyst and len(themes) <= 2:
            theme_note += f"具备事件催化（{catalyst.rstrip('。')}）。"
        elif len(themes) <= 2:
            theme_note += "具备龙头带动效应，形成强主线。"
        return (
            f"领涨板块高度集中：{lead}。{theme_note}"
            f"领跌方面：{lag}，多为周期或防御板块，资金存在流出迹象，但暂未形成扩散风险。"
        )

    # ------------------------------------------------------------------
    # 4. 资金与情绪
    # ------------------------------------------------------------------
    def _compute_market_breadth(self, latest: str) -> dict:
        """计算市场宽度/情绪子指标：新高新低、均线占比、ADL，全部基于本地 stock_daily。"""
        with sqlite3.connect(self.db_path) as conn:
            # 取近 252 个交易日（52 周）和近 60 个交易日
            dates_252 = [r[0] for r in conn.execute(
                "SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT 252"
            ).fetchall()]
            if len(dates_252) < 252:
                return {}
            year_start = dates_252[-1]

            # 1. 新高新低（NH-NL）：当日 high/low 是否触及 52 周极值
            nh_nl = conn.execute("""
                WITH range AS (
                    SELECT symbol, MAX(high) AS max_h, MIN(low) AS min_l
                    FROM stock_daily WHERE date >= ? AND date <= ? GROUP BY symbol
                ),
                cur AS (SELECT symbol, high, low FROM stock_daily WHERE date = ?)
                SELECT
                    SUM(CASE WHEN c.high >= r.max_h THEN 1 ELSE 0 END) AS nh,
                    SUM(CASE WHEN c.low <= r.min_l THEN 1 ELSE 0 END) AS nl
                FROM cur c JOIN range r USING(symbol)
            """, (year_start, latest, latest)).fetchone()
            nh, nl = (nh_nl[0] or 0), (nh_nl[1] or 0)

            # 2. 站上均线比例（MA20 / MA60）
            ma20_dates = [r[0] for r in conn.execute(
                "SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT 20"
            ).fetchall()]
            ma60_dates = dates_252[:60]
            ma20_start, ma60_start = ma20_dates[-1], ma60_dates[-1]

            def _above_ma(start_date: str) -> tuple[int, int]:
                row = conn.execute("""
                    WITH cur AS (SELECT symbol, close FROM stock_daily WHERE date = ?),
                         ma AS (SELECT symbol, AVG(close) AS val FROM stock_daily
                                WHERE date >= ? AND date <= ? GROUP BY symbol)
                    SELECT COUNT(*) FROM cur JOIN ma USING(symbol) WHERE cur.close > ma.val
                """, (latest, start_date, latest)).fetchone()
                total = conn.execute("SELECT COUNT(*) FROM stock_daily WHERE date = ?", (latest,)).fetchone()[0]
                return row[0] or 0, total or 1

            ma20_above, total = _above_ma(ma20_start)
            ma60_above, _ = _above_ma(ma60_start)

            # 3. ADL 腾落指数（近 20 日累积）
            adl_dates = dates_252[:21]
            adl_dates.reverse()
            adl_daily: list[int] = []
            for i in range(1, len(adl_dates)):
                today, prev = adl_dates[i], adl_dates[i - 1]
                net = conn.execute("""
                    SELECT SUM(CASE WHEN t.close > p.close THEN 1 ELSE 0 END) -
                           SUM(CASE WHEN t.close < p.close THEN 1 ELSE 0 END)
                    FROM (SELECT symbol, close FROM stock_daily WHERE date = ?) t
                    JOIN (SELECT symbol, close FROM stock_daily WHERE date = ?) p USING(symbol)
                """, (today, prev)).fetchone()[0]
                adl_daily.append(net or 0)
            adl_cumul = sum(adl_daily)
            adl_recent5 = sum(adl_daily[-5:]) / 5 if len(adl_daily) >= 5 else 0

        return {
            "nh": nh, "nl": nl, "nh_nl": nh - nl,
            "ma20_above": ma20_above, "ma20_total": total,
            "ma20_pct": round(ma20_above / total * 100, 1) if total else 0,
            "ma60_above": ma60_above, "ma60_total": total,
            "ma60_pct": round(ma60_above / total * 100, 1) if total else 0,
            "adl_cumul": adl_cumul, "adl_recent5": round(adl_recent5, 0),
            "adl_today": adl_daily[-1] if adl_daily else 0,
        }

    def _sentiment_score(self, b: dict, sb: dict) -> tuple[int, str]:
        """恐贪指数：融合趋势强度 + 短期情绪极端 + 中期结构，合成 0-100 反向情绪分。

        与 signal_score（纯盘面顺势强度）的区别：
          - signal_score 测"今天多强"（顺势，越高越该进攻）
          - fear_greed 测"情绪温度"（>75极度贪婪=见顶警告，<25极度恐惧=见底机会）
        故加入逆向成分（短期超买超卖）+ 独立维度（ADL中期趋势），降低与signal_score的共线。
        """
        # 1. 趋势成分（20%）：MA20占比，市场中线趋势
        s_trend = sb.get("ma20_pct", 50.0) if sb else 50.0

        # 2. 短期情绪极端（35%，逆向核心）：平均涨跌幅非线性映射
        # avg_change > +2.5% = 极度贪婪（超买风险），< -2.5% = 极度恐惧（超卖机会）
        avg_chg = b.get("avg_change", 0)
        s_extreme = max(0.0, min(100.0, 50.0 + avg_chg * 10.0))

        # 3. NH-NL 结构（20%）：净新高占比，反映突破/破位结构
        if sb and sb.get("nh_nl") is not None:
            decided = b["up"] + b["down"]
            nhnl_pct = sb["nh_nl"] / decided * 100 if decided else 0
            s_nhnl = max(0.0, min(100.0, (nhnl_pct + 10) / 20 * 100))
        else:
            s_nhnl = 50.0

        # 4. ADL 中期趋势（25%）：累积涨跌线近5日变化，独立于当日breadth
        # adl_recent5 > 0 = 中期资金净流入偏贪婪，< 0 = 中期流出偏恐惧
        adl_r5 = sb.get("adl_recent5", 0) if sb else 0
        s_adl = max(0.0, min(100.0, 50.0 + adl_r5 * 0.1))

        score = round(0.20 * s_trend + 0.35 * s_extreme + 0.20 * s_nhnl + 0.25 * s_adl)
        if score >= 75:
            label = "极度贪婪"
        elif score >= 60:
            label = "贪婪"
        elif score >= 40:
            label = "中性"
        elif score >= 25:
            label = "恐惧"
        else:
            label = "极度恐惧"
        return score, label

    def _build_sentiment(self, b: dict, indices: list[dict], sectors: dict,
                         turnover_stats: dict | None = None,
                         sentiment_breadth: dict | None = None) -> dict:
        lu, ld = b["limit_up"], b["limit_down"]
        ratio = f"{lu}:{ld}" if ld else f"{lu}:0"
        decided = b["up"] + b["down"]
        width = round(b["up"] / b["down"], 2) if b["down"] else 0

        # 恐贪指数
        fear_greed, fg_label = self._sentiment_score(b, sentiment_breadth or {})

        # 成交额纵向对比
        vol_part = f"两市成交额 {b['turnover_yi']:.0f} 亿元，{b['turnover_label']}"
        if turnover_stats and turnover_stats["avg5"] > 0:
            cur = b["turnover_yi"]
            pct5 = (cur - turnover_stats["avg5"]) / turnover_stats["avg5"] * 100
            trend = "放量" if pct5 > 5 else ("缩量" if pct5 < -5 else "持平")
            vol_part += f"，较近5日均值（{turnover_stats['avg5']:.0f} 亿）{trend} {pct5:+.0f}%，{'增量资金入场信号明确' if pct5 > 5 else ('资金有所收缩' if pct5 < -5 else '资金面相对平稳')}"
        text = (
            f"{vol_part}。"
            f"涨跌停比 {ratio}，短线资金活跃度{'极高' if lu - ld > 50 else '一般'}；"
            f"市场宽度（涨跌家数比）约 {width}:1，"
            f"整体赚钱效应{'尚可' if b['up_ratio'] >= 55 else '偏弱'}，"
            f"{'但分化明显，非主线个股赚钱难度较大。' if 45 <= b['up_ratio'] <= 62 else '需警惕结构性风险。'}"
        )

        result: dict = {
            "text": text,
            "limit_ratio": ratio,
            "breadth_width": width,
            "fear_greed_score": fear_greed,
            "fear_greed_label": fg_label,
        }
        if sentiment_breadth:
            result["breadth_indicators"] = sentiment_breadth
        return result

    # ------------------------------------------------------------------
    # 4.5 资金流向（主力资金 + 融资融券 + 北向资金）
    # ------------------------------------------------------------------
    def _fetch_fund_flow_em(self) -> dict:
        """从东方财富 push2delay 拉取主力资金净流入（行业板块 + 个股）。"""
        import requests

        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        base = "https://push2delay.eastmoney.com/api/qt/clist/get"

        def _fetch(fs: str, limit: int) -> list[dict]:
            params = {
                "pn": 1, "pz": limit, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                "fid": "f62", "fs": fs, "fields": "f12,f14,f62,f184,f3,f6",
            }
            r = requests.get(base, params=params, headers=headers, timeout=10)
            diff = (r.json().get("data") or {}).get("diff", []) or []
            return [
                {
                    "code": str(b["f12"]),
                    "name": str(b["f14"]),
                    "net_flow_yi": round(float(b.get("f62") or 0) / 1e8, 2),
                    "net_flow_pct": round(float(b.get("f184") or 0), 2),
                    "change_pct": round(float(b.get("f3") or 0), 2),
                }
                for b in diff
            ]

        result: dict = {}
        try:
            # 行业板块主力资金净流入 TOP10
            sectors_all = _fetch("m:90 t:2", 600)
            sectors_sorted = sorted(sectors_all, key=lambda x: x["net_flow_yi"], reverse=True)
            result["sector_inflow"] = sectors_sorted[:8]
            result["sector_outflow"] = list(reversed(sectors_sorted[-8:]))
        except Exception as exc:
            logger.warning(f"行业主力资金获取失败：{exc}")
            result["sector_inflow"] = []
            result["sector_outflow"] = []

        try:
            # 个股主力资金净流入 TOP10
            stocks_all = _fetch("m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048", 200)
            stocks_sorted = sorted(stocks_all, key=lambda x: x["net_flow_yi"], reverse=True)
            result["stock_inflow"] = stocks_sorted[:10]
        except Exception as exc:
            logger.warning(f"个股主力资金获取失败：{exc}")
            result["stock_inflow"] = []

        return result

    def _fetch_margin_data(self) -> dict:
        """获取融资融券余额及近期变化趋势（杠杆资金方向）。"""
        import akshare as ak

        df = ak.stock_margin_account_info()
        df = df.sort_values("日期").tail(6).reset_index(drop=True)  # 近6个交易日
        if len(df) < 2:
            return {}
        latest_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        rq_balance = float(latest_row["融资余额"])    # 单位：亿元
        rq_prev = float(prev_row["融资余额"])
        rq_change = round(rq_balance - rq_prev, 1)
        rq_change_pct = round((rq_balance - rq_prev) / rq_prev * 100, 2)

        # 近5日融资余额序列（单位：亿元，用于趋势判断）
        rq_series = [round(float(x), 0) for x in df["融资余额"].tolist()]

        trend = "持续加杠杆" if all(rq_series[i] >= rq_series[i - 1] for i in range(1, len(rq_series))) else \
                ("持续降杠杆" if all(rq_series[i] <= rq_series[i - 1] for i in range(1, len(rq_series))) else "震荡")

        return {
            "rq_balance": round(rq_balance, 0),                   # 亿元
            "rq_change": round(rq_change, 1),
            "rq_change_pct": rq_change_pct,
            "rq_series": rq_series,
            "trend": trend,
            "date": str(latest_row["日期"]),
        }

    def _build_fund_flow(self, latest: str) -> dict:
        """组装资金流向数据：主力资金 + 融资融券 + 北向资金提示。"""
        em_data = self._fetch_fund_flow_em()

        # 融资融券
        try:
            margin = self._fetch_margin_data()
        except Exception as exc:
            logger.warning(f"融资融券数据获取失败：{exc}")
            margin = {}

        # 生成研判文本
        parts = []
        si = em_data.get("sector_inflow", [])
        if si:
            lead = si[0]
            parts.append(f"主力资金主要流入 {lead['name']}（净流入 {lead['net_flow_yi']:+.1f} 亿），资金聚焦主线方向")
        so = em_data.get("sector_outflow", [])
        if so:
            lag = so[0]
            parts.append(f"主力资金流出 {lag['name']}（净流出 {lag['net_flow_yi']:.1f} 亿）")
        if margin:
            direction = "增加" if margin["rq_change"] > 0 else "减少"
            parts.append(
                f"融资余额 {margin['rq_balance']:.0f} 亿元，较前日{direction} {abs(margin['rq_change']):.1f} 亿"
                f"（{margin['trend']}）"
            )
        parts.append("北向资金实时净流入数据自 2024 年 8 月起已停止披露，暂不可用。")

        text = "；".join(parts) + "。"

        return {
            "text": text,
            "sector_inflow": em_data.get("sector_inflow", []),
            "sector_outflow": em_data.get("sector_outflow", []),
            "stock_inflow": em_data.get("stock_inflow", []),
            "margin": margin,
            "north_note": "北向资金实时净流入数据自2024年8月起已停止披露",
        }

    # ------------------------------------------------------------------
    # 5. 消息催化
    # ------------------------------------------------------------------
    # 新闻关键词 → 主题分类（用于过滤和归类财经新闻）
    _NEWS_KEYWORDS: list[tuple[str, str]] = [
        ("半导体", "半导体"), ("芯片", "半导体"), ("集成电路", "半导体"),
        ("国产替代", "半导体"), ("光刻", "半导体"), ("EDA", "半导体"),
        ("AI", "人工智能"), ("人工智能", "人工智能"), ("算力", "人工智能"),
        ("大模型", "人工智能"), ("智能", "人工智能"),
        ("消费电子", "消费电子"), ("手机", "消费电子"), ("面板", "消费电子"),
        ("新能源", "新能源"), ("光伏", "新能源"), ("锂电", "新能源"), ("储能", "新能源"),
        ("军工", "军工"), ("国防", "军工"),
        ("降息", "货币政策"), ("降准", "货币政策"), ("MLF", "货币政策"), ("LPR", "货币政策"),
        ("PMI", "宏观经济"), ("社融", "宏观经济"), ("GDP", "宏观经济"), ("CPI", "宏观经济"),
        ("美联储", "海外"), ("美股", "海外"), ("关税", "海外"), ("制裁", "海外"),
        ("房地产", "房地产"), ("楼市", "房地产"),
        ("医药", "医药"), ("医疗", "医药"), ("集采", "医药"),
        ("稀土", "资源品"), ("有色", "资源品"), ("煤炭", "资源品"), ("原油", "资源品"),
    ]

    def _fetch_news(self, latest: str) -> list[dict]:
        """抓取当日财经快讯，返回过滤后的结构化新闻列表。"""
        import akshare as ak

        all_news: list[dict] = []
        # 1. 东财全球财经快讯
        try:
            df = ak.stock_info_global_em()
            latest_short = latest.replace("-", "")[4:]  # MMDD
            for _, row in df.iterrows():
                title = str(row.get("标题", ""))
                pub_time = str(row.get("发布时间", ""))
                # 只取当天相关新闻
                if latest in pub_time or pub_time[:10] == latest:
                    matched = self._classify_news(title)
                    all_news.append({
                        "title": title,
                        "summary": str(row.get("摘要", ""))[:120],
                        "time": pub_time,
                        "url": str(row.get("链接", "")),
                        "theme": matched,
                    })
        except Exception as exc:
            logger.warning(f"财经快讯获取失败：{exc}")

        # 2. 新闻联播（政策面）
        try:
            df = ak.news_cctv(date=latest.replace("-", ""))
            for _, row in df.iterrows():
                title = str(row.get("title", ""))
                content = str(row.get("content", ""))[:120]
                matched = self._classify_news(title) or self._classify_news(content)
                if matched:  # 只保留与市场相关的政策
                    all_news.append({
                        "title": f"[政策] {title}",
                        "summary": content,
                        "time": str(row.get("date", "")),
                        "url": "",
                        "theme": matched,
                        "source": "新闻联播",
                    })
        except Exception as exc:
            logger.warning(f"新闻联播获取失败：{exc}")

        return all_news

    def _classify_news(self, text: str) -> str:
        """根据关键词将新闻归类到主题，返回主题名或空字符串。"""
        for kw, theme in self._NEWS_KEYWORDS:
            if kw in text:
                return theme
        return ""

    def _build_news(self, sectors: dict, latest: str) -> dict:
        """消息催化：聚合实时新闻 + 板块关联分析。"""
        top = sectors.get("top", [])
        themes_from_sectors = self._detect_themes(top[:3])

        news_items = self._fetch_news(latest)

        # 统计各主题新闻数量
        theme_counts: dict[str, int] = {}
        themed_news: dict[str, list[dict]] = {}
        for item in news_items:
            theme = item.get("theme", "")
            if theme:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
                themed_news.setdefault(theme, []).append(item)

        # 找催化最集中的主题
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        top_themes = [t for t, _ in sorted_themes[:3]]

        # 判断与领涨板块的关联度
        sector_news: list[dict] = []
        matched_themes = []
        for sector_theme in themes_from_sectors:
            # 产业链主题 → 新闻主题映射
            news_theme_map = {
                "半导体产业链": "半导体", "消费电子链": "消费电子", "信创与软件": "人工智能",
                "AI 算力": "人工智能", "周期资源品": "资源品", "大金融": "宏观经济",
                "大消费": "宏观经济", "农业": "宏观经济", "医药生物": "医药",
            }
            nt = news_theme_map.get(sector_theme)
            if nt and nt in theme_counts:
                matched_themes.append(nt)
                for item in themed_news[nt][:3]:
                    sector_news.append(item)

        # 无主题的新闻取综合要闻 TOP5
        general_news = [n for n in news_items if not n.get("theme")][:5]

        # 生成研判文本
        parts = []
        if sorted_themes:
            theme_str = "、".join(f"{t}（{c}条）" for t, c in sorted_themes[:3])
            parts.append(f"今日新闻催化主要集中在：{theme_str}")
        if matched_themes:
            parts.append(f"与领涨主线（{'、'.join(set(matched_themes))}）高度相关，消息面支撑主线延续")
        else:
            parts.append("暂无明显与领涨主线直接相关的新闻催化")
        parts.append(f"共抓取{len(news_items)}条当日财经新闻")
        text = "；".join(parts) + "。"

        return {
            "items": news_items[:15],
            "themed_news": themed_news,
            "sector_news": sector_news[:8],
            "general_news": general_news,
            "theme_counts": theme_counts,
            "top_themes": top_themes,
            "text": text,
        }

    # ------------------------------------------------------------------
    # 6. 明日交易计划
    # ------------------------------------------------------------------
    def _build_plan(self, signal: int, indices: list[dict], sectors: dict, turnover_stats: dict | None = None) -> dict:
        if signal >= 55:
            action, position = "进攻", "7-8 成"
        elif signal >= 45:
            action, position = "均衡", "5-6 成"
        else:
            action, position = "防守", "3-4 成"

        top = sectors.get("top", [])
        bottom = sectors.get("bottom", [])
        focus = [s["name"] for s in top[:3]] or ["暂无明显主线"]
        avoid = [s["name"] for s in bottom[:3]] or ["无明确回避方向"]

        strong_idx = next((i for i in indices if i["change_pct"] >= 2), None)
        focus_note = f"关注 {strong_idx['name']} 能否延续强势" if strong_idx else ""

        # 量化失效条件：成交额阈值 + 龙头板块炸板
        trigger_parts = []
        if turnover_stats and turnover_stats["avg5"] > 0:
            threshold = round(turnover_stats["avg5"] * 0.85 / 10000, 2)  # 近5日均值的85%，亿→万亿
            trigger_parts.append(f"明日两市成交额萎缩至 {threshold:.2f} 万亿以下")
        else:
            trigger_parts.append("明日成交额显著萎缩")
        if top:
            trigger_parts.append(f"{top[0]['name']} 龙头出现炸板")
        trigger_note = "或".join(trigger_parts)

        return {
            "action": action,
            "position": position,
            "focus": focus,
            "avoid": avoid,
            "note": focus_note,
            "trigger": trigger_note,
            "text": (
                f"结论：{action}。仓位可维持/提升至 {position}。"
                f"核心关注方向：{'、'.join(focus)} 等主线龙头{('；' + focus_note + '。') if focus_note else '。'}"
                f"回避方向：{'、'.join(avoid)} 等弱势板块。"
                f"触发失效条件：若{trigger_note}，则转为均衡，减仓至5成。"
            ),
        }

    # ------------------------------------------------------------------
    # 四、历史纵向对比
    # ------------------------------------------------------------------
    def _compute_history(self, latest: str, prev: str) -> dict:
        """历史纵向对比：信号评分趋势、成交额趋势、相似日匹配、季节性。"""
        result: dict = {}
        # 取近60个交易日
        with sqlite3.connect(self.db_path) as conn:
            dates = [r[0] for r in conn.execute(
                "SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT 61"
            ).fetchall()]
            dates.reverse()
            if len(dates) < 3:
                return {"error": "历史数据不足"}

            start_date = dates[0]

            # 一次性算出每日 breadth（用窗口函数 LAG 高效计算）
            daily = conn.execute("""
                WITH ranked AS (
                    SELECT date, symbol, close, turnover,
                           LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close
                    FROM stock_daily WHERE date >= ?
                )
                SELECT date,
                    SUM(CASE WHEN close > prev_close THEN 1 ELSE 0 END) AS up,
                    SUM(CASE WHEN close < prev_close THEN 1 ELSE 0 END) AS down,
                    SUM(CASE WHEN close = prev_close THEN 1 ELSE 0 END) AS flat,
                    COUNT(*) AS total,
                    SUM(turnover) AS turnover,
                    AVG((close - prev_close) / prev_close * 100) AS avg_chg
                FROM ranked WHERE prev_close IS NOT NULL
                GROUP BY date ORDER BY date
            """, (start_date,)).fetchall()

        # 涨跌停（逐日，按限幅近似）
        limit_daily = self._compute_limit_history(dates)

        # 组装趋势序列
        signal_series, turnover_series, up_ratio_series = [], [], []
        features: list[list[float]] = []
        feat_dates: list[str] = []
        for i, (d, up, down, flat, total, turnover, avg_chg) in enumerate(daily):
            up, down, total = up or 0, down or 0, total or 0
            decided = up + down
            up_ratio = round(up / decided * 100, 1) if decided else 0
            lim = limit_daily.get(d, {"lu": 0, "ld": 0})
            lu, ld = lim["lu"], lim["ld"]
            lim_total = lu + ld
            limit_ratio = (lu / lim_total * 100) if lim_total else 50
            chg_score = max(0.0, min(100.0, ((avg_chg or 0) + 5) / 10 * 100))
            score = round(0.45 * up_ratio + 0.30 * chg_score + 0.25 * limit_ratio)
            signal_series.append({"date": d, "value": score})
            turnover_series.append({"date": d, "value": round((turnover or 0) / 1e8, 0)})
            up_ratio_series.append({"date": d, "value": up_ratio})
            # 特征向量（归一化，用于相似日匹配）
            features.append([up_ratio / 100, min(1, max(0, (avg_chg or 0 + 5) / 10)), lu / 200, ld / 50])
            feat_dates.append(d)

        result["trends"] = {
            "signal": signal_series,
            "turnover": turnover_series,
            "up_ratio": up_ratio_series,
        }

        # 信号趋势判断（近5日 vs 近20日）
        recent5 = signal_series[-5:] if len(signal_series) >= 5 else signal_series
        avg5 = sum(s["value"] for s in recent5) / len(recent5)
        avg_total = sum(s["value"] for s in signal_series) / len(signal_series)
        result["trend_summary"] = {
            "signal_now": signal_series[-1]["value"] if signal_series else 0,
            "signal_avg5": round(avg5, 0),
            "signal_avg20": round(avg_total, 0),
            "warming": "升温" if avg5 > avg_total else ("退潮" if avg5 < avg_total else "持平"),
        }

        # 相似日匹配
        result["similar_days"] = self._find_similar_days(features, feat_dates, daily)

        # 季节性
        result["seasonality"] = self._compute_seasonality(conn_dates=dates)

        return result

    def _compute_limit_history(self, dates: list[str]) -> dict:
        """计算指定日期列表的每日涨跌停数。"""
        result: dict[str, dict] = {}
        if len(dates) < 2:
            return result
        date_pairs = list(zip(dates[1:], dates[:-1]))
        with sqlite3.connect(self.db_path) as conn:
            for today, yest in date_pairs:
                lu, ld = conn.execute("""
                    WITH c AS (
                        SELECT t.symbol, (t.close - p.close) / p.close * 100 AS chg
                        FROM (SELECT symbol, close FROM stock_daily WHERE date = ?) t
                        JOIN (SELECT symbol, close FROM stock_daily WHERE date = ?) p USING(symbol)
                    )
                    SELECT
                        SUM(CASE WHEN age > 5 AND chg >=  th THEN 1 ELSE 0 END) AS lu,
                        SUM(CASE WHEN age > 5 AND chg <= -th THEN 1 ELSE 0 END) AS ld
                    FROM (
                        SELECT c.*,
                               CASE
                                   WHEN c.symbol LIKE '30%' OR c.symbol LIKE '68%' THEN 19.5
                                   WHEN c.symbol LIKE '8%' OR c.symbol LIKE '4%' THEN 29.0
                                   WHEN COALESCE(b.name, '') LIKE '%ST%' THEN 4.6
                                   ELSE 9.7
                               END AS th,
                               julianday(?) - julianday(COALESCE(b.ipo_date, '2000-01-01')) AS age
                        FROM c LEFT JOIN stock_basic b ON b.symbol = c.symbol
                    )
                """, (today, yest, today)).fetchone()
                result[today] = {"lu": lu or 0, "ld": ld or 0}
        return result

    def _find_similar_days(self, features: list[list[float]], dates: list[str],
                           daily: list) -> list[dict]:
        """用欧氏距离找今日最相似的历史交易日，统计次日表现。"""
        if len(features) < 2:
            return []
        today_feat = features[-1]
        # 计算历史各日与今日的距离（排除今日）
        scored: list[tuple[float, str]] = []
        for i in range(len(features) - 1):
            dist = sum((features[i][j] - today_feat[j]) ** 2 for j in range(len(today_feat))) ** 0.5
            scored.append((dist, dates[i]))
        scored.sort()

        # 取最相似5天，查次日涨跌
        similar: list[dict] = []
        for dist, d in scored[:5]:
            # 找 d 在 daily 中的索引，取次日 avg_chg
            idx = next((i for i, row in enumerate(daily) if row[0] == d), None)
            next_chg = None
            if idx is not None and idx + 1 < len(daily):
                next_chg = round(daily[idx + 1][6] or 0, 2)  # avg_chg of next day
            similar.append({"date": d, "distance": round(dist, 4), "next_day_avg_chg": next_chg})
        # 相似日统计
        next_chgs = [s["next_day_avg_chg"] for s in similar if s["next_day_avg_chg"] is not None]
        win_rate = round(sum(1 for c in next_chgs if c > 0) / len(next_chgs) * 100, 0) if next_chgs else None
        avg_next = round(sum(next_chgs) / len(next_chgs), 2) if next_chgs else None
        return {"days": similar, "next_day_win_rate": win_rate, "next_day_avg_chg": avg_next}

    def _compute_seasonality(self, conn_dates: list[str]) -> dict:
        """计算月度胜率与季节性效应（基于全市场个股涨跌统计）。"""
        with sqlite3.connect(self.db_path) as conn:
            daily_all = conn.execute("""
                WITH ranked AS (
                    SELECT date, symbol, close,
                           LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev
                    FROM stock_daily
                )
                SELECT substr(date, 6, 2) AS mon,
                       SUM(CASE WHEN close > prev THEN 1 ELSE 0 END) AS up,
                       SUM(CASE WHEN close < prev THEN 1 ELSE 0 END) AS dn,
                       COUNT(*) AS total
                FROM ranked WHERE prev IS NOT NULL AND date >= '2024-01-01'
                GROUP BY substr(date, 6, 2) ORDER BY mon
            """).fetchall()

        season_data = []
        for mon, up, dn, total in daily_all:
            ud = (up or 0) + (dn or 0)
            win = round((up or 0) / ud * 100, 1) if ud else 0
            season_data.append({"month": mon, "win_rate": win, "up": up or 0, "down": dn or 0})
        return {"monthly": season_data}

    # ------------------------------------------------------------------
    # 五、估值与宏观
    # ------------------------------------------------------------------
    def _fetch_valuation(self) -> dict:
        """指数估值分位 + 股权风险溢价 ERP。"""
        import akshare as ak

        result: dict = {"indices": [], "erp": {}}
        # 指数PE/PB
        idx_map = {"沪深300": "沪深300", "上证50": "上证50", "创业板指": "创业板指"}
        for display, ak_name in idx_map.items():
            try:
                df = ak.stock_index_pe_lg(symbol=ak_name)
                pe_col = "滚动市盈率" if "滚动市盈率" in df.columns else df.columns[2]
                pe_series = pd.to_numeric(df[pe_col], errors="coerce").dropna()
                latest_pe = round(float(pe_series.iloc[-1]), 2)
                percentile = round(float((pe_series < latest_pe).sum() / len(pe_series) * 100), 1)
                valuation = "低估" if percentile < 30 else ("高估" if percentile > 70 else "合理")
                result["indices"].append({
                    "name": display, "pe": latest_pe,
                    "pe_percentile": percentile, "valuation": valuation,
                })
            except Exception as exc:
                logger.warning(f"{display}估值获取失败：{exc}")

        # ERP：沪深300盈利收益率 - 10年期国债收益率
        try:
            from datetime import timedelta
            _now = datetime.now()
            bond_df = ak.bond_china_yield(
                start_date=(_now - timedelta(days=30)).strftime("%Y%m%d"),
                end_date=_now.strftime("%Y%m%d"),
            )
            bond_row = bond_df[bond_df["曲线名称"] == "中债国债收益率曲线"]
            treasury_10y = float(bond_row.iloc[-1]["10年"]) if len(bond_row) > 0 else 2.0
            hs300 = next((i for i in result["indices"] if i["name"] == "沪深300"), None)
            if hs300:
                earnings_yield = 100 / hs300["pe"]  # 盈利收益率
                erp = round(earnings_yield - treasury_10y, 2)
                level = "股优于债" if erp > 2 else ("债优于股" if erp < 0 else "股债均衡")
                result["erp"] = {
                    "earnings_yield": round(earnings_yield, 2),
                    "treasury_10y": treasury_10y,
                    "erp": erp, "level": level,
                }
        except Exception as exc:
            logger.warning(f"ERP计算失败：{exc}")

        return result

    def _fetch_macro(self) -> dict:
        """宏观数据：PMI、M2、社融。"""
        import akshare as ak

        result: dict = {}
        # PMI
        try:
            df = ak.macro_china_pmi()
            latest = df.iloc[0]
            pmi_val = float(latest["制造业-指数"])
            result["pmi"] = {
                "month": str(latest["月份"]),
                "value": pmi_val,
                "yoy": round(float(latest["制造业-同比增长"]), 2),
                "signal": "扩张" if pmi_val >= 50 else "收缩",
            }
        except Exception as exc:
            logger.warning(f"PMI获取失败：{exc}")

        # M2
        try:
            df = ak.macro_china_money_supply()
            latest = df.iloc[0]
            result["m2"] = {
                "month": str(latest["月份"]),
                "yoy": round(float(latest["货币和准货币(M2)-同比增长"]), 2),
            }
        except Exception as exc:
            logger.warning(f"M2获取失败：{exc}")

        # 社融
        try:
            df = ak.macro_china_shrzgm()
            latest = df.iloc[0]
            result["social_financing"] = {
                "month": str(latest["月份"]),
                "value": float(latest["社会融资规模增量"]),
            }
        except Exception as exc:
            logger.warning(f"社融获取失败：{exc}")

        return result

    # ------------------------------------------------------------------
    # 六、涨跌停结构 + 龙虎榜
    # ------------------------------------------------------------------
    def _compute_limit_structure(self, latest: str, prev: str) -> dict:
        """涨停板结构分析：首板/连板/梯队高度。"""
        with sqlite3.connect(self.db_path) as conn:
            # 近5日涨停股集合（按日）
            dates_5 = [r[0] for r in conn.execute(
                "SELECT DISTINCT date FROM stock_daily ORDER BY date DESC LIMIT 6"
            ).fetchall()]
            dates_5.reverse()

            # 每日涨停股集合
            daily_lu: dict[str, set] = {}
            for i in range(1, len(dates_5)):
                today, yest = dates_5[i], dates_5[i - 1]
                rows = conn.execute("""
                    SELECT symbol FROM (
                        SELECT t.symbol,
                               (t.close - p.close) / p.close * 100 AS chg,
                               CASE
                                   WHEN t.symbol LIKE '30%' OR t.symbol LIKE '68%' THEN 19.5
                                   WHEN t.symbol LIKE '8%' OR t.symbol LIKE '4%' THEN 29.0
                                   WHEN COALESCE(b.name, '') LIKE '%ST%' THEN 4.6
                                   ELSE 9.7
                               END AS th,
                               julianday(?) - julianday(COALESCE(b.ipo_date, '2000-01-01')) AS age
                        FROM (SELECT symbol, close FROM stock_daily WHERE date = ?) t
                        JOIN (SELECT symbol, close FROM stock_daily WHERE date = ?) p USING(symbol)
                        LEFT JOIN stock_basic b ON b.symbol = t.symbol
                    )
                    WHERE age > 5 AND chg >= th
                """, (today, today, yest)).fetchall()
                daily_lu[today] = {r[0] for r in rows}

            today_lu = daily_lu.get(latest, set())
            yest_lu = daily_lu.get(dates_5[-2], set()) if len(dates_5) >= 2 else set()

            # 首板：今日涨停但昨日未涨停
            first_board = today_lu - yest_lu
            # 连板：今日涨停且昨日也涨停
            consecutive = today_lu & yest_lu

            # 计算连板梯队：对连板股追踪连续天数
            echelon: dict[int, int] = {}
            for sym in consecutive:
                streak = 1
                for i in range(len(dates_5) - 2, 0, -1):
                    if sym in daily_lu.get(dates_5[i], set()):
                        streak += 1
                    else:
                        break
                echelon[streak] = echelon.get(streak, 0) + 1

            # 最高连板高度
            max_height = max(echelon.keys()) if echelon else 0

            # 炸板率：触及涨停(high 涨幅达标) − 封板(close 涨幅达标)
            broken = conn.execute("""
                SELECT
                    SUM(CASE WHEN age > 5 AND hi  >=  th THEN 1 ELSE 0 END) AS touched,
                    SUM(CASE WHEN age > 5 AND chg >=  th THEN 1 ELSE 0 END) AS sealed
                FROM (
                    SELECT t.symbol,
                           (t.high  - p.close) / p.close * 100 AS hi,
                           (t.close - p.close) / p.close * 100 AS chg,
                           CASE
                               WHEN t.symbol LIKE '30%' OR t.symbol LIKE '68%' THEN 19.5
                               WHEN t.symbol LIKE '8%' OR t.symbol LIKE '4%' THEN 29.0
                               WHEN COALESCE(b.name, '') LIKE '%ST%' THEN 4.6
                               ELSE 9.7
                           END AS th,
                           julianday(?) - julianday(COALESCE(b.ipo_date, '2000-01-01')) AS age
                    FROM (SELECT symbol, close, high FROM stock_daily WHERE date = ?) t
                    JOIN (SELECT symbol, close FROM stock_daily WHERE date = ?) p USING(symbol)
                    LEFT JOIN stock_basic b ON b.symbol = t.symbol
                )
            """, (latest, latest, prev)).fetchone()
            touched_up, sealed_up = broken[0] or 0, broken[1] or 0
            broken_up = touched_up - sealed_up
            broken_rate = round(broken_up / touched_up * 100, 1) if touched_up else 0.0

            # 跌停
            ld = conn.execute("""
                SELECT COUNT(*) FROM (
                    SELECT t.symbol,
                           (t.close - p.close) / p.close * 100 AS chg,
                           CASE
                               WHEN t.symbol LIKE '30%' OR t.symbol LIKE '68%' THEN 19.5
                               WHEN t.symbol LIKE '8%' OR t.symbol LIKE '4%' THEN 29.0
                               WHEN COALESCE(b.name, '') LIKE '%ST%' THEN 4.6
                               ELSE 9.7
                           END AS th,
                           julianday(?) - julianday(COALESCE(b.ipo_date, '2000-01-01')) AS age
                    FROM (SELECT symbol, close FROM stock_daily WHERE date = ?) t
                    JOIN (SELECT symbol, close FROM stock_daily WHERE date = ?) p USING(symbol)
                    LEFT JOIN stock_basic b ON b.symbol = t.symbol
                ) WHERE age > 5 AND chg <= -th
            """, (latest, latest, prev)).fetchone()[0]

        echelon_list = [{"height": k, "count": v} for k, v in sorted(echelon.items(), reverse=True)]
        return {
            "limit_up_total": len(today_lu),
            "limit_down_total": ld or 0,
            "first_board": len(first_board),
            "consecutive": len(consecutive),
            "max_height": max_height,
            "echelon": echelon_list,
            "touched_up": touched_up,
            "broken_up": broken_up,
            "broken_rate": broken_rate,
            "text": (
                f"涨停 {len(today_lu)} 家（首板 {len(first_board)} 家，连板 {len(consecutive)} 家），"
                f"最高连板高度 {max_height} 板；跌停 {ld or 0} 家。"
                f"盘中触及涨停 {touched_up} 家，炸板 {broken_up} 家，炸板率 {broken_rate}%。"
                f"连板梯队：{'、'.join(f'{k}板{v}家' for k, v in sorted(echelon.items(), reverse=True)) or '无'}。"
            ),
        }

    def _fetch_dragon_tiger(self) -> dict:
        """龙虎榜：游资席位、机构净买卖。"""
        import akshare as ak
        from datetime import datetime as _dt, timedelta as _td

        end = _dt.now().strftime("%Y%m%d")
        start = (_dt.now() - _td(days=5)).strftime("%Y%m%d")
        result: dict = {}

        # 龙虎榜个股明细
        try:
            df = ak.stock_lhb_detail_em(start_date=start, end_date=end)
            if len(df) > 0:
                # 取净买额最大的TOP10
                if "龙虎榜净买额" in df.columns:
                    df["净买额"] = pd.to_numeric(df["龙虎榜净买额"], errors="coerce")
                    top = df.nlargest(10, "净买额")
                    result["top_stocks"] = [
                        {"code": str(r["代码"]), "name": str(r["名称"]),
                         "net_buy": round(float(r["净买额"]) / 1e8, 2),
                         "change_pct": round(float(r.get("涨跌幅", 0)), 2) if "涨跌幅" in df.columns else 0}
                        for _, r in top.iterrows()
                    ]
                result["total_count"] = len(df)
        except Exception as exc:
            logger.warning(f"龙虎榜明细获取失败：{exc}")

        # 机构买卖统计
        try:
            df = ak.stock_lhb_jgmmtj_em(start_date=start, end_date=end)
            if len(df) > 0:
                # 机构净买入TOP5
                for col in df.columns:
                    if "净买入" in col or "净买额" in col:
                        df["_net"] = pd.to_numeric(df[col], errors="coerce")
                        top_inst = df.nlargest(5, "_net")
                        result["top_institutions"] = [
                            {"name": str(r.get("名称", "")),
                             "net_buy": round(float(r["_net"]) / 1e8, 2)}
                            for _, r in top_inst.iterrows()
                        ]
                        break
        except Exception as exc:
            logger.warning(f"龙虎榜机构统计获取失败：{exc}")

        result["text"] = (
            f"龙虎榜共 {result.get('total_count', 0)} 只个股上榜。"
            + (f"游资/机构重点关注：{result.get('top_stocks', [{}])[0].get('name', '')}。"
               if result.get("top_stocks") else "")
        )
        return result

    # ------------------------------------------------------------------
    # 7. 风险提示
    # ------------------------------------------------------------------
    def _build_risks(self, b: dict, indices: list[dict], sectors: dict) -> list[str]:
        risks: list[str] = []
        # 指数分化：某指数逆势下跌
        if indices:
            weak = [i for i in indices if i["change_pct"] < 0]
            strong = [i for i in indices if i["change_pct"] > 1]
            if weak and strong:
                names = "、".join(i["name"] for i in weak)
                risks.append(f"{names}逆势走弱，可能拖累大盘整体重心。")
            top_idx = max(indices, key=lambda x: x["change_pct"])
            if top_idx["change_pct"] >= 2.5:
                risks.append(f"{top_idx['name']}短期涨幅过大，存在获利盘回吐风险。")

        # 板块高度集中
        top = sectors.get("top", [])
        if top and top[0]["change_pct"] >= 5:
            risks.append(f"领涨板块 {top[0]['name']} 单日涨幅 {top[0]['change_pct']:+.1f}%，"
                         "资金过度集中，需警惕分歧后的回调。")

        # 涨跌停背离
        if b["limit_up"] > 0 and b["limit_down"] > b["limit_up"] * 0.3:
            risks.append(f"跌停家数达 {b['limit_down']} 家，亏钱效应扩散，需警惕情绪转弱。")

        if not risks:
            risks.append("当前盘面信号平稳，建议关注后续量能变化与主线延续性。")
        return risks
