"""大盘分析 - 盘面总览（MarketAnalyzer mixin 组件）。

由 :class:`sequoia_x.analysis.market.MarketAnalyzer` 多重继承组合，
不单独实例化；方法通过 ``self.db_path`` 等访问门面状态。
"""

from __future__ import annotations

import sqlite3

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class BreadthMixin:
    """盘面总览计算逻辑（MarketAnalyzer 的 mixin 组件）。"""

    def _compute_breadth(self, latest: str, prev: str) -> dict:
        sql = """
        WITH t AS (SELECT symbol, close, turnover, pct_chg FROM stock_daily WHERE date = ?),
             p AS (SELECT symbol, close FROM stock_daily WHERE date = ?)
        SELECT
            SUM(CASE WHEN t.close > p.close THEN 1 ELSE 0 END) AS up,
            SUM(CASE WHEN t.close < p.close THEN 1 ELSE 0 END) AS down,
            SUM(CASE WHEN t.close = p.close THEN 1 ELSE 0 END) AS flat,
            COUNT(*) AS total,
            SUM(t.turnover) AS turnover,
            AVG(COALESCE(t.pct_chg, (t.close - p.close) / p.close * 100)) AS avg_chg
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
                   COALESCE(t.pct_chg, (t.close - p.close) / p.close * 100) AS chg,
                   (t.high  - p.close) / p.close * 100 AS hi,
                   (t.low   - p.close) / p.close * 100 AS lo
            FROM (SELECT symbol, close, high, low, pct_chg FROM stock_daily WHERE date = ?) t
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
                    FROM (SELECT symbol, close, pct_chg FROM stock_daily WHERE date = ?) t
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
