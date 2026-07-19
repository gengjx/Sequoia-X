"""大盘分析 - 历史纵向对比（MarketAnalyzer mixin 组件）。

由 :class:`sequoia_x.analysis.market.MarketAnalyzer` 多重继承组合，
不单独实例化；方法通过 ``self.db_path`` 等访问门面状态。
"""

from __future__ import annotations

import sqlite3

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class HistoryMixin:
    """历史纵向对比计算逻辑（MarketAnalyzer 的 mixin 组件）。"""

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
                    SELECT date, symbol, close, turnover, pct_chg,
                           LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close
                    FROM stock_daily WHERE date >= ?
                )
                SELECT date,
                    SUM(CASE WHEN close > prev_close THEN 1 ELSE 0 END) AS up,
                    SUM(CASE WHEN close < prev_close THEN 1 ELSE 0 END) AS down,
                    SUM(CASE WHEN close = prev_close THEN 1 ELSE 0 END) AS flat,
                    COUNT(*) AS total,
                    SUM(turnover) AS turnover,
                    AVG(COALESCE(pct_chg, (close - prev_close) / prev_close * 100)) AS avg_chg
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
                        SELECT t.symbol, COALESCE(t.pct_chg, (t.close - p.close) / p.close * 100) AS chg
                        FROM (SELECT symbol, close, pct_chg FROM stock_daily WHERE date = ?) t
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
