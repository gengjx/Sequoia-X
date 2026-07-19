"""大盘分析 - 涨跌停结构与龙虎榜（MarketAnalyzer mixin 组件）。

由 :class:`sequoia_x.analysis.market.MarketAnalyzer` 多重继承组合，
不单独实例化；方法通过 ``self.db_path`` 等访问门面状态。
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class MicrostructureMixin:
    """涨跌停结构与龙虎榜计算逻辑（MarketAnalyzer 的 mixin 组件）。"""

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
                               COALESCE(t.pct_chg, (t.close - p.close) / p.close * 100) AS chg,
                               CASE
                                   WHEN t.symbol LIKE '30%' OR t.symbol LIKE '68%' THEN 19.5
                                   WHEN t.symbol LIKE '8%' OR t.symbol LIKE '4%' THEN 29.0
                                   WHEN COALESCE(b.name, '') LIKE '%ST%' THEN 4.6
                                   ELSE 9.7
                               END AS th,
                               julianday(?) - julianday(COALESCE(b.ipo_date, '2000-01-01')) AS age
                        FROM (SELECT symbol, close, pct_chg FROM stock_daily WHERE date = ?) t
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
                           COALESCE(t.pct_chg, (t.close - p.close) / p.close * 100) AS chg,
                           CASE
                               WHEN t.symbol LIKE '30%' OR t.symbol LIKE '68%' THEN 19.5
                               WHEN t.symbol LIKE '8%' OR t.symbol LIKE '4%' THEN 29.0
                               WHEN COALESCE(b.name, '') LIKE '%ST%' THEN 4.6
                               ELSE 9.7
                           END AS th,
                           julianday(?) - julianday(COALESCE(b.ipo_date, '2000-01-01')) AS age
                    FROM (SELECT symbol, close, high, pct_chg FROM stock_daily WHERE date = ?) t
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
                           COALESCE(t.pct_chg, (t.close - p.close) / p.close * 100) AS chg,
                           CASE
                               WHEN t.symbol LIKE '30%' OR t.symbol LIKE '68%' THEN 19.5
                               WHEN t.symbol LIKE '8%' OR t.symbol LIKE '4%' THEN 29.0
                               WHEN COALESCE(b.name, '') LIKE '%ST%' THEN 4.6
                               ELSE 9.7
                           END AS th,
                           julianday(?) - julianday(COALESCE(b.ipo_date, '2000-01-01')) AS age
                    FROM (SELECT symbol, close, pct_chg FROM stock_daily WHERE date = ?) t
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
        """龙虎榜：游资席位、机构净买卖。优先查本地 DB，无数据时 fallback akshare。"""
        result = self._fetch_dragon_tiger_local()
        if result.get("total_count", 0) > 0:
            return result
        # fallback：实时拉 akshare
        return self._fetch_dragon_tiger_remote()

    def _fetch_dragon_tiger_local(self) -> dict:
        """从本地 lhb_detail + lhb_seats 查龙虎榜（毫秒级，无网络依赖）。"""
        result: dict = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # 近5日龙虎榜个股东涛
                stocks = conn.execute(
                    """SELECT symbol, name, date, net_buy, pct_chg
                       FROM lhb_detail
                       WHERE date IN (
                           SELECT DISTINCT date FROM lhb_detail ORDER BY date DESC LIMIT 5
                       )
                       ORDER BY net_buy DESC LIMIT 10"""
                ).fetchall()
                if stocks:
                    result["top_stocks"] = [
                        {"code": r["symbol"], "name": r["name"],
                         "net_buy": round((r["net_buy"] or 0) / 1e8, 2),
                         "change_pct": round(r["pct_chg"] or 0, 2)}
                        for r in stocks
                    ]
                    result["total_count"] = conn.execute(
                        """SELECT COUNT(DISTINCT symbol) FROM lhb_detail WHERE date=(
                           SELECT MAX(date) FROM lhb_detail)"""
                    ).fetchone()[0]
                    # 席位净买入 TOP5（近5日汇总）
                    seats = conn.execute(
                        """SELECT seat_name, SUM(net_amount) AS net
                           FROM lhb_seats
                           WHERE date IN (
                               SELECT DISTINCT date FROM lhb_detail ORDER BY date DESC LIMIT 5
                           )
                           GROUP BY seat_name ORDER BY net DESC LIMIT 5"""
                    ).fetchall()
                    if seats:
                        result["top_institutions"] = [
                            {"name": r["seat_name"],
                             "net_buy": round((r["net"] or 0) / 1e8, 2)}
                            for r in seats
                        ]
        except Exception as exc:
            logger.debug(f"本地龙虎榜查询失败（表可能未创建）：{exc}")

        result["text"] = (
            f"龙虎榜共 {result.get('total_count', 0)} 只个股上榜。"
            + (f"游资/机构重点关注：{result.get('top_stocks', [{}])[0].get('name', '')}。"
               if result.get("top_stocks") else "")
        )
        return result

    def _fetch_dragon_tiger_remote(self) -> dict:
        """fallback：实时拉 akshare 龙虎榜（网络慢，仅在本地无数据时调用）。"""
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        import akshare as ak

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
