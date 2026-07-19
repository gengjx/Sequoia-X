"""大盘分析 - 板块主线（MarketAnalyzer mixin 组件）。

由 :class:`sequoia_x.analysis.market.MarketAnalyzer` 多重继承组合，
不单独实例化；方法通过 ``self.db_path`` 等访问门面状态。
"""

from __future__ import annotations

import sqlite3
import time

from sequoia_x.analysis.market_common import (
    _BOARD_TABLE,
    _INDUSTRY_TABLE,
    _MARKET_CAP_TABLE,
)
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class SectorsMixin:
    """板块主线计算逻辑（MarketAnalyzer 的 mixin 组件）。"""

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
                   COALESCE(t.pct_chg, (t.close - p.close) / p.close * 100) AS pct,
                   COALESCE(mc.circ_mv, t.turnover, 1) AS w
            FROM (SELECT symbol, close, turnover, pct_chg FROM stock_daily WHERE date = ?) t
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
                   COALESCE(t.pct_chg, (t.close - p.close) / p.close * 100) AS chg,
                   COALESCE(mc.circ_mv, t.turnover, 1) AS w
            FROM (SELECT symbol, close, turnover, pct_chg FROM stock_daily WHERE date = ?) t
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

    @staticmethod
    def fetch_sectors_realtime() -> list[dict]:
        """拉取东财行业板块实时涨幅榜（盘中可用）。

        Returns:
            [{code, name, change_pct, turnover_rate, up_count, down_count, leading_stock}, ...]
        """
        import requests

        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        result = []
        try:
            r = requests.get(
                "https://push2delay.eastmoney.com/api/qt/clist/get",
                params={
                    "pn": 1, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                    "fs": "m:90+t:2",  # 行业板块
                    "fields": "f12,f14,f2,f3,f8,f104,f105,f128,f136",
                },
                headers=headers, timeout=10,
            )
            data = r.json().get("data", {}).get("diff", [])
            for item in data:
                try:
                    result.append({
                        "code": str(item.get("f12", "")),
                        "name": str(item.get("f14", "")),
                        "price": float(item.get("f2", 0) or 0),
                        "change_pct": round(float(item.get("f3", 0) or 0), 2),
                        "turnover_rate": round(float(item.get("f8", 0) or 0), 2),
                        "up_count": int(item.get("f104", 0) or 0),
                        "down_count": int(item.get("f105", 0) or 0),
                        "leading_stock": str(item.get("f128", "")),
                    })
                except (ValueError, TypeError):
                    continue
            result.sort(key=lambda x: x["change_pct"], reverse=True)
        except Exception as e:
            logger.warning(f"板块实时行情拉取失败：{e!r}")
        return result
