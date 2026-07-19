"""大盘分析 - 消息催化（MarketAnalyzer mixin 组件）。

由 :class:`sequoia_x.analysis.market.MarketAnalyzer` 多重继承组合，
不单独实例化；方法通过 ``self.db_path`` 等访问门面状态。
"""

from __future__ import annotations

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class NewsMixin:
    """消息催化计算逻辑（MarketAnalyzer 的 mixin 组件）。"""

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
