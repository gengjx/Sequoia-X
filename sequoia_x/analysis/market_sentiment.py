"""大盘分析 - 资金与情绪（MarketAnalyzer mixin 组件）。

由 :class:`sequoia_x.analysis.market.MarketAnalyzer` 多重继承组合，
不单独实例化；方法通过 ``self.db_path`` 等访问门面状态。
"""

from __future__ import annotations

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class SentimentMixin:
    """资金与情绪计算逻辑（MarketAnalyzer 的 mixin 组件）。"""

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
