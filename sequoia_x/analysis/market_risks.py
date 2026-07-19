"""大盘分析 - 风险提示（MarketAnalyzer mixin 组件）。

由 :class:`sequoia_x.analysis.market.MarketAnalyzer` 多重继承组合，
不单独实例化；方法通过 ``self.db_path`` 等访问门面状态。
"""

from __future__ import annotations

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class RisksMixin:
    """风险提示计算逻辑（MarketAnalyzer 的 mixin 组件）。"""

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
