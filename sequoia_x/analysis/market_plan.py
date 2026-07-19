"""大盘分析 - 明日交易计划（MarketAnalyzer mixin 组件）。

由 :class:`sequoia_x.analysis.market.MarketAnalyzer` 多重继承组合，
不单独实例化；方法通过 ``self.db_path`` 等访问门面状态。
"""

from __future__ import annotations

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class PlanMixin:
    """明日交易计划计算逻辑（MarketAnalyzer 的 mixin 组件）。"""

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
