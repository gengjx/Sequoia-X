"""龙虎榜盘中预判：根据资金流向+涨跌幅+换手率，预判哪些股票可能上榜。

龙虎榜上榜条件（满足任一）：
  - 日涨跌幅偏离值 ≥ 7%
  - 日换手率 ≥ 20%
  - 日振幅 ≥ 15%

盘中用实时涨跌幅+换手率+主力净流入预判，盘后验证命中率。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LhbPrediction:
    symbol: str
    name: str = ""
    price: float = 0.0
    pct_chg: float = 0.0
    turnover_rate: float = 0.0
    main_net: float = 0.0
    main_pct: float = 0.0
    reasons: list[str] = field(default_factory=list)
    probability: str = "中"  # 高/中/低


def predict_lhb_candidates(db_path: str | None = None) -> list[LhbPrediction]:
    """预判今日可能上龙虎榜的股票。

    用最新资金流向 + 日K最新涨跌幅/换手率做预判。
    盘中数据来自 fund_flow 表（东财 push2delay）。

    Returns:
        预判上榜的股票列表（按概率排序）
    """
    settings = Settings()
    path = db_path or settings.db_path
    predictions: list[LhbPrediction] = []

    with sqlite3.connect(path) as conn:
        # 联合资金流向 + 日K最新行情
        rows = conn.execute("""
            SELECT ff.symbol,
                   COALESCE(b.name, '') AS name,
                   ff.main_net, ff.main_pct,
                   d.pct_chg, d.turn, d.close
            FROM fund_flow ff
            LEFT JOIN stock_basic b ON b.symbol = ff.symbol
            LEFT JOIN stock_daily d ON d.symbol = ff.symbol
                AND d.date = (SELECT MAX(date) FROM stock_daily)
            WHERE ff.date = (SELECT MAX(date) FROM fund_flow)
        """).fetchall()

    for r in rows:
        symbol, name, main_net, main_pct, pct_chg, turn, close = r
        pct_chg = pct_chg or 0
        turn = turn or 0
        main_net = main_net or 0

        reasons: list[str] = []
        hit_conditions = 0

        # 条件1：涨跌幅 ≥ 7%
        if abs(pct_chg) >= 7:
            reasons.append(f"涨跌幅{pct_chg:+.1f}%≥7%")
            hit_conditions += 1

        # 条件2：换手率 ≥ 20%
        if turn >= 20:
            reasons.append(f"换手率{turn:.1f}%≥20%")
            hit_conditions += 1

        # 条件3：主力净流入 > 1亿（大资金异动）
        if main_net > 1e8:
            reasons.append(f"主力净流入{main_net/1e4:.0f}万")
            hit_conditions += 1

        # 条件4：主力净流入占比 > 15%
        if (main_pct or 0) > 15:
            reasons.append(f"主力占比{main_pct:.1f}%")
            hit_conditions += 1

        if hit_conditions == 0:
            continue

        prob = "高" if hit_conditions >= 3 else ("中" if hit_conditions >= 2 else "低")
        if prob == "低":
            continue  # 只输出中概率以上

        predictions.append(LhbPrediction(
            symbol=symbol, name=name, price=close or 0,
            pct_chg=round(pct_chg, 2), turnover_rate=round(turn, 2),
            main_net=main_net, main_pct=round(main_pct or 0, 2),
            reasons=reasons, probability=prob,
        ))

    # 按概率+主力净流入排序
    prob_order = {"高": 0, "中": 1, "低": 2}
    predictions.sort(key=lambda x: (prob_order[x.probability], -x.main_net))

    logger.info(f"龙虎榜预判：{len(predictions)} 只可能上榜（高{sum(1 for p in predictions if p.probability=='高')} 中{sum(1 for p in predictions if p.probability=='中')}）")
    return predictions


def predict_to_list(predictions: list[LhbPrediction]) -> list[dict]:
    """转 dict 列表（供 API）。"""
    return [
        {
            "symbol": p.symbol, "name": p.name, "price": p.price,
            "pct_chg": p.pct_chg, "turnover_rate": p.turnover_rate,
            "main_net": round(p.main_net / 1e4, 0),  # 万元
            "main_pct": p.main_pct, "probability": p.probability,
            "reasons": p.reasons,
        }
        for p in predictions
    ]
