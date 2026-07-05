"""龙头策略：板块轮动中识别领涨龙头，A股短线核心赚钱模式。"""

import sqlite3

import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)

_BOARD_TABLE = "stock_board_em"
_MARKET_CAP_TABLE = "stock_market_cap"


class DragonHeadStrategy(BaseStrategy):
    """龙头选股策略。

    填补 sequoia-x 无"板块联动选股"缺口——现有策略都是独立扫描个股，
    完全没利用已有的 stock_board_em 细分板块数据。本策略捕捉板块轮动启动期
    的领涨龙头，是 A 股短线最核心的赚钱模式。

    两步筛选：
    Step 1 — 找领涨板块：全市场按市值加权计算细分板块涨幅，取 TOP 10
    Step 2 — 在领涨板块内选龙头：
        a. 个股涨幅 > 板块均值 + 2%（跑赢板块）
        b. 成交额 > 1 亿（流动性）
        c. 个股涨幅进入所在板块成分股 TOP 3（板块内领涨）
        d. 板块当日加权涨幅 > 2%（板块确有启动，排除弱势板块蹭热度）

    Attributes:
        webhook_key: 路由到 'dragon' 专属飞书机器人。
    """

    webhook_key: str = "dragon"
    _MIN_BARS: int = 2
    _TOP_BOARDS: int = 10  # 取领涨板块数
    _TOP_STOCKS_PER_BOARD: int = 3  # 每板块取龙头数

    def run(self) -> list[str]:
        """两步筛选：先算领涨板块，再选板块内龙头。"""
        # 取最新两个交易日
        latest, prev = self._latest_dates()
        if not latest or not prev:
            logger.warning("DragonHeadStrategy：行情日期不足")
            return []

        # Step 1：全市场板块加权涨幅排名
        boards = self._compute_board_ranking(latest, prev)
        if not boards:
            logger.warning("DragonHeadStrategy：无板块数据（需先构建 stock_board_em 缓存）")
            return []

        # 领涨 TOP 10 且涨幅 > 2%（确有启动）
        hot_boards = [b for b in boards[: self._TOP_BOARDS] if b["chg"] > 2.0]
        if not hot_boards:
            logger.info("DragonHeadStrategy：无板块加权涨幅 > 2% 的领涨板块")
            return []

        # Step 2：在领涨板块内选龙头
        selected = self._select_dragons(hot_boards, latest, prev)
        logger.info(f"DragonHeadStrategy 选出 {len(selected)} 只股票（{len(hot_boards)} 个领涨板块）")
        return selected

    # ------------------------------------------------------------------
    # 数据获取
    # ------------------------------------------------------------------
    def _latest_dates(self) -> tuple[str | None, str | None]:
        with sqlite3.connect(self.engine.db_path) as conn:
            row = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()
            latest = row[0] if row else None
            if not latest:
                return None, None
            row = conn.execute(
                "SELECT MAX(date) FROM stock_daily WHERE date < ?", (latest,)
            ).fetchone()
            prev = row[0] if row else None
        return latest, prev

    def _compute_board_ranking(self, latest: str, prev: str) -> list[dict]:
        """全市场细分板块市值加权涨幅排名（与 MarketAnalyzer 口径一致）。"""
        sql = f"""
        WITH chg AS (
            SELECT bd.board AS board,
                   (t.close - p.close) / p.close * 100 AS pct,
                   COALESCE(mc.circ_mv, t.turnover, 1) AS w
            FROM (SELECT symbol, close, turnover FROM stock_daily WHERE date = ?) t
            JOIN (SELECT symbol, close FROM stock_daily WHERE date = ?) p USING (symbol)
            JOIN {_BOARD_TABLE} bd ON bd.symbol = t.symbol
            LEFT JOIN {_MARKET_CAP_TABLE} mc ON mc.symbol = t.symbol
        )
        SELECT board,
               COUNT(*) AS cnt,
               SUM(w) AS total_w,
               ROUND(SUM(pct * w) / SUM(w), 2) AS w_chg
        FROM chg
        GROUP BY board
        HAVING cnt >= 5
        ORDER BY w_chg DESC
        """
        with sqlite3.connect(self.engine.db_path) as conn:
            rows = conn.execute(sql, (latest, prev)).fetchall()
        return [{"board": r[0], "cnt": r[1], "total_w": r[2], "chg": r[3]} for r in rows]

    def _select_dragons(self, hot_boards: list[dict], latest: str, prev: str) -> list[str]:
        """在领涨板块内选龙头：跑赢板块 + 成交额过亿 + 板块内 TOP3 涨幅。"""
        selected: list[str] = []
        seen: set[str] = set()

        with sqlite3.connect(self.engine.db_path) as conn:
            for b in hot_boards:
                board_name = b["board"]
                board_chg = b["chg"]
                # 板块成分股当日涨跌幅 + 成交额
                sql = """
                    SELECT t.symbol,
                           (t.close - p.close) / p.close * 100 AS pct,
                           t.turnover
                    FROM (SELECT symbol, close, turnover FROM stock_daily WHERE date = ?) t
                    JOIN (SELECT symbol, close FROM stock_daily WHERE date = ?) p USING (symbol)
                    JOIN stock_board_em bd ON bd.symbol = t.symbol
                    WHERE bd.board = ?
                """
                rows = conn.execute(sql, (latest, prev, board_name)).fetchall()
                if not rows:
                    continue

                # 过滤流动性 + 排序取 TOP3
                liquid = [(sym, pct, turn) for sym, pct, turn in rows if turn and turn > 100_000_000]
                liquid.sort(key=lambda x: x[1], reverse=True)

                for sym, pct, _ in liquid[: self._TOP_STOCKS_PER_BOARD]:
                    # 龙头条件：跑赢板块 2% 以上
                    if pct > board_chg + 2.0 and sym not in seen:
                        selected.append(sym)
                        seen.add(sym)

        return selected
