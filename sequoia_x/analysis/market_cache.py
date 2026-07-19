"""大盘分析 - 数据缓存管理（MarketAnalyzer mixin 组件）。

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
    _STOCK_BASIC_TABLE,
    _clean_industry_name,
)
from sequoia_x.core.logger import get_logger
from sequoia_x.core.rate_limiter import em_get

logger = get_logger(__name__)


class CacheMixin:
    """数据缓存管理计算逻辑（MarketAnalyzer 的 mixin 组件）。"""

    def _ensure_stock_basic_cache(self) -> None:
        """确保 stock_basic 缓存表存在；不存在则从 baostock 拉取股票名称与上市日。"""
        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (_STOCK_BASIC_TABLE,)
            ).fetchone()
            if exists:
                return

        logger.info("首次构建股票元数据缓存（baostock，约 10~20s）...")
        import baostock as bs

        bs.login()
        rows: list[tuple[str, str, str]] = []
        try:
            rs = bs.query_stock_basic(code_name="", code="")
            while rs.next():
                data = rs.get_row_data()
                if data[4] == "1":  # type == "1"：股票（含已退市，保留以覆盖历史回放）
                    symbol = data[0].split(".")[-1]
                    rows.append((symbol, data[1], data[2]))  # symbol, name, ipo_date
        finally:
            bs.logout()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_STOCK_BASIC_TABLE} ("
                "symbol TEXT PRIMARY KEY, name TEXT, ipo_date TEXT)"
            )
            conn.executemany(
                f"INSERT OR REPLACE INTO {_STOCK_BASIC_TABLE} (symbol, name, ipo_date) VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()
        logger.info(f"股票元数据缓存完成，共 {len(rows)} 条")

    def refresh_stock_basic_cache(self) -> int:
        """强制刷新股票元数据缓存，返回写入条数。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {_STOCK_BASIC_TABLE}")
            conn.commit()
        self._ensure_stock_basic_cache()
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {_STOCK_BASIC_TABLE}").fetchone()[0]

    # ------------------------------------------------------------------
    # 流通市值缓存（板块市值加权用；缓变，周期刷新）
    # ------------------------------------------------------------------

    def _ensure_market_cap_cache(self) -> None:
        """确保流通市值缓存表存在；不存在则从东财延迟行情拉取全市场快照。"""
        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (_MARKET_CAP_TABLE,)
            ).fetchone()
            if exists:
                return
        logger.info("首次构建流通市值缓存（东财延迟行情，约 5~10s）...")
        rows = self._fetch_market_cap_all()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_MARKET_CAP_TABLE} ("
                "symbol TEXT PRIMARY KEY, circ_mv REAL)"
            )
            conn.executemany(
                f"INSERT OR REPLACE INTO {_MARKET_CAP_TABLE} (symbol, circ_mv) VALUES (?, ?)", rows
            )
            conn.commit()
        logger.info(f"流通市值缓存完成，共 {len(rows)} 条")

    def refresh_market_cap_cache(self) -> int:
        """强制刷新流通市值缓存，返回写入条数。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {_MARKET_CAP_TABLE}")
            conn.commit()
        self._ensure_market_cap_cache()
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {_MARKET_CAP_TABLE}").fetchone()[0]

    @staticmethod
    def _fetch_market_cap_all() -> list[tuple[str, float]]:
        """从东财 push2delay 拉取全市场流通市值（单位：元，与 turnover 同口径）。

        push2delay 强制每页上限 100 条（total≈5800），故按页全量分页拉取；
        端点为延迟行情，盘后稳定可用。字段：f12=代码, f21=流通市值。
        """
        from sequoia_x.core.rate_limiter import em_get

        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        base_url = "https://push2delay.eastmoney.com/api/qt/clist/get"
        fs = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
        page_size = 100
        rows: list[tuple[str, float]] = []
        page = 1
        while True:
            params = {
                "pn": page, "pz": page_size, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                "fid": "f3", "fs": fs, "fields": "f12,f21",
            }
            try:
                r = em_get(base_url, params=params, headers=headers, timeout=10)
                data = r.json().get("data") or {}
                diff = data.get("diff", []) or []
                total = data.get("total", 0)
            except Exception:
                diff = []
                total = 0
            if not diff:
                break
            for b in diff:
                sym, mv = str(b.get("f12", "")), b.get("f21")
                if sym and mv is not None and mv != "-":
                    rows.append((sym, float(mv)))
            if total and len(rows) >= total:
                break
            page += 1
            time.sleep(0.1)
        return rows

    def _ensure_industry_cache(self) -> None:
        """确保 stock_industry 缓存表存在；不存在则从 baostock 拉取并写入。"""
        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (_INDUSTRY_TABLE,)
            ).fetchone()
            if exists:
                return

        logger.info("首次构建行业分类缓存（baostock，约 30~45s）...")
        import baostock as bs

        bs.login()
        rows: list[tuple[str, str]] = []
        try:
            rs = bs.query_stock_industry()
            while rs.next():
                data = rs.get_row_data()
                code = data[1]          # sh.600000
                industry = data[3]      # C39计算机...
                symbol = code.split(".")[-1]
                rows.append((symbol, _clean_industry_name(industry)))
        finally:
            bs.logout()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_INDUSTRY_TABLE} ("
                "symbol TEXT PRIMARY KEY, industry TEXT)"
            )
            conn.executemany(
                f"INSERT OR REPLACE INTO {_INDUSTRY_TABLE} (symbol, industry) VALUES (?, ?)", rows
            )
            conn.commit()
        logger.info(f"行业分类缓存完成，共 {len(rows)} 条")

    def refresh_industry_cache(self) -> int:
        """强制刷新行业分类缓存，返回写入条数。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {_INDUSTRY_TABLE}")
            conn.commit()
        self._ensure_industry_cache()
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {_INDUSTRY_TABLE}").fetchone()[0]

    # ------------------------------------------------------------------
    # 东财细分行业板块缓存（个股→最细板块映射，本地计算涨跌幅）
    # ------------------------------------------------------------------

    def _has_board_cache(self) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {_BOARD_TABLE}").fetchone()
            return row[0] > 0

    def refresh_board_cache(self) -> int:
        """从东方财富拉取全部行业板块成分股，构建「个股→最细板块」映射并持久化。

        东财板块存在层级（如「半导体」含「模拟芯片设计」），处理策略：按成员数
        升序遍历，每只股票首次出现即归属（最细板块优先），粗板块自动仅保留无细分的个股。
        一次构建约 3~5 分钟，之后板块涨跌幅全部从本地 stock_daily 计算，零网络依赖。
        """

        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        base_url = "https://push2delay.eastmoney.com/api/qt/clist/get"

        # 1. 拉取全部行业板块（代码、名称、成员数）
        boards: list[tuple[str, str, int]] = []
        for page in range(1, 8):
            params = {
                "pn": page, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                "fid": "f3", "fs": "m:90 t:2", "fields": "f12,f14,f104,f105",
            }
            try:
                r = em_get(base_url, params=params, headers=headers, timeout=10)
                diff = (r.json().get("data") or {}).get("diff", []) or []
            except Exception:
                diff = []
            if not diff:
                break
            for b in diff:
                count = int(b["f104"]) + int(b["f105"])
                boards.append((b["f12"], str(b["f14"]), count))
            if len(diff) < 100:
                break
            time.sleep(0.3)

        if not boards:
            raise RuntimeError("无法获取板块列表（东财接口不可用）")

        # 2. 按成员数升序（最细板块优先），逐板块取成分股并分配
        boards.sort(key=lambda x: x[2])
        stock_board: dict[str, str] = {}
        for idx, (code, name, _) in enumerate(boards):
            for pn in range(1, 4):  # 每板块最多 3 页（300 只），覆盖绝大多数
                params = {
                    "pn": pn, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                    "fid": "f3", "fs": f"b:{code}", "fields": "f12",
                }
                try:
                    r = em_get(base_url, params=params, headers=headers, timeout=10)
                    diff = (r.json().get("data") or {}).get("diff", []) or []
                except Exception:
                    diff = []
                if not diff:
                    break
                for b in diff:
                    sym = str(b["f12"])
                    if sym not in stock_board:
                        stock_board[sym] = name
                if len(diff) < 100:
                    break
                time.sleep(0.15)
            time.sleep(0.15)
            if (idx + 1) % 50 == 0:
                logger.info(f"板块映射构建中... {idx + 1}/{len(boards)} 板块，已映射 {len(stock_board)} 只")

        # 3. 持久化到 stock_board_em 表
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {_BOARD_TABLE}")
            conn.execute(
                f"CREATE TABLE {_BOARD_TABLE} (symbol TEXT PRIMARY KEY, board TEXT)"
            )
            conn.executemany(
                f"INSERT OR REPLACE INTO {_BOARD_TABLE} (symbol, board) VALUES (?, ?)",
                list(stock_board.items()),
            )
            conn.commit()
        logger.info(f"东财板块映射缓存完成：{len(stock_board)} 只股票 → {len(set(stock_board.values()))} 个板块")
        return len(stock_board)
