"""集合竞价选股引擎：9:25竞价快照采集 → 5维评分 → 分级推送。

A股短线核心战场：集合竞价是全天资金意图的首次集中暴露。
高开+放量+量比大 = 主力认可；高开+缩量 = 诱多风险。

评分模型（5维度，参考主流游资竞价战法）：
  - 竞价涨幅 30%：高开幅度（1%-7%为佳，>9.5%炸板风险）
  - 量比 25%：竞价量相对昨日均量倍数（>3倍强势）
  - 成交额 20%：竞价成交额（>5000万资金认可度高）
  - 昨封板属性 15%：昨日涨停今日继续高开=连板预期
  - 板块共振 10%：同板块多只竞价强=板块效应
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

_CREATE_AUCTION_SQL = """
CREATE TABLE IF NOT EXISTS auction_snap (
    date             TEXT    NOT NULL,
    symbol           TEXT    NOT NULL,
    name             TEXT,
    auction_price    REAL,
    auction_pct      REAL,
    prev_close       REAL,
    auction_amount   REAL,
    volume_ratio     REAL,
    yesterday_limit  INTEGER,
    board            TEXT,
    score            REAL,
    grade            TEXT,
    signal           TEXT,
    detail           TEXT,
    created_at       TEXT,
    PRIMARY KEY (date, symbol)
);
"""

_CREATE_VERIFY_SQL = """
CREATE TABLE IF NOT EXISTS auction_verify (
    verify_date      TEXT    NOT NULL,
    auction_date     TEXT    NOT NULL,
    grade            TEXT    NOT NULL,
    symbol           TEXT    NOT NULL,
    name             TEXT,
    score            REAL,
    t1_return        REAL,    -- T+1 涨跌幅%
    t5_return        REAL,    -- T+5 累计涨跌幅%
    is_win           INTEGER, -- T+1 是否盈利
    created_at       TEXT,
    PRIMARY KEY (verify_date, symbol)
);
"""


@dataclass
class AuctionItem:
    symbol: str
    name: str = ""
    auction_price: float = 0.0
    auction_pct: float = 0.0      # 竞价涨跌幅%
    prev_close: float = 0.0       # 昨收（真实价）
    auction_amount: float = 0.0   # 竞价成交额（万元）
    volume_ratio: float = 0.0     # 量比
    yesterday_limit: bool = False  # 昨日是否涨停
    board: str = ""               # 所属板块
    score: float = 0.0
    grade: str = ""               # A / B / 风险
    signal: str = ""              # 强势竞价/标准竞价/风险竞价
    detail: dict = field(default_factory=dict)


class AuctionScanner:
    """集合竞价扫描器：采集全市场竞价快照 → 评分 → 分级。"""

    # 评分权重
    W_PCT = 0.30
    W_VR = 0.25
    W_AMT = 0.20
    W_LIMIT = 0.15
    W_BOARD = 0.10

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_AUCTION_SQL)
            conn.execute(_CREATE_VERIFY_SQL)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_auction_date ON auction_snap(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_verify_auction ON auction_verify(auction_date)")
            conn.commit()

    def _fetch_spot(self) -> list[dict]:
        """拉取东财全市场实时快照（竞价撮合后反映开盘意图）。"""
        import requests

        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        all_rows: list[dict] = []
        # 分页拉全市场
        for page in range(1, 20):
            try:
                r = requests.get(
                    "https://push2delay.eastmoney.com/api/qt/clist/get",
                    params={
                        "pn": page, "pz": 200, "po": 1, "np": 1,
                        "fltt": 2, "invt": 2,
                        "fs": "m:0+t:6+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:81+f:!2,m:1+t:23+f:!2",
                        "fields": "f12,f14,f2,f3,f5,f6,f10,f17",
                    },
                    headers=headers, timeout=10,
                )
                d = r.json().get("data", {})
                diff = d.get("diff") or []
                if not diff:
                    break
                for item in diff:
                    sym = item.get("f12")
                    if not sym:
                        continue
                    def _num(v):
                        try: return float(v)
                        except (TypeError, ValueError): return 0.0
                    all_rows.append({
                        "symbol": sym,
                        "name": item.get("f14", ""),
                        "price": _num(item.get("f2")),
                        "pct": _num(item.get("f3")),
                        "volume": _num(item.get("f5")),
                        "amount": _num(item.get("f6")),
                        "vr": _num(item.get("f10")),
                        "open": _num(item.get("f17")),
                    })
            except Exception as e:
                logger.warning(f"竞价快照第{page}页失败：{e!r}")
                break
        logger.info(f"竞价快照采集：{len(all_rows)} 只")
        return all_rows

    def _load_yesterday(self) -> dict[str, dict]:
        """加载全市场昨日数据（昨收/是否涨停/板块），用于竞价评分。"""
        with sqlite3.connect(self.db_path) as conn:
            # 最新交易日
            row = conn.execute("SELECT MAX(date) FROM stock_daily").fetchone()
            if not row or not row[0]:
                return {}
            latest = row[0]
            rows = conn.execute(
                f"""SELECT d.symbol, d.close, d.high, d.low, d.open,
                           (h.high - h.low) / h.low * 100 as amplitude
                    FROM stock_daily d
                    LEFT JOIN stock_daily h ON h.symbol = d.symbol AND h.date = ?
                    WHERE d.date = ?""",
                (latest, latest),
            ).fetchall()
        result = {}
        limit_pct_threshold = 9.7  # 涨停近似（含ST 5%、主板10%、创业板20%的简化处理）
        for r in rows:
            sym, close, high, low, opn = r[0], r[1] or 0, r[2] or 0, r[3] or 0, r[4] or 0
            if not close or not opn:
                continue
            chg = (close - opn) / opn * 100 if opn else 0
            # 涨停判定：涨幅接近涨停板（主板>=9.7% 或 创业板>=19.7%）
            is_limit = chg >= limit_pct_threshold or (sym.startswith(("3", "68")) and chg >= 19.5)
            result[sym] = {"prev_close": close, "y_limit": is_limit}
        return result

    def _load_boards(self) -> dict[str, str]:
        """加载 symbol → 板块映射。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, board FROM stock_board_em"
                ).fetchall()
            return {r[0]: r[1] for r in rows if r[0] and r[1]}
        except sqlite3.OperationalError:
            return {}

    @staticmethod
    def _score_pct(pct: float) -> float:
        """竞价涨幅评分：1%-7%为佳，>9.5%炸板风险降分，<0不计分。"""
        if pct <= 0:
            return max(0, 50 + pct * 3)  # 平开微绿给中低分
        if pct > 9.5:
            return 40  # 接近涨停高开，炸板风险大
        if pct >= 7:
            return 75
        if pct >= 3:
            return 90
        if pct >= 1:
            return 80
        return 60  # 0-1%微高开

    @staticmethod
    def _score_vr(vr: float) -> float:
        """量比评分：>3倍强势，<1缩量。"""
        if vr >= 5:
            return 95
        if vr >= 3:
            return 85
        if vr >= 2:
            return 70
        if vr >= 1:
            return 55
        return max(10, vr * 40)

    @staticmethod
    def _score_amt(amt_yi: float) -> float:
        """成交额评分（单位亿元）：>0.5亿强势。"""
        if amt_yi >= 1.0:
            return 90
        if amt_yi >= 0.5:
            return 75
        if amt_yi >= 0.2:
            return 55
        if amt_yi >= 0.05:
            return 35
        return 15

    def scan(self, top_n: int = 50, push: bool = False, notifier=None) -> dict:
        """执行竞价扫描：采集 → 评分 → 分级 → 落库（可选推送）。

        Args:
            top_n: 返回前N只
            push: 是否触发飞书推送
            notifier: FeishuNotifier 实例
        """
        spots = self._fetch_spot()
        if not spots:
            return {"error": "竞价快照采集失败", "items": []}

        yest = self._load_yesterday()
        boards = self._load_boards()

        # 统计各板块竞价强势股数（板块共振）
        board_strong: dict[str, int] = {}
        for s in spots:
            if s["pct"] >= 3 and s["amount"] >= 5e7:
                bd = boards.get(s["symbol"], "")
                if bd:
                    board_strong[bd] = board_strong.get(bd, 0) + 1

        items: list[AuctionItem] = []
        today = datetime.now().strftime("%Y-%m-%d")

        for s in spots:
            sym = s["symbol"]
            yd = yest.get(sym, {})
            bd = boards.get(sym, "")
            pct = s["pct"]
            amt_yi = s["amount"] / 1e8  # 转亿元

            # 过滤ST、停牌（价格为0或名称含ST）
            if "ST" in s["name"].upper() or s["price"] <= 0:
                continue

            s_pct = self._score_pct(pct)
            s_vr = self._score_vr(s["vr"])
            s_amt = self._score_amt(amt_yi)
            s_limit = 90 if yd.get("y_limit") else 40
            s_board = min(90, 40 + board_strong.get(bd, 0) * 15) if bd else 40

            score = round(
                self.W_PCT * s_pct + self.W_VR * s_vr + self.W_AMT * s_amt
                + self.W_LIMIT * s_limit + self.W_BOARD * s_board
            )

            # 分级
            if pct > 9.5 or (pct >= 5 and s["vr"] < 1):
                grade, signal = "风险", "风险竞价（高开炸板/缩量诱多）"
            elif score >= 80:
                grade, signal = "A", "强势竞价"
            elif score >= 65:
                grade, signal = "B", "标准竞价"
            else:
                continue  # 低分不入榜

            items.append(AuctionItem(
                symbol=sym, name=s["name"], auction_price=s["price"],
                auction_pct=round(pct, 2), prev_close=yd.get("prev_close", 0),
                auction_amount=round(amt_yi, 2), volume_ratio=round(s["vr"], 2),
                yesterday_limit=yd.get("y_limit", False), board=bd,
                score=score, grade=grade, signal=signal,
                detail={"s_pct": round(s_pct), "s_vr": round(s_vr),
                        "s_amt": round(s_amt), "s_limit": round(s_limit),
                        "s_board": round(s_board)},
            ))

        items.sort(key=lambda x: -x.score)
        top = items[:top_n]
        self._save(today, top)

        # 飞书推送
        push_result = None
        if push and notifier and top:
            push_result = self._push(notifier, today, top)

        logger.info(f"竞价扫描完成：候选{len(top)}只（A:{sum(1 for i in top if i.grade=='A')} "
                    f"B:{sum(1 for i in top if i.grade=='B')} 风险:{sum(1 for i in top if i.grade=='风险')})")

        return {
            "date": today,
            "count": len(top),
            "grade_a": sum(1 for i in top if i.grade == "A"),
            "grade_b": sum(1 for i in top if i.grade == "B"),
            "grade_risk": sum(1 for i in top if i.grade == "风险"),
            "items": [self._item_dict(i) for i in top],
            "push": push_result,
        }

    def _save(self, date: str, items: list[AuctionItem]) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = ("INSERT OR REPLACE INTO auction_snap "
               "(date, symbol, name, auction_price, auction_pct, prev_close, "
               "auction_amount, volume_ratio, yesterday_limit, board, score, grade, signal, detail, created_at) "
               "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
        with sqlite3.connect(self.db_path) as conn:
            for i in items:
                import json
                conn.execute(sql, (
                    date, i.symbol, i.name, i.auction_price, i.auction_pct, i.prev_close,
                    i.auction_amount, i.volume_ratio, int(i.yesterday_limit), i.board,
                    i.score, i.grade, i.signal, json.dumps(i.detail, ensure_ascii=False), now,
                ))
            conn.commit()

    @staticmethod
    def _item_dict(i: AuctionItem) -> dict:
        return {
            "symbol": i.symbol, "name": i.name, "price": i.auction_price,
            "pct": i.auction_pct, "amount_yi": i.auction_amount, "vr": i.volume_ratio,
            "y_limit": i.yesterday_limit, "board": i.board,
            "score": i.score, "grade": i.grade, "signal": i.signal, "detail": i.detail,
        }

    def _push(self, notifier, date: str, items: list[AuctionItem]) -> dict:
        """竞价榜飞书推送。"""
        grade_a = [i for i in items if i.grade == "A"]
        grade_b = [i for i in items if i.grade == "B"]
        risks = [i for i in items if i.grade == "风险"]

        lines = [f"🔥 **{date} 集合竞价强度榜**", f"A级(强势) {len(grade_a)} | B级(标准) {len(grade_b)} | 风险 {len(risks)}", ""]

        if grade_a:
            lines.append("🅰️ **强势竞价（重点关注）**")
            for i in grade_a[:10]:
                lim = " 昨涨停" if i.yesterday_limit else ""
                lines.append(
                    f"  {i.symbol} {i.name} | 评分{i.score} | 高开{i.auction_pct}% | "
                    f"量比{i.volume_ratio} | 额{i.auction_amount}亿{lim}"
                )
            lines.append("")

        if grade_b:
            lines.append("🅱️ **标准竞价（备选池）**")
            for i in grade_b[:8]:
                lines.append(f"  {i.symbol} {i.name} | 评分{i.score} | 高开{i.auction_pct}% | 量比{i.volume_ratio}")
            lines.append("")

        if risks:
            lines.append("⚠️ **风险竞价（规避）**")
            for i in risks[:5]:
                lines.append(f"  {i.symbol} {i.name} | {i.signal}")

        text = "\n".join(lines)
        try:
            ok = notifier.send_text(title=f"Sequoia-X | {date}竞价榜", content=text, webhook_key="auction")
            return {"status": "ok" if ok else "error", "pushed": len(items) if ok else 0}
        except Exception as e:
            logger.warning(f"竞价推送失败：{e!r}")
            return {"status": "error", "msg": str(e)}

    def verify_t1(self, auction_date: str | None = None) -> dict:
        """验证竞价命中率：算竞价日次日的收益（T+1）及5日收益（T+5）。

        用日K相邻close比算涨跌幅（后复权相邻日比值=真实涨跌，口径一致）。
        竞价价是真实价，但验证不看绝对价格，只看次日方向，故用涨跌幅避免复权口径问题。

        Args:
            auction_date: 指定竞价日验证；None=验证所有未验证的竞价日
        Returns:
            {verified, summary: {grade: {count, win_rate, avg_t1, avg_t5}}}
        """
        import pandas as pd

        # 找需验证的竞价日
        with sqlite3.connect(self.db_path) as conn:
            if auction_date:
                auctions = conn.execute(
                    "SELECT date, symbol, name, grade, score FROM auction_snap WHERE date=?",
                    (auction_date,),
                ).fetchall()
                dates_to_verify = [auction_date]
            else:
                # 所有竞价日，排除已验证的
                all_dates = [r[0] for r in conn.execute(
                    "SELECT DISTINCT date FROM auction_snap ORDER BY date"
                ).fetchall()]
                verified = set(r[0] for r in conn.execute(
                    "SELECT DISTINCT auction_date FROM auction_verify"
                ).fetchall())
                dates_to_verify = [d for d in all_dates if d not in verified]
                if not dates_to_verify:
                    return {"verified": 0, "summary": {}, "msg": "所有竞价日已验证"}
                auctions = conn.execute(
                    f"SELECT date, symbol, name, grade, score FROM auction_snap "
                    f"WHERE date IN ({','.join('?'*len(dates_to_verify))})",
                    dates_to_verify,
                ).fetchall()

        if not auctions:
            return {"verified": 0, "summary": {}, "msg": "无竞价记录可验证"}

        # 取日K，按symbol分组找竞价日后N日的收盘价
        verified_rows: list[tuple] = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as conn:
            # 全市场交易日后序列（用于找T+1/T+5）
            trade_dates = [r[0] for r in conn.execute(
                "SELECT DISTINCT date FROM stock_daily ORDER BY date"
            ).fetchall()]

            for a_date, symbol, name, grade, score in auctions:
                if a_date not in trade_dates:
                    continue  # 竞价日无日K数据（可能未同步）
                idx = trade_dates.index(a_date)
                # T+1 = 竞价日的次日
                if idx + 1 >= len(trade_dates):
                    continue  # 次日还没数据，跳过
                next_date = trade_dates[idx + 1]
                row_cur = conn.execute(
                    "SELECT close FROM stock_daily WHERE symbol=? AND date=?",
                    (symbol, a_date),
                ).fetchone()
                row_next = conn.execute(
                    "SELECT close FROM stock_daily WHERE symbol=? AND date=?",
                    (symbol, next_date),
                ).fetchone()
                if not row_cur or not row_next or not row_cur[0]:
                    continue
                t1_ret = round((row_next[0] / row_cur[0] - 1) * 100, 2)
                # T+5
                t5_ret = None
                if idx + 6 < len(trade_dates):
                    t5_date = trade_dates[idx + 5]
                    row_t5 = conn.execute(
                        "SELECT close FROM stock_daily WHERE symbol=? AND date=?",
                        (symbol, t5_date),
                    ).fetchone()
                    if row_t5 and row_t5[0]:
                        t5_ret = round((row_t5[0] / row_cur[0] - 1) * 100, 2)

                verified_rows.append((
                    next_date, a_date, grade, symbol, name, score,
                    t1_ret, t5_ret, int(t1_ret > 0), now,
                ))

        if not verified_rows:
            return {"verified": 0, "summary": {},
                    "msg": f"竞价日 {dates_to_verify} 次日日K尚未就绪，需数据同步后验证"}

        # 落库
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO auction_verify "
                "(verify_date, auction_date, grade, symbol, name, score, "
                "t1_return, t5_return, is_win, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                verified_rows,
            )
            conn.commit()

        summary = self._verify_summary()
        logger.info(f"竞价T+1验证完成：{len(verified_rows)}条，竞价日{dates_to_verify}")
        return {"verified": len(verified_rows), "dates": dates_to_verify, "summary": summary}

    def _verify_summary(self) -> dict:
        """汇总各分级命中率统计。"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT grade, COUNT(*), SUM(is_win), AVG(t1_return), AVG(t5_return) "
                "FROM auction_verify GROUP BY grade"
            ).fetchall()
        result = {}
        for grade, cnt, wins, avg_t1, avg_t5 in rows:
            result[grade] = {
                "count": cnt,
                "win_rate": round(wins / cnt * 100, 1) if cnt else 0,
                "avg_t1": round(avg_t1 or 0, 2),
                "avg_t5": round(avg_t5 or 0, 2),
            }
        return result

    def get_verify_detail(self, limit: int = 100) -> list[dict]:
        """查询验证明细记录。"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT verify_date, auction_date, grade, symbol, name, score, "
                "t1_return, t5_return, is_win FROM auction_verify "
                "ORDER BY verify_date DESC, score DESC LIMIT ?",
                (limit,),
            ).fetchall()
        cols = ["verify_date", "auction_date", "grade", "symbol", "name", "score",
                "t1", "t5", "win"]
        return [dict(zip(cols, r)) for r in rows]

    def get_history(self, date: str | None = None, limit: int = 50) -> list[dict]:
        """查询历史竞价记录。"""
        with sqlite3.connect(self.db_path) as conn:
            if date:
                rows = conn.execute(
                    "SELECT * FROM auction_snap WHERE date=? ORDER BY score DESC LIMIT ?",
                    (date, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM auction_snap ORDER BY date DESC, score DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        cols = ["date", "symbol", "name", "price", "pct", "prev", "amt", "vr",
                "y_lim", "board", "score", "grade", "signal", "detail", "created"]
        return [dict(zip(cols, r)) for r in rows]
