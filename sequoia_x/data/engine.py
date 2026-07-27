"""数据引擎模块：负责 SQLite 行情数据存储与 baostock 增量同步。"""

import sqlite3
from pathlib import Path

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_daily (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol   TEXT    NOT NULL,
    date     TEXT    NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    turnover REAL,
    turn        REAL,      -- 换手率(%)
    pct_chg     REAL,      -- 涨跌幅(%)
    tradestatus INTEGER,   -- 交易状态 1=正常 0=停牌
    isst        INTEGER,   -- 是否ST 1=ST 0=非ST
    UNIQUE (symbol, date)
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_daily (symbol, date);
"""

_CREATE_MINUTE_SQL = """
CREATE TABLE IF NOT EXISTS stock_minute (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol   TEXT    NOT NULL,
    datetime TEXT    NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    amount   REAL,
    UNIQUE (symbol, datetime)
);
"""

_CREATE_HOLDING_SQL = """
CREATE TABLE IF NOT EXISTS portfolio_holding (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    name        TEXT    DEFAULT '',
    entry_price REAL    NOT NULL,
    shares      INTEGER NOT NULL,
    entry_date  TEXT    NOT NULL,
    stop_loss   REAL    DEFAULT 0,      -- 当前止损价（移动止损会更新）
    initial_stop REAL   DEFAULT 0,      -- 初始止损（记录）
    target      REAL    DEFAULT 0,
    grade       TEXT    DEFAULT '',
    hit_strategies TEXT DEFAULT '',
    cost        REAL    DEFAULT 0,      -- 买入总成本
    status      TEXT    DEFAULT 'open', -- open / closed
    close_reason TEXT   DEFAULT '',
    closed_price REAL   DEFAULT 0,
    closed_date TEXT    DEFAULT '',
    notes       TEXT    DEFAULT '',
    UNIQUE (symbol, status)
);
"""

_CREATE_WEIGHTS_SQL = """
CREATE TABLE IF NOT EXISTS strategy_weights (
    strategy_key  TEXT PRIMARY KEY,
    quality_score INTEGER NOT NULL,
    sharpe        REAL DEFAULT 0,
    max_dd        REAL DEFAULT 0,
    alpha         REAL DEFAULT 0,
    calmar        REAL DEFAULT 0,
    win_rate      REAL DEFAULT 0,
    pl_ratio      REAL DEFAULT 0,
    annual_return REAL DEFAULT 0,
    sample_trades INTEGER DEFAULT 0,
    updated_at    TEXT NOT NULL
);
"""

_CREATE_DECISION_POOL_SQL = """
CREATE TABLE IF NOT EXISTS decision_pool (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    grade       TEXT    DEFAULT '',   -- A/B/C/观望/淘汰
    score       INTEGER DEFAULT 0,
    action      TEXT    DEFAULT '',
    source      TEXT    DEFAULT '',   -- buy/watch/竞价/持仓
    UNIQUE (symbol, date)
);
"""

_CREATE_PAPER_ACCOUNT_SQL = """
CREATE TABLE IF NOT EXISTS paper_account (
    id              INTEGER PRIMARY KEY,
    name            TEXT    DEFAULT 'default',
    initial_capital REAL    NOT NULL,    -- 初始本金
    cash            REAL    NOT NULL,    -- 可用现金
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);
"""

_CREATE_PAPER_TRADES_SQL = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   INTEGER NOT NULL DEFAULT 1,
    symbol       TEXT    NOT NULL,
    name         TEXT    DEFAULT '',
    side         TEXT    NOT NULL,   -- buy / sell
    price        REAL    NOT NULL,
    shares       INTEGER NOT NULL,
    amount       REAL    NOT NULL,   -- 成交金额
    date         TEXT    NOT NULL,
    reason       TEXT    DEFAULT '',  -- 买入理由/卖出原因
    pnl          REAL    DEFAULT 0,   -- 卖出时记录本次盈亏
    pnl_pct      REAL    DEFAULT 0,
    hold_days    INTEGER DEFAULT 0,
    entry_price  REAL    DEFAULT 0,   -- 卖出记录对应买入价
    created_at   TEXT    NOT NULL
);
"""

_CREATE_PAPER_HOLDINGS_SQL = """
CREATE TABLE IF NOT EXISTS paper_holdings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   INTEGER NOT NULL DEFAULT 1,
    symbol       TEXT    NOT NULL,
    name         TEXT    DEFAULT '',
    entry_price  REAL    NOT NULL,
    shares       INTEGER NOT NULL,
    entry_date   TEXT    NOT NULL,
    stop_loss    REAL    DEFAULT 0,
    initial_stop REAL    DEFAULT 0,
    target       REAL    DEFAULT 0,
    grade        TEXT    DEFAULT '',
    hit_strategies TEXT DEFAULT '',
    cost         REAL    DEFAULT 0,
    UNIQUE (account_id, symbol)
);
"""

_CREATE_PAPER_NAV_SQL = """
CREATE TABLE IF NOT EXISTS paper_nav (
    date            TEXT    NOT NULL,
    total_assets    REAL    NOT NULL,
    cash            REAL    NOT NULL,
    market_value    REAL    NOT NULL,
    daily_return    REAL    DEFAULT 0,    -- 当日收益率%
    cum_return      REAL    DEFAULT 0,    -- 累计收益率%
    benchmark_return REAL   DEFAULT 0,    -- 沪深300当日收益率%
    benchmark_cum   REAL    DEFAULT 0,    -- 沪深300累计收益率%
    holding_count   INTEGER DEFAULT 0,
    UNIQUE (date)
);
"""


_CREATE_FACTOR_WEIGHTS_SQL = """
CREATE TABLE IF NOT EXISTS factor_weights (
    factor_name TEXT PRIMARY KEY,
    category    TEXT DEFAULT '',
    ic_mean     REAL NOT NULL,
    icir        REAL DEFAULT 0,
    win_rate    REAL DEFAULT 0,
    weight      REAL NOT NULL,
    crowding    REAL DEFAULT 0,
    updated_at  TEXT NOT NULL
);
"""

_CREATE_MARKET_FACTOR_WEIGHTS_SQL = """
CREATE TABLE IF NOT EXISTS market_factor_weights (
    market_state TEXT    NOT NULL,   -- bull / neutral / bear
    factor_name  TEXT    NOT NULL,
    category     TEXT    DEFAULT '',
    ic_mean      REAL    NOT NULL,
    icir         REAL    DEFAULT 0,
    win_rate     REAL    DEFAULT 0,
    weight       REAL    NOT NULL,
    updated_at   TEXT    NOT NULL,
    PRIMARY KEY (market_state, factor_name)
);
"""

_CREATE_FUND_HOLD_SQL = """
CREATE TABLE IF NOT EXISTS fund_hold (
    symbol        TEXT    NOT NULL,
    report_date   TEXT    NOT NULL,
    name          TEXT    DEFAULT '',
    fund_count    INTEGER DEFAULT 0,    -- 持有基金家数
    hold_shares   REAL,                  -- 持股总数
    hold_value    REAL,                  -- 持股市值
    change_dir    TEXT,                  -- 增仓/减仓
    change_shares REAL,                  -- 变动股数
    change_pct    REAL,                  -- 变动比例%
    PRIMARY KEY (symbol, report_date)
);
"""

_CREATE_INDEX_DAILY_SQL = """
CREATE TABLE IF NOT EXISTS index_daily (
    symbol  TEXT    NOT NULL,
    date    TEXT    NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    PRIMARY KEY (symbol, date)
);
"""

_CREATE_MACRO_MONEY_SQL = """
CREATE TABLE IF NOT EXISTS macro_money (
    month   TEXT PRIMARY KEY,
    m2      REAL,       -- M2货币供应量(亿元)
    m2_yoy  REAL,       -- M2同比%
    m1      REAL,       -- M1
    m1_yoy  REAL,
    m0      REAL,       -- M0流通现金
    m0_yoy  REAL
);
"""

_CREATE_MACRO_SF_SQL = """
CREATE TABLE IF NOT EXISTS macro_sf (
    month           TEXT PRIMARY KEY,
    sf_total        REAL,   -- 社融增量(亿元)
    rmb_loan        REAL,   -- 人民币贷款
    entrust_loan    REAL,   -- 委托贷款
    trust_loan      REAL,   -- 信托贷款
    corp_bond       REAL,   -- 企业债券
    equity_finance  REAL    -- 股票融资
);
"""

_CREATE_MARGIN_DETAIL_SQL = """
CREATE TABLE IF NOT EXISTS margin_detail (
    symbol    TEXT    NOT NULL,
    date      TEXT    NOT NULL,
    rzye      REAL,           -- 融资余额（看多杠杆水平）
    rzbuy     REAL,           -- 融资买入额
    rzrepay   REAL,           -- 融资偿还额
    rqlts     REAL,           -- 融券余量（看空杠杆水平）
    rqsell    REAL,           -- 融券卖出量
    rqrepay   REAL,           -- 融券偿还量
    rqye      REAL,           -- 融券余额
    PRIMARY KEY (symbol, date)
);
"""

_CREATE_ML_SCORES_SQL = """
CREATE TABLE IF NOT EXISTS ml_scores (
    run_date      TEXT    NOT NULL,
    symbol        TEXT    NOT NULL,
    ml_score      REAL,
    ic_mean       REAL    DEFAULT 0,
    icir          REAL    DEFAULT 0,
    t_stat        REAL    DEFAULT 0,
    model_version TEXT    DEFAULT '',
    PRIMARY KEY (run_date, symbol)
);
"""

_CREATE_LHB_DETAIL_SQL = """
CREATE TABLE IF NOT EXISTS lhb_detail (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT    NOT NULL,
    name          TEXT    DEFAULT '',
    date          TEXT    NOT NULL,
    close         REAL    DEFAULT 0,
    pct_chg       REAL    DEFAULT 0,
    net_buy       REAL    DEFAULT 0,
    buy_amount    REAL    DEFAULT 0,
    sell_amount   REAL    DEFAULT 0,
    total_amount  REAL    DEFAULT 0,
    market_amount REAL    DEFAULT 0,
    net_ratio     REAL    DEFAULT 0,
    turnover_rate REAL    DEFAULT 0,
    circ_mv       REAL    DEFAULT 0,
    reason        TEXT    DEFAULT '',
    interp        TEXT    DEFAULT '',
    UNIQUE (symbol, date)
);
"""

_CREATE_LHB_SEATS_SQL = """
CREATE TABLE IF NOT EXISTS lhb_seats (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT    NOT NULL,
    date       TEXT    NOT NULL,
    direction  TEXT    NOT NULL,
    seat_name  TEXT    NOT NULL,
    buy_amount REAL    DEFAULT 0,
    sell_amount REAL   DEFAULT 0,
    net_amount REAL    DEFAULT 0,
    buy_ratio  REAL    DEFAULT 0,
    sell_ratio REAL    DEFAULT 0,
    reason     TEXT    DEFAULT '',
    UNIQUE (symbol, date, direction, seat_name)
);
"""



def _bs_fetch_batch(tasks: list) -> list:
    """多进程 worker：独立 login，批量拉取 baostock 数据（含单只重试）。"""
    import time
    import baostock as bs
    bs.login()
    results = []
    for symbol, bs_code, start, end in tasks:
        fetched = False
        for attempt in range(3):
            try:
                if attempt > 0:
                    bs.logout()
                    time.sleep(2 * attempt)
                    bs.login()
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,volume,amount,turn,pctChg,tradestatus,isST",
                    start_date=start,
                    end_date=end,
                    frequency="d",
                    adjustflag="2",  # 前复权（最新价≈真实交易价，K线连续）
                )
                if rs.error_code != "0":
                    continue
                while rs.next():
                    results.append([symbol] + rs.get_row_data())
                fetched = True
                break
            except Exception as e:
                logger.debug(f"baostock查询重试: {e!r}")
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                continue
        if not fetched:
            logger.warning(f"worker: {symbol} 拉取失败（重试3次）")
    bs.logout()
    return results


class DataEngine:
    """行情数据引擎，负责 SQLite 存储和 baostock 数据同步。"""

    def __init__(self, settings: Settings) -> None:
        self.db_path: str = settings.db_path
        self.start_date: str = settings.start_date
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.execute(_CREATE_HOLDING_SQL)
            conn.execute(_CREATE_WEIGHTS_SQL)
            conn.execute(_CREATE_FACTOR_WEIGHTS_SQL)
            conn.execute(_CREATE_LHB_DETAIL_SQL)
            conn.execute(_CREATE_LHB_SEATS_SQL)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lhb_detail_date ON lhb_detail(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lhb_seats_sym_dt ON lhb_seats(symbol, date)")
            conn.execute(_CREATE_DECISION_POOL_SQL)
            conn.execute(_CREATE_PAPER_ACCOUNT_SQL)
            conn.execute(_CREATE_PAPER_TRADES_SQL)
            conn.execute(_CREATE_PAPER_HOLDINGS_SQL)
            conn.execute(_CREATE_PAPER_NAV_SQL)
            conn.execute(_CREATE_MARKET_FACTOR_WEIGHTS_SQL)
            conn.execute(_CREATE_ML_SCORES_SQL)
            conn.execute(_CREATE_MARGIN_DETAIL_SQL)
            conn.execute(_CREATE_FUND_HOLD_SQL)
            conn.execute(_CREATE_INDEX_DAILY_SQL)
            conn.execute(_CREATE_MACRO_MONEY_SQL)
            conn.execute(_CREATE_MACRO_SF_SQL)
            conn.execute(_CREATE_MINUTE_SQL)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_minute_sym_dt ON stock_minute(symbol, datetime)")
            # 增量迁移：给已有DB补列（旧数据新列为NULL，不破坏）
            _existing = {r[1] for r in conn.execute("PRAGMA table_info(stock_daily)")}
            for _col, _ddl in [
                ("turn", "REAL"), ("pct_chg", "REAL"),
                ("tradestatus", "INTEGER"), ("isst", "INTEGER"),
            ]:
                if _col not in _existing:
                    conn.execute(f"ALTER TABLE stock_daily ADD COLUMN {_col} {_ddl}")
            # P6: factor_weights 增 crowding 列（幂等）
            _fw_cols = {r[1] for r in conn.execute("PRAGMA table_info(factor_weights)")}
            if "crowding" not in _fw_cols:
                conn.execute("ALTER TABLE factor_weights ADD COLUMN crowding REAL DEFAULT 0")
            conn.commit()
        logger.info(f"数据库初始化完成：{self.db_path}")

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        """幂等给已有表补列（旧数据新列取DEFAULT，不破坏现有行）。"""
        with sqlite3.connect(self.db_path) as conn:
            cols = {c[1] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if column not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                conn.commit()

    def save_factor_weights(self, weights: list[dict]) -> None:
        """批量写入因子IC权重（UPSERT）。

        Args:
            weights: [{factor_name, category, ic_mean, icir, win_rate, weight}, ...]
        """
        import time
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        sql = ("INSERT OR REPLACE INTO factor_weights "
               "(factor_name, category, ic_mean, icir, win_rate, weight, crowding, updated_at) "
               "VALUES (?,?,?,?,?,?,?,?)")
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            for w in weights:
                conn.execute(sql, (
                    w["factor_name"], w.get("category", ""),
                    w["ic_mean"], w.get("icir", 0), w.get("win_rate", 0),
                    w["weight"], w.get("crowding", 0), now,
                ))
        finally:
            conn.close()

    def load_factor_weights(self) -> dict[str, dict]:
        """读取全部因子权重，返回 {factor_name: {weight, ic_mean, ...}}。"""
        sql = ("SELECT factor_name, category, ic_mean, icir, win_rate, weight, crowding, updated_at "
               "FROM factor_weights")
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
        return {
            r[0]: {
                "category": r[1], "ic_mean": r[2], "icir": r[3],
                "win_rate": r[4], "weight": r[5], "crowding": r[6],
                "updated_at": r[7],
            }
            for r in rows
        }

    def save_market_factor_weights(self, weights_by_state: dict[str, list[dict]]) -> None:
        """批量写入三态因子权重（bull/neutral/bear）。

        Args:
            weights_by_state: {"bull": [{factor_name, weight, ...}], "neutral": [...], ...}
        """
        import time
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("DELETE FROM market_factor_weights")
            for state, weights in weights_by_state.items():
                for w in weights:
                    conn.execute(
                        "INSERT INTO market_factor_weights "
                        "(market_state, factor_name, category, ic_mean, icir, win_rate, weight, updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (state, w["factor_name"], w.get("category", ""),
                         w["ic_mean"], w.get("icir", 0), w.get("win_rate", 0),
                         w["weight"], now),
                    )
        finally:
            conn.close()

    def load_market_factor_weights(self) -> dict[str, dict[str, float]]:
        """读取三态因子权重，返回 {market_state: {factor_name: weight}}。"""
        sql = ("SELECT market_state, factor_name, weight "
               "FROM market_factor_weights WHERE weight != 0")
        result: dict[str, dict[str, float]] = {}
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
        for state, fname, weight in rows:
            result.setdefault(state, {})[fname] = weight
        return result

    def save_strategy_weights(self, weights: list[dict]) -> None:
        """批量写入策略评估权重（UPSERT）。"""
        import time
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        # oos_decay 列迁移（幂等）：样本外衰减率，默认1.0=完全延续
        self._ensure_column("strategy_weights", "oos_decay", "REAL DEFAULT 1.0")
        self._ensure_column("strategy_weights", "marginal_alpha", "REAL DEFAULT 0")
        sql = ("INSERT OR REPLACE INTO strategy_weights "
               "(strategy_key, quality_score, sharpe, max_dd, alpha, calmar, "
               "win_rate, pl_ratio, annual_return, sample_trades, oos_decay, marginal_alpha, updated_at) "
               "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)")
        with sqlite3.connect(self.db_path) as conn:
            for w in weights:
                conn.execute(sql, (
                    w["strategy_key"], w["quality_score"],
                    w.get("sharpe", 0), w.get("max_dd", 0), w.get("alpha", 0),
                    w.get("calmar", 0), w.get("win_rate", 0), w.get("pl_ratio", 0),
                    w.get("annual_return", 0), w.get("sample_trades", 0),
                    w.get("oos_decay", 1.0), w.get("marginal_alpha", 0), now,
                ))
            conn.commit()

    def load_strategy_weights(self) -> dict[str, dict]:
        """读取全部策略权重，返回 {strategy_key: {quality_score, ...}}。"""
        self._ensure_column("strategy_weights", "marginal_alpha", "REAL DEFAULT 0")
        sql = ("SELECT strategy_key, quality_score, sharpe, max_dd, alpha, "
               "calmar, win_rate, pl_ratio, annual_return, sample_trades, marginal_alpha, updated_at "
               "FROM strategy_weights")
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
        return {
            r[0]: {
                "quality_score": r[1], "sharpe": r[2], "max_dd": r[3], "alpha": r[4],
                "calmar": r[5], "win_rate": r[6], "pl_ratio": r[7],
                "annual_return": r[8], "sample_trades": r[9], "marginal_alpha": r[10],
                "updated_at": r[11],
            }
            for r in rows
        }

    def _get_last_date(self, symbol: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM stock_daily WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return row[0] if row and row[0] else None

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        for _attempt in range(3):
            try:
                with sqlite3.connect(self.db_path, timeout=10) as conn:
                    conn.execute("PRAGMA busy_timeout=5000")
                    df = pd.read_sql(
                        "SELECT * FROM stock_daily WHERE symbol = ? ORDER BY date",
                        conn,
                        params=(symbol,),
                    )
                return df
            except sqlite3.OperationalError:
                if _attempt < 2:
                    time.sleep(0.1 * (_attempt + 1))
                else:
                    raise

    def get_all_daily(self) -> pd.DataFrame:
        """一次性加载全市场K线（3M行），内存缓存供多策略共用。

        9个策略原本各自逐只get_ohlcv，累计9×5000次SQL查询+connect，
        且并发时SQLite读竞争导致rps从2.8s暴涨到46s。
        共用一份全量DataFrame，避免重复I/O。结果按data_date缓存。
        """
        if getattr(self, "_all_daily_cache", None) is not None:
            return self._all_daily_cache
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                "SELECT * FROM stock_daily ORDER BY symbol, date", conn,
            )
        self._all_daily_cache = df
        return df

    def get_daily_groups(self) -> dict:
        """全量K线按symbol预分组（dict），供策略O(1)取单股切片。

        注意：策略逐只用 df[df.symbol==x] 是O(n)全表扫描（5000只要1200s），
        groupby预分组0.6s后dict取片O(1)，是正确做法。
        """
        if getattr(self, "_daily_groups_cache", None) is not None:
            return self._daily_groups_cache
        df = self.get_all_daily()
        self._daily_groups_cache = dict(iter(df.groupby("symbol", sort=False)))
        return self._daily_groups_cache

    @staticmethod
    def _to_baostock_code(symbol: str) -> str:
        """将纯数字代码转为 baostock 格式：6/9开头 -> sh，其余 -> sz。"""
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        return f"{prefix}.{symbol}"

    # ── 数据同步 ──

    def sync_today_bulk(self, force_full: bool = False) -> int:
        """多进程并行通过 baostock 拉取日K数据（前复权），写入 SQLite。"""
        from datetime import date, timedelta
        from multiprocessing import Pool

        today_str = date.today().strftime("%Y-%m-%d")

        tasks = []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT symbol, MAX(date) FROM stock_daily GROUP BY symbol"
            ).fetchall()

        if not rows:
            logger.warning("本地无股票数据，请先执行 --backfill")
            return 0

        for symbol, last_date in rows:
            if last_date and last_date >= today_str:
                continue
            start = today_str
            if force_full:
                start = self.start_date  # 全量刷新：从最早日期开始
            elif last_date:
                start = (date.fromisoformat(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            tasks.append((symbol, self._to_baostock_code(symbol), start, today_str))

        if not tasks:
            logger.info("所有股票已是最新，无需更新")
            return 0

        logger.info(f"需要更新 {len(tasks)} 只股票，启动多进程并行拉取...")

        # baostock 日额度检查（每只约2次API调用：login+query）
        from sequoia_x.core.rate_limiter import _rate_limiter
        estimated_calls = len(tasks) * 2
        if not _rate_limiter.baostock_check(estimated_calls):
            status = _rate_limiter.baostock_status()
            logger.warning(
                f"baostock 日额度不足，跳过同步。"
                f"今日已用 {status['used']}/{status['limit']}（{status['usage_pct']}%）"
            )
            return 0

        n_workers = min(3, len(tasks))
        chunks = [tasks[i::n_workers] for i in range(n_workers)]
        logger.info(f"sync_today_bulk: {len(tasks)}只 分{n_workers}worker 每worker~{len(chunks[0])}只")

        # 多进程并行拉取（baostock盘后高峰单login需8s，必须并行加速）
        with Pool(n_workers) as pool:
            batch_results = pool.map(_bs_fetch_batch, chunks)

        # 拉取结果合并后一次性落库（Pool.map全部完成才返回，无中途丢数据风险）
        all_rows = []
        for batch in batch_results:
            all_rows.extend(batch)

        if not all_rows:
            logger.warning("baostock无数据返回，切换东财fallback...")
            em_count = self.sync_today_eastmoney()
            if em_count > 0:
                return em_count
            logger.warning("东财也无数据，切换腾讯第三备源...")
            return self.sync_today_tencent()

        # 若baostock只拉到部分（<50%），补充东财fallback
        synced_syms = len({r[0] for r in all_rows})
        if synced_syms < len(tasks) * 0.5:
            logger.warning(f"baostock仅拉到{synced_syms}/{len(tasks)}只，东财补充剩余...")
            em_count = self.sync_today_eastmoney()
            # baostock结果也落库（可能有东财没有的票）
            pass

        df = pd.DataFrame(all_rows, columns=[
            "symbol", "date", "open", "high", "low", "close", "volume", "turnover",
            "turn", "pct_chg", "tradestatus", "isst",
        ])
        for col in ["open", "high", "low", "close", "volume", "turnover", "turn", "pct_chg"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["tradestatus", "isst"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        df = df.dropna(subset=["close"])
        df = df[df["volume"] > 0]

        count = len(df)
        # UPSERT累加写入：只更新本轮实际拉到的(symbol,date)，不DELETE已有数据
        # 避免多次同步时本轮拉取不完整覆盖掉上轮已成功的票
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO stock_daily "
                "(symbol, date, open, high, low, close, volume, turnover, turn, pct_chg, tradestatus, isst) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                df[["symbol", "date", "open", "high", "low", "close", "volume", "turnover",
                    "turn", "pct_chg", "tradestatus", "isst"]].values.tolist(),
            )
            conn.commit()

        logger.info(f"sync_today_bulk: UPSERT {count} 条（{df['symbol'].nunique()}只×{df['date'].nunique()}日）")
        # 记录 baostock 消耗
        from sequoia_x.core.rate_limiter import _rate_limiter
        _rate_limiter.baostock_consume(len(tasks) * 2)
        return count

    def sync_valuation(self) -> int:
        """从东财 push2delay 批量拉取全市场 PE/PB，写入 stock_market_cap 表。

        东财 clist 接口返回全市场实时快照（f9=PE-TTM, f23=PB），
        全市场3秒搞定，不依赖baostock，不限额度。
        """
        from sequoia_x.core.rate_limiter import _rate_limiter
        from datetime import date as _date
        import requests as _req

        # 确保 PE/PB 列存在
        with sqlite3.connect(self.db_path) as conn:
            cols = {c[1] for c in conn.execute("PRAGMA table_info(stock_market_cap)").fetchall()}
            if "pe" not in cols:
                conn.execute("ALTER TABLE stock_market_cap ADD COLUMN pe REAL")
            if "pb" not in cols:
                conn.execute("ALTER TABLE stock_market_cap ADD COLUMN pb REAL")
            conn.commit()

        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
        all_spot: dict[str, dict] = {}
        for page in range(1, 80):
            if not _rate_limiter.eastmoney_acquire():
                logger.warning("东财熔断，估值同步中止")
                break
            try:
                r = _req.get(
                    "https://push2delay.eastmoney.com/api/qt/clist/get",
                    params={
                        "pn": page, "pz": 200, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                        "fs": "m:0+t:6+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2",
                        "fields": "f12,f9,f23",
                    },
                    headers=headers, timeout=10,
                )
                d = r.json().get("data") or {}
                diff = d.get("diff") or []
                if not diff:
                    break
                for item in diff:
                    sym = item.get("f12", "")
                    if sym and len(sym) == 6 and sym.isdigit():
                        all_spot[sym] = item
                _rate_limiter.eastmoney_success()
            except Exception as e:
                _rate_limiter.eastmoney_failure()
                logger.debug(f"估值同步第{page}页失败: {e!r}")
                continue

        logger.info(f"估值同步：获取 {len(all_spot)} 只快照")

        rows = []
        for sym, item in all_spot.items():
            pe = item.get("f9")
            pb = item.get("f23")
            if pe is not None or pb is not None:
                rows.append((pe, pb, sym))

        if rows:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "UPDATE stock_market_cap SET pe=?, pb=? WHERE symbol=?",
                    rows,
                )
                # 不在 stock_market_cap 的新股也插入
                existing = {r[0] for r in conn.execute("SELECT symbol FROM stock_market_cap").fetchall()}
                new_rows = [(sym, None, pe, pb) for pe, pb, sym in rows if sym not in existing]
                if new_rows:
                    conn.executemany(
                        "INSERT INTO stock_market_cap (symbol, circ_mv, pe, pb) VALUES (?,?,?,?)",
                        new_rows,
                    )
                conn.commit()

        logger.info(f"估值同步完成：{len(rows)} 只更新PE/PB")
        return len(rows)

    def sync_today_tencent(self) -> int:
        """腾讯日K批量补全（第三备源：baostock额度耗尽 + 东财被封时）。

        原理：腾讯 web.ifzq.gtimg.cn 前复权日K接口，
        用 DB后复权昨收 / 腾讯前复权昨收 = 复权系数，
        前复权今日OHLC × 系数 = 后复权今日OHLC。
        逐只请求（无批量接口），但不限频率、不封IP，5000只约15分钟。
        """
        import requests as _req
        from datetime import date as _date

        today_str = _date.today().strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT symbol, MAX(date) FROM stock_daily GROUP BY symbol"
            ).fetchall()
            last_db = dict(conn.execute("""
                SELECT symbol, close FROM stock_daily d
                WHERE date = (SELECT MAX(date) FROM stock_daily WHERE symbol = d.symbol)
            """).fetchall())

        missing = {sym: last for sym, last in rows if not last or last < today_str}
        if not missing:
            logger.info("腾讯fallback: 所有股票已是最新")
            return 0

        logger.info(f"腾讯fallback: 需补 {len(missing)} 只")
        rows_to_insert = []
        success = 0
        for i, (sym, _) in enumerate(missing.items()):
            try:
                prefix = "sh" if sym.startswith(("6", "9")) else "sz"
                tc_sym = f"{prefix}{sym}"
                r = _req.get(
                    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                    params={"param": f"{tc_sym},day,,,5,qfq"},
                    timeout=8,
                )
                data = (r.json().get("data") or {}).get(tc_sym, {})
                klines = data.get("qfqday") or data.get("day") or []
                if not klines:
                    continue

                # 找最新一条
                latest = klines[-1]
                d_str = latest[0]
                if d_str < today_str:
                    continue  # 没有今天数据

                o, c, h, l, vol = float(latest[1]), float(latest[2]), float(latest[3]), float(latest[4]), float(latest[5])

                # 腾讯返回qfq前复权数据，直接使用（无需复权因子）

                rows_to_insert.append((
                    sym, d_str, o, h, l, c,
                    int(vol * 100), 0.0, None, None, 1, 0
                ))
                success += 1
            except Exception as e:
                logger.debug(f"腾讯fallback拉取失败: {e!r}")
                continue

            if (i + 1) % 500 == 0:
                logger.info(f"腾讯fallback进度: {i+1}/{len(missing)}, 成功{success}")

        if rows_to_insert:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO stock_daily "
                    "(symbol, date, open, high, low, close, volume, turnover, turn, pct_chg, tradestatus, isst) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows_to_insert,
                )
                conn.commit()

        logger.info(f"腾讯fallback完成: {success}/{len(missing)} 只")
        return success

    def save_decision_pool(self, items: list[dict]) -> int:
        """决策结果落库：盘后选出的票纳入盘中关注池。

        每次 generate_decision 后调用，UPSERT 今日决策池。
        build_watchlist 读取此表扩展盘中监控范围。
        """
        if not items:
            return 0
        import sqlite3 as _sql
        from datetime import datetime as _dt
        today = _dt.now().strftime("%Y-%m-%d")
        rows = [
            (it["symbol"], today, it.get("grade", ""),
             it.get("score", 0), it.get("action", ""), it.get("source", "buy"))
            for it in items
        ]
        with _sql.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO decision_pool (symbol, date, grade, score, action, source) "
                "VALUES (?, ?, ?, ?, ?, ?)", rows
            )
            conn.commit()
        return len(rows)

    def sync_today_eastmoney(self) -> int:
        """东财批量快照补全今日日K（baostock宕机时的fallback）。

        原理：东财clist返回全市场实时OHLCV（不复权）+昨收(f18)，
        用 DB后复权昨收 / 东财raw昨收 = 复权系数，
        raw今日OHLC × 系数 = 后复权今日OHLC。
        volume×100（手→股），turnover直接用amt。
        全市场3秒搞定（vs baostock 8worker×5分钟）。
        """
        import requests as _req
        from datetime import date as _date

        today_str = _date.today().strftime("%Y-%m-%d")
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}

        # 查出缺今日数据的股票及DB最后一天后复权close
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT symbol, MAX(date) FROM stock_daily GROUP BY symbol"
            ).fetchall()
            last_hfq = dict(conn.execute("""
                SELECT symbol, close FROM stock_daily d
                WHERE date = (SELECT MAX(date) FROM stock_daily WHERE symbol = d.symbol)
            """).fetchall())

        missing = {sym: last for sym, last in rows if not last or last < today_str}

        # 今日已有数据但缺turn/pct_chg的，也纳入更新（东财补全字段）
        with sqlite3.connect(self.db_path) as conn:
            stale = conn.execute(
                "SELECT symbol FROM stock_daily WHERE date=? AND turn IS NULL", (today_str,)
            ).fetchall()
        for r in stale:
            if r[0] not in missing:
                missing[r[0]] = today_str
        if stale:
            logger.info(f"东财fallback: 含{len(stale)}只今日缺turn字段，一并补全")

        if not missing:
            logger.info("东财fallback: 所有股票已是最新，字段完整")
            return 0
        logger.info(f"东财fallback: 需补 {len(missing)} 只")

        # 东财批量拉全市场沪深A股快照
        all_spot: dict[str, dict] = {}
        for page in range(1, 80):
            try:
                r = _req.get(
                    "https://push2delay.eastmoney.com/api/qt/clist/get",
                    params={
                        "pn": page, "pz": 200, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                        "fs": "m:0+t:6+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2",
                        "fields": "f12,f2,f3,f5,f6,f8,f15,f16,f17,f18",
                    },
                    headers=headers, timeout=10,
                )
                d = r.json().get("data") or {}
                diff = d.get("diff") or []
                if not diff:
                    break
                for item in diff:
                    sym = item.get("f12", "")
                    if sym and len(sym) == 6 and sym.isdigit():
                        all_spot[sym] = item
            except Exception as e:
                logger.debug(f"东财fallback快照拉取失败: {e!r}")
                continue

        logger.info(f"东财fallback: 快照获取 {len(all_spot)} 只")

        # 转换为后复权并落库
        rows_to_insert = []
        for sym in missing:
            spot = all_spot.get(sym)
            if not spot:
                continue

            def _num(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return 0.0

            raw_close = _num(spot.get("f2"))
            raw_open = _num(spot.get("f17"))
            raw_high = _num(spot.get("f15"))
            raw_low = _num(spot.get("f16"))
            raw_yest = _num(spot.get("f18"))
            vol = _num(spot.get("f5"))
            amt = _num(spot.get("f6"))
            pct_chg = _num(spot.get("f3"))   # 东财涨跌幅%
            turn = _num(spot.get("f8"))       # 东财换手率%

            if raw_close <= 0 or vol <= 0:
                continue
            # 前复权：今日价=真实交易价，直接使用东财raw（无需复权因子）

            rows_to_insert.append((
                sym, today_str,
                round(raw_open, 4), round(raw_high, 4),
                round(raw_low, 4), round(raw_close, 4),
                round(vol * 100, 2), round(amt, 2),
                round(turn, 4) if turn > 0 else None,         # 换手率
                round(pct_chg, 4) if pct_chg != 0 else None,  # 涨跌幅
                1,    # tradestatus=1 正常交易（停牌的close=0已跳过）
                0,    # isst=0（东财快照无ST标记，保守设0，baostock恢复后可修正）
            ))

        if rows_to_insert:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO stock_daily "
                    "(symbol, date, open, high, low, close, volume, turnover, turn, pct_chg, tradestatus, isst) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows_to_insert,
                )
                conn.commit()

        logger.info(f"东财fallback: UPSERT {len(rows_to_insert)} 条")
        return len(rows_to_insert)

    def backfill(self, symbols: list[str]) -> None:
        """通过 baostock 批量回填历史日 K 线数据（后复权）。

        容错机制：
        - 单只股票失败自动重试 3 次，间隔递增（2s/4s/8s）
        - 每 200 只股票自动重连 baostock（防止长连接超时）
        - 已入库的自动 skip，中断后可重跑续传
        """
        import time
        from datetime import date, timedelta

        import baostock as bs

        today_str = date.today().strftime("%Y-%m-%d")
        max_retries = 3
        reconnect_interval = 200  # 每处理 N 只股票重连一次

        def _login():
            lg = bs.login()
            if lg.error_code != "0":
                logger.error(f"baostock 登录失败: {lg.error_msg}")
                return False
            return True

        if not _login():
            return

        success = 0
        skipped = 0
        failed = 0
        since_reconnect = 0

        try:
            for i, symbol in enumerate(symbols):
                last_date = self._get_last_date(symbol)
                if last_date and last_date >= today_str:
                    skipped += 1
                    if (i + 1) % 500 == 0:
                        logger.info(
                            f"已处理 {i + 1}/{len(symbols)}，"
                            f"成功 {success} 跳过 {skipped} 失败 {failed}"
                        )
                    continue

                # 定期重连，防止长连接超时
                since_reconnect += 1
                if since_reconnect >= reconnect_interval:
                    bs.logout()
                    time.sleep(1)
                    if not _login():
                        logger.error("重连失败，终止回填")
                        return
                    since_reconnect = 0

                start = last_date or self.start_date
                if last_date:
                    start = (date.fromisoformat(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")

                bs_code = self._to_baostock_code(symbol)

                # 带重试的查询
                rows = []
                query_ok = False
                for attempt in range(max_retries):
                    try:
                        rs = bs.query_history_k_data_plus(
                            bs_code,
                            "date,open,high,low,close,volume,amount,turn,pctChg,tradestatus,isST",
                            start_date=start,
                            end_date=today_str,
                            frequency="d",
                            adjustflag="2",  # 前复权（最新价≈真实交易价，K线连续）
                        )

                        if rs.error_code != "0":
                            raise RuntimeError(rs.error_msg)

                        rows = []
                        while rs.next():
                            rows.append(rs.get_row_data())
                        query_ok = True
                        break

                    except Exception as exc:
                        if attempt < max_retries - 1:
                            wait = 2 ** (attempt + 1)
                            logger.warning(
                                f"[{symbol}] 第{attempt + 1}次失败: {exc}，{wait}s 后重试"
                            )
                            time.sleep(wait)
                            # 重连 baostock
                            bs.logout()
                            time.sleep(1)
                            _login()
                        else:
                            logger.warning(f"[{symbol}] {max_retries}次重试均失败，跳过")

                if not query_ok:
                    failed += 1
                    continue

                if not rows:
                    skipped += 1
                    continue

                df = pd.DataFrame(rows, columns=rs.fields)
                for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                for col in ["tradestatus", "isST"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"])
                df = df[df["volume"] > 0]

                if df.empty:
                    skipped += 1
                    continue

                df["symbol"] = symbol
                df = df.rename(columns={
                    "amount": "turnover", "pctChg": "pct_chg", "isST": "isst",
                })
                df = df[["symbol", "date", "open", "high", "low", "close", "volume", "turnover",
                         "turn", "pct_chg", "tradestatus", "isst"]]

                try:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.executemany(
                            "INSERT OR REPLACE INTO stock_daily "
                            "(symbol, date, open, high, low, close, volume, turnover, turn, pct_chg, tradestatus, isst) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            df.values.tolist(),
                        )
                        conn.commit()
                except sqlite3.IntegrityError:
                    pass

                success += 1

                if (i + 1) % 500 == 0:
                    logger.info(
                        f"已处理 {i + 1}/{len(symbols)}，"
                        f"成功 {success} 跳过 {skipped} 失败 {failed}"
                    )

        finally:
            bs.logout()

        logger.info(f"回填完成 — 成功: {success} | 跳过: {skipped} | 失败: {failed}")

    # ── 股票列表 ──

    def get_all_symbols(self) -> list[str]:
        """通过 baostock 获取全市场 A 股代码列表。"""
        import baostock as bs

        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"baostock 登录失败: {lg.error_msg}")
            return []

        try:
            rs = bs.query_stock_basic(code_name="", code="")
            symbols = []
            while rs.next():
                row = rs.get_row_data()
                code = row[0]           # "sh.600000" or "sz.000001"
                status = row[4]         # "1" = 上市
                stock_type = row[5]     # "1" = 股票
                if status == "1" and stock_type == "1":
                    symbols.append(code.split(".")[1])  # 提取纯数字代码
            logger.info(f"获取股票列表完成，共 {len(symbols)} 只")
            return symbols
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
        finally:
            bs.logout()

    def get_local_symbols(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM stock_daily"
            ).fetchall()
        return [row[0] for row in rows]

    def get_ipo_map(self) -> dict[str, str]:
        """返回 {symbol: ipo_date}，用于回测过滤新股（缓解幸存者偏差）。

        退市股历史行情当前缺失，本方法仅过滤新股上市初期的非理性波动。
        """
        with sqlite3.connect(self.db_path) as conn:
            try:
                rows = conn.execute(
                    "SELECT symbol, ipo_date FROM stock_basic"
                ).fetchall()
            except sqlite3.OperationalError:
                return {}
        return {r[0]: r[1] for r in rows if r[0] and r[1]}

    def get_ipo_cutoff_map(self, min_age_days: int = 365) -> dict[str, str]:
        """返回 {symbol: cutoff_date}，cutoff = 上市日 + min_age_days。

        回测时 df[date >= cutoff] 过滤新股上市初期的非理性波动，
        缓解新股偏差（注意：退市股历史缺失仍需数据层重建）。
        """
        from datetime import datetime, timedelta
        ipo_map = self.get_ipo_map()
        out: dict[str, str] = {}
        delta = timedelta(days=min_age_days)
        for sym, ipo in ipo_map.items():
            try:
                d = datetime.strptime(str(ipo)[:10], "%Y-%m-%d")
                out[sym] = (d + delta).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue
        return out

    def save_minute_klines(self, rows: list[dict]) -> int:
        """批量写入分钟K线（UPSERT）。"""
        if not rows:
            return 0
        sql = ("INSERT OR REPLACE INTO stock_minute "
               "(symbol, datetime, open, high, low, close, volume, amount) "
               "VALUES (?,?,?,?,?,?,?,?)")
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(sql, [(
                r["symbol"], r["datetime"], r["open"], r["high"],
                r["low"], r["close"], r["volume"], r["amount"],
            ) for r in rows])
            conn.commit()
        return len(rows)

    def get_minute_klines(self, symbol: str, date: str | None = None) -> pd.DataFrame:
        """读取分钟K线。date=None 取最新一天。"""
        with sqlite3.connect(self.db_path) as conn:
            if date:
                df = pd.read_sql_query(
                    "SELECT * FROM stock_minute WHERE symbol=? AND datetime LIKE ? ORDER BY datetime",
                    conn, params=(symbol, f"{date}%"),
                )
            else:
                df = pd.read_sql_query(
                    "SELECT * FROM stock_minute WHERE symbol=? ORDER BY datetime DESC LIMIT 240",
                    conn, params=(symbol,),
                )
                if not df.empty:
                    latest_date = str(df.iloc[0]["datetime"])[:10]
                    df = pd.read_sql_query(
                        "SELECT * FROM stock_minute WHERE symbol=? AND datetime LIKE ? ORDER BY datetime",
                        conn, params=(symbol, f"{latest_date}%"),
                    )
        return df

    # ------------------------------------------------------------------
    # 龙虎榜数据采集
    # ------------------------------------------------------------------
    def sync_lhb(self, date_str: str | None = None) -> int:
        """同步龙虎榜数据到本地 DB。

        Args:
            date_str: 日期 YYYYMMDD 格式，默认当天。

        Returns:
            写入的个股明细行数。
        """
        import akshare as ak
        from datetime import datetime as dt

        target = date_str or dt.now().strftime("%Y%m%d")
        db_date = f"{target[:4]}-{target[4:6]}-{target[6:8]}"

        with sqlite3.connect(self.db_path) as conn:
            # 跳过已同步的日期（幂等）
            exists = conn.execute(
                "SELECT COUNT(*) FROM lhb_detail WHERE date=?", (db_date,)
            ).fetchone()[0]
            if exists > 0:
                logger.info(f"龙虎榜 {db_date} 已存在 {exists} 行，跳过")
                return exists

        try:
            df = ak.stock_lhb_detail_em(start_date=target, end_date=target)
        except Exception as exc:
            logger.warning(f"龙虎榜拉取失败 {target}: {exc}")
            return 0

        if df is None or len(df) == 0:
            logger.info(f"龙虎榜 {target} 无数据（可能未发布或非交易日）")
            return 0

        # 数值清洗
        def _num(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0

        rows = []
        for _, r in df.iterrows():
            sym = str(r.get("代码", "")).strip()
            if not sym:
                continue
            rows.append((
                sym,
                str(r.get("名称", "")),
                db_date,
                _num(r.get("收盘价")),
                _num(r.get("涨跌幅")),
                _num(r.get("龙虎榜净买额")),
                _num(r.get("龙虎榜买入额")),
                _num(r.get("龙虎榜卖出额")),
                _num(r.get("龙虎榜成交额")),
                _num(r.get("市场总成交额")),
                _num(r.get("净买额占总成交比")),
                _num(r.get("换手率")),
                _num(r.get("流通市值")),
                str(r.get("上榜原因", "")),
                str(r.get("解读", "")),
            ))

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO lhb_detail
                   (symbol, name, date, close, pct_chg, net_buy, buy_amount, sell_amount,
                    total_amount, market_amount, net_ratio, turnover_rate, circ_mv,
                    reason, interp)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            conn.commit()

        logger.info(f"龙虎榜 {db_date} 写入 {len(rows)} 行个股明细")
        return len(rows)

    def sync_lhb_seats(self, date_str: str | None = None) -> int:
        """同步龙虎榜席位明细（买卖席位）。

        需要先有 lhb_detail 才知道有哪些股票上榜。

        Returns:
            写入的席位行数。
        """
        import akshare as ak
        from datetime import datetime as dt

        target = date_str or dt.now().strftime("%Y%m%d")
        db_date = f"{target[:4]}-{target[4:6]}-{target[6:8]}"

        with sqlite3.connect(self.db_path) as conn:
            # 获取当日上榜股票
            symbols = [r[0] for r in conn.execute(
                "SELECT symbol FROM lhb_detail WHERE date=?", (db_date,)
            ).fetchall()]
            # 跳过已同步席位的日期
            seat_exists = conn.execute(
                "SELECT COUNT(*) FROM lhb_seats WHERE date=?", (db_date,)
            ).fetchone()[0]

        if not symbols:
            logger.info(f"龙虎榜席位：{db_date} 无上榜个股，跳过")
            return 0
        if seat_exists > 0:
            logger.info(f"龙虎榜席位 {db_date} 已存在 {seat_exists} 行，跳过")
            return seat_exists

        def _num(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0

        all_rows = []
        for sym in symbols:
            for flag in ("买入", "卖出"):
                try:
                    sdf = ak.stock_lhb_stock_detail_em(
                        symbol=sym, date=target, flag=flag
                    )
                except Exception as exc:
                    logger.debug(f"席位拉取失败 {sym} {flag}: {exc}")
                    continue
                if sdf is None or len(sdf) == 0:
                    continue
                direction = "buy" if flag == "买入" else "sell"
                for _, r in sdf.iterrows():
                    seat = str(r.get("交易营业部名称", "")).strip()
                    if not seat:
                        continue
                    all_rows.append((
                        sym, db_date, direction, seat,
                        _num(r.get("买入金额")),
                        _num(r.get("卖出金额")),
                        _num(r.get("净额")),
                        _num(r.get("买入金额-占总成交比例")),
                        _num(r.get("卖出金额-占总成交比例")),
                        str(r.get("类型", "")),
                    ))

        if all_rows:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    """INSERT OR REPLACE INTO lhb_seats
                       (symbol, date, direction, seat_name, buy_amount, sell_amount,
                        net_amount, buy_ratio, sell_ratio, reason)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    all_rows,
                )
                conn.commit()

        logger.info(f"龙虎榜席位 {db_date} 写入 {len(all_rows)} 行（{len(symbols)} 只股票）")
        return len(all_rows)

    def get_lhb_date(self, date_str: str) -> list[dict]:
        """查本地 DB 获取某日龙虎榜个股明细。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT symbol, name, close, pct_chg, net_buy, buy_amount,
                          sell_amount, reason, interp
                   FROM lhb_detail WHERE date=? ORDER BY net_buy DESC""",
                (date_str,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_lhb_recent(self, days: int = 5) -> list[dict]:
        """查本地 DB 获取近 N 日龙虎榜净买入 TOP。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT symbol, name, date, net_buy, close, pct_chg, reason
                   FROM lhb_detail
                   WHERE date IN (
                       SELECT DISTINCT date FROM lhb_detail ORDER BY date DESC LIMIT ?
                   )
                   ORDER BY net_buy DESC LIMIT 20""",
                (days,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_lhb_by_symbol(self, symbol: str, days: int = 30) -> list[dict]:
        """查本地 DB 获取某只股票近 N 日龙虎榜记录。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT date, net_buy, buy_amount, sell_amount, reason, interp
                   FROM lhb_detail WHERE symbol=?
                   ORDER BY date DESC LIMIT ?""",
                (symbol, days),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_lhb_latest_date(self) -> str | None:
        """获取本地 DB 中龙虎榜的最新日期。"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM lhb_detail"
            ).fetchone()
        return row[0] if row and row[0] else None

    # ------------------------------------------------------------------
    # 主力资金流向采集（东财 clist API）
    # ------------------------------------------------------------------
    def sync_fund_flow(self, date_str: str | None = None) -> int:
        """同步全市场主力资金流向到 fund_flow 表。

        数据来源：东财 push2delay clist API
        字段：f62=主力净流入, f184=主力净流入占比, f66=超大单, f72=大单, f78=中单, f81=小单

        Returns:
            写入行数。
        """
        import requests
        from datetime import datetime as dt

        target_date = date_str or dt.now().strftime("%Y-%m-%d")

        # 建表（幂等）
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS fund_flow (
                    symbol     TEXT    NOT NULL,
                    date       TEXT    NOT NULL,
                    main_net   REAL    DEFAULT 0,
                    main_pct   REAL    DEFAULT 0,
                    super_net  REAL    DEFAULT 0,
                    big_net    REAL    DEFAULT 0,
                    mid_net    REAL    DEFAULT 0,
                    small_net  REAL    DEFAULT 0,
                    UNIQUE (symbol, date)
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fund_flow_date ON fund_flow(date)"
            )
            conn.commit()

        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
        all_rows: list[tuple] = []

        for page in range(1, 30):  # 最多30页
            try:
                r = requests.get(
                    "https://push2delay.eastmoney.com/api/qt/clist/get",
                    params={
                        "pn": page, "pz": 200, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                        "fs": "m:0+t:6+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2",
                        "fields": "f12,f14,f62,f184,f66,f72,f78,f81",
                    },
                    headers=headers, timeout=10,
                )
                data = r.json().get("data", {})
                diff = data.get("diff", [])
                if not diff:
                    break
                for item in diff:
                    sym = str(item.get("f12", "")).strip()
                    if not sym:
                        continue
                    try:
                        all_rows.append((
                            sym, target_date,
                            float(item.get("f62", 0) or 0),
                            float(item.get("f184", 0) or 0),
                            float(item.get("f66", 0) or 0),
                            float(item.get("f72", 0) or 0),
                            float(item.get("f78", 0) or 0),
                            float(item.get("f81", 0) or 0),
                        ))
                    except (ValueError, TypeError):
                        continue
                if len(all_rows) >= data.get("total", 0):
                    break
            except Exception as exc:
                logger.warning(f"资金流向拉取失败 page {page}: {exc}")
                break

        if all_rows:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    """INSERT OR REPLACE INTO fund_flow
                       (symbol, date, main_net, main_pct, super_net, big_net, mid_net, small_net)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    all_rows,
                )
                conn.commit()

        logger.info(f"资金流向 {target_date} 写入 {len(all_rows)} 行")
        return len(all_rows)

    def get_fund_flow(self, date_str: str | None = None) -> list[dict]:
        """查本地资金流向（按主力净流入排序）。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if date_str:
                rows = conn.execute(
                    """SELECT symbol, date, main_net, main_pct, super_net, big_net
                       FROM fund_flow WHERE date=? ORDER BY main_net DESC""",
                    (date_str,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT symbol, date, main_net, main_pct, super_net, big_net
                       FROM fund_flow WHERE date=(
                           SELECT MAX(date) FROM fund_flow
                       ) ORDER BY main_net DESC""",
                ).fetchall()
        return [dict(r) for r in rows]
