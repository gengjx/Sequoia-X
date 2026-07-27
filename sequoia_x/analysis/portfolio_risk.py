"""组合级风控引擎：Beta暴露 + 行业集中度 + 相关性 + 回撤预算 + VaR。

解决核心问题：当前系统只有单股止损，缺乏组合层面的系统性风险控制。
对标：掘金/聚宽 portfolio analyzer、米筐风险模型、VNPY风控模块。

5个风控维度：
  1. Beta暴露：持仓加权Beta vs 沪深300，>1.5告警、>2.0熔断
  2. 行业/板块集中度：单行业>40%告警、>50%熔断
  3. 持仓相关性矩阵：高相关组合缺乏分散，计算组合波动率
  4. 最大回撤预算：回撤>10%告警、>15%触发降仓
  5. VaR/CVaR：95%置信度1日最大亏损，超过总资产3%告警
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskAlert:
    """风控告警。"""
    level: str          # ok / warning / danger
    category: str       # beta / concentration / drawdown / var / correlation
    message: str
    metric: float
    threshold: float


@dataclass
class PortfolioRiskReport:
    """组合风控报告。"""
    total_value: float = 0
    cash: float = 0
    # Beta
    portfolio_beta: float = 0
    # 集中度
    top_industry: str = ""
    top_industry_pct: float = 0
    top_board: str = ""
    top_board_pct: float = 0
    holding_count: int = 0
    # 波动率
    portfolio_volatility: float = 0  # 年化波动率%
    avg_correlation: float = 0
    # VaR
    var_95: float = 0       # 95% VaR（元）
    cvar_95: float = 0      # 95% CVaR（元）
    var_pct: float = 0      # VaR占总资产%
    # 回撤
    current_drawdown: float = 0
    max_drawdown: float = 0
    # 评分
    risk_score: int = 100   # 100=最安全，0=最危险
    alerts: list[RiskAlert] = field(default_factory=list)
    # HHI集中度指数
    hhi: float = 0          # 赫芬达尔指数（0-10000，越高越集中）

    def to_dict(self) -> dict:
        return {
            "total_value": round(self.total_value, 0),
            "cash": round(self.cash, 0),
            "portfolio_beta": round(self.portfolio_beta, 2),
            "top_industry": self.top_industry,
            "top_industry_pct": round(self.top_industry_pct, 1),
            "top_board": self.top_board,
            "top_board_pct": round(self.top_board_pct, 1),
            "holding_count": self.holding_count,
            "portfolio_volatility": round(self.portfolio_volatility, 1),
            "avg_correlation": round(self.avg_correlation, 2),
            "var_95": round(self.var_95, 0),
            "cvar_95": round(self.cvar_95, 0),
            "var_pct": round(self.var_pct, 2),
            "current_drawdown": round(self.current_drawdown, 1),
            "max_drawdown": round(self.max_drawdown, 1),
            "risk_score": self.risk_score,
            "hhi": round(self.hhi, 0),
            "alerts": [
                {"level": a.level, "category": a.category, "message": a.message,
                 "metric": round(a.metric, 2), "threshold": a.threshold}
                for a in self.alerts
            ],
        }


class PortfolioRiskMonitor:
    """组合级风控监控器。

    数据源：
      - paper_holdings / portfolio_holding：持仓
      - stock_daily：日K收益序列（计算Beta/相关性/VaR）
      - stock_industry：行业分类
      - stock_board_em：细分板块
      - stock_market_cap：市值（权重计算）
    """

    # 风控阈值
    BETA_WARN = 1.3
    BETA_DANGER = 1.6
    INDUSTRY_WARN = 0.30      # 单行业30%
    INDUSTRY_DANGER = 0.45    # 单行业45%
    HHI_WARN = 2000           # HHI>2000=中等集中
    HHI_DANGER = 3500         # HHI>3500=高度集中
    VAR_PCT_WARN = 0.02       # VaR占总资产2%
    VAR_PCT_DANGER = 0.04     # VaR占总资产4%
    DD_WARN = 0.08            # 回撤8%
    DD_DANGER = 0.13          # 回撤13%

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def analyze(self, holdings: list[dict] | None = None,
                cash: float = 0) -> PortfolioRiskReport:
        """执行完整风控分析。

        Args:
            holdings: 持仓列表[{symbol, shares, entry_price, ...}]，None则从DB读
            cash: 当前现金
        """
        report = PortfolioRiskReport(cash=cash)

        if holdings is None:
            holdings = self._load_holdings()
        if not holdings:
            report.alerts.append(RiskAlert("ok", "general", "空仓无风险", 0, 0))
            return report

        # ── 基础数据 ──
        weights, total_value = self._calc_weights(holdings, cash)
        report.total_value = total_value + cash
        report.holding_count = len(holdings)

        # ── 1. Beta暴露 ──
        beta = self._calc_portfolio_beta(holdings, weights)
        report.portfolio_beta = beta
        if beta > self.BETA_DANGER:
            report.alerts.append(RiskAlert("danger", "beta",
                f"组合Beta={beta:.2f}，市场敏感度过高（>{self.BETA_DANGER}），下跌行情损失放大",
                beta, self.BETA_DANGER))
        elif beta > self.BETA_WARN:
            report.alerts.append(RiskAlert("warning", "beta",
                f"组合Beta={beta:.2f}偏高（>{self.BETA_WARN}），注意市场系统性风险",
                beta, self.BETA_WARN))

        # ── 2. 行业/板块集中度 ──
        industry_pct, top_ind = self._calc_concentration(holdings, weights, "stock_industry", "industry")
        report.top_industry = top_ind
        report.top_industry_pct = industry_pct * 100
        if industry_pct > self.INDUSTRY_DANGER:
            report.alerts.append(RiskAlert("danger", "concentration",
                f"行业「{top_ind}」占比{industry_pct*100:.0f}%过高（>{self.INDUSTRY_DANGER*100:.0f}%），行业风险集中",
                industry_pct, self.INDUSTRY_DANGER))
        elif industry_pct > self.INDUSTRY_WARN:
            report.alerts.append(RiskAlert("warning", "concentration",
                f"行业「{top_ind}」占比{industry_pct*100:.0f}%偏高",
                industry_pct, self.INDUSTRY_WARN))

        board_pct, top_bd = self._calc_concentration(holdings, weights, "stock_board_em", "board")
        report.top_board = top_bd
        report.top_board_pct = board_pct * 100

        # HHI集中度指数
        report.hhi = sum(w * w for w in weights.values()) * 10000
        if report.hhi > self.HHI_DANGER:
            report.alerts.append(RiskAlert("danger", "concentration",
                f"持仓集中度HHI={report.hhi:.0f}，过度集中于少数个股",
                report.hhi, self.HHI_DANGER))
        elif report.hhi > self.HHI_WARN:
            report.alerts.append(RiskAlert("warning", "concentration",
                f"持仓集中度HHI={report.hhi:.0f}偏高",
                report.hhi, self.HHI_WARN))

        # ── 3. 相关性 + 波动率 ──
        vol, avg_corr = self._calc_volatility_correlation(holdings, weights)
        report.portfolio_volatility = vol
        report.avg_correlation = avg_corr
        if avg_corr > 0.7 and len(holdings) >= 5:
            report.alerts.append(RiskAlert("warning", "correlation",
                f"持仓平均相关性={avg_corr:.2f}，分散化不足（同涨同跌风险）",
                avg_corr, 0.7))

        # ── 4. VaR / CVaR ──
        var, cvar = self._calc_var(holdings, weights, total_value)
        report.var_95 = var
        report.cvar_95 = cvar
        report.var_pct = var / report.total_value * 100 if report.total_value > 0 else 0
        if report.var_pct > self.VAR_PCT_DANGER * 100:
            report.alerts.append(RiskAlert("danger", "var",
                f"1日VaR(95%)=¥{var:,.0f}（占资产{report.var_pct:.1f}%），单日亏损风险过高",
                report.var_pct, self.VAR_PCT_DANGER * 100))
        elif report.var_pct > self.VAR_PCT_WARN * 100:
            report.alerts.append(RiskAlert("warning", "var",
                f"1日VaR(95%)=¥{var:,.0f}（占资产{report.var_pct:.1f}%）",
                report.var_pct, self.VAR_PCT_WARN * 100))

        # ── 5. 回撤 ──
        dd = self._calc_drawdown()
        report.current_drawdown = dd * 100
        if dd > self.DD_DANGER:
            report.alerts.append(RiskAlert("danger", "drawdown",
                f"当前回撤{dd*100:.1f}%，超过{self.DD_DANGER*100:.0f}%熔断线，建议降仓防守",
                dd * 100, self.DD_DANGER * 100))
        elif dd > self.DD_WARN:
            report.alerts.append(RiskAlert("warning", "drawdown",
                f"当前回撤{dd*100:.1f}%，接近风控阈值",
                dd * 100, self.DD_WARN * 100))

        # ── 综合评分 ──
        report.risk_score = self._calc_risk_score(report)
        return report

    # ════════════════════════════════════════════
    # 数据加载
    # ════════════════════════════════════════════
    def _load_holdings(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM paper_holdings WHERE account_id=1"
            ).fetchall()
        return [dict(r) for r in rows]

    def _calc_weights(self, holdings: list[dict], cash: float) -> tuple[dict[str, float], float]:
        """计算每只股票的市值权重。"""
        prices = {}
        with sqlite3.connect(self.db_path) as conn:
            for h in holdings:
                sym = h["symbol"]
                row = conn.execute(
                    "SELECT close FROM stock_daily WHERE symbol=? ORDER BY date DESC LIMIT 1",
                    (sym,)
                ).fetchone()
                prices[sym] = row[0] if row else h.get("entry_price", 0)

        values = {h["symbol"]: prices.get(h["symbol"], 0) * h["shares"] for h in holdings}
        total = sum(values.values())
        if total == 0:
            return {}, 0
        weights = {sym: val / (total + cash) for sym, val in values.items()}
        return weights, total

    # ════════════════════════════════════════════
    # Beta计算
    # ════════════════════════════════════════════
    def _calc_portfolio_beta(self, holdings: list[dict], weights: dict[str, float]) -> float:
        """计算组合加权Beta vs 沪深300。

        Beta = Cov(Ri, Rm) / Var(Rm)
        用最近60个交易日日收益率计算。
        """
        market_returns = self._get_market_returns()
        if len(market_returns) < 30:
            return 1.0  # 无足够数据，默认中性

        market_var = float(np.var(market_returns))
        if market_var == 0:
            return 1.0

        portfolio_beta = 0
        valid_count = 0
        for h in holdings:
            sym = h["symbol"]
            w = weights.get(sym, 0)
            if w == 0:
                continue
            stock_returns = self._get_stock_returns(sym)
            if len(stock_returns) < 30:
                continue
            # 对齐长度
            min_len = min(len(stock_returns), len(market_returns))
            cov = float(np.cov(stock_returns[-min_len:], market_returns[-min_len:])[0, 1])
            beta = cov / market_var if market_var > 0 else 1.0
            # 限制极端值
            beta = max(-2.0, min(3.0, beta))
            portfolio_beta += beta * w
            valid_count += 1

        if valid_count == 0:
            return 1.0
        return portfolio_beta

    # ════════════════════════════════════════════
    # 集中度计算
    # ════════════════════════════════════════════
    def _calc_concentration(self, holdings: list[dict], weights: dict[str, float],
                            table: str, col: str) -> tuple[float, str]:
        """计算行业/板块集中度，返回(最大占比, 名称)。"""
        with sqlite3.connect(self.db_path) as conn:
            for h in holdings:
                sym = h["symbol"]
                row = conn.execute(
                    f"SELECT {col} FROM {table} WHERE symbol=?", (sym,)
                ).fetchone()
                h["_category"] = row[0] if row else "未分类"

        cat_weights: dict[str, float] = {}
        for h in holdings:
            cat = h.get("_category", "未分类")
            cat_weights[cat] = cat_weights.get(cat, 0) + weights.get(h["symbol"], 0)

        if not cat_weights:
            return 0, ""
        top_cat = max(cat_weights, key=cat_weights.get)
        return cat_weights[top_cat], top_cat

    # ════════════════════════════════════════════
    # 波动率 + 相关性
    # ════════════════════════════════════════════
    def _calc_volatility_correlation(self, holdings: list[dict],
                                     weights: dict[str, float]) -> tuple[float, float]:
        """计算组合年化波动率和平均相关性。

        组合波动率 = sqrt(W^T × Σ × W)
        Σ = 个股收益率协方差矩阵
        """
        returns_data = {}
        for h in holdings:
            sym = h["symbol"]
            if weights.get(sym, 0) == 0:
                continue
            rets = self._get_stock_returns(sym)
            if len(rets) >= 30:
                returns_data[sym] = rets[-60:]  # 最近60日

        if len(returns_data) < 2:
            # 单只持仓，用自身波动率
            if len(returns_data) == 1:
                r = list(returns_data.values())[0]
                vol = float(np.std(r) * np.sqrt(252) * 100)
                return vol, 0
            return 0, 0

        # 对齐到等长
        min_len = min(len(r) for r in returns_data.values())
        df = pd.DataFrame({sym: r[-min_len:] for sym, r in returns_data.items()})
        cov_matrix = df.cov().values * 252  # 年化协方差

        w_vec = np.array([weights.get(sym, 0) for sym in df.columns])
        total_w = w_vec.sum()
        if total_w > 0:
            w_vec = w_vec / total_w
        portfolio_var = float(w_vec @ cov_matrix @ w_vec)
        portfolio_vol = np.sqrt(portfolio_var) * 100  # 转为%

        # 平均相关性（相关系数矩阵上三角均值）
        corr_matrix = df.corr().fillna(0).values
        n = len(df.columns)
        if n > 1:
            upper_tri = corr_matrix[np.triu_indices(n, k=1)]
            avg_corr = float(np.mean(np.abs(upper_tri)))
        else:
            avg_corr = 0

        return portfolio_vol, avg_corr

    # ════════════════════════════════════════════
    # VaR / CVaR（历史模拟法）
    # ════════════════════════════════════════════
    def _calc_var(self, holdings: list[dict], weights: dict[str, float],
                  total_value: float) -> tuple[float, float]:
        """历史模拟法计算95% VaR和CVaR。

        用最近250个交易日的组合收益率分布，取5%分位点。
        """
        # 构建组合日收益率序列
        returns_data = {}
        for h in holdings:
            sym = h["symbol"]
            if weights.get(sym, 0) == 0:
                continue
            rets = self._get_stock_returns(sym)
            if len(rets) >= 30:
                returns_data[sym] = rets

        if not returns_data:
            return 0, 0

        min_len = min(len(r) for r in returns_data.values())
        df = pd.DataFrame({sym: r[-min_len:] for sym, r in returns_data.items()})

        w_vec = np.array([weights.get(sym, 0) for sym in df.columns])
        total_w = w_vec.sum()
        if total_w > 0:
            w_vec = w_vec / total_w

        portfolio_daily = df.values @ w_vec

        var_5 = float(np.percentile(portfolio_daily, 5))
        cvar_5 = float(np.mean(portfolio_daily[portfolio_daily <= var_5]))

        # 转为金额（负收益率 × 总市值 = 亏损金额）
        var_amount = abs(var_5) * total_value
        cvar_amount = abs(cvar_5) * total_value

        return var_amount, cvar_amount

    # ════════════════════════════════════════════
    # 回撤
    # ════════════════════════════════════════════
    def _calc_drawdown(self) -> float:
        """基于NAV序列计算当前回撤。"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT total_assets FROM paper_nav ORDER BY date"
            ).fetchall()
        if len(rows) < 2:
            return 0
        assets = [r[0] for r in rows]
        current = assets[-1]
        peak = max(assets)
        if peak <= 0:
            return 0
        return (peak - current) / peak

    # ════════════════════════════════════════════
    # 综合评分
    # ════════════════════════════════════════════
    def _calc_risk_score(self, report: PortfolioRiskReport) -> int:
        """0-100 风险评分（100=最安全）。"""
        score = 100
        for alert in report.alerts:
            if alert.level == "danger":
                score -= 20
            elif alert.level == "warning":
                score -= 8
        return max(0, min(100, score))

    # ════════════════════════════════════════════
    # 辅助：收益率序列
    # ════════════════════════════════════════════
    def _get_stock_returns(self, symbol: str, days: int = 250) -> np.ndarray:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT close FROM stock_daily WHERE symbol=? "
                "ORDER BY date DESC LIMIT ?",
                (symbol, days + 1)
            ).fetchall()
        if len(rows) < 2:
            return np.array([])
        closes = np.array([r[0] for r in reversed(rows)])
        return np.diff(closes) / closes[:-1]

    def _get_market_returns(self, days: int = 250) -> np.ndarray:
        """沪深300真实收益率（从 index_daily 表读取，无则回退全市场等权近似）。"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT close FROM index_daily WHERE symbol='000300' "
                "ORDER BY date DESC LIMIT ?", (days + 1,)
            ).fetchall()
            if len(rows) >= 2:
                closes = np.array([r[0] for r in reversed(rows)])
                return np.diff(closes) / closes[:-1]
            # 回退：全市场等权均值近似
            rows = conn.execute(
                "SELECT date, AVG(pct_chg) as avg_ret FROM stock_daily "
                "WHERE pct_chg IS NOT NULL AND date >= "
                "(SELECT MAX(date) FROM stock_daily WHERE pct_chg IS NOT NULL) "
                f"GROUP BY date ORDER BY date DESC LIMIT {days}"
            ).fetchall()
        if not rows:
            return np.array([])
        rets = np.array([r[1] / 100 for r in reversed(rows)])
        return rets


class UnlockAvoidance:
    """解禁回避风控过滤器。

    检查个股未来 N 天内是否有大额解禁（占流通市值 > threshold），
    有则拦截买入（回避供给冲击风险）。

    解禁日期是公司公告的未来事件（非预测），as_of 日已知，无前视问题。
    """

    LOOKFORWARD_DAYS = 30
    RATIO_THRESHOLD = 0.20  # 解禁占比 > 20% 才拦截（p90量级）

    def __init__(self, db_path: str, lookforward_days: int | None = None,
                 ratio_threshold: float | None = None) -> None:
        self.db_path = db_path
        self.lookforward_days = lookforward_days or self.LOOKFORWARD_DAYS
        self.ratio_threshold = ratio_threshold if ratio_threshold is not None else self.RATIO_THRESHOLD
        # 预加载全部大额解禁事件（symbol -> [(release_date, ratio)]，按日期排序）
        self._events: dict[str, list[tuple[str, float]]] = self._load_events()

    def _load_events(self) -> dict[str, list[tuple[str, float]]]:
        """加载全部大额解禁事件（ratio > threshold），按 release_date 排序。"""
        import sqlite3 as _sq
        events: dict[str, list[tuple[str, float]]] = {}
        try:
            with _sq.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, release_date, unlock_ratio FROM restricted_release "
                    "WHERE unlock_ratio > ? ORDER BY symbol, release_date",
                    (self.ratio_threshold,),
                ).fetchall()
            for r in rows:
                events.setdefault(r[0], []).append((r[1], r[2]))
        except Exception:
            pass
        return events

    def should_avoid(self, symbol: str, as_of_date: str) -> bool:
        """检查该股在 [as_of_date, as_of_date + lookforward_days] 内是否有大额解禁。

        PIT 正确：只查 release_date >= as_of_date（解禁前的供给冲击预期），
        不查已过去的解禁（冲击已释放）。

        Args:
            symbol: 股票代码
            as_of_date: PIT 截止日期 YYYY-MM-DD

        Returns:
            True=应回避（有大额解禁），False=可买
        """
        import datetime as _dt
        evs = self._events.get(symbol)
        if not evs:
            return False
        try:
            ref = _dt.datetime.strptime(as_of_date, "%Y-%m-%d")
            deadline = (ref + _dt.timedelta(days=self.lookforward_days)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return False
        # 查 release_date ∈ [as_of_date, deadline]
        for release_date, _ratio in evs:
            if as_of_date <= release_date <= deadline:
                return True
        return False
