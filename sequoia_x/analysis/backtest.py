"""信号评分回测引擎：验证 signal_score / fear_greed_score 对次日大盘方向的预测力。

核心指标（对标主流量化系统因子评价体系）：
  - Rank IC（Spearman 秩相关）：因子值与次日收益的秩相关，抗异常值
  - IC均值 / ICIR（信息比率） / IC胜率
  - 分档回测（5档单调性验证）
  - 子指标独立 IC 贡献度

方法论约束：
  - 评分仅用当日及之前数据（无前视偏误）
  - 次日收益作为 label（前瞻变量，仅回测用）
  - 幸存者偏差：退市股可能缺失，基于现有库样本
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# 纯函数：评分公式（与 market.py 实盘完全一致，回测/实盘共享）
# ------------------------------------------------------------------
def signal_score(up_ratio: float, avg_change: float, limit_up: int, limit_down: int) -> float:
    """盘面信号评分（0-100），与 MarketAnalyzer._signal_score 数值一致。"""
    breadth_score = up_ratio
    change_score = max(0.0, min(100.0, (avg_change + 5) / 10 * 100))
    lim = limit_up + limit_down
    limit_score = (limit_up / lim * 100) if lim else 50.0
    return round(0.45 * breadth_score + 0.30 * change_score + 0.25 * limit_score)


def fear_greed_score(
    limit_up: int, limit_down: int, up_ratio: float,
    nh: int, nl: int, decided: int,
    ma20_pct: float, ma60_pct: float,
) -> float:
    """恐贪指数（0-100），与 MarketAnalyzer._sentiment_score 数值一致。"""
    lim = limit_up + limit_down
    s1 = (limit_up / lim * 100) if lim else 50.0
    s2 = up_ratio
    nhnl_pct = (nh - nl) / decided * 100 if decided else 0
    s3 = max(0.0, min(100.0, (nhnl_pct + 10) / 20 * 100))
    s4 = ma20_pct
    s5 = ma60_pct
    return round(0.20 * s1 + 0.25 * s2 + 0.20 * s3 + 0.20 * s4 + 0.15 * s5)


@dataclass
class BacktestReport:
    """信号评分回测报告。"""
    sample_days: int
    date_range: str
    # 整体 IC 统计
    signal_ic: dict = field(default_factory=dict)
    fear_greed_ic: dict = field(default_factory=dict)
    # 分档回测
    signal_quantiles: list[dict] = field(default_factory=list)
    fear_greed_quantiles: list[dict] = field(default_factory=list)
    # 子指标独立 IC
    subindicator_ic: list[dict] = field(default_factory=list)
    # 结论
    conclusion: dict = field(default_factory=dict)
    # 权重优化（IC加权 + 全样本/OOS对比）
    optimization: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sample_days": self.sample_days,
            "date_range": self.date_range,
            "signal_ic": self.signal_ic,
            "fear_greed_ic": self.fear_greed_ic,
            "signal_quantiles": self.signal_quantiles,
            "fear_greed_quantiles": self.fear_greed_quantiles,
            "subindicator_ic": self.subindicator_ic,
            "conclusion": self.conclusion,
            "optimization": self.optimization,
        }


class SignalBacktester:
    """信号评分回测器：基于本地全量行情，向量化计算历史评分并评估预测力。"""

    def __init__(self, settings: Settings) -> None:
        self.db_path: str = settings.db_path

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self, min_days: int = 120) -> dict:
        """执行完整回测，返回结构化报告。"""
        daily = self._build_daily_metrics()
        if len(daily) < min_days:
            return {"error": f"历史数据不足：仅 {len(daily)} 日，需 ≥{min_days} 日"}

        report = BacktestReport(
            sample_days=len(daily),
            date_range=f"{daily.index[0]} ~ {daily.index[-1]}",
        )

        # 1. 整体 IC
        report.signal_ic = self._compute_ic(daily, "signal_score", "next_avg_chg")
        report.fear_greed_ic = self._compute_ic(daily, "fear_greed_score", "next_avg_chg")

        # 2. 分档回测
        report.signal_quantiles = self._compute_quantiles(daily, "signal_score")
        report.fear_greed_quantiles = self._compute_quantiles(daily, "fear_greed_score")

        # 3. 子指标独立 IC
        report.subindicator_ic = self._compute_subindicator_ic(daily)

        # 4. 权重优化（IC加权，含样本外验证）
        report.optimization = self._optimize_weights(daily)

        # 5. 结论
        report.conclusion = self._build_conclusion(report)

        logger.info(
            f"回测完成：{len(daily)} 日样本，"
            f"signal IC={report.signal_ic['ic_mean']:.3f} ICIR={report.signal_ic['icir']:.3f}，"
            f"fear_greed IC={report.fear_greed_ic['ic_mean']:.3f} ICIR={report.fear_greed_ic['icir']:.3f}"
        )
        return report.to_dict()

    # ------------------------------------------------------------------
    # 1. 批量构建日级指标（pandas 向量化，零网络）
    # ------------------------------------------------------------------
    def _build_daily_metrics(self) -> pd.DataFrame:
        """加载全量行情，向量化计算所有历史子指标，聚合为日级 DataFrame。"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT symbol, date, open, high, low, close, turnover FROM stock_daily ORDER BY symbol, date",
                conn,
            )
            try:
                basic = pd.read_sql_query(
                    "SELECT symbol, name, ipo_date FROM stock_basic", conn
                )
            except Exception:
                basic = pd.DataFrame(columns=["symbol", "name", "ipo_date"])

        logger.info(f"加载行情：{len(df)} 行，{df['symbol'].nunique()} 只股票")

        # 涨跌幅（后复权下比例不变）
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df["prev_close"] = df.groupby("symbol")["close"].shift(1)
        df = df.dropna(subset=["prev_close"])
        df["pct"] = (df["close"] - df["prev_close"]) / df["prev_close"] * 100
        df["high_pct"] = (df["high"] - df["prev_close"]) / df["prev_close"] * 100

        # 限幅阈值（板块 + ST）
        df = df.merge(basic[["symbol", "name", "ipo_date"]], on="symbol", how="left")
        is_st = df["name"].fillna("").str.upper().str.contains("ST")
        df["threshold"] = 9.7
        df.loc[df["symbol"].str.startswith(("30", "68")), "threshold"] = 19.5
        df.loc[df["symbol"].str.startswith(("8", "4")), "threshold"] = 29.0
        df.loc[is_st, "threshold"] = 4.6

        # 新股过滤（上市 ≤5 日）
        df["ipo_date"] = pd.to_datetime(df["ipo_date"], errors="coerce")
        df["date_dt"] = pd.to_datetime(df["date"])
        df["age"] = (df["date_dt"] - df["ipo_date"]).dt.days
        df["valid"] = df["age"].fillna(999) > 5

        # 涨跌停标记
        v = df["valid"]
        df["is_up"] = (df["close"] > df["prev_close"]) & v
        df["is_down"] = (df["close"] < df["prev_close"]) & v
        df["limit_up"] = (df["pct"] >= df["threshold"]) & v
        df["limit_down"] = (df["pct"] <= -df["threshold"]) & v

        # NH-NL：52 周滚动极值
        df["max_h_252"] = df.groupby("symbol")["high"].transform(
            lambda x: x.rolling(252, min_periods=60).max()
        )
        df["min_l_252"] = df.groupby("symbol")["low"].transform(
            lambda x: x.rolling(252, min_periods=60).min()
        )
        df["is_nh"] = (df["high"] >= df["max_h_252"]) & df["max_h_252"].notna() & v
        df["is_nl"] = (df["low"] <= df["min_l_252"]) & df["min_l_252"].notna() & v

        # MA20 / MA60
        df["ma20"] = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(20, min_periods=20).mean()
        )
        df["ma60"] = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(60, min_periods=60).mean()
        )
        df["above_ma20"] = (df["close"] > df["ma20"]) & df["ma20"].notna() & v
        df["above_ma60"] = (df["close"] > df["ma60"]) & df["ma60"].notna() & v

        # 日级聚合
        g = df.groupby("date")
        total = g["valid"].sum()
        up = g["is_up"].sum()
        down = g["is_down"].sum()
        decided = up + down
        daily = pd.DataFrame({
            "up": up,
            "down": down,
            "total": total,
            "decided": decided,
            "up_ratio": (up / decided * 100).fillna(50.0),
            "avg_change": g["pct"].mean(),
            "limit_up": g["limit_up"].sum(),
            "limit_down": g["limit_down"].sum(),
            "nh": g["is_nh"].sum(),
            "nl": g["is_nl"].sum(),
            "ma20_above": g["above_ma20"].sum(),
            "ma60_above": g["above_ma60"].sum(),
            "turnover_yi": g["turnover"].sum() / 1e8,
        })
        daily["ma20_pct"] = (daily["ma20_above"] / daily["total"] * 100).fillna(50.0)
        daily["ma60_pct"] = (daily["ma60_above"] / daily["total"] * 100).fillna(50.0)

        # 计算评分（复用纯函数，与实盘一致）
        daily["signal_score"] = daily.apply(
            lambda r: signal_score(r["up_ratio"], r["avg_change"], r["limit_up"], r["limit_down"]),
            axis=1,
        )
        daily["fear_greed_score"] = daily.apply(
            lambda r: fear_greed_score(
                r["limit_up"], r["limit_down"], r["up_ratio"],
                r["nh"], r["nl"], r["decided"], r["ma20_pct"], r["ma60_pct"],
            ),
            axis=1,
        )

        # 次日收益（label，前瞻变量）
        daily["next_avg_chg"] = daily["avg_change"].shift(-1)
        daily = daily.dropna(subset=["next_avg_chg"])

        logger.info(f"日级指标构建完成：{len(daily)} 个有效交易日")
        return daily

    # ------------------------------------------------------------------
    # 2. IC 统计
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_ic(daily: pd.DataFrame, factor: str, target: str) -> dict:
        """计算 Rank IC（Spearman）及统计量。"""
        valid = daily[[factor, target]].dropna()
        if len(valid) < 30:
            return {"error": "样本不足"}
        # Spearman = Pearson on ranked data（避免 scipy 依赖）
        ic_mean = float(valid[factor].rank().corr(valid[target].rank()))
        # 逐月 IC（滚动窗口评估稳定性）
        monthly = valid.groupby(valid.index.to_series().str[:7]).apply(
            lambda x: x[factor].rank().corr(x[target].rank()) if len(x) >= 5 else np.nan,
            include_groups=False,
        ).dropna()
        ic_std = float(monthly.std()) if len(monthly) >= 2 else 0.0
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        win_rate = float((monthly > 0).sum() / len(monthly) * 100) if len(monthly) else 0.0
        return {
            "ic_mean": round(ic_mean, 4),
            "ic_std": round(ic_std, 4),
            "icir": round(icir, 4),
            "win_rate": round(win_rate, 1),
            "monthly_count": len(monthly),
            "assessment": _ic_assessment(ic_mean, icir, win_rate),
        }

    # ------------------------------------------------------------------
    # 3. 分档回测（单调性验证）
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_quantiles(daily: pd.DataFrame, factor: str, n_bins: int = 5) -> list[dict]:
        """按因子值分 N 档，统计各档次日收益均值、胜率、样本数。"""
        valid = daily[[factor, "next_avg_chg"]].dropna().copy()
        if len(valid) < n_bins * 5:
            return []
        valid["bin"] = pd.qcut(valid[factor], n_bins, labels=False, duplicates="drop")
        results = []
        for b in sorted(valid["bin"].unique()):
            sub = valid[valid["bin"] == b]
            rets = sub["next_avg_chg"]
            results.append({
                "bin": int(b),
                "count": len(sub),
                "factor_range": f"{sub[factor].min():.0f}~{sub[factor].max():.0f}",
                "next_avg_return": round(float(rets.mean()), 3),
                "next_win_rate": round(float((rets > 0).sum() / len(rets) * 100), 1),
            })
        return results

    # ------------------------------------------------------------------
    # 4. 子指标独立 IC 贡献度
    # ------------------------------------------------------------------
    def _compute_subindicator_ic(self, daily: pd.DataFrame) -> list[dict]:
        """计算各子指标对次日收益的独立 IC，用于诊断权重合理性。"""
        subs = {
            "up_ratio（上涨占比）": "up_ratio",
            "avg_change（平均涨跌幅）": "avg_change",
            "limit_ratio（涨跌停比）": None,
            "nh_nl（净新高）": None,
            "ma20_pct（站上MA20占比）": "ma20_pct",
            "ma60_pct（站上MA60占比）": "ma60_pct",
            "turnover_yi（成交额）": "turnover_yi",
        }
        results = []
        for label, col in subs.items():
            if col is None:
                # 派生指标
                if "limit_ratio" in label:
                    lim = daily["limit_up"] + daily["limit_down"]
                    series = (daily["limit_up"] / lim.replace(0, np.nan) * 100).fillna(50)
                elif "nh_nl" in label:
                    series = (daily["nh"] - daily["nl"]).fillna(0)
            else:
                series = daily[col]
            valid = pd.DataFrame({"f": series, "t": daily["next_avg_chg"]}).dropna()
            if len(valid) < 30:
                continue
            ic = valid["f"].rank().corr(valid["t"].rank())
            results.append({
                "indicator": label,
                "ic": round(float(ic), 4),
                "abs_ic": round(abs(float(ic)), 4),
            })
        results.sort(key=lambda x: x["abs_ic"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # 5a. 权重优化（IC加权 + 全样本/OOS 双轨验证）
    # ------------------------------------------------------------------
    # 评分子指标定义：(逻辑名, 0-100分项列名)
    # 分项列在 _build_subscores 中构造，映射逻辑与实盘 _signal_score/_sentiment_score 一致
    _SIGNAL_SUBS: list[tuple[str, str]] = [
        ("up_ratio", "s_up_ratio"),
        ("avg_change", "s_avg_change"),
        ("limit_ratio", "s_limit_ratio"),
    ]
    _FEAR_GREED_SUBS: list[tuple[str, str]] = [
        ("limit_ratio", "s_limit_ratio"),
        ("up_ratio", "s_up_ratio"),
        ("nh_nl", "s_nh_nl"),
        ("ma20_pct", "s_ma20_pct"),
        ("ma60_pct", "s_ma60_pct"),
    ]

    @staticmethod
    def _build_subscores(daily: pd.DataFrame) -> pd.DataFrame:
        """将原始子指标映射为 0-100 看多方向分项（与实盘映射一致）。"""
        d = daily.copy()
        d["s_up_ratio"] = d["up_ratio"]
        d["s_avg_change"] = np.clip((d["avg_change"] + 5) / 10 * 100, 0, 100)
        lim = d["limit_up"] + d["limit_down"]
        d["s_limit_ratio"] = (d["limit_up"] / lim * 100).fillna(50).clip(0, 100)
        nhnl_pct = (d["nh"] - d["nl"]) / d["decided"] * 100
        d["s_nh_nl"] = np.clip((nhnl_pct + 10) / 20 * 100, 0, 100)
        d["s_ma20_pct"] = d["ma20_pct"]
        d["s_ma60_pct"] = d["ma60_pct"]
        return d

    def _optimize_weights(self, daily: pd.DataFrame) -> dict:
        """IC加权优化：全样本（理论天花板）+ 样本外（真实可达）双轨验证。

        方法论：w_i = max(IC_i, 0) / Σ max(IC_j, 0)，仅保留正向预测力子指标。
        OOS：前半段定权重，后半段测IC，杜绝数据窥探。
        """
        d = self._build_subscores(daily)
        target = "next_avg_chg"
        split = len(d) // 2
        return {
            "signal_score": self._optimize_one(d, self._SIGNAL_SUBS, target, split),
            "fear_greed": self._optimize_one(d, self._FEAR_GREED_SUBS, target, split),
        }

    def _optimize_one(
        self, daily: pd.DataFrame, subs: list[tuple[str, str]],
        target: str, split: int,
    ) -> dict:
        """对单一评分做 IC 加权优化与三段对比。"""
        # 原权重（实盘当前权重，硬编码以避免循环依赖）
        # 注：signal = 0.45/0.30/0.25；fear_greed = 0.20/0.25/0.20/0.20/0.15
        original = {
            "signal_score": {"up_ratio": 0.45, "avg_change": 0.30, "limit_ratio": 0.25},
            "fear_greed": {
                "limit_ratio": 0.20, "up_ratio": 0.25, "nh_nl": 0.20,
                "ma20_pct": 0.20, "ma60_pct": 0.15,
            },
        }
        name_key = "fear_greed" if len(subs) == 5 else "signal_score"
        orig_w = original[name_key]

        # 各子指标全样本 IC
        ic_full = {}
        for label, col in subs:
            valid = daily[[col, target]].dropna()
            ic_full[label] = float(valid[col].rank().corr(valid[target].rank())) if len(valid) >= 30 else 0.0

        # IC加权权重：只保留正IC
        pos = {k: v for k, v in ic_full.items() if v > 0}
        total = sum(pos.values())
        opt_w = {k: (v / total if total > 0 else 0.0) for k, v in ic_full.items()}
        if not pos:  # 无正IC子指标，回退等权
            opt_w = {k: 1.0 / len(subs) for k, _ in subs}

        def _weighted_score(df: pd.DataFrame, weights: dict) -> pd.Series:
            s = pd.Series(0.0, index=df.index)
            for label, col in subs:
                s += df[col] * weights.get(label, 0)
            return s

        # 三段评分
        score_orig = _weighted_score(daily, orig_w)
        score_full = _weighted_score(daily, opt_w)

        # OOS：前半段定权重（已在opt_w用全样本算，这里严格用前半段重算）
        ic_train = {}
        for label, col in subs:
            train = daily.iloc[:split][[col, target]].dropna()
            ic_train[label] = float(train[col].rank().corr(train[target].rank())) if len(train) >= 20 else 0.0
        pos_tr = {k: v for k, v in ic_train.items() if v > 0}
        tot_tr = sum(pos_tr.values())
        oos_w = {k: (v / tot_tr if tot_tr > 0 else 0.0) for k, v in ic_train.items()}
        if not pos_tr:
            oos_w = {k: 1.0 / len(subs) for k, _ in subs}
        score_oos = _weighted_score(daily, oos_w)

        # 后半段（OOS测试集）IC
        test = daily.iloc[split:]
        def _ic(df, score):
            v = pd.DataFrame({"f": score, "t": df[target]}).dropna()
            return round(float(v["f"].rank().corr(v["t"].rank())), 4) if len(v) >= 20 else 0.0
        # 原权重全样本IC、全样本IC加权全样本IC、OOS权重后半段IC
        return {
            "original_weights": orig_w,
            "optimized_weights": {k: round(v, 3) for k, v in opt_w.items()},
            "oos_weights": {k: round(v, 3) for k, v in oos_w.items()},
            "indicator_ic": {k: round(v, 4) for k, v in ic_full.items()},
            "ic_comparison": {
                "original_full": _ic(daily, score_orig),
                "optimized_full": _ic(daily, score_full),
                "oos_test": _ic(test, score_oos.iloc[split:]),
            },
            "improvement": round(_ic(daily, score_full) - _ic(daily, score_orig), 4),
        }

    # ------------------------------------------------------------------
    # 5. 结论
    # ------------------------------------------------------------------
    @staticmethod
    def _build_conclusion(report: BacktestReport) -> dict:
        sig = report.signal_ic
        fg = report.fear_greed_ic
        parts = []

        def _assess(name: str, ic: dict) -> None:
            if "error" in ic:
                parts.append(f"{name}：样本不足，无法评估。")
                return
            parts.append(
                f"{name}：IC={ic['ic_mean']:.3f}，ICIR={ic['icir']:.3f}，"
                f"月胜率={ic['win_rate']:.0f}%（{ic['assessment']}）。"
            )

        _assess("盘面信号评分", sig)
        _assess("恐贪指数", fg)

        # 单调性检验
        for name, quants in [("signal", report.signal_quantiles), ("fear_greed", report.fear_greed_quantiles)]:
            if len(quants) >= 2:
                rets = [q["next_avg_return"] for q in quants]
                monotonic = all(rets[i] <= rets[i + 1] for i in range(len(rets) - 1)) or \
                            all(rets[i] >= rets[i + 1] for i in range(len(rets) - 1))
                parts.append(
                    f"{name} 分档{'呈单调性' if monotonic else '非单调，区分力不足'}"
                    f"（最低档均收益 {rets[0]:.2f}%，最高档 {rets[-1]:.2f}%）。"
                )

        # 子指标建议
        if report.subindicator_ic:
            best = report.subindicator_ic[0]
            worst = report.subindicator_ic[-1]
            parts.append(
                f"预测力最强子指标：{best['indicator']}（IC={best['ic']:.3f}）；"
                f"最弱：{worst['indicator']}（IC={worst['ic']:.3f}）。"
            )

        # 权重优化解读
        opt = report.optimization
        if opt:
            for name, key in [("盘面信号", "signal_score"), ("恐贪指数", "fear_greed")]:
                o = opt.get(key, {})
                cmp = o.get("ic_comparison", {})
                if cmp:
                    parts.append(
                        f"{name}权重优化：原 IC={cmp.get('original_full', 0):.3f} → "
 f"IC加权全样本 IC={cmp.get('optimized_full', 0):.3f}（"
 f"样本外 IC={cmp.get('oos_test', 0):.3f}）。"
                    )

        return {"text": " ".join(parts)}


def _ic_assessment(ic_mean: float, icir: float, win_rate: float) -> str:
    """根据 IC 统计量给出有效性评级。"""
    if icir >= 1.0 and win_rate >= 70:
        return "强有效因子"
    if icir >= 0.5 and win_rate >= 55:
        return "有效因子"
    if abs(ic_mean) >= 0.03 and win_rate >= 50:
        return "弱有效，可优化权重"
    return "预测力不足"
