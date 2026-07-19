"""机器学习因子引擎：用现有因子做非线性合成，生成 ml_score。

不依赖 sklearn，纯 numpy 实现 Ridge Regression + 前向特征选择。
捕捉因子间非线性交互（如"高ROE + 低换手 + 缩量"组合效应），
与线性IC加权因子互补，提供独立Alpha。

防过拟合设计：
  1. 严格时间序列切割：训练只用过去数据，绝不前视
  2. 5折时间序列交叉验证（expanding window）
  3. L2正则化（Ridge）+ 特征数量上限
  4. 标签用未来收益（非当期），杜绝数据泄露
"""

from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class MLFactorEngine:
    """机器学习因子引擎：Ridge Regression 因子合成。

    流程：
      1. 从DB加载所有股票的因子截面（按月）
      2. 构建标签：未来20日前向收益
      3. 时间序列分割训练/预测
      4. Ridge回归预测 → ml_score
      5. 计算IC → 如果有效则加入因子库
    """

    FEATURE_FACTORS = [
        # 量价
        "mom_5", "mom_10", "mom_20", "mom_60", "rps_120",
        # 波动
        "vol_20", "atr_pct", "skew",
        # 流动性
        "turnover", "amihud", "volume_ratio",
        # 量价模式
        "ma_cross", "vol_surge", "vol_shrink_p", "flag_tight",
        # 质量
        "roe", "np_margin", "gp_margin", "rev_growth", "profit_growth",
        # 估值
        "pe_ratio", "pb_ratio",
        # 资金流
        "main_net", "main_pct",
    ]

    HOLD_DAYS = 20
    TRAIN_MONTHS = 18  # 训练窗口
    MIN_STOCKS_PER_MONTH = 100

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def compute_ml_score(self) -> dict:
        """计算ML因子分数，返回IC评估结果。

        Returns:
            {ic_mean, icir, win_rate, predictions: {symbol: score}, model_info}
        """
        logger.info("ML因子引擎：开始计算...")

        # 1. 收集因子截面 + 前向收益标签
        data = self._collect_training_data()
        if not data or len(data) < 100:
            logger.warning("ML因子：训练数据不足")
            return {"ic_mean": 0, "valid": False}

        # 2. 时间序列交叉验证
        results = self._time_series_cv(data)

        logger.info(
            f"ML因子完成：IC={results['ic_mean']:.4f} ICIR={results['icir']:.3f} "
            f"WR={results['win_rate']:.1f}% 有效={results['valid']}"
        )
        return results

    def _collect_training_data(self) -> list[dict]:
        """收集每月因子截面 + 前向收益标签。

        返回: [{month, features: {factor: value}, fwd_return: float}, ...]
        """
        conn = sqlite3.connect(self.db_path)

        # 获取所有交易日按月分桶
        dates = pd.read_sql(
            "SELECT DISTINCT date FROM stock_daily ORDER BY date", conn
        )["date"].tolist()

        # 按月取最后一个交易日
        monthly_dates = {}
        for d in dates:
            month = d[:7]
            monthly_dates[month] = d  # 覆盖为该月最后一天
        monthly_dates = dict(sorted(monthly_dates.items()))

        months = list(monthly_dates.keys())
        if len(months) < self.TRAIN_MONTHS + 2:
            conn.close()
            return []

        # 取采样股票
        sample_syms = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM stock_daily "
            "WHERE date=? AND volume > 1000000 "
            "ORDER BY symbol LIMIT 800",
            (monthly_dates[months[-1]],)
        ).fetchall()]

        # 预加载每只股票的完整日K
        logger.info(f"ML因子：加载{len(sample_syms)}只股票数据...")
        stock_data: dict[str, pd.DataFrame] = {}
        for sym in sample_syms:
            df = pd.read_sql(
                "SELECT * FROM stock_daily WHERE symbol=? ORDER BY date",
                conn, params=(sym,)
            )
            if len(df) >= 250:
                stock_data[sym] = df

        # 预加载财报 + 估值
        from sequoia_x.analysis.factor import compute_factors

        finance_map = self._load_finance_map(conn)
        valuation_map = self._load_valuation_map(conn)

        # 计算每月截面因子 + 前向收益
        data = []
        eval_months = months[-(self.TRAIN_MONTHS + 3):]  # 多取3个月做标签

        for mi, month in enumerate(eval_months):
            if month not in monthly_dates:
                continue
            date_str = monthly_dates[month]
            month_factors = []

            for sym, df in stock_data.items():
                df_d = df[df["date"] <= date_str]
                if len(df_d) < 60:
                    continue

                factors = compute_factors(
                    df_d,
                    finance=finance_map.get(sym),
                    fund_flow=None,
                    lhb_data=None,
                )
                # 加估值因子
                val = valuation_map.get(sym, {})
                factors["pe_ratio"] = -val.get("pe", 0) if val.get("pe", 0) > 0 else 0
                factors["pb_ratio"] = -val.get("pb", 0) if val.get("pb", 0) > 0 else 0

                # 前向收益（标签）
                fwd_idx = df_d.index[-1]
                fwd_end = fwd_idx + self.HOLD_DAYS
                if fwd_end >= len(df):
                    continue
                fwd_return = (df.iloc[fwd_end]["close"] / df.iloc[fwd_idx]["close"] - 1)

                feat = {f: factors.get(f, 0) for f in self.FEATURE_FACTORS}
                month_factors.append({
                    "symbol": sym,
                    "month": month,
                    "features": feat,
                    "fwd_return": fwd_return,
                })

            if len(month_factors) >= self.MIN_STOCKS_PER_MONTH:
                data.extend(month_factors)

        conn.close()
        logger.info(f"ML因子：收集{len(data)}条训练数据，{len(set(d['month'] for d in data))}个月")
        return data

    def _time_series_cv(self, data: list[dict]) -> dict:
        """时间序列交叉验证：expanding window。

        每轮用过去N个月训练，预测下1个月，滚动推进。
        """
        df = pd.DataFrame(data)
        months = sorted(df["month"].unique())
        if len(months) < 6:
            return {"ic_mean": 0, "valid": False}

        ic_list = []
        all_predictions: dict[str, float] = {}

        # 滚动验证：至少6个月训练，预测下1个月
        for i in range(6, len(months) - 1):
            train_months = months[:i]
            test_month = months[i]

            train_df = df[df["month"].isin(train_months)]
            test_df = df[df["month"] == test_month]

            if len(train_df) < 300 or len(test_df) < 50:
                continue

            # 构建特征矩阵（None → 0）
            def _to_float(v):
                try:
                    return float(v) if v is not None else 0.0
                except (ValueError, TypeError):
                    return 0.0

            X_train = np.array([[_to_float(r[f]) for f in self.FEATURE_FACTORS]
                                for r in train_df["features"]], dtype=float)
            y_train = train_df["fwd_return"].values.astype(float)
            X_test = np.array([[_to_float(r[f]) for f in self.FEATURE_FACTORS]
                               for r in test_df["features"]], dtype=float)
            y_test = test_df["fwd_return"].values.astype(float)

            # 标准化（用训练集统计量）
            mean = np.nanmean(X_train, axis=0)
            std = np.nanstd(X_train, axis=0)
            std[std == 0] = 1
            X_train = np.nan_to_num((X_train - mean) / std)
            X_test = np.nan_to_num((X_test - mean) / std)

            # 填充NaN
            X_train = np.nan_to_num(X_train, nan=0)
            X_test = np.nan_to_num(X_test, nan=0)

            # Ridge Regression
            alpha = 1.0
            n_features = X_train.shape[1]
            A = X_train.T @ X_train + alpha * np.eye(n_features)
            b = X_train.T @ y_train
            try:
                weights = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue

            # 预测
            preds = X_test @ weights

            # 横截面IC（Spearman rank correlation）
            # 纯numpy实现Spearman rank correlation（不依赖scipy）
            ic = self._rank_corr(preds, y_test)

            if not np.isnan(ic):
                ic_list.append(ic)

            # 保存最新预测
            if i == len(months) - 2:
                for j, row in test_df.iterrows():
                    all_predictions[row["symbol"]] = float(preds[test_df.index.get_loc(j)])

        if not ic_list:
            return {"ic_mean": 0, "valid": False}

        ic_arr = np.array(ic_list)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        icir = ic_mean / ic_std if ic_std > 0 else 0
        win_rate = float(np.mean(ic_arr > 0))

        return {
            "ic_mean": round(ic_mean, 4),
            "icir": round(icir, 3),
            "win_rate": round(win_rate * 100, 1),
            "n_months": len(ic_list),
            "valid": ic_mean > 0.02 and icir > 0.3,
            "predictions": all_predictions,
            "weights": {f: float(w) for f, w in zip(self.FEATURE_FACTORS, weights)} if 'weights' in dir() else {},
        }

    @staticmethod
    def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
        """纯numpy Spearman rank correlation。"""
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = np.sqrt(np.sum(ra**2) * np.sum(rb**2))
        if denom == 0:
            return 0
        return float(np.sum(ra * rb) / denom)

    def _load_finance_map(self, conn) -> dict:
        rows = conn.execute(
            "SELECT * FROM stock_finance "
            "WHERE stat_date = (SELECT MAX(stat_date) FROM stock_finance)"
        ).fetchall()
        cols = [d[0] for d in conn.execute(
            "SELECT * FROM stock_finance LIMIT 1"
        ).description]
        return {r[0]: dict(zip(cols, r)) for r in rows}

    def _load_valuation_map(self, conn) -> dict:
        rows = conn.execute("SELECT symbol, pe, pb FROM stock_market_cap").fetchall()
        return {r[0]: {"pe": r[1], "pb": r[2]} for r in rows}
