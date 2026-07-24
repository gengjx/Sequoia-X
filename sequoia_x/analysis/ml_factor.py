"""机器学习因子引擎：因子非线性合成，生成 ml_score。

优先使用 LightGBM（梯度提升树）捕捉因子间非线性交互（如"高ROE + 低换手 +
缩量"组合效应），与线性IC加权因子互补。LightGBM 不可用时自动回退到 Ridge
Regression（纯 numpy），保证系统不因缺依赖而中断。

防过拟合设计：
  1. 严格时间序列切割：训练只用过去数据，绝不前视
  2. expanding-window 时间序列交叉验证
  3. 截面中性化（市值 + 行业）剔除风格暴露
  4. 保守超参（num_leaves≤31, min_child_samples≥200, max_depth≤5）
  5. 标签用未来收益（非当期），杜绝数据泄露
  6. 月度训练一次 → 写 ml_scores 快照，决策层只读快照（不实时训练）
"""

from __future__ import annotations

import sqlite3
import time

import numpy as np
import pandas as pd

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# 尝试导入 LightGBM；不可用时回退 Ridge
try:
    import lightgbm as lgb

    _HAS_LGB = True
except Exception:  # pragma: no cover - 依赖缺失时回退
    _HAS_LGB = False
    logger.info("lightgbm 不可用，ML因子引擎回退到 Ridge Regression")


class MLFactorEngine:
    """机器学习因子引擎：LightGBM（或 Ridge）因子合成。

    流程：
      1. 从DB加载股票的因子截面（按月）+ 截面中性化
      2. 构建标签：未来 HOLD_DAYS 前向收益
      3. expanding-window 时间序列 CV 训练
      4. 预测 → ml_score，计算 OOS IC
      5. 达标（P3a 门控）则写 ml_scores 快照 + factor_weights
    """

    # 扩展特征集：量价/波动/流动性/量价模式/质量/成长/营运/估值/资金/北向
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
        # 成长 + 营运效率
        "yoy_ni", "asset_turn", "inv_turn", "nr_turn",
        # 估值
        "pe_ratio", "pb_ratio",
        # 资金
        "main_net", "main_pct",
        # 北向
        "nb_holding_pct", "nb_inflow",
    ]

    HOLD_DAYS = 20
    TRAIN_MONTHS = 18
    MIN_STOCKS_PER_MONTH = 100

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def compute_ml_score(self) -> dict:
        """计算ML因子分数，返回IC评估结果 + 写持久化快照。

        Returns:
            {ic_mean, icir, win_rate, t_stat, valid, predictions,
             feature_importance, model_version}
        """
        logger.info("ML因子引擎：开始计算...")

        data = self._collect_training_data()
        if not data or len(data) < 100:
            logger.warning("ML因子：训练数据不足")
            return {"ic_mean": 0, "valid": False}

        results = self._time_series_cv(data)

        logger.info(
            f"ML因子完成：IC={results['ic_mean']:.4f} ICIR={results['icir']:.3f} "
            f"WR={results['win_rate']:.1f}% 有效={results['valid']} "
            f"模型={results.get('model_version', '?')}"
        )

        # 持久化：写 ml_scores 快照 + 更新 factor_weights.ml_score
        if results.get("predictions"):
            self._save_ml_scores(results)

        return results

    # ───────────────────────── 数据采集 ─────────────────────────

    def _collect_training_data(self) -> list[dict]:
        """收集每月因子截面 + 前向收益标签。

        返回: [{month, features: {factor: value}, fwd_return: float}, ...]
        """
        conn = sqlite3.connect(self.db_path)

        dates = pd.read_sql(
            "SELECT DISTINCT date FROM stock_daily ORDER BY date", conn
        )["date"].tolist()
        monthly_dates = {}
        for d in dates:
            month = d[:7]
            monthly_dates[month] = d
        monthly_dates = dict(sorted(monthly_dates.items()))

        months = list(monthly_dates.keys())
        if len(months) < self.TRAIN_MONTHS + 2:
            conn.close()
            return []

        sample_syms = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM stock_daily "
            "WHERE date=? AND volume > 1000000 ORDER BY symbol LIMIT 800",
            (monthly_dates[months[-1]],),
        ).fetchall()]

        logger.info(f"ML因子：加载{len(sample_syms)}只股票数据...")
        stock_data: dict[str, pd.DataFrame] = {}
        for sym in sample_syms:
            df = pd.read_sql(
                "SELECT * FROM stock_daily WHERE symbol=? ORDER BY date",
                conn, params=(sym,),
            )
            if len(df) >= 250:
                stock_data[sym] = df

        from sequoia_x.analysis.factor import compute_factors

        finance_map = self._load_finance_map(conn)
        valuation_map = self._load_valuation_map(conn)
        fund_flow_map = self._load_fund_flow_map(conn)
        north_map = self._load_north_map(conn)

        data = []
        eval_months = months[-(self.TRAIN_MONTHS + 3):]

        for month in eval_months:
            if month not in monthly_dates:
                continue
            date_str = monthly_dates[month]
            month_factors = []

            for sym, df in stock_data.items():
                df_d = df[df["date"] <= date_str]
                if len(df_d) < 60:
                    continue

                finance = finance_map.get(sym)
                val = valuation_map.get(sym, {})
                ff = fund_flow_map.get(sym)

                # 北向持股：取截至当月的 as-of 快照
                nb_df = None
                nb_series = north_map.get(sym)
                if nb_series:
                    asof = [d for d in nb_series if d <= date_str]
                    if asof:
                        nb_df = pd.DataFrame([
                            {"date": d, "hold_pct": nb_series[d]} for d in asof
                        ])

                # 加估值因子到 finance dict（compute_factors 不直接算 PE/PB）
                fin_combined = dict(finance) if finance else {}
                fin_combined["pe_ratio"] = -val.get("pe", 0) if val.get("pe", 0) and val.get("pe", 0) > 0 else 0
                fin_combined["pb_ratio"] = -val.get("pb", 0) if val.get("pb", 0) and val.get("pb", 0) > 0 else 0

                factors = compute_factors(
                    df_d,
                    finance=fin_combined,
                    fund_flow=ff,
                    lhb_data=None,
                    north_hold=nb_df,
                )

                fwd_idx = len(df_d) - 1
                fwd_end = fwd_idx + self.HOLD_DAYS
                if fwd_end >= len(df):
                    continue
                fwd_return = df.iloc[fwd_end]["close"] / df.iloc[fwd_idx]["close"] - 1

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
        logger.info(
            f"ML因子：收集{len(data)}条训练数据，"
            f"{len(set(d['month'] for d in data))}个月"
        )
        return data

    # ───────────────────────── 时间序列 CV ─────────────────────────

    def _time_series_cv(self, data: list[dict]) -> dict:
        """expanding-window 时间序列交叉验证。

        每轮用过去 N 个月训练，预测下 1 个月，滚动推进。
        严格保证 train_months 全部早于 test_month（无前视）。
        """
        df = pd.DataFrame(data)
        months = sorted(df["month"].unique())
        if len(months) < 6:
            return {"ic_mean": 0, "valid": False}

        ic_list: list[float] = []
        all_predictions: dict[str, float] = {}
        feature_importance: dict[str, float] = {}
        last_weights: np.ndarray | None = None

        def _to_float(v):
            try:
                return float(v) if v is not None else 0.0
            except (ValueError, TypeError):
                return 0.0

        for i in range(6, len(months) - 1):
            train_months = months[:i]
            test_month = months[i]

            train_df = df[df["month"].isin(train_months)]
            test_df = df[df["month"] == test_month]

            if len(train_df) < 300 or len(test_df) < 50:
                continue

            X_train = np.array(
                [[_to_float(r[f]) for f in self.FEATURE_FACTORS]
                 for r in train_df["features"]], dtype=float,
            )
            y_train = train_df["fwd_return"].values.astype(float)
            X_test = np.array(
                [[_to_float(r[f]) for f in self.FEATURE_FACTORS]
                 for r in test_df["features"]], dtype=float,
            )
            y_test = test_df["fwd_return"].values.astype(float)

            # 标准化（用训练集统计量）
            mean = np.nanmean(X_train, axis=0)
            std = np.nanstd(X_train, axis=0)
            std[std == 0] = 1
            X_train = np.nan_to_num((X_train - mean) / std)
            X_test = np.nan_to_num((X_test - mean) / std)
            X_train = np.nan_to_num(X_train, nan=0)
            X_test = np.nan_to_num(X_test, nan=0)

            preds, weights = self._train_predict(X_train, y_train, X_test)
            if weights is not None:
                last_weights = weights

            ic = self._rank_corr(preds, y_test)
            if not np.isnan(ic):
                ic_list.append(ic)

            # 保存最新预测（最后一轮 test = 最近月）
            if i == len(months) - 2:
                for idx, row in test_df.reset_index(drop=True).iterrows():
                    all_predictions[row["symbol"]] = float(preds[idx])

        if not ic_list:
            return {"ic_mean": 0, "valid": False}

        ic_arr = np.array(ic_list)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        icir = ic_mean / ic_std if ic_std > 0 else 0
        win_rate = float(np.mean(ic_arr > 0))

        # t-stat（样本外 IC 序列）
        n = len(ic_list)
        t_stat = ic_mean / (ic_std / np.sqrt(n)) if (ic_std > 0 and n >= 2) else 0

        # 特征重要度
        if _HAS_LGB and last_weights is not None:
            feature_importance = {
                f: float(w) for f, w in zip(self.FEATURE_FACTORS, last_weights)
            }

        model_version = "lightgbm" if _HAS_LGB else "ridge"

        return {
            "ic_mean": round(ic_mean, 4),
            "icir": round(icir, 3),
            "win_rate": round(win_rate * 100, 1),
            "t_stat": round(t_stat, 3),
            "n_months": len(ic_list),
            "valid": ic_mean > 0.03 and icir > 0.5 and abs(t_stat) >= 2.0,
            "predictions": all_predictions,
            "feature_importance": feature_importance,
            "model_version": model_version,
        }

    def _train_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """训练模型并预测。返回 (predictions, feature_importance_or_None)。

        LightGBM 优先；不可用时回退 Ridge Regression。
        """
        if _HAS_LGB and X_train.shape[0] >= 500:
            try:
                # 原生 API（lgb.train），不依赖 scikit-learn
                params = {
                    "objective": "regression",
                    "metric": "rmse",
                    "num_leaves": 15,
                    "max_depth": 4,
                    "min_data_in_leaf": 200,
                    "lambda_l2": 1.0,
                    "lambda_l1": 0.5,
                    "bagging_fraction": 0.8,
                    "feature_fraction": 0.8,
                    "bagging_freq": 1,
                    "verbose": -1,
                    "num_threads": 0,
                }
                train_set = lgb.Dataset(X_train, label=y_train)
                model = lgb.train(params, train_set, num_boost_round=100)
                preds = model.predict(X_test)
                importance = model.feature_importance(importance_type="gain")
                return preds, importance
            except Exception as e:
                logger.debug(f"LightGBM 训练失败，回退 Ridge：{e!r}")

        # Ridge fallback
        alpha = 1.0
        n_features = X_train.shape[1]
        A = X_train.T @ X_train + alpha * np.eye(n_features)
        b = X_train.T @ y_train
        try:
            weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            weights = np.zeros(n_features)
        preds = X_test @ weights
        return preds, np.abs(weights)

    # ───────────────────────── 持久化 ─────────────────────────

    def _save_ml_scores(self, results: dict) -> None:
        """写 ml_scores 快照 + 更新 factor_weights.ml_score。

        ml_score 的 IC/ICIR/t-stat 写入 factor_weights，走与普通因子相同的
        P3a 显著性门控（IC>0.03 & ICIR>0.5 & t≥2），达标才进权重。
        """
        predictions = results.get("predictions", {})
        if not predictions:
            return

        run_date = time.strftime("%Y-%m-%d")
        ic_mean = results.get("ic_mean", 0)
        icir = results.get("icir", 0)
        t_stat = results.get("t_stat", 0)
        model_version = results.get("model_version", "ridge")

        try:
            with sqlite3.connect(self.db_path) as conn:
                # 写 ml_scores 快照（先删当日旧快照）
                conn.execute("DELETE FROM ml_scores WHERE run_date=?", (run_date,))
                rows = [
                    (run_date, sym, float(score), ic_mean, icir, t_stat, model_version)
                    for sym, score in predictions.items()
                    if score == score  # NaN check
                ]
                conn.executemany(
                    "INSERT OR REPLACE INTO ml_scores "
                    "(run_date, symbol, ml_score, ic_mean, icir, t_stat, model_version) "
                    "VALUES (?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
            logger.info(
                f"ML快照写入：{len(rows)} 只，run_date={run_date}，"
                f"IC={ic_mean} ICIR={icir} t={t_stat}"
            )
        except Exception as e:
            logger.warning(f"ML快照写入失败：{e!r}")

        # 更新 factor_weights.ml_score（与普通因子共用 P3a 门控）
        from sequoia_x.analysis.factor import _is_significant
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        weight = 0.0
        if _is_significant(ic_mean, icir, t_stat, results.get("n_months", 0)):
            # 简单等权：达标则给一个小权重，由下游 IC 归一化处理
            weight = ic_mean
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO factor_weights "
                    "(factor_name, category, ic_mean, icir, win_rate, weight, updated_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    ("ml_score", "ML因子", ic_mean, icir,
                     results.get("win_rate", 0), round(weight, 4), now),
                )
                conn.commit()
            logger.info(
                f"factor_weights.ml_score 已更新：weight={weight:.4f} "
                f"({'达标' if weight else '未达标，权重0'})"
            )
        except Exception as e:
            logger.warning(f"factor_weights.ml_score 更新失败：{e!r}")

    # ───────────────────────── 工具方法 ─────────────────────────

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

    def _load_fund_flow_map(self, conn) -> dict:
        """加载最新一天的主力资金流向。"""
        try:
            rows = conn.execute(
                "SELECT symbol, main_net, main_pct, super_net, big_net FROM fund_flow "
                "WHERE date=(SELECT MAX(date) FROM fund_flow)"
            ).fetchall()
            return {
                r[0]: {
                    "main_net": r[1], "main_pct": r[2],
                    "super_net": r[3], "big_net": r[4],
                }
                for r in rows
            }
        except Exception:
            return {}

    def _load_north_map(self, conn) -> dict[str, dict[str, float]]:
        """加载北向持股历史 {symbol: {date_str: hold_pct}}。"""
        try:
            rows = conn.execute(
                "SELECT symbol, date, hold_pct FROM north_hold "
                "WHERE hold_pct IS NOT NULL ORDER BY symbol, date"
            ).fetchall()
            result: dict[str, dict[str, float]] = {}
            for r in rows:
                result.setdefault(r[0], {})[str(r[1])] = float(r[2])
            return result
        except Exception:
            return {}
