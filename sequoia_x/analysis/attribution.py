"""收益归因分析：分解多因子策略的超额收益来源。

核心方法（对标BARRA/Brinson归因模型）：
  1. 因子贡献分解：每个因子对超额收益的边际贡献 = 权重 × IC × 因子离散度
  2. 因子暴露分析：多因子选出的Top组合 vs 全市场在各个因子上的暴露差异
  3. 净收益分解：alpha贡献排序，识别哪些因子是收益驱动力

输出：各因子的收益贡献度排名 + 有效/无效判定 + 优化建议
"""

from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

from sequoia_x.analysis.factor import compute_factors, FACTOR_META
from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

HOLD_DAYS = 20
SAMPLE_SIZE = 500


class AttributionAnalyzer:
    """收益归因分析器。"""

    def __init__(self, engine, settings: Settings | None = None) -> None:
        self.engine = engine
        self.settings = settings

    def analyze(self, hold_days: int = HOLD_DAYS, sample_size: int = SAMPLE_SIZE,
                seed: int = 42) -> dict:
        """执行收益归因分析。

        流程：
          1. 采样N只股票，计算每月截面因子值 + 前向收益
          2. 对多因子综合分排序，取Top50为"策略组合"
          3. 计算策略组合 vs 全市场的各因子暴露差异
          4. 用因子权重×IC×暴露差 = 因子贡献度
          5. 按贡献度排名输出

        Returns:
            {
                "factor_contributions": [{name, category, weight, ic, exposure_gap, contribution, ...}],
                "strategy_return": 年化收益,
                "benchmark_return": 基准年化,
                "alpha": 超额年化,
                "top_factors": 贡献最大的因子,
                "drag_factors": 拖累收益的因子,
                "months": 回测月数,
                "sample_size": 采样数,
            }
        """
        symbols = self.engine.get_local_symbols()
        if sample_size and len(symbols) > sample_size:
            rng = random.Random(seed)
            symbols = rng.sample(symbols, sample_size)

        logger.info(f"收益归因：采样{len(symbols)}只，持有期{hold_days}天")

        # 加载三态因子权重
        state_weights = self._load_state_weights()
        # 加载财报数据
        finance_map = self._load_finance()
        valuation_map = self._load_valuation()

        cutoff_map = self.engine.get_ipo_cutoff_map()

        # 收集月度截面数据
        # {month: [{symbol, factors: {}, fwd_return: float}]}
        monthly_data: dict[str, list[dict]] = {}
        processed = 0

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < hold_days + 60:
                    continue
                co = cutoff_map.get(symbol)
                if co and "date" in df.columns:
                    df = df[df["date"].astype(str) >= co]
                df = df.reset_index(drop=True)
                if len(df) < hold_days + 20:
                    continue

                close = df["close"]
                dates = df["date"].astype(str).values
                # 前向收益
                fwd = (close.shift(-(hold_days + 1)) / close.shift(-1) - 1).values

                fin = finance_map.get(symbol, {})
                val = valuation_map.get(symbol)

                # 每月初取截面（避免逐日重复）
                prev_month = ""
                for i in range(60, len(df) - hold_days - 1):
                    if np.isnan(fwd[i]):
                        continue
                    m = str(dates[i])[:7]
                    if m == prev_month:
                        continue
                    prev_month = m

                    # 计算截面因子值
                    slice_df = df.iloc[: i + 1]
                    factors = compute_factors(slice_df, finance=fin)
                    if val:
                        factors.update(val)
                    if not factors:
                        continue

                    monthly_data.setdefault(m, []).append({
                        "symbol": symbol,
                        "factors": factors,
                        "fwd_return": float(fwd[i]),
                    })

                processed += 1
            except Exception:
                continue

        months = sorted(monthly_data.keys())
        logger.info(f"收益归因：处理{processed}只，{len(months)}个月")

        if len(months) < 6:
            return self._empty_result(sample_size)

        # 计算每月因子IC + 多因子综合分排序
        all_factor_names = list(FACTOR_META.keys())
        # 添加估值因子
        for extra in ["pe_ratio", "pb_ratio"]:
            if extra not in all_factor_names:
                all_factor_names.append(extra)

        # 因子贡献累加
        factor_ic_sum: dict[str, float] = {f: 0.0 for f in all_factor_names}
        factor_exposure_gap_sum: dict[str, float] = {f: 0.0 for f in all_factor_names}
        factor_weight_avg: dict[str, float] = {f: 0.0 for f in all_factor_names}
        month_count = 0

        strategy_rets = []
        benchmark_rets = []

        for m in months:
            batch = monthly_data[m]
            if len(batch) < 30:
                continue
            month_count += 1

            # 提取因子矩阵
            for fname in all_factor_names:
                def _to_float(v):
                    if v is None:
                        return np.nan
                    try:
                        f = float(v)
                        return f if f == f else np.nan
                    except (TypeError, ValueError):
                        return np.nan
                vals = np.array([_to_float(b["factors"].get(fname)) for b in batch], dtype=float)
                rets = np.array([_to_float(b.get("fwd_return")) for b in batch], dtype=float)
                mask = ~(np.isnan(vals)) & ~(np.isnan(rets))
                if mask.sum() < 20:
                    continue
                v = vals[mask]
                r = rets[mask]

                # Rank IC (Spearman via numpy rank)
                v_rank = pd.Series(v).rank().values
                r_rank = pd.Series(r).rank().values
                ic_val = np.corrcoef(v_rank, r_rank)[0, 1] if len(v) >= 20 else 0.0
                if np.isnan(ic_val):
                    ic_val = 0.0
                factor_ic_sum[fname] += ic_val

                # 因子权重（三态加权）
                month_state = self._classify_month(m, months)
                weights = state_weights.get(month_state, state_weights.get("neutral", {}))
                fw = weights.get(fname, 0)  # signed weight (sign matches IC)
                factor_weight_avg[fname] += fw

                # 暴露差异：Top50组合因子均值 - 全市场因子均值
                sorted_idx = np.argsort(v)
                top_n = min(50, len(v) // 3)
                top_factor_mean = np.mean(v[sorted_idx[-top_n:]])
                all_factor_mean = np.mean(v)
                exposure_gap = top_factor_mean - all_factor_mean
                factor_exposure_gap_sum[fname] += exposure_gap

            # 多因子综合分 → Top50前向收益 vs 全市场
            rets_all = np.array([_to_float(b.get("fwd_return")) for b in batch], dtype=float)
            month_state = self._classify_month(m, months)
            weights = state_weights.get(month_state, state_weights.get("neutral", {}))

            scores = np.zeros(len(batch))
            for b_idx, b in enumerate(batch):
                s = 0.0
                for fname, fw in weights.items():
                    fv = _to_float(b["factors"].get(fname))
                    if fv == fv and fw != 0:  # NaN check without np.isnan
                        s += fv * fw
                scores[b_idx] = s

            if np.std(scores) > 0:
                top_mask = scores >= np.percentile(scores, 80)
                strategy_rets.append(np.mean(rets_all[top_mask]))
                benchmark_rets.append(np.mean(rets_all))
            else:
                benchmark_rets.append(np.mean(rets_all))

        if not strategy_rets:
            return self._empty_result(sample_size)

        # 计算归因结果
        contributions = []
        for fname in all_factor_names:
            avg_ic = float(factor_ic_sum[fname] / month_count) if month_count else 0.0
            avg_weight = float(factor_weight_avg[fname] / month_count) if month_count else 0.0
            avg_exp_gap = float(factor_exposure_gap_sum[fname] / month_count) if month_count else 0.0

            # 贡献度 = 权重 × IC × 100（符号乘法）
            # 同号(IC>0,w>0 或 IC<0,w<0) → 正贡献（因子帮助选股）
            # 异号 → 负贡献（因子拖累选股，需修正）
            same_sign = bool((avg_ic > 0) == (avg_weight > 0)) if avg_weight != 0 else False
            contribution = float(avg_weight * avg_ic * 100)

            meta = FACTOR_META.get(fname, {})
            contributions.append({
                "name": fname,
                "category": meta.get("category", ""),
                "desc": meta.get("desc", fname),
                "ic_mean": round(avg_ic, 4),
                "weight": round(avg_weight, 4),
                "exposure_gap": round(avg_exp_gap, 4),
                "contribution": round(contribution, 4),
                "effective": bool(abs(avg_ic) > 0.02 and abs(avg_weight) > 0.005 and same_sign),
                "assessment": self._assess(avg_ic, avg_weight),
            })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        # 策略 vs 基准年化
        strat_curve = np.cumprod(1 + np.array(strategy_rets))
        bench_curve = np.cumprod(1 + np.array(benchmark_rets))
        n_months = len(strategy_rets)
        strat_annual = float((strat_curve[-1] ** (12 / n_months) - 1) * 100) if n_months > 0 else 0.0
        bench_annual = float((bench_curve[-1] ** (12 / n_months) - 1) * 100) if n_months > 0 else 0.0

        # 分类别汇总
        category_summary = {}
        for c in contributions:
            cat = c["category"] or "其他"
            if cat not in category_summary:
                category_summary[cat] = {"total_contribution": 0, "count": 0, "effective": 0}
            category_summary[cat]["total_contribution"] += c["contribution"]
            category_summary[cat]["count"] += 1
            if c["effective"]:
                category_summary[cat]["effective"] += 1

        cat_list = sorted(
            [{"category": k, **v} for k, v in category_summary.items()],
            key=lambda x: abs(x["total_contribution"]),
            reverse=True,
        )

        # ── 闭环1：归因结果写回因子权重表 ──
        weight_adjustments = self._apply_weight_feedback(contributions)

        return {
            "factor_contributions": contributions[:15],
            "all_factors_count": len(contributions),
            "category_summary": cat_list,
            "strategy_annual_return": round(strat_annual, 2),
            "benchmark_annual_return": round(bench_annual, 2),
            "alpha": round(strat_annual - bench_annual, 2),
            "months": n_months,
            "sample_size": len(symbols),
            "processed": processed,
            "hold_days": hold_days,
            "top_drivers": [c["name"] for c in contributions[:8] if c["contribution"] > 0][:5],
            "top_drags": [c["name"] for c in contributions if c["contribution"] < 0][:5],
            "weight_adjustments": weight_adjustments,
        }

    def _apply_weight_feedback(self, contributions: list[dict]) -> list[dict]:
        """闭环1：归因结果反馈到因子权重表。

        规则：
          - 贡献度 < 0 的因子（拖累收益）→ 权重 ×0.7
          - 贡献度 Top5 且 > 0 的因子 → 权重 ×1.2
          - 贡献度接近0（|contrib| < 0.005）→ 权重 ×0.5（噪声因子降权）
          - 每个因子单次调整幅度上限 ±30%，防止过度调整
        """
        import sqlite3 as _sq
        import time as _time

        # 构建贡献度查找表
        contrib_map = {c["name"]: c["contribution"] for c in contributions}
        top_drivers = set(c["name"] for c in contributions[:5] if c["contribution"] > 0)

        adjustments = []
        now = _time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            with _sq.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT market_state, factor_name, weight FROM market_factor_weights WHERE weight != 0"
                ).fetchall()

                updates = []
                for state, fname, old_wt in rows:
                    contrib = contrib_map.get(fname, 0)
                    multiplier = 1.0
                    reason = ""

                    if contrib < -0.01:
                        multiplier = 0.85
                        reason = f"贡献度{contrib:+.4f}<0，降权15%"
                    elif fname in top_drivers and contrib > 0.01:
                        multiplier = 1.10
                        reason = f"核心驱动(贡献{contrib:+.4f})，提权10%"
                    elif abs(contrib) < 0.005:
                        multiplier = 0.75
                        reason = f"贡献度接近0({contrib:+.4f})，噪声降权25%"

                    if multiplier != 1.0:
                        new_wt = old_wt * multiplier
                        # 限制单次调整幅度
                        if abs(new_wt) > abs(old_wt) * 1.05:
                            new_wt = old_wt * 1.05
                        elif abs(new_wt) < abs(old_wt) * 0.95:
                            new_wt = old_wt * 0.95

                        updates.append((state, fname, new_wt, now))
                        adjustments.append({
                            "factor": fname, "state": state,
                            "old_weight": round(old_wt, 4),
                            "new_weight": round(new_wt, 4),
                            "multiplier": multiplier,
                            "contribution": round(contrib, 4),
                            "reason": reason,
                        })

                # 批量更新
                if updates:
                    conn.executemany(
                        "UPDATE market_factor_weights SET weight=?, updated_at=? "
                        "WHERE market_state=? AND factor_name=?",
                        [(u[2], u[3], u[0], u[1]) for u in updates]
                    )
                    logger.info(f"归因反馈：调整{len(updates)}个因子权重")
        except Exception as e:
            logger.warning(f"归因权重反馈失败：{e!r}")

        return adjustments

    def _load_state_weights(self) -> dict[str, dict[str, float]]:
        """从DB加载三态市场状态因子权重。"""
        try:
            result = self.engine.load_market_factor_weights()
            if result:
                return result
        except Exception:
            pass
        return {}

    def _load_finance(self) -> dict[str, dict]:
        """加载财报因子。"""
        import sqlite3 as _sq
        result: dict[str, dict] = {}
        try:
            with _sq.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, roe, np_margin, gp_margin, rev_growth, profit_growth "
                    "FROM stock_finance ORDER BY stat_date DESC"
                ).fetchall()
                seen = set()
                for r in rows:
                    if r[0] in seen:
                        continue
                    seen.add(r[0])
                    result[r[0]] = {
                        "roe": r[1], "np_margin": r[2], "gp_margin": r[3],
                        "rev_growth": r[4], "profit_growth": r[5],
                    }
        except Exception:
            pass
        return result

    def _load_valuation(self) -> dict[str, dict[str, float]]:
        """加载估值快照（PE/PB）。"""
        import sqlite3 as _sq
        result: dict[str, dict[str, float]] = {}
        try:
            with _sq.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, pe_ratio, pb_ratio FROM stock_finance "
                    "WHERE pe_ratio IS NOT NULL ORDER BY stat_date DESC"
                ).fetchall()
                seen = set()
                for r in rows:
                    if r[0] in seen:
                        continue
                    seen.add(r[0])
                    val = {}
                    if r[1] and r[1] > 0:
                        # PE取倒数（低PE=高分）
                        val["pe_ratio"] = -1.0 / r[1]
                    if r[2] and r[2] > 0:
                        val["pb_ratio"] = -1.0 / r[2]
                    if val:
                        result[r[0]] = val
        except Exception:
            pass
        return result

    @staticmethod
    def _classify_month(month: str, all_months: list[str]) -> str:
        """根据月份在全周期中的位置粗略分类市场状态。"""
        if not all_months:
            return "neutral"
        idx = all_months.index(month) if month in all_months else 0
        total = len(all_months)
        if total < 6:
            return "neutral"
        # 前1/3为早期（2021-2022熊市），中1/3震荡，后1/3牛市
        if idx < total * 0.35:
            return "bear"
        elif idx < total * 0.65:
            return "neutral"
        else:
            return "bull"

    @staticmethod
    def _assess(ic: float, weight: float) -> str:
        if abs(ic) < 0.01:
            return "无效"
        if abs(weight) < 0.003:
            return "权重过低"
        same_sign = (ic > 0) == (weight > 0)
        if not same_sign and abs(ic) > 0.02:
            return "⚠️ 方向错误"
        if abs(ic) > 0.03 and abs(weight) > 0.01:
            return "✅ 核心驱动"
        if abs(ic) > 0.02:
            return "✅ 有效"
        return "中性"

    @staticmethod
    def _empty_result(sample_size: int) -> dict:
        return {
            "factor_contributions": [],
            "all_factors_count": 0,
            "category_summary": [],
            "strategy_annual_return": 0,
            "benchmark_annual_return": 0,
            "alpha": 0,
            "months": 0,
            "sample_size": sample_size,
            "processed": 0,
            "hold_days": HOLD_DAYS,
            "top_drivers": [],
            "top_drags": [],
            "error": "样本不足（需至少6个月数据）",
        }
