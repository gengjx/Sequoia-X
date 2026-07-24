"""回测可信度验证引擎：Walk-forward + 参数敏感性 + Bootstrap + PSR。

解决核心问题：30.7%年化收益是否可信，还是过拟合产物？

4个验证维度：
  1. Walk-forward滚动验证：训练期→预测期，看样本外是否稳定
  2. 参数敏感性测试：hold_days参数扫描，看收益是否依赖特定参数
  3. Bootstrap置信区间：月度收益重采样，年化收益的95%置信区间
  4. PSR(Deflated Sharpe)：修正夏普比率的运气成分，判断统计显著性

对标：QuantConnect Out-of-Sample / 聚宽 walkforward / 米筐 bootstrap
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """单个验证维度的结果。"""
    name: str
    passed: bool           # 是否通过验证
    score: float           # 0-100 可信度分数
    detail: dict           # 详细数据


class BacktestValidator:
    """回测可信度验证器。

    依赖 StrategyEvaluator 的月度收益数据，不重复计算信号。
    """

    def __init__(self, evaluator):
        """Args: evaluator: StrategyEvaluator 实例（已初始化engine/settings）。"""
        self.ev = evaluator
        self.cost = evaluator.cost
        self.engine = evaluator.engine

    # ════════════════════════════════════════════════════════
    # 1. Walk-Forward 滚动验证
    # ════════════════════════════════════════════════════════
    def walk_forward(
        self, train_months: int = 6, test_months: int = 2,
        sample_size: int = 300, hold_days: int = 20,
    ) -> ValidationResult:
        """滚动窗口验证：训练期评估策略→预测期检验稳定性。

        逻辑：
          1. 把数据按月切片，每 train_months 训练 + test_months 预测
          2. 训练期的月度收益 vs 预测期，计算衰减率
          3. 衰减率<50% = 通过（策略在样本外仍有效）

        Args:
            train_months: 训练窗口长度
            test_months: 预测窗口长度
            sample_size: 采样股票数（减少加速）
            hold_days: 持有期
        """
        logger.info(f"Walk-forward验证：train={train_months}月 test={test_months}月")

        collected = self.ev._collect(hold_days, sample_size, seed=42)
        months = collected["months"]
        strat_monthly = collected["strategy_monthly"]
        bench_monthly = collected["benchmark_monthly"]

        if len(months) < train_months + test_months:
            return ValidationResult(
                "walk_forward", False, 0,
                {"error": f"数据不足：仅{len(months)}个月，需≥{train_months + test_months}"},
            )

        window = train_months + test_months
        results = []

        for start in range(0, len(months) - window + 1, test_months):
            train_ms = months[start:start + train_months]
            test_ms = months[start + train_months:start + window]
            if len(test_ms) < test_months:
                break

            for skey in strat_monthly:
                train_rets = []
                for m in train_ms:
                    train_rets.extend(strat_monthly[skey].get(m, []))
                test_rets = []
                for m in test_ms:
                    test_rets.extend(strat_monthly[skey].get(m, []))

                if len(train_rets) < 30 or len(test_rets) < 10:
                    continue

                train_mean = float(np.mean(train_rets))
                test_mean = float(np.mean(test_rets))
                decay = 1.0
                if train_mean > 0:
                    decay = test_mean / train_mean if train_mean != 0 else 1.0
                elif train_mean < 0:
                    decay = 1.0  # 训练期亏损无法判断衰减

                results.append({
                    "strategy": skey,
                    "period": f"{train_ms[0]}~{test_ms[-1]}",
                    "train_return": round(train_mean * 100, 2),
                    "test_return": round(test_mean * 100, 2),
                    "decay": round(decay, 2),
                })

        if not results:
            return ValidationResult("walk_forward", False, 0, {"error": "有效样本不足"})

        # 计算整体衰减率中位数
        decays = [r["decay"] for r in results]
        median_decay = float(np.median(decays))
        positive_test = sum(1 for r in results if r["test_return"] > 0)
        total_valid = len(results)

        # 按 strategy 分组取 median decay（供质量分折扣使用）
        from statistics import median as _median
        per_strategy_decay: dict[str, float] = {}
        strat_decay_groups: dict[str, list[float]] = {}
        for r in results:
            strat_decay_groups.setdefault(r["strategy"], []).append(r["decay"])
        for skey, group in strat_decay_groups.items():
            per_strategy_decay[skey] = round(float(_median(group)), 2)

        # 通过条件：中位衰减率>0.3 且 样本外正收益占比>50%
        passed = median_decay > 0.3 and positive_test / total_valid > 0.5
        score = min(100, max(0, median_decay * 100))

        return ValidationResult(
            "walk_forward", passed, round(score, 1),
            {
                "median_decay": round(median_decay, 2),
                "positive_test_ratio": round(positive_test / total_valid, 2),
                "windows": len(results),
                "per_strategy_decay": per_strategy_decay,
                "detail": results[:20],
            },
        )

    # ════════════════════════════════════════════════════════
    # 2. 参数敏感性测试
    # ════════════════════════════════════════════════════════
    def parameter_sensitivity(
        self, hold_days_list: list[int] | None = None,
        sample_size: int = 300,
    ) -> ValidationResult:
        """持有期参数扫描：不同hold_days下年化收益是否稳定。

        过拟合特征：只在某个特定参数下表现好，稍微偏移就崩。
        健康特征：在合理参数范围内（5-60天）都为正收益。

        Args:
            hold_days_list: 要测试的持有期列表
            sample_size: 采样股票数
        """
        hold_days_list = hold_days_list or [5, 10, 20, 30, 60]
        logger.info(f"参数敏感性测试：hold_days={hold_days_list}")

        param_results = []
        for hd in hold_days_list:
            result = self.ev.evaluate(hold_days=hd, sample_size=sample_size)
            bench_annual = result["benchmark"]["annual_return"]
            best_annual = 0
            for s in result["strategies"]:
                if s["annual_return"] > best_annual:
                    best_annual = s["annual_return"]
            mf_annual = next(
                (s["annual_return"] for s in result["strategies"]
                 if s["key"] == "multi_factor"), 0
            )
            param_results.append({
                "hold_days": hd,
                "multi_factor_annual": round(mf_annual, 1),
                "best_annual": round(best_annual, 1),
                "benchmark": round(bench_annual, 1),
                "alpha": round(mf_annual - bench_annual, 1),
            })

        # 多因子的alpha序列
        alphas = [p["alpha"] for p in param_results]
        positive_count = sum(1 for a in alphas if a > 0)
        alpha_std = float(np.std(alphas)) if len(alphas) > 1 else 0
        alpha_mean = float(np.mean(alphas))

        # 通过条件：>60%参数下alpha为正 且 标准差<15%
        passed = positive_count / len(alphas) > 0.6 and alpha_std < 15
        # 分数：正alpha占比×50 + 稳定性×50
        stability_score = max(0, 100 - alpha_std * 5)
        score = (positive_count / len(alphas) * 50) + (stability_score * 0.5)

        return ValidationResult(
            "parameter_sensitivity", passed, round(score, 1),
            {
                "params": param_results,
                "positive_ratio": round(positive_count / len(alphas), 2),
                "alpha_mean": round(alpha_mean, 1),
                "alpha_std": round(alpha_std, 1),
            },
        )

    # ════════════════════════════════════════════════════════
    # 3. Bootstrap 置信区间
    # ════════════════════════════════════════════════════════
    def bootstrap(
        self, sample_size: int = 500, hold_days: int = 20,
        n_resamples: int = 1000,
    ) -> ValidationResult:
        """Bootstrap重采样：月度收益重采样，年化收益的置信区间。

        原理：
          1. 提取策略月度收益序列
          2. 有放回重采样n次，每次计算年化收益
          3. 95%置信区间下界>0 = 策略有效

        Args:
            n_resamples: 重采样次数
        """
        logger.info(f"Bootstrap验证：重采样{n_resamples}次")

        collected = self.ev._collect(hold_days, sample_size, seed=42)
        months = collected["months"]
        strat_monthly = collected["strategy_monthly"]
        bench_monthly = collected["benchmark_monthly"]

        rng = np.random.RandomState(42)
        results = {}

        for skey in strat_monthly:
            monthly_returns = []
            for m in months:
                rets = strat_monthly[skey].get(m, [])
                monthly_returns.append(float(np.mean(rets)) if rets else 0.0)

            if len(monthly_returns) < 6:
                continue

            arr = np.array(monthly_returns)
            annuals = []
            for _ in range(n_resamples):
                sample = rng.choice(arr, size=len(arr), replace=True)
                curve = np.cumprod(1 + sample)
                annual = (curve[-1] ** (12 / len(curve)) - 1) * 100
                annuals.append(annual)

            annuals = np.array(annuals)
            p5 = float(np.percentile(annuals, 5))
            p50 = float(np.percentile(annuals, 50))
            p95 = float(np.percentile(annuals, 95))

            results[skey] = {
                "ci_lower": round(p5, 1),
                "median": round(p50, 1),
                "ci_upper": round(p95, 1),
                "prob_positive": round(float(np.mean(annuals > 0)), 2),
            }

        # 多因子重点检查
        mf = results.get("multi_factor", {})
        ci_lower = mf.get("ci_lower", -100)
        prob_pos = mf.get("prob_positive", 0)

        passed = ci_lower > 0 and prob_pos > 0.9
        score = min(100, max(0, prob_pos * 100))

        return ValidationResult(
            "bootstrap", passed, round(score, 1),
            {
                "multi_factor": mf,
                "all_strategies": {k: v for k, v in sorted(
                    results.items(), key=lambda x: x[1]["median"], reverse=True
                )[:8]},
                "n_resamples": n_resamples,
            },
        )

    # ════════════════════════════════════════════════════════
    # 4. PSR (Probabilistic Sharpe Ratio) / Deflated Sharpe
    # ════════════════════════════════════════════════════════
    def psr_test(
        self, sample_size: int = 500, hold_days: int = 20,
        benchmark_sharpe: float = 0.5,
    ) -> ValidationResult:
        """PSR检验：夏普比率中"运气"成分有多大。

        原理（Bailey & Lopez de Prado 2012）：
          PSR = Φ( ((SR - SR₀) * sqrt(n-1)) / sqrt(1 - skew*SR + (kurt-1)/4 * SR²) )
          其中 SR=样本夏普，SR₀=基准夏普，n=样本数，skew/kurt=偏度/峰度

          PSR > 0.95 = 夏普统计显著（95%置信度下真实夏普>基准）
          PSR < 0.50 = 夏普不可靠（可能是运气）

        Args:
            benchmark_sharpe: 基准夏普（默认0.5，年化夏普>0.5才值得实盘）
        """
        logger.info(f"PSR验证：基准夏普={benchmark_sharpe}")

        collected = self.ev._collect(hold_days, sample_size, seed=42)
        months = collected["months"]
        strat_monthly = collected["strategy_monthly"]

        # 纯numpy实现标准正态CDF（不依赖scipy）
        import math
        def _norm_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        results = {}
        n = len(months)

        for skey in strat_monthly:
            monthly_returns = []
            for m in months:
                rets = strat_monthly[skey].get(m, [])
                monthly_returns.append(float(np.mean(rets)) if rets else 0.0)

            if len(monthly_returns) < 6:
                continue

            arr = np.array(monthly_returns)
            std = float(arr.std())
            sr_annual = float(arr.mean() / std * np.sqrt(12)) if std > 0 else 0
            sr_monthly = sr_annual / np.sqrt(12)

            # 偏度和峰度（纯numpy）
            mean = float(arr.mean())
            if std > 0:
                skew = float(np.mean(((arr - mean) / std) ** 3))
                kurt = float(np.mean(((arr - mean) / std) ** 4))
            else:
                skew = 0.0
                kurt = 3.0

            # PSR公式（Bailey & Lopez de Prado 2012）
            sr0_monthly = benchmark_sharpe / np.sqrt(12)
            numerator = (sr_monthly - sr0_monthly) * np.sqrt(n - 1)
            denominator = np.sqrt(1 - skew * sr_monthly + (kurt - 1) / 4 * sr_monthly ** 2)

            if denominator > 0:
                psr = float(_norm_cdf(numerator / denominator))
            else:
                psr = 0.5

            results[skey] = {
                "sharpe": round(sr_annual, 2),
                "psr": round(psr, 3),
                "skew": round(skew, 2),
                "kurtosis": round(kurt, 2),
                "significant": psr > 0.95,
                "n_months": n,
            }

        # 多因子重点
        mf = results.get("multi_factor", {})
        psr_val = mf.get("psr", 0)

        passed = psr_val > 0.95
        score = min(100, max(0, psr_val * 100))

        return ValidationResult(
            "psr", passed, round(score, 1),
            {
                "multi_factor": mf,
                "all_strategies": {k: v for k, v in sorted(
                    results.items(), key=lambda x: x[1]["psr"], reverse=True
                )[:8]},
                "benchmark_sharpe": benchmark_sharpe,
            },
        )

    # ════════════════════════════════════════════════════════
    # 综合验证
    # ════════════════════════════════════════════════════════
    def validate_all(
        self, sample_size: int = 300, hold_days: int = 20,
    ) -> dict:
        """运行全部4项验证，返回综合可信度报告。

        Returns:
            {
                "overall_score": 0-100,
                "overall_passed": bool,
                "tests": {name: ValidationResult.detail, ...},
                "summary": "文字总结",
            }
        """
        tests = {}
        scores = []

        for name, fn in [
            ("walk_forward", lambda: self.walk_forward(sample_size=sample_size, hold_days=hold_days)),
            ("parameter_sensitivity", lambda: self.parameter_sensitivity(sample_size=sample_size)),
            ("bootstrap", lambda: self.bootstrap(sample_size=sample_size, hold_days=hold_days)),
            ("psr", lambda: self.psr_test(sample_size=sample_size, hold_days=hold_days)),
        ]:
            logger.info(f"执行验证：{name}")
            try:
                r = fn()
                tests[name] = {
                    "passed": r.passed,
                    "score": r.score,
                    "detail": r.detail,
                }
                scores.append(r.score)
            except Exception as e:
                logger.warning(f"验证{name}失败：{e!r}")
                tests[name] = {"passed": False, "score": 0, "detail": {"error": str(e)}}
                scores.append(0)

        overall = float(np.mean(scores)) if scores else 0
        passed_count = sum(1 for t in tests.values() if t["passed"])

        # 总结文字
        if overall >= 75:
            summary = f"✅ 回测可信度高（{overall:.0f}分），{passed_count}/4项验证通过"
        elif overall >= 50:
            summary = f"⚠️ 回测可信度中等（{overall:.0f}分），{passed_count}/4项验证通过，建议关注未通过项"
        else:
            summary = f"❌ 回测可信度低（{overall:.0f}分），仅{passed_count}/4项验证通过，策略可能过拟合"

        return {
            "overall_score": round(overall, 1),
            "overall_passed": overall >= 60,
            "passed_count": passed_count,
            "total_count": len(tests),
            "tests": tests,
            "summary": summary,
        }
