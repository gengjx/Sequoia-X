"""Web 服务 - 回测与因子评估（WebServices mixin 组件）。

由 :class:`sequoia_x.web.services.WebServices` 多重继承组合，不单独实例化；
方法通过 ``self.engine`` / ``self.settings`` / ``self._task_store`` 等访问门面状态。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from sequoia_x.analysis.backtest import SignalBacktester
from sequoia_x.analysis.combo_backtest import ComboBacktester
from sequoia_x.analysis.factor import evaluate_factor_ic
from sequoia_x.analysis.strategy_eval import StrategyEvaluator
from sequoia_x.web.services_common import (
    TaskRecord,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class BacktestMixin:
    """回测与因子评估（WebServices 的 mixin 组件）。"""

    def backtest_combos(self, combos: dict[str, list[str]] | None = None,
                         hold_days: list[int] | None = None) -> dict:
        """组合历史回测对比（向量化，采样500只，秒级返回）。"""
        if combos is None:
            combos = {
                "趋势突破": ["turtle", "ma_volume", "rps"],
                "低吸埋伏": ["pullback", "bottom"],
                "均衡全天候": ["rps", "turtle", "pullback", "bottom"],
            }
        hold_days = hold_days or [5, 10, 20]
        bt = ComboBacktester(self.engine, self.settings)
        return bt.run(combos, hold_days=hold_days)

    def backtest_resonance(self, hold_days: list[int] | None = None) -> dict:
        """共振度分档回测（验证多策略共振是否带来超额收益）。"""
        hold_days = hold_days or [5, 10, 20]
        bt = ComboBacktester(self.engine, self.settings)
        return bt.run_resonance(hold_days=hold_days)

    def evaluate_ml_factor(self) -> dict:
        """评估ML因子（Ridge因子合成）。"""
        from sequoia_x.analysis.ml_factor import MLFactorEngine
        engine = MLFactorEngine(self.engine.db_path)
        result = engine.compute_ml_score()
        # 如果有效，写入全局因子权重表
        if result.get("valid"):
            ic = result["ic_mean"]
            icir = result["icir"]
            wr = result["win_rate"]
            weight = ic / max(abs(icir), 0.01) * icir  # 符号IC权重
            self.engine.save_factor_weights([{
                "factor_name": "ml_score",
                "category": "ML因子",
                "ic_mean": ic,
                "icir": icir,
                "win_rate": wr,
                "weight": weight,
            }])
            # 同时写入三态权重表（所有市场状态使用相同权重，因为ML因子是自适应的）
            import sqlite3
            ml_entry = [{
                "factor_name": "ml_score",
                "category": "ML因子",
                "ic_mean": ic,
                "icir": icir,
                "win_rate": wr,
                "weight": abs(ic),
            }]
            with sqlite3.connect(self.engine.db_path) as conn:
                for state in ("bull", "neutral", "bear"):
                    from datetime import datetime as _dt
                    now_str = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
                    conn.execute(
                        "INSERT OR REPLACE INTO market_factor_weights "
                        "(market_state, factor_name, category, ic_mean, icir, win_rate, weight, updated_at) "
                        "VALUES (?, 'ml_score', 'ML因子', ?, ?, ?, ?, ?)",
                        (state, ic, icir, wr, abs(ic), now_str),
                    )
                conn.commit()
            logging.getLogger(__name__).info(
                f"ML因子已写入全局+三态权重表（IC={ic} ICIR={icir}）"
            )
        return result

    def evaluate_strategies(self, hold_days: int = 20, sample_size: int = 500) -> dict:
        """策略评估：时间序列净值 + 全维度评分卡 + 基准对比。"""
        ev = StrategyEvaluator(self.engine, self.settings)
        return ev.evaluate(hold_days=hold_days, sample_size=sample_size)

    def get_strategy_weights(self) -> dict:
        """读取DB中的策略权重快照（前端展示当前权重+更新时间）。"""
        return self.engine.load_strategy_weights()

    def find_optimal_combos(self, hold_days: int = 20, sample_size: int = 500,
                            max_strategies: int = 5, top_n: int = 10) -> dict:
        """网格搜索最优策略组合（数据驱动，替代主观预设）。"""
        ev = StrategyEvaluator(self.engine, self.settings)
        return ev.find_optimal_combos(
            hold_days=hold_days, sample_size=sample_size,
            max_strategies=max_strategies, top_n=top_n,
        )

    def compare_combos(self, hold_days: int = 20, sample_size: int = 500) -> dict:
        """主观预设组合 vs 数据驱动最优组合 对比。"""
        ev = StrategyEvaluator(self.engine, self.settings)
        return ev.compare_combos(hold_days=hold_days, sample_size=sample_size)

    def evaluate_factors(self, hold_days: int = 20, sample_size: int = 500,
                         rolling_months: int = 0) -> dict:
        """因子IC评估：全部因子的预测力评估（Rank IC/ICIR/分层）。

        Args:
            rolling_months: 滚动窗口月数（0=全样本，6=最近6个月）
        """
        return evaluate_factor_ic(
            self.engine, hold_days=hold_days, sample_size=sample_size,
            rolling_months=rolling_months,
        )

    def get_factor_weights(self) -> dict:
        """读取DB中的因子权重快照。"""
        return self.engine.load_factor_weights()

    # ------------------------------------------------------------------
    # 持仓跟踪 PositionTracker
    # ------------------------------------------------------------------

    def backtest_async(self) -> str:
        """异步执行信号评分 IC 回测，结果缓存到内存。"""
        task_id = uuid.uuid4().hex[:8]
        record = TaskRecord(task_id=task_id, strategy_key="__backtest__")
        self._task_store[task_id] = record
        self._executor.submit(self._backtest_task, task_id)
        return task_id

    def _backtest_task(self, task_id: str) -> None:
        record = self._task_store[task_id]
        record.status = TaskStatus.RUNNING
        record.started_at = datetime.now()
        try:
            bt = SignalBacktester(self.settings)
            report = bt.run(min_days=120)
            self._backtest_cache = report
            record.results = [f"sample_days:{report.get('sample_days', 0)}"]
            record.status = TaskStatus.DONE
        except Exception as e:
            record.status = TaskStatus.ERROR
            record.error = str(e)
        finally:
            record.finished_at = datetime.now()

    def get_backtest_report(self) -> dict | None:
        """返回缓存的回测报告。"""
        return self._backtest_cache
