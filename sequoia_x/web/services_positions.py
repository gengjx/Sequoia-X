"""Web 服务 - 持仓管理（WebServices mixin 组件）。

由 :class:`sequoia_x.web.services.WebServices` 多重继承组合，不单独实例化；
方法通过 ``self.engine`` / ``self.settings`` / ``self._task_store`` 等访问门面状态。
"""

from __future__ import annotations

import logging
import sqlite3

from sequoia_x.analysis.position import PositionTracker
from sequoia_x.notify.feishu import FeishuNotifier

logger = logging.getLogger(__name__)


class PositionMixin:
    """持仓管理（WebServices 的 mixin 组件）。"""

    @property
    def positions(self) -> PositionTracker:
        """懒加载持仓跟踪器。"""
        if self._position_tracker is None:
            self._position_tracker = PositionTracker(self.engine, self.settings)
        return self._position_tracker

    def list_holdings(self, status: str = "open") -> list[dict]:
        return self.positions.list_holdings(status)

    def add_holding(self, data: dict) -> int:
        """从决策 buy_list 条目或手动录入新增持仓。"""
        return self.positions.add_holding(
            symbol=data["symbol"], name=data.get("name", ""),
            entry_price=float(data["entry_price"]), shares=int(data["shares"]),
            entry_date=data.get("entry_date"), stop_loss=float(data.get("stop_loss", 0)),
            target=float(data.get("target", 0)), grade=data.get("grade", ""),
            hit_strategies=data.get("hit_strategies", ""), notes=data.get("notes", ""),
        )

    def update_holding(self, hid: int, **fields) -> bool:
        return self.positions.update_holding(hid, **fields)

    def close_holding(self, hid: int, close_price: float, reason: str = "") -> bool:
        return self.positions.close_holding(hid, close_price, reason)

    def delete_holding(self, hid: int) -> bool:
        return self.positions.delete_holding(hid)

    def sync_paper_to_portfolio(self) -> None:
        """同步 paper_holdings → portfolio_holding（PositionTracker 读后者）。

        清空 portfolio_holding 的 open 记录，用 paper_holdings 当前持仓覆盖。
        """
        import sqlite3
        try:
            with sqlite3.connect(self.settings.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # 取 paper_holdings 当前持仓
                rows = conn.execute("SELECT * FROM paper_holdings").fetchall()
                # 清空 portfolio_holding 的 open 记录
                conn.execute("DELETE FROM portfolio_holding WHERE status='open'")
                # 写入
                for r in rows:
                    d = dict(r)
                    conn.execute(
                        "INSERT OR REPLACE INTO portfolio_holding "
                        "(symbol, name, entry_price, shares, entry_date, stop_loss, "
                        "initial_stop, target, grade, hit_strategies, cost, status) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?, 'open')",
                        (d["symbol"], d.get("name",""), d["entry_price"], d["shares"],
                         d.get("entry_date",""), d.get("stop_loss",0), d.get("initial_stop",0),
                         d.get("target",0), d.get("grade",""), d.get("hit_strategies",""),
                         d.get("cost",0))
                    )
                conn.commit()
                logging.getLogger(__name__).info(f"持仓同步: paper_holdings → portfolio_holding ({len(rows)}只)")
        except Exception as e:
            logging.getLogger(__name__).warning(f"持仓同步失败: {e!r}")

    def scan_positions(self, apply_stop_move: bool = False) -> dict:
        """扫描所有持仓，返回信号列表 + 组合摘要。"""
        signals = self.positions.scan_all(apply_stop_move=apply_stop_move)
        return {
            "signals": [self.positions.signal_to_dict(s) for s in signals],
            "summary": self.positions.summary(signals),
        }

    def scan_positions_intraday(self) -> dict:
        """盘中实时盯盘扫描：批量快照 + 实时MA估算。"""
        signals = self.positions.scan_intraday()
        return {
            "signals": [self.positions.signal_to_dict(s) for s in signals],
            "summary": self.positions.summary(signals),
        }

    def _get_auction_a_grade(self) -> list[str]:
        """取今日竞价A级票代码列表。"""
        try:
            with sqlite3.connect(self.engine.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol FROM auction_snap WHERE grade='A' "
                    "AND date=(SELECT MAX(date) FROM auction_snap)"
                ).fetchall()
            return [r[0] for r in rows if r[0]]
        except sqlite3.OperationalError:
            return []

    def import_decision_to_holdings(self, buy_list: list[dict]) -> dict:
        """把决策买入清单批量导入持仓表（跳过已持仓的）。"""
        added, skipped = 0, 0
        for r in buy_list:
            if r.get("shares", 0) <= 0 or r.get("price", 0) <= 0:
                skipped += 1
                continue
            try:
                self.positions.add_holding(
                    symbol=r["symbol"], name=r.get("name", ""),
                    entry_price=r["price"], shares=r["shares"],
                    stop_loss=r.get("stop_loss", 0), target=r.get("target", 0),
                    grade=r.get("grade", ""),
                    hit_strategies=",".join(r.get("hit_strategies", [])),
                )
                added += 1
            except Exception as e:
                logger.warning(f"导入持仓 {r.get('symbol')} 失败：{e!r}")
                skipped += 1
        logger.info(f"决策导入持仓：新增 {added} 只，跳过 {skipped} 只")
        return {"added": added, "skipped": skipped}

    def push_positions_feishu(self) -> dict:
        """推送持仓扫描报告到飞书。"""
        signals = self.positions.scan_all()
        summary = self.positions.summary(signals)
        notifier = FeishuNotifier(self.settings)
        ok = notifier.send_positions(
            [self.positions.signal_to_dict(s) for s in signals], summary,
        )
        return {"success": ok, "count": summary["count"]}

    def push_decision_feishu(self, decision: dict | None = None,
                             strategy_keys: list[str] | None = None,
                             capital: float = 100000.0, min_score: int = 50,
                             exclude_markets: list[str] | None = None,
                             exclude_st: bool = False) -> dict:
        """推送交易决策清单到飞书。无 decision 参数时自动生成。"""
        if decision is None:
            decision = self.generate_decision(
                strategy_keys=strategy_keys, capital=capital, min_score=min_score,
                exclude_markets=exclude_markets, exclude_st=exclude_st,
            )
        notifier = FeishuNotifier(self.settings)
        ok = notifier.send_decision(decision)
        return {"success": ok, "buy_count": decision.get("summary", {}).get("buy_count", 0)}
