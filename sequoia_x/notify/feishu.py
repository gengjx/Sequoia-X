"""飞书通知模块：将选股结果通过 Webhook 推送至飞书群。"""

import json
from datetime import date

import requests

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class FeishuNotifier:
    """飞书 Webhook 推送器。

    根据策略的 webhook_key 路由到对应的飞书机器人。
    若 webhook_key 未在 Settings.strategy_webhooks 中配置，
    则 fallback 到 Settings.feishu_webhook_url。
    """

    def __init__(self, settings: Settings) -> None:
        """
        初始化 FeishuNotifier。

        Args:
            settings: Settings 实例，提供 Webhook URL 配置。
        """
        self.settings = settings

    @staticmethod
    def _to_xueqiu_code(code: str) -> str:
        """将纯数字代码转为雪球格式：6开头→SH，4/8开头→BJ，其余→SZ。"""
        if code.startswith("6"):
            return f"SH{code}"
        elif code.startswith(("4", "8")):
            return f"BJ{code}"
        return f"SZ{code}"

    @staticmethod
    def _get_stock_names(symbols: list[str]) -> dict[str, str]:
        """通过 baostock 批量查询股票名称，返回 {code: name} 映射。"""
        import baostock as bs
        bs.login()
        mapping = {}
        for code in symbols:
            prefix = "sh" if code.startswith(("6", "9")) else "sz"
            rs = bs.query_stock_basic(code=f"{prefix}.{code}")
            while rs.next():
                row = rs.get_row_data()
                mapping[code] = row[1]  # 第2个字段是股票名称
        bs.logout()
        return mapping

    def _build_card(self, symbols: list[str], strategy_name: str) -> dict:
        today = date.today().strftime("%Y-%m-%d")
        names = self._get_stock_names(symbols)

        links: list[str] = []
        for code in symbols:
            xq_code = self._to_xueqiu_code(code)
            name = names.get(code, xq_code)
            links.append(f"[{name}](https://xueqiu.com/S/{xq_code})")

        symbol_text = " ".join(links) if links else "（无选股结果）"

        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"📈 Sequoia-X 选股播报 | {strategy_name}",
                    },
                    "template": "blue",
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**日期：** {today}\n**策略：** {strategy_name}\n**选股数量：** {len(symbols)}",
                        },
                    },
                    {"tag": "hr"},
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**选股列表：**\n{symbol_text}",
                        },
                    },
                ],
            },
        }

    def send(
        self,
        symbols: list[str],
        strategy_name: str,
        webhook_key: str = "default",
    ) -> None:
        """
        将选股结果格式化为飞书卡片消息并 POST 至对应 Webhook。

        根据 webhook_key 从 Settings 中查找专属 URL；
        若未配置，则 fallback 到 feishu_webhook_url。

        Args:
            symbols: 选股结果代码列表。
            strategy_name: 策略名称，用于卡片标题。
            webhook_key: 策略标识，用于路由到对应飞书机器人。

        Raises:
            不抛出异常，HTTP 失败时记录 ERROR 日志。
        """
        url = self.settings.get_webhook_url(webhook_key)
        payload = self._build_card(symbols, strategy_name)

        try:
            resp = requests.post(
                url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            # 解析飞书真正的返回体
            resp_json = resp.json()

            # 飞书真正的成功标志是内部的 code == 0
            if resp.status_code != 200 or resp_json.get("code") != 0:
                logger.error(
                    f"飞书推送失败 [{webhook_key}] "
                    f"HTTP状态={resp.status_code} 飞书响应={resp.text}"
                )
            else:
                logger.info(f"飞书推送成功 [{webhook_key}]，共 {len(symbols)} 只股票")

        except requests.RequestException as exc:
            logger.error(f"飞书推送请求异常 [{webhook_key}]：{exc}")

    def send_decision(self, decision: dict, webhook_key: str = "default") -> bool:
        """推送交易决策清单到飞书。

        Args:
            decision: DecisionEngine.generate 的返回（含 buy_list/summary）
            webhook_key: 路由标识
        Returns:
            True=成功
        """
        today = date.today().strftime("%Y-%m-%d")
        sm = decision.get("summary", {})
        buys = decision.get("buy_list", [])[:10]  # 最多展示10只，避免卡片过长

        # 构建买入清单文本（含现价/止损/目标/盈亏比 → 可执行交易指令单）

        def _fp(v):
            return f"{v:.2f}" if isinstance(v, (int, float)) and v else "-"

        def _pct(cur, ref):
            if not cur or not ref:
                return None
            return (cur / ref - 1) * 100

        lines = []
        for i, r in enumerate(buys, 1):
            grade_emoji = {"A": "🔴", "B": "🟡", "C": "🔵"}.get(r.get("grade"), "⚪")
            xq = self._to_xueqiu_code(r["symbol"])
            price = r.get("price", 0) or 0
            sl = r.get("stop_loss", 0) or 0
            tgt = r.get("target")
            sl_pct = _pct(sl, price)
            tgt_pct = _pct(tgt, price) if tgt else None
            rr = None
            if price and sl and tgt and (price - sl) > 0:
                rr = round((tgt - price) / (price - sl), 1)
            cap = r.get("capital", 0) or 0
            cap_str = f"{cap/10000:.1f}万" if cap >= 10000 else f"{int(cap)}"

            head = (
                f"{i}. {grade_emoji}[{r.get('name','')}]"
                f"(https://xueqiu.com/S/{xq})`{r['symbol']}` "
                f"评分{r.get('score','-')} 共振{r.get('resonance','-')}"
            )
            trade = f"现价 {_fp(price)}"
            if sl:
                trade += f" ｜ 止损 {_fp(sl)}"
                if sl_pct is not None:
                    trade += f"({sl_pct:+.1f}%)"
            if tgt:
                trade += f" ｜ 目标 {_fp(tgt)}"
                if tgt_pct is not None:
                    trade += f"({tgt_pct:+.1f}%)"
            if rr is not None:
                trade += f" ｜ 盈亏比 1:{rr}"
            pos = f"仓位{r.get('position_pct',0)}% · {r.get('shares',0)}股 · 资金{cap_str}"
            lines.append(f"{head}\n{trade}\n{pos}")
        buy_text = "\n".join(lines) if lines else "（暂无符合买入条件的标的）"

        # 淘汰摘要
        gc = sm.get("grade_count", {})
        summary_text = (
            f"**日期：** {today}\n"
            f"**候选池：** {decision.get('pool_size', 0)} 只\n"
            f"**买入：** {sm.get('buy_count', 0)} 只（A{gc.get('A',0)} B{gc.get('B',0)} C{gc.get('C',0)}）\n"
            f"**淘汰：** {sm.get('reject_count', 0)} 只\n"
            f"**仓位占用：** {sm.get('position_ratio', 0)}%（现金{sm.get('cash_ratio', 0)}%）"
        )

        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": f"🎯 Sequoia-X 交易决策 | {today}"},
                    "template": "red",
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": summary_text}},
                    {"tag": "hr"},
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"**买入清单：**\n{buy_text}"}},
                    {"tag": "hr"},
                    {"tag": "note",
                     "elements": [{"tag": "plain_text",
                                   "content": "⚠️ 量化决策仅供参考，A股T+1，请结合大盘环境决策"}]},
                ],
            },
        }
        return self._post(payload, webhook_key)

    def send_positions(self, signals: list[dict], summary: dict,
                      webhook_key: str = "default") -> bool:
        """推送持仓扫描报告到飞书（含移动止损/减仓/止盈信号）。"""
        today = date.today().strftime("%Y-%m-%d")
        head = (
            f"**日期：** {today}\n"
            f"**持仓：** {summary.get('count', 0)} 只 "
            f"｜ 总市值 {summary.get('total_value', 0)/10000:.1f}万 "
            f"｜ 浮盈 {summary.get('total_pnl', 0):+,.0f}({summary.get('total_pnl_pct', 0):+.1f}%)\n"
            f"**信号：** ⚠️需处理 {summary.get('danger_count', 0)}/{summary.get('warn_count', 0)} "
            f"｜ 胜率 {summary.get('win_rate', 0)}%({summary.get('winners', 0)}/{summary.get('count', 0)})"
        )
        # 按信号等级分组：danger/warn 优先，其余折叠
        level_emoji = {"danger": "🔴", "warn": "🟡", "success": "🟢", "info": "⚪"}
        urgent, normal = [], []
        for s in signals:
            lv = s.get("signal_level", "info")
            sym = s.get("symbol", "")
            xq = self._to_xueqiu_code(sym) if sym else ""
            name = s.get("name", "") or sym
            line = (
                f"{level_emoji.get(lv, '⚪')}[{name}]"
                f"(https://xueqiu.com/S/{xq})`{sym}` "
                f"现价 {s.get('price', 0):.2f} 浮盈 {s.get('pnl_pct', 0):+.1f}%({s.get('r_multiple', 0):+.1f}R)\n"
                f"▶ **{s.get('action', '持有')}**：{'；'.join(s.get('reasons', ['持有']))}"
            )
            if lv in ("danger", "warn"):
                urgent.append(line)
            else:
                normal.append(line)
        urgent_text = "\n".join(urgent) if urgent else "（无紧急信号）"
        normal_text = "\n".join(normal[:8]) if normal else "（无）"
        if len(normal) > 8:
            normal_text += f"\n…等共 {len(normal)} 只正常持有"
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": f"📊 Sequoia-X 持仓扫描 | {today}"},
                    "template": "orange",
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": head}},
                    {"tag": "hr"},
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"**⚠️ 需处理信号：**\n{urgent_text}"}},
                    {"tag": "hr"},
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"**✅ 正常持有：**\n{normal_text}"}},
                    {"tag": "note", "elements": [{"tag": "plain_text",
                     "content": "移动止损规则：跌破止损/MA20清仓 ｜ 跌破MA10减半仓 ｜ 盈利1R保本 ｜ 2R锁利"}]},
                ],
            },
        }
        return self._post(payload, webhook_key)

    def _post(self, payload: dict, webhook_key: str = "default") -> bool:
        """底层 POST，复用于 send 和 send_decision。"""
        url = self.settings.get_webhook_url(webhook_key)
        try:
            resp = requests.post(url, data=json.dumps(payload),
                                 headers={"Content-Type": "application/json"}, timeout=10)
            resp_json = resp.json()
            if resp.status_code != 200 or resp_json.get("code") != 0:
                logger.error(f"飞书推送失败 [{webhook_key}] HTTP={resp.status_code} {resp.text}")
                return False
            logger.info(f"飞书推送成功 [{webhook_key}]")
            return True
        except requests.RequestException as exc:
            logger.error(f"飞书推送异常 [{webhook_key}]：{exc}")
            return False
