"""大盘分析 - 估值与宏观（MarketAnalyzer mixin 组件）。

由 :class:`sequoia_x.analysis.market.MarketAnalyzer` 多重继承组合，
不单独实例化；方法通过 ``self.db_path`` 等访问门面状态。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


class FundamentalsMixin:
    """估值与宏观计算逻辑（MarketAnalyzer 的 mixin 组件）。"""

    def _fetch_valuation(self) -> dict:
        """指数估值分位 + 股权风险溢价 ERP。"""
        import akshare as ak

        result: dict = {"indices": [], "erp": {}}
        # 指数PE/PB
        idx_map = {"沪深300": "沪深300", "上证50": "上证50", "创业板指": "创业板指"}
        for display, ak_name in idx_map.items():
            try:
                df = ak.stock_index_pe_lg(symbol=ak_name)
                pe_col = "滚动市盈率" if "滚动市盈率" in df.columns else df.columns[2]
                pe_series = pd.to_numeric(df[pe_col], errors="coerce").dropna()
                latest_pe = round(float(pe_series.iloc[-1]), 2)
                percentile = round(float((pe_series < latest_pe).sum() / len(pe_series) * 100), 1)
                valuation = "低估" if percentile < 30 else ("高估" if percentile > 70 else "合理")
                result["indices"].append({
                    "name": display, "pe": latest_pe,
                    "pe_percentile": percentile, "valuation": valuation,
                })
            except Exception as exc:
                logger.warning(f"{display}估值获取失败：{exc}")

        # ERP：沪深300盈利收益率 - 10年期国债收益率
        try:
            from datetime import timedelta
            _now = datetime.now()
            bond_df = ak.bond_china_yield(
                start_date=(_now - timedelta(days=30)).strftime("%Y%m%d"),
                end_date=_now.strftime("%Y%m%d"),
            )
            bond_row = bond_df[bond_df["曲线名称"] == "中债国债收益率曲线"]
            treasury_10y = float(bond_row.iloc[-1]["10年"]) if len(bond_row) > 0 else 2.0
            hs300 = next((i for i in result["indices"] if i["name"] == "沪深300"), None)
            if hs300:
                earnings_yield = 100 / hs300["pe"]  # 盈利收益率
                erp = round(earnings_yield - treasury_10y, 2)
                level = "股优于债" if erp > 2 else ("债优于股" if erp < 0 else "股债均衡")
                result["erp"] = {
                    "earnings_yield": round(earnings_yield, 2),
                    "treasury_10y": treasury_10y,
                    "erp": erp, "level": level,
                }
        except Exception as exc:
            logger.warning(f"ERP计算失败：{exc}")

        return result

    def _fetch_macro(self) -> dict:
        """宏观数据：PMI、M2、社融。"""
        import akshare as ak

        result: dict = {}
        # PMI
        try:
            df = ak.macro_china_pmi()
            latest = df.iloc[0]
            pmi_val = float(latest["制造业-指数"])
            result["pmi"] = {
                "month": str(latest["月份"]),
                "value": pmi_val,
                "yoy": round(float(latest["制造业-同比增长"]), 2),
                "signal": "扩张" if pmi_val >= 50 else "收缩",
            }
        except Exception as exc:
            logger.warning(f"PMI获取失败：{exc}")

        # M2
        try:
            df = ak.macro_china_money_supply()
            latest = df.iloc[0]
            result["m2"] = {
                "month": str(latest["月份"]),
                "yoy": round(float(latest["货币和准货币(M2)-同比增长"]), 2),
            }
        except Exception as exc:
            logger.warning(f"M2获取失败：{exc}")

        # 社融
        try:
            df = ak.macro_china_shrzgm()
            latest = df.iloc[0]
            result["social_financing"] = {
                "month": str(latest["月份"]),
                "value": float(latest["社会融资规模增量"]),
            }
        except Exception as exc:
            logger.warning(f"社融获取失败：{exc}")

        return result

    # ------------------------------------------------------------------
    # 六、涨跌停结构 + 龙虎榜
    # ------------------------------------------------------------------
