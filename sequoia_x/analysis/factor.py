"""多因子库：从日K+财务数据提取30个因子（6大类），全向量化计算。

因子分类（对标Barra/Axioma多因子模型，A股实证有效）：
  - 动量(8): 趋势延续，强者恒强（A股中期动量效应显著）
  - 反转(4): 短周期反转（A股5-20日反转效应强于美股）
  - 波动(4): 低波动溢价 + 收敛→突破信号
  - 流动性(4): 流动性溢价，避开流动性陷阱
  - 量价(5): 量价关系揭示主力行为（现有策略拆解）
  - 质量(5): 高质量公司溢价（来自stock_finance）

设计原则：
  - 每个因子返回 float（横截面标准化前的原始值），横截面Rank IC评估
  - 因子值越大→预期收益越高（反转/波动因子取负，保证方向一致）
  - 纯向量化（pandas rolling/shift），无iterrows，5000只×60日<3s
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


# ── 因子显著性过滤门槛（A 股月频工业标准）──
# alpha 泄漏修复：旧门槛 |IC|>0.015 & ICIR>0.3 过低，放噪声因子进权重稀释强信号。
# 新门槛三者全过才进权重（宁可少不要错），样本不足(n<MIN_IC_SAMPLES)的因子默认不显著。
MIN_IC_ABS = 0.03       # IC 绝对值下限
MIN_ICIR_ABS = 0.5      # ICIR 绝对值下限（信息比率）
MIN_T_STAT = 2.0        # t 统计量下限（≈95% 置信），样本不足时可下调到 1.65(90%)
MIN_IC_SAMPLES = 6      # 最少月度 IC 观测数，否则默认不显著

# ── 因子拥挤度衰减（行业集中度 HHI → 软连续权重衰减）──
# 高拥挤因子选出的票集中在少数行业，反转风险大。用多头组合行业 HHI
# 衡量拥挤，对拥挤因子权重施加软衰减（保留 30% 下限，与 oos_decay 哲学一致）。
# 阈值为 84 行业、500 采样（top20%≈100 名）的工程估值，首跑后看日志分布再调。
CROWDING_TOP_PCT = 0.20       # 多头侧分位数（top 20%）
CROWDING_LOOKBACK_MONTHS = 3  # 取最近 N 月 HHI 均值降噪
CROWDING_SAFE = 0.08          # HHI ≤ 此值不衰减（分散健康）
CROWDING_MAX = 0.30           # HHI ≥ 此值衰减到下限（重度拥挤）
CROWDING_FLOOR = 0.30         # 衰减下限（保留 30% 分散价值，不归零）

# P10b: ML 因子近期 IC 稳定性门控——抑制衰退期噪声、保留强周期 alpha。
# 仿 P6 拥挤度衰减的连续软门控，用 ML 近期样本外 IC 均值做稳定性度量。
ML_IC_FULL = 0.06            # 近期均 IC ≥ 此值 → 满权重（略低于全样本均0.078）
ML_IC_WEAK = 0.015           # 近期均 IC ≤ 此值 → 降到下限（衰退窗口均IC≈0.002~0.02）
ML_PENALTY_FLOOR = 0.35      # 重度衰退仍保留 35%（与 P6/P2 半保留下限哲学一致）

# P16: 因子正交化——IC 相关性驱动的权重去冗余。
# 高相关因子簇内，非基准因子的权重按 (1-corr²) 折扣（增量信息系数）。
ORTH_THRESHOLD = 0.95         # IC 相关性 |r| > 此值才归簇（P16实测：0.6折扣最强alpha致年化降5pp，调高至近完全镜像等效回退）


# 因子分类（key=因子名, category=大类, direction=+1正向/-1反向）
FACTOR_META: dict[str, dict] = {
    # ── 动量(8) ──
    "mom_5":       {"category": "动量", "desc": "5日收益率"},
    "mom_10":      {"category": "动量", "desc": "10日收益率"},
    "mom_20":      {"category": "动量", "desc": "20日收益率"},
    "mom_60":      {"category": "动量", "desc": "60日收益率"},
    "rps_120":     {"category": "动量", "desc": "120日相对强度"},
    "high_dist":   {"category": "动量", "desc": "距20日新高(越近越强)"},
    "mom_52w":     {"category": "动量", "desc": "52周动量"},
    "mom_accel":   {"category": "动量", "desc": "动量加速度(近5日-前5日)"},
    # ── 反转(4)（取负，跌得多的分高）──
    "rev_5":       {"category": "反转", "desc": "5日反转", "reverse": True},
    "rev_10":      {"category": "反转", "desc": "10日反转", "reverse": True},
    "rev_20":      {"category": "反转", "desc": "20日反转", "reverse": True},
    "oversold":    {"category": "反转", "desc": "超跌反弹", "reverse": True},
    # ── 波动(4)（取负，低波动溢价）──
    "vol_20":      {"category": "波动", "desc": "20日波动率", "reverse": True},
    "atr_pct":     {"category": "波动", "desc": "ATR占比", "reverse": True},
    "vol_shrink":  {"category": "波动", "desc": "波动率收敛"},
    "skew":        {"category": "波动", "desc": "偏度", "reverse": True},
    # ── 流动性(4) ──
    "turnover":    {"category": "流动性", "desc": "成交额"},
    "volume_ratio":{"category": "流动性", "desc": "量比"},
    "liq_rank":    {"category": "流动性", "desc": "流动性分位"},
    "amihud":      {"category": "流动性", "desc": "Amihud非流动性", "reverse": True},
    # ── 换手率(4)（A股核心量能维度，baostock turn字段）──
    "turn_ratio":    {"category": "换手率", "desc": "当日换手率"},
    "turn_ma5":      {"category": "换手率", "desc": "5日均换手率"},
    "turn_surge":    {"category": "换手率", "desc": "换手率倍数(今日/20日均)"},
    "turn_reversal": {"category": "换手率", "desc": "低换手率反转", "reverse": True},
    # ── 量价(5) ──
    "ma_cross":    {"category": "量价", "desc": "均线金叉信号"},
    "vol_surge":   {"category": "量价", "desc": "放量倍数"},
    "vol_shrink_p":{"category": "量价", "desc": "缩量程度", "reverse": True},
    "flag_tight":  {"category": "量价", "desc": "旗形收敛度"},
    "vp_divergence":{"category": "量价", "desc": "量价共振度"},
    # ── 结构(4)（A股微观结构因子，纯量价，学术验证有效）──
    "close_pos":   {"category": "结构", "desc": "收盘价位置（越高越强）"},
    "gap":         {"category": "结构", "desc": "隔夜跳空幅度"},
    "range_pct":   {"category": "结构", "desc": "日内振幅", "reverse": True},
    "vol_wt_mom":  {"category": "结构", "desc": "量加权动量"},
    # ── 资金(2)（主力资金流向，东财fund_flow表）──
    "main_net":    {"category": "资金", "desc": "主力净流入额"},
    "main_pct":    {"category": "资金", "desc": "主力净流入占比"},
    # ── 质量(5) ──
    "roe":         {"category": "质量", "desc": "净资产收益率"},
    "np_margin":   {"category": "质量", "desc": "净利率"},
    "gp_margin":   {"category": "质量", "desc": "毛利率"},
    "rev_growth":  {"category": "质量", "desc": "营收增速"},
    "profit_growth":{"category": "质量", "desc": "利润增速"},
    "cfo_yield":   {"category": "盈利质量", "desc": "经营现金流/营收（现金收益率，高=盈利扎实）"},
    "earnings_quality": {"category": "盈利质量", "desc": "经营现金流/净利润（盈余质量，高=真实盈利）"},
    # ── 成长(1)（同比增长率，来自stock_finance）──
    "yoy_ni":      {"category": "成长", "desc": "净利润同比增长率"},
    # ── 估值(2) ──
    "pe_ratio":    {"category": "估值", "desc": "市盈率TTM（低估值溢价）"},
    "pb_ratio":    {"category": "估值", "desc": "市净率（低估值溢价）"},
    # ── 营运效率(3)（Barra风格周转率，来自stock_finance）──
    "asset_turn":  {"category": "营运", "desc": "总资产周转率"},
    "inv_turn":    {"category": "营运", "desc": "存货周转率"},
    "nr_turn":     {"category": "营运", "desc": "应收账款周转率"},
    # ── 龙虎榜(2) ──
    "lhb_count":   {"category": "龙虎榜", "desc": "近30天上榜次数"},
    "lhb_netbuy":  {"category": "龙虎榜", "desc": "近30天龙虎榜净买入额"},
    # ── 资金流近似(3)（用日K构建，无需外部接口）──
    "flow_strength":  {"category": "资金流", "desc": "上涨日成交额占比"},
    "flow_weighted":  {"category": "资金流", "desc": "涨跌幅加权资金流"},
    "flow_trend":       {"category": "资金流", "desc": "5日vs20日资金流趋势"},
    "flow_super_ratio": {"category": "资金流", "desc": "超大单占比（机构资金方向）"},
    "flow_intensity":   {"category": "资金流", "desc": "主力资金净流入强度"},
    # ── 北向资金(2)（沪深港通持股，来自north_hold表）──
    "nb_holding_pct":  {"category": "北向", "desc": "北向持股占比（机构持仓水平）"},
    "nb_inflow":       {"category": "北向", "desc": "近20日北向持股占比变化（增量资金方向）"},
    # ── 融资融券(3)（杠杆资金方向，北向断供后的替代信号）──
    "margin_balance":  {"category": "融资融券", "desc": "融资余额（杠杆看多水平，高=拥挤）"},
    "margin_netbuy":   {"category": "融资融券", "desc": "融资净买入额（当日杠杆资金流入）"},
    "short_ratio":     {"category": "融资融券", "desc": "融券占比（看空比例，高=空头情绪）"},
    # ── 基金持仓(2)（公募基金重仓，来自fund_hold表）──
    "fund_holding":    {"category": "基金持仓", "desc": "基金持有家数（机构共识/拥挤，高=拥挤）", "reverse": True},
    "fund_inflow":     {"category": "基金持仓", "desc": "基金季度增仓比例（高=机构加仓）"},
    # ── 沪深300 Beta(2)（相对沪深300的系统性暴露）──
    "beta_300":        {"category": "Beta", "desc": "相对沪深300 Beta（低Beta防御溢价）", "reverse": True},
    "rel_strength_300":{"category": "Beta", "desc": "相对沪深300超额收益（20日相对强度）"},
    "block_discount":  {"category": "大宗交易", "desc": "近30天大宗交易加权折价率（折价=机构接货，正向）"},
    "holder_change":   {"category": "筹码集中度", "desc": "股东户数环比变化（减少=筹码集中→看涨，负向）"},
    "holder_count":    {"category": "筹码集中度", "desc": "股东户数（多=分散，负向）"},
    "holder_avg_value":{"category": "筹码集中度", "desc": "户均持股市值（高=大户持仓，正向）"},
}


def compute_factors(df: pd.DataFrame, finance: dict | None = None,
                   fund_flow: dict | None = None, lhb_data: dict | None = None,
                   north_hold: pd.DataFrame | None = None,
                   margin: dict | None = None,
                   fund_hold: dict | None = None,
                   block_data: dict | None = None,
                   holder_data: dict | None = None,
                   index_ret: pd.Series | None = None, ) -> dict[str, float]:
    """计算单只股票的全部因子值（向量化，基于完整K线序列）。

    Args:
        df: OHLCV DataFrame（按日期升序），需含 open/high/low/close/volume/turnover
        finance: 财务数据 dict（roe/np_margin等），可选
        fund_flow: 资金流向 dict，可选
        lhb_data: 龙虎榜 dict，可选
        north_hold: 北向资金持股 DataFrame（列含 date/hold_pct，升序），可选
        margin: 融资融券 dict（rzye/rzbuy/rqlts/rqye），可选

   Returns:
        {因子名: 因子值}，取最后一日（最新截面）。无效因子返回nan。
    """
    if len(df) < 20:
        return {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    turnover = df["turnover"]
    open_ = df["open"]

    factors: dict[str, float] = {}

    def _safe_float(val):
        """安全转float，None/NaN→NaN。"""
        if val is None:
            return np.nan
        try:
            f = float(val)
            return f if f == f else np.nan  # NaN check
        except (TypeError, ValueError):
            return np.nan

    # ════════ 动量(8) ════════
    factors["mom_5"] = _ret(close, 5)
    factors["mom_10"] = _ret(close, 10)
    factors["mom_20"] = _ret(close, 20)
    factors["mom_60"] = _ret(close, 60)
    factors["rps_120"] = _ret(close, 120) if len(df) >= 120 else np.nan
    high_20 = high.rolling(20).max()
    factors["high_dist"] = (close.iloc[-1] / high_20.iloc[-1] - 1) * 100 if pd.notna(high_20.iloc[-1]) else np.nan
    factors["mom_52w"] = _ret(close, 250) if len(df) >= 250 else np.nan
    recent_mom = _ret(close, 5)
    prior_mom = (close.iloc[-6] / close.iloc[-11] - 1) * 100 if len(df) >= 11 else np.nan
    factors["mom_accel"] = (recent_mom - prior_mom) if pd.notna(prior_mom) and pd.notna(recent_mom) else np.nan

    # ════════ 反转(4)（取负）════════
    factors["rev_5"] = -factors["mom_5"]
    factors["rev_10"] = -factors["mom_10"]
    factors["rev_20"] = -factors["mom_20"]
    high_20b = high.rolling(20).max()
    dd = (high_20b.iloc[-1] - close.iloc[-1]) / high_20b.iloc[-1] * 100 if pd.notna(high_20b.iloc[-1]) else 0
    factors["oversold"] = dd

    # ════════ 波动(4)（取负）════════
    rets = close.pct_change()
    factors["vol_20"] = -float(rets.tail(20).std() * np.sqrt(252) * 100) if len(rets) >= 20 else np.nan
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    factors["atr_pct"] = -(atr.iloc[-1] / close.iloc[-1] * 100) if pd.notna(atr.iloc[-1]) and close.iloc[-1] else np.nan
    vol_recent = rets.tail(5).std()
    vol_prior = rets.iloc[-25:-5].std() if len(rets) >= 25 else 0
    factors["vol_shrink"] = float((vol_prior - vol_recent) / vol_prior * 100) if vol_prior and pd.notna(vol_prior) and pd.notna(vol_recent) else np.nan
    factors["skew"] = -float(rets.tail(20).skew()) if len(rets) >= 20 else np.nan

    # ════════ 流动性(4) ════════
    factors["turnover"] = float(turnover.iloc[-1] / 1e8)  # 亿元
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    factors["volume_ratio"] = float(volume.iloc[-1] / vol_ma5.iloc[-1]) if pd.notna(vol_ma5.iloc[-1]) and vol_ma5.iloc[-1] else 1.0
    factors["liq_rank"] = float(turnover.iloc[-1])  # 原始值，横截面排名时用
    # Amihud非流动性 = |收益率|/成交额
    if turnover.iloc[-1] > 0 and pd.notna(rets.iloc[-1]):
        factors["amihud"] = -float(abs(rets.iloc[-1]) / turnover.iloc[-1] * 1e10)
    else:
        factors["amihud"] = np.nan

    # ════════ 换手率(4)（baostock turn字段，A股核心量能维度）════════
    turn = df["turn"] if "turn" in df.columns else pd.Series(dtype=float)
    turn_valid = turn.notna().any() and len(turn) > 0
    if turn_valid:
        # turn可能含NaN（历史数据），前向填充后取最新
        turn_ff = turn.ffill()
        factors["turn_ratio"] = float(turn_ff.iloc[-1]) if pd.notna(turn_ff.iloc[-1]) else np.nan
        turn_ma5_v = turn_ff.rolling(5).mean()
        factors["turn_ma5"] = float(turn_ma5_v.iloc[-1]) if pd.notna(turn_ma5_v.iloc[-1]) else np.nan
        turn_ma20_v = turn_ff.rolling(20).mean()
        if pd.notna(turn_ma20_v.iloc[-1]) and turn_ma20_v.iloc[-1] > 0:
            factors["turn_surge"] = float(turn_ff.iloc[-1] / turn_ma20_v.iloc[-1])
        else:
            factors["turn_surge"] = np.nan
        # 低换手率反转：20日均换手的负值（换手越低=分越高，左侧信号）
        factors["turn_reversal"] = -float(turn_ma20_v.iloc[-1]) if pd.notna(turn_ma20_v.iloc[-1]) else np.nan
    else:
        for k in ["turn_ratio", "turn_ma5", "turn_surge", "turn_reversal"]:
            factors[k] = np.nan

    # ════════ 量价(5) ════════
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    if len(df) >= 21:
        # 金叉：昨日ma5<ma20，今日ma5>ma20
        cross = 1.0 if (ma5.iloc[-2] < ma20.iloc[-2] and ma5.iloc[-1] > ma20.iloc[-1]) else 0.0
    else:
        cross = 0.0
    factors["ma_cross"] = cross
    factors["vol_surge"] = float(volume.iloc[-1] / vol_ma20.iloc[-1]) if pd.notna(vol_ma20.iloc[-1]) and vol_ma20.iloc[-1] else 1.0
    recent_vol = volume.tail(3).mean()
    factors["vol_shrink_p"] = -float(recent_vol / vol_ma20.iloc[-1]) if pd.notna(vol_ma20.iloc[-1]) and vol_ma20.iloc[-1] else np.nan
    # 旗形收敛：10日振幅/40日振幅（越小越收敛，取负让收敛=高分）
    if len(df) >= 40:
        lo10 = low.tail(10).min()
        lo40 = low.tail(40).min()
        range_10 = (high.tail(10).max() - lo10) / lo10 if lo10 else np.nan
        range_40 = (high.tail(40).max() - lo40) / lo40 if lo40 else np.nan
        factors["flag_tight"] = -float(range_10 / range_40) if range_40 and range_40 != 0 and pd.notna(range_40) and pd.notna(range_10) else np.nan
    else:
        factors["flag_tight"] = np.nan
    # 量价共振：涨+放量 = 正，跌+放量=负（简化）
    chg = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
    vr = volume.iloc[-1] / vol_ma20.iloc[-1] if pd.notna(vol_ma20.iloc[-1]) and vol_ma20.iloc[-1] else 1
    factors["vp_divergence"] = float(chg * vr * 100)

    # ════════ 结构(4) ════════
    if len(df) >= 20:
        lo20 = float(low.tail(20).min())
        hi20 = float(high.tail(20).max())
        denom = hi20 - lo20 if (hi20 - lo20) != 0 else 1
        factors["close_pos"] = float((close.iloc[-1] - lo20) / denom * 100)
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else float(close.iloc[-1])
        factors["gap"] = float((prev_close - float(open_.iloc[-1])) / prev_close * 100) if prev_close else np.nan
        factors["range_pct"] = -float((float(high.iloc[-1]) - float(low.iloc[-1])) / float(close.iloc[-1]) * 100)
        vol_ma20_last = float(volume.tail(20).mean()) if len(volume) >= 20 else 1
        factors["vol_wt_mom"] = float((close.iloc[-1] / close.iloc[-20] - 1) * (volume.iloc[-1] / vol_ma20_last if vol_ma20_last else 1) * 100)
    else:
        for k in ["close_pos", "gap", "range_pct", "vol_wt_mom"]:
            factors[k] = np.nan

    # ════════ 资金(2) ════════
    if fund_flow:
        factors["main_net"] = _safe_float(fund_flow.get("main_net"))
        factors["main_pct"] = _safe_float(fund_flow.get("main_pct"))
        # 复合资金流因子：超大单占比（机构资金方向）
        super_net = _safe_float(fund_flow.get("super_net"))
        big_net = _safe_float(fund_flow.get("big_net"))
        total_flow = abs(super_net) + abs(big_net) + 1
        factors["flow_super_ratio"] = float(super_net / total_flow)
        # 资金流强度（主力净流入绝对值 / 成交额）
        factors["flow_intensity"] = float(abs(_safe_float(fund_flow.get("main_net"))) / (turnover.iloc[-1] + 1))
    else:
        for k in ["main_net", "main_pct", "flow_super_ratio", "flow_intensity"]:
            factors[k] = np.nan

    # ════════ 龙虎榜(2) ════════
    if lhb_data:
        factors["lhb_count"] = float(lhb_data.get("count", 0))
        factors["lhb_netbuy"] = float(lhb_data.get("net_buy", 0))
    else:
        for k in ["lhb_count", "lhb_netbuy"]:
            factors[k] = np.nan

    # ════════ 大宗交易(1) ════════
    if block_data:
        # block_data: {"discount": 近30天加权折价率, "count": 次数}
        factors["block_discount"] = float(block_data.get("discount", 0))
    else:
        factors["block_discount"] = np.nan

    # ════════ 股东户数/筹码集中度(3) ════════
    if holder_data:
        factors["holder_change"] = _safe_float(holder_data.get("holder_change"))
        factors["holder_count"] = _safe_float(holder_data.get("holder_num"))
        factors["holder_avg_value"] = _safe_float(holder_data.get("avg_value"))
    else:
        for k in ["holder_change", "holder_count", "holder_avg_value"]:
            factors[k] = np.nan

    # ════════ 质量(5) + 成长(1) + 营运效率(3) ════════
    if finance:
        factors["roe"] = _safe_float(finance.get("roe"))             # 净资产收益率
        factors["np_margin"] = _safe_float(finance.get("np_margin")) # 净利率
        factors["gp_margin"] = _safe_float(finance.get("gp_margin")) # 毛利率
        factors["rev_growth"] = _safe_float(finance.get("yoy_eps"))  # EPS增速（营收增速代理）
        factors["profit_growth"] = _safe_float(finance.get("yoy_pni"))  # 扣非净利润增速
        # 成长：净利润同比增长率
        factors["yoy_ni"] = _safe_float(finance.get("yoy_ni"))
        # 营运效率（Barra风格周转率）
        factors["asset_turn"] = _safe_float(finance.get("asset_turn"))
        factors["inv_turn"] = _safe_float(finance.get("inv_turn"))
        factors["nr_turn"] = _safe_float(finance.get("nr_turn"))
        # 盈利质量（现金流比率，P15）
        factors["cfo_yield"] = _safe_float(finance.get("cfo_to_or"))      # 经营现金流/营收
        factors["earnings_quality"] = _safe_float(finance.get("cfo_to_np"))  # 经营现金流/净利润
    else:
        for k in ["roe", "np_margin", "gp_margin", "rev_growth", "profit_growth",
                  "yoy_ni", "asset_turn", "inv_turn", "nr_turn",
                  "cfo_yield", "earnings_quality"]:
            factors[k] = np.nan

    # ════════ 北向资金(2)（沪深港通持股）════════
    if north_hold is not None and len(north_hold) > 0:
        try:
            nh = north_hold.sort_values("date")
            last_pct = _safe_float(nh.iloc[-1]["hold_pct"])
            factors["nb_holding_pct"] = last_pct
            if len(nh) >= 21:
                pct_20d_ago = _safe_float(nh.iloc[-21]["hold_pct"])
                if not np.isnan(last_pct) and not np.isnan(pct_20d_ago):
                    factors["nb_inflow"] = last_pct - pct_20d_ago
                else:
                    factors["nb_inflow"] = np.nan
            else:
                factors["nb_inflow"] = np.nan
        except Exception:
            factors["nb_holding_pct"] = np.nan
            factors["nb_inflow"] = np.nan
    else:
        factors["nb_holding_pct"] = np.nan
        factors["nb_inflow"] = np.nan

    # ══════ 融资融券(3)（杠杆资金方向，北向断供后的替代信号）══════
    if margin:
        rzye = _safe_float(margin.get("rzye"))
        rqye = _safe_float(margin.get("rqye"))
        factors["margin_balance"] = rzye
        factors["margin_netbuy"] = _safe_float(margin.get("rzbuy"))
        if rqye is not None and rzye is not None and rzye > 0:
            factors["short_ratio"] = rqye / (rzye + rqye)
        else:
            factors["short_ratio"] = np.nan
    else:
        for k in ["margin_balance", "margin_netbuy", "short_ratio"]:
            factors[k] = np.nan

    # ════════ 基金持仓(2)（公募基金重仓，来自fund_hold表）════════
    if fund_hold:
        factors["fund_holding"] = _safe_float(fund_hold.get("fund_count"))
        factors["fund_inflow"] = _safe_float(fund_hold.get("change_pct"))
    else:
        for k in ["fund_holding", "fund_inflow"]:
            factors[k] = np.nan

    # ════════ 沪深300 Beta(2)（相对沪深300系统性暴露）════════
    if index_ret is not None and len(index_ret) >= 60:
        try:
            stock_ret = close.pct_change().dropna()
            sr = stock_ret.iloc[-60:]
            mr = pd.Series(index_ret).dropna().iloc[-60:]
            min_len = min(len(sr), len(mr))
            if min_len >= 30:
                sr = sr.iloc[-min_len:].values
                mr = mr.iloc[-min_len:].values
                m_var = float(np.var(mr))
                if m_var > 0:
                    cov = float(np.cov(sr, mr)[0, 1])
                    factors["beta_300"] = max(-2.0, min(3.0, cov / m_var))
                else:
                    factors["beta_300"] = np.nan
                # 相对强度：20日个股超额收益
                if len(stock_ret) >= 21 and len(mr) >= 21:
                    s20 = float(stock_ret.iloc[-21:].sum())
                    m20 = float(mr[-21:].sum())
                    factors["rel_strength_300"] = s20 - m20
                else:
                    factors["rel_strength_300"] = np.nan
            else:
                factors["beta_300"] = np.nan
                factors["rel_strength_300"] = np.nan
        except Exception:
            factors["beta_300"] = np.nan
            factors["rel_strength_300"] = np.nan
    else:
        factors["beta_300"] = np.nan
        factors["rel_strength_300"] = np.nan

    # 清理nan→None（序列化友好）
    return {k: (None if (v != v) else round(v, 4)) for k, v in factors.items()}


def _ret(close: pd.Series, n: int) -> float:
    """N日收益率%。"""
    if len(close) <= n:
        return np.nan
    prev = close.iloc[-n - 1]
    if prev == 0 or pd.isna(prev):
        return np.nan
    return float((close.iloc[-1] / prev - 1) * 100)


def compute_all_factors(
    engine, symbols: list[str], finance_map: dict | None = None,
) -> pd.DataFrame:
    """批量计算全市场因子截面（最新一日），返回 DataFrame。

    横截面标准化前的原始值，下游做Rank IC评估或多因子合成。

    Args:
        engine: DataEngine
        symbols: 股票代码列表
        finance_map: {symbol: finance_dict}，可选

    Returns:
        DataFrame，index=symbol，columns=因子名
    """
    rows = []
    total = len(symbols)
    for i, sym in enumerate(symbols):
        try:
            df = engine.get_ohlcv(sym)
            if len(df) < 20:
                continue
            finance = finance_map.get(sym) if finance_map else None
            factors = compute_factors(df, finance)
            factors["symbol"] = sym
            rows.append(factors)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).set_index("symbol")
    logger.info(f"因子计算完成：{len(result)}/{total} 只")
    return result


def cross_section_rank(df: pd.DataFrame) -> pd.DataFrame:
    """横截面百分位排名（0-100），消除量纲差异。

    每个因子列独立排名，值为该股票在全市场的百分位。
    用于多因子合成（等权或IC加权前必须标准化）。
    """
    return df.rank(pct=True) * 100


def compute_factor_series(df: pd.DataFrame, factor_names: list[str] | None = None, finance_series: dict[str, pd.Series] | None = None, index_ret: pd.Series | None = None) -> dict[str, pd.Series]:
    """向量化计算完整序列的因子值（性能优化核心）。

    一次性算出每个因子在每个交易日的值，下游按月取截面。
    相比逐月重算 compute_factors，性能提升50倍以上。

    Args:
        df: OHLCV DataFrame（按日期升序）
        factor_names: 需要计算的因子列表，None=全部因子
        finance_series: 质量因子的point-in-time序列（{因子名: pd.Series}），None=跳过质量因子

    Returns:
        {因子名: pd.Series（与df等长，每日因子值）}
    """
    if len(df) < 20:
        return {}
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    turnover = df["turnover"]
    rets = close.pct_change()

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    high_20 = high.rolling(20).max()
    high_120 = high.rolling(120).max() if len(df) >= 120 else pd.Series(np.nan, index=df.index)
    high_250 = high.rolling(250).max() if len(df) >= 250 else pd.Series(np.nan, index=df.index)

    series: dict[str, pd.Series] = {}

    # 动量
    series["mom_5"] = close.pct_change(5) * 100
    series["mom_10"] = close.pct_change(10) * 100
    series["mom_20"] = close.pct_change(20) * 100
    series["mom_60"] = close.pct_change(60) * 100
    series["rps_120"] = close.pct_change(120) * 100 if len(df) >= 120 else pd.Series(np.nan, index=df.index)
    series["high_dist"] = (close / high_20 - 1) * 100
    series["mom_52w"] = close.pct_change(250) * 100 if len(df) >= 250 else pd.Series(np.nan, index=df.index)
    series["mom_accel"] = series["mom_5"] - series["mom_5"].shift(5)

    # 反转（取负）
    series["rev_5"] = -series["mom_5"]
    series["rev_10"] = -series["mom_10"]
    series["rev_20"] = -series["mom_20"]
    series["oversold"] = (high_20 - close) / high_20 * 100

    # 波动（取负）
    series["vol_20"] = -rets.rolling(20).std() * np.sqrt(252) * 100
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    series["atr_pct"] = -atr / close * 100
    vol_recent = rets.rolling(5).std()
    vol_prior = rets.shift(5).rolling(20).std()
    series["vol_shrink"] = (vol_prior - vol_recent) / vol_prior.replace(0, np.nan) * 100
    series["skew"] = -rets.rolling(20).skew()

    # 流动性
    series["turnover"] = turnover / 1e8
    series["volume_ratio"] = volume / vol_ma5.replace(0, np.nan)
    series["liq_rank"] = turnover
    series["amihud"] = -rets.abs() / turnover.replace(0, np.nan) * 1e10

    # 换手率（baostock turn字段）
    if "turn" in df.columns:
        turn_s = df["turn"].ffill()
        series["turn_ratio"] = turn_s
        series["turn_ma5"] = turn_s.rolling(5).mean()
        turn_ma20_s = turn_s.rolling(20).mean()
        series["turn_surge"] = turn_s / turn_ma20_s.replace(0, np.nan)
        series["turn_reversal"] = -turn_ma20_s

    # 量价
    series["ma_cross"] = ((ma5.shift(1) < ma20.shift(1)) & (ma5 > ma20)).astype(float)
    series["vol_surge"] = volume / vol_ma20.replace(0, np.nan)
    series["vol_shrink_p"] = -volume.rolling(3).mean() / vol_ma20.replace(0, np.nan)
    range_10 = (high.rolling(10).max() - low.rolling(10).min()) / low.rolling(10).min().replace(0, np.nan)
    range_40 = (high.rolling(40).max() - low.rolling(40).min()) / low.rolling(40).min().replace(0, np.nan)
    series["flag_tight"] = -range_10 / range_40.replace(0, np.nan)
    series["vp_divergence"] = rets * (volume / vol_ma20.replace(0, np.nan)) * 100

    # 结构（A股微观结构因子）
    series["close_pos"] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min().replace(0, np.nan)) * 100
    series["gap"] = (close.shift(1) - open_) / close.shift(1).replace(0, np.nan) * 100  # 开盘跳空
    series["range_pct"] = -(high - low) / close.replace(0, np.nan) * 100  # 日内振幅（取负=小振幅溢价）
    series["vol_wt_mom"] = (close.pct_change(20) * (volume / vol_ma20.replace(0, np.nan))).rolling(20).mean() * 100

    # 资金流近似（用日K构建，无需外部接口）
    up_mask = (close > open_).astype(float)
    turnover_total_60 = turnover.rolling(60, min_periods=20).sum()
    up_turnover_60 = (turnover * up_mask).rolling(60, min_periods=20).sum()
    series["flow_strength"] = (up_turnover_60 / turnover_total_60.replace(0, np.nan) - 0.5) * 100
    rets_all = close.pct_change()
    flow_num = (rets_all * turnover).rolling(20, min_periods=10).sum()
    flow_den = turnover.rolling(20, min_periods=10).sum().replace(0, np.nan)
    series["flow_weighted"] = flow_num / flow_den * 10000
    flow_5 = (rets_all * turnover).rolling(5).sum() / turnover.rolling(5).sum().replace(0, np.nan)
    flow_20v = (rets_all * turnover).rolling(20, min_periods=10).sum() / turnover.rolling(20, min_periods=10).sum().replace(0, np.nan)
    series["flow_trend"] = (flow_5 - flow_20v) * 10000

    # 沪深300 Beta（相对沪深300系统性暴露）
    if index_ret is not None and "date" in df.columns:
        try:
            _idx_map = index_ret.to_dict() if hasattr(index_ret, "to_dict") else dict(index_ret)
            idx_ret_aligned = df["date"].astype(str).map(_idx_map)
            if idx_ret_aligned.notna().sum() >= 60:
                _ir = idx_ret_aligned.astype(float)
                _mvar = _ir.rolling(60, min_periods=30).var()
                _cov = rets.rolling(60, min_periods=30).cov(_ir)
                _beta = (_cov / _mvar.replace(0, np.nan)).clip(-2.0, 3.0)
                series["beta_300"] = _beta
                series["rel_strength_300"] = rets.rolling(20, min_periods=10).sum() - _ir.rolling(20, min_periods=10).sum()
        except Exception:
            pass

    # 质量因子（point-in-time，从finance_series注入）
    if finance_series:
        for qname in ["roe", "np_margin", "gp_margin", "rev_growth", "profit_growth",
                       "pe_ratio", "pb_ratio"]:
            if qname in finance_series and len(finance_series[qname]) == len(df):
                series[qname] = finance_series[qname]

    if factor_names:
        return {k: v for k, v in series.items() if k in factor_names}
    return series


def compute_composite_score(
    df: pd.DataFrame, weights: dict[str, float] | None = None,
    finance_series: dict[str, pd.Series] | None = None,
) -> pd.Series:
    """计算多因子综合得分序列（每日一个分，用于回测信号）。

    流程：compute_factor_series → 横截面排名(单股内时序百分位) → IC加权合成。
    注意：严格横截面排名需要全市场快照，这里用单股时序百分位近似
    （该股当前因子值在自身历史中的位置），回测信号够用。

    Args:
        df: OHLCV DataFrame
        weights: 因子权重，None=默认IC权重

    Returns:
        pd.Series：每日综合因子分（0-100），越高越值得买
    """
    if len(df) < 60:
        return pd.Series(0.0, index=df.index)

    series = compute_factor_series(df, finance_series=finance_series)
    if not series:
        return pd.Series(0.0, index=df.index)

    from sequoia_x.strategy.multi_factor import _DEFAULT_FACTOR_WEIGHTS
    weights = weights or _DEFAULT_FACTOR_WEIGHTS
    valid = {k: v for k, v in weights.items() if k in series}
    if not valid:
        return pd.Series(0.0, index=df.index)

    # 带符号加权：正IC因子高分=好，负IC因子高分=差（权重为负自动反向）
    total_w = sum(abs(v) for v in valid.values())
    composite = pd.Series(0.0, index=df.index)
    for fname, w in valid.items():
        # 时序百分位：当前值在过去252日的排名位置(0-100)
        pct = series[fname].rolling(252, min_periods=20).rank(pct=True) * 100
        # 带符号权重：正权重直接加，负权重做减法
        composite += pct.fillna(50) * (w / total_w)
    return composite





def _neutralize截面(df截面: pd.DataFrame, factor_cols: list[str],
                     market_cap_map: dict, industry_map: dict) -> pd.DataFrame:
    """对截面做市值+行业中性化。

    对每个因子值，回归 log(市值) + 行业哑变量，取残差作为中性化后的因子值。
    这样消除"大市值公司普遍 ROE 低/波动小"等系统性偏差。

    Args:
        df截面: 单月截面数据（含 factor_cols + symbol）
        factor_cols: 需要中性化的因子列
        market_cap_map: {symbol: 流通市值}
        industry_map: {symbol: 行业}

    Returns:
        df截面（因子列替换为残差）
    """
    import numpy as np

    # 构建 log(市值) 和行业哑变量
    symbols = df截面["symbol"].values if "symbol" in df截面.columns else None
    if symbols is None:
        return df截面

    log_mv = np.array([np.log(market_cap_map.get(s, 0) + 1) if market_cap_map.get(s, 0) > 0 else np.nan for s in symbols])

    # 只对有市值的行做中性化
    valid_mask = ~np.isnan(log_mv)
    if valid_mask.sum() < 30:
        return df截面  # 样本太少，跳过中性化

    # 行业哑变量
    industries = [industry_map.get(s, "未知") for s in symbols]
    unique_ind = list(set(industries))
    ind_to_idx = {ind: i for i, ind in enumerate(unique_ind)}

    n = len(symbols)
    n_ind = len(unique_ind)
    # X = [截距, log_mv, 行业哑变量...]
    X = np.column_stack([
        np.ones(n),
        log_mv,
        *[np.array([1.0 if ind == ui else 0.0 for ind in industries]) for ui in unique_ind]
    ])

    for f in factor_cols:
        y = df截面[f].values
        mask = valid_mask & ~np.isnan(y.astype(float))
        if mask.sum() < 30:
            continue
        try:
            X_valid = X[mask]
            y_valid = y[mask].astype(float)
            # OLS: beta = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
            residual = y_valid - X_valid @ beta
            # 填回残差
            result_col = np.full(n, np.nan)
            result_col[mask] = residual
            df截面[f + "_neutral"] = result_col
            df截面[f] = result_col  # 替换原值
        except Exception:
            continue

    return df截面


# ══════════════════════════════════════════════════════════════════
# 因子IC评估引擎
# ══════════════════════════════════════════════════════════════════

def evaluate_factor_ic(
    engine, factor_names: list[str] | None = None,
    hold_days: int = 20, sample_size: int = 500, seed: int = 42,
    neutralize: bool = True, rolling_months: int = 0,
) -> dict:
    """评估全部因子的预测力（月度Rank IC + ICIR + 分层收益）。

    核心方法（对标专业因子评价体系）：
      - 每月初计算全市场因子截面值
      - 算因子值与未来N天收益的Spearman秩相关（Rank IC）
      - IC>0.02且ICIR>0.3为有效因子

    Args:
        engine: DataEngine
        factor_names: 待评估因子列表，None=全部
        hold_days: 持有期（收益计算窗口），默认20天
        sample_size: 采样股票数
        seed: 随机种子

    Returns:
        {
            "factors": [{name, category, desc, ic_mean, ic_std, icir,
                         win_rate, assessment, quantile_spread}, ...],
            "hold_days": 20, "sample_size": N,
            "months": [...], "ic_series": {factor: [月度IC序列]},
        }
    """
    import random

    symbols = engine.get_local_symbols()
    if sample_size and len(symbols) > sample_size:
        rng = random.Random(seed)
        symbols = rng.sample(symbols, sample_size)
    logger.info(f"因子IC评估：采样 {len(symbols)} 只，持有期 {hold_days} 天")
    cutoff_map = engine.get_ipo_cutoff_map()
    factor_set = factor_names or list(FACTOR_META.keys())

    # 加载市值/行业映射（用于中性化）
    market_cap_map: dict[str, float] = {}
    industry_map: dict[str, str] = {}
    if neutralize:
        import sqlite3 as _sq
        try:
            with _sq.connect(engine.db_path) as _conn:
                for r in _conn.execute("SELECT symbol, circ_mv FROM stock_market_cap").fetchall():
                    if r[1] and r[1] > 0:
                        market_cap_map[r[0]] = float(r[1])
                for r in _conn.execute("SELECT symbol, industry FROM stock_industry").fetchall():
                    industry_map[r[0]] = r[1]
            logger.info(f"因子中性化：市值{len(market_cap_map)}只，行业{len(industry_map)}只")
        except Exception as e:
            logger.warning(f"市值/行业加载失败，跳过中性化：{e!r}")
            neutralize = False

    # 加载质量因子的月度截面（按财报季度月份匹配）
    # stat_date 格式 YYYY-MM-DD，取 YYYY-MM 作为截面月份
    quality_factors = {"roe", "np_margin", "gp_margin", "rev_growth", "profit_growth",
                       "yoy_ni", "asset_turn", "inv_turn", "nr_turn",
                       "cfo_yield", "earnings_quality"}
    valuation_factors = {"pe_ratio", "pb_ratio"}
    # 因子名 → stock_finance 字段名映射（IC评估逐月取截面值用）
    _finance_field_map = {
        "roe": "roe", "np_margin": "np_margin", "gp_margin": "gp_margin",
        "rev_growth": "yoy_eps", "profit_growth": "yoy_pni",
        "yoy_ni": "yoy_ni", "asset_turn": "asset_turn",
        "inv_turn": "inv_turn", "nr_turn": "nr_turn",
        "cfo_yield": "cfo_to_or", "earnings_quality": "cfo_to_np",
    }
    has_quality = bool(set(factor_set) & quality_factors)
    has_valuation = bool(set(factor_set) & valuation_factors)
    finance_map: dict[str, list[tuple]] = {}  # {symbol: [(stat_date, {fields})], 排序}
    if has_quality:
        import sqlite3 as _sq
        try:
            with _sq.connect(engine.db_path) as _conn:
                _rows = _conn.execute(
                    """SELECT symbol, stat_date, roe, np_margin, gp_margin, yoy_eps, yoy_pni,
                              yoy_ni, asset_turn, inv_turn, nr_turn, cfo_to_or, cfo_to_np
                       FROM stock_finance ORDER BY symbol, stat_date"""
                ).fetchall()
            for r in _rows:
                finance_map.setdefault(r[0], []).append((r[1], {
                    "roe": r[2], "np_margin": r[3], "gp_margin": r[4],
                    "yoy_eps": r[5], "yoy_pni": r[6],
                    "yoy_ni": r[7], "asset_turn": r[8], "inv_turn": r[9], "nr_turn": r[10],
                    "cfo_to_or": r[11], "cfo_to_np": r[12],
                }))
            logger.info(f"因子IC评估：加载财报 {len(finance_map)} 只股票")
        except Exception as e:
            logger.warning(f"因子IC评估：财报加载失败：{e!r}")

    # 加载估值因子（PE/PB，来自东财快照）
    valuation_map: dict[str, dict] = {}
    if has_valuation:
        import sqlite3 as _sq
        try:
            with _sq.connect(engine.db_path) as _conn:
                for r in _conn.execute("SELECT symbol, pe, pb FROM stock_market_cap WHERE pe IS NOT NULL OR pb IS NOT NULL").fetchall():
                    valuation_map[r[0]] = {"pe_ratio": r[1], "pb_ratio": r[2]}
            logger.info(f"因子IC评估：加载估值 {len(valuation_map)} 只股票")
        except Exception as e:
            logger.warning(f"因子IC评估：估值加载失败：{e!r}")

    # 加载资金流向（main_net / main_pct / flow_super_ratio / flow_intensity）
    fund_flow_factors = {"main_net", "main_pct", "flow_super_ratio", "flow_intensity"}
    has_fund_flow = bool(set(factor_set) & fund_flow_factors)
    fund_flow_map: dict[str, list[tuple]] = {}  # {symbol: [(date, {fields})], 排序}
    if has_fund_flow:
        import sqlite3 as _sq2
        try:
            with _sq2.connect(engine.db_path) as _conn2:
                _ff_rows = _conn2.execute(
                    "SELECT symbol, date, main_net, main_pct, super_net, big_net "
                    "FROM fund_flow ORDER BY symbol, date"
                ).fetchall()
            for r in _ff_rows:
                fund_flow_map.setdefault(r[0], []).append((r[1], {
                    "main_net": r[2], "main_pct": r[3],
                    "super_net": r[4], "big_net": r[5],
                }))
            logger.info(f"因子IC评估：加载资金流向 {len(fund_flow_map)} 只股票")
        except Exception as e:
            logger.warning(f"因子IC评估：资金流向加载失败：{e!r}")

    # 加载龙虎榜因子（近30天上榜次数 + 净买入额）
    lhb_factors = {"lhb_count", "lhb_netbuy"}
    has_lhb = bool(set(factor_set) & lhb_factors)
    lhb_map: dict[str, dict] = {}
    if has_lhb:
        import sqlite3 as _sq3
        try:
            with _sq3.connect(engine.db_path) as _conn3:
                _lhb_rows = _conn3.execute(
                    "SELECT symbol, date, net_buy FROM lhb_detail "
                    "ORDER BY symbol, date"
                ).fetchall()
                for r in _lhb_rows:
                    lhb_map.setdefault(r[0], []).append((r[1], r[2] or 0))
            logger.info(f"因子IC评估：加载龙虎榜 {len(lhb_map)} 只股票")
        except Exception as e:
            logger.warning(f"因子IC评估：龙虎榜加载失败：{e!r}")

    # 加载北向资金持股历史（按股票→{date: hold_pct}，用于月度截面计算）
    north_factors = {"nb_holding_pct", "nb_inflow"}
    north_map: dict[str, dict[str, float]] = {}  # {symbol: {date_str: hold_pct}}
    if set(factor_set) & north_factors:
        import sqlite3 as _sq4
        try:
            with _sq4.connect(engine.db_path) as _conn4:
                _nb_rows = _conn4.execute(
                    "SELECT symbol, date, hold_pct FROM north_hold "
                    "WHERE hold_pct IS NOT NULL ORDER BY symbol, date"
                ).fetchall()
            for r in _nb_rows:
                north_map.setdefault(r[0], {})[str(r[1])] = float(r[2])
            logger.info(f"因子IC评估：加载北向持股 {len(north_map)} 只股票")
        except Exception as e:
            logger.warning(f"因子IC评估：北向持股加载失败（表可能不存在）：{e!r}")

    # 加载融资融券历史（杠杆资金方向，北向断供后的替代信号）
    margin_factors = {"margin_balance", "margin_netbuy", "short_ratio"}
    margin_map: dict[str, list[tuple]] = {}  # {symbol: [(date, {fields})], 排序}
    if set(factor_set) & margin_factors:
        import sqlite3 as _sq5
        try:
            with _sq5.connect(engine.db_path) as _conn5:
                _mg_rows = _conn5.execute(
                    "SELECT symbol, date, rzye, rzbuy, rqlts, rqye "
                    "FROM margin_detail ORDER BY symbol, date"
                ).fetchall()
            for r in _mg_rows:
                margin_map.setdefault(r[0], []).append((r[1], {
                    "rzye": r[2], "rzbuy": r[3], "rqlts": r[4], "rqye": r[5],
                }))
            logger.info(f"因子IC评估：加载融资融券 {len(margin_map)} 只股票")
        except Exception as e:
            logger.warning(f"因子IC评估：融资融券加载失败：{e!r}")

    # 加载基金持仓（公募重仓，来自fund_hold表，按季度截面）
    fund_hold_factors = {"fund_holding", "fund_inflow"}
    fund_hold_map: dict[str, list[tuple]] = {}  # {symbol: [(report_date, {fields})], 排序}
    if set(factor_set) & fund_hold_factors:
        import sqlite3 as _sq6
        try:
            with _sq6.connect(engine.db_path) as _conn6:
                _fh_rows = _conn6.execute(
                    "SELECT symbol, report_date, fund_count, change_pct "
                    "FROM fund_hold ORDER BY symbol, report_date"
                ).fetchall()
            for r in _fh_rows:
                fund_hold_map.setdefault(r[0], []).append((r[1], {
                    "fund_count": r[2], "change_pct": r[3],
                }))
            logger.info(f"因子IC评估：加载基金持仓 {len(fund_hold_map)} 只股票")
        except Exception as e:
            logger.warning(f"因子IC评估：基金持仓加载失败：{e!r}")

    # 加载大宗交易历史（折溢率=机构接货信号）
    block_factors = {"block_discount"}
    block_map: dict[str, list[tuple]] = {}  # {symbol: [(date, discount)], 排序}
    if set(factor_set) & block_factors:
        import sqlite3 as _sq8
        try:
            with _sq8.connect(engine.db_path) as _conn8:
                _bt_rows = _conn8.execute(
                    "SELECT symbol, date, discount, amount FROM block_trade "
                    "ORDER BY symbol, date"
                ).fetchall()
            for r in _bt_rows:
                block_map.setdefault(r[0], []).append((r[1], r[2] or 0, r[3] or 0))
            logger.info(f"因子IC评估：加载大宗交易 {len(block_map)} 只股票")
        except Exception as e:
            logger.warning(f"因子IC评估：大宗交易加载失败（表可能不存在）：{e!r}")

    # 加载股东户数历史（筹码集中度，季频）
    holder_factors = {"holder_change", "holder_count", "holder_avg_value"}
    holder_map: dict[str, list[tuple]] = {}  # {symbol: [(end_date, {fields})], 排序}
    if set(factor_set) & holder_factors:
        import sqlite3 as _sq9
        try:
            with _sq9.connect(engine.db_path) as _conn9:
                _hd_rows = _conn9.execute(
                    "SELECT symbol, end_date, holder_num, holder_change, avg_value "
                    "FROM holder_count ORDER BY symbol, end_date"
                ).fetchall()
            for r in _hd_rows:
                holder_map.setdefault(r[0], []).append((r[1], {
                    "holder_num": r[2], "holder_change": r[3], "avg_value": r[4],
                }))
            logger.info(f"因子IC评估：加载股东户数 {len(holder_map)} 只股票")
        except Exception as e:
            logger.warning(f"因子IC评估：股东户数加载失败（表可能不存在）：{e!r}")

    # 加载沪深300指数收益率序列（用于 beta_300 / rel_strength_300 的IC评估）
    index_ret_series = None
    if set(factor_set) & {"beta_300", "rel_strength_300"}:
        import sqlite3 as _sq7
        try:
            with _sq7.connect(engine.db_path) as _conn7:
                _idx_rows = _conn7.execute(
                    "SELECT date, close FROM index_daily "
                    "WHERE symbol='000300' ORDER BY date"
                ).fetchall()
            if len(_idx_rows) >= 60:
                _idx_df = pd.DataFrame(_idx_rows, columns=["date", "close"])
                _idx_df["date"] = _idx_df["date"].astype(str)
                _idx_ret = _idx_df["close"].pct_change()
                index_ret_series = pd.Series(_idx_ret.values, index=_idx_df["date"].values).dropna()
                logger.info(f"因子IC评估：加载沪深300指数 {len(index_ret_series)} 日")
        except Exception as e:
            logger.warning(f"因子IC评估：沪深300指数加载失败：{e!r}")

    # 采集每只股票的 (月份, 因子值, 未来收益)
    # 性能优化：一次性向量化算完整序列因子，再按月取截面，避免逐月重算
    def _pit_asof(seq: list[tuple], asof_date: str) -> dict | None:
        """从已排序的 [(date, fields), ...] 取 ≤ asof_date 的最新一条 fields。"""
        import bisect
        if not seq:
            return None
        dates_list = [d for d, _ in seq]
        idx = bisect.bisect_right(dates_list, asof_date) - 1
        return seq[idx][1] if idx >= 0 else None

    def _pit_recent(seq: list[tuple], asof_date: str, days: int) -> list[tuple]:
        """取 ≤ asof_date 的最近 days 个自然日内的记录。"""
        if not seq:
            return []
        import datetime as _dtm
        try:
            ref = _dtm.datetime.strptime(asof_date, "%Y-%m-%d")
            ws = (ref - _dtm.timedelta(days=days)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return []
        return [(d, f) for d, f in seq if ws <= d <= asof_date]

    records: dict[str, list[dict]] = {}  # {month: [{factor_values..., fwd_return}]}
    processed = 0

    for sym in symbols:
        try:
            df = engine.get_ohlcv(sym)
            if len(df) < hold_days + 60:
                continue
            co = cutoff_map.get(sym)
            if co and "date" in df.columns:
                df = df[df["date"].astype(str) >= co]
            df = df.reset_index(drop=True)
            # 向量化算完整序列的因子值（一次算完，按月取截面）
            # compute_factor_series 只支持量价时序因子，质量因子无时序跳过
            # 所有 point-in-time 因子（非时序，按月取截面值）需排除出 compute_factor_series
            margin_factors = {"margin_balance", "margin_netbuy", "short_ratio"}
            block_factors = {"block_discount"}
            holder_factors = {"holder_change", "holder_count", "holder_avg_value"}
            _pit_factors = (quality_factors | valuation_factors | fund_flow_factors
                            | lhb_factors | north_factors | margin_factors | fund_hold_factors
                            | block_factors | holder_factors)
            ts_factors = [f for f in factor_set if f not in _pit_factors]
            series = compute_factor_series(df, ts_factors, index_ret=index_ret_series)
            dates = df["date"].astype(str).values
            months = np.array([d[:7] for d in dates])
            close = df["close"]
            # 次日成交口径（信号当日生成→次日收盘买入持N天卖出），杜绝前视
            fwd = (close.shift(-(hold_days + 1)) / close.shift(-1) - 1).values

            # 每月取第一个交易日作为截面日
            seen_months = set()
            for i in range(len(df)):
                m = months[i]
                if m in seen_months:
                    continue
                if i + hold_days >= len(df):
                    break
                seen_months.add(m)
                fwd_ret = fwd[i]
                if np.isnan(fwd_ret):
                    continue
                row = {"symbol": sym}
                for k in factor_set:
                    if k in fund_flow_factors:
                        # as-of 取截面前最新资金流（杜绝未来函数）
                        ff = _pit_asof(fund_flow_map.get(sym, []), dates[i])
                        if ff:
                            if k == "main_net":
                                row[k] = float(ff.get("main_net", 0)) if ff.get("main_net") is not None else None
                            elif k == "main_pct":
                                row[k] = float(ff.get("main_pct", 0)) if ff.get("main_pct") is not None else None
                            elif k == "flow_super_ratio":
                                sn = ff.get("super_net")
                                bn = ff.get("big_net")
                                if sn is not None and bn is not None:
                                    tot = abs(float(sn)) + abs(float(bn)) + 1
                                    row[k] = float(sn) / tot
                                else:
                                    row[k] = None
                            elif k == "flow_intensity":
                                mn = ff.get("main_net")
                                if mn is not None:
                                    row[k] = abs(float(mn))
                                else:
                                    row[k] = None
                            else:
                                row[k] = None
                        else:
                            row[k] = None
                    elif k in quality_factors:
                        # as-of 取截面前最新季报（杜绝未来函数：用已披露的财报）
                        fin = _pit_asof(finance_map.get(sym, []), dates[i])
                        field = _finance_field_map.get(k)
                        if fin and field:
                            v = fin.get(field)
                            row[k] = float(v) if v is not None else None
                        else:
                            row[k] = None
                    elif k in valuation_factors:
                        val = valuation_map.get(sym)
                        if val and val.get(k) is not None:
                            v = float(val[k])
                            if not np.isnan(v) and abs(v) < 10000:
                                row[k] = v
                            else:
                                row[k] = None
                        else:
                            row[k] = None
                    elif k in lhb_factors:
                        # as-of 取近30天龙虎榜（杜绝未来函数），lhb_map 为 [(date, net_buy)]
                        _recent = _pit_recent(lhb_map.get(sym, []), dates[i], 30)
                        if k == "lhb_count":
                            row[k] = float(len(_recent))
                        elif k == "lhb_netbuy":
                            row[k] = float(sum(nb for _, nb in _recent))
                        else:
                            row[k] = None
                    elif k in north_factors:
                        nb_series = north_map.get(sym)
                        if nb_series:
                            asof_dates = [d for d in nb_series if d <= dates[i]]
                            if asof_dates:
                                cur_pct = nb_series[asof_dates[-1]]
                                if k == "nb_holding_pct":
                                    row[k] = cur_pct
                                elif k == "nb_inflow":
                                    idx20 = len(asof_dates) - 21
                                    old_pct = nb_series[asof_dates[idx20]] if idx20 >= 0 else None
                                    row[k] = (cur_pct - old_pct) if old_pct is not None else None
                                else:
                                    row[k] = None
                            else:
                                row[k] = None
                        else:
                            row[k] = None
                    elif k in holder_factors:
                        # as-of 取截面前最新季度股东户数（季频，杜绝未来函数）
                        hd = _pit_asof(holder_map.get(sym, []), dates[i])
                        if hd:
                            if k == "holder_change":
                                row[k] = float(hd["holder_change"]) if hd["holder_change"] is not None else None
                            elif k == "holder_count":
                                row[k] = float(hd["holder_num"]) if hd["holder_num"] is not None else None
                            elif k == "holder_avg_value":
                                row[k] = float(hd["avg_value"]) if hd["avg_value"] is not None else None
                            else:
                                row[k] = None
                        else:
                            row[k] = None
                    elif k in block_factors:
                        # as-of 取近30天大宗交易（杜绝未来函数），加权折价率
                        # block_map 存 3 元组 (date, discount, amount)，内联过滤
                        import datetime as _bdt
                        _bt_seq = block_map.get(sym, [])
                        if _bt_seq:
                            try:
                                _ref = _bdt.datetime.strptime(dates[i], "%Y-%m-%d")
                                _ws = (_ref - _bdt.timedelta(days=30)).strftime("%Y-%m-%d")
                            except (ValueError, TypeError):
                                _ws = "1900-01-01"
                            _bt_recent = [rec for rec in _bt_seq if _ws <= rec[0] <= dates[i]]
                        else:
                            _bt_recent = []
                        if _bt_recent:
                            total_amt = sum(rec[2] for rec in _bt_recent if rec[2] and rec[2] > 0)
                            if total_amt > 0:
                                row[k] = sum(rec[1] * (rec[2] if rec[2] and rec[2] > 0 else 0) for rec in _bt_recent) / total_amt
                            else:
                                row[k] = sum(rec[1] for rec in _bt_recent) / len(_bt_recent)
                        else:
                            row[k] = None
                    elif k in margin_factors:
                        # as-of 取截面前最新融资融券（杜绝未来函数）
                        mg = _pit_asof(margin_map.get(sym, []), dates[i])
                        if mg:
                            rzye = mg.get("rzye")
                            rqye = mg.get("rqye")
                            if k == "margin_balance":
                                row[k] = float(rzye) if rzye is not None else None
                            elif k == "margin_netbuy":
                                rzbuy = mg.get("rzbuy")
                                row[k] = float(rzbuy) if rzbuy is not None else None
                            elif k == "short_ratio":
                                if rqye is not None and rzye is not None and rzye > 0:
                                    row[k] = float(rqye) / (rzye + rqye)
                                else:
                                    row[k] = None
                            else:
                                row[k] = None
                        else:
                            row[k] = None
                    elif k in fund_hold_factors:
                        # as-of 取截面前最新基金持仓（按季度披露，杜绝未来函数）
                        fh = _pit_asof(fund_hold_map.get(sym, []), dates[i])
                        if fh:
                            if k == "fund_holding":
                                fc = fh.get("fund_count")
                                row[k] = float(fc) if fc is not None else None
                            elif k == "fund_inflow":
                                cp = fh.get("change_pct")
                                row[k] = float(cp) if cp is not None else None
                            else:
                                row[k] = None
                        else:
                            row[k] = None
                    elif k in series and not np.isnan(series[k].iloc[i]):
                        row[k] = float(series[k].iloc[i])
                    else:
                        row[k] = None
                row["fwd_return"] = float(fwd_ret)
                records.setdefault(m, []).append(row)
            processed += 1
        except Exception:
            continue

    logger.info(f"因子IC评估：处理 {processed}/{len(symbols)} 只，{len(records)} 个月份")

    # 计算每个因子的月度IC序列
    sorted_months = sorted(records.keys())
    # P3: 滚动窗口——只取最近N个月（0=全样本）
    if rolling_months > 0 and len(sorted_months) > rolling_months:
        cutoff_month = sorted_months[-rolling_months]
        sorted_months = [m for m in sorted_months if m >= cutoff_month]
        records = {m: records[m] for m in sorted_months if m in records}
        logger.info(f"滚动IC窗口：最近{rolling_months}个月（{sorted_months[0]}~{sorted_months[-1]}）")
    ic_series: dict[str, list[float]] = {f: [] for f in factor_set}

    for m in sorted_months:
        batch = records[m]
        df_batch = pd.DataFrame(batch)
        # 截面中性化（市值+行业）
        if neutralize and len(df_batch) >= 50:
            df_batch = _neutralize截面(df_batch, factor_set, market_cap_map, industry_map)
        for f in factor_set:
            valid = df_batch[[f, "fwd_return"]].dropna()
            if len(valid) < 20:
                ic_series[f].append(np.nan)
                continue
            ic = float(valid[f].rank().corr(valid["fwd_return"].rank()))
            ic_series[f].append(ic)

    # P6: 因子拥挤度评估（多头组合行业集中度 HHI）—— 在三态/全局权重计算前算，
    # 以便对拥挤因子施加软衰减乘子。ic_by_factor 从 ic_series 均值推导
    # （此处 factor_reports 尚未构建），industry_map 已加载。
    ic_by_factor = {
        f: float(np.mean([x for x in ic_series[f] if not np.isnan(x)]))
        for f in factor_set if any(not np.isnan(x) for x in ic_series[f])
    }
    crowding_scores = _compute_crowding(
        records, sorted_months, ic_by_factor, industry_map,
    )
    # P6修正：拥挤度阈值从绝对值改为分位数自适应——SAFE=median、MAX=p90。
    # 实测固定值(0.08/0.30)脱离分布：37因子中位数仅0.056、p90=0.103，
    # 绝对阈值导致惩罚几乎不触发。分位数随分布自适应，牛市拥挤上升自动收紧。
    dyn_crowd_safe = CROWDING_SAFE
    dyn_crowd_max = CROWDING_MAX
    if crowding_scores:
        vals = sorted(crowding_scores.values())
        med = vals[len(vals) // 2]
        # 样本充足(≥10)时用分位数；不足则回退模块常量(安全降级)
        if len(vals) >= 10:
            dyn_crowd_safe = med
            dyn_crowd_max = vals[int(len(vals) * 0.9)]
        logger.info(
            f"因子拥挤度(HHI)：{len(vals)}个因子 "
            f"min={vals[0]:.3f} median={med:.3f} max={vals[-1]:.3f} "
            f"(衰减阈值 SAFE={dyn_crowd_safe:.3f}/MAX={dyn_crowd_max:.3f})"
        )

    # ════════ P16: 因子正交化——IC 相关性聚类 + 增量 IC 权重折扣 ════════
    ic_corr = _build_ic_correlation(ic_series, sorted(factor_set))
    orth_clusters = _cluster_factors(ic_corr, ORTH_THRESHOLD) if not ic_corr.empty else {}
    orth_penalties = _orthogonal_penalty(ic_corr, orth_clusters, ic_by_factor) if orth_clusters else {}
    if orth_clusters:
        multi_clusters = {cid: m for cid, m in orth_clusters.items() if len(m) > 1}
        if multi_clusters:
            parts = []
            for cid, members in sorted(multi_clusters.items(), key=lambda x: -len(x[1])):
                base = max(members, key=lambda f: abs(ic_by_factor.get(f, 0)))
                parts.append(f"[{base}+{len(members)-1}]")
            discounted = sum(1 for v in orth_penalties.values() if v < 0.99)
            logger.info(
                f"因子正交化：{len(multi_clusters)}个高相关簇 "
                f"({' '.join(parts)})，{discounted}个因子被折扣"
            )

    # ════════ P1: 三态市场状态分组IC（bull/neutral/bear 各一套权重）════════
    # 用全市场等权月度收益判定该月市场状态：
    #   月收益 >3% → bull, <-3% → bear, 中间 → neutral
    month_returns = {}
    for m in sorted_months:
        rets = [r["fwd_return"] for r in records[m] if r["fwd_return"] is not None]
        month_returns[m] = float(np.median(rets)) if rets else 0.0

    def _state_of(m_ret: float) -> str:
        if m_ret > 0.03:
            return "bull"
        elif m_ret < -0.03:
            return "bear"
        return "neutral"

    month_states = {m: _state_of(month_returns[m]) for m in sorted_months}
    state_ic_series: dict[str, dict[str, list[float]]] = {
        s: {f: [] for f in factor_set} for s in ("bull", "neutral", "bear")
    }
    for m in sorted_months:
        state = month_states[m]
        batch = records[m]
        df_batch = pd.DataFrame(batch)
        if neutralize and len(df_batch) >= 50:
            df_batch = _neutralize截面(df_batch, factor_set, market_cap_map, industry_map)
        for f in factor_set:
            valid = df_batch[[f, "fwd_return"]].dropna()
            if len(valid) < 10:
                continue
            ic = float(valid[f].rank().corr(valid["fwd_return"].rank()))
            state_ic_series[state][f].append(ic)

    # 计算三态权重
    state_weights: dict[str, list[dict]] = {}
    for state in ("bull", "neutral", "bear"):
        state_reports = []
        for f in factor_set:
            ics = [x for x in state_ic_series[state][f] if not np.isnan(x)]
            if len(ics) < 3:
                continue
            ic_mean = float(np.mean(ics))
            ic_std = float(np.std(ics))
            icir = ic_mean / ic_std if ic_std > 0 else 0
            win_rate = float((np.array(ics) > 0).mean() * 100)
            t_stat = _t_stat(ic_mean, ic_std, len(ics))
            if _is_significant(ic_mean, icir, t_stat, len(ics)):  # 三重显著性过滤，与主表口径一致
                state_reports.append({
                    "factor_name": f,
                    "category": FACTOR_META.get(f, {}).get("category", ""),
                    "ic_mean": round(ic_mean, 4),
                    "icir": round(icir, 4),
                    "t_stat": round(t_stat, 3),
                    "win_rate": round(win_rate, 1),
                })
        total_ic = sum(abs(r["ic_mean"]) for r in state_reports)
        if total_ic > 0:
            for r in state_reports:
                # P6: 三态权重同样施加拥挤度软衰减（用同一全局 crowding：
                # 行业集中度是结构性属性，与市场态无关）
                r["weight"] = round(
                    (r["ic_mean"] / total_ic)
                    * _crowding_penalty(crowding_scores.get(r["factor_name"], 0.0),
                                          dyn_crowd_safe, dyn_crowd_max)
                    * orth_penalties.get(r["factor_name"], 1.0), 4)
            state_weights[state] = state_reports

    # 写入三态权重表
    try:
        engine.save_market_factor_weights(state_weights)
        parts = [f"{s}:{len(state_weights.get(s,[]))}" for s in ("bull", "neutral", "bear")]
        logger.info(f"三态因子权重已写入DB（{' '.join(parts)}）")
    except Exception as e:
        logger.warning(f"三态权重写入失败：{e!r}")

    # 记录各月市场状态（供回测/展示）
    for m in sorted_months:
        month_returns[m] = round(month_returns[m], 4)

    # 汇总统计
    factor_reports = []
    for f in factor_set:
        ics = [x for x in ic_series[f] if not np.isnan(x)]
        if len(ics) < 4:
            factor_reports.append({
                "name": f, **FACTOR_META.get(f, {}),
                "ic_mean": 0, "icir": 0, "win_rate": 0,
                "assessment": "样本不足", "quantile_spread": 0,
            })
            continue
        ic_mean = float(np.mean(ics))
        ic_std = float(np.std(ics))
        icir = ic_mean / ic_std if ic_std > 0 else 0
        win_rate = float((np.array(ics) > 0).mean() * 100)
        n_ics = len(ics)
        t_stat = _t_stat(ic_mean, ic_std, n_ics)

        # 分层收益：Top组-Q5组（最后一个月）
        last_batch = pd.DataFrame(records[sorted_months[-1]])
        valid = last_batch[[f, "fwd_return"]].dropna()
        q_spread = 0.0
        if len(valid) >= 50:
            valid["q"] = pd.qcut(valid[f], 5, labels=False, duplicates="drop")
            if valid["q"].nunique() >= 2:
                top_ret = valid[valid["q"] == valid["q"].max()]["fwd_return"].mean()
                bot_ret = valid[valid["q"] == valid["q"].min()]["fwd_return"].mean()
                q_spread = round(float((top_ret - bot_ret) * 100), 2)

        factor_reports.append({
            "name": f,
            "category": FACTOR_META.get(f, {}).get("category", ""),
            "desc": FACTOR_META.get(f, {}).get("desc", f),
            "ic_mean": round(ic_mean, 4),
            "ic_std": round(ic_std, 4),
            "icir": round(icir, 4),
            "t_stat": round(t_stat, 3),
            "n_samples": n_ics,
            "win_rate": round(win_rate, 1),
            "assessment": _factor_assessment(ic_mean, icir, win_rate),
            "quantile_spread": q_spread,
        })

    # 按IC绝对值降序
    factor_reports.sort(key=lambda x: abs(x["ic_mean"]), reverse=True)

    # 写回DB：带符号IC权重（正IC→正权重，负IC→负权重）+ ICIR筛选门槛
    #
    # 专业做法：
    #   1. 只保留统计显著因子（|IC|>0.03 且 |ICIR|>0.5 且 |t|≥2，样本≥6）
    #   2. 权重 = IC_signed / sum(|IC_signed|)，保留IC方向
    #   3. 负IC因子获得负权重，在综合分中做减法（相当于反向指标的正确使用）
    #   4. 综合分 = Σ(因子排名 × signed_weight)，正因子贡献高分，负因子拖累
    try:
        effective = [f for f in factor_reports
                     if _is_significant(f["ic_mean"], f.get("icir", 0),
                                        f.get("t_stat", 0), f.get("n_samples", 0))]

        # P4: 因子去重——剔除同义因子，只保留每组IC最强的
        # 定义同义因子组（因子值高度相关，来自截面相关性分析）
        _SYNONYM_GROUPS = [
            # 动量/反转互为镜像，只保留IC绝对值最大的一个
            {"mom_5", "rev_5"},
            {"mom_10", "rev_10"},
            {"mom_20", "rev_20"},
            # 流动性三重计数
            {"turnover", "liq_rank"},
            # 换手率重复
            {"turn_surge", "vol_surge"},
            {"turn_ma5", "turn_ratio"},
            # 高位距离/超卖镜像
            {"high_dist", "oversold"},
            # 波动率高度相关
            {"atr_pct", "vol_20"},
            # 营运效率周转率高度相关
            {"asset_turn", "inv_turn", "nr_turn"},
        ]
        removed = set()
        for group in _SYNONYM_GROUPS:
            in_effective = [f for f in effective if f["name"] in group]
            if len(in_effective) > 1:
                # 保留 |IC| 最大的
                best = max(in_effective, key=lambda x: abs(x["ic_mean"]))
                for f in in_effective:
                    if f["name"] != best["name"]:
                        removed.add(f["name"])
        if removed:
            effective = [f for f in effective if f["name"] not in removed]
            logger.info(f"因子去重：剔除{len(removed)}个同义因子({', '.join(sorted(removed))})")

        total_ic = sum(abs(f["ic_mean"]) for f in effective)
        # 先备份 ML 因子权重（全量清零会抹掉 ml_factor 月度训练写入的 ml_score）
        import sqlite3 as _sq3
        _ml_backup = None
        with _sq3.connect(engine.db_path, isolation_level=None) as _conn:
            _ml_row = _conn.execute(
                "SELECT weight, ic_mean, icir, win_rate, updated_at "
                "FROM factor_weights WHERE factor_name='ml_score'"
            ).fetchone()
            if _ml_row and _ml_row[0] and _ml_row[0] != 0:
                _ml_backup = _ml_row
            # 清零所有因子的旧权重（无条件执行，防 total_ic==0 时 stale 权重残留）
            _conn.execute("UPDATE factor_weights SET weight=0")
        if total_ic > 0:
            # 写入新的带符号权重
            weights = [{
                "factor_name": f["name"],
                "category": f.get("category", ""),
                "ic_mean": f["ic_mean"],
                "icir": f.get("icir", 0),
                "t_stat": f.get("t_stat", 0),
                "win_rate": f.get("win_rate", 0),
                "crowding": crowding_scores.get(f["name"], 0.0),
                # P6: 权重 = (IC/|ΣIC|) × 拥挤度软衰减 × P16 正交化折扣
                # 乘子在归一化分母后施加，正确穿透 multi_factor 的 Σ|w| 再归一化。
                "weight": round(
                    (f["ic_mean"] / total_ic)
                    * _crowding_penalty(crowding_scores.get(f["name"], 0.0),
                                          dyn_crowd_safe, dyn_crowd_max)
                    * orth_penalties.get(f["name"], 1.0), 4),
            } for f in effective]
            engine.save_factor_weights(weights)
            pos_cnt = sum(1 for w in weights if w["weight"] > 0)
            neg_cnt = sum(1 for w in weights if w["weight"] < 0)
            logger.info(f"因子权重已刷新写入DB：{len(weights)}个有效因子（{pos_cnt}正+{neg_cnt}负）")
        else:
            logger.warning("无因子通过显著性过滤(0.03/0.5/2.0)，权重已全量清零，检查数据/窗口")
        # 恢复 ML 因子权重（防止全量清零泄漏 ml_factor 月度训练结果）
        if _ml_backup:
            with _sq3.connect(engine.db_path, isolation_level=None) as _conn:
                _conn.execute(
                    "INSERT OR REPLACE INTO factor_weights "
                    "(factor_name, category, ic_mean, icir, win_rate, weight, updated_at) "
                    "VALUES ('ml_score','ML因子',?,?,?,?,?)",
                    (_ml_backup[1], _ml_backup[2], _ml_backup[3], _ml_backup[0], _ml_backup[4]),
                )
            logger.info(f"ML因子权重已恢复：weight={_ml_backup[0]:.4f}（不被IC刷新清零）")
    except Exception as e:
        logger.warning(f"因子权重写DB失败（不影响评估结果）：{e!r}")

    return {
        "factors": factor_reports,
        "hold_days": hold_days,
        "sample_size": processed,
        "months": sorted_months,
        "ic_series": {f: [round(x, 4) if not np.isnan(x) else None for x in ic_series[f]]
                      for f in factor_set},
        "market_state_weights": {s: {w["factor_name"]: w["weight"] for w in ws}
                                 for s, ws in state_weights.items()},
        "month_states": month_states,
        "crowding": crowding_scores,
        "ic_corr": ic_corr,
    }


def _t_stat(ic_mean: float, ic_std: float, n: int) -> float:
    """月度 Rank IC 的 t 统计量 = ic_mean / (ic_std / sqrt(n))。

    衡量 IC 是否统计显著区别于 0（|t|≥2 ≈ 95% 置信）。样本不足时返回 0
    （视为不显著），避免小样本噪声被当作有效信号。
    """
    if n < MIN_IC_SAMPLES or ic_std <= 0:
        return 0.0
    return float(ic_mean / (ic_std / math.sqrt(n)))


def _is_significant(ic_mean: float, icir: float, t_stat: float, n: int) -> bool:
    """三重显著性过滤：IC/ICIR/t-stat 同时达标 且 样本充足。

    alpha 泄漏修复：旧门槛仅 |IC|>0.015 & ICIR>0.3 过低，让一堆 IC≈0.03 的
    噪声因子获得小权重稀释强信号。新门槛三者全过才进权重。
    """
    if n < MIN_IC_SAMPLES:
        return False
    return (abs(ic_mean) > MIN_IC_ABS
            and abs(icir) > MIN_ICIR_ABS
            and abs(t_stat) >= MIN_T_STAT)


def _factor_assessment(ic_mean: float, icir: float, win_rate: float) -> str:
    """因子有效性评级。"""
    if ic_mean >= 0.03 and icir >= 0.5:
        return "强有效因子"
    if ic_mean >= 0.02 and win_rate >= 55:
        return "有效因子"
    if abs(ic_mean) >= 0.02:
        return "弱有效，方向" + ("正向" if ic_mean > 0 else "负向")
    return "无效因子"


def _industry_hhi(symbols, industry_map: dict) -> float:
    """计算一组股票的行业集中度（Herfindahl-Hirschman Index）。

    HHI = Σ(industry_share²)，范围 (0,1]：全部同行业=1.0，完全均匀分散≈1/行业数。
    衡量因子多头组合是否过度集中于少数行业（拥挤度代理，反转风险信号）。

    Args:
        symbols: 多头组合的股票代码列表
        industry_map: {symbol: 行业名} 映射

    Returns:
        HHI 值 (0,1]；无行业数据或空集时返回 0（视为不拥挤，不惩罚）。
    """
    if not symbols or not industry_map:
        return 0.0
    counts: dict[str, int] = {}
    n = 0
    for sym in symbols:
        ind = industry_map.get(sym)
        if not ind:
            continue
        counts[ind] = counts.get(ind, 0) + 1
        n += 1
    if n == 0:
        return 0.0
    return sum((c / n) ** 2 for c in counts.values())


def _compute_crowding(
    records: dict, sorted_months: list, ic_by_factor: dict, industry_map: dict,
) -> dict:
    """计算各因子的多头组合行业集中度（HHI），取近 N 月均值降噪。

    对每个因子按 IC 符号取多头侧（IC≥0→top、IC<0→bottom，修正旧 nlargest
    不分方向导致负 IC 因子取错侧的缺陷），用原始（未中性化）因子值——中性化
    会扭曲真实集中度语义。取最近 CROWDING_LOOKBACK_MONTHS 个月 HHI 均值。

    Args:
        records: {month: [{symbol, factor..., fwd_return}, ...]}
        sorted_months: 升序月份列表
        ic_by_factor: {factor_name: ic_mean}（决定多头侧方向）
        industry_map: {symbol: 行业名}

    Returns:
        {factor_name: avg_hhi}，仅含有足够数据的因子。
    """
    if not sorted_months or not industry_map:
        return {}
    lookback = min(CROWDING_LOOKBACK_MONTHS, len(sorted_months))
    months = sorted_months[-lookback:]
    crowding: dict[str, float] = {}
    for f, ic in ic_by_factor.items():
        hhis = []
        for m in months:
            batch = records.get(m, [])
            if len(batch) < 50 or f not in batch[0]:
                continue
            df = pd.DataFrame(batch)
            valid = df[[f, "symbol"]].dropna()
            if len(valid) < 50:
                continue
            k = max(len(valid) // int(1 / CROWDING_TOP_PCT), 10)  # top20%
            # 多头侧：IC≥0 取 top、IC<0 取 bottom（负 IC 因子多头是低值侧）
            if ic >= 0:
                long_syms = valid.nlargest(k, f)["symbol"].tolist()
            else:
                long_syms = valid.nsmallest(k, f)["symbol"].tolist()
            hhi = _industry_hhi(long_syms, industry_map)
            if hhi > 0:
                hhis.append(hhi)
        if hhis:
            crowding[f] = round(sum(hhis) / len(hhis), 4)
    return crowding


def _crowding_penalty(crowding: float, safe: float = CROWDING_SAFE,
                        max_pct: float = CROWDING_MAX) -> float:
    """拥挤度 → 软连续衰减乘子。

    线性映射：c≤safe→1.0（不衰减）、c≥max_pct→FLOOR（保留下限）、中间线性递减。
    重度拥挤也不归零，保留分散价值（与 oos_decay 半保留下限哲学一致）。

    Args:
        crowding: HHI 拥挤度值（通常 0~1）。
        safe: 安全线阈值，≤此值不衰减（默认模块常量，生产用动态分位数）。
        max_pct: 重度拥挤线，≥此值衰减到 FLOOR。

    Returns:
        衰减乘子 [CROWDING_FLOOR, 1.0]。
    """
    if crowding <= safe:
        return 1.0
    if crowding >= max_pct:
        return CROWDING_FLOOR
    # 线性：1.0 在 safe，CROWDING_FLOOR 在 max_pct
    frac = (crowding - safe) / (max_pct - safe) if max_pct > safe else 1.0
    return round(1.0 - (1.0 - CROWDING_FLOOR) * frac, 4)


def _ml_stability_penalty(recent_ic_mean: float) -> float:
    """ML 近期 IC 均值 → 连续软惩罚乘子。

    线性映射：ic≥ML_IC_FULL→1.0（满权重）、ic≤ML_IC_WEAK→ML_PENALTY_FLOOR
    （保留下限）、中间线性递增。衰退期 ML 注入噪声时自动降权，
    强周期满权重保留 alpha。结构与 _crowding_penalty 镜像（方向相反）。

    Args:
        recent_ic_mean: ML 近期 n 月样本外 IC 均值。

    Returns:
        权重乘子 [ML_PENALTY_FLOOR, 1.0]。
    """
    if recent_ic_mean >= ML_IC_FULL:
        return 1.0
    if recent_ic_mean <= ML_IC_WEAK:
        return ML_PENALTY_FLOOR
    # 线性：ML_PENALTY_FLOOR 在 WEAK，1.0 在 FULL
    frac = (recent_ic_mean - ML_IC_WEAK) / (ML_IC_FULL - ML_IC_WEAK)
    return round(ML_PENALTY_FLOOR + (1.0 - ML_PENALTY_FLOOR) * frac, 4)


def _recent_ml_ic_mean(db_path: str, as_of_date: str | None = None, n: int = 6) -> float | None:
    """读 ml_scores 的 run_date 级 IC 均值，取最近 n 个的均值。

    Args:
        db_path: 数据库路径。
        as_of_date: PIT 截止日期（回测）。None（实盘）取全部最近 n 个 run_date。
        n: 取最近 n 个月。可用 run_date < 3 时返回 None（样本不足不惩罚）。

    Returns:
        近期 IC 均值；无数据或样本不足返回 None（调用方按满权重处理）。
    """
    import sqlite3
    try:
        with sqlite3.connect(db_path) as conn:
            if as_of_date:
                rows = conn.execute(
                    "SELECT ic_mean FROM ml_scores "
                    "WHERE run_date <= ? AND ic_mean IS NOT NULL "
                    "GROUP BY run_date ORDER BY run_date DESC LIMIT ?",
                    (as_of_date, n),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT ic_mean FROM ml_scores "
                    "WHERE ic_mean IS NOT NULL "
                    "GROUP BY run_date ORDER BY run_date DESC LIMIT ?",
                    (n,),
                ).fetchall()
    except Exception:
        return None
    if len(rows) < 3:
        return None
    return round(sum(r[0] for r in rows) / len(rows), 4)


# ════════════════════════════════════════════════════════════════════════
# P16: 因子正交化——IC 相关性矩阵 + 层次聚类 + 增量 IC 折扣
# ════════════════════════════════════════════════════════════════════════

def _build_ic_correlation(
    ic_series: dict[str, list[float]],
    factor_names: list[str],
) -> pd.DataFrame:
    """从逐月 IC 序列构建因子间 IC 相关性矩阵。

    IC 相关性衡量"两个因子是否在同一时段同时有效/失效"——
    比截面值相关性更能反映信号冗余（同一信号的不同代理）。

    Returns:
        factor × factor 的 Pearson 相关矩阵。
    """
    data = {}
    for f in factor_names:
        s = ic_series.get(f, [])
        data[f] = pd.Series(s, dtype=float).dropna()
    # 对齐长度（不同因子可能因 NaN 导致长度不同）
    df = pd.DataFrame(data)
    if df.empty or len(df) < 3:
        return pd.DataFrame()
    return df.corr()


def _cluster_factors(
    ic_corr: pd.DataFrame,
    threshold: float = ORTH_THRESHOLD,
) -> dict[int, list[str]]:
    """层次聚类自动识别高相关因子簇。

    距离 = 1 - |corr|，用 scipy linkage + fcluster 分组。
    负相关因子（方向相反的独立信号）距离大、不会被错误合并。

    Returns:
        {cluster_id: [factor_names]}，含单因子簇（独立因子）。
    """
    if ic_corr.empty:
        return {}
    factors = list(ic_corr.columns)
    n = len(factors)
    if n < 2:
        return {0: factors}

    dist = np.array(1.0 - ic_corr.abs().values, dtype=float)
    np.fill_diagonal(dist, 0.0)
    # 对称化 + 归零负值（距离非负）
    dist = np.clip(dist, 0, 2)

    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    except Exception:
        # 回退：贪心分组
        labels = _greedy_cluster(ic_corr, threshold, factors)

    clusters: dict[int, list[str]] = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(factors[i])
    return clusters


def _greedy_cluster(
    ic_corr: pd.DataFrame,
    threshold: float,
    factors: list[str],
) -> list[int]:
    """scipy 不可用时的贪心分组回退。"""
    parent = list(range(len(factors)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            if abs(ic_corr.iloc[i, j]) > threshold:
                union(i, j)
    return [find(i) for i in range(len(factors))]


def _orthogonal_penalty(
    ic_corr: pd.DataFrame,
    clusters: dict[int, list[str]],
    ic_means: dict[str, float],
) -> dict[str, float]:
    """计算每个因子的正交化权重折扣乘子。

    每簇内选 |IC| 最大的为基准（乘子=1.0）；
    非基准因子按 (1 - corr²_with_base) 折扣——
    这是增量信息系数的标准公式（partial IC after removing base）。

    corr=0.9 → 折扣 0.44；corr=0.6 → 折扣 0.80；corr=0.3 → 折扣 0.95。

    Returns:
        {factor: multiplier [0, 1]}。
    """
    penalties: dict[str, float] = {}
    for cid, members in clusters.items():
        if len(members) <= 1:
            for f in members:
                penalties[f] = 1.0
            continue
        # 选 |IC| 最大者为基准
        base = max(members, key=lambda f: abs(ic_means.get(f, 0)))
        penalties[base] = 1.0
        for f in members:
            if f == base:
                continue
            r = 0.0
            if f in ic_corr.index and base in ic_corr.columns:
                r = abs(float(ic_corr.loc[f, base]))
            penalties[f] = round((1.0 - r ** 2) ** 0.5, 4)
    return penalties
