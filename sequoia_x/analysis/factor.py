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

import numpy as np
import pandas as pd

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


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
    # ── 量价(5) ──
    "ma_cross":    {"category": "量价", "desc": "均线金叉信号"},
    "vol_surge":   {"category": "量价", "desc": "放量倍数"},
    "vol_shrink_p":{"category": "量价", "desc": "缩量程度", "reverse": True},
    "flag_tight":  {"category": "量价", "desc": "旗形收敛度"},
    "vp_divergence":{"category": "量价", "desc": "量价共振度"},
    # ── 质量(5) ──
    "roe":         {"category": "质量", "desc": "净资产收益率"},
    "np_margin":   {"category": "质量", "desc": "净利率"},
    "gp_margin":   {"category": "质量", "desc": "毛利率"},
    "rev_growth":  {"category": "质量", "desc": "营收增速"},
    "profit_growth":{"category": "质量", "desc": "利润增速"},
}


def compute_factors(df: pd.DataFrame, finance: dict | None = None) -> dict[str, float]:
    """计算单只股票的全部因子值（向量化，基于完整K线序列）。

    Args:
        df: OHLCV DataFrame（按日期升序），需含 open/high/low/close/volume/turnover
        finance: 财务数据 dict（roe/np_margin等），可选

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
    factors["mom_accel"] = recent_mom - prior_mom if prior_mom is not np.nan else np.nan

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
    factors["vol_shrink"] = float((vol_prior - vol_recent) / vol_prior * 100) if vol_prior else 0
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
        range_10 = (high.tail(10).max() - low.tail(10).min()) / low.tail(10).min()
        range_40 = (high.tail(40).max() - low.tail(40).min()) / low.tail(40).min()
        factors["flag_tight"] = -float(range_10 / range_40) if range_40 else np.nan
    else:
        factors["flag_tight"] = np.nan
    # 量价共振：涨+放量 = 正，跌+放量=负（简化）
    chg = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
    vr = volume.iloc[-1] / vol_ma20.iloc[-1] if pd.notna(vol_ma20.iloc[-1]) and vol_ma20.iloc[-1] else 1
    factors["vp_divergence"] = float(chg * vr * 100)

    # ════════ 质量(5) ════════
    if finance:
        factors["roe"] = float(finance.get("roe", 0))
        factors["np_margin"] = float(finance.get("np_margin", 0))
        factors["gp_margin"] = float(finance.get("gp_margin", 0))
        factors["rev_growth"] = float(finance.get("yoy_ni", 0))  # 营收增速用yoy_ni近似
        factors["profit_growth"] = float(finance.get("yoy_ni", 0))
    else:
        for k in ["roe", "np_margin", "gp_margin", "rev_growth", "profit_growth"]:
            factors[k] = np.nan

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


def compute_factor_series(df: pd.DataFrame, factor_names: list[str] | None = None) -> dict[str, pd.Series]:
    """向量化计算完整序列的因子值（性能优化核心）。

    一次性算出每个因子在每个交易日的值，下游按月取截面。
    相比逐月重算 compute_factors，性能提升50倍以上。

    Args:
        df: OHLCV DataFrame（按日期升序）
        factor_names: 需要计算的因子列表，None=全部非质量因子（质量因子无时序）

    Returns:
        {因子名: pd.Series（与df等长，每日因子值）}
    """
    if len(df) < 20:
        return {}
    close = df["close"]
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

    # 量价
    series["ma_cross"] = ((ma5.shift(1) < ma20.shift(1)) & (ma5 > ma20)).astype(float)
    series["vol_surge"] = volume / vol_ma20.replace(0, np.nan)
    series["vol_shrink_p"] = -volume.rolling(3).mean() / vol_ma20.replace(0, np.nan)
    range_10 = (high.rolling(10).max() - low.rolling(10).min()) / low.rolling(10).min().replace(0, np.nan)
    range_40 = (high.rolling(40).max() - low.rolling(40).min()) / low.rolling(40).min().replace(0, np.nan)
    series["flag_tight"] = -range_10 / range_40.replace(0, np.nan)
    series["vp_divergence"] = rets * (volume / vol_ma20.replace(0, np.nan)) * 100

    # 质量因子无时序，IC评估跳过
    if factor_names:
        return {k: v for k, v in series.items() if k in factor_names}
    return series


def compute_composite_score(
    df: pd.DataFrame, weights: dict[str, float] | None = None,
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

    series = compute_factor_series(df)
    if not series:
        return pd.Series(0.0, index=df.index)

    from sequoia_x.strategy.multi_factor import _DEFAULT_FACTOR_WEIGHTS
    weights = weights or _DEFAULT_FACTOR_WEIGHTS
    valid = {k: v for k, v in weights.items() if k in series}
    if not valid:
        return pd.Series(0.0, index=df.index)

    # 单股时序百分位排名（rolling rank近似横截面）
    total_w = sum(abs(v) for v in valid.values())
    composite = pd.Series(0.0, index=df.index)
    for fname, w in valid.items():
        # 时序百分位：当前值在过去252日的排名位置(0-1)
        pct = series[fname].rolling(252, min_periods=20).rank(pct=True) * 100
        composite += pct.fillna(50) * (abs(w) / total_w)
    return composite




# ══════════════════════════════════════════════════════════════════
# 因子IC评估引擎
# ══════════════════════════════════════════════════════════════════

def evaluate_factor_ic(
    engine, factor_names: list[str] | None = None,
    hold_days: int = 20, sample_size: int = 500, seed: int = 42,
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

    # 采集每只股票的 (月份, 因子值, 未来收益)
    # 性能优化：一次性向量化算完整序列因子，再按月取截面，避免逐月重算
    records: dict[str, list[dict]] = {}  # {month: [{factor_values..., fwd_return}]}
    factor_set = factor_names or list(FACTOR_META.keys())
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
            ts_factors = [f for f in factor_set if f not in
                          ("roe","np_margin","gp_margin","rev_growth","profit_growth")]
            series = compute_factor_series(df, ts_factors)
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
                row = {k: (float(series[k].iloc[i])
                         if k in series and not np.isnan(series[k].iloc[i]) else None)
                       for k in factor_set}
                row["fwd_return"] = float(fwd_ret)
                records.setdefault(m, []).append(row)
            processed += 1
        except Exception:
            continue

    logger.info(f"因子IC评估：处理 {processed}/{len(symbols)} 只，{len(records)} 个月份")

    # 计算每个因子的月度IC序列
    sorted_months = sorted(records.keys())
    ic_series: dict[str, list[float]] = {f: [] for f in factor_set}

    for m in sorted_months:
        batch = records[m]
        df_batch = pd.DataFrame(batch)
        for f in factor_set:
            valid = df_batch[[f, "fwd_return"]].dropna()
            if len(valid) < 20:
                ic_series[f].append(np.nan)
                continue
            ic = float(valid[f].rank().corr(valid["fwd_return"].rank()))
            ic_series[f].append(ic)

    # 汇总统计
    factor_reports = []
    for f in factor_set:
        ics = [x for x in ic_series[f] if not np.isnan(x)]
        if len(ics) < 6:
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
            "win_rate": round(win_rate, 1),
            "assessment": _factor_assessment(ic_mean, icir, win_rate),
            "quantile_spread": q_spread,
        })

    # 按IC绝对值降序
    factor_reports.sort(key=lambda x: abs(x["ic_mean"]), reverse=True)

    # 写回DB：有效因子(|IC|>0.015)按IC归一化为权重，自动刷新多因子策略
    try:
        effective = [f for f in factor_reports if abs(f["ic_mean"]) > 0.015]
        total_ic = sum(abs(f["ic_mean"]) for f in effective)
        if total_ic > 0:
            weights = [{
                "factor_name": f["name"],
                "category": f.get("category", ""),
                "ic_mean": f["ic_mean"],
                "icir": f.get("icir", 0),
                "win_rate": f.get("win_rate", 0),
                "weight": round(abs(f["ic_mean"]) / total_ic, 4),
            } for f in effective]
            engine.save_factor_weights(weights)
            logger.info(f"因子权重已刷新写入DB：{len(weights)}个有效因子")
    except Exception as e:
        logger.warning(f"因子权重写DB失败（不影响评估结果）：{e!r}")

    return {
        "factors": factor_reports,
        "hold_days": hold_days,
        "sample_size": processed,
        "months": sorted_months,
        "ic_series": {f: [round(x, 4) if not np.isnan(x) else None for x in ic_series[f]]
                      for f in factor_set},
    }


def _factor_assessment(ic_mean: float, icir: float, win_rate: float) -> str:
    """因子有效性评级。"""
    if ic_mean >= 0.03 and icir >= 0.5:
        return "强有效因子"
    if ic_mean >= 0.02 and win_rate >= 55:
        return "有效因子"
    if abs(ic_mean) >= 0.02:
        return "弱有效，方向" + ("正向" if ic_mean > 0 else "负向")
    return "无效因子"
