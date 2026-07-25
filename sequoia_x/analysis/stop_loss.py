"""ATR/波动率自适应止损（共享工具）。

P5 之前，回测的 `_calc_atr_stop`（2.5×ATR、8-15% 封顶）与实盘的入场/持仓
止损是两套独立逻辑：入场止损来自 `stock_analysis` 的 `price-2*ATR` 且无封顶
（散乱到 -4%~-20%），持仓期间从不按新 ATR 收紧。本模块把 ATR 止损提取为
单一数据源，供回测、入场、持仓三处复用，实现"一套逻辑两边用"。

约束：
  - ATR 用 True Range（max(H-L, |H-PrevC|, |L-PrevC|)）的窗口均值。
  - 止损幅度 clamp 到 [min_pct, max_pct]（默认 8%~15%），防假止损与过宽止损。
  - 持仓期间只收紧不放宽（单向原则）：波动率放大维持原止损，收敛时上移。
"""

from __future__ import annotations

import pandas as pd

# 默认参数：2.5×ATR，封顶 8%~15%（回测已验证、用户确认）
DEFAULT_ATR_MULT = 2.5
DEFAULT_MIN_PCT = 0.08
DEFAULT_MAX_PCT = 0.15
DEFAULT_LOOKBACK = 20        # ATR 窗口（与回测一致）
DEFAULT_FALLBACK_PCT = 0.12  # 数据不足时的默认止损 12%


def calc_atr_stop(
    df_ohlcv: pd.DataFrame | None,
    entry_price: float,
    *,
    atr_mult: float = DEFAULT_ATR_MULT,
    min_pct: float = DEFAULT_MIN_PCT,
    max_pct: float = DEFAULT_MAX_PCT,
    as_of_date: str | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    fallback_pct: float = DEFAULT_FALLBACK_PCT,
) -> float:
    """根据股票自身波动率（ATR）计算止损价。

    Args:
        df_ohlcv: 含 date/high/low/close 列的 OHLCV DataFrame（可全历史，
            内部按 as_of_date 取末尾 lookback 根做 True Range）。None 或空
            时回退到 entry_price*(1-fallback_pct)。
        entry_price: 入场价（止损基准）。
        atr_mult: ATR 乘数（2.5=2.5 倍 ATR，适配 A 股波动）。
        min_pct: 最小止损幅度（A 股日内波动常 3-5%，过窄易假止损）。
        max_pct: 最大止损幅度（过宽等于没止损）。
        as_of_date: 仅取 ≤ 该日期的 K 线（回测/入场时点一致性，防未来函数）。
        lookback: True Range 窗口根数（默认 20，与回测一致）。
        fallback_pct: 数据不足 / ATR<=0 时的回退止损幅度。

    Returns:
        止损价（entry_price 的下方）。注意：不四舍五入，保持与原回测一致，
        调用方按需自行 round。
    """
    if not entry_price or entry_price <= 0:
        return 0.0
    fallback = entry_price * (1 - fallback_pct)
    if df_ohlcv is None or len(df_ohlcv) == 0:
        return fallback

    df = df_ohlcv
    if as_of_date is not None:
        df = df[df["date"].astype(str) <= as_of_date]
    if len(df) < lookback + 1:
        return fallback

    high = df["high"].astype(float).iloc[-lookback:]
    low = df["low"].astype(float).iloc[-lookback:]
    close = df["close"].astype(float).iloc[-lookback:]
    prev_close = float(df["close"].astype(float).iloc[-(lookback + 1)])

    # True Range = max(H-L, |H-PrevC|, |L-PrevC|)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.mean())
    if atr <= 0:
        return fallback

    stop_pct = min(max(atr * atr_mult / entry_price, min_pct), max_pct)
    return entry_price * (1 - stop_pct)


def resolve_entry_stop(raw_stop: float | None, entry_price: float, atr_stop: float) -> float:
    """选择入场止损价：把 decision 散乱值统一为 ATR 自适应口径。

    规则（与 P5 计划一致）：
      - raw_stop 为空/≤0（未给止损）→ 用 ATR 值；
      - 止损距离 > max_pct（过宽，等于没止损）→ 用 ATR 值；
      - 否则保留 decision 给的止损（可能比 ATR 更紧，不强制放宽）。

    Args:
        raw_stop: decision/stock_analysis 传入的原始止损价（可能散乱或 0）。
        entry_price: 入场价。
        atr_stop: calc_atr_stop 计算的 ATR 止损价。

    Returns:
        最终写入持仓的入场止损价。
    """
    if raw_stop is None or raw_stop <= 0:
        return atr_stop
    if entry_price <= 0:
        return atr_stop
    distance = (entry_price - raw_stop) / entry_price  # >0 表示止损在下方
    if distance > DEFAULT_MAX_PCT:
        return atr_stop
    return raw_stop
