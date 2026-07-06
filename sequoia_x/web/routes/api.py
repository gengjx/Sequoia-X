"""JSON API 路由。"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from sequoia_x.web.services import STRATEGY_META, TaskStatus

router = APIRouter()


# ---------------------------------------------------------------------------
# Strategy endpoints
# ---------------------------------------------------------------------------

@router.get("/strategies")
async def list_strategies(request: Request):
    services = request.app.state.services
    return {"strategies": services.list_strategies()}


@router.post("/strategies/{key}/run")
async def run_strategy(key: str, request: Request):
    services = request.app.state.services
    if key not in STRATEGY_META:
        raise HTTPException(404, f"未知策略: {key}")
    task_id = services.run_strategy_async(key)
    return {"task_id": task_id, "status": TaskStatus.PENDING}


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    services = request.app.state.services
    record = services.get_task_status(task_id)
    if record is None:
        raise HTTPException(404, f"任务不存在: {task_id}")
    return {
        "task_id": record.task_id,
        "strategy_key": record.strategy_key,
        "status": record.status,
        "started_at": record.started_at.isoformat() if record.started_at else None,
        "finished_at": record.finished_at.isoformat() if record.finished_at else None,
        "result_count": len(record.results),
        "results": record.results,
        "error": record.error,
    }


@router.get("/strategies/{key}/results")
async def get_strategy_results(key: str, request: Request):
    services = request.app.state.services
    if key not in STRATEGY_META:
        raise HTTPException(404, f"未知策略: {key}")
    results = services.get_cached_results(key)
    return {"key": key, "results": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Stock endpoints
# ---------------------------------------------------------------------------

@router.get("/stocks/search")
async def search_stocks(q: str, request: Request):
    services = request.app.state.services
    symbols = services.search_symbols(q)
    return {"symbols": symbols}


@router.get("/stocks/{symbol}")
async def get_stock(symbol: str, request: Request, days: int = 120):
    services = request.app.state.services
    data = services.get_stock_data(symbol, days)
    return {"symbol": symbol, "data": data}


@router.get("/stocks/{symbol}/summary")
async def get_stock_summary(symbol: str, request: Request):
    services = request.app.state.services
    summary = services.get_stock_summary(symbol)
    if summary is None:
        raise HTTPException(404, f"未找到股票: {symbol}")
    return summary
@router.get("/stocks/{symbol}/analysis")
async def analyze_stock(symbol: str, request: Request):
    """个股深度分析：技术面+相对强度+市场环境+策略命中+基本面+资金面。

    分析在后台线程执行（asyncio.to_thread），不阻塞事件循环，
    分析期间其他页面/请求正常响应。
    """
    import asyncio
    services = request.app.state.services
    return await asyncio.to_thread(services.analyze_stock, symbol)


@router.post("/portfolio/analyze")
async def analyze_portfolio(body: PortfolioRequest, request: Request):
    """批量分析多只股票，返回组合体检报告（持仓扫描）。"""
    import asyncio
    services = request.app.state.services
    return await asyncio.to_thread(services.analyze_portfolio, body.symbols)


@router.post("/decision/generate")
async def generate_decision(body: DecisionRequest, request: Request):
    """交易决策中枢：多策略融合生成买卖清单（10分钟缓存）。"""
    import asyncio
    services = request.app.state.services
    return await asyncio.to_thread(
        services.generate_decision,
        body.strategy_keys, body.capital, body.min_score,
        body.exclude_markets, body.exclude_st, body.max_candidates,
        body.include_auction,
    )


@router.get("/decision/backtest")
async def backtest_combos(request: Request):
    """组合历史回测对比（向量化，秒级返回）。"""
    import asyncio
    services = request.app.state.services
    return await asyncio.to_thread(services.backtest_combos)


@router.get("/decision/backtest-resonance")
async def backtest_resonance(request: Request):
    """共振度分档回测（1/2/3+共振的收益对比）。"""
    import asyncio
    services = request.app.state.services
    return await asyncio.to_thread(services.backtest_resonance)


@router.get("/strategy/evaluate")
async def strategy_evaluate(request: Request):
    """策略评估：全维度评分卡 + 净值曲线对比（约3-5秒）。"""
    import asyncio
    services = request.app.state.services
    return await asyncio.to_thread(services.evaluate_strategies)


@router.get("/factor/weights")
async def factor_weights(request: Request):
    """读取当前因子IC权重快照。"""
    services = request.app.state.services
    return {"weights": services.get_factor_weights()}


@router.get("/factor/evaluate")
async def factor_evaluate(request: Request):
    """因子IC评估（30因子预测力，约5-10秒）。"""
    import asyncio
    services = request.app.state.services
    return await asyncio.to_thread(services.evaluate_factors)


@router.get("/strategy/compare-combos")
async def compare_combos(request: Request):
    """主观预设组合 vs 数据驱动最优组合 对比（约5-10秒）。"""
    import asyncio
    services = request.app.state.services
    return await asyncio.to_thread(services.compare_combos)


@router.get("/strategy/optimal-combos")
async def optimal_combos(request: Request):
    """网格搜索最优策略组合（数据驱动，约5-10秒）。"""
    import asyncio
    services = request.app.state.services
    return await asyncio.to_thread(services.find_optimal_combos)


@router.get("/strategy/weights")
async def strategy_weights(request: Request):
    """读取当前策略质量权重快照（来自策略评估引擎最近一次刷新）。"""
    services = request.app.state.services
    return {"weights": services.get_strategy_weights()}


@router.post("/decision/push-feishu")
async def push_decision_feishu(body: DecisionRequest, request: Request):
    """推送交易决策清单到飞书。"""
    import asyncio
    services = request.app.state.services
    return await asyncio.to_thread(
        services.push_decision_feishu,
        None, body.strategy_keys, body.capital, body.min_score,
        body.exclude_markets, body.exclude_st,
    )


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

@router.get("/system")
async def get_system(request: Request):
    services = request.app.state.services
    return services.get_system_info()


@router.get("/system/logs")
async def get_logs(request: Request, limit: int = 100):
    handler = request.app.state.log_handler
    services = request.app.state.services
    return {"logs": services.get_recent_logs(handler, limit)}


@router.post("/system/sync")
async def sync_data(request: Request):
    services = request.app.state.services
    task_id = services.sync_data_async()
    return {"task_id": task_id, "status": TaskStatus.PENDING}


@router.post("/system/backfill")
async def backfill_data(request: Request):
    services = request.app.state.services
    task_id = services.backfill_async()
    return {"task_id": task_id, "status": TaskStatus.PENDING}


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------

@router.get("/config")
async def get_config(request: Request):
    settings = request.app.state.settings
    return {
        "feishu_webhook_url": settings.feishu_webhook_url,
        "strategy_webhooks": settings.strategy_webhooks,
        "db_path": settings.db_path,
        "start_date": settings.start_date,
    }


class PortfolioRequest(BaseModel):
    symbols: list[str]


class DecisionRequest(BaseModel):
    strategy_keys: list[str] | None = None
    capital: float = 100000.0
    min_score: int = 50
    exclude_markets: list[str] | None = None
    exclude_st: bool = False
    max_candidates: int | None = None  # 分析池上限，None=60（个股并行后可放宽）
    include_auction: bool = False  # 纳入今日竞价A级票到决策池


class ConfigUpdate(BaseModel):
    feishu_webhook_url: str | None = None
    strategy_webhooks: dict[str, str] | None = None


@router.put("/config")
async def update_config(body: ConfigUpdate, request: Request):
    from sequoia_x.web.env_writer import write_env, reload_settings

    updates: dict[str, str] = {}
    if body.feishu_webhook_url is not None:
        updates["FEISHU_WEBHOOK_URL"] = body.feishu_webhook_url
    if body.strategy_webhooks is not None:
        for key, url in body.strategy_webhooks.items():
            updates[f"STRATEGY_WEBHOOK_{key.upper()}"] = url

    if not updates:
        return {"ok": True, "message": "无需更新"}

    write_env(updates)
    new_settings = reload_settings()
    request.app.state.settings = new_settings
    request.app.state.services.settings = new_settings
    return {"ok": True, "message": "配置已更新"}


class WebhookTest(BaseModel):
    url: str


@router.post("/config/test-webhook")
async def test_webhook(body: WebhookTest):
    import asyncio
    import requests as req

    def _do_test() -> dict:
        payload = {"msg_type": "text", "content": {"text": "Sequoia-X Webhook Test - OK"}}
        try:
            resp = req.post(body.url, json=payload, timeout=10)
            data = resp.json()
            return {
                "ok": resp.status_code == 200 and data.get("code") == 0,
                "status_code": resp.status_code,
                "feishu_code": data.get("code"),
                "message": data.get("msg", ""),
            }
        except Exception as e:
            return {"ok": False, "status_code": 0, "feishu_code": -1, "message": str(e)}

    result = await asyncio.to_thread(_do_test)
    return result


# ---------------------------------------------------------------------------
# Market analysis endpoints (大盘分析)
# ---------------------------------------------------------------------------

@router.post("/market/analyze")
async def analyze_market(request: Request, date: str | None = None):
    services = request.app.state.services
    task_id = services.analyze_market_async(date)
    return {"task_id": task_id, "status": TaskStatus.PENDING}


@router.get("/market/report")
async def get_market_report(request: Request, date: str | None = None):
    services = request.app.state.services
    report = services.get_market_report(date)
    if report is None:
        raise HTTPException(404, "暂无大盘分析报告，请先生成")
    return report


@router.post("/market/refresh-industry")
async def refresh_industry(request: Request):
    services = request.app.state.services
    task_id = services.refresh_industry_cache_async()
    return {"task_id": task_id, "status": TaskStatus.PENDING}


@router.post("/market/backtest")
async def run_backtest(request: Request):
    services = request.app.state.services
    task_id = services.backtest_async()
    return {"task_id": task_id, "status": TaskStatus.PENDING}


@router.get("/market/backtest/report")
async def get_backtest_report(request: Request):
    services = request.app.state.services
    report = services.get_backtest_report()
    if report is None:
        raise HTTPException(404, "暂无回测报告，请先运行回测")
    return report


# ---------------------------------------------------------------------------
# 持仓跟踪 Position Tracking endpoints
# ---------------------------------------------------------------------------

class HoldingCreate(BaseModel):
    symbol: str
    name: str = ""
    entry_price: float
    shares: int
    entry_date: str | None = None
    stop_loss: float = 0
    target: float = 0
    grade: str = ""
    hit_strategies: str = ""
    notes: str = ""


class HoldingUpdate(BaseModel):
    entry_price: float | None = None
    shares: int | None = None
    stop_loss: float | None = None
    target: float | None = None
    notes: str | None = None
    name: str | None = None
    entry_date: str | None = None


class HoldingClose(BaseModel):
    close_price: float
    reason: str = ""


class HoldingImport(BaseModel):
    buy_list: list[dict]


@router.get("/positions")
async def list_positions(request: Request, status: str = "open"):
    services = request.app.state.services
    return {"holdings": services.list_holdings(status)}


@router.post("/positions")
async def create_position(body: HoldingCreate, request: Request):
    services = request.app.state.services
    hid = services.add_holding(body.model_dump())
    return {"id": hid, "ok": True}


@router.put("/positions/{hid}")
async def update_position(hid: int, body: HoldingUpdate, request: Request):
    services = request.app.state.services
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    ok = services.update_holding(hid, **fields)
    return {"ok": ok}


@router.post("/positions/{hid}/close")
async def close_position(hid: int, body: HoldingClose, request: Request):
    services = request.app.state.services
    ok = services.close_holding(hid, body.close_price, body.reason)
    return {"ok": ok}


@router.delete("/positions/{hid}")
async def delete_position(hid: int, request: Request):
    services = request.app.state.services
    ok = services.delete_holding(hid)
    return {"ok": ok}


@router.post("/positions/scan")
async def scan_positions(request: Request, apply_stop_move: bool = False):
    """扫描所有持仓，返回移动止损/减仓/止盈信号 + 组合摘要。"""
    services = request.app.state.services
    return services.scan_positions(apply_stop_move=apply_stop_move)


@router.post("/positions/import-decision")
async def import_decision_holdings(body: HoldingImport, request: Request):
    """把决策买入清单批量导入持仓表。"""
    services = request.app.state.services
    return services.import_decision_to_holdings(body.buy_list)


@router.post("/positions/push-feishu")
async def push_positions_feishu(request: Request):
    services = request.app.state.services
    return services.push_positions_feishu()


# ---------------------------------------------------------------------------
# 分钟K线
# ---------------------------------------------------------------------------

@router.get("/minute/watchlist")
async def minute_watchlist(request: Request):
    """查询关注池（竞价A+持仓）。"""
    from sequoia_x.analysis.minute import build_watchlist
    engine = request.app.state.engine
    return {"symbols": build_watchlist(engine.db_path)}


@router.post("/minute/collect")
async def minute_collect(request: Request, klt: int = 1, days: int = 1):
    """采集关注池全量分钟K线并落库。"""
    from sequoia_x.analysis.minute import MinuteCollector
    engine = request.app.state.engine
    collector = MinuteCollector(engine.db_path)
    return collector.collect_watchlist(klt=klt, days=days)


@router.get("/minute/{symbol}")
async def minute_data(request: Request, symbol: str, date: str | None = None):
    """查询单只股票分钟K线。"""
    engine = request.app.state.engine
    df = engine.get_minute_klines(symbol, date=date)
    return {"symbol": symbol, "count": len(df), "rows": df.to_dict("records") if not df.empty else []}


# ---------------------------------------------------------------------------
# 盘中实时信号
# ---------------------------------------------------------------------------

@router.post("/intraday/scan")
async def intraday_scan(request: Request, push: bool = False):
    """手动触发一次盘中信号扫描（9:30-15:00有效）。"""
    from sequoia_x.analysis.intraday_scanner import IntradayScanner
    from sequoia_x.notify.feishu import FeishuNotifier
    engine = request.app.state.engine
    settings = request.app.state.settings
    scanner = IntradayScanner(engine.db_path)
    notifier = FeishuNotifier(settings) if push else None
    signals = scanner.scan_once(notifier=notifier)
    return {"count": len(signals), "signals": [
        {"symbol": s.symbol, "name": s.name, "type": s.signal_type,
         "price": s.price, "detail": s.detail, "severity": s.severity, "ts": s.ts}
        for s in signals
    ]}




# ---------------------------------------------------------------------------
# 集合竞价
# ---------------------------------------------------------------------------

@router.get("/auction/history")
async def auction_history(request: Request, date: str | None = None, limit: int = 50):
    """查询历史竞价记录。"""
    from sequoia_x.analysis.auction import AuctionScanner
    engine = request.app.state.engine
    scanner = AuctionScanner(engine.db_path)
    rows = scanner.get_history(date=date, limit=limit)
    return {"rows": rows, "count": len(rows)}


@router.post("/auction/verify")
async def auction_verify(request: Request, auction_date: str | None = None):
    """触发竞价T+1命中验证。"""
    from sequoia_x.analysis.auction import AuctionScanner
    engine = request.app.state.engine
    scanner = AuctionScanner(engine.db_path)
    return scanner.verify_t1(auction_date=auction_date)


@router.get("/auction/verify/detail")
async def auction_verify_detail(request: Request, limit: int = 100):
    """查询验证明细。"""
    from sequoia_x.analysis.auction import AuctionScanner
    engine = request.app.state.engine
    scanner = AuctionScanner(engine.db_path)
    rows = scanner.get_verify_detail(limit=limit)
    return {"rows": rows, "count": len(rows)}


@router.post("/auction/scan")
async def auction_scan(request: Request, top_n: int = 50, push: bool = False):
    """手动触发竞价扫描（竞价时段9:25后有效，非竞价时段返回实时行情近似）。"""
    from sequoia_x.analysis.auction import AuctionScanner
    from sequoia_x.notify.feishu import FeishuNotifier
    engine = request.app.state.engine
    settings = request.app.state.settings
    scanner = AuctionScanner(engine.db_path)
    notifier = FeishuNotifier(settings) if push else None
    result = scanner.scan(top_n=top_n, push=push, notifier=notifier)
    return result
