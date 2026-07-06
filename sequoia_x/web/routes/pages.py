"""HTML 页面路由。"""

from fastapi import APIRouter, Request

from sequoia_x.web.services import STRATEGY_META

router = APIRouter()


@router.get("/")
async def index(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={"strategies": request.app.state.services.list_strategies()},
    )


@router.get("/strategy/{key}")
async def strategy_detail(key: str, request: Request):
    templates = request.app.state.templates
    if key not in STRATEGY_META:
        return templates.TemplateResponse(
            request=request, name="index.html",
            context={"strategies": request.app.state.services.list_strategies(), "error": f"未知策略: {key}"},
        )
    meta = STRATEGY_META[key]
    services = request.app.state.services
    return templates.TemplateResponse(
        request=request, name="strategy_detail.html",
        context={
            "key": key,
            "meta": meta,
            "strategy": next(s for s in services.list_strategies() if s["key"] == key),
            "results": services.get_cached_results(key),
        },
    )


@router.get("/config")
async def config_page(request: Request):
    templates = request.app.state.templates
    settings = request.app.state.settings
    return templates.TemplateResponse(
        request=request, name="config.html",
        context={"settings": settings, "strategies": STRATEGY_META},
    )


@router.get("/stocks")
async def stocks_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request, name="stock_browser.html",
        context={},
    )


@router.get("/auction")
async def auction_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request, name="auction.html",
        context={},
    )


@router.get("/intraday")
async def intraday_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request, name="intraday.html",
        context={},
    )


@router.get("/system")
async def system_page(request: Request):
    templates = request.app.state.templates
    services = request.app.state.services
    info = services.get_system_info()
    logs = services.get_recent_logs(request.app.state.log_handler, limit=50)
    return templates.TemplateResponse(
        request=request, name="system.html",
        context={"info": info, "logs": logs},
    )


@router.get("/market")
async def market_page(request: Request):
    templates = request.app.state.templates
    services = request.app.state.services
    report = services.get_market_report()
    return templates.TemplateResponse(
        request=request, name="market.html",
        context={"report": report},
    )


@router.get("/stock-analysis")
async def stock_analysis_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request, name="stock_analysis.html",
        context={},
    )


@router.get("/portfolio")
async def portfolio_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request, name="portfolio.html",
        context={},
    )


@router.get("/decision")
async def decision_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request, name="decision.html",
        context={},
    )

@router.get("/positions")
async def positions_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request, name="positions.html",
        context={},
    )

@router.get("/strategy-compare")
async def strategy_compare_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request, name="strategy_compare.html",
        context={},
    )

@router.get("/factor")
async def factor_page(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request, name="factor.html",
        context={},
    )
