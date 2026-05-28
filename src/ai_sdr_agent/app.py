from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from ai_sdr_agent.config import SDRSettings, get_settings
from ai_sdr_agent.db.engine import init_db
from ai_sdr_agent.routers import (
    agent_previews_router,
    bots_router,
    conversation_shares_router,
    hybrid_voice_router,
    openai_realtime_voice_router,
    test_sessions_router,
    web_voice_router,
)
from ai_sdr_agent.services.latency_analytics import shared_latency_analytics


def create_app(settings: SDRSettings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logger.info("Initialising database...")
        await init_db(settings.database_url)
        logger.info("Database ready.")
        yield

    app = FastAPI(title=settings.app_name, version="0.4.0", lifespan=lifespan)

    @app.middleware("http")
    async def log_unhandled_errors(request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled request error path={}", request.url.path)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:3001",
            "http://localhost:3000",
            "https://akoi.ai",
            "https://www.akoi.ai",
            "https://voice-agent-zeta-tawny.vercel.app",
        ],
        allow_origin_regex=r"https://.*\.vercel\.app",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.latency_analytics = shared_latency_analytics

    app.include_router(agent_previews_router)
    app.include_router(bots_router)
    app.include_router(conversation_shares_router)
    app.include_router(test_sessions_router)
    app.include_router(hybrid_voice_router)
    app.include_router(openai_realtime_voice_router)
    app.include_router(web_voice_router)

    @app.get("/healthz")
    async def healthz():
        return {
            "status": "ok",
            "app_name": settings.app_name,
            "llm_provider": settings.llm_provider,
            "provider_summary": settings.provider_summary(),
        }

    @app.get("/analytics/latency")
    async def latency_analytics(
        recent_limit: int = Query(default=50, ge=1, le=200),
    ):
        """Aggregated turn latencies for graph turns and browser web_voice timings."""
        return await shared_latency_analytics.snapshot(recent_limit=recent_limit)

    registered_routes = sorted(
        route.path for route in app.routes if hasattr(route, "path")
    )
    logger.info("Application startup routes={}", registered_routes)

    return app
