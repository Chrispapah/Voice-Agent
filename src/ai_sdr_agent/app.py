from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
from vocode.streaming.models.telephony import BaseCallConfig
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.telephony.config_manager.in_memory_config_manager import InMemoryConfigManager
from vocode.streaming.telephony.config_manager.redis_config_manager import RedisConfigManager
from vocode.streaming.telephony.server.base import TelephonyServer, TwilioInboundCallConfig

from ai_sdr_agent.agent_factory import SDRAgentFactory
from ai_sdr_agent.config import SDRSettings, get_settings
from ai_sdr_agent.db.engine import init_db
from ai_sdr_agent.graph.service import SDRConversationService, SDRRuntimeDependencies
from ai_sdr_agent.routers import bots_router, test_sessions_router
from ai_sdr_agent.services.brain import build_conversation_brain
from ai_sdr_agent.services.call_scheduler import CallScheduler
from ai_sdr_agent.services.latency_analytics import shared_latency_analytics
from ai_sdr_agent.services.persistence import (
    InMemoryCallLogRepository,
    InMemoryLeadRepository,
    InMemorySessionStore,
)
from ai_sdr_agent.services.pre_call_loader import PreCallLoader
from ai_sdr_agent.synthesizer_factory import SDRSynthesizerFactory
from ai_sdr_agent.transcriber_factory import (
    SDRTranscriberFactory,
    build_telephony_deepgram_transcriber_config,
    build_telephony_google_transcriber_config,
    resolve_telephony_deepgram_model,
)
from ai_sdr_agent.tools import StubCRMGateway, StubCalendarGateway, StubEmailGateway
from ai_sdr_agent.vocode_agent import build_agent_config


class StartSessionRequest(BaseModel):
    lead_id: str


class TurnRequest(BaseModel):
    human_input: str


class OutboundCallRequest(BaseModel):
    lead_id: str


class HybridCallConfigManager(BaseConfigManager):
    """Prefer in-process configs, with Redis fallback across replicas."""

    def __init__(self, redis_enabled: bool):
        self._memory = InMemoryConfigManager()
        self._redis = RedisConfigManager() if redis_enabled else None

    async def save_config(self, conversation_id: str, config: BaseCallConfig):
        await self._memory.save_config(conversation_id, config)
        if self._redis is not None:
            await self._redis.save_config(conversation_id, config)

    async def get_config(self, conversation_id: str):
        config = await self._memory.get_config(conversation_id)
        if config is not None:
            return config
        if self._redis is None:
            return None
        config = await self._redis.get_config(conversation_id)
        if config is not None:
            await self._memory.save_config(conversation_id, config)
        return config

    async def delete_config(self, conversation_id: str):
        await self._memory.delete_config(conversation_id)
        if self._redis is not None:
            await self._redis.delete_config(conversation_id)


def create_app(settings: SDRSettings | None = None) -> FastAPI:
    settings = settings or get_settings()
    missing_runtime_values = settings.missing_runtime_values()
    telephony_ready = not missing_runtime_values

    # Legacy in-memory wiring (kept for backward compat / telephony path)
    lead_repository = InMemoryLeadRepository()
    seed_lead = lead_repository._leads["lead-001"]
    calendar_gateway = StubCalendarGateway()
    email_gateway = StubEmailGateway()
    crm_gateway = StubCRMGateway(seed_leads=[seed_lead])
    call_log_repository = InMemoryCallLogRepository()
    session_store = InMemorySessionStore()
    pre_call_loader = PreCallLoader(
        lead_repository=lead_repository,
        calendar_gateway=calendar_gateway,
    )
    conversation_service = SDRConversationService(
        SDRRuntimeDependencies(
            brain=build_conversation_brain(settings),
            calendar_gateway=calendar_gateway,
            email_gateway=email_gateway,
            crm_gateway=crm_gateway,
            pre_call_loader=pre_call_loader,
            session_store=session_store,
            call_log_repository=call_log_repository,
            email_template_path=Path("templates/follow_up_email.html"),
            sales_rep_name=settings.default_sales_rep_name,
            from_name=settings.default_from_name,
            latency_analytics=shared_latency_analytics,
        )
    )
    config_manager = _build_config_manager(settings)
    call_scheduler = CallScheduler(settings=settings, config_manager=config_manager)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logger.info("Initialising database...")
        await init_db(settings.database_url)
        logger.info("Database ready.")
        yield

    app = FastAPI(title=settings.app_name, version="0.3.0", lifespan=lifespan)

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

    # CORS for deployed frontend and local development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3001",
            "http://localhost:3000",
            "https://voice-agent-zeta-tawny.vercel.app",
        ],
        allow_origin_regex=r"https://.*\.vercel\.app",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.conversation_service = conversation_service
    app.state.lead_repository = lead_repository
    app.state.call_log_repository = call_log_repository
    app.state.calendar_gateway = calendar_gateway
    app.state.email_gateway = email_gateway
    app.state.crm_gateway = crm_gateway
    app.state.config_manager = config_manager
    app.state.call_scheduler = call_scheduler
    app.state.latency_analytics = shared_latency_analytics

    # ── AI engine router (authenticated via Supabase JWT) ──────────
    app.include_router(bots_router)
    app.include_router(test_sessions_router)

    # ── Legacy routes (telephony + backward compat) ────────────────

    @app.get("/healthz")
    async def healthz():
        return {
            "status": "ok" if telephony_ready else "degraded",
            "app_name": settings.app_name,
            "llm_provider": settings.llm_provider,
            "stt_provider": settings.stt_provider,
            "tts_provider": "elevenlabs",
            "telephony_ready": telephony_ready,
            "missing_runtime_values": missing_runtime_values,
            "config_manager": settings.config_manager_kind(),
            "provider_summary": settings.provider_summary(),
        }

    @app.get("/analytics/latency")
    async def latency_analytics(
        recent_limit: int = Query(default=50, ge=1, le=200),
    ):
        """Aggregated turn latencies (graph + persist) and perceived phone latency (graph → first TTS chunk)."""
        return await shared_latency_analytics.snapshot(recent_limit=recent_limit)

    @app.get("/debug/telephony")
    async def debug_telephony():
        routes = sorted(
            {
                route.path
                for route in app.routes
                if hasattr(route, "path")
            }
        )
        return {
            "telephony_ready": telephony_ready,
            "base_url": settings.normalized_base_url(),
            "missing_runtime_values": missing_runtime_values,
            "config_manager": settings.config_manager_kind(),
            "inbound_call_route_registered": "/inbound_call" in routes,
            "routes": routes,
        }

    @app.get("/leads")
    async def list_leads():
        return [lead.model_dump() for lead in lead_repository._leads.values()]

    @app.post("/sessions")
    async def start_session(request: StartSessionRequest):
        conversation_id = await conversation_service.start_session(request.lead_id)
        state = await conversation_service.get_state(conversation_id)
        state = await conversation_service.handle_turn(conversation_id, "")
        return {
            "conversation_id": conversation_id,
            "agent_response": state["last_agent_response"],
            "state": state,
        }

    @app.post("/sessions/{conversation_id}/turns")
    async def run_turn(conversation_id: str, request: TurnRequest):
        try:
            state = await conversation_service.handle_turn(conversation_id, request.human_input)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "conversation_id": conversation_id,
            "agent_response": state["last_agent_response"],
            "state": state,
        }

    @app.get("/sessions/{conversation_id}")
    async def get_session(conversation_id: str):
        try:
            return await conversation_service.get_state(conversation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/outbound/calls")
    async def start_outbound_call(request: OutboundCallRequest):
        try:
            lead = await lead_repository.get_lead(request.lead_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        result = await call_scheduler.schedule_outbound_call(lead)
        return result.__dict__

    if telephony_ready and settings.normalized_base_url():
        telephony_server = TelephonyServer(
            base_url=settings.normalized_base_url(),
            config_manager=config_manager,
            transcriber_factory=SDRTranscriberFactory(),
            synthesizer_factory=SDRSynthesizerFactory(),
            inbound_call_configs=[
                TwilioInboundCallConfig(
                    url="/inbound_call",
                    agent_config=build_agent_config(
                        lead_id="lead-001",
                        calendar_id=settings.default_calendar_id,
                        sales_rep_name=settings.default_sales_rep_name,
                        initial_message_text=settings.initial_greeting,
                    ),
                    transcriber_config=_build_transcriber_config(settings),
                    synthesizer_config=_build_synthesizer_config(settings),
                    twilio_config=TwilioConfig(
                        account_sid=settings.twilio_account_sid or "",
                        auth_token=settings.twilio_auth_token or "",
                    ),
                )
            ],
            agent_factory=SDRAgentFactory(conversation_service),
        )
        app.include_router(telephony_server.get_router())

    registered_routes = sorted(
        {
            route.path
            for route in app.routes
            if hasattr(route, "path")
        }
    )
    logger.info(
        "Application startup telephony_ready={} base_url={} config_manager={} "
        "stt_provider={} deepgram_model={} tts_provider={} elevenlabs_model_id={} "
        "elevenlabs_websocket={} missing_runtime_values={} inbound_call_route_registered={} routes={}",
        telephony_ready,
        settings.normalized_base_url(),
        settings.config_manager_kind(),
        settings.stt_provider,
        resolve_telephony_deepgram_model(settings.deepgram_model),
        "elevenlabs",
        settings.elevenlabs_model_id,
        settings.elevenlabs_use_websocket,
        missing_runtime_values,
        "/inbound_call" in registered_routes,
        registered_routes,
    )

    return app


def _build_config_manager(settings: SDRSettings):
    redis_enabled = settings.use_redis_config_manager and bool(settings.redis_url)
    if redis_enabled:
        logger.info(
            "Using hybrid call config manager with in-memory primary and Redis fallback."
        )
    else:
        logger.info("Using in-memory call config manager.")
    return HybridCallConfigManager(redis_enabled=redis_enabled)


def _build_synthesizer_config(settings: SDRSettings):
    return ElevenLabsSynthesizerConfig.from_telephone_output_device(
        api_key=settings.elevenlabs_api_key or "",
        voice_id=settings.elevenlabs_voice_id or "",
        model_id=settings.elevenlabs_model_id,
        optimize_streaming_latency=settings.elevenlabs_optimize_streaming_latency,
        experimental_websocket=settings.elevenlabs_use_websocket,
    )


def _build_transcriber_config(settings: SDRSettings):
    if settings.stt_provider == "google":
        return build_telephony_google_transcriber_config(settings)
    return build_telephony_deepgram_transcriber_config(settings)
