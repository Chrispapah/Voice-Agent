from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig
from vocode.streaming.telephony.config_manager.in_memory_config_manager import InMemoryConfigManager
from vocode.streaming.telephony.config_manager.redis_config_manager import RedisConfigManager
from vocode.streaming.telephony.server.base import TelephonyServer, TwilioInboundCallConfig
from vocode.streaming.transcriber.deepgram_transcriber import (
    DeepgramEndpointingConfig,
    TimeSilentConfig,
)

from ai_sdr_agent.agent_factory import SDRAgentFactory
from ai_sdr_agent.config import SDRSettings, get_settings
from ai_sdr_agent.graph.service import SDRConversationService, SDRRuntimeDependencies
from ai_sdr_agent.services.brain import build_conversation_brain
from ai_sdr_agent.services.call_scheduler import CallScheduler
from ai_sdr_agent.services.persistence import (
    InMemoryCallLogRepository,
    InMemoryLeadRepository,
    InMemorySessionStore,
)
from ai_sdr_agent.services.pre_call_loader import PreCallLoader
from ai_sdr_agent.tools import StubCRMGateway, StubCalendarGateway, StubEmailGateway
from ai_sdr_agent.vocode_agent import build_agent_config


class StartSessionRequest(BaseModel):
    lead_id: str


class TurnRequest(BaseModel):
    human_input: str


class OutboundCallRequest(BaseModel):
    lead_id: str


def create_app(settings: SDRSettings | None = None) -> FastAPI:
    settings = settings or get_settings()
    missing_runtime_values = settings.missing_runtime_values()
    telephony_ready = not missing_runtime_values

    lead_repository = InMemoryLeadRepository()
    seed_lead = lead_repository._leads["lead-001"]  # typed seed used by all stubs
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
        )
    )
    config_manager = _build_config_manager(settings)
    call_scheduler = CallScheduler(settings=settings, config_manager=config_manager)

    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.state.settings = settings
    app.state.conversation_service = conversation_service
    app.state.lead_repository = lead_repository
    app.state.call_log_repository = call_log_repository
    app.state.calendar_gateway = calendar_gateway
    app.state.email_gateway = email_gateway
    app.state.crm_gateway = crm_gateway
    app.state.config_manager = config_manager
    app.state.call_scheduler = call_scheduler

    @app.get("/healthz")
    async def healthz():
        return {
            "status": "ok" if telephony_ready else "degraded",
            "app_name": settings.app_name,
            "llm_provider": settings.llm_provider,
            "stt_provider": "deepgram",
            "tts_provider": "elevenlabs",
            "telephony_ready": telephony_ready,
            "missing_runtime_values": missing_runtime_values,
            "config_manager": settings.config_manager_kind(),
            "provider_summary": settings.provider_summary(),
        }

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
        "deepgram",
        settings.deepgram_model,
        "elevenlabs",
        settings.elevenlabs_model_id,
        settings.elevenlabs_use_websocket,
        missing_runtime_values,
        "/inbound_call" in registered_routes,
        registered_routes,
    )

    return app


def _build_config_manager(settings: SDRSettings):
    if settings.use_redis_config_manager and settings.redis_url:
        return RedisConfigManager()
    return InMemoryConfigManager()


def _build_synthesizer_config(settings: SDRSettings):
    return ElevenLabsSynthesizerConfig.from_telephone_output_device(
        api_key=settings.elevenlabs_api_key or "",
        voice_id=settings.elevenlabs_voice_id or "",
        model_id=settings.elevenlabs_model_id,
        optimize_streaming_latency=settings.elevenlabs_optimize_streaming_latency,
        experimental_websocket=settings.elevenlabs_use_websocket,
    )


def _build_transcriber_config(settings: SDRSettings):
    return DeepgramTranscriberConfig.from_telephone_input_device(
        endpointing_config=DeepgramEndpointingConfig(
            vad_threshold_ms=settings.deepgram_vad_threshold_ms,
            utterance_cutoff_ms=settings.deepgram_utterance_cutoff_ms,
            time_silent_config=TimeSilentConfig(
                time_cutoff_seconds=settings.deepgram_time_cutoff_seconds,
                post_punctuation_time_seconds=settings.deepgram_post_punctuation_time_seconds,
            ),
            use_single_utterance_endpointing_for_first_utterance=(
                settings.deepgram_single_utterance_for_first_response
            ),
        ),
        api_key=settings.deepgram_api_key,
        model=settings.deepgram_model,
        mute_during_speech=settings.deepgram_mute_during_speech,
    )
