from ai_sdr_agent.routers.bots import router as bots_router
from ai_sdr_agent.routers.agent_previews import router as agent_previews_router
from ai_sdr_agent.routers.conversation_shares import router as conversation_shares_router
from ai_sdr_agent.routers.hybrid_voice import router as hybrid_voice_router
from ai_sdr_agent.routers.openai_realtime_voice import router as openai_realtime_voice_router
from ai_sdr_agent.routers.test_sessions import router as test_sessions_router
from ai_sdr_agent.routers.web_voice import router as web_voice_router

__all__ = [
    "bots_router",
    "agent_previews_router",
    "conversation_shares_router",
    "hybrid_voice_router",
    "openai_realtime_voice_router",
    "test_sessions_router",
    "web_voice_router",
]
