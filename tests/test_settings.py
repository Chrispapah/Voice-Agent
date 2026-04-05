import os

from vocode_contact_center.settings import ContactCenterSettings


def test_missing_runtime_values_for_openai_setup():
    settings = ContactCenterSettings(
        twilio_account_sid="sid",
        twilio_auth_token="token",
        deepgram_api_key="deepgram",
        elevenlabs_api_key="eleven",
        elevenlabs_voice_id="voice",
        langchain_provider="openai",
        openai_api_key=None,
        base_url=None,
        ngrok_auth_token=None,
    )

    missing = settings.missing_runtime_values()

    assert "OPENAI_API_KEY" in missing
    assert "BASE_URL or NGROK_AUTH_TOKEN" in missing


def test_missing_runtime_values_are_empty_when_minimum_config_is_present():
    settings = ContactCenterSettings(
        base_url="demo.example.com",
        twilio_account_sid="sid",
        twilio_auth_token="token",
        deepgram_api_key="deepgram",
        elevenlabs_api_key="eleven",
        elevenlabs_voice_id="voice",
        langchain_provider="openai",
        openai_api_key="openai",
    )

    assert settings.missing_runtime_values() == []


def test_railway_public_domain_counts_as_public_endpoint():
    settings = ContactCenterSettings(
        railway_public_domain="demo-production.up.railway.app",
        twilio_account_sid="sid",
        twilio_auth_token="token",
        deepgram_api_key="deepgram",
        elevenlabs_api_key="eleven",
        elevenlabs_voice_id="voice",
        langchain_provider="openai",
        openai_api_key="openai",
    )

    assert settings.missing_runtime_values() == []
    assert settings.normalized_base_url() == "demo-production.up.railway.app"


def test_redis_url_is_split_into_vocode_redis_environment_variables():
    settings = ContactCenterSettings(
        redis_url="rediss://default:secret@example.upstash.io:6379"
    )

    settings.apply_redis_env()

    assert os.environ["REDISHOST"] == "example.upstash.io"
    assert os.environ["REDISPORT"] == "6379"
    assert os.environ["REDISUSER"] == "default"
    assert os.environ["REDISPASSWORD"] == "secret"
    assert os.environ["REDISSSL"] == "1"


def test_non_ssl_redis_url_disables_redisssl_flag():
    settings = ContactCenterSettings(
        redis_url="redis://default:secret@redis.railway.internal:6379"
    )

    settings.apply_redis_env()

    assert os.environ["REDISHOST"] == "redis.railway.internal"
    assert os.environ["REDISPORT"] == "6379"
    assert os.environ["REDISSSL"] == ""


def test_latency_defaults_are_tuned_for_faster_turn_taking():
    settings = ContactCenterSettings(
        deepgram_vad_threshold_ms=140,
        deepgram_utterance_cutoff_ms=600,
        deepgram_time_cutoff_seconds=0.12,
        deepgram_post_punctuation_time_seconds=0.05,
        non_streaming_chunk_min_words=3,
        non_streaming_chunk_max_words=8,
        non_streaming_chunk_min_chars=12,
        realtime_enabled=True,
        realtime_audio_encoding="linear16",
        realtime_sample_rate=16000,
    )

    assert settings.deepgram_vad_threshold_ms == 140
    assert settings.deepgram_utterance_cutoff_ms == 600
    assert settings.deepgram_time_cutoff_seconds == 0.12
    assert settings.deepgram_post_punctuation_time_seconds == 0.05
    assert settings.langchain_max_tokens == 64
    assert settings.langchain_recent_message_limit == 6
    assert settings.langchain_summary_max_chars == 600
    assert settings.require_streaming_synthesizer is True
    assert settings.non_streaming_chunk_min_words == 3
    assert settings.non_streaming_chunk_max_words == 8
    assert settings.non_streaming_chunk_min_chars == 12
    assert settings.realtime_enabled is True
    assert settings.realtime_audio_encoding == "linear16"
    assert settings.realtime_sample_rate == 16000


def test_missing_realtime_values_ignore_telephony_dependencies():
    settings = ContactCenterSettings(
        elevenlabs_api_key="eleven",
        elevenlabs_voice_id="voice",
        langchain_provider="openai",
        openai_api_key="openai",
        twilio_account_sid=None,
        twilio_auth_token=None,
        deepgram_api_key=None,
    )

    assert settings.missing_realtime_values() == []


def test_nltk_auto_download_defaults_off_on_railway():
    settings = ContactCenterSettings(railway_public_domain="demo-production.up.railway.app")

    assert settings.should_auto_download_nltk() is False


def test_nltk_auto_download_can_be_forced_on():
    settings = ContactCenterSettings(nltk_auto_download=True)

    assert settings.should_auto_download_nltk() is True
