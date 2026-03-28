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
