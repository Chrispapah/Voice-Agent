from __future__ import annotations

import base64
import json
from pathlib import Path
import sys

from jose import jwt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_sdr_agent.auth import dependencies
from ai_sdr_agent.config import SDRSettings


ES256_TOKEN = (
    "eyJhbGciOiJFUzI1NiIsImtpZCI6IjMzNWNlNGY4LWQ5YWEtNDk5OC1hY2UwLTE0MzQ4ZTFlZmRjZiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJodHRwczovL2FodG9kZXNmdnBvanVxaGpyemFuLnN1cGFiYXNlLmNvL2F1dGgvdjEiLCJzdWIiOiI1YTA3ZjI0ZC1jOGVk"
    "LTRiMTEtYTFkNi1mNDgwYmM0ZDkzZGYiLCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzc2MDk0NjkwLCJpYXQiOjE3NzYwOTEw"
    "OTAsImVtYWlsIjoiYy5wYXBhaGFyYWxhYm91c0BnbWFpbC5jb20iLCJwaG9uZSI6IiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6"
    "ImVtYWlsIiwicHJvdmlkZXJzIjpbImVtYWlsIl19LCJ1c2VyX21ldGFkYXRhIjp7ImRpc3BsYXlfbmFtZSI6ImNwYXBhIiwiZW1haWwi"
    "OiJjLnBhcGFoYXJhbGFib3VzQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJwaG9uZV92ZXJpZmllZCI6ZmFsc2UsInN1"
    "YiI6IjVhMDdmMjRkLWM4ZWQtNGIxMS1hMWQ2LWY0ODBiYzRkOTNkZiJ9LCJyb2xlIjoiYXV0aGVudGljYXRlZCIsImFhbCI6ImFhbDEi"
    "LCJhbXIiOlt7Im1ldGhvZCI6InBhc3N3b3JkIiwidGltZXN0YW1wIjoxNzc2MDkxMDkwfV0sInNlc3Npb25faWQiOiJlZDJkOGQ2Zi03"
    "ZTZhLTRlNDMtOTdiNS01NWM1MmEwMDdmNTMiLCJpc19hbm9ueW1vdXMiOmZhbHNlfQ."
    "PfNWnJTDyjkWlOF4BlkA-YpuKD2UTDtjsKVMWs7-1xXzbYgj2LV-cvH4NDP4kwRV_pFxQSlwCJMo-rfWy4q4bg"
)


def test_decode_supabase_jwt_supports_es256_jwks(monkeypatch):
    # Fixture JWT uses a fixed exp; avoid time-dependent failures.
    real_decode = jwt.decode

    def decode_without_exp(*args: object, **kwargs: object):
        kw = dict(kwargs)
        opts = dict(kw.get("options") or {})
        opts["verify_exp"] = False
        kw["options"] = opts
        return real_decode(*args, **kw)

    monkeypatch.setattr(jwt, "decode", decode_without_exp)
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SDRSettings(
            supabase_url="https://ahtodesfvpojuqhjrzan.supabase.co",
            supabase_jwt_secret="CHANGE-ME-set-SUPABASE_JWT_SECRET-in-env",
        ),
    )
    monkeypatch.setattr(
        dependencies,
        "_fetch_jwks",
        lambda: [
            {
                "alg": "ES256",
                "crv": "P-256",
                "ext": True,
                "key_ops": ["verify"],
                "kid": "335ce4f8-d9aa-4998-ace0-14348e1efdcf",
                "kty": "EC",
                "use": "sig",
                "x": "TO72RGYvWRyAbBeXz1XuRCU5wl61LntL3bMSncOHbV4",
                "y": "LRpxyjgBAaKz3S8oHjFPHD8m5hktAJZjvBv-kEXTCVE",
            }
        ],
    )

    payload = dependencies.decode_supabase_jwt(ES256_TOKEN)

    assert payload["sub"] == "5a07f24d-c8ed-4b11-a1d6-f480bc4d93df"
    assert payload["aud"] == "authenticated"


def test_decode_supabase_jwt_supports_hs256_fallback(monkeypatch):
    secret = "unit-test-secret"
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SDRSettings(
            supabase_url="https://example.supabase.co",
            supabase_jwt_secret=secret,
        ),
    )

    token = jwt.encode(
        {
            "iss": "https://example.supabase.co/auth/v1",
            "sub": "11111111-1111-1111-1111-111111111111",
            "aud": "authenticated",
        },
        secret,
        algorithm="HS256",
    )

    payload = dependencies.decode_supabase_jwt(token)

    assert payload["sub"] == "11111111-1111-1111-1111-111111111111"


def test_decode_supabase_jwt_rejects_unknown_algorithm(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: SDRSettings(supabase_url="https://example.supabase.co"),
    )

    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "11111111-1111-1111-1111-111111111111", "aud": "authenticated"}).encode()
    ).decode().rstrip("=")
    signature = base64.urlsafe_b64encode(b"sig").decode().rstrip("=")
    token = f"{header}.{payload}.{signature}"

    try:
        dependencies.decode_supabase_jwt(token)
    except Exception as exc:
        assert "Unsupported JWT algorithm" in str(exc)
    else:
        raise AssertionError("Expected decode_supabase_jwt to reject unsupported algorithms")
