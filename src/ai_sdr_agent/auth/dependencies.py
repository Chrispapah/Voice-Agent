from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from ai_sdr_agent.config import get_settings

_bearer = HTTPBearer()
_JWKS_CACHE_TTL_SECONDS = 300
_SUPABASE_JWKS_CACHE: tuple[float, list[dict[str, Any]]] | None = None


def _get_supabase_issuer() -> str | None:
    settings = get_settings()
    if not settings.supabase_url:
        return None
    return settings.supabase_url.rstrip("/") + "/auth/v1"


def _get_supabase_jwks_url() -> str | None:
    issuer = _get_supabase_issuer()
    if issuer is None:
        return None
    return issuer + "/.well-known/jwks.json"


def _get_jwt_secret() -> str | None:
    settings = get_settings()
    secret = settings.supabase_jwt_secret.strip()
    if not secret or secret == "CHANGE-ME-set-SUPABASE_JWT_SECRET-in-env":
        return None
    return secret


def _fetch_jwks() -> list[dict[str, Any]]:
    global _SUPABASE_JWKS_CACHE

    now = time.time()
    if _SUPABASE_JWKS_CACHE is not None:
        expires_at, cached_keys = _SUPABASE_JWKS_CACHE
        if now < expires_at:
            return cached_keys

    jwks_url = _get_supabase_jwks_url()
    if jwks_url is None:
        return []

    with urllib.request.urlopen(jwks_url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))

    keys = payload.get("keys", [])
    _SUPABASE_JWKS_CACHE = (now + _JWKS_CACHE_TTL_SECONDS, keys)
    return keys


def _find_jwk(kid: str | None) -> dict[str, Any] | None:
    if not kid:
        return None

    try:
        keys = _fetch_jwks()
    except (OSError, urllib.error.URLError, ValueError):
        return None

    for key in keys:
        if key.get("kid") == kid:
            return key
    return None


def _decode_with_jwks(token: str) -> dict[str, Any]:
    header = jwt.get_unverified_header(token)
    jwk = _find_jwk(header.get("kid"))
    if jwk is None:
        raise JWTError("No matching JWK found")

    issuer = _get_supabase_issuer()
    options: dict[str, Any] = {"verify_aud": True}
    kwargs: dict[str, Any] = {"audience": "authenticated", "options": options}
    if issuer is not None:
        kwargs["issuer"] = issuer

    return jwt.decode(token, jwk, algorithms=[jwk["alg"]], **kwargs)


def _decode_with_hs256(token: str) -> dict[str, Any]:
    secret = _get_jwt_secret()
    if secret is None:
        raise JWTError("SUPABASE_JWT_SECRET is not configured")

    issuer = _get_supabase_issuer()
    options: dict[str, Any] = {"verify_aud": True}
    kwargs: dict[str, Any] = {"audience": "authenticated", "options": options}
    if issuer is not None:
        kwargs["issuer"] = issuer

    return jwt.decode(token, secret, algorithms=["HS256"], **kwargs)


def decode_supabase_jwt(token: str) -> dict[str, Any]:
    header = jwt.get_unverified_header(token)
    algorithm = header.get("alg")

    if algorithm == "HS256":
        return _decode_with_hs256(token)

    if algorithm in {"RS256", "ES256"}:
        return _decode_with_jwks(token)

    raise JWTError(f"Unsupported JWT algorithm: {algorithm}")


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> uuid.UUID:
    """Decode a Supabase-issued JWT and return the user's UUID.

    No database round-trip is needed -- the ``sub`` claim in a Supabase
    JWT is the user's ``auth.users.id``.
    """
    try:
        payload = decode_supabase_jwt(credentials.credentials)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    return uuid.UUID(user_id_str)
