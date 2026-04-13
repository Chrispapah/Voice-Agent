from __future__ import annotations

import os
import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

_bearer = HTTPBearer()

_SUPABASE_JWT_SECRET: str | None = None


def _get_jwt_secret() -> str:
    global _SUPABASE_JWT_SECRET
    if _SUPABASE_JWT_SECRET is None:
        _SUPABASE_JWT_SECRET = os.environ.get(
            "SUPABASE_JWT_SECRET", "CHANGE-ME-set-SUPABASE_JWT_SECRET-in-env"
        )
    return _SUPABASE_JWT_SECRET


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> uuid.UUID:
    """Decode a Supabase-issued JWT and return the user's UUID.

    No database round-trip is needed -- the ``sub`` claim in a Supabase
    JWT is the user's ``auth.users.id``.
    """
    try:
        payload = jwt.decode(
            credentials.credentials,
            _get_jwt_secret(),
            algorithms=["HS256"],
            audience="authenticated",
        )
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
