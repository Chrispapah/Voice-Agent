from __future__ import annotations

import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ai_sdr_agent.auth.dependencies import get_current_user_id
from ai_sdr_agent.db.engine import get_async_session
from ai_sdr_agent.db.repositories import PgAuthConnectionRepository, PgWorkspaceEnvVarRepository
from ai_sdr_agent.services.tool_config import parse_tool_config

router = APIRouter(prefix="/api/workspace", tags=["workspace"])

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _mask_value(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "****"
    return value[:2] + "****" + value[-2:]


class EnvVarCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    value: str = ""


class EnvVarUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    value: str | None = None


class EnvVarResponse(BaseModel):
    id: str
    name: str
    value_masked: str
    created_at: str | None
    updated_at: str | None


class AuthConnectionCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    type: str = Field(default="api_key_header", max_length=40)
    config_json: dict = Field(default_factory=dict)


class AuthConnectionResponse(BaseModel):
    id: str
    label: str
    type: str
    config_json: dict
    created_at: str | None
    updated_at: str | None


class ToolValidateRequest(BaseModel):
    kind: str = "http"
    config_json: dict = Field(default_factory=dict)


class ToolValidateResponse(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


@router.get("/env-vars")
async def list_env_vars(
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    repo = PgWorkspaceEnvVarRepository(session)
    rows = await repo.list_for_user(user_id)
    return [
        EnvVarResponse(
            id=str(r.id),
            name=r.name,
            value_masked=_mask_value(r.value),
            created_at=r.created_at.isoformat() if r.created_at else None,
            updated_at=r.updated_at.isoformat() if r.updated_at else None,
        )
        for r in rows
    ]


@router.post("/env-vars", status_code=status.HTTP_201_CREATED)
async def create_env_var(
    body: EnvVarCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    name = body.name.strip()
    if not _ENV_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid env var name")
    repo = PgWorkspaceEnvVarRepository(session)
    if await repo.get_by_name(user_id, name):
        raise HTTPException(status_code=409, detail="Env var already exists")
    row = await repo.create(user_id=user_id, name=name, value=body.value)
    await session.commit()
    return EnvVarResponse(
        id=str(row.id),
        name=row.name,
        value_masked=_mask_value(row.value),
        created_at=row.created_at.isoformat() if row.created_at else None,
        updated_at=row.updated_at.isoformat() if row.updated_at else None,
    )


@router.patch("/env-vars/{var_id}")
async def update_env_var(
    var_id: str,
    body: EnvVarUpdate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    repo = PgWorkspaceEnvVarRepository(session)
    fields: dict[str, str] = {}
    if body.name is not None:
        name = body.name.strip()
        if not _ENV_NAME_RE.match(name):
            raise HTTPException(status_code=400, detail="Invalid env var name")
        fields["name"] = name
    if body.value is not None:
        fields["value"] = body.value
    row = await repo.update(uuid.UUID(var_id), user_id, **fields)
    if row is None:
        raise HTTPException(status_code=404, detail="Env var not found")
    await session.commit()
    return EnvVarResponse(
        id=str(row.id),
        name=row.name,
        value_masked=_mask_value(row.value),
        created_at=row.created_at.isoformat() if row.created_at else None,
        updated_at=row.updated_at.isoformat() if row.updated_at else None,
    )


@router.delete("/env-vars/{var_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_env_var(
    var_id: str,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    repo = PgWorkspaceEnvVarRepository(session)
    if not await repo.delete(uuid.UUID(var_id), user_id):
        raise HTTPException(status_code=404, detail="Env var not found")
    await session.commit()


@router.get("/auth-connections")
async def list_auth_connections(
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    repo = PgAuthConnectionRepository(session)
    rows = await repo.list_for_user(user_id)
    return [
        AuthConnectionResponse(
            id=str(r.id),
            label=r.label,
            type=r.type,
            config_json=_mask_auth_config(r.config_json),
            created_at=r.created_at.isoformat() if r.created_at else None,
            updated_at=r.updated_at.isoformat() if r.updated_at else None,
        )
        for r in rows
    ]


def _mask_auth_config(cfg: dict) -> dict:
    masked = dict(cfg)
    for key in ("api_key", "bearer_token", "password"):
        if key in masked and masked[key]:
            masked[key] = _mask_value(str(masked[key]))
    return masked


@router.post("/auth-connections", status_code=status.HTTP_201_CREATED)
async def create_auth_connection(
    body: AuthConnectionCreate,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    repo = PgAuthConnectionRepository(session)
    row = await repo.create(
        user_id=user_id,
        label=body.label.strip(),
        type=body.type,
        config_json=body.config_json,
    )
    await session.commit()
    return AuthConnectionResponse(
        id=str(row.id),
        label=row.label,
        type=row.type,
        config_json=_mask_auth_config(row.config_json),
        created_at=row.created_at.isoformat() if row.created_at else None,
        updated_at=row.updated_at.isoformat() if row.updated_at else None,
    )


@router.delete("/auth-connections/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_auth_connection(
    connection_id: str,
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_async_session),
):
    repo = PgAuthConnectionRepository(session)
    if not await repo.delete(uuid.UUID(connection_id), user_id):
        raise HTTPException(status_code=404, detail="Auth connection not found")
    await session.commit()


@router.post("/tools/validate")
async def validate_tool_config(body: ToolValidateRequest) -> ToolValidateResponse:
    errors: list[str] = []
    if body.kind in ("http", "webhook"):
        try:
            cfg = parse_tool_config(body.config_json)
            if not cfg.url.strip():
                errors.append("URL is required")
        except Exception as exc:
            errors.append(str(exc))
    elif body.kind == "custom":
        errors.append("Custom tools are not yet supported at runtime")
    return ToolValidateResponse(valid=len(errors) == 0, errors=errors)
