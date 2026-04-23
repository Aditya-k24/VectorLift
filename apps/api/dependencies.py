"""
FastAPI dependency injection providers for VectorLift.

All providers are thin wrappers around objects stored in ``app.state`` so the
same singletons are reused across every request.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Annotated, Any

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def get_settings(request: Request) -> Any:
    """Return the application :class:`~core.config.Settings` singleton."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application settings not initialised.",
        )
    return settings


SettingsDep = Annotated[Any, Depends(get_settings)]


# ---------------------------------------------------------------------------
# Search pipeline
# ---------------------------------------------------------------------------


def get_search_pipeline(request: Request) -> Any:
    """Return the :class:`~pipelines.SearchPipeline` singleton loaded at startup.

    Raises 503 if the pipeline was not initialised (e.g. models still loading).
    """
    pipeline = getattr(request.app.state, "search_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search pipeline not ready.  Try again shortly.",
        )
    return pipeline


SearchPipelineDep = Annotated[Any, Depends(get_search_pipeline)]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a single :class:`AsyncSession` per request, auto-closing on exit.

    The session factory is stored at ``app.state.db_session_factory``.
    """
    session_factory = getattr(request.app.state, "db_session_factory", None)
    if session_factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database session factory not initialised.",
        )
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DbSessionDep = Annotated[AsyncSession, Depends(get_db)]


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


def get_redis(request: Request) -> aioredis.Redis:  # type: ignore[type-arg]
    """Return the shared async Redis client stored in ``app.state.redis``."""
    client = getattr(request.app.state, "redis", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis client not initialised.",
        )
    return client  # type: ignore[return-value]


RedisDep = Annotated[aioredis.Redis, Depends(get_redis)]  # type: ignore[type-arg]
