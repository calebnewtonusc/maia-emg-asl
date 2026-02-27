"""API key authentication middleware for MAIA inference server."""
from __future__ import annotations

import os
import logging
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Paths that bypass auth
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


def _get_valid_keys() -> set[str]:
    raw = os.environ.get("MAIA_API_KEYS", "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    if not keys:
        default = "dev-key-change-me"
        logger.warning("No MAIA_API_KEYS set — using default dev key. DO NOT use in production.")
        keys = {default}
    return keys


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Check X-API-Key header for protected routes.

    Bypass:
      - Public paths (/health, /docs, etc.)
      - WebSocket upgrade requests
      - MAIA_DISABLE_AUTH=true env var
    """

    async def dispatch(self, request: Request, call_next):
        # Disable auth entirely for development
        if os.environ.get("MAIA_DISABLE_AUTH", "").lower() in ("1", "true", "yes"):
            return await call_next(request)

        # Skip public paths
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Skip WebSocket upgrades
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        # Validate API key
        key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if key not in _get_valid_keys():
            return JSONResponse(
                {"detail": "Invalid or missing API key"},
                status_code=401,
            )

        return await call_next(request)
