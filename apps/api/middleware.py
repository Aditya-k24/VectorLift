"""
Custom Starlette middleware for VectorLift API.

Provides:
- RequestIDMiddleware  – attaches a unique X-Request-ID to every request/response.
- TimingMiddleware     – measures wall-clock processing time and returns it as
                         X-Response-Time (milliseconds).
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a UUID-based request identifier to every request/response.

    The middleware first checks for an incoming ``X-Request-ID`` header so that
    clients or load-balancers that already set one have their value honoured.
    Otherwise a fresh UUID4 is generated.  The final value is propagated via:

    * ``request.state.request_id``  – accessible throughout the handler chain.
    * ``X-Request-ID`` response header.
    """

    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id: str = request.headers.get(
            self.header_name, str(uuid.uuid4())
        )
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers[self.header_name] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Measure and expose the total request processing time.

    The elapsed time (wall-clock, milliseconds) is returned in the
    ``X-Response-Time`` response header.  The value is formatted as an integer
    number of milliseconds (e.g. ``"42ms"``).
    """

    HEADER_NAME: str = "X-Response-Time"

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start: float = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms: int = int((time.perf_counter() - start) * 1_000)
        response.headers[self.HEADER_NAME] = f"{elapsed_ms}ms"
        return response
