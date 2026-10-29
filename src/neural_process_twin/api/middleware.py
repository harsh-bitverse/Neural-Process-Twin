"""ASGI middleware components for request size bounding, observability, and error sanitization."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from neural_process_twin.api.deployment_config import (
    METRICS,
    SETTINGS,
    ConcurrencyLimitExceededError,
)

logger = logging.getLogger("neural_process_twin.api")


class ResourceSafetyMiddleware(BaseHTTPMiddleware):
    """Enforces request payload size limits, observability tracking, and error sanitization."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = req_id
        t0 = time.perf_counter()

        # 1. Enforce payload size limits before body reading/buffering
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                cl_val = int(content_length)
                if cl_val > SETTINGS.max_request_body_bytes:
                    METRICS.record_error("payload_too_large")
                    logger.warning(
                        "[%s] Request rejected: Content-Length %d exceeds limit of %d bytes",
                        req_id,
                        cl_val,
                        SETTINGS.max_request_body_bytes,
                    )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": (
                                f"Request payload size ({cl_val} bytes) exceeds the maximum "
                                f"allowed limit of {SETTINGS.max_request_body_bytes} bytes."
                            ),
                            "error_code": "PAYLOAD_TOO_LARGE",
                            "request_id": req_id,
                        },
                        headers={"X-Request-ID": req_id},
                    )
            except ValueError:
                pass

        # 2. Process request with observability and error envelope
        endpoint = request.url.path
        status_code = 500
        mode: str | None = None

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
        except ConcurrencyLimitExceededError as exc:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            METRICS.record_error("concurrency_limit_exceeded")
            METRICS.record_request(endpoint, mode, 503, duration_ms)
            logger.warning("[%s] Concurrency limit exceeded on %s: %s", req_id, endpoint, exc)
            return JSONResponse(
                status_code=503,
                content={
                    "detail": str(exc),
                    "error_code": "CONCURRENCY_LIMIT_EXCEEDED",
                    "request_id": req_id,
                },
                headers={
                    "X-Request-ID": req_id,
                    "X-Response-Time-Ms": f"{duration_ms:.2f}",
                    "Retry-After": "2",
                },
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            METRICS.record_error("unhandled_server_exception")
            METRICS.record_request(endpoint, mode, 500, duration_ms)
            logger.exception("[%s] Unhandled exception processing %s: %s", req_id, endpoint, exc)
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "An internal execution error occurred while processing the scientific simulation.",
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "request_id": req_id,
                },
                headers={
                    "X-Request-ID": req_id,
                    "X-Response-Time-Ms": f"{duration_ms:.2f}",
                },
            )

        duration_ms = (time.perf_counter() - t0) * 1000.0
        METRICS.record_request(endpoint, mode, status_code, duration_ms)

        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

        return response
