"""FastAPI application factory and middleware configuration."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from neural_process_twin.api.deployment_config import SETTINGS
from neural_process_twin.api.middleware import ResourceSafetyMiddleware
from neural_process_twin.api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context managing startup warmup and graceful shutdown."""
    logger.info("Initializing Neural Process Twin application lifecycle...")
    try:
        from neural_process_twin.api.routes import get_experimental_surrogate, get_solver

        get_solver()
        get_experimental_surrogate()
        logger.info("Scientific solvers and neural surrogate models initialized successfully.")
    except Exception as exc:
        logger.warning("Startup pre-warmup encountered issue (will initialize lazily): %s", exc)
    yield
    logger.info("Shutting down Neural Process Twin application.")


def create_app(static_dir: str | Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application for Neural Process Twin."""
    app = FastAPI(
        title="Neural Process Twin — Scientific Digital Twin API",
        description=(
            "Research-grade inference and physical diagnostic API exposing the conservative "
            "finite-volume solver, frozen FNO-2D ensemble, split-conformal uncertainty estimators, "
            "and fail-closed hybrid safety screening."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # 1. Enforce payload size limits, request ID injection, and error sanitization
    app.add_middleware(ResourceSafetyMiddleware)

    # 2. Configure CORS securely for deployment
    allow_wildcard = SETTINGS.cors_allowed_origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=SETTINGS.cors_allowed_origins,
        allow_credentials=not allow_wildcard,  # Wildcard credentials violate browser CORS standards
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router)

    # Mount static frontend build if present
    resolved_static_dir = Path("frontend/dist") if static_dir is None else Path(static_dir)

    if resolved_static_dir.is_dir():
        logger.info("Mounting frontend static assets from %s", resolved_static_dir)
        app.mount("/", StaticFiles(directory=str(resolved_static_dir), html=True), name="frontend")
    else:
        logger.info(
            "No frontend dist found at %s. API docs available at /docs", resolved_static_dir
        )

        @app.get("/")
        def root_info() -> dict[str, str]:
            return {
                "message": "Neural Process Twin API running. Frontend static build not detected.",
                "docs_url": "/docs",
                "health_url": "/api/health",
                "capabilities_url": "/api/capabilities",
            }

    return app
