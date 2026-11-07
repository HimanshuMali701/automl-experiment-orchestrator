"""FastAPI application factory, global exception handling, and middleware setup."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from automl import __version__
from automl.api.jobs import get_job_manager
from automl.api.routes import router
from automl.core.config import Settings, get_settings
from automl.core.exceptions import AutoMLError
from automl.core.logging import get_logger

logger = get_logger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and graceful shutdown."""
    settings = get_settings()
    settings.ensure_directories()
    logger.info(f"Starting {settings.app_name} v{__version__} API service...")
    yield
    logger.info("Stopping API service; terminating background worker pool...")
    get_job_manager().shutdown()


def create_app(settings: Settings | None = None, enable_lifespan: bool = True) -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    cfg = settings or get_settings()

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        description=(
            "Automated tabular machine learning experimentation platform API. "
            "Supports dataset profiling, task detection, stateful workflow orchestration, "
            "leakage-free model benchmarking, and MLflow experiment tracking."
        ),
        lifespan=lifespan if enable_lifespan else None,
    )

    # 1. CORS Middleware (allows local frontend dashboard communication)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Standardized Exception Handlers (no stack traces to clients)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "error_type": "HTTPException"},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        error_details = []
        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "Invalid parameter")
            error_details.append(f"{loc}: {msg}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "; ".join(error_details), "error_type": "RequestValidationError"},
        )

    @app.exception_handler(AutoMLError)
    async def automl_exception_handler(request: Request, exc: AutoMLError) -> JSONResponse:
        logger.warning(f"Business logic exception on {request.method} {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc), "error_type": exc.__class__.__name__},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Full stack trace logged ONLY internally on server side
        logger.exception(f"Unhandled server error processing {request.method} {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred while processing your request. Please check server logs.",
                "error_type": "InternalServerError",
            },
        )

    # 3. Mount Routes
    app.include_router(router)

    return app


# Application entrypoint
app = create_app()

