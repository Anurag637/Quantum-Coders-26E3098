import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

from backend.app.core.config import get_settings


settings = get_settings()
logger = logging.getLogger(__name__)


def init_telemetry(app, engine: Optional[object] = None):
    """
    Initialize OpenTelemetry tracing.

    - Instruments FastAPI
    - Instruments HTTP clients
    - Instruments SQLAlchemy (optional)
    - Supports OTLP exporter
    """

    if not settings.ENABLE_TRACING:
        logger.info("Telemetry disabled.")
        return

    resource = Resource.create(
        {
            "service.name": settings.APP_NAME,
            "service.version": settings.VERSION,
            "deployment.environment": settings.APP_ENV,
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    # Choose exporter
    if settings.APP_ENV == "development":
        exporter = ConsoleSpanExporter()
        logger.info("Using console telemetry exporter.")
    else:
        exporter = OTLPSpanExporter()
        logger.info("Using OTLP telemetry exporter.")

    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    # Instrument HTTP clients
    HTTPXClientInstrumentor().instrument()

    # Instrument SQLAlchemy (if provided)
    if engine:
        SQLAlchemyInstrumentor().instrument(engine=engine)

    logger.info("Telemetry initialized successfully.")


def get_tracer(name: str = "llm-gateway"):
    """
    Get a tracer instance for manual spans.
    """
    return trace.get_tracer(name)