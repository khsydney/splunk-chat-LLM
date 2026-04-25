# sitecustomize.py
import os
import logging

# ---------- Helpful defaults ----------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")
# Capture prompts/choices on OpenAI spans (otlp attr size—dev only)
os.environ.setdefault("OTEL_INSTRUMENTATION_OPENAI_CAPTURE_PROMPTS", "true")

OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
OTLP_INSECURE = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"

SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "chat-rag")
DEPLOY_ENV   = os.getenv("DEPLOY_ENV", "Nick-LLM")

# ---------- One Resource for everything ----------
from opentelemetry.sdk.resources import Resource
resource = Resource.create({
    "service.name": SERVICE_NAME,
    "deployment.environment": DEPLOY_ENV,
})

# ---------- Traces ----------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON
    # gRPC exporter for 4317
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    tp = TracerProvider(resource=resource, sampler=ALWAYS_ON)
    tp.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=OTLP_INSECURE)
        )
    )
    trace.set_tracer_provider(tp)
except Exception:
    pass

# ---------- Metrics (optional but nice) ----------
try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

    metric_exporter = OTLPMetricExporter(endpoint=OTLP_ENDPOINT, insecure=OTLP_INSECURE)
    reader = PeriodicExportingMetricReader(metric_exporter)
    mp = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(mp)
except Exception:
    pass

# ---------- Logs (so AI Events land in Splunk) ----------
try:
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    # gRPC logs exporter for 4317
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

    lp = LoggerProvider(resource=resource)
    set_logger_provider(lp)
    lp.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(endpoint=OTLP_ENDPOINT, insecure=OTLP_INSECURE)
        )
    )
except Exception:
    pass

# ---------- Correlate std logging with traces ----------
try:
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    LoggingInstrumentor().instrument(set_logging_format=True, log_level=logging.INFO)
except Exception:
    pass

# ---------- Traceloop (let it handle LangChain/OpenAI only) ----------
# USE_TRACELOOP = True
# if USE_TRACELOOP:
#     try:
#         from traceloop.sdk import Traceloop
#         from traceloop.sdk.instruments import Instruments
#         Traceloop.init(
#             app_name=SERVICE_NAME,
#             resource_attributes={"deployment.environment": DEPLOY_ENV},
#             instruments={Instruments.LANGCHAIN, Instruments.OPENAI},
#             disable_batch=True,  # dev immediate flush
#         )
#     except Exception:
#         pass

# ---------- Helper to instrument safely ----------
def _safe(instr_cls, method="instrument", **kw):
    try:
        getattr(instr_cls(), method)(**kw)
    except Exception:
        pass

# ---------- Infra + Milvus (keep these) ----------
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    _safe(FastAPIInstrumentor)
except Exception:
    pass

try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    _safe(HTTPXClientInstrumentor)
except Exception:
    pass

try:
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    _safe(RequestsInstrumentor)
except Exception:
    pass

try:
    from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
    _safe(GrpcInstrumentorClient)
except Exception:
    pass

try:
    from opentelemetry.instrumentation.milvus import MilvusInstrumentor
    _safe(MilvusInstrumentor)
except Exception:
    pass
