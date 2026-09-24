"""Test bootstrap. Set PRESET_PROVIDER=1 to simulate an environment where a pytest plugin
(or an OTel distro) has already installed a global SDK TracerProvider before the tests import
the app — OpenTelemetry ignores later set_tracer_provider() calls, so the tests must cope."""
import os

if os.getenv("PRESET_PROVIDER") == "1":
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    trace.set_tracer_provider(TracerProvider())
