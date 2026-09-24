"""Enable Luna-2 evaluators (plus LLM-judge twins) on the demo's Agent Stream.

    python scripts/enable_luna_evaluators.py                 # full set (needs on-prem / Enterprise)
    python scripts/enable_luna_evaluators.py --saas          # only the 4 Luna evaluators SaaS supports
    python scripts/enable_luna_evaluators.py --only prompt_injection_luna input_pii_luna

NOTE: `enable_evaluators` REPLACES the set currently enabled on the stream.
Evaluators run on *sampled* traces — set the sampling rule to 100% in the
Agent Stream's "Configure Evaluators" page for a like-for-like demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/x.py` from the repo root

import argparse

from app.jev import splunk_ao as splunk_side

SAAS_LUNA = ["prompt_injection_luna", "input_toxicity_luna", "output_toxicity_luna", "input_sexism_luna",
             "output_sexism_luna", "input_pii_luna", "output_pii_luna"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saas", action="store_true", help="limit to the Luna evaluators available in Observability Cloud SaaS")
    ap.add_argument("--only", nargs="*", help="explicit SplunkAOEvaluators member names")
    ap.add_argument("--no-llm-twins", action="store_true")
    args = ap.parse_args()

    only = args.only or (SAAS_LUNA if args.saas else None)
    names = splunk_side.enable_luna_evaluators(include_llm_judge_twins=not args.no_llm_twins, only=only)
    print("enabled:", ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
