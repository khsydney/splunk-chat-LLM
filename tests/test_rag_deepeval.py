# tests/test_rag_deepeval.py
"""DeepEval-based RAG quality tests.

Run with:  pytest tests/test_rag_deepeval.py -v
Requires:  OPENAI_API_KEY set in the environment.
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

EVAL_MODEL = "gpt-4o-mini"
THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Sample test cases – replace / extend with real query/answer/context triples
# gathered from your Splunk documentation corpus.
# ---------------------------------------------------------------------------
_TEST_CASES = [
    LLMTestCase(
        input="What is a Splunk index?",
        actual_output=(
            "A Splunk index is a repository where Splunk stores all of its processed data. "
            "When data is ingested it is parsed, transformed, and written to the index in "
            "compressed format for fast search and retrieval."
        ),
        retrieval_context=[
            "An index is a Splunk repository for storing all processed machine data. "
            "Data is parsed, transformed, and written to the index in compressed format.",
            "Splunk indexes allow you to search, monitor, and analyze machine-generated data "
            "at scale by organizing data into named repositories called indexes.",
        ],
    ),
    LLMTestCase(
        input="How does Splunk handle search concurrency?",
        actual_output=(
            "Splunk manages search concurrency through search quotas and the scheduler. "
            "The search scheduler prioritizes scheduled searches and limits ad-hoc searches "
            "using per-user and system-wide concurrency caps defined in limits.conf."
        ),
        retrieval_context=[
            "Search concurrency is controlled by limits.conf settings such as max_searches_per_cpu "
            "and max_rt_search_multiplier. The search scheduler enforces these limits.",
            "Splunk's scheduler queues and prioritizes scheduled searches to prevent resource "
            "exhaustion. Ad-hoc searches are subject to per-user quotas.",
        ],
    ),
]


@pytest.mark.parametrize("test_case", _TEST_CASES)
def test_rag_answer_quality(test_case: LLMTestCase):
    """Each answer must pass answer relevancy, faithfulness, and contextual relevancy."""
    assert_test(
        test_case,
        [
            AnswerRelevancyMetric(model=EVAL_MODEL, threshold=THRESHOLD),
            FaithfulnessMetric(model=EVAL_MODEL, threshold=THRESHOLD),
            ContextualRelevancyMetric(model=EVAL_MODEL, threshold=THRESHOLD),
        ],
    )
