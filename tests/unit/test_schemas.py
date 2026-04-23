"""
Unit tests for Pydantic schemas used across the API and experiment tracking.

Tests cover:
- apps/api/schemas.py  (SearchRequest, PassageResult, ExperimentConfig, MetricSet)
- core/schemas/experiment.py  (MetricsResult, ExperimentConfig)
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from apps.api.schemas import (
    ExperimentConfig,
    MetricSet,
    PassageResult,
    SearchMode,
    SearchRequest,
)


# ---------------------------------------------------------------------------
# SearchRequest
# ---------------------------------------------------------------------------


class TestSearchRequest:
    def test_defaults_top_k_is_10(self):
        req = SearchRequest(query="machine learning")
        assert req.top_k == 10

    def test_defaults_mode_is_hybrid(self):
        req = SearchRequest(query="machine learning")
        assert req.mode == SearchMode.HYBRID

    def test_accepts_valid_modes(self):
        for mode in ("bm25", "dense", "hybrid", "rerank"):
            req = SearchRequest(query="test", mode=mode)
            assert req.mode.value == mode

    def test_invalid_mode_raises_validation_error(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", mode="invalid_mode")

    def test_empty_query_raises_validation_error(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_query_too_long_raises_validation_error(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="x" * 2049)

    def test_query_stripped_of_whitespace(self):
        req = SearchRequest(query="  hello world  ")
        assert req.query == "hello world"

    def test_top_k_minimum_1(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=0)

    def test_top_k_maximum_100(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=101)

    def test_valid_top_k(self):
        req = SearchRequest(query="test", top_k=50)
        assert req.top_k == 50

    def test_retrieval_multiplier_default(self):
        req = SearchRequest(query="test")
        assert req.retrieval_multiplier == 5

    def test_custom_filters_accepted(self):
        req = SearchRequest(query="test", filters={"source": "wiki"})
        assert req.filters["source"] == "wiki"


# ---------------------------------------------------------------------------
# PassageResult
# ---------------------------------------------------------------------------


class TestPassageResult:
    def test_score_can_be_zero(self):
        p = PassageResult(passage_id="1", text="text", title="title", score=0.0, rank=1)
        assert p.score == 0.0

    def test_score_can_be_negative(self):
        # PassageResult does not constrain score to [0,1]
        p = PassageResult(passage_id="1", text="text", title="title", score=-0.5, rank=1)
        assert p.score == -0.5

    def test_score_optional_defaults(self):
        # score has no default — must be provided
        with pytest.raises(ValidationError):
            PassageResult(passage_id="1", text="text", title="title", rank=1)

    def test_metadata_defaults_to_empty_dict(self):
        p = PassageResult(passage_id="1", text="text", title="title", score=0.5, rank=1)
        assert p.metadata == {}

    def test_metadata_accepts_arbitrary_keys(self):
        p = PassageResult(
            passage_id="1",
            text="text",
            title="title",
            score=0.5,
            rank=1,
            metadata={"source": "wiki", "date": "2024-01-01"},
        )
        assert p.metadata["source"] == "wiki"

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            PassageResult(text="text", title="title", score=0.5, rank=1)  # missing passage_id


# ---------------------------------------------------------------------------
# ExperimentConfig (apps.api.schemas)
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def _make_config(self, **kwargs):
        defaults = {
            "name": "test-experiment",
            "mode": SearchMode.HYBRID,
            "dataset": "ms_marco_dev_small",
            "top_k": 10,
        }
        defaults.update(kwargs)
        return ExperimentConfig(**defaults)

    def test_serialisation_to_json(self):
        config = self._make_config()
        json_str = config.model_dump_json()
        data = json.loads(json_str)
        assert data["name"] == "test-experiment"
        assert data["top_k"] == 10

    def test_deserialisation_from_json(self):
        config = self._make_config()
        json_str = config.model_dump_json()
        restored = ExperimentConfig.model_validate_json(json_str)
        assert restored.name == config.name
        assert restored.top_k == config.top_k
        assert restored.mode == config.mode

    def test_default_mode(self):
        config = ExperimentConfig(name="x")
        assert config.mode == SearchMode.HYBRID

    def test_extra_field_accepted(self):
        config = self._make_config(extra={"custom_key": "custom_value"})
        assert config.extra["custom_key"] == "custom_value"

    def test_round_trip_preserves_all_fields(self):
        config = self._make_config(
            name="my-exp",
            description="a description",
            mode=SearchMode.BM25,
            dataset="custom_dataset",
            top_k=25,
            retrieval_multiplier=3,
        )
        data = config.model_dump()
        restored = ExperimentConfig(**data)
        assert restored.name == "my-exp"
        assert restored.top_k == 25
        assert restored.mode == SearchMode.BM25


# ---------------------------------------------------------------------------
# MetricSet
# ---------------------------------------------------------------------------


class TestMetricSet:
    def _make_valid(self, **kwargs):
        return MetricSet(**kwargs)

    def test_defaults_all_zero(self):
        m = MetricSet()
        assert m.ndcg_at_10 == 0.0
        assert m.map_score == 0.0
        assert m.mrr_at_10 == 0.0

    def test_valid_ndcg_in_range(self):
        m = MetricSet(ndcg_at_10=0.75)
        assert m.ndcg_at_10 == 0.75

    def test_latency_fields_accepted(self):
        m = MetricSet(mean_latency_ms=42.5, p95_latency_ms=120.3, p99_latency_ms=200.0)
        assert m.mean_latency_ms == 42.5

    def test_serialisation(self):
        m = MetricSet(ndcg_at_10=0.6, map_score=0.55, mrr_at_10=0.7)
        data = m.model_dump()
        assert data["ndcg_at_10"] == 0.6
        assert "map_score" in data

    def test_all_fields_present_in_dump(self):
        m = MetricSet()
        data = m.model_dump()
        for key in ["ndcg_at_10", "mrr_at_10", "map_score", "recall_at_10", "precision_at_10"]:
            assert key in data


# ---------------------------------------------------------------------------
# core/schemas/experiment.py — MetricsResult
# ---------------------------------------------------------------------------


class TestMetricsResult:
    """Tests for the stricter MetricsResult in core/schemas/experiment.py."""

    from core.schemas.experiment import MetricsResult

    def _make_valid(self, **overrides):
        from core.schemas.experiment import MetricsResult

        defaults = {
            "ndcg_10": 0.5,
            "mrr_10": 0.6,
            "map": 0.45,
            "recall_k": {"10": 0.7, "100": 0.9},
            "precision_k": {"10": 0.3},
            "latency_p50": 50.0,
            "latency_p95": 100.0,
            "latency_p99": 200.0,
        }
        defaults.update(overrides)
        return MetricsResult(**defaults)

    def test_ndcg_must_be_in_0_1(self):
        from core.schemas.experiment import MetricsResult
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            self._make_valid(ndcg_10=1.5)

    def test_ndcg_exactly_0_accepted(self):
        m = self._make_valid(ndcg_10=0.0)
        assert m.ndcg_10 == 0.0

    def test_ndcg_exactly_1_accepted(self):
        m = self._make_valid(ndcg_10=1.0)
        assert m.ndcg_10 == 1.0

    def test_recall_values_must_be_in_unit_interval(self):
        from core.schemas.experiment import MetricsResult
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            self._make_valid(recall_k={"10": 1.5})

    def test_latency_percentile_ordering_enforced(self):
        from core.schemas.experiment import MetricsResult
        from pydantic import ValidationError as PydanticValidationError

        # p50 > p95 should fail
        with pytest.raises(PydanticValidationError):
            self._make_valid(latency_p50=200.0, latency_p95=100.0, latency_p99=300.0)

    def test_valid_metrics_result(self):
        m = self._make_valid()
        assert m.ndcg_10 == 0.5
        assert m.recall_k["10"] == 0.7
