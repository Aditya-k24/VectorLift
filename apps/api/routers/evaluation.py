"""
Evaluation router – endpoints to trigger evaluation runs and query results.

Endpoints
---------
POST /evaluate                              – kick off an async evaluation job.
GET  /experiments                           – list all experiment results.
GET  /experiments/{experiment_id}           – fetch a single experiment.
GET  /experiments/{experiment_id}/compare  – compare two experiments.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, status

from apps.api.schemas import (
    EvaluationJobResponse,
    ExperimentComparison,
    ExperimentConfig,
    ExperimentResult,
)
from apps.api.services.evaluation_service import EvaluationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


# ---------------------------------------------------------------------------
# Helper: build EvaluationService from app state
# ---------------------------------------------------------------------------


def _get_eval_service(request: Request) -> EvaluationService:
    pipeline = getattr(request.app.state, "search_pipeline", None)
    db_factory = getattr(request.app.state, "db_session_factory", None)
    return EvaluationService(pipeline=pipeline, db_session_factory=db_factory)


# ---------------------------------------------------------------------------
# POST /evaluate
# ---------------------------------------------------------------------------


@router.post(
    "/evaluate",
    response_model=EvaluationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger an evaluation run",
    description=(
        "Starts an asynchronous evaluation job over a benchmark dataset. "
        "Returns a job_id immediately; poll GET /experiments to see when it completes."
    ),
)
async def trigger_evaluation(
    config: ExperimentConfig,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> EvaluationJobResponse:
    service = _get_eval_service(http_request)

    import uuid
    job_id = str(uuid.uuid4())

    async def _run_job() -> None:
        try:
            result = await service.run_evaluation(config)
            logger.info(
                "eval.job.done",
                extra={"job_id": job_id, "experiment_id": result.experiment_id},
            )
        except Exception:
            logger.exception("eval.job.failed", extra={"job_id": job_id})

    background_tasks.add_task(asyncio.ensure_future, _run_job())

    return EvaluationJobResponse(
        job_id=job_id,
        status="pending",
        message=(
            f"Evaluation job '{config.name}' started. "
            "Check GET /experiments for results."
        ),
    )


# ---------------------------------------------------------------------------
# GET /experiments
# ---------------------------------------------------------------------------


@router.get(
    "/experiments",
    response_model=list[ExperimentResult],
    summary="List all experiments",
    description="Returns all completed experiment results, newest first.",
)
async def list_experiments(http_request: Request) -> list[ExperimentResult]:
    service = _get_eval_service(http_request)
    return await service.list_experiments()


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}
# ---------------------------------------------------------------------------


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentResult,
    summary="Get a single experiment",
)
async def get_experiment(experiment_id: str, http_request: Request) -> ExperimentResult:
    service = _get_eval_service(http_request)
    result = await service.get_experiment(experiment_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found.",
        )
    return result


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}/compare
# ---------------------------------------------------------------------------


@router.get(
    "/experiments/{experiment_id}/compare",
    response_model=ExperimentComparison,
    summary="Compare two experiments",
    description=(
        "Compare experiment *experiment_id* (baseline) against *candidate_id*. "
        "Returns per-metric deltas and paired t-test significance results."
    ),
)
async def compare_experiments(
    experiment_id: str,
    candidate_id: str = Query(..., description="ID of the candidate experiment to compare against."),
    http_request: Request = ...,  # type: ignore[assignment]
) -> ExperimentComparison:
    service = _get_eval_service(http_request)
    comparison = await service.compare_experiments(experiment_id, candidate_id)
    if comparison is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Could not compare: one or both experiments not found "
                f"(baseline='{experiment_id}', candidate='{candidate_id}')."
            ),
        )
    return comparison
