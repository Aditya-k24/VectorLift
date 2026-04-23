"""
VectorLiftClient – synchronous HTTP client for the Streamlit dashboard.

Uses ``requests`` (sync) because Streamlit's event model is single-threaded.
Implements retry logic via ``tenacity`` for transient connection errors.
"""

from __future__ import annotations

import logging
from typing import Any

import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30  # seconds
_MAX_RETRIES = 3
_WAIT_MIN = 0.5
_WAIT_MAX = 4.0


class VectorLiftClientError(Exception):
    """Raised when the API returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class VectorLiftClient:
    """Thin wrapper around the VectorLift REST API.

    Parameters
    ----------
    base_url:
        Root URL of the running FastAPI service, e.g. ``http://localhost:8000``.
    timeout:
        Request timeout in seconds (applied to every call).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(requests.exceptions.ConnectionError),
        stop=stop_after_attempt(_MAX_RETRIES),
        wait=wait_exponential(min=_WAIT_MIN, max=_WAIT_MAX),
        reraise=True,
    )
    def _get(self, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.get(url, timeout=self.timeout, **kwargs)
        except requests.exceptions.ConnectionError:
            logger.warning("Connection error reaching %s – retrying …", url)
            raise
        self._raise_for_status(resp)
        return resp.json()

    @retry(
        retry=retry_if_exception_type(requests.exceptions.ConnectionError),
        stop=stop_after_attempt(_MAX_RETRIES),
        wait=wait_exponential(min=_WAIT_MIN, max=_WAIT_MAX),
        reraise=True,
    )
    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError:
            logger.warning("Connection error reaching %s – retrying …", url)
            raise
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise VectorLiftClientError(status_code=resp.status_code, detail=str(detail))

    # ------------------------------------------------------------------
    # Search API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        retrieval_multiplier: int = 5,
    ) -> dict[str, Any]:
        """Call POST /search and return the parsed JSON response.

        Parameters
        ----------
        query:
            Natural-language search query.
        mode:
            One of ``bm25``, ``dense``, ``hybrid``, ``rerank``.
        top_k:
            Maximum number of results to retrieve.
        filters:
            Optional metadata filters forwarded to the retriever.
        retrieval_multiplier:
            Over-fetch multiplier used in rerank mode.
        """
        payload: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "retrieval_multiplier": retrieval_multiplier,
            "filters": filters or {},
        }
        return self._post("/search", payload)  # type: ignore[return-value]

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Call POST /search/rerank with caller-supplied candidates."""
        payload: dict[str, Any] = {
            "query": query,
            "candidates": candidates,
            "top_n": top_n,
        }
        return self._post("/search/rerank", payload)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Health / model info
    # ------------------------------------------------------------------

    def get_health(self) -> dict[str, Any]:
        """Call GET /health and return the parsed JSON response."""
        return self._get("/health")  # type: ignore[return-value]

    def get_model_info(self) -> dict[str, Any]:
        """Call GET /model-info and return the parsed JSON response."""
        return self._get("/model-info")  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Experiments / evaluation
    # ------------------------------------------------------------------

    def get_experiments(self) -> list[dict[str, Any]]:
        """Call GET /evaluation/experiments and return the list."""
        result = self._get("/evaluation/experiments")
        if isinstance(result, list):
            return result  # type: ignore[return-value]
        return []

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Call GET /evaluation/experiments/{id}."""
        return self._get(f"/evaluation/experiments/{experiment_id}")  # type: ignore[return-value]

    def compare_experiments(
        self, baseline_id: str, candidate_id: str
    ) -> dict[str, Any]:
        """Call GET /evaluation/experiments/{baseline_id}/compare?candidate_id=…"""
        return self._get(
            f"/evaluation/experiments/{baseline_id}/compare",
            params={"candidate_id": candidate_id},
        )  # type: ignore[return-value]

    def trigger_evaluation(self, config: dict[str, Any]) -> dict[str, Any]:
        """Call POST /evaluation/evaluate to kick off an async evaluation job."""
        return self._post("/evaluation/evaluate", config)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "VectorLiftClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
