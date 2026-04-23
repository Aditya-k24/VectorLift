"""
Root conftest.py — applies to the entire test suite.

Responsibilities
----------------
1. Register custom pytest markers so ``--strict-markers`` does not fail.
2. Add a ``--runslow`` command-line option to gate slow tests.
3. Automatically skip tests marked ``slow`` unless ``--runslow`` is passed.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Custom command-line options
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Include tests marked with @pytest.mark.slow in this run.",
    )
    parser.addoption(
        "--rune2e",
        action="store_true",
        default=False,
        help="Include tests marked with @pytest.mark.e2e in this run.",
    )


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: Tests that require live external services "
        "(Elasticsearch, Qdrant, Postgres, Redis). "
        "Run with: pytest -m integration",
    )
    config.addinivalue_line(
        "markers",
        "e2e: Full end-to-end tests through the HTTP API or full pipeline. "
        "Run with: pytest -m e2e  or  pytest --rune2e",
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests that take more than 5 seconds. "
        "Run with: pytest --runslow",
    )
    config.addinivalue_line(
        "markers",
        "unit: Fast, self-contained unit tests with no external dependencies.",
    )


# ---------------------------------------------------------------------------
# Automatic skip logic
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip slow tests unless --runslow is passed; skip e2e unless --rune2e."""
    run_slow = config.getoption("--runslow", default=False)
    run_e2e = config.getoption("--rune2e", default=False)

    skip_slow = pytest.mark.skip(reason="Slow test skipped — pass --runslow to include.")
    skip_e2e = pytest.mark.skip(reason="E2E test skipped — pass --rune2e to include.")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "e2e" in item.keywords and not run_e2e:
            item.add_marker(skip_e2e)
