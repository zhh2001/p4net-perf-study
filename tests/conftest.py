"""pytest configuration: gate ``integration`` / ``requires_p4c`` /
``requires_bmv2`` markers behind explicit CLI flags so the default
``pytest`` invocation only runs unit tests.

Same shape as p4net's conftest — flag names are stable across both
projects.
"""

from __future__ import annotations

import shutil

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked ``integration`` (require root/netns).",
    )
    parser.addoption(
        "--run-p4c",
        action="store_true",
        default=False,
        help="Run tests marked ``requires_p4c`` (need the p4c binary).",
    )
    parser.addoption(
        "--run-bmv2",
        action="store_true",
        default=False,
        help="Run tests marked ``requires_bmv2`` (need simple_switch_grpc).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_integration = pytest.mark.skip(reason="needs --run-integration")
    skip_p4c = pytest.mark.skip(reason="needs --run-p4c (or p4c missing on PATH)")
    skip_bmv2 = pytest.mark.skip(reason="needs --run-bmv2 (or simple_switch_grpc missing on PATH)")

    have_p4c = shutil.which("p4c") is not None
    have_bmv2 = shutil.which("simple_switch_grpc") is not None

    for item in items:
        if "integration" in item.keywords and not config.getoption("--run-integration"):
            item.add_marker(skip_integration)
        if "requires_p4c" in item.keywords and (not config.getoption("--run-p4c") or not have_p4c):
            item.add_marker(skip_p4c)
        if "requires_bmv2" in item.keywords and (
            not config.getoption("--run-bmv2") or not have_bmv2
        ):
            item.add_marker(skip_bmv2)
