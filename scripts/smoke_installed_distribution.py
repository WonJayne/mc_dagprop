"""Smoke-test an installed distribution without importing from the checkout."""

from __future__ import annotations

import argparse
import importlib.metadata
import tomllib
from pathlib import Path

import mc_dagprop as mc


def _expected_version(source_root: Path) -> str:
    with (source_root / "pyproject.toml").open("rb") as pyproject_file:
        return str(tomllib.load(pyproject_file)["project"]["version"])


def _assert_distribution_metadata() -> None:
    distribution = importlib.metadata.distribution("mc-dagprop")
    packaged_files = {str(path) for path in distribution.files or ()}
    assert any(path.endswith("mc_dagprop/py.typed") for path in packaged_files)
    assert any(path.endswith("mc_dagprop/monte_carlo/_core.pyi") for path in packaged_files)
    assert any(path.endswith("THIRD_PARTY_NOTICES.md") for path in packaged_files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    source_root = args.source_root.resolve()

    module_path = Path(mc.__file__).resolve()
    assert not module_path.is_relative_to(source_root), f"imported mc_dagprop from checkout: {module_path}"
    assert mc.__version__ == _expected_version(source_root)
    _assert_distribution_metadata()

    events = (mc.Event("root", mc.EventTimestamp(0, 100, 0)), mc.Event("child", mc.EventTimestamp(0, 100, 0)))
    activities = {(0, 1): mc.Activity(0, 10, 1)}
    precedence = ((1, ((0, 0),)),)
    context = mc.PropagationContext(events, activities, precedence)
    registry = mc.DelayFamilyRegistry()
    registry.add_empirical(1, [0, 5], [1, 0])

    analytic = mc.AnalyticPropagator.from_context(
        context, registry, step=1, underflow_rule=mc.UnderflowRule.TRUNCATE, overflow_rule=mc.OverflowRule.TRUNCATE
    )
    analytic_result = analytic.run()
    nonzero_mass = {
        value: probability
        for value, probability in zip(
            analytic_result[1].pmf.values.tolist(), analytic_result[1].pmf.probabilities.tolist(), strict=True
        )
        if probability > 0
    }
    assert nonzero_mass == {10.0: 1.0}

    monte_carlo = mc.MonteCarloPropagator.from_context(context, registry)
    monte_carlo_result = monte_carlo.run(123)
    assert float(monte_carlo_result.realized[1]) == 10.0
    assert float(monte_carlo_result.durations[0]) == 10.0
    print(f"installed mc-dagprop {mc.__version__} smoke test passed")


if __name__ == "__main__":
    main()
