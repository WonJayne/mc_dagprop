from __future__ import annotations

import math

import numpy as np
import pytest

import mc_dagprop as mc
from mc_dagprop.analytic import exponential_pmf, gamma_pmf


@pytest.mark.parametrize("scale,start,stop", [(0.05, 0.0, 1.5), (0.2, 0.3, 2.7), (2.0, 1000.2, 1003.7), (1e15, 0.0, 2.3)])
def test_linear_exponential_preserves_conditional_mean(scale: float, start: float, stop: float) -> None:
    pmf = exponential_pmf(scale, 1, start, stop, quantization="linear")
    width = stop - start
    ratio = width / scale
    expected = start + (width * (0.5 - ratio / 12.0) if ratio < 1e-4 else scale - width / math.expm1(ratio))
    assert np.dot(pmf.values, pmf.probabilities) == pytest.approx(expected, abs=1e-11)
    pmf.validate()
    pmf.validate_alignment(1)


@pytest.mark.parametrize(
    "shape,scale,start,stop",
    [(1.0, 0.2, 0.0, 2.3), (2.5, 2.0, 0.2, 6.7), (2.0, 2.0, 2000.2, 2004.7), (200.0, 2.0, 0.0, 3.3)],
)
def test_linear_gamma_preserves_conditional_mean(shape: float, scale: float, start: float, stop: float) -> None:
    pmf = gamma_pmf(shape, scale, 1, start, stop, quantization="linear")
    # Independent numerical integration, including a vanishing lower-tail
    # interval and an upper tail whose absolute probability underflows.
    points, weights = np.polynomial.legendre.leggauss(160)
    values = start + (points + 1.0) * (stop - start) / 2.0
    log_density = (shape - 1.0) * np.log(values) - values / scale
    density = np.exp(log_density - np.max(log_density))
    expected = np.dot(weights, values * density) / np.dot(weights, density)
    assert np.dot(pmf.values, pmf.probabilities) == pytest.approx(expected, rel=1e-10)
    pmf.validate()


def test_original_family_base_is_independent_of_aligned_duration() -> None:
    registry = mc.DelayFamilyRegistry(continuous_quantization="linear")
    registry.add_exponential(1, 0.05, 1.5)
    activity = mc.Activity(0, 0.0, 1)
    pmf = registry.increment_pmf(activity, 1, base_duration=0.4)
    assert np.dot(pmf.values, pmf.probabilities) == pytest.approx(0.02, abs=1e-12)
    context = mc.PropagationContext(
        (mc.Event("a", mc.EventTimestamp(0, 100, 0)), mc.Event("b", mc.EventTimestamp(0, 100, 0))),
        {(0, 1): activity}, ((1, ((0, 0),)),),
    )
    result = mc.analytic_from_context(context, registry, step=1, underflow_rule=mc.UnderflowRule.TRUNCATE,
                                    overflow_rule=mc.OverflowRule.TRUNCATE, activity_base_durations={0: 0.4}).run()
    assert np.dot(result[1].pmf.values, result[1].pmf.probabilities) == pytest.approx(0.02, abs=1e-12)
    with pytest.raises(ValueError, match="unknown activity"):
        mc.analytic_from_context(context, registry, step=1, underflow_rule=mc.UnderflowRule.TRUNCATE,
                                 overflow_rule=mc.OverflowRule.TRUNCATE, activity_base_durations={1: 0.4})


def test_linear_constant_retains_original_fractional_base_mean() -> None:
    registry = mc.DelayFamilyRegistry(continuous_quantization="linear")
    registry.add_constant(1, 0.5)
    pmf = registry.increment_pmf(mc.Activity(0, 0.0, 1), 1, base_duration=0.4)
    assert pmf.values.tolist() == [0.0, 1.0]
    assert np.dot(pmf.values, pmf.probabilities) == pytest.approx(0.2)


def test_linear_chain_restores_mean_but_retains_quantization_variance() -> None:
    count = 100
    context = mc.PropagationContext(
        tuple(mc.Event(str(index), mc.EventTimestamp(2.0 * index, 1000, 2.0 * index)) for index in range(count + 1)),
        {(index, index + 1): mc.Activity(index, 2, 1) for index in range(count)},
        tuple((index + 1, ((index, index),)) for index in range(count)),
    )
    means: dict[str, float] = {}
    quantiles: dict[str, float] = {}
    for mode in ("floor", "linear"):
        registry = mc.DelayFamilyRegistry(continuous_quantization=mode)
        registry.add_exponential(1, 0.05, 1.5)
        result = mc.analytic_from_context(context, registry, step=1, underflow_rule=mc.UnderflowRule.TRUNCATE,
                                         overflow_rule=mc.OverflowRule.TRUNCATE).run()[-1].pmf
        means[mode] = float(np.dot(result.values - 2 * count, result.probabilities))
        quantiles[mode] = float(result.values[np.searchsorted(np.cumsum(result.probabilities), 0.95)] - 2 * count)
    simulator = mc.monte_carlo_from_context(context, registry)
    samples = np.array([simulator.run(seed).realized[-1] - 2 * count for seed in range(3333)])
    assert means["floor"] < 0.005
    assert means["linear"] == pytest.approx(10.0, abs=1e-9)
    assert samples.mean() == pytest.approx(means["linear"], abs=0.1)
    # Linear projection fixes first moments, not the entire distribution.
    assert quantiles["linear"] > np.quantile(samples, 0.95) + 2.0
    print(f"chain: floor mean={means['floor']:.6f}, P95={quantiles['floor']}; "
          f"linear mean={means['linear']:.6f}, P95={quantiles['linear']}; "
          f"MC mean={samples.mean():.6f}, P95={np.quantile(samples, 0.95):.6f}")


def test_invalid_quantization_is_rejected() -> None:
    with pytest.raises(ValueError, match="quantization"):
        mc.DelayFamilyRegistry(continuous_quantization="nearest")
    with pytest.raises(ValueError, match="quantization"):
        exponential_pmf(1, 1, 0, 2, quantization="nearest")


def test_floor_equivalence_contract_rejects_linear_mode() -> None:
    registry = mc.DelayFamilyRegistry(continuous_quantization="linear")
    registry.add_exponential(1, 0.05, 1.5)
    context = mc.PropagationContext(
        (mc.Event("a", mc.EventTimestamp(0, 100, 0)), mc.Event("b", mc.EventTimestamp(0, 100, 0))),
        {(0, 1): mc.Activity(0, 2, 1)}, ((1, ((0, 0),)),),
    )
    with pytest.raises(mc.EquivalenceDomainError, match="floor quantization"):
        mc.validate_equivalence_domain(context, registry, step=1, mode=mc.EquivalenceMode.QUANTIZED_CONTINUOUS)
