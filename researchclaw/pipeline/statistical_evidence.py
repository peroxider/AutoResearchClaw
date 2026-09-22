"""Deterministic seed-level summaries with explicit comparison identity.

Seeds describe training randomness, not independent datasets or subjects.
Regimes are never pooled. Missing scipy leaves p-values unavailable rather
than substituting a normal approximation for a small-sample t-test.
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Any


def seed_groups(metrics: dict[str, Any], metric: str) -> dict[str, dict[int, float]]:
    """Read method/[dataset/regime/...]/seed/metric without losing identity."""
    groups: dict[str, dict[int, float]] = {}
    for key, value in metrics.items():
        parts = key.split("/")
        if len(parts) < 3 or parts[-1] != metric or isinstance(value, bool):
            continue
        try:
            seed = int(parts[-2].removeprefix("seed_").removeprefix("seed"))
            number = float(value)
        except (ValueError, TypeError):
            continue
        if math.isfinite(number):
            group = "/".join(parts[:-2])
            if seed in groups.get(group, {}):
                raise ValueError(f"Duplicate seed identity: {group}/{seed}/{metric}")
            groups.setdefault(group, {})[seed] = number
    return groups


def summarize_seeds(values: dict[int, float]) -> dict[str, Any]:
    sample = [values[k] for k in sorted(values)]
    result: dict[str, Any] = {
        "mean": statistics.mean(sample), "n_seeds": len(sample),
        "std": statistics.stdev(sample) if len(sample) > 1 else None,
        "resampling_unit": "training_seed", "ci_method": "unavailable",
    }
    if len(sample) >= 3:
        rng = random.Random(42)
        boot = sorted(statistics.mean(rng.choices(sample, k=len(sample))) for _ in range(2000))
        result.update(ci95_low=boot[49], ci95_high=boot[1949],
                      ci_method="percentile_bootstrap", bootstrap_replicates=2000)
    return result


def paired_comparisons(groups: dict[str, dict[int, float]], baseline: str) -> list[dict[str, Any]]:
    """Compare each method only to a declared baseline in the same stratum.

Report constant differences as degenerate (no defined t statistic). Holm
adjustment is applied across the available nondegenerate tests. No baseline
declaration means no inferential comparison, regardless of method names.
    """
    if not baseline:
        return []
    results: list[dict[str, Any]] = []
    for group, samples in sorted(groups.items()):
        method, sep, stratum = group.partition("/")
        if method == baseline:
            continue
        control = groups.get(baseline + (sep + stratum if sep else ""), {})
        seeds = sorted(samples.keys() & control.keys())
        if len(seeds) < 2:
            continue
        differences = {s: samples[s] - control[s] for s in seeds}
        stats = summarize_seeds(differences)
        mean, std = stats["mean"], stats["std"]
        t_stat = None
        p_value = None
        status = "unavailable"
        if std == 0:
            status = "identical" if mean == 0 else "constant_nonzero_difference"
            p_value = 1.0 if mean == 0 else None
        else:
            t_stat = mean / (std / math.sqrt(len(seeds)))
            try:
                from scipy.stats import t
                p_value = float(2 * t.sf(abs(t_stat), len(seeds) - 1))
                status = "computed"
            except ImportError:
                pass
        results.append({
            "method": method, "baseline": baseline, "stratum": stratum,
            "mean_diff": mean, "std_diff": std, "t_stat": t_stat,
            "p_value": p_value, "n_seeds": len(seeds), "seeds": seeds,
            "test": "paired_t", "status": status, "source": "pipeline_computed",
            "resampling_unit": "training_seed", "effect_size": mean,
            "ci95_low": stats.get("ci95_low"), "ci95_high": stats.get("ci95_high"),
            "ci_method": stats["ci_method"], "multiple_comparison": "holm",
        })
    ordered = sorted((r for r in results if r["p_value"] is not None), key=lambda r: r["p_value"])
    previous = 0.0
    for i, result in enumerate(ordered):
        previous = max(previous, min(1.0, result["p_value"] * (len(ordered) - i)))
        result["p_value_adjusted"] = previous
    return results
