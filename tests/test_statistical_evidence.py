import json

import pytest

from researchclaw.pipeline.statistical_evidence import paired_comparisons, seed_groups


def test_reused_seeds_do_not_overwrite_regimes():
    groups = seed_groups({"A/easy/0/accuracy": 0.9, "A/hard/0/accuracy": 0.4,
                          "A/easy/0/auroc": 0.7}, "accuracy")
    assert groups == {"A/easy": {0: 0.9}, "A/hard": {0: 0.4}}


def test_baseline_is_declared_and_seeds_are_paired_per_stratum():
    groups = {"Z/base": {0: 1, 1: 2, 2: 3}, "A/base": {0: 2, 1: 4, 2: 6},
              "Other/base": {8: 10, 9: 11}, "Z/shift": {0: 10, 1: 11},
              "A/shift": {0: 12, 1: 13}}
    assert paired_comparisons(groups, "") == []
    comparisons = paired_comparisons(groups, "Z")
    assert len(comparisons) == 2
    assert all(r["baseline"] == "Z" and r["method"] == "A" for r in comparisons)
    renamed = {k.replace("Z/", "B/"): v for k, v in groups.items()}
    assert [r["mean_diff"] for r in paired_comparisons(renamed, "B")] == [r["mean_diff"] for r in comparisons]


@pytest.mark.parametrize("offset,status,p", [(0, "identical", 1.0), (3, "constant_nonzero_difference", None)])
def test_degenerate_differences_are_not_zero_t_for_nonzero_effect(offset, status, p):
    result = paired_comparisons({"control": {i: i for i in range(3)},
                                 "method": {i: i + offset for i in range(3)}}, "control")[0]
    assert result["status"] == status
    assert result["p_value"] == p and result["t_stat"] is None
    assert result["mean_diff"] == offset
    json.dumps(result, allow_nan=False)


def test_duplicate_seed_spelling_is_not_silently_overwritten():
    with pytest.raises(ValueError, match="Duplicate"):
        seed_groups({"A/seed_1/acc": 1, "A/1/acc": 2}, "acc")
