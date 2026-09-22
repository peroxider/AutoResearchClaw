import copy

import pytest

from researchclaw.pipeline.manuscript import ManuscriptError, _result_tables, render_manuscript
from tests.test_manuscript import build, study
from tests.test_experiment_protocol import spec
from tests.test_research_inputs import inputs


def result(index=0, **changes):
    record = {"result_id": f"result-{index}", "key": {"dataset": "demo", "dataset_version": "v1",
        "split": "test", "method": "method_A", "config": "a" * 63 + str(index % 10),
        "seed": str(index), "metric": "accuracy", "aggregation": "per_seed", "regime": "default"},
        "unit": "fraction", "value": 0.0}
    record["key"].update(changes)
    return record


def assert_spans(md, tex, claims):
    for claim in claims:
        for filename, text in (("paper_final.md", md), ("paper.tex", tex)):
            span = claim["spans"][filename]
            assert text[span["start"]:span["end"]] == claim["rendered"]


@pytest.mark.parametrize("columns", [1, 2])
def test_table_aliases_preserve_colliding_full_hashes_and_exact_values(columns):
    records = [result(0), result(1)]
    records[1]["value"] = -0.000003
    records[1]["unit"] = records[0]["unit"] = "raw"
    md, tex, claims, following = _result_tables(records, columns=columns, first_table=7)
    assert following == 8 and len(claims) == 2
    assert {c["configuration_alias"] for c in claims} == {"C1", "C2"}
    assert {c["table"] for c in claims} == {"R7"}
    for record in records:
        assert record["key"]["config"] in md
        assert record["key"]["config"] in tex.replace(r"\allowbreak{}", "")
    assert "0.000000" in md and "-0.000003" in md
    assert_spans(md, tex, claims)


@pytest.mark.parametrize("field", ["dataset", "dataset_version", "split", "metric", "aggregation", "regime", "unit"])
def test_incompatible_contexts_never_share_a_table(field):
    records = [result(0), result(1)]
    (records[1] if field == "unit" else records[1]["key"])[field] = "different"
    md, tex, claims, following = _result_tables(records, columns=2)
    assert following == 3 and {c["table"] for c in claims} == {"R1", "R2"}
    assert "different" in md and "different" in tex
    assert_spans(md, tex, claims)


def test_bounded_tables_keep_every_row_and_repeat_complete_context():
    records = [result(i, config="same_config") for i in range(29)]
    md, tex, claims, following = _result_tables(records, columns=1)
    assert following == 4 and len(claims) == 29
    assert md.count("dataset version: v1") == 3
    assert tex.count(r"\endhead") == 3
    assert {c["result_id"] for c in claims} == {r["result_id"] for r in records}
    assert max(c["row"] for c in claims) == 12
    assert_spans(md, tex, claims)
    assert _result_tables(list(reversed(records)), columns=1) == (md, tex, claims, following)


def test_configuration_legends_are_bounded_without_prefix_aliasing():
    records = [result(i) for i in range(9)]
    md, tex, claims, following = _result_tables(records, columns=2)
    assert following == 4 and len(claims) == 9
    for table in {c["table"] for c in claims}:
        assert len({c["key"]["config"] for c in claims if c["table"] == table}) <= 4
    assert_spans(md, tex, claims)


def test_duplicates_deduplicated_but_conflicting_records_rejected():
    record = result()
    assert len(_result_tables([record, copy.deepcopy(record)], columns=1)[2]) == 1
    altered = copy.deepcopy(record)
    altered["value"] = 1.0
    with pytest.raises(ManuscriptError, match="Conflicting"):
        _result_tables([record, altered], columns=1)


def test_literal_labels_cannot_inject_markdown_or_tex():
    record = result(method=r"method|\input{private}&_%", dataset=r"dataset|\input{secret}")
    md, tex, claims, _ = _result_tables([record], columns=2)
    assert r"\input{private}" not in tex and r"\input{secret}" not in tex
    assert r"method\|" in md
    assert_spans(md, tex, claims)


def test_publication_groups_results_and_preserves_paragraph_bindings(study):
    root, _ = study
    report = build(study)
    texts, bindings = render_manuscript(root, report)
    assert len({c["table"] for c in bindings["numeric"]}) < len(bindings["numeric"])
    assert_spans(texts["paper_final.md"], texts["paper.tex"], bindings["numeric"])
    for paragraph in bindings["paragraphs"]:
        block = next(s for s in report["sections"] if s["task"]["id"] == paragraph["section"])["blocks"][paragraph["block"]]
        for filename, span in paragraph["spans"].items():
            assert block["text"] in texts[filename][span["start"]:span["end"]]


def test_writer_packet_boundaries_do_not_duplicate_result_tables(inputs, spec):
    spec["seeds"] = list(range(8))
    spec["budget"]["max_total_seconds"] = 400
    expanded = study.__wrapped__(inputs, spec)
    root, _ = expanded
    report = build(expanded)
    result_sections = [s for s in report["sections"] if s["task"]["role"] == "results"]
    assert len(result_sections) == 6  # Three bounded writer packets per RQ.
    texts, bindings = render_manuscript(root, report)
    claims = bindings["numeric"]
    assert len(claims) == 32
    assert len({c["table"] for c in claims}) == 4  # Twelve + four rows per RQ.
    for question in ("Does A improve on Z?", "Does component contribute?"):
        assert texts["paper_final.md"].count("### " + question) == 1
    assert_spans(texts["paper_final.md"], texts["paper.tex"], claims)
