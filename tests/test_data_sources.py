"""External data adapters: pinned acquisition, content-addressed version cache."""
import json
import shutil

import pytest

from researchclaw.data_sources import (
    PROVENANCE_ADAPTER, DataAcquisitionError, materialize_fetch, normalize_fetch,
)
from researchclaw.pipeline.evidence_store import file_hash
from researchclaw.research_inputs import InputContractError, prepare_inputs, verify_inputs
from tests.test_research_inputs import dump, inputs


@pytest.fixture
def source_csv(tmp_path):
    payload = tmp_path / "records.csv"
    payload.write_text(
        "id,feature,target\n" + "".join(f"r{i:03d},{i + 0.125},{i % 2}\n" for i in range(12)),
        encoding="utf-8", newline="")
    return payload


def declaration(source, name="records.csv", **changes):
    payload = source.parent / name if source.is_dir() else source
    return {"url": payload.as_uri(), "sha256": file_hash(payload), "size_cap": 1 << 20, **changes}


def test_normalization_accepts_only_pinned_bounded_https_or_local_sources(source_csv):
    good = declaration(source_csv)
    assert normalize_fetch(good) == good
    assert normalize_fetch({**good, "url": "https://example.org/data/records.csv?v=2"})["url"].startswith("https://")
    assert normalize_fetch({**good, "url": f"file://localhost{source_csv.as_uri()[len('file:'):]}"})
    for bad in (
        None, [], {}, {"url": good["url"], "sha256": good["sha256"]},
        {**good, "extra": 1}, {**good, "url": 7}, {**good, "url": "  "},
        {**good, "url": " " + good["url"]}, {**good, "url": good["url"] + " "},
        {**good, "url": "\t" + good["url"]}, {**good, "url": "https://example.org/r.csv\n"},
        {**good, "url": "https:// example.org/r.csv"},
        {**good, "sha256": file_hash(source_csv).upper()},
        {**good, "sha256": "deadbeef"}, {**good, "sha256": "z" * 64}, {**good, "sha256": 123},
        {**good, "size_cap": 0}, {**good, "size_cap": -1}, {**good, "size_cap": True},
        {**good, "size_cap": (1 << 30) + 1}, {**good, "size_cap": 1.5},
        {**good, "url": "ftp://example.org/r.csv"}, {**good, "url": "http://example.org/r.csv"},
        {**good, "url": "javascript:alert(1)"}, {**good, "url": "https:///r.csv"},
        {**good, "url": "file://server/share/r.csv"}, {**good, "url": "file://localhost"},
        {**good, "url": "https://user:pw@example.org/r.csv"},
        {**good, "url": "https://example.org/r.csv#fragment"},
    ):
        with pytest.raises(DataAcquisitionError):
            normalize_fetch(bad)


def test_materialize_freezes_payload_and_deterministic_provenance(tmp_path, source_csv):
    cache = tmp_path / "cache"
    payload = materialize_fetch(declaration(source_csv), cache)
    assert payload.name == f"{file_hash(source_csv)}.csv"
    assert payload.read_bytes() == source_csv.read_bytes()
    record = json.loads((cache / f"{file_hash(source_csv)}.provenance.json").read_text(encoding="utf-8"))
    assert record == {"adapter": PROVENANCE_ADAPTER, "sha256": file_hash(source_csv),
                      "size": source_csv.stat().st_size, "url": source_csv.as_uri()}
    before = sorted(p.name for p in cache.iterdir())
    assert materialize_fetch(declaration(source_csv), cache) == payload
    assert sorted(p.name for p in cache.iterdir()) == before


def test_fetched_content_that_misses_the_pin_writes_nothing(tmp_path, source_csv):
    cache = tmp_path / "cache"
    original = declaration(source_csv)
    wrong = tmp_path / "mutated.csv"
    shutil.copyfile(source_csv, wrong)
    wrong.write_bytes(wrong.read_bytes() + b"tail\n")
    with pytest.raises(DataAcquisitionError, match="does not match its pinned sha256"):
        materialize_fetch({**original, "url": wrong.as_uri()}, cache)
    assert not cache.exists() or not list(cache.iterdir())


def test_tampered_cache_fails_closed_without_refetching(tmp_path, source_csv):
    cache = tmp_path / "cache"
    pinned = declaration(source_csv)
    payload = materialize_fetch(pinned, cache)
    source_csv.unlink()  # A refetch attempt would fail; the cache must not try one.
    payload.write_bytes(b"tampered\n")
    with pytest.raises(DataAcquisitionError, match="no longer matches its pinned digest"):
        materialize_fetch(pinned, cache)
    payload.write_bytes(b"")
    with pytest.raises(DataAcquisitionError, match="no longer matches"):
        materialize_fetch(pinned, cache)


def test_concurrent_writers_under_contention_all_succeed_cleanly(tmp_path, source_csv):
    import threading
    cache = tmp_path / "cache"
    pinned = declaration(source_csv)
    barrier = threading.Barrier(5)
    results, failures = [], []

    def worker():
        try:
            barrier.wait()
            results.append(materialize_fetch(pinned, cache))
        except Exception as exc:  # noqa: BLE001
            failures.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
    assert {file_hash(payload) for payload in results} == {pinned["sha256"]}
    assert sorted(p.name for p in cache.iterdir()) == [
        f"{pinned['sha256']}.csv", f"{pinned['sha256']}.provenance.json"]


def test_lost_replace_race_without_a_winner_fails_closed_and_clean(tmp_path, source_csv, monkeypatch):
    import pathlib
    cache = tmp_path / "cache"
    pinned = declaration(source_csv)

    def replace(self, target):
        raise PermissionError(13, "no concurrent winner ever completes the address")

    monkeypatch.setattr(pathlib.Path, "replace", replace)
    with pytest.raises(PermissionError):
        materialize_fetch(pinned, cache)
    monkeypatch.undo()
    assert list(cache.iterdir()) == []  # Own staging is cleaned up on every path.


def test_cache_hit_requires_a_consistent_provenance_record(tmp_path, source_csv):
    cache = tmp_path / "cache"
    materialize_fetch(declaration(source_csv), cache)
    sidecar = cache / f"{file_hash(source_csv)}.provenance.json"
    sidecar.unlink()
    with pytest.raises(DataAcquisitionError, match="lacks its provenance"):
        materialize_fetch(declaration(source_csv), cache)
    sidecar.write_text(json.dumps({"adapter": "other/v9", "sha256": "0" * 64, "size": 1, "url": "x"}))
    with pytest.raises(DataAcquisitionError, match="inconsistent"):
        materialize_fetch(declaration(source_csv), cache)


def test_declared_size_cap_bounds_acquisition(tmp_path, source_csv):
    cache = tmp_path / "cache"
    with pytest.raises(DataAcquisitionError, match="size cap"):
        materialize_fetch(declaration(source_csv, size_cap=10), cache)
    assert not cache.exists() or not list(cache.iterdir())


def test_identical_content_from_different_origins_shares_one_cache_entry(tmp_path, source_csv):
    other = tmp_path / "mirror"
    other.mkdir()
    shutil.copyfile(source_csv, other / "records.csv")
    cache = tmp_path / "cache"
    first = materialize_fetch(declaration(source_csv), cache)
    second = materialize_fetch(declaration(other / "records.csv"), cache)
    assert first == second
    # Provenance records the first origin and is not rewritten by a hit.
    sidecar = cache / f"{file_hash(source_csv)}.provenance.json"
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["url"] == source_csv.as_uri()


def test_concurrent_writer_to_the_same_content_address_is_accepted(tmp_path, source_csv, monkeypatch):
    import pathlib
    cache = tmp_path / "cache"
    pinned = declaration(source_csv)
    real_replace, raced = pathlib.Path.replace, {"done": False}

    def replace(self, target):
        if not raced["done"] and self.name.endswith(".staging") and target.suffix == ".csv":
            raced["done"] = True
            shutil.copyfile(source_csv, target)  # A concurrent writer completes the address first.
            raise PermissionError(13, "simulated concurrent replace")
        return real_replace(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", replace)
    payload = materialize_fetch(pinned, cache)
    monkeypatch.undo()
    assert file_hash(payload) == pinned["sha256"]
    assert sorted(p.name for p in cache.iterdir()) == [
        f"{pinned['sha256']}.csv", f"{pinned['sha256']}.provenance.json"]  # No staging leftovers.


@pytest.fixture
def external_inputs(inputs):
    brief_path, root, manifest, brief = inputs
    source_csv = brief_path.parent / "records.csv"
    manifest = {**manifest, "path": f"{file_hash(source_csv)}.csv",
                "fetch": declaration(source_csv)}
    dump(brief_path.parent / "manifest.json", manifest)
    dump(brief_path, {**brief, "allow_external_data": True})
    return brief_path, root, manifest, brief


def test_external_dataset_prepares_from_pinned_cache_with_provenance(external_inputs):
    brief_path, root, manifest, _ = external_inputs
    contract = prepare_inputs(brief_path, root)
    assert verify_inputs(brief_path, root) == contract
    card = contract["datasets"][0]["card"]
    assert card["external_source"] == {"sha256": manifest["fetch"]["sha256"],
                                       "size_bytes": card["external_source"]["size_bytes"],
                                       "cache": f"external_data/{manifest['path']}"}
    assert card["external_source"]["size_bytes"] > 0
    payload = root / "external_data" / manifest["path"]
    assert payload.is_file() and file_hash(payload) == manifest["fetch"]["sha256"]
    assert str(payload.resolve()) in contract["source_files"]
    assert contract["datasets"][0]["manifest"]["fetch"] == manifest["fetch"]
    assert prepare_inputs(brief_path, root) == contract  # Resume does not refetch or diverge.


def test_external_acquisition_requires_the_declared_permission(inputs):
    brief_path, root, manifest, _ = inputs
    source_csv = brief_path.parent / "records.csv"
    manifest = {**manifest, "path": f"{file_hash(source_csv)}.csv",
                "fetch": declaration(source_csv)}
    dump(brief_path.parent / "manifest.json", manifest)
    with pytest.raises(InputContractError, match="allow_external_data"):
        prepare_inputs(brief_path, root)


def test_manifest_path_and_pin_must_agree(inputs):
    brief_path, root, manifest, _ = inputs
    source_csv = brief_path.parent / "records.csv"
    manifest = {**manifest, "path": "someone.csv", "fetch": declaration(source_csv)}
    dump(brief_path.parent / "manifest.json", manifest)
    with pytest.raises(InputContractError, match="pinned payload name"):
        prepare_inputs(brief_path, root)
    manifest = {**manifest, "fetch": declaration(source_csv, sha256="z" * 64)}
    dump(brief_path.parent / "manifest.json", manifest)
    with pytest.raises(InputContractError, match="External dataset declaration is invalid"):
        prepare_inputs(brief_path, root)


def test_tampered_cache_breaks_resume_verification(external_inputs):
    brief_path, root, manifest, _ = external_inputs
    prepare_inputs(brief_path, root)
    payload = root / "external_data" / manifest["path"]
    payload.write_bytes(payload.read_bytes().replace(b"feature", b"tampered", 1))
    with pytest.raises(InputContractError, match="Frozen source changed or missing"):
        verify_inputs(brief_path, root)


def test_wrong_fresh_fetch_fails_before_any_cache_write(external_inputs):
    brief_path, root, manifest, _ = external_inputs
    bad = {**manifest["fetch"], "sha256": "b" * 64}
    manifest = {**manifest, "path": f"{'b' * 64}.csv", "fetch": bad}
    dump(brief_path.parent / "manifest.json", manifest)
    with pytest.raises(DataAcquisitionError, match="does not match its pinned sha256"):
        prepare_inputs(brief_path, root)
    assert not (root / "external_data").exists() or not list((root / "external_data").iterdir())
