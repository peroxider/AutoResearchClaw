from __future__ import annotations

import pytest

from researchclaw.experiment.multi_device_probe import (
    MultiDeviceProbeError, _base_report, run_multi_device_probe,
    verify_multi_device_probe,
)


def passed_report(world_size=2):
    expected = world_size * (world_size + 1) / 2
    rows = [{"rank": rank, "device": rank, "value": expected, "initial_seed": 0,
             "name": f"GPU {rank}", "capability": [8, 0], "total_memory": 1024}
            for rank in range(world_size)]
    return _base_report(status="passed", reason=None, torch_version="2.8",
                        cuda_version="12.8", world_size=world_size, results=rows)


def test_passed_collective_report_replays_exact_rank_evidence():
    report = passed_report(3)
    assert verify_multi_device_probe(report) == report


@pytest.mark.parametrize("mutate", [
    lambda r: r["results"][0].update(value=1.0),
    lambda r: r["results"][1].update(device=0),
    lambda r: r["results"][0].update(initial_seed=7),
    lambda r: r.update(world_size=1),
])
def test_collective_report_rejects_forged_success(mutate):
    report = passed_report()
    mutate(report)
    payload = dict(report)
    payload.pop("version")
    from researchclaw.pipeline.evidence_store import content_hash
    report["version"] = content_hash(payload)
    with pytest.raises(MultiDeviceProbeError):
        verify_multi_device_probe(report)


@pytest.mark.parametrize(("field", "value"), [
    ("scope", "cross-host verified"),
    ("torch_version", ""),
    ("cuda_version", None),
    ("seed", False),
])
def test_collective_report_rejects_self_consistent_identity_forgery(field, value):
    report = passed_report()
    report[field] = value
    payload = dict(report)
    payload.pop("version")
    from researchclaw.pipeline.evidence_store import content_hash
    report["version"] = content_hash(payload)
    with pytest.raises(MultiDeviceProbeError):
        verify_multi_device_probe(report)


@pytest.mark.parametrize(("field", "value"), [
    ("rank", False),
    ("device", False),
    ("value", True),
    ("initial_seed", False),
])
def test_collective_report_rejects_bool_rank_evidence(field, value):
    report = passed_report()
    report["results"][0][field] = value
    payload = dict(report)
    payload.pop("version")
    from researchclaw.pipeline.evidence_store import content_hash
    report["version"] = content_hash(payload)
    with pytest.raises(MultiDeviceProbeError):
        verify_multi_device_probe(report)


def test_current_host_probe_is_honest_when_hardware_is_unavailable(tmp_path):
    report = run_multi_device_probe(tmp_path, timeout_seconds=10, max_devices=2)
    assert report["status"] in {"passed", "unavailable", "failed"}
    verify_multi_device_probe(report)
    if report["status"] != "passed":
        assert report["results"] == [] and report["reason"]


def test_probe_budget_is_bounded(tmp_path):
    with pytest.raises(MultiDeviceProbeError):
        run_multi_device_probe(tmp_path, timeout_seconds=0)
    with pytest.raises(MultiDeviceProbeError):
        run_multi_device_probe(tmp_path, max_devices=9)


@pytest.mark.parametrize(("status", "reason", "world_size"), [
    ("unavailable", "fewer_than_two_cuda_devices", 2),
    ("unavailable", "nccl_unavailable", 1),
    ("failed", "discovery:RuntimeError", 2),
    ("failed", "execution:RuntimeError", 0),
])
def test_non_passed_report_rejects_reason_state_conflicts(status, reason, world_size):
    report = _base_report(status=status, reason=reason, torch_version=None,
                          cuda_version=None, world_size=world_size, results=[])
    with pytest.raises(MultiDeviceProbeError):
        verify_multi_device_probe(report)
