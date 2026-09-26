"""Opt-in, bounded real CUDA/NCCL collective execution probe."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from researchclaw.pipeline.evidence_store import content_hash


_CHECKER = "cuda-nccl-all-reduce/v1"
_SCOPE = "single-host real CUDA processes and one NCCL all-reduce; no cross-host claim"


class MultiDeviceProbeError(ValueError):
    pass


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _worker(rank: int, world_size: int, rendezvous: str, output: Path, seed: int) -> int:
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dist.init_process_group(
        backend="nccl", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        value = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(rank)
        properties = torch.cuda.get_device_properties(rank)
        _write(output, {
            "rank": rank, "device": rank, "value": float(value.cpu().item()),
            "initial_seed": int(torch.initial_seed()),
            "name": str(properties.name),
            "capability": list(map(int, torch.cuda.get_device_capability(rank))),
            "total_memory": int(properties.total_memory),
        })
    finally:
        dist.destroy_process_group()
    return 0


def _base_report(*, status: str, reason: str | None, torch_version: str | None,
                 cuda_version: str | None, world_size: int, results: list[dict]) -> dict:
    report = {
        "schema_version": 1, "checker": _CHECKER,
        "status": status, "reason": reason, "backend": "nccl",
        "torch_version": torch_version, "cuda_version": cuda_version,
        "world_size": world_size, "seed": 0, "results": results,
        "scope": _SCOPE,
    }
    report["version"] = content_hash(report)
    return report


def run_multi_device_probe(root: Path, *, timeout_seconds: int = 120,
                           max_devices: int = 8) -> dict:
    """Execute one process per real CUDA device and freeze a fail-closed report."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600
            or type(max_devices) is not int or not 2 <= max_devices <= 8):
        raise MultiDeviceProbeError("Invalid multi-device probe budget")
    try:
        import torch
        import torch.distributed as dist
        torch_version, cuda_version = str(torch.__version__), str(torch.version.cuda or "") or None
        count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        nccl = bool(dist.is_available() and dist.is_nccl_available())
    except ImportError:
        torch_version = cuda_version = None
        count, nccl = 0, False
    except Exception as exc:
        report = _base_report(status="failed", reason="discovery:" + type(exc).__name__,
                              torch_version=None, cuda_version=None, world_size=0, results=[])
        _write(root / "multi_device_probe.json", report)
        return report
    if count < 2 or not nccl:
        reason = "fewer_than_two_cuda_devices" if count < 2 else "nccl_unavailable"
        report = _base_report(status="unavailable", reason=reason,
                              torch_version=torch_version, cuda_version=cuda_version,
                              world_size=min(count, max_devices), results=[])
        _write(root / "multi_device_probe.json", report)
        return report
    world_size = min(count, max_devices)
    run_token = uuid.uuid4().hex
    rendezvous_file = (root / f".multi_device_rendezvous_{run_token}").resolve()
    outputs = [root / f".multi_device_rank_{run_token}_{rank}.json"
               for rank in range(world_size)]
    processes = []
    try:
        for rank, output in enumerate(outputs):
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "researchclaw.experiment.multi_device_probe", "--worker", str(rank),
                 str(world_size), rendezvous_file.as_uri(), str(output), "0"],
                cwd=Path(__file__).resolve().parents[2], stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL))
        deadline = time.monotonic() + timeout_seconds
        for process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout_seconds)
            process.wait(timeout=remaining)
        if any(process.returncode != 0 for process in processes):
            raise MultiDeviceProbeError("A CUDA rank process failed")
        results = [json.loads(path.read_text(encoding="utf-8")) for path in outputs]
        report = _base_report(status="passed", reason=None,
                              torch_version=torch_version, cuda_version=cuda_version,
                              world_size=world_size, results=results)
        verify_multi_device_probe(report)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        report = _base_report(status="failed", reason="execution:" + type(exc).__name__,
                              torch_version=torch_version, cuda_version=cuda_version,
                              world_size=world_size, results=[])
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        rendezvous_file.unlink(missing_ok=True)
        for output in outputs:
            output.unlink(missing_ok=True)
    _write(root / "multi_device_probe.json", report)
    return report


def verify_multi_device_probe(report: dict) -> dict:
    """Verify a frozen report without claiming unavailable hardware executed."""
    fields = {"schema_version", "checker", "status", "reason", "backend",
              "torch_version", "cuda_version", "world_size", "seed", "results",
              "scope", "version"}
    if not isinstance(report, dict) or set(report) != fields:
        raise MultiDeviceProbeError("Malformed multi-device report")
    payload = dict(report)
    version = payload.pop("version")
    if version != content_hash(payload):
        raise MultiDeviceProbeError("Multi-device report changed")
    versions_are_valid = all(
        value is None or (isinstance(value, str) and 0 < len(value) <= 128)
        for value in (report["torch_version"], report["cuda_version"])
    )
    if (type(report["schema_version"]) is not int or report["schema_version"] != 1
            or report["checker"] != _CHECKER or report["backend"] != "nccl"
            or report["scope"] != _SCOPE or type(report["seed"]) is not int
            or report["seed"] != 0 or not versions_are_valid
            or report["status"] not in {"passed", "unavailable", "failed"}
            or type(report["world_size"]) is not int or not 0 <= report["world_size"] <= 8
            or not isinstance(report["results"], list)):
        raise MultiDeviceProbeError("Invalid multi-device report identity")
    if report["status"] != "passed":
        if (not isinstance(report["reason"], str)
                or not 0 < len(report["reason"]) <= 256 or report["results"]):
            raise MultiDeviceProbeError("Invalid unavailable/failed multi-device report")
        if (report["status"] == "unavailable"
                and report["reason"] not in {"fewer_than_two_cuda_devices", "nccl_unavailable"}):
            raise MultiDeviceProbeError("Invalid unavailable multi-device reason")
        if (report["status"] == "unavailable"
                and ((report["reason"] == "fewer_than_two_cuda_devices"
                      and report["world_size"] not in {0, 1})
                     or (report["reason"] == "nccl_unavailable"
                         and not 2 <= report["world_size"] <= 8))):
            raise MultiDeviceProbeError("Unavailable reason conflicts with device count")
        if report["status"] == "failed" and not report["reason"].startswith(
                ("discovery:", "execution:")):
            raise MultiDeviceProbeError("Invalid failed multi-device reason")
        if (report["status"] == "failed"
                and ((report["reason"].startswith("discovery:") and report["world_size"] != 0)
                     or (report["reason"].startswith("execution:")
                         and not 2 <= report["world_size"] <= 8))):
            raise MultiDeviceProbeError("Failed reason conflicts with execution state")
        return report
    if (report["reason"] is not None or not 2 <= report["world_size"] <= 8
            or len(report["results"]) != report["world_size"]
            or not isinstance(report["torch_version"], str) or not report["torch_version"]
            or not isinstance(report["cuda_version"], str) or not report["cuda_version"]):
        raise MultiDeviceProbeError("Incomplete passed multi-device report")
    expected = report["world_size"] * (report["world_size"] + 1) / 2
    for rank, row in enumerate(report["results"]):
        if (not isinstance(row, dict) or set(row) != {"rank", "device", "value", "initial_seed",
                                                     "name", "capability", "total_memory"}
                or type(row["rank"]) is not int or row["rank"] != rank
                or type(row["device"]) is not int or row["device"] != rank
                or type(row["value"]) not in {int, float} or row["value"] != expected
                or type(row["initial_seed"]) is not int or row["initial_seed"] != 0
                or not isinstance(row["name"], str) or not 0 < len(row["name"]) <= 256
                or not isinstance(row["capability"], list) or len(row["capability"]) != 2
                or any(type(value) is not int or value < 0 for value in row["capability"])
                or type(row["total_memory"]) is not int or row["total_memory"] <= 0):
            raise MultiDeviceProbeError("Invalid CUDA rank evidence")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a real local CUDA/NCCL collective probe")
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-devices", type=int, default=8)
    parser.add_argument("--worker", nargs=5, metavar=("RANK", "WORLD", "RENDEZVOUS", "OUTPUT", "SEED"))
    args = parser.parse_args(argv)
    if args.worker:
        rank, world, rendezvous, output, seed = args.worker
        return _worker(int(rank), int(world), rendezvous, Path(output), int(seed))
    if args.root is None:
        parser.error("root is required unless --worker is used")
    report = run_multi_device_probe(
        args.root, timeout_seconds=args.timeout_seconds, max_devices=args.max_devices)
    print(json.dumps({"status": report["status"], "reason": report["reason"],
                      "world_size": report["world_size"]}, sort_keys=True))
    return 0 if report["status"] in {"passed", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
