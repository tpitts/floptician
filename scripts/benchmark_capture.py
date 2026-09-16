"""Benchmark the production OpenCV camera path on Mac or Windows, without inference."""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil

from floptician.camera_devices import discover_cameras
from floptician.camera_manager import CameraManager


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List names and persistent device IDs")
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument("--camera", help="Exact, unique camera name")
    identity.add_argument("--device-id", help="Persistent device ID from --list")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30)
    parser.add_argument("--sample-fps", type=float, default=1.8)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.list and not (args.camera or args.device_id):
        parser.error("Select --camera or --device-id; use --list to discover devices")
    for field in ("camera_width", "camera_height", "camera_fps", "sample_fps", "seconds", "repeats"):
        if not np.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            parser.error(f"{field} must be positive and finite")
    if not np.isfinite(args.warmup) or args.warmup < 0:
        parser.error("warmup must be nonnegative and finite")
    return args


def worker(args):
    devices = discover_cameras()
    matches = (
        [d for d in devices if d.device_id == args.device_id]
        if args.device_id
        else [d for d in devices if d.name == args.camera]
    )
    if len(matches) != 1:
        raise ValueError("Camera missing or name ambiguous. Use --list and --device-id.")
    started = time.perf_counter()
    with CameraManager() as camera:
        camera.open_camera(
            matches[0],
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
        )
        startup_ms = (time.perf_counter() - started) * 1000
        mode = asdict(camera.info)
        warmup_end = time.perf_counter() + args.warmup
        while time.perf_counter() < warmup_end:
            camera.get_frame()
            time.sleep(1 / args.sample_fps)
        process = psutil.Process()
        cpu_before = sum(process.cpu_times()[:2])
        read_ms, memories = [], []
        duplicates = 0
        previous = None
        started = time.perf_counter()
        next_read = started
        while time.perf_counter() - started < args.seconds:
            time.sleep(max(0, next_read - time.perf_counter()))
            if time.perf_counter() - started >= args.seconds:
                break
            before = time.perf_counter()
            ok, frame = camera.get_frame()
            if not ok:
                raise RuntimeError("Camera closed unexpectedly")
            read_ms.append((time.perf_counter() - before) * 1000)
            if previous is not None and np.array_equal(previous, frame):
                duplicates += 1
            previous = frame.copy()
            memories.append(process.memory_info().rss / 1024**2)
            next_read = max(next_read + 1 / args.sample_fps, time.perf_counter())
        elapsed = time.perf_counter() - started
        cpu_after = sum(process.cpu_times()[:2])
        return {
            "camera": camera.device.name,
            "native_mode": mode,
            "output_size": [mode["width"], mode["height"]],
            "sample_fps": args.sample_fps,
            "requested_camera_fps": args.camera_fps,
            "startup_ms": round(startup_ms, 1),
            "elapsed_seconds": round(elapsed, 2),
            "successful_reads": len(read_ms),
            "successful_reads_per_second": round(len(read_ms) / elapsed, 2),
            "identical_adjacent_frames": duplicates,
            "cpu_percent_one_core": round((cpu_after - cpu_before) / elapsed * 100, 2),
            "peak_sampled_rss_mib": round(max(memories), 1) if memories else None,
            "read_ms_median": round(float(np.median(read_ms)), 3) if read_ms else None,
            "read_ms_p95": round(float(np.percentile(read_ms, 95)), 3) if read_ms else None,
            "sensor_to_application_latency_ms": None,
        }


def run_worker(command, timeout):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        lines = stdout.strip().splitlines()
        result = json.loads(lines[-1]) if lines else {"error": stderr[-2000:] or "No worker response"}
        if process.returncode:
            result["exit_code"] = process.returncode
        return result
    finally:
        if process.poll() is None:
            children = psutil.Process(process.pid).children(recursive=True)
            for child in children:
                with contextlib.suppress(psutil.NoSuchProcess):
                    child.kill()
            process.kill()
            process.communicate()
            psutil.wait_procs(children, timeout=3)


def main():
    args = parse_args()
    if args.list:
        for device in discover_cameras():
            print(json.dumps(asdict(device)))
        return
    if args.worker:
        try:
            result = worker(args)
        except Exception as exc:
            result = {"error": str(exc)}
        print(json.dumps(result), flush=True)
        return
    print("Stop Floptician first. Keep the same OBS workload throughout. CPU: 100% = one core.", flush=True)
    results = []
    for repeat in range(args.repeats):
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        try:
            result = run_worker(command, args.seconds + args.warmup + 45)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            result = {"error": str(exc)}
        result["round"] = repeat + 1
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {
        "platform": platform.platform(),
        "notes": "No inference; read time is not optical latency.",
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if any("error" in result for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
