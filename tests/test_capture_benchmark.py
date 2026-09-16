from __future__ import annotations

import subprocess
import sys

import pytest
from scripts import benchmark_capture


def test_worker_timeout_cleans_up_descendants(monkeypatch):
    children = []
    real_wait = benchmark_capture.psutil.wait_procs

    def remember_wait(processes, timeout):
        children.extend(processes)
        return real_wait(processes, timeout=timeout)

    monkeypatch.setattr(benchmark_capture.psutil, "wait_procs", remember_wait)
    program = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        benchmark_capture.run_worker([sys.executable, "-c", program], timeout=1)
    assert children
    assert all(not child.is_running() or child.status() == "zombie" for child in children)


def test_worker_reads_json_result():
    result = benchmark_capture.run_worker([sys.executable, "-c", "print('{\"successful_reads\": 3}')"], timeout=5)
    assert result["successful_reads"] == 3


def test_worker_reports_crash():
    result = benchmark_capture.run_worker(
        [sys.executable, "-c", "import sys; sys.stderr.write('capture failed'); sys.exit(2)"], timeout=5
    )
    assert result["error"] == "capture failed"
    assert result["exit_code"] == 2
