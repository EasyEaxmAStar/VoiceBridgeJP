"""Continuous nvidia-smi sampling without adding GPU/runtime dependencies."""
from __future__ import annotations

from collections import deque
import csv
from datetime import datetime
import subprocess
import threading
import time


class GPUSampler:
    """Observed device peaks are sampled, never sums of allocator peaks."""

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.samples = deque(maxlen=20000)
        self.process_samples = deque(maxlen=20000)
        self.errors = []
        self.count = 0
        self.peak = 0
        self.resident_peak = 0
        self.resident_count = 0
        self.max_capture_gap = 0.0
        self._last_capture = None
        self._processes = []
        self._threads = []
        self.resident = False
        self._lock = threading.Lock()

    def start(self):
        queries = [("gpu", "timestamp,memory.used,memory.total,temperature.gpu,clocks.current.graphics"),
                   ("compute-apps", "pid,used_gpu_memory")]
        for kind, fields in queries:
            try:
                process = subprocess.Popen(["nvidia-smi", f"--query-{kind}={fields}",
                    "--format=csv,noheader,nounits", "--id=0", f"--loop-ms={self.interval_ms}"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self._processes.append(process)
                thread = threading.Thread(target=self._read, args=(process, kind), daemon=True)
                self._threads.append(thread)
                thread.start()
            except OSError as exc:
                self.errors.append(str(exc))
        return self

    def _read(self, process, kind):
        for line in process.stdout:
            if not line.strip():
                continue
            try:
                fields = [x.strip() for x in next(csv.reader([line]))]
                self.ingest(kind, fields, time.perf_counter())
            except (ValueError, IndexError):
                if len(self.errors) < 20:
                    self.errors.append(f"Unparsed {kind} sample: {line.strip()}")

    def ingest(self, kind, fields, received):
        with self._lock:
            if kind == "compute-apps":
                self.process_samples.append({"received_at":received, "pid":int(fields[0]),
                    "used_mib":float(fields[1]) if fields[1] not in {"[N/A]", "N/A", "[Not Supported]"} else None,
                    "raw":fields[1]})
                return
            captured = datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f")
            used = float(fields[1])
            row = {"driver_timestamp":fields[0], "received_at":received, "used_mib":used,
                   "total_mib":float(fields[2]), "temperature_c":float(fields[3]),
                   "graphics_clock_mhz":float(fields[4]), "both_models_resident":self.resident}
            if self._last_capture is not None:
                self.max_capture_gap = max(self.max_capture_gap, (captured-self._last_capture).total_seconds())
            self._last_capture = captured
            self.count += 1
            self.peak = max(self.peak, used)
            if self.resident:
                self.resident_peak = max(self.resident_peak, used)
                self.resident_count += 1
            self.samples.append(row)

    def stop(self):
        for process in self._processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        for thread in self._threads:
            thread.join(timeout=3)
        for process in self._processes:
            error = process.stderr.read().strip()
            if error:
                self.errors.append(error[:1000])
            process.stdout.close()
            process.stderr.close()

    def report(self, pids=()):
        per_process = {}
        for pid in pids:
            rows = [x for x in self.process_samples if x["pid"] == pid]
            available = [x["used_mib"] for x in rows if x["used_mib"] is not None]
            per_process[str(pid)] = {"sample_count":len(rows), "observed_peak_mib":max(available) if available else None,
                "unavailable_reason":None if available else "Driver exposes no numeric per-process memory (common with Windows WDDM)"}
        return {"source":"nvidia-smi continuous loop, device 0", "requested_interval_ms":self.interval_ms,
                "sample_count":self.count, "trace_retained":len(self.samples),
                "max_driver_capture_gap_seconds":self.max_capture_gap,
                "observed_device_peak_mib":self.peak if self.count else None,
                "both_resident_samples":self.resident_count,
                "observed_both_resident_peak_mib":self.resident_peak if self.resident_count else None,
                "target_mib":14*1024,
                "observed_below_target":self.resident_peak < 14*1024 if self.resident_count else None,
                "per_process":per_process, "errors":self.errors,
                "limitations":"Sampled device-wide framebuffer usage includes desktop/other processes; transients between samples may be missed. Allocator peaks are separate and are never added to infer a simultaneous peak. Per-process accounting can be unavailable under WDDM.",
                "device_trace":list(self.samples), "process_trace":list(self.process_samples)}
