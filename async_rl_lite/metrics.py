"""Metrics collection and monitoring for the RL training pipeline.

Provides real-time tracking of throughput, staleness, GPU utilization,
and other system health indicators used by the training loop and
the repack / fault-tolerance subsystems.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class MetricSample:
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)


class TimeSeriesBuffer:
    """Fixed-size circular buffer of metric samples."""

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._buffer: deque[MetricSample] = deque(maxlen=max_size)

    def add(self, sample: MetricSample) -> None:
        self._buffer.append(sample)

    def get_recent(self, window_sec: float = 60.0) -> list[MetricSample]:
        cutoff = time.time() - window_sec
        return [s for s in self._buffer if s.timestamp >= cutoff]

    def get_values(self, window_sec: float = 60.0) -> list[float]:
        return [s.value for s in self.get_recent(window_sec)]

    def mean(self, window_sec: float = 60.0) -> float:
        values = self.get_values(window_sec)
        return float(np.mean(values)) if values else 0.0

    def percentile(self, p: float, window_sec: float = 60.0) -> float:
        values = self.get_values(window_sec)
        return float(np.percentile(values, p)) if values else 0.0

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def latest(self) -> MetricSample | None:
        return self._buffer[-1] if self._buffer else None


class ThroughputTracker:
    """Tracks trajectory generation and training throughput."""

    def __init__(self, window_sec: float = 60.0) -> None:
        self.window_sec = window_sec
        self._generation_events: deque[float] = deque()
        self._training_events: deque[float] = deque()
        self._token_counts: deque[tuple[float, int]] = deque()

    def record_generation(self, count: int = 1) -> None:
        now = time.time()
        for _ in range(count):
            self._generation_events.append(now)
        self._evict_old()

    def record_training(self, count: int = 1) -> None:
        now = time.time()
        for _ in range(count):
            self._training_events.append(now)
        self._evict_old()

    def record_tokens(self, count: int) -> None:
        self._token_counts.append((time.time(), count))

    def _evict_old(self) -> None:
        cutoff = time.time() - self.window_sec * 2
        while self._generation_events and self._generation_events[0] < cutoff:
            self._generation_events.popleft()
        while self._training_events and self._training_events[0] < cutoff:
            self._training_events.popleft()
        while self._token_counts and self._token_counts[0][0] < cutoff:
            self._token_counts.popleft()

    @property
    def generation_throughput(self) -> float:
        cutoff = time.time() - self.window_sec
        count = sum(1 for t in self._generation_events if t >= cutoff)
        return count / self.window_sec

    @property
    def training_throughput(self) -> float:
        cutoff = time.time() - self.window_sec
        count = sum(1 for t in self._training_events if t >= cutoff)
        return count / self.window_sec

    @property
    def token_throughput(self) -> float:
        cutoff = time.time() - self.window_sec
        total = sum(c for t, c in self._token_counts if t >= cutoff)
        return total / self.window_sec

    def stats(self) -> dict[str, float]:
        return {
            "generation_throughput_traj_per_sec": self.generation_throughput,
            "training_throughput_updates_per_sec": self.training_throughput,
            "token_throughput_per_sec": self.token_throughput,
        }


class StalenessMonitor:
    """Monitors data staleness across the training pipeline."""

    def __init__(self, window_size: int = 100) -> None:
        self._staleness_values: deque[float] = deque(maxlen=window_size)
        self._version_spans: deque[float] = deque(maxlen=window_size)
        self._inherent_staleness: deque[float] = deque(maxlen=window_size)
        self.window_size = window_size

    def record(
        self,
        staleness: float,
        version_span: float = 0.0,
        inherent_staleness: float = 0.0,
    ) -> None:
        self._staleness_values.append(staleness)
        self._version_spans.append(version_span)
        self._inherent_staleness.append(inherent_staleness)

    @property
    def avg_staleness(self) -> float:
        if not self._staleness_values:
            return 0.0
        return float(np.mean(list(self._staleness_values)))

    @property
    def max_staleness(self) -> float:
        if not self._staleness_values:
            return 0.0
        return float(max(self._staleness_values))

    @property
    def avg_version_span(self) -> float:
        if not self._version_spans:
            return 0.0
        return float(np.mean(list(self._version_spans)))

    @property
    def avg_inherent_staleness(self) -> float:
        if not self._inherent_staleness:
            return 0.0
        return float(np.mean(list(self._inherent_staleness)))

    def stats(self) -> dict[str, float]:
        return {
            "avg_staleness": self.avg_staleness,
            "max_staleness": self.max_staleness,
            "avg_version_span": self.avg_version_span,
            "avg_inherent_staleness": self.avg_inherent_staleness,
            "samples": float(len(self._staleness_values)),
        }


class GPUUtilizationTracker:
    """Tracks GPU utilization across the cluster."""

    def __init__(self, num_gpus: int, sample_buffer_size: int = 100) -> None:
        self.num_gpus = num_gpus
        self._utilization: dict[int, TimeSeriesBuffer] = {
            i: TimeSeriesBuffer(max_size=sample_buffer_size)
            for i in range(num_gpus)
        }
        self._memory_util: dict[int, TimeSeriesBuffer] = {
            i: TimeSeriesBuffer(max_size=sample_buffer_size)
            for i in range(num_gpus)
        }

    def record(
        self,
        gpu_id: int,
        compute_util: float,
        memory_util: float,
    ) -> None:
        if gpu_id in self._utilization:
            self._utilization[gpu_id].add(
                MetricSample(name=f"gpu_{gpu_id}_compute", value=compute_util)
            )
            self._memory_util[gpu_id].add(
                MetricSample(name=f"gpu_{gpu_id}_memory", value=memory_util)
            )

    def avg_compute_util(self, window_sec: float = 60.0) -> float:
        means = [buf.mean(window_sec) for buf in self._utilization.values()]
        return float(np.mean(means)) if means else 0.0

    def avg_memory_util(self, window_sec: float = 60.0) -> float:
        means = [buf.mean(window_sec) for buf in self._memory_util.values()]
        return float(np.mean(means)) if means else 0.0

    def stats(self) -> dict[str, float]:
        return {
            "num_gpus": float(self.num_gpus),
            "avg_compute_utilization": self.avg_compute_util(),
            "avg_memory_utilization": self.avg_memory_util(),
        }


class MetricsCollector:
    """Central metrics aggregator for the entire training pipeline."""

    def __init__(
        self,
        num_gpus: int = 0,
        throughput_window_sec: float = 60.0,
    ) -> None:
        self.throughput = ThroughputTracker(window_sec=throughput_window_sec)
        self.staleness = StalenessMonitor()
        self.gpu_tracker = GPUUtilizationTracker(num_gpus=num_gpus) if num_gpus > 0 else None

        self._custom_metrics: dict[str, TimeSeriesBuffer] = {}
        self._counters: dict[str, int] = {}
        self._start_time = time.time()

    def record_custom(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        if name not in self._custom_metrics:
            self._custom_metrics[name] = TimeSeriesBuffer()
        self._custom_metrics[name].add(
            MetricSample(name=name, value=value, tags=tags or {})
        )

    def increment(self, name: str, count: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + count

    def get_counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    def get_custom_mean(self, name: str, window_sec: float = 60.0) -> float:
        buf = self._custom_metrics.get(name)
        return buf.mean(window_sec) if buf else 0.0

    @property
    def uptime_sec(self) -> float:
        return time.time() - self._start_time

    def full_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "uptime_sec": self.uptime_sec,
            "throughput": self.throughput.stats(),
            "staleness": self.staleness.stats(),
            "counters": dict(self._counters),
        }

        if self.gpu_tracker is not None:
            report["gpu"] = self.gpu_tracker.stats()

        custom_summary = {}
        for name, buf in self._custom_metrics.items():
            custom_summary[name] = {
                "latest": buf.latest.value if buf.latest else None,
                "mean_60s": buf.mean(60.0),
                "p50_60s": buf.percentile(50.0, 60.0),
                "p99_60s": buf.percentile(99.0, 60.0),
                "samples": buf.size,
            }
        if custom_summary:
            report["custom"] = custom_summary

        return report
