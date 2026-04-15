"""KVCache utilization monitoring and idleness detection.

Tracks GPU KVCache usage per rollout to identify underutilized
rollouts that are candidates for the repack mechanism.  In production
this would read real GPU memory counters; here we simulate the
lifecycle described in the report (ramp-up → plateau → decline).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class RolloutPhase(Enum):
    IDLE = "idle"
    RAMPING_UP = "ramping_up"
    ACTIVE = "active"
    DECLINING = "declining"
    COMPLETED = "completed"


@dataclass(slots=True)
class KVCacheSnapshot:
    """Point-in-time KVCache usage for one rollout."""

    rollout_id: int
    utilization: float          # 0.0 .. 1.0
    num_active_sequences: int
    total_tokens_cached: int
    max_capacity_tokens: int
    timestamp: float = field(default_factory=time.time)
    phase: RolloutPhase = RolloutPhase.IDLE

    @property
    def headroom(self) -> float:
        return 1.0 - self.utilization

    @property
    def tokens_remaining(self) -> int:
        return self.max_capacity_tokens - self.total_tokens_cached


@dataclass(slots=True)
class KVCacheHistory:
    """Time series of utilization for one rollout."""

    rollout_id: int
    snapshots: list[KVCacheSnapshot] = field(default_factory=list)
    window_size: int = 10

    def add(self, snapshot: KVCacheSnapshot) -> None:
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.window_size * 3:
            self.snapshots = self.snapshots[-self.window_size * 2 :]

    @property
    def latest(self) -> KVCacheSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    @property
    def recent_window(self) -> list[KVCacheSnapshot]:
        return self.snapshots[-self.window_size :]

    @property
    def utilization_trend(self) -> float:
        """Return the slope of utilization over the recent window.

        Positive = still filling, negative = draining, ~0 = idle plateau.
        """
        window = self.recent_window
        if len(window) < 3:
            return 0.0
        utils = [s.utilization for s in window]
        x = np.arange(len(utils), dtype=np.float64)
        coeffs = np.polyfit(x, utils, 1)
        return float(coeffs[0])

    @property
    def is_declining(self) -> bool:
        return self.utilization_trend < -0.01

    @property
    def is_stable(self) -> bool:
        return abs(self.utilization_trend) < 0.005


class IdlenessDetector:
    """Determines whether a rollout is in long-tail generation.

    A rollout is considered idle (= candidate for repack) when its
    KVCache utilization is non-increasing and below a threshold,
    which indicates the decode step count is in its ramp-down phase.
    """

    def __init__(
        self,
        utilization_threshold: float = 0.3,
        trend_threshold: float = -0.005,
        min_observations: int = 5,
    ) -> None:
        self.utilization_threshold = utilization_threshold
        self.trend_threshold = trend_threshold
        self.min_observations = min_observations

    def is_idle(self, history: KVCacheHistory) -> bool:
        if len(history.snapshots) < self.min_observations:
            return False

        latest = history.latest
        if latest is None:
            return False

        # Low utilization + non-increasing trend
        if latest.utilization > self.utilization_threshold:
            return False

        trend = history.utilization_trend
        return trend <= self.trend_threshold or history.is_stable

    def compute_idleness_score(self, history: KVCacheHistory) -> float:
        """Score between 0 (fully active) and 1 (completely idle)."""
        if not history.snapshots:
            return 0.0

        latest = history.latest
        if latest is None:
            return 0.0

        # Combine utilization and trend into a score
        util_score = 1.0 - min(latest.utilization / max(self.utilization_threshold, 0.01), 1.0)
        trend = history.utilization_trend
        trend_score = max(0.0, min(1.0, -trend / max(abs(self.trend_threshold), 1e-6)))
        seq_score = 1.0 - min(latest.num_active_sequences / 4.0, 1.0)

        return 0.5 * util_score + 0.3 * trend_score + 0.2 * seq_score


class KVCacheMonitor:
    """Monitors KVCache utilization across all rollouts.

    Periodically samples utilization from each rollout, maintains
    per-rollout history, and provides idleness rankings to the
    repack mechanism.
    """

    def __init__(
        self,
        num_rollouts: int,
        max_capacity_tokens: int = 32768,
        sample_interval_sec: float = 1.0,
        idleness_threshold: float = 0.3,
        window_size: int = 10,
    ) -> None:
        self.num_rollouts = num_rollouts
        self.max_capacity_tokens = max_capacity_tokens
        self.sample_interval_sec = sample_interval_sec
        self.window_size = window_size

        self.detector = IdlenessDetector(
            utilization_threshold=idleness_threshold,
        )
        self.histories: dict[int, KVCacheHistory] = {
            i: KVCacheHistory(rollout_id=i, window_size=window_size)
            for i in range(num_rollouts)
        }

        self._total_samples = 0
        self._idle_detections = 0

    def record_utilization(
        self,
        rollout_id: int,
        utilization: float,
        num_active_sequences: int,
        total_tokens_cached: int,
        phase: RolloutPhase = RolloutPhase.ACTIVE,
    ) -> None:
        """Record a new utilization sample for a rollout."""
        snapshot = KVCacheSnapshot(
            rollout_id=rollout_id,
            utilization=utilization,
            num_active_sequences=num_active_sequences,
            total_tokens_cached=total_tokens_cached,
            max_capacity_tokens=self.max_capacity_tokens,
            phase=phase,
        )

        if rollout_id not in self.histories:
            self.histories[rollout_id] = KVCacheHistory(
                rollout_id=rollout_id,
                window_size=self.window_size,
            )

        self.histories[rollout_id].add(snapshot)
        self._total_samples += 1

    def get_idle_rollouts(self) -> list[int]:
        """Return IDs of rollouts detected as idle (repack candidates)."""
        idle = []
        for rollout_id, history in self.histories.items():
            if self.detector.is_idle(history):
                idle.append(rollout_id)
                self._idle_detections += 1
        return idle

    def get_idleness_ranking(self) -> list[tuple[int, float]]:
        """Return (rollout_id, idleness_score) sorted by most idle first."""
        scores = []
        for rollout_id, history in self.histories.items():
            score = self.detector.compute_idleness_score(history)
            scores.append((rollout_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def get_utilization_summary(self) -> dict[str, float]:
        """Return aggregate utilization stats across all rollouts."""
        utils = []
        for history in self.histories.values():
            if history.latest is not None:
                utils.append(history.latest.utilization)

        if not utils:
            return {"avg_utilization": 0.0, "min_utilization": 0.0, "max_utilization": 0.0}

        return {
            "avg_utilization": float(np.mean(utils)),
            "min_utilization": float(np.min(utils)),
            "max_utilization": float(np.max(utils)),
            "std_utilization": float(np.std(utils)),
            "num_monitored": float(len(utils)),
        }

    def get_rollout_phase(self, rollout_id: int) -> RolloutPhase:
        history = self.histories.get(rollout_id)
        if history is None or history.latest is None:
            return RolloutPhase.IDLE
        return history.latest.phase

    def stats(self) -> dict[str, Any]:
        return {
            "num_rollouts": self.num_rollouts,
            "total_samples": self._total_samples,
            "idle_detections": self._idle_detections,
            "utilization": self.get_utilization_summary(),
        }
