"""Trajectory scheduling and staleness-bounded dispatch.

Extends the basic rollout controller with production-grade scheduling
features: priority queues, machine-aware dispatch, staleness estimation,
and adaptive concurrency control.
"""

from __future__ import annotations

import asyncio
import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(order=True, slots=True)
class ScheduledPrompt:
    """A prompt entry in the priority scheduling queue."""

    priority: float
    prompt_id: int = field(compare=False)
    difficulty: int = field(compare=False)
    estimated_tokens: int = field(compare=False)
    created_at: float = field(compare=False, default_factory=time.time)
    max_retries: int = field(compare=False, default=3)
    retry_count: int = field(compare=False, default=0)


class StalenessBoundScheduler:
    """Trajectory-level staleness-aware scheduler.

    Under our fully decoupled design the staleness of each trajectory
    is determined by the gap between the model version used for
    generation and the current trainer version.  This scheduler
    tracks per-trajectory staleness and uses it to prioritize which
    completed trajectories should be consumed first.
    """

    def __init__(
        self,
        max_staleness: int = 4,
        staleness_decay: float = 0.95,
    ) -> None:
        self.max_staleness = max_staleness
        self.staleness_decay = staleness_decay
        self._trajectory_versions: dict[str, int] = {}
        self._staleness_violations = 0
        self._total_checked = 0

    def register_trajectory(self, trajectory_id: str, model_version: int) -> None:
        self._trajectory_versions[trajectory_id] = model_version

    def check_staleness(self, trajectory_id: str, current_version: int) -> float:
        """Return the staleness of a trajectory (version gap)."""
        self._total_checked += 1
        gen_version = self._trajectory_versions.get(trajectory_id, current_version)
        staleness = current_version - gen_version
        if staleness > self.max_staleness:
            self._staleness_violations += 1
        return float(staleness)

    def compute_importance_weight(
        self,
        gen_version: int,
        current_version: int,
    ) -> float:
        """Compute a staleness-aware importance weight for off-policy correction."""
        staleness = current_version - gen_version
        return self.staleness_decay ** max(staleness, 0)

    def should_discard(self, trajectory_id: str, current_version: int) -> bool:
        """Check if a trajectory is too stale to be useful."""
        staleness = self.check_staleness(trajectory_id, current_version)
        return staleness > self.max_staleness * 2

    def unregister(self, trajectory_id: str) -> None:
        self._trajectory_versions.pop(trajectory_id, None)

    def stats(self) -> dict[str, float]:
        return {
            "tracked_trajectories": float(len(self._trajectory_versions)),
            "staleness_violations": float(self._staleness_violations),
            "total_checked": float(self._total_checked),
            "violation_rate": (
                self._staleness_violations / max(self._total_checked, 1)
            ),
        }


class AdaptiveConcurrencyController:
    """Dynamically adjusts the maximum number of concurrent rollouts.

    Monitors buffer fill rate and training consumption rate to avoid
    both buffer starvation and overflow.  Increases concurrency when
    the buffer is draining faster than it fills, and decreases when
    the buffer is nearing capacity.
    """

    def __init__(
        self,
        initial_concurrency: int = 48,
        min_concurrency: int = 8,
        max_concurrency: int = 256,
        buffer_low_watermark: float = 0.25,
        buffer_high_watermark: float = 0.75,
        adjustment_step: int = 4,
        cooldown_sec: float = 2.0,
    ) -> None:
        self.current_concurrency = initial_concurrency
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.buffer_low_watermark = buffer_low_watermark
        self.buffer_high_watermark = buffer_high_watermark
        self.adjustment_step = adjustment_step
        self.cooldown_sec = cooldown_sec

        self._last_adjustment_time = 0.0
        self._adjustments: list[tuple[float, int]] = []
        self._history: list[dict[str, float]] = []

    def adjust(self, buffer_fill_ratio: float) -> int:
        """Adjust concurrency based on buffer fill ratio."""
        now = time.time()
        if now - self._last_adjustment_time < self.cooldown_sec:
            return self.current_concurrency

        old = self.current_concurrency

        if buffer_fill_ratio < self.buffer_low_watermark:
            # Buffer draining too fast — increase generation
            self.current_concurrency = min(
                self.current_concurrency + self.adjustment_step,
                self.max_concurrency,
            )
        elif buffer_fill_ratio > self.buffer_high_watermark:
            # Buffer filling too fast — slow down generation
            self.current_concurrency = max(
                self.current_concurrency - self.adjustment_step,
                self.min_concurrency,
            )

        if self.current_concurrency != old:
            self._last_adjustment_time = now
            self._adjustments.append((now, self.current_concurrency))

        self._history.append({
            "timestamp": now,
            "fill_ratio": buffer_fill_ratio,
            "concurrency": float(self.current_concurrency),
        })

        return self.current_concurrency

    def stats(self) -> dict[str, float]:
        return {
            "current_concurrency": float(self.current_concurrency),
            "total_adjustments": float(len(self._adjustments)),
            "min_concurrency": float(self.min_concurrency),
            "max_concurrency": float(self.max_concurrency),
        }


class MachineAwareDispatcher:
    """Dispatch prompts to machines considering load and locality.

    Uses a scoring function that balances:
    - Machine load (prefer less-loaded machines)
    - Model version freshness (prefer machines with newer weights)
    - Prompt difficulty (assign hard prompts to faster machines)
    """

    def __init__(
        self,
        num_machines: int,
        load_weight: float = 0.4,
        freshness_weight: float = 0.3,
        affinity_weight: float = 0.3,
    ) -> None:
        self.num_machines = num_machines
        self.load_weight = load_weight
        self.freshness_weight = freshness_weight
        self.affinity_weight = affinity_weight

        self._machine_loads: dict[int, int] = {i: 0 for i in range(num_machines)}
        self._machine_versions: dict[int, int] = {i: 0 for i in range(num_machines)}
        self._dispatch_counts: dict[int, int] = {i: 0 for i in range(num_machines)}
        self._max_load = 1  # Avoid division by zero

    def update_machine_state(
        self,
        machine_id: int,
        active_rollouts: int,
        model_version: int,
    ) -> None:
        self._machine_loads[machine_id] = active_rollouts
        self._machine_versions[machine_id] = model_version
        self._max_load = max(self._max_load, active_rollouts)

    def _score_machine(
        self,
        machine_id: int,
        prompt_difficulty: int,
        current_version: int,
    ) -> float:
        # Lower load → higher score
        load = self._machine_loads.get(machine_id, 0)
        load_score = 1.0 - (load / max(self._max_load, 1))

        # Fresher version → higher score
        version = self._machine_versions.get(machine_id, 0)
        version_gap = max(0, current_version - version)
        freshness_score = 1.0 / (1.0 + version_gap)

        # Affinity: assign harder prompts to less-loaded machines
        affinity_score = load_score * (1.0 + prompt_difficulty * 0.2)

        return (
            self.load_weight * load_score
            + self.freshness_weight * freshness_score
            + self.affinity_weight * affinity_score
        )

    def select_machine(
        self,
        prompt_difficulty: int = 0,
        current_version: int = 0,
        exclude: set[int] | None = None,
    ) -> int:
        best_machine = 0
        best_score = -float("inf")

        for mid in range(self.num_machines):
            if exclude and mid in exclude:
                continue
            score = self._score_machine(mid, prompt_difficulty, current_version)
            if score > best_score:
                best_score = score
                best_machine = mid

        self._dispatch_counts[best_machine] = (
            self._dispatch_counts.get(best_machine, 0) + 1
        )
        return best_machine

    def stats(self) -> dict[str, Any]:
        return {
            "dispatch_counts": dict(self._dispatch_counts),
            "machine_loads": dict(self._machine_loads),
            "machine_versions": dict(self._machine_versions),
        }


class TrajectoryScheduler:
    """Top-level scheduler combining all scheduling components."""

    def __init__(
        self,
        num_machines: int,
        max_concurrent: int = 48,
        max_staleness: int = 4,
        buffer_capacity: int = 256,
    ) -> None:
        self.staleness_scheduler = StalenessBoundScheduler(
            max_staleness=max_staleness,
        )
        self.concurrency_controller = AdaptiveConcurrencyController(
            initial_concurrency=max_concurrent,
        )
        self.dispatcher = MachineAwareDispatcher(
            num_machines=num_machines,
        )

        self.buffer_capacity = buffer_capacity
        self._queue: list[ScheduledPrompt] = []
        self._total_scheduled = 0
        self._total_dispatched = 0

    def enqueue(self, prompt: ScheduledPrompt) -> None:
        heapq.heappush(self._queue, prompt)
        self._total_scheduled += 1

    def dequeue(self) -> ScheduledPrompt | None:
        if not self._queue:
            return None
        prompt = heapq.heappop(self._queue)
        self._total_dispatched += 1
        return prompt

    def can_dispatch(
        self,
        in_flight: int,
        buffer_size: int,
    ) -> bool:
        max_conc = self.concurrency_controller.adjust(
            buffer_size / max(self.buffer_capacity, 1)
        )
        return in_flight < max_conc and len(self._queue) > 0

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    def stats(self) -> dict[str, Any]:
        return {
            "queue_size": self.queue_size,
            "total_scheduled": self._total_scheduled,
            "total_dispatched": self._total_dispatched,
            "staleness": self.staleness_scheduler.stats(),
            "concurrency": self.concurrency_controller.stats(),
            "dispatcher": self.dispatcher.stats(),
        }
