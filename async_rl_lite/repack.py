"""Dynamic repack mechanism for consolidating long-tail trajectories.

When a few straggler rollouts hold GPU resources while generating
long-tail trajectories, the repack mechanism transfers their
in-progress work to fewer destination rollouts so that freed
machines can begin processing new prompts.

The algorithm is inspired by Best-Fit bin-packing: each destination
rollout is treated as a bin with capacity determined by its KVCache
headroom, and source trajectories are items to be placed.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .kvcache import KVCacheMonitor, RolloutPhase


@dataclass(slots=True)
class InProgressTrajectory:
    """Metadata for a trajectory that is still being generated."""

    trajectory_id: str
    rollout_id: int
    prompt_id: int
    tokens_generated: int
    max_tokens: int
    kvcache_usage: float            # fraction of rollout's KVCache
    model_version: int
    started_at: float
    estimated_remaining_tokens: int = 0

    @property
    def progress(self) -> float:
        return self.tokens_generated / max(self.max_tokens, 1)

    @property
    def kvcache_footprint(self) -> float:
        return self.kvcache_usage

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.started_at


@dataclass(slots=True)
class RepackPlan:
    """A single move: transfer a trajectory from source to destination."""

    trajectory_id: str
    source_rollout: int
    destination_rollout: int
    estimated_kvcache_cost: float
    model_version: int


@dataclass(slots=True)
class RepackResult:
    """Outcome of executing a repack plan."""

    plan: list[RepackPlan]
    trajectories_moved: int
    rollouts_freed: int
    total_time_sec: float
    kvcache_utilization_delta: float
    success: bool


class BestFitPacker:
    """Best-Fit bin-packing algorithm for trajectory consolidation.

    Each destination rollout has a KVCache capacity (headroom).
    Each source trajectory has a KVCache footprint.  We greedily
    assign each trajectory to the destination that will become
    most densely packed after the transfer, preserving capacity
    in other destinations for larger workloads.
    """

    def __init__(
        self,
        max_kvcache_utilization: float = 0.90,
        max_requests_per_rollout: int = 8,
    ) -> None:
        self.max_kvcache_utilization = max_kvcache_utilization
        self.max_requests_per_rollout = max_requests_per_rollout

    def can_fit(
        self,
        trajectory: InProgressTrajectory,
        dest_current_util: float,
        dest_current_requests: int,
    ) -> bool:
        projected_util = dest_current_util + trajectory.kvcache_footprint
        projected_requests = dest_current_requests + 1
        return (
            projected_util <= self.max_kvcache_utilization
            and projected_requests <= self.max_requests_per_rollout
        )

    def pack(
        self,
        source_trajectories: list[InProgressTrajectory],
        destination_rollouts: list[dict[str, Any]],
    ) -> list[RepackPlan]:
        """Generate a packing plan.

        Parameters
        ----------
        source_trajectories : list
            Trajectories to relocate, sorted smallest-footprint first.
        destination_rollouts : list
            Each dict has ``rollout_id``, ``current_utilization``,
            ``current_requests``.

        Returns
        -------
        list[RepackPlan]
            Sequence of moves to execute.
        """
        # Sort sources by footprint ascending (small items first = better packing)
        sources = sorted(source_trajectories, key=lambda t: t.kvcache_footprint)

        # Track mutable state per destination
        dest_state = {
            d["rollout_id"]: {
                "util": d["current_utilization"],
                "reqs": d["current_requests"],
            }
            for d in destination_rollouts
        }

        plan: list[RepackPlan] = []

        for traj in sources:
            best_dest: int | None = None
            best_remaining: float = float("inf")

            for d in destination_rollouts:
                did = d["rollout_id"]
                state = dest_state[did]
                if did == traj.rollout_id:
                    continue  # Cannot repack to self
                if not self.can_fit(traj, state["util"], state["reqs"]):
                    continue

                remaining = self.max_kvcache_utilization - (state["util"] + traj.kvcache_footprint)
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_dest = did

            if best_dest is not None:
                plan.append(
                    RepackPlan(
                        trajectory_id=traj.trajectory_id,
                        source_rollout=traj.rollout_id,
                        destination_rollout=best_dest,
                        estimated_kvcache_cost=traj.kvcache_footprint,
                        model_version=traj.model_version,
                    )
                )
                dest_state[best_dest]["util"] += traj.kvcache_footprint
                dest_state[best_dest]["reqs"] += 1

        return plan


class RepackTrigger:
    """Decides when to trigger the repack mechanism.

    Monitors the fraction of idle rollouts and triggers repack when
    a configurable threshold is exceeded, with a cooldown to prevent
    excessive repacking.
    """

    def __init__(
        self,
        idle_fraction_threshold: float = 0.25,
        cooldown_sec: float = 5.0,
        min_idle_rollouts: int = 2,
    ) -> None:
        self.idle_fraction_threshold = idle_fraction_threshold
        self.cooldown_sec = cooldown_sec
        self.min_idle_rollouts = min_idle_rollouts
        self._last_trigger_time: float = 0.0
        self._trigger_count = 0

    def should_trigger(
        self,
        num_idle: int,
        total_rollouts: int,
    ) -> bool:
        if num_idle < self.min_idle_rollouts:
            return False
        if total_rollouts == 0:
            return False
        idle_fraction = num_idle / total_rollouts
        if idle_fraction < self.idle_fraction_threshold:
            return False
        elapsed = time.time() - self._last_trigger_time
        if elapsed < self.cooldown_sec:
            return False
        return True

    def record_trigger(self) -> None:
        self._last_trigger_time = time.time()
        self._trigger_count += 1


class RepackManager:
    """Top-level manager for the repack mechanism.

    Coordinates between KVCache monitoring, the packing algorithm,
    and the actual trajectory transfer.
    """

    def __init__(
        self,
        kvcache_monitor: KVCacheMonitor,
        packer: BestFitPacker | None = None,
        trigger: RepackTrigger | None = None,
    ) -> None:
        self.kvcache_monitor = kvcache_monitor
        self.packer = packer or BestFitPacker()
        self.trigger = trigger or RepackTrigger()

        self._in_progress: dict[str, InProgressTrajectory] = {}
        self._repack_history: list[RepackResult] = []
        self._total_moves = 0
        self._total_rollouts_freed = 0
        self._total_repack_time = 0.0
        self._lock = asyncio.Lock()

    def register_trajectory(self, traj: InProgressTrajectory) -> None:
        self._in_progress[traj.trajectory_id] = traj

    def unregister_trajectory(self, trajectory_id: str) -> None:
        self._in_progress.pop(trajectory_id, None)

    def update_trajectory(
        self,
        trajectory_id: str,
        tokens_generated: int,
        kvcache_usage: float,
    ) -> None:
        if trajectory_id in self._in_progress:
            self._in_progress[trajectory_id].tokens_generated = tokens_generated
            self._in_progress[trajectory_id].kvcache_usage = kvcache_usage

    async def maybe_repack(self) -> RepackResult | None:
        """Check whether repack is needed and execute if so."""
        idle_rollouts = self.kvcache_monitor.get_idle_rollouts()
        if not self.trigger.should_trigger(
            num_idle=len(idle_rollouts),
            total_rollouts=self.kvcache_monitor.num_rollouts,
        ):
            return None

        return await self.execute_repack(idle_rollouts)

    async def execute_repack(self, source_rollout_ids: list[int]) -> RepackResult:
        """Execute the full repack pipeline."""
        async with self._lock:
            start = time.perf_counter()
            self.trigger.record_trigger()

            # Gather source trajectories from idle rollouts
            source_trajectories = [
                traj
                for traj in self._in_progress.values()
                if traj.rollout_id in source_rollout_ids
            ]

            # Identify candidate destinations (non-idle rollouts with headroom)
            all_rollout_ids = set(range(self.kvcache_monitor.num_rollouts))
            dest_ids = all_rollout_ids - set(source_rollout_ids)
            destination_rollouts = []
            for rid in dest_ids:
                history = self.kvcache_monitor.histories.get(rid)
                util = 0.0
                reqs = 0
                if history and history.latest:
                    util = history.latest.utilization
                    reqs = history.latest.num_active_sequences
                destination_rollouts.append({
                    "rollout_id": rid,
                    "current_utilization": util,
                    "current_requests": reqs,
                })

            # Run packing algorithm
            plan = self.packer.pack(source_trajectories, destination_rollouts)

            # Execute moves (simulated)
            moved_rollouts: set[int] = set()
            for move in plan:
                # In production: transfer KVCache state via network
                if move.trajectory_id in self._in_progress:
                    self._in_progress[move.trajectory_id].rollout_id = (
                        move.destination_rollout
                    )
                    moved_rollouts.add(move.source_rollout)

            # Compute KVCache delta
            util_before = self.kvcache_monitor.get_utilization_summary()
            elapsed = time.perf_counter() - start

            result = RepackResult(
                plan=plan,
                trajectories_moved=len(plan),
                rollouts_freed=len(moved_rollouts),
                total_time_sec=elapsed,
                kvcache_utilization_delta=0.0,  # Would be measured in production
                success=True,
            )

            self._repack_history.append(result)
            self._total_moves += len(plan)
            self._total_rollouts_freed += len(moved_rollouts)
            self._total_repack_time += elapsed

            return result

    def stats(self) -> dict[str, Any]:
        return {
            "in_progress_trajectories": len(self._in_progress),
            "total_repacks": len(self._repack_history),
            "total_moves": self._total_moves,
            "total_rollouts_freed": self._total_rollouts_freed,
            "total_repack_time_sec": self._total_repack_time,
            "avg_repack_time_sec": (
                self._total_repack_time / max(len(self._repack_history), 1)
            ),
            "avg_moves_per_repack": (
                self._total_moves / max(len(self._repack_history), 1)
            ),
            "trigger_count": self.trigger._trigger_count,
        }
