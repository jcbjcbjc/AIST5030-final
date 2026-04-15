"""Rollout manager: orchestrates rollout execution, failure recovery, and repack.

Sits above individual rollout workers and coordinates the full
lifecycle of trajectory generation including dispatch, monitoring,
failure detection, and the repack mechanism.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .data_module import PartialResponse, PartialResponsePool
from .kvcache import KVCacheMonitor, RolloutPhase
from .repack import InProgressTrajectory, RepackManager, RepackResult


class RolloutStatus:
    IDLE = "idle"
    GENERATING = "generating"
    WAITING_WEIGHTS = "waiting_weights"
    REPACKING = "repacking"
    FAILED = "failed"


@dataclass(slots=True)
class RolloutSlot:
    """Tracks the state of a single rollout slot on a machine."""

    slot_id: int
    machine_id: int
    status: str = RolloutStatus.IDLE
    current_trajectory_id: str | None = None
    current_prompt_id: int | None = None
    model_version: int = -1
    tokens_generated: int = 0
    max_tokens: int = 0
    started_at: float = 0.0
    completed_count: int = 0
    failed_count: int = 0
    total_generation_time: float = 0.0


@dataclass(slots=True)
class MachineInfo:
    """Information about a rollout machine."""

    machine_id: int
    num_gpus: int
    rollout_slots: list[RolloutSlot] = field(default_factory=list)
    is_healthy: bool = True
    kvcache_utilization: float = 0.0
    total_completed: int = 0
    total_failed: int = 0

    @property
    def active_slots(self) -> int:
        return sum(1 for s in self.rollout_slots if s.status == RolloutStatus.GENERATING)

    @property
    def idle_slots(self) -> int:
        return sum(1 for s in self.rollout_slots if s.status == RolloutStatus.IDLE)


class RolloutManager:
    """Manages rollout execution across multiple machines.

    Responsibilities:
    - Dispatch prompts to available rollout slots
    - Monitor generation progress via partial response pool
    - Coordinate with KVCache monitor for repack decisions
    - Handle rollout failures and reassignment
    - Collect metrics for the training loop
    """

    def __init__(
        self,
        num_machines: int,
        slots_per_machine: int,
        num_gpus_per_machine: int = 4,
        config: ExperimentConfig | None = None,
    ) -> None:
        self.num_machines = num_machines
        self.slots_per_machine = slots_per_machine
        self.config = config

        self.machines: dict[int, MachineInfo] = {}
        for mid in range(num_machines):
            slots = [
                RolloutSlot(
                    slot_id=mid * slots_per_machine + j,
                    machine_id=mid,
                )
                for j in range(slots_per_machine)
            ]
            self.machines[mid] = MachineInfo(
                machine_id=mid,
                num_gpus=num_gpus_per_machine,
                rollout_slots=slots,
            )

        self.partial_pool = PartialResponsePool(
            max_size=num_machines * slots_per_machine * 2,
        )

        self.kvcache_monitor = KVCacheMonitor(
            num_rollouts=num_machines * slots_per_machine,
        )

        self.repack_manager = RepackManager(
            kvcache_monitor=self.kvcache_monitor,
        )

        # Metrics
        self._total_dispatched = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_repacks = 0
        self._dispatch_latencies: list[float] = []
        self._lock = asyncio.Lock()

    def _find_idle_slot(self) -> RolloutSlot | None:
        """Find an idle slot using round-robin across machines."""
        for machine in self.machines.values():
            if not machine.is_healthy:
                continue
            for slot in machine.rollout_slots:
                if slot.status == RolloutStatus.IDLE:
                    return slot
        return None

    def _find_slot_on_machine(self, machine_id: int) -> RolloutSlot | None:
        machine = self.machines.get(machine_id)
        if machine is None or not machine.is_healthy:
            return None
        for slot in machine.rollout_slots:
            if slot.status == RolloutStatus.IDLE:
                return slot
        return None

    async def dispatch_prompt(
        self,
        prompt_id: int,
        prompt_text: str,
        max_tokens: int,
        model_version: int,
        preferred_machine: int | None = None,
    ) -> str | None:
        """Dispatch a prompt to an available rollout slot.

        Returns the trajectory_id on success, None if no slot available.
        """
        async with self._lock:
            start = time.perf_counter()

            if preferred_machine is not None:
                slot = self._find_slot_on_machine(preferred_machine)
            else:
                slot = self._find_idle_slot()

            if slot is None:
                return None

            trajectory_id = f"traj_{uuid.uuid4().hex[:12]}"
            slot.status = RolloutStatus.GENERATING
            slot.current_trajectory_id = trajectory_id
            slot.current_prompt_id = prompt_id
            slot.model_version = model_version
            slot.tokens_generated = 0
            slot.max_tokens = max_tokens
            slot.started_at = time.perf_counter()

            # Register in partial response pool
            partial = PartialResponse(
                trajectory_id=trajectory_id,
                prompt_id=prompt_id,
                rollout_id=slot.slot_id,
                tokens_so_far=0,
                max_tokens=max_tokens,
                model_version=model_version,
                started_at=time.time(),
                last_updated=time.time(),
            )
            await self.partial_pool.register(partial)

            # Register in repack manager
            in_progress = InProgressTrajectory(
                trajectory_id=trajectory_id,
                rollout_id=slot.slot_id,
                prompt_id=prompt_id,
                tokens_generated=0,
                max_tokens=max_tokens,
                kvcache_usage=0.0,
                model_version=model_version,
                started_at=time.time(),
            )
            self.repack_manager.register_trajectory(in_progress)

            self._total_dispatched += 1
            dispatch_latency = time.perf_counter() - start
            self._dispatch_latencies.append(dispatch_latency)

            return trajectory_id

    async def report_progress(
        self,
        trajectory_id: str,
        tokens_generated: int,
        kvcache_usage: float,
        model_version: int | None = None,
    ) -> None:
        """Report generation progress for a trajectory."""
        await self.partial_pool.update(
            trajectory_id, tokens_generated, model_version
        )
        self.repack_manager.update_trajectory(
            trajectory_id, tokens_generated, kvcache_usage
        )

    async def complete_trajectory(self, trajectory_id: str) -> None:
        """Mark a trajectory as completed and free the slot."""
        async with self._lock:
            await self.partial_pool.complete(trajectory_id)
            self.repack_manager.unregister_trajectory(trajectory_id)

            for machine in self.machines.values():
                for slot in machine.rollout_slots:
                    if slot.current_trajectory_id == trajectory_id:
                        generation_time = time.perf_counter() - slot.started_at
                        slot.status = RolloutStatus.IDLE
                        slot.current_trajectory_id = None
                        slot.current_prompt_id = None
                        slot.completed_count += 1
                        slot.total_generation_time += generation_time
                        machine.total_completed += 1
                        self._total_completed += 1
                        return

    async def fail_trajectory(self, trajectory_id: str, reason: str = "") -> None:
        """Mark a trajectory as failed."""
        async with self._lock:
            await self.partial_pool.complete(trajectory_id)
            self.repack_manager.unregister_trajectory(trajectory_id)

            for machine in self.machines.values():
                for slot in machine.rollout_slots:
                    if slot.current_trajectory_id == trajectory_id:
                        slot.status = RolloutStatus.IDLE
                        slot.current_trajectory_id = None
                        slot.failed_count += 1
                        machine.total_failed += 1
                        self._total_failed += 1
                        return

    async def handle_machine_failure(self, machine_id: int) -> list[str]:
        """Handle a machine failure: mark all its slots as failed."""
        async with self._lock:
            machine = self.machines.get(machine_id)
            if machine is None:
                return []

            machine.is_healthy = False
            affected = []
            for slot in machine.rollout_slots:
                if slot.current_trajectory_id is not None:
                    affected.append(slot.current_trajectory_id)
                    slot.status = RolloutStatus.FAILED
                    slot.failed_count += 1

            return affected

    async def maybe_repack(self) -> RepackResult | None:
        """Check and execute repack if conditions are met."""
        result = await self.repack_manager.maybe_repack()
        if result is not None:
            self._total_repacks += 1
        return result

    def update_kvcache(
        self,
        rollout_id: int,
        utilization: float,
        num_sequences: int,
        tokens_cached: int,
    ) -> None:
        """Update KVCache metrics for a rollout."""
        self.kvcache_monitor.record_utilization(
            rollout_id=rollout_id,
            utilization=utilization,
            num_active_sequences=num_sequences,
            total_tokens_cached=tokens_cached,
            phase=RolloutPhase.ACTIVE if utilization > 0.1 else RolloutPhase.IDLE,
        )

    def get_cluster_summary(self) -> dict[str, float]:
        total_slots = sum(len(m.rollout_slots) for m in self.machines.values())
        active_slots = sum(m.active_slots for m in self.machines.values())
        idle_slots = sum(m.idle_slots for m in self.machines.values())
        healthy_machines = sum(1 for m in self.machines.values() if m.is_healthy)

        return {
            "total_machines": float(self.num_machines),
            "healthy_machines": float(healthy_machines),
            "total_slots": float(total_slots),
            "active_slots": float(active_slots),
            "idle_slots": float(idle_slots),
            "slot_utilization": active_slots / max(total_slots, 1),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "cluster": self.get_cluster_summary(),
            "total_dispatched": self._total_dispatched,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "total_repacks": self._total_repacks,
            "avg_dispatch_latency_sec": (
                float(np.mean(self._dispatch_latencies))
                if self._dispatch_latencies
                else 0.0
            ),
            "partial_pool": self.partial_pool.stats(),
            "kvcache_monitor": self.kvcache_monitor.stats(),
            "repack_manager": self.repack_manager.stats(),
        }
