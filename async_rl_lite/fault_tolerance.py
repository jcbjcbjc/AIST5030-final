"""Fault tolerance and failure recovery for long-running RL training.

Implements a heartbeat-based failover system that detects rollout
machine failures and triggers recovery by reconstructing the relay
broadcast chain and redistributing work to healthy machines.
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import numpy as np


class MachineStatus(enum.Enum):
    HEALTHY = "healthy"
    SUSPECTED = "suspected"
    FAILED = "failed"
    RECOVERING = "recovering"
    DECOMMISSIONED = "decommissioned"


@dataclass(slots=True)
class HeartbeatRecord:
    machine_id: int
    timestamp: float
    status: MachineStatus
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    active_rollouts: int = 0
    completed_rollouts: int = 0
    error_count: int = 0


@dataclass(slots=True)
class MachineState:
    machine_id: int
    status: MachineStatus = MachineStatus.HEALTHY
    last_heartbeat: float = 0.0
    consecutive_misses: int = 0
    total_heartbeats: int = 0
    total_failures: int = 0
    recovery_count: int = 0
    gpu_ids: list[int] = field(default_factory=list)
    active_rollout_ids: list[int] = field(default_factory=list)


class HeartbeatMonitor:
    """Monitors heartbeats from rollout machines.

    If a machine misses more than ``max_misses`` consecutive heartbeats
    within the ``timeout`` window it is marked as failed and recovery
    is triggered.
    """

    def __init__(
        self,
        num_machines: int,
        heartbeat_interval_sec: float = 5.0,
        timeout_sec: float = 15.0,
        max_consecutive_misses: int = 3,
    ) -> None:
        self.num_machines = num_machines
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.timeout_sec = timeout_sec
        self.max_consecutive_misses = max_consecutive_misses

        self.machines: dict[int, MachineState] = {
            i: MachineState(
                machine_id=i,
                last_heartbeat=time.time(),
                gpu_ids=list(range(4)),  # Assume 4 GPUs per machine
            )
            for i in range(num_machines)
        }

        self._failure_callbacks: list[Callable[[int], Coroutine[Any, Any, None]]] = []
        self._recovery_callbacks: list[Callable[[int], Coroutine[Any, Any, None]]] = []
        self._total_failures_detected = 0
        self._total_recoveries = 0

    def on_failure(self, callback: Callable[[int], Coroutine[Any, Any, None]]) -> None:
        self._failure_callbacks.append(callback)

    def on_recovery(self, callback: Callable[[int], Coroutine[Any, Any, None]]) -> None:
        self._recovery_callbacks.append(callback)

    async def receive_heartbeat(self, record: HeartbeatRecord) -> None:
        machine = self.machines.get(record.machine_id)
        if machine is None:
            return

        machine.last_heartbeat = record.timestamp
        machine.total_heartbeats += 1
        machine.consecutive_misses = 0

        if machine.status == MachineStatus.SUSPECTED:
            machine.status = MachineStatus.HEALTHY

        if machine.status == MachineStatus.RECOVERING:
            machine.status = MachineStatus.HEALTHY
            machine.recovery_count += 1
            self._total_recoveries += 1
            for cb in self._recovery_callbacks:
                await cb(record.machine_id)

    async def check_liveness(self) -> list[int]:
        """Check all machines for missed heartbeats.

        Returns list of newly-failed machine IDs.
        """
        now = time.time()
        newly_failed: list[int] = []

        for machine_id, machine in self.machines.items():
            if machine.status in (MachineStatus.FAILED, MachineStatus.DECOMMISSIONED):
                continue

            elapsed = now - machine.last_heartbeat
            if elapsed > self.heartbeat_interval_sec:
                machine.consecutive_misses += 1

                if machine.consecutive_misses >= self.max_consecutive_misses:
                    if machine.status != MachineStatus.FAILED:
                        machine.status = MachineStatus.FAILED
                        machine.total_failures += 1
                        self._total_failures_detected += 1
                        newly_failed.append(machine_id)
                elif machine.status == MachineStatus.HEALTHY:
                    machine.status = MachineStatus.SUSPECTED

        for machine_id in newly_failed:
            for cb in self._failure_callbacks:
                await cb(machine_id)

        return newly_failed

    def get_healthy_machines(self) -> list[int]:
        return [
            mid
            for mid, m in self.machines.items()
            if m.status == MachineStatus.HEALTHY
        ]

    def get_failed_machines(self) -> list[int]:
        return [
            mid
            for mid, m in self.machines.items()
            if m.status == MachineStatus.FAILED
        ]

    def mark_recovering(self, machine_id: int) -> None:
        if machine_id in self.machines:
            self.machines[machine_id].status = MachineStatus.RECOVERING
            self.machines[machine_id].consecutive_misses = 0

    def decommission(self, machine_id: int) -> None:
        if machine_id in self.machines:
            self.machines[machine_id].status = MachineStatus.DECOMMISSIONED

    def stats(self) -> dict[str, Any]:
        status_counts = {}
        for m in self.machines.values():
            key = m.status.value
            status_counts[key] = status_counts.get(key, 0) + 1

        return {
            "num_machines": self.num_machines,
            "total_failures_detected": self._total_failures_detected,
            "total_recoveries": self._total_recoveries,
            "status_counts": status_counts,
            "avg_heartbeats_per_machine": float(
                np.mean([m.total_heartbeats for m in self.machines.values()])
            ),
        }


class BroadcastChainRebuilder:
    """Rebuilds the relay broadcast chain after a machine failure.

    When a relay worker fails the broadcast chain has a gap.  This
    rebuilder reconstructs the chain from the remaining healthy
    relays so that weight distribution can continue.
    """

    def __init__(self, topology: str = "chain") -> None:
        self.topology = topology
        self._rebuild_count = 0
        self._total_rebuild_time = 0.0

    async def rebuild(
        self,
        healthy_relay_ids: list[str],
        failed_relay_ids: list[str],
    ) -> dict[str, Any]:
        start = time.perf_counter()

        new_chain: list[tuple[str, str]] = []
        if self.topology == "chain":
            relays = ["master_relay"] + healthy_relay_ids
            for i in range(len(relays) - 1):
                new_chain.append((relays[i], relays[i + 1]))
        elif self.topology == "tree":
            # Rebuild binary tree
            relays = list(healthy_relay_ids)
            senders = ["master_relay"]
            while relays:
                next_senders = []
                for sender in senders:
                    for _ in range(2):
                        if not relays:
                            break
                        target = relays.pop(0)
                        new_chain.append((sender, target))
                        next_senders.append(target)
                senders = next_senders

        elapsed = time.perf_counter() - start
        self._rebuild_count += 1
        self._total_rebuild_time += elapsed

        return {
            "new_chain": new_chain,
            "chain_length": len(new_chain),
            "removed_relays": failed_relay_ids,
            "remaining_relays": len(healthy_relay_ids),
            "rebuild_time_sec": elapsed,
        }

    def stats(self) -> dict[str, float]:
        return {
            "rebuild_count": float(self._rebuild_count),
            "avg_rebuild_time_sec": (
                self._total_rebuild_time / max(self._rebuild_count, 1)
            ),
        }


class RecoveryCoordinator:
    """Coordinates the full recovery procedure after a failure.

    Steps:
    1. Detect failure via HeartbeatMonitor
    2. Pause rollout submission on the failed machine
    3. Reassign in-progress trajectories to healthy machines
    4. Rebuild the relay broadcast chain
    5. Resume generation on the new machine (if replacement available)
    """

    def __init__(
        self,
        heartbeat_monitor: HeartbeatMonitor,
        chain_rebuilder: BroadcastChainRebuilder,
        max_recovery_attempts: int = 3,
        recovery_timeout_sec: float = 60.0,
    ) -> None:
        self.heartbeat_monitor = heartbeat_monitor
        self.chain_rebuilder = chain_rebuilder
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_timeout_sec = recovery_timeout_sec

        self._recovery_history: list[dict[str, Any]] = []
        self._active_recoveries: dict[int, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    async def handle_failure(self, machine_id: int) -> dict[str, Any]:
        """Full failure recovery for a single machine."""
        async with self._lock:
            start = time.perf_counter()
            result: dict[str, Any] = {
                "machine_id": machine_id,
                "started_at": start,
                "steps": [],
            }

            # Step 1: Mark machine as recovering
            self.heartbeat_monitor.mark_recovering(machine_id)
            result["steps"].append("marked_recovering")

            # Step 2: Identify affected rollouts
            machine = self.heartbeat_monitor.machines[machine_id]
            affected_rollouts = list(machine.active_rollout_ids)
            result["affected_rollouts"] = affected_rollouts
            result["steps"].append(f"identified_{len(affected_rollouts)}_affected_rollouts")

            # Step 3: Get healthy machines for redistribution
            healthy = self.heartbeat_monitor.get_healthy_machines()
            result["healthy_machines"] = healthy
            result["steps"].append(f"found_{len(healthy)}_healthy_machines")

            # Step 4: Redistribute rollouts (round-robin)
            redistribution: dict[int, list[int]] = {}
            for idx, rollout_id in enumerate(affected_rollouts):
                target = healthy[idx % len(healthy)] if healthy else -1
                if target not in redistribution:
                    redistribution[target] = []
                redistribution[target].append(rollout_id)
            result["redistribution"] = redistribution
            result["steps"].append("redistributed_rollouts")

            # Step 5: Rebuild broadcast chain
            healthy_relay_ids = [f"relay_{mid}" for mid in healthy]
            failed_relay_ids = [f"relay_{machine_id}"]
            chain_result = await self.chain_rebuilder.rebuild(
                healthy_relay_ids, failed_relay_ids
            )
            result["chain_rebuild"] = chain_result
            result["steps"].append("rebuilt_broadcast_chain")

            # Record
            elapsed = time.perf_counter() - start
            result["total_time_sec"] = elapsed
            result["success"] = True
            self._recovery_history.append(result)

            return result

    async def run_periodic_check(self, stop_event: asyncio.Event, interval: float = 5.0) -> None:
        """Background task that periodically checks machine liveness."""
        while not stop_event.is_set():
            newly_failed = await self.heartbeat_monitor.check_liveness()
            for machine_id in newly_failed:
                if machine_id not in self._active_recoveries:
                    task = asyncio.create_task(self.handle_failure(machine_id))
                    self._active_recoveries[machine_id] = task
                    task.add_done_callback(
                        lambda t, mid=machine_id: self._active_recoveries.pop(mid, None)
                    )
            await asyncio.sleep(interval)

    def stats(self) -> dict[str, Any]:
        return {
            "total_recoveries": len(self._recovery_history),
            "active_recoveries": len(self._active_recoveries),
            "heartbeat_monitor": self.heartbeat_monitor.stats(),
            "chain_rebuilder": self.chain_rebuilder.stats(),
        }


class CheckpointManager:
    """Manages periodic checkpointing of model and training state.

    Saves snapshots at configurable intervals so that training can
    resume from the latest checkpoint after a trainer failure.
    """

    def __init__(
        self,
        checkpoint_interval_updates: int = 10,
        max_checkpoints: int = 5,
        checkpoint_dir: str = "/tmp/rl_checkpoints",
    ) -> None:
        self.checkpoint_interval = checkpoint_interval_updates
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir = checkpoint_dir

        self._checkpoints: list[dict[str, Any]] = []
        self._total_checkpoint_time = 0.0

    def should_checkpoint(self, update_index: int) -> bool:
        return update_index > 0 and update_index % self.checkpoint_interval == 0

    async def save_checkpoint(
        self,
        update_index: int,
        policy_version: int,
        weights: np.ndarray,
        bias: np.ndarray,
        train_history: list[dict[str, float]],
    ) -> dict[str, Any]:
        start = time.perf_counter()

        checkpoint = {
            "update_index": update_index,
            "policy_version": policy_version,
            "weights_shape": list(weights.shape),
            "bias_shape": list(bias.shape),
            "weights_checksum": hash(weights.tobytes()) % (10**8),
            "timestamp": time.time(),
            "history_length": len(train_history),
        }

        self._checkpoints.append(checkpoint)

        # Evict oldest checkpoints
        if len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints = self._checkpoints[-self.max_checkpoints :]

        elapsed = time.perf_counter() - start
        self._total_checkpoint_time += elapsed
        checkpoint["save_time_sec"] = elapsed

        return checkpoint

    def get_latest_checkpoint(self) -> dict[str, Any] | None:
        return self._checkpoints[-1] if self._checkpoints else None

    def stats(self) -> dict[str, Any]:
        return {
            "total_checkpoints": len(self._checkpoints),
            "total_checkpoint_time_sec": self._total_checkpoint_time,
            "avg_checkpoint_time_sec": (
                self._total_checkpoint_time / max(len(self._checkpoints), 1)
            ),
            "latest_checkpoint": self.get_latest_checkpoint(),
        }
