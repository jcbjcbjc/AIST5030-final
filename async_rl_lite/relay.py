"""Relay worker system for asynchronous weight synchronization.

Implements the hierarchical relay worker architecture described in the
report.  The master relay receives updated weights from the trainer and
broadcasts them to colocated relay workers on each rollout machine.
Individual rollouts pull weights from their local relay over PCIe,
avoiding any global synchronization barrier.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .comm import (
    BroadcastProtocol,
    CommunicationManager,
    PCIeTransfer,
    RDMASimulator,
    TransferProtocol,
    TransferRequest,
    WeightPayload,
)
from .policy import PolicySnapshot


@dataclass(slots=True)
class RelayWeightEntry:
    """A cached weight snapshot held by a relay worker."""

    version: int
    snapshot: PolicySnapshot
    received_at: float
    access_count: int = 0
    last_accessed: float = 0.0


class RelayWorker:
    """A single relay worker colocated with rollout machines on CPU.

    Each relay worker maintains the latest model weights in CPU memory
    and serves weight-pull requests from colocated rollout processes
    over PCIe.  It receives weight updates from the master relay via
    RDMA broadcast.
    """

    def __init__(
        self,
        relay_id: str,
        machine_id: int,
        num_colocated_rollouts: int,
        pcie: PCIeTransfer,
        cache_size: int = 4,
    ) -> None:
        self.relay_id = relay_id
        self.machine_id = machine_id
        self.num_colocated_rollouts = num_colocated_rollouts
        self.pcie = pcie
        self.cache_size = cache_size

        self._weight_cache: dict[int, RelayWeightEntry] = {}
        self._latest_version: int = -1
        self._latest_snapshot: PolicySnapshot | None = None
        self._lock = asyncio.Lock()

        # Metrics
        self._updates_received = 0
        self._pull_requests = 0
        self._pull_served = 0
        self._pull_stale = 0
        self._total_serve_time = 0.0
        self._total_receive_time = 0.0

    async def receive_weights(self, version: int, snapshot: PolicySnapshot) -> None:
        """Receive a new weight snapshot from the master relay."""
        start = time.perf_counter()
        async with self._lock:
            entry = RelayWeightEntry(
                version=version,
                snapshot=snapshot.copy(),
                received_at=time.perf_counter(),
            )
            self._weight_cache[version] = entry
            if version > self._latest_version:
                self._latest_version = version
                self._latest_snapshot = snapshot.copy()

            # Evict old entries beyond cache_size
            if len(self._weight_cache) > self.cache_size:
                oldest = sorted(self._weight_cache.keys())
                for old_version in oldest[: len(self._weight_cache) - self.cache_size]:
                    del self._weight_cache[old_version]

            self._updates_received += 1
        self._total_receive_time += time.perf_counter() - start

    async def pull_weights(self, rollout_id: int, current_version: int) -> tuple[int, PolicySnapshot] | None:
        """Serve a weight-pull request from a colocated rollout.

        Returns the latest snapshot if it is newer than *current_version*,
        otherwise ``None``.  Simulates PCIe transfer latency for the copy
        from CPU host memory to the GPU.
        """
        start = time.perf_counter()
        async with self._lock:
            self._pull_requests += 1

            if self._latest_snapshot is None or self._latest_version <= current_version:
                self._pull_stale += 1
                return None

            snapshot_copy = self._latest_snapshot.copy()
            version = self._latest_version

            if version in self._weight_cache:
                self._weight_cache[version].access_count += 1
                self._weight_cache[version].last_accessed = time.perf_counter()

            self._pull_served += 1

        # Simulate PCIe transfer from CPU memory to GPU
        await self.pcie.host_to_device(snapshot_copy.weights)
        self._total_serve_time += time.perf_counter() - start

        return version, snapshot_copy

    async def get_latest_version(self) -> int:
        async with self._lock:
            return self._latest_version

    def stats(self) -> dict[str, float]:
        return {
            "relay_id_hash": float(hash(self.relay_id) % 10000),
            "machine_id": float(self.machine_id),
            "num_colocated_rollouts": float(self.num_colocated_rollouts),
            "latest_version": float(self._latest_version),
            "cache_size": float(len(self._weight_cache)),
            "updates_received": float(self._updates_received),
            "pull_requests": float(self._pull_requests),
            "pull_served": float(self._pull_served),
            "pull_stale": float(self._pull_stale),
            "pull_hit_rate": self._pull_served / max(self._pull_requests, 1),
            "avg_serve_time_sec": self._total_serve_time / max(self._pull_served, 1),
            "avg_receive_time_sec": self._total_receive_time / max(self._updates_received, 1),
        }


class MasterRelay:
    """Master relay worker that receives weights from the trainer.

    Acts as the root of the broadcast tree.  On receiving updated
    weights it immediately begins broadcasting to all other relay
    workers without blocking the trainer.
    """

    def __init__(
        self,
        relay_workers: list[RelayWorker],
        comm: CommunicationManager,
        broadcast_timeout: float = 30.0,
    ) -> None:
        self.relay_workers = relay_workers
        self.comm = comm
        self.broadcast_timeout = broadcast_timeout

        self._latest_version = -1
        self._broadcasts_started = 0
        self._broadcasts_completed = 0
        self._broadcasts_failed = 0
        self._total_broadcast_time = 0.0
        self._lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[None]] = set()

    async def receive_from_trainer(self, version: int, snapshot: PolicySnapshot) -> None:
        """Receive updated weights from the trainer, broadcast asynchronously."""
        async with self._lock:
            self._latest_version = version
            self._broadcasts_started += 1

        # Launch broadcast in background so trainer is not blocked
        task = asyncio.create_task(self._broadcast_to_relays(version, snapshot))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _broadcast_to_relays(self, version: int, snapshot: PolicySnapshot) -> None:
        """Broadcast weights to all relay workers."""
        start = time.perf_counter()
        try:
            tasks = [
                relay.receive_weights(version, snapshot)
                for relay in self.relay_workers
            ]
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.broadcast_timeout,
            )
            async with self._lock:
                self._broadcasts_completed += 1
        except asyncio.TimeoutError:
            async with self._lock:
                self._broadcasts_failed += 1
        finally:
            elapsed = time.perf_counter() - start
            async with self._lock:
                self._total_broadcast_time += elapsed

    async def wait_for_pending_broadcasts(self) -> None:
        """Wait for all in-flight background broadcasts to finish."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

    def stats(self) -> dict[str, float]:
        return {
            "latest_version": float(self._latest_version),
            "num_relay_workers": float(len(self.relay_workers)),
            "broadcasts_started": float(self._broadcasts_started),
            "broadcasts_completed": float(self._broadcasts_completed),
            "broadcasts_failed": float(self._broadcasts_failed),
            "avg_broadcast_time_sec": (
                self._total_broadcast_time / max(self._broadcasts_completed, 1)
            ),
            "pending_broadcasts": float(len(self._background_tasks)),
        }


class RelayWeightSyncService:
    """High-level orchestrator for the full relay-based weight sync pipeline.

    Wraps MasterRelay + RelayWorker hierarchy and provides a simple
    ``publish`` / ``pull`` interface matching the ParameterService API
    so that rollout workers and the trainer can use it transparently.
    """

    def __init__(
        self,
        num_machines: int,
        rollouts_per_machine: int = 2,
        pcie_gen: int = 5,
        rdma_bandwidth_gbps: float = 200.0,
        cache_size: int = 4,
    ) -> None:
        self.num_machines = num_machines
        self.rollouts_per_machine = rollouts_per_machine

        self.comm = CommunicationManager(
            num_rollout_machines=num_machines,
            num_trainer_machines=1,
            rdma_bandwidth_gbps=rdma_bandwidth_gbps,
            pcie_gen=pcie_gen,
        )

        pcie = PCIeTransfer(gen=pcie_gen)
        self.relay_workers = [
            RelayWorker(
                relay_id=f"relay_{i}",
                machine_id=i,
                num_colocated_rollouts=rollouts_per_machine,
                pcie=pcie,
                cache_size=cache_size,
            )
            for i in range(num_machines)
        ]

        self.master_relay = MasterRelay(
            relay_workers=self.relay_workers,
            comm=self.comm,
        )

        self._version = 0
        self._publish_count = 0

    async def publish(self, snapshot: PolicySnapshot) -> int:
        """Publish a new snapshot—mirrors ParameterService.publish."""
        self._version += 1
        self._publish_count += 1
        await self.master_relay.receive_from_trainer(self._version, snapshot)
        return self._version

    async def pull_if_new(
        self, machine_id: int, rollout_id: int, current_version: int
    ) -> tuple[int, PolicySnapshot] | None:
        """Pull weights from the local relay worker."""
        if machine_id >= len(self.relay_workers):
            return None
        return await self.relay_workers[machine_id].pull_weights(
            rollout_id, current_version
        )

    @property
    def current_version(self) -> int:
        return self._version

    def stats(self) -> dict[str, Any]:
        return {
            "version": self._version,
            "publish_count": self._publish_count,
            "master_relay": self.master_relay.stats(),
            "relay_workers": [rw.stats() for rw in self.relay_workers],
            "communication": self.comm.stats(),
        }
