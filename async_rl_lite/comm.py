"""Distributed communication layer.

Simulates RDMA-based weight transfer and PCIe bus communication
between trainer, relay workers, and rollout machines. In production
this would be backed by NCCL / gRPC / RDMA verbs; here we model
the latency characteristics so the rest of the system can be
developed and tested against realistic timing.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import struct
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class TransferProtocol(enum.Enum):
    RDMA = "rdma"
    PCIE = "pcie"
    TCP = "tcp"
    SHARED_MEMORY = "shared_memory"


@dataclass(slots=True)
class TransferStats:
    bytes_sent: int = 0
    bytes_received: int = 0
    transfers_completed: int = 0
    transfers_failed: int = 0
    total_latency_sec: float = 0.0
    max_latency_sec: float = 0.0
    min_latency_sec: float = float("inf")

    @property
    def avg_latency_sec(self) -> float:
        if self.transfers_completed == 0:
            return 0.0
        return self.total_latency_sec / self.transfers_completed

    @property
    def throughput_gbps(self) -> float:
        if self.total_latency_sec == 0.0:
            return 0.0
        return (self.bytes_sent * 8) / (self.total_latency_sec * 1e9)

    def as_dict(self) -> dict[str, float]:
        return {
            "bytes_sent": float(self.bytes_sent),
            "bytes_received": float(self.bytes_received),
            "transfers_completed": float(self.transfers_completed),
            "transfers_failed": float(self.transfers_failed),
            "avg_latency_sec": self.avg_latency_sec,
            "max_latency_sec": self.max_latency_sec,
            "min_latency_sec": self.min_latency_sec if self.min_latency_sec != float("inf") else 0.0,
            "throughput_gbps": self.throughput_gbps,
        }


@dataclass(slots=True)
class WeightPayload:
    """Serialized model weight chunk for network transfer."""

    version: int
    shard_id: int
    total_shards: int
    data: np.ndarray
    checksum: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = self._compute_checksum()
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def _compute_checksum(self) -> str:
        raw = self.data.tobytes()
        return hashlib.md5(raw).hexdigest()

    def verify_integrity(self) -> bool:
        return self._compute_checksum() == self.checksum

    @property
    def size_bytes(self) -> int:
        return self.data.nbytes


@dataclass(slots=True)
class TransferRequest:
    payload: WeightPayload
    source_id: str
    destination_id: str
    protocol: TransferProtocol
    priority: int = 0
    created_at: float = field(default_factory=time.time)


class RDMASimulator:
    """Simulates RDMA (Remote Direct Memory Access) transfers.

    Models the bandwidth and latency characteristics of InfiniBand
    or RoCE networks used for inter-node weight synchronization.
    """

    BANDWIDTH_GBPS = 200.0  # 200 Gbps InfiniBand HDR
    BASE_LATENCY_US = 1.5   # ~1.5 microsecond RDMA latency
    MTU_BYTES = 4096

    def __init__(self, bandwidth_gbps: float | None = None, jitter_factor: float = 0.05) -> None:
        self.bandwidth_gbps = bandwidth_gbps or self.BANDWIDTH_GBPS
        self.jitter_factor = jitter_factor
        self._stats = TransferStats()
        self._lock = asyncio.Lock()
        self._rng = np.random.default_rng(42)

    def _compute_transfer_time(self, size_bytes: int) -> float:
        bandwidth_bytes_per_sec = (self.bandwidth_gbps * 1e9) / 8.0
        transfer_time = size_bytes / bandwidth_bytes_per_sec
        base_latency = self.BASE_LATENCY_US * 1e-6
        num_packets = max(1, size_bytes // self.MTU_BYTES)
        protocol_overhead = num_packets * 0.5e-6  # 0.5us per packet
        jitter = self._rng.normal(0, self.jitter_factor * transfer_time)
        return max(0.0, base_latency + transfer_time + protocol_overhead + jitter)

    async def transfer(self, request: TransferRequest) -> bool:
        size = request.payload.size_bytes
        delay = self._compute_transfer_time(size)
        start = time.perf_counter()

        # Simulate the network transfer
        await asyncio.sleep(delay * 0.001)  # Scale down for simulation

        if not request.payload.verify_integrity():
            async with self._lock:
                self._stats.transfers_failed += 1
            return False

        latency = time.perf_counter() - start
        async with self._lock:
            self._stats.bytes_sent += size
            self._stats.bytes_received += size
            self._stats.transfers_completed += 1
            self._stats.total_latency_sec += latency
            self._stats.max_latency_sec = max(self._stats.max_latency_sec, latency)
            self._stats.min_latency_sec = min(self._stats.min_latency_sec, latency)

        return True

    def stats(self) -> dict[str, float]:
        return self._stats.as_dict()


class PCIeTransfer:
    """Simulates PCIe bus transfers between CPU and GPU on the same node.

    Models the bandwidth of PCIe Gen4/Gen5 x16 links for weight
    transfer from relay worker (CPU memory) to rollout GPU.
    """

    PCIE_GEN4_BW_GBPS = 31.5   # PCIe Gen4 x16
    PCIE_GEN5_BW_GBPS = 63.0   # PCIe Gen5 x16

    def __init__(self, gen: int = 5, lanes: int = 16) -> None:
        base_bw = self.PCIE_GEN5_BW_GBPS if gen >= 5 else self.PCIE_GEN4_BW_GBPS
        self.bandwidth_gbps = base_bw * (lanes / 16)
        self.gen = gen
        self.lanes = lanes
        self._stats = TransferStats()
        self._lock = asyncio.Lock()

    async def host_to_device(self, data: np.ndarray, device_id: int = 0) -> float:
        size_bytes = data.nbytes
        bandwidth_bytes_per_sec = (self.bandwidth_gbps * 1e9) / 8.0
        transfer_time = size_bytes / bandwidth_bytes_per_sec
        # DMA setup overhead
        overhead = 2e-6  # ~2 microseconds
        total_time = transfer_time + overhead

        start = time.perf_counter()
        await asyncio.sleep(total_time * 0.001)  # Scale for simulation
        latency = time.perf_counter() - start

        async with self._lock:
            self._stats.bytes_sent += size_bytes
            self._stats.transfers_completed += 1
            self._stats.total_latency_sec += latency
            self._stats.max_latency_sec = max(self._stats.max_latency_sec, latency)
            self._stats.min_latency_sec = min(self._stats.min_latency_sec, latency)

        return latency

    async def device_to_host(self, data: np.ndarray, device_id: int = 0) -> float:
        return await self.host_to_device(data, device_id)

    def stats(self) -> dict[str, float]:
        return {
            "gen": float(self.gen),
            "lanes": float(self.lanes),
            "bandwidth_gbps": self.bandwidth_gbps,
            **self._stats.as_dict(),
        }


class BroadcastProtocol:
    """Implements a chain-based broadcast protocol for relay workers.

    The master relay receives weights from the trainer and broadcasts
    them to all other relay workers using a tree or chain topology
    to minimize the load on any single node.
    """

    def __init__(
        self,
        node_ids: list[str],
        rdma: RDMASimulator,
        topology: str = "chain",
    ) -> None:
        self.node_ids = list(node_ids)
        self.rdma = rdma
        self.topology = topology
        self._broadcast_count = 0
        self._total_broadcast_time = 0.0
        self._failed_broadcasts = 0

    def _build_chain_schedule(self, source: str) -> list[tuple[str, str]]:
        """Build a chain broadcast schedule: source -> n1 -> n2 -> ..."""
        others = [nid for nid in self.node_ids if nid != source]
        schedule = []
        prev = source
        for node in others:
            schedule.append((prev, node))
            prev = node
        return schedule

    def _build_tree_schedule(self, source: str) -> list[list[tuple[str, str]]]:
        """Build a binary tree broadcast schedule for parallelism."""
        others = [nid for nid in self.node_ids if nid != source]
        levels: list[list[tuple[str, str]]] = []
        current_senders = [source]
        remaining = list(others)

        while remaining:
            level = []
            next_senders = []
            for sender in current_senders:
                for _ in range(2):  # binary fan-out
                    if not remaining:
                        break
                    target = remaining.pop(0)
                    level.append((sender, target))
                    next_senders.append(target)
            levels.append(level)
            current_senders = next_senders

        return levels

    async def broadcast(self, payload: WeightPayload, source: str) -> dict[str, Any]:
        start = time.perf_counter()
        results: dict[str, bool] = {}

        if self.topology == "chain":
            schedule = self._build_chain_schedule(source)
            for src, dst in schedule:
                request = TransferRequest(
                    payload=payload,
                    source_id=src,
                    destination_id=dst,
                    protocol=TransferProtocol.RDMA,
                )
                success = await self.rdma.transfer(request)
                results[dst] = success
                if not success:
                    self._failed_broadcasts += 1
                    break

        elif self.topology == "tree":
            levels = self._build_tree_schedule(source)
            for level in levels:
                tasks = []
                for src, dst in level:
                    request = TransferRequest(
                        payload=payload,
                        source_id=src,
                        destination_id=dst,
                        protocol=TransferProtocol.RDMA,
                    )
                    tasks.append(self.rdma.transfer(request))
                level_results = await asyncio.gather(*tasks)
                for (src, dst), success in zip(level, level_results):
                    results[dst] = success
                    if not success:
                        self._failed_broadcasts += 1

        elapsed = time.perf_counter() - start
        self._broadcast_count += 1
        self._total_broadcast_time += elapsed

        return {
            "version": payload.version,
            "recipients": len(results),
            "successful": sum(1 for v in results.values() if v),
            "failed": sum(1 for v in results.values() if not v),
            "elapsed_sec": elapsed,
        }

    def stats(self) -> dict[str, float]:
        return {
            "topology": 0.0 if self.topology == "chain" else 1.0,
            "num_nodes": float(len(self.node_ids)),
            "broadcast_count": float(self._broadcast_count),
            "total_broadcast_time_sec": self._total_broadcast_time,
            "avg_broadcast_time_sec": (
                self._total_broadcast_time / max(self._broadcast_count, 1)
            ),
            "failed_broadcasts": float(self._failed_broadcasts),
        }


class CommunicationManager:
    """Top-level manager coordinating all communication channels."""

    def __init__(
        self,
        num_rollout_machines: int,
        num_trainer_machines: int,
        rdma_bandwidth_gbps: float = 200.0,
        pcie_gen: int = 5,
    ) -> None:
        self.num_rollout_machines = num_rollout_machines
        self.num_trainer_machines = num_trainer_machines

        self.rdma = RDMASimulator(bandwidth_gbps=rdma_bandwidth_gbps)
        self.pcie = PCIeTransfer(gen=pcie_gen)

        relay_ids = [f"relay_{i}" for i in range(num_rollout_machines)]
        self.broadcast = BroadcastProtocol(
            node_ids=["master_relay"] + relay_ids,
            rdma=self.rdma,
            topology="tree" if num_rollout_machines > 8 else "chain",
        )

        self._total_weight_syncs = 0
        self._total_sync_time = 0.0

    async def sync_weights(self, weights: np.ndarray, version: int) -> dict[str, Any]:
        """Full weight synchronization: trainer -> master relay -> all relays."""
        self._total_weight_syncs += 1
        start = time.perf_counter()

        payload = WeightPayload(
            version=version,
            shard_id=0,
            total_shards=1,
            data=weights,
        )

        # Step 1: Trainer pushes to master relay via RDMA
        trainer_request = TransferRequest(
            payload=payload,
            source_id="trainer_0",
            destination_id="master_relay",
            protocol=TransferProtocol.RDMA,
        )
        await self.rdma.transfer(trainer_request)

        # Step 2: Master relay broadcasts to all relay workers
        broadcast_result = await self.broadcast.broadcast(payload, "master_relay")

        elapsed = time.perf_counter() - start
        self._total_sync_time += elapsed

        return {
            "version": version,
            "total_time_sec": elapsed,
            "broadcast_result": broadcast_result,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "total_weight_syncs": self._total_weight_syncs,
            "avg_sync_time_sec": self._total_sync_time / max(self._total_weight_syncs, 1),
            "rdma": self.rdma.stats(),
            "pcie": self.pcie.stats(),
            "broadcast": self.broadcast.stats(),
        }
