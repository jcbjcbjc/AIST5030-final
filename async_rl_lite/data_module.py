"""Data module: experience buffer, partial response pool, and prompt management.

Implements the Data Module from the system architecture, handling
the lifecycle of trajectories from generation through to training
consumption.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class PartialResponse:
    """An in-progress trajectory that has not yet completed generation."""

    trajectory_id: str
    prompt_id: int
    rollout_id: int
    tokens_so_far: int
    max_tokens: int
    model_version: int
    started_at: float
    last_updated: float = 0.0
    estimated_completion_pct: float = 0.0

    @property
    def is_stale(self) -> bool:
        return time.time() - self.last_updated > 30.0

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.started_at


@dataclass(slots=True)
class PromptEntry:
    """A prompt waiting in the prompt pool to be dispatched."""

    prompt_id: int
    text: str
    difficulty: int
    priority: float = 0.0
    created_at: float = field(default_factory=time.time)
    dispatched: bool = False
    dispatched_to: int = -1
    dispatch_time: float = 0.0


class PartialResponsePool:
    """Tracks in-progress trajectories across all rollout machines.

    The pool allows the system to monitor generation progress,
    estimate completion times, and identify stale trajectories
    that may need recovery or repacking.
    """

    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size
        self._pool: dict[str, PartialResponse] = {}
        self._lock = asyncio.Lock()
        self._total_registered = 0
        self._total_completed = 0
        self._total_expired = 0

    async def register(self, response: PartialResponse) -> bool:
        async with self._lock:
            if len(self._pool) >= self.max_size:
                return False
            self._pool[response.trajectory_id] = response
            self._total_registered += 1
            return True

    async def update(
        self,
        trajectory_id: str,
        tokens_so_far: int,
        model_version: int | None = None,
    ) -> None:
        async with self._lock:
            if trajectory_id in self._pool:
                entry = self._pool[trajectory_id]
                entry.tokens_so_far = tokens_so_far
                entry.last_updated = time.time()
                entry.estimated_completion_pct = tokens_so_far / max(entry.max_tokens, 1)
                if model_version is not None:
                    entry.model_version = model_version

    async def complete(self, trajectory_id: str) -> PartialResponse | None:
        async with self._lock:
            entry = self._pool.pop(trajectory_id, None)
            if entry is not None:
                self._total_completed += 1
            return entry

    async def get_stale(self, timeout_sec: float = 30.0) -> list[PartialResponse]:
        async with self._lock:
            now = time.time()
            stale = [
                pr
                for pr in self._pool.values()
                if now - pr.last_updated > timeout_sec
            ]
            return stale

    async def expire_stale(self, timeout_sec: float = 60.0) -> int:
        async with self._lock:
            now = time.time()
            expired_ids = [
                tid
                for tid, pr in self._pool.items()
                if now - pr.last_updated > timeout_sec
            ]
            for tid in expired_ids:
                del self._pool[tid]
            self._total_expired += len(expired_ids)
            return len(expired_ids)

    def get_by_rollout(self, rollout_id: int) -> list[PartialResponse]:
        return [
            pr for pr in self._pool.values() if pr.rollout_id == rollout_id
        ]

    def get_by_version(self, model_version: int) -> list[PartialResponse]:
        return [
            pr for pr in self._pool.values() if pr.model_version == model_version
        ]

    @property
    def size(self) -> int:
        return len(self._pool)

    def stats(self) -> dict[str, float]:
        now = time.time()
        ages = [now - pr.started_at for pr in self._pool.values()]
        return {
            "size": float(len(self._pool)),
            "max_size": float(self.max_size),
            "total_registered": float(self._total_registered),
            "total_completed": float(self._total_completed),
            "total_expired": float(self._total_expired),
            "avg_age_sec": float(np.mean(ages)) if ages else 0.0,
            "max_age_sec": float(max(ages)) if ages else 0.0,
            "completion_rate": (
                self._total_completed / max(self._total_registered, 1)
            ),
        }


class PromptPool:
    """Manages prompt dispatch to rollout machines.

    Supports priority-based scheduling where harder prompts can
    be dispatched to faster machines or given scheduling priority
    based on curriculum or difficulty-aware strategies.
    """

    def __init__(
        self,
        max_size: int = 4096,
        priority_by_difficulty: bool = True,
    ) -> None:
        self.max_size = max_size
        self.priority_by_difficulty = priority_by_difficulty
        self._queue: list[PromptEntry] = []  # min-heap by priority
        self._dispatched: dict[int, PromptEntry] = {}
        self._lock = asyncio.Lock()
        self._total_added = 0
        self._total_dispatched = 0

    async def add_prompt(self, entry: PromptEntry) -> bool:
        async with self._lock:
            if len(self._queue) >= self.max_size:
                return False
            if self.priority_by_difficulty:
                entry.priority = -entry.difficulty  # Higher difficulty = higher priority
            heapq.heappush(self._queue, (entry.priority, entry.prompt_id, entry))
            self._total_added += 1
            return True

    async def add_prompts_batch(self, entries: list[PromptEntry]) -> int:
        added = 0
        for entry in entries:
            if await self.add_prompt(entry):
                added += 1
        return added

    async def dispatch(self, rollout_id: int) -> PromptEntry | None:
        async with self._lock:
            if not self._queue:
                return None
            _, _, entry = heapq.heappop(self._queue)
            entry.dispatched = True
            entry.dispatched_to = rollout_id
            entry.dispatch_time = time.time()
            self._dispatched[entry.prompt_id] = entry
            self._total_dispatched += 1
            return entry

    async def dispatch_batch(self, rollout_id: int, count: int) -> list[PromptEntry]:
        entries = []
        for _ in range(count):
            entry = await self.dispatch(rollout_id)
            if entry is None:
                break
            entries.append(entry)
        return entries

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def dispatched_count(self) -> int:
        return len(self._dispatched)

    def stats(self) -> dict[str, float]:
        return {
            "pending": float(self.pending_count),
            "dispatched": float(self.dispatched_count),
            "total_added": float(self._total_added),
            "total_dispatched": float(self._total_dispatched),
            "max_size": float(self.max_size),
        }


class ExperienceWriter:
    """Writes completed trajectories into the experience buffer.

    Handles serialization, batching, and priority tagging before
    inserting into the shared experience buffer.
    """

    def __init__(
        self,
        buffer_put_fn: Any,
        batch_write_size: int = 16,
        write_interval_sec: float = 0.1,
    ) -> None:
        self._buffer_put_fn = buffer_put_fn
        self.batch_write_size = batch_write_size
        self.write_interval_sec = write_interval_sec

        self._pending: deque[Any] = deque()
        self._total_written = 0
        self._total_dropped = 0
        self._lock = asyncio.Lock()

    async def submit(self, trajectory: Any) -> None:
        async with self._lock:
            self._pending.append(trajectory)

    async def flush(self) -> int:
        """Write pending trajectories to the buffer."""
        async with self._lock:
            written = 0
            while self._pending:
                traj = self._pending.popleft()
                success = await self._buffer_put_fn(traj, drop_if_full=True)
                if success:
                    written += 1
                else:
                    self._total_dropped += 1
                self._total_written += written
            return written

    async def run_writer_loop(self, stop_event: asyncio.Event) -> None:
        """Background loop that periodically flushes pending trajectories."""
        while not stop_event.is_set():
            await self.flush()
            await asyncio.sleep(self.write_interval_sec)
        # Final flush
        await self.flush()

    def stats(self) -> dict[str, float]:
        return {
            "pending": float(len(self._pending)),
            "total_written": float(self._total_written),
            "total_dropped": float(self._total_dropped),
        }


class ExperienceSampler:
    """Samples training batches from the experience buffer.

    Supports multiple sampling strategies: FIFO, oldest-first
    (prioritize stale data), importance-weighted, and
    difficulty-stratified.
    """

    STRATEGIES = ("fifo", "oldest_first", "importance_weighted", "stratified")

    def __init__(
        self,
        strategy: str = "oldest_first",
        importance_temperature: float = 1.0,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.importance_temperature = importance_temperature
        self._total_samples = 0
        self._total_batches = 0

    def select_batch(
        self,
        items: list[Any],
        batch_size: int,
        current_version: int = 0,
    ) -> list[Any]:
        """Select a batch from available items using the configured strategy."""
        if len(items) <= batch_size:
            selected = list(items)
        elif self.strategy == "fifo":
            selected = items[:batch_size]
        elif self.strategy == "oldest_first":
            sorted_items = sorted(
                items,
                key=lambda t: getattr(t, "max_behavior_version", 0),
            )
            selected = sorted_items[:batch_size]
        elif self.strategy == "importance_weighted":
            staleness = np.array([
                max(1, current_version - getattr(t, "max_behavior_version", current_version))
                for t in items
            ], dtype=np.float64)
            weights = np.exp(staleness / self.importance_temperature)
            weights /= weights.sum()
            indices = np.random.choice(len(items), size=batch_size, replace=False, p=weights)
            selected = [items[i] for i in indices]
        elif self.strategy == "stratified":
            # Split by difficulty, sample proportionally
            easy = [t for t in items if getattr(getattr(t, "prompt", None), "difficulty", 0) == 0]
            hard = [t for t in items if getattr(getattr(t, "prompt", None), "difficulty", 0) > 0]
            n_hard = min(len(hard), batch_size // 2)
            n_easy = min(len(easy), batch_size - n_hard)
            selected = easy[:n_easy] + hard[:n_hard]
            # Fill remaining if needed
            remaining = batch_size - len(selected)
            if remaining > 0:
                rest = [t for t in items if t not in selected]
                selected.extend(rest[:remaining])
        else:
            selected = items[:batch_size]

        self._total_samples += len(selected)
        self._total_batches += 1
        return selected

    def stats(self) -> dict[str, float]:
        return {
            "strategy": float(self.STRATEGIES.index(self.strategy)),
            "total_samples": float(self._total_samples),
            "total_batches": float(self._total_batches),
            "avg_batch_size": (
                self._total_samples / max(self._total_batches, 1)
            ),
        }
