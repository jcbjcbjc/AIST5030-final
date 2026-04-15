"""Common utilities for the async RL framework."""

from __future__ import annotations

import asyncio
import functools
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(slots=True)
class MovingAverage:
    """Exponential moving average tracker."""

    alpha: float = 0.1
    value: float = 0.0
    count: int = 0

    def update(self, sample: float) -> float:
        if self.count == 0:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value
        self.count += 1
        return self.value


class RateLimiter:
    """Token-bucket rate limiter for controlling throughput."""

    def __init__(self, rate: float, burst: int = 1) -> None:
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0

            wait_time = (1.0 - self._tokens) / self.rate
            self._tokens = 0.0
            await asyncio.sleep(wait_time)
            return wait_time


class WindowedCounter:
    """Counts events within a sliding time window."""

    def __init__(self, window_sec: float = 60.0) -> None:
        self.window_sec = window_sec
        self._events: deque[float] = deque()

    def record(self) -> None:
        now = time.time()
        self._events.append(now)
        self._evict()

    def _evict(self) -> None:
        cutoff = time.time() - self.window_sec
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    @property
    def count(self) -> int:
        self._evict()
        return len(self._events)

    @property
    def rate_per_sec(self) -> float:
        return self.count / self.window_sec


class VersionTracker:
    """Tracks and compares model versions across components."""

    def __init__(self) -> None:
        self._versions: dict[str, int] = {}
        self._update_times: dict[str, float] = {}

    def set_version(self, component: str, version: int) -> None:
        self._versions[component] = version
        self._update_times[component] = time.time()

    def get_version(self, component: str) -> int:
        return self._versions.get(component, -1)

    def max_version(self) -> int:
        return max(self._versions.values()) if self._versions else -1

    def min_version(self) -> int:
        return min(self._versions.values()) if self._versions else -1

    def version_spread(self) -> int:
        if len(self._versions) < 2:
            return 0
        return self.max_version() - self.min_version()

    def components_at_version(self, version: int) -> list[str]:
        return [c for c, v in self._versions.items() if v == version]

    def as_dict(self) -> dict[str, Any]:
        return {
            "versions": dict(self._versions),
            "max_version": self.max_version(),
            "min_version": self.min_version(),
            "spread": self.version_spread(),
        }


def compute_advantages(
    rewards: np.ndarray,
    method: str = "normalize",
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute advantages from rewards using the specified method."""
    if method == "normalize":
        mean = float(np.mean(rewards))
        std = float(np.std(rewards) + 1e-8)
        return (rewards - mean) / std
    elif method == "baseline":
        baseline = float(np.mean(rewards))
        return rewards - baseline
    elif method == "rank":
        # Rank-based advantage (used by some GRPO variants)
        order = np.argsort(np.argsort(rewards))
        n = len(rewards)
        return (2.0 * order / max(n - 1, 1) - 1.0).astype(np.float64)
    else:
        return rewards


def clip_importance_ratio(
    log_ratio: float,
    clip_range: float = 0.2,
) -> tuple[float, bool]:
    """Clip an importance-sampling ratio and report whether clipping occurred."""
    ratio = float(np.exp(np.clip(log_ratio, -5.0, 5.0)))
    low = 1.0 - clip_range
    high = 1.0 + clip_range
    clipped_ratio = float(np.clip(ratio, low, high))
    was_clipped = clipped_ratio != ratio
    return clipped_ratio, was_clipped


def cosine_schedule(
    step: int,
    total_steps: int,
    initial_lr: float,
    min_lr: float = 0.0,
    warmup_steps: int = 0,
) -> float:
    """Cosine annealing learning rate schedule with optional warmup."""
    if step < warmup_steps:
        return initial_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr + (initial_lr - min_lr) * cosine_decay


def linear_schedule(
    step: int,
    total_steps: int,
    initial_value: float,
    final_value: float,
) -> float:
    """Linear interpolation between initial and final values."""
    progress = min(step / max(total_steps, 1), 1.0)
    return initial_value + (final_value - initial_value) * progress


class AsyncBarrier:
    """Asyncio barrier for synchronizing N coroutines."""

    def __init__(self, parties: int) -> None:
        self.parties = parties
        self._count = 0
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            self._count += 1
            if self._count >= self.parties:
                self._event.set()

        await self._event.wait()

    def reset(self) -> None:
        self._count = 0
        self._event.clear()


class GradientAccumulator:
    """Accumulates gradients across mini-batches for large effective batch sizes."""

    def __init__(self, weight_shape: tuple[int, ...], bias_shape: tuple[int, ...]) -> None:
        self._grad_w = np.zeros(weight_shape, dtype=np.float64)
        self._grad_b = np.zeros(bias_shape, dtype=np.float64)
        self._step_count = 0
        self._token_count = 0

    def accumulate(
        self,
        grad_w: np.ndarray,
        grad_b: np.ndarray,
        num_tokens: int = 1,
    ) -> None:
        self._grad_w += grad_w
        self._grad_b += grad_b
        self._step_count += 1
        self._token_count += num_tokens

    def get_and_reset(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Return accumulated gradients and reset."""
        gw = self._grad_w.copy()
        gb = self._grad_b.copy()
        tc = self._token_count
        self._grad_w.fill(0.0)
        self._grad_b.fill(0.0)
        self._step_count = 0
        self._token_count = 0
        return gw, gb, tc

    @property
    def step_count(self) -> int:
        return self._step_count

    def grad_norm(self) -> float:
        return float(np.sqrt(np.sum(self._grad_w ** 2) + np.sum(self._grad_b ** 2)))
