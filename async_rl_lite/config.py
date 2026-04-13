from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExperimentConfig:
    seed: int = 7
    num_updates: int = 24
    train_batch_size: int = 32
    max_concurrent_rollouts: int = 48
    num_workers: int = 6
    max_decode_steps: int = 4
    max_staleness: int = 2
    learning_rate: float = 0.18
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    buffer_capacity: int = 256
    history_limit: int = 64
    base_rollout_delay: float = 0.006
    reward_delay: float = 0.002
    refresh_overhead: float = 0.0015
    controller_wait_timeout: float = 0.010
    controller_idle_sleep: float = 0.001
    prioritize_oldest: bool = True
    eval_size: int = 100
    trace_examples: int = 6
    verbose: bool = True
