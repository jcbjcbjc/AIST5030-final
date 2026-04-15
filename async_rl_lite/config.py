from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RepackConfig:
    """Configuration for the dynamic repack mechanism."""

    enabled: bool = True
    idle_fraction_threshold: float = 0.25
    cooldown_sec: float = 5.0
    min_idle_rollouts: int = 2
    max_kvcache_utilization: float = 0.90
    max_requests_per_rollout: int = 8
    kvcache_idleness_threshold: float = 0.3
    kvcache_trend_threshold: float = -0.005
    kvcache_window_size: int = 10
    check_interval_sec: float = 2.0


@dataclass(slots=True)
class FaultToleranceConfig:
    """Configuration for fault tolerance and recovery."""

    heartbeat_interval_sec: float = 5.0
    heartbeat_timeout_sec: float = 15.0
    max_consecutive_misses: int = 3
    max_recovery_attempts: int = 3
    recovery_timeout_sec: float = 60.0
    checkpoint_interval_updates: int = 10
    max_checkpoints: int = 5
    checkpoint_dir: str = "/tmp/rl_checkpoints"


@dataclass(slots=True)
class DistributedConfig:
    """Configuration for the distributed communication layer."""

    num_rollout_machines: int = 4
    num_trainer_machines: int = 1
    rollouts_per_machine: int = 2
    gpus_per_machine: int = 4
    rdma_bandwidth_gbps: float = 200.0
    pcie_gen: int = 5
    pcie_lanes: int = 16
    broadcast_topology: str = "chain"     # "chain" or "tree"
    relay_cache_size: int = 4
    broadcast_timeout_sec: float = 30.0


@dataclass(slots=True)
class ExperimentConfig:
    """Full experiment configuration."""

    # --- Core training ---
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

    # --- Timing simulation ---
    base_rollout_delay: float = 0.006
    reward_delay: float = 0.002
    refresh_overhead: float = 0.0015
    controller_wait_timeout: float = 0.010
    controller_idle_sleep: float = 0.001

    # --- Replay buffer ---
    prioritize_oldest: bool = True
    sampling_strategy: str = "oldest_first"   # fifo, oldest_first, importance_weighted, stratified
    importance_temperature: float = 1.0

    # --- Evaluation ---
    eval_size: int = 100
    trace_examples: int = 6
    verbose: bool = True

    # --- Learning rate schedule ---
    lr_schedule: str = "constant"   # constant, cosine, linear
    lr_warmup_steps: int = 0
    lr_min: float = 0.01

    # --- Advantage computation ---
    advantage_method: str = "normalize"   # normalize, baseline, rank

    # --- Gradient accumulation ---
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # --- Distributed ---
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # --- Repack ---
    repack: RepackConfig = field(default_factory=RepackConfig)

    # --- Fault tolerance ---
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)
