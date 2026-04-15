# Scalable Asynchronous RL Framework

This repository contains the implementation for our scalable asynchronous RL framework for language reasoning post-training, as described in the accompanying AIST5030 Final Report (Group 11).

## Architecture

The framework implements a fully decoupled architecture with four core modules:

1. **Rollout Module** — manages trajectory generation across GPU machines
2. **Data Module** — experience buffer, partial response pool, prompt dispatch
3. **Trainer Workers** — decoupled PPO-style model training
4. **Relay Workers** — asynchronous weight synchronization via RDMA

Key features:

- **Trajectory-level asynchrony**: each trajectory is generated and consumed independently at its own pace, eliminating global synchronization barriers
- **Relay worker hierarchy**: a tier of CPU-based relay workers provide fine-grained, anytime weight synchronization without stalling training
- **Dynamic repack mechanism**: consolidates long-tail trajectories onto fewer rollouts to maximize GPU utilization
- **Fault tolerance**: heartbeat-based failover with automatic broadcast chain rebuild
- **KVCache monitoring**: tracks GPU KVCache utilization for repack decisions

## Prototype

The compact prototype preserves the core architectural ideas while running on a single machine:

- A toy arithmetic reasoning task instead of real LLM benchmarks
- numpy-only autoregressive policy
- Single-process asyncio runtime simulating distributed execution
- Asynchronous rollout generation with mid-trajectory weight refreshes
- Staleness-aware scheduling and replay buffer
- Decoupled PPO-style training objective

## Files

### Core system
- `run_demo.py` — entry point; runs async and sync experiments
- `async_rl_lite/config.py` — experiment, distributed, repack, and fault-tolerance configuration
- `async_rl_lite/toy_env.py` — toy arithmetic environment, prompt difficulty, reward shaping
- `async_rl_lite/policy.py` — lightweight autoregressive policy
- `async_rl_lite/system.py` — rollout workers, controller, replay buffer, trainer, and runners

### Relay workers
- `async_rl_lite/relay.py` — relay worker hierarchy for asynchronous weight synchronization
- `async_rl_lite/comm.py` — RDMA simulation, PCIe transfer, broadcast protocol

### Repack mechanism
- `async_rl_lite/repack.py` — Best-Fit packing algorithm, repack trigger, repack manager
- `async_rl_lite/kvcache.py` — KVCache utilization monitoring and idleness detection

### Data module
- `async_rl_lite/data_module.py` — experience buffer writer/sampler, partial response pool, prompt pool

### Rollout management
- `async_rl_lite/rollout_manager.py` — orchestrates rollout dispatch, monitoring, and failure recovery

### Scheduling
- `async_rl_lite/scheduler.py` — staleness-bounded scheduling, adaptive concurrency, machine-aware dispatch

### Fault tolerance
- `async_rl_lite/fault_tolerance.py` — heartbeat monitor, broadcast chain rebuilder, recovery coordinator, checkpointing

### Monitoring
- `async_rl_lite/metrics.py` — throughput tracking, staleness monitoring, GPU utilization tracking

### Utilities
- `async_rl_lite/utils.py` — gradient accumulation, learning rate schedules, rate limiting, version tracking

## Run

```bash
python3 run_demo.py
```

The script writes the latest summary to `latest_results.json`.

Useful options:

```bash
python3 run_demo.py --mode both --updates 24 --batch-size 32 --decode-steps 4 --trace-examples 6
python3 run_demo.py --mode async --workers 8 --max-concurrent-rollouts 64 --max-staleness 4
```

Result JSON structure:

- `config`: resolved experiment config
- `async` / `sync`: end-to-end experiment summaries
- `comparison`: async-vs-sync deltas and speedups when `--mode both`

Each experiment summary includes:

- final scalar metrics such as throughput, train reward, eval accuracy, interrupted rollout count
- `train_history`: one record per update
- `sample_trajectories`: traced training rollouts with token versions
- `parameter_service`, `reward_service`, `workers`
- `controller` and `replay_buffer` for the async run
- `sample_eval_traces` for quick qualitative inspection
