# Async RL Lite

This directory contains a compact async RL prototype inspired by the system and algorithm ideas discussed in the AReaL paper, our report, and the presentation deck.

What it keeps from the paper:

- asynchronous rollout generation and training
- interruptible rollout workers that refresh weights mid-trajectory
- a replay buffer between generation and training
- staleness-aware rollout throttling
- oldest-first replay consumption to limit stale data accumulation
- a simplified decoupled PPO-style update

What it simplifies:

- no real LLMs or distributed GPUs
- a toy arithmetic reasoning task instead of math/code benchmarks
- numpy-only policy and hand-written gradient updates
- a single-process asyncio runtime

Files:

- `run_demo.py`: runs both async and sync training and writes a JSON summary
- `async_rl_lite/config.py`: experiment configuration
- `async_rl_lite/toy_env.py`: toy reasoning environment, prompt difficulty, and reward shaping
- `async_rl_lite/policy.py`: lightweight autoregressive policy with reasoning-budget-aware decoding
- `async_rl_lite/system.py`: rollout workers, controller, replay buffer, trainer, metrics, traces, and runners

Core behaviors in the expanded version:

- prompts now expose `difficulty`, `required_reasoning_steps`, and `estimated_cost`
- hard prompts are forced to emit a few reasoning tokens before answers are allowed
- async rollouts can switch parameter versions mid-trajectory and record token-level version traces
- the replay buffer can prioritize older trajectories when building the next batch
- the trainer records per-update history instead of only final metrics
- evaluation now reports easy/hard splits and sample decoded traces

Run:

```bash
python3 run_demo.py
```

The script writes the latest summary to:

```text
latest_results.json
```

Useful options:

```bash
python3 run_demo.py --mode both --updates 24 --batch-size 32 --decode-steps 4 --trace-examples 6
```

Result JSON structure:

- `config`: resolved experiment config
- `async` / `sync`: end-to-end experiment summaries
- `comparison`: async-vs-sync deltas and speedups when `--mode both`

Each experiment summary includes:

- final scalar metrics such as throughput, train reward, eval accuracy, and interrupted rollout count
- `train_history`: one record per update
- `sample_trajectories`: traced training rollouts with token versions
- `parameter_service`, `reward_service`, `workers`
- `controller` and `replay_buffer` for the async run
- `sample_eval_traces` for quick qualitative inspection

Recommended next extensions:

- replace the linear numpy policy with a tiny PyTorch network
- split generation and training into separate processes
- add ablations over `max_staleness`, `num_workers`, and `buffer_capacity`
- replace the arithmetic task with a string-based reasoning environment
