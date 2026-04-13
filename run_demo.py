from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from async_rl_lite import ExperimentConfig, run_async_experiment, run_sync_experiment


RESULT_PATH = Path("latest_results.json")


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        seed=args.seed,
        num_updates=args.updates,
        train_batch_size=args.batch_size,
        num_workers=args.workers,
        max_concurrent_rollouts=args.max_concurrent_rollouts,
        max_staleness=args.max_staleness,
        max_decode_steps=args.decode_steps,
        trace_examples=args.trace_examples,
        verbose=not args.quiet,
    )


def build_comparison(results: dict[str, object]) -> dict[str, float]:
    async_result = results.get("async")
    sync_result = results.get("sync")
    if not isinstance(async_result, dict) or not isinstance(sync_result, dict):
        return {}

    async_wall = float(async_result["wall_time_sec"])
    sync_wall = float(sync_result["wall_time_sec"])
    async_tp = float(async_result["throughput_traj_per_sec"])
    sync_tp = float(sync_result["throughput_traj_per_sec"])
    async_reward = float(async_result["eval_reward"])
    sync_reward = float(sync_result["eval_reward"])
    async_acc = float(async_result["eval_accuracy"])
    sync_acc = float(sync_result["eval_accuracy"])

    return {
        "wall_time_speedup_sync_over_async": sync_wall / max(async_wall, 1e-12),
        "throughput_speedup_async_over_sync": async_tp / max(sync_tp, 1e-12),
        "eval_reward_delta_async_minus_sync": async_reward - sync_reward,
        "eval_accuracy_delta_async_minus_sync": async_acc - sync_acc,
    }


async def main_async(args: argparse.Namespace) -> dict[str, object]:
    config = build_config(args)
    results: dict[str, object] = {"config": asdict(config)}

    if args.mode in {"async", "both"}:
        results["async"] = await run_async_experiment(config)
    if args.mode in {"sync", "both"}:
        results["sync"] = await run_sync_experiment(config)
    if args.mode == "both":
        results["comparison"] = build_comparison(results)

    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the async RL lite demo.")
    parser.add_argument("--mode", choices=("async", "sync", "both"), default="both")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=48)
    parser.add_argument("--max-staleness", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--trace-examples", type=int, default=6)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results = asyncio.run(main_async(args))
    print(json.dumps(results, indent=2))
    print(f"\nSaved summary to {RESULT_PATH}")


if __name__ == "__main__":
    main()
