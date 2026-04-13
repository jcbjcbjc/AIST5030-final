from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .policy import PolicySnapshot, ToyAutoregressivePolicy
from .toy_env import ANSWER_TOKENS, Prompt, REASONING_TOKENS, ToyReasoningEnv, VOCAB_SIZE, token_name


@dataclass(slots=True)
class RewardResult:
    reward: float
    latency_sec: float


@dataclass(slots=True)
class TokenRecord:
    step_index: int
    action: int
    behavior_log_prob: float
    behavior_version: int
    features: np.ndarray
    allow_answers: bool
    answer_only: bool


@dataclass(slots=True)
class Trajectory:
    prompt: Prompt
    tokens: list[TokenRecord]
    predicted_answer: int | None
    reward: float
    reasoning_steps: int
    interrupted: bool
    wall_time: float
    reward_latency_sec: float
    worker_id: int
    started_version: int
    final_version: int
    enqueued_at: float = 0.0
    buffer_wait_sec: float = 0.0
    sampled_version: int = -1

    @property
    def accuracy(self) -> float:
        return 1.0 if self.predicted_answer == self.prompt.target else 0.0

    @property
    def decode_len(self) -> int:
        return len(self.tokens)

    @property
    def mean_behavior_version(self) -> float:
        return float(sum(token.behavior_version for token in self.tokens) / len(self.tokens))

    @property
    def min_behavior_version(self) -> int:
        return min(token.behavior_version for token in self.tokens)

    @property
    def max_behavior_version(self) -> int:
        return max(token.behavior_version for token in self.tokens)

    @property
    def version_span(self) -> int:
        return self.max_behavior_version - self.min_behavior_version

    @property
    def sampled_staleness(self) -> float:
        if self.sampled_version < 0:
            return 0.0
        return float(max(0, self.sampled_version - self.max_behavior_version))

    def to_trace(self) -> dict[str, object]:
        return {
            "prompt_id": self.prompt.prompt_id,
            "prompt": self.prompt.text,
            "target": self.prompt.target,
            "difficulty": self.prompt.difficulty,
            "required_reasoning_steps": self.prompt.required_reasoning_steps,
            "predicted_answer": self.predicted_answer,
            "reward": round(self.reward, 4),
            "accuracy": self.accuracy,
            "interrupted": self.interrupted,
            "worker_id": self.worker_id,
            "started_version": self.started_version,
            "final_version": self.final_version,
            "version_span": self.version_span,
            "buffer_wait_sec": round(self.buffer_wait_sec, 5),
            "wall_time_sec": round(self.wall_time, 5),
            "reward_latency_sec": round(self.reward_latency_sec, 5),
            "tokens": [token_name(token.action) for token in self.tokens],
            "token_versions": [token.behavior_version for token in self.tokens],
        }


@dataclass(slots=True)
class UpdateStats:
    reward_mean: float
    batch_accuracy: float
    avg_behavior_version: float
    clip_fraction: float
    avg_importance: float
    avg_reasoning_steps: float
    avg_decode_len: float
    avg_traj_time_sec: float
    avg_reward_latency_sec: float
    avg_buffer_wait_sec: float
    avg_staleness: float
    avg_version_span: float
    interruption_rate: float
    hard_prompt_fraction: float

    def as_dict(self, update_index: int, policy_version: int, buffer_size: int) -> dict[str, float]:
        return {
            "update": float(update_index),
            "policy_version": float(policy_version),
            "buffer_size": float(buffer_size),
            "reward_mean": self.reward_mean,
            "batch_accuracy": self.batch_accuracy,
            "avg_behavior_version": self.avg_behavior_version,
            "clip_fraction": self.clip_fraction,
            "avg_importance": self.avg_importance,
            "avg_reasoning_steps": self.avg_reasoning_steps,
            "avg_decode_len": self.avg_decode_len,
            "avg_traj_time_sec": self.avg_traj_time_sec,
            "avg_reward_latency_sec": self.avg_reward_latency_sec,
            "avg_buffer_wait_sec": self.avg_buffer_wait_sec,
            "avg_staleness": self.avg_staleness,
            "avg_version_span": self.avg_version_span,
            "interruption_rate": self.interruption_rate,
            "hard_prompt_fraction": self.hard_prompt_fraction,
        }


class ParameterService:
    def __init__(self, initial_snapshot: PolicySnapshot, history_limit: int) -> None:
        self.version = 0
        self.history_limit = history_limit
        self.snapshot = initial_snapshot.copy()
        self.history: dict[int, PolicySnapshot] = {0: initial_snapshot.copy()}
        self.publish_count = 0
        self.pull_requests = 0
        self.pull_hits = 0
        self.max_refresh_gap = 0
        self._lock = asyncio.Lock()

    @property
    def current_version(self) -> int:
        return self.version

    async def get_latest(self) -> tuple[int, PolicySnapshot]:
        async with self._lock:
            return self.version, self.snapshot.copy()

    async def pull_if_new(self, local_version: int) -> tuple[int, PolicySnapshot] | None:
        async with self._lock:
            self.pull_requests += 1
            if self.version > local_version:
                self.pull_hits += 1
                self.max_refresh_gap = max(self.max_refresh_gap, self.version - local_version)
                return self.version, self.snapshot.copy()
            return None

    async def publish(self, snapshot: PolicySnapshot) -> int:
        async with self._lock:
            self.version += 1
            self.publish_count += 1
            self.snapshot = snapshot.copy()
            self.history[self.version] = snapshot.copy()
            obsolete = sorted(self.history)[:-self.history_limit]
            for version in obsolete:
                self.history.pop(version, None)
            return self.version

    def stats(self) -> dict[str, float]:
        return {
            "current_version": float(self.version),
            "publish_count": float(self.publish_count),
            "pull_requests": float(self.pull_requests),
            "pull_hits": float(self.pull_hits),
            "pull_hit_rate": self.pull_hits / max(self.pull_requests, 1),
            "max_refresh_gap": float(self.max_refresh_gap),
            "history_size": float(len(self.history)),
        }


class RewardService:
    def __init__(self, env: ToyReasoningEnv, base_delay: float) -> None:
        self.env = env
        self.base_delay = base_delay
        self.request_count = 0
        self.total_latency_sec = 0.0
        self.max_latency_sec = 0.0
        self.total_reward = 0.0

    async def score(self, prompt: Prompt, predicted_answer: int | None, reasoning_steps: int) -> RewardResult:
        delay = self.base_delay * (1.0 + 0.30 * prompt.estimated_cost + 0.15 * reasoning_steps)
        start = time.perf_counter()
        await asyncio.sleep(delay)
        reward = self.env.score(prompt, predicted_answer, reasoning_steps)
        latency = time.perf_counter() - start
        self.request_count += 1
        self.total_latency_sec += latency
        self.max_latency_sec = max(self.max_latency_sec, latency)
        self.total_reward += reward
        return RewardResult(reward=reward, latency_sec=latency)

    def stats(self) -> dict[str, float]:
        return {
            "request_count": float(self.request_count),
            "avg_latency_sec": self.total_latency_sec / max(self.request_count, 1),
            "max_latency_sec": self.max_latency_sec,
            "avg_reward": self.total_reward / max(self.request_count, 1),
        }


class ReplayBuffer:
    def __init__(self, capacity: int, prioritize_oldest: bool) -> None:
        self.capacity = capacity
        self.prioritize_oldest = prioritize_oldest
        self._items: list[Trajectory] = []
        self._condition = asyncio.Condition()
        self.total_puts = 0
        self.total_gets = 0
        self.dropped_trajectories = 0
        self.max_size_seen = 0
        self.total_wait_sec = 0.0
        self.total_sample_staleness = 0.0

    async def put(self, trajectory: Trajectory, drop_if_full: bool = False) -> bool:
        async with self._condition:
            if drop_if_full and len(self._items) >= self.capacity:
                self.dropped_trajectories += 1
                return False

            while len(self._items) >= self.capacity:
                await self._condition.wait()

            trajectory.enqueued_at = time.perf_counter()
            self._items.append(trajectory)
            self.total_puts += 1
            self.max_size_seen = max(self.max_size_seen, len(self._items))
            self._condition.notify_all()
            return True

    async def get_many(self, batch_size: int, current_version: int) -> list[Trajectory]:
        async with self._condition:
            while len(self._items) < batch_size:
                await self._condition.wait()

            if self.prioritize_oldest:
                self._items.sort(
                    key=lambda trajectory: (
                        trajectory.max_behavior_version,
                        trajectory.prompt.required_reasoning_steps,
                        trajectory.prompt.prompt_id,
                    )
                )

            batch = self._items[:batch_size]
            del self._items[:batch_size]

            now = time.perf_counter()
            for trajectory in batch:
                trajectory.buffer_wait_sec = now - trajectory.enqueued_at
                trajectory.sampled_version = current_version
                self.total_wait_sec += trajectory.buffer_wait_sec
                self.total_sample_staleness += trajectory.sampled_staleness

            self.total_gets += len(batch)
            self._condition.notify_all()
            return batch

    def size(self) -> int:
        return len(self._items)

    def stats(self) -> dict[str, float]:
        return {
            "capacity": float(self.capacity),
            "prioritize_oldest": 1.0 if self.prioritize_oldest else 0.0,
            "size": float(len(self._items)),
            "total_puts": float(self.total_puts),
            "total_gets": float(self.total_gets),
            "dropped_trajectories": float(self.dropped_trajectories),
            "max_size_seen": float(self.max_size_seen),
            "avg_wait_sec": self.total_wait_sec / max(self.total_gets, 1),
            "avg_sample_staleness": self.total_sample_staleness / max(self.total_gets, 1),
        }


class InterruptibleRolloutWorker:
    def __init__(
        self,
        worker_id: int,
        policy: ToyAutoregressivePolicy,
        parameter_service: ParameterService,
        reward_service: RewardService,
        config: ExperimentConfig,
        seed: int,
    ) -> None:
        self.worker_id = worker_id
        self.policy = policy
        self.parameter_service = parameter_service
        self.reward_service = reward_service
        self.config = config
        self.seed_stream = np.random.default_rng(seed)
        self.rollouts_started = 0
        self.rollouts_completed = 0
        self.refresh_count = 0
        self.total_rollout_time = 0.0
        self.total_reasoning_steps = 0

    def _make_rng(self) -> np.random.Generator:
        return np.random.default_rng(int(self.seed_stream.integers(0, 2**31 - 1)))

    async def rollout(
        self,
        prompt: Prompt,
        frozen_snapshot: PolicySnapshot | None = None,
        frozen_version: int | None = None,
    ) -> Trajectory:
        self.rollouts_started += 1
        rng = self._make_rng()

        if frozen_snapshot is None:
            local_version, local_snapshot = await self.parameter_service.get_latest()
            frozen = False
        else:
            local_version = 0 if frozen_version is None else frozen_version
            local_snapshot = frozen_snapshot.copy()
            frozen = True

        started_version = local_version
        tokens: list[TokenRecord] = []
        prev_token: int | None = None
        predicted_answer: int | None = None
        reasoning_steps = 0
        interrupted = False
        start = time.perf_counter()

        for step in range(self.config.max_decode_steps):
            answer_only = step == self.config.max_decode_steps - 1
            allow_answers = answer_only or step >= prompt.required_reasoning_steps
            action, log_prob, features = self.policy.sample_action(
                prompt=prompt,
                step_index=step,
                prev_token=prev_token,
                rng=rng,
                snapshot=local_snapshot,
                allow_answers=allow_answers,
                answer_only=answer_only,
            )
            tokens.append(
                TokenRecord(
                    step_index=step,
                    action=action,
                    behavior_log_prob=log_prob,
                    behavior_version=local_version,
                    features=features,
                    allow_answers=allow_answers,
                    answer_only=answer_only,
                )
            )

            if action in REASONING_TOKENS:
                reasoning_steps += 1
            else:
                predicted_answer = action

            await asyncio.sleep(
                self.config.base_rollout_delay
                * (prompt.estimated_cost + 0.20 * step + 0.25 * float(rng.random()))
            )

            if not frozen:
                refreshed = await self.parameter_service.pull_if_new(local_version)
                if refreshed is not None:
                    local_version, local_snapshot = refreshed
                    interrupted = True
                    self.refresh_count += 1
                    await asyncio.sleep(self.config.refresh_overhead * (1.0 + step))

            if action in ANSWER_TOKENS:
                break
            prev_token = action

        reward_result = await self.reward_service.score(prompt, predicted_answer, reasoning_steps)
        wall_time = time.perf_counter() - start
        self.rollouts_completed += 1
        self.total_rollout_time += wall_time
        self.total_reasoning_steps += reasoning_steps

        return Trajectory(
            prompt=prompt,
            tokens=tokens,
            predicted_answer=predicted_answer,
            reward=reward_result.reward,
            reasoning_steps=reasoning_steps,
            interrupted=interrupted,
            wall_time=wall_time,
            reward_latency_sec=reward_result.latency_sec,
            worker_id=self.worker_id,
            started_version=started_version,
            final_version=local_version,
        )

    def stats(self) -> dict[str, float]:
        return {
            "worker_id": float(self.worker_id),
            "rollouts_started": float(self.rollouts_started),
            "rollouts_completed": float(self.rollouts_completed),
            "refresh_count": float(self.refresh_count),
            "avg_rollout_time_sec": self.total_rollout_time / max(self.rollouts_completed, 1),
            "avg_reasoning_steps": self.total_reasoning_steps / max(self.rollouts_completed, 1),
        }


class RolloutController:
    def __init__(
        self,
        env: ToyReasoningEnv,
        workers: list[InterruptibleRolloutWorker],
        replay_buffer: ReplayBuffer,
        parameter_service: ParameterService,
        config: ExperimentConfig,
    ) -> None:
        self.env = env
        self.workers = workers
        self.replay_buffer = replay_buffer
        self.parameter_service = parameter_service
        self.config = config
        self.submitted_rollouts = 0
        self.completed_rollouts = 0
        self.interrupted_rollouts = 0
        self.staleness_pauses = 0
        self.idle_loops = 0
        self.max_in_flight = 0
        self.dropped_after_stop = 0

    def _can_submit(self, in_flight: int) -> bool:
        if in_flight >= self.config.max_concurrent_rollouts:
            return False
        future_rollouts = self.submitted_rollouts + 1
        lag_bucket = math.floor((future_rollouts - 1) / self.config.train_batch_size)
        return lag_bucket <= self.parameter_service.current_version + self.config.max_staleness

    async def run(self, stop_event: asyncio.Event) -> None:
        in_flight: set[asyncio.Task[Trajectory]] = set()
        worker_index = 0

        while not stop_event.is_set() or in_flight:
            submitted = False

            while not stop_event.is_set() and self._can_submit(len(in_flight)):
                prompt = self.env.sample_prompt()
                worker = self.workers[worker_index % len(self.workers)]
                worker_index += 1
                task = asyncio.create_task(worker.rollout(prompt))
                in_flight.add(task)
                self.submitted_rollouts += 1
                self.max_in_flight = max(self.max_in_flight, len(in_flight))
                submitted = True

            if not submitted and not stop_event.is_set() and not self._can_submit(len(in_flight)):
                self.staleness_pauses += 1

            if not in_flight:
                self.idle_loops += 1
                await asyncio.sleep(self.config.controller_idle_sleep)
                continue

            done, pending = await asyncio.wait(
                in_flight,
                timeout=self.config.controller_wait_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            in_flight = set(pending)

            if not done:
                self.idle_loops += 1
                continue

            for task in done:
                trajectory = task.result()
                self.completed_rollouts += 1
                if trajectory.interrupted:
                    self.interrupted_rollouts += 1
                stored = await self.replay_buffer.put(trajectory, drop_if_full=stop_event.is_set())
                if not stored:
                    self.dropped_after_stop += 1

    def stats(self) -> dict[str, float]:
        return {
            "submitted_rollouts": float(self.submitted_rollouts),
            "completed_rollouts": float(self.completed_rollouts),
            "interrupted_rollouts": float(self.interrupted_rollouts),
            "interrupt_rate": self.interrupted_rollouts / max(self.completed_rollouts, 1),
            "staleness_pauses": float(self.staleness_pauses),
            "idle_loops": float(self.idle_loops),
            "max_in_flight": float(self.max_in_flight),
            "dropped_after_stop": float(self.dropped_after_stop),
        }


class DecoupledPPOTrainer:
    def __init__(
        self,
        policy: ToyAutoregressivePolicy,
        parameter_service: ParameterService,
        config: ExperimentConfig,
    ) -> None:
        self.policy = policy
        self.parameter_service = parameter_service
        self.config = config

    def _token_gradient(
        self,
        token: TokenRecord,
        advantage: float,
        current_snapshot: PolicySnapshot,
        prox_snapshot: PolicySnapshot,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        probs_cur = self.policy.probs(token.features, current_snapshot, token.allow_answers, token.answer_only)
        probs_prox = self.policy.probs(token.features, prox_snapshot, token.allow_answers, token.answer_only)

        logp_cur = float(np.log(probs_cur[token.action] + 1e-12))
        logp_prox = float(np.log(probs_prox[token.action] + 1e-12))

        off_policy_correction = float(np.exp(np.clip(logp_prox - token.behavior_log_prob, -3.0, 3.0)))
        prox_ratio = float(np.exp(np.clip(logp_cur - logp_prox, -3.0, 3.0)))

        low = 1.0 - self.config.clip_epsilon
        high = 1.0 + self.config.clip_epsilon
        clipped_ratio = float(np.clip(prox_ratio, low, high))
        clipped = clipped_ratio != prox_ratio
        coeff = off_policy_correction * clipped_ratio * advantage

        one_hot = np.zeros(VOCAB_SIZE, dtype=np.float64)
        one_hot[token.action] = 1.0
        grad_logits = coeff * (one_hot - probs_cur)
        grad_w = np.outer(token.features, grad_logits)
        grad_b = grad_logits
        return grad_w, grad_b, off_policy_correction, 1.0 if clipped else 0.0

    def update(self, batch: list[Trajectory]) -> UpdateStats:
        rewards = np.array([trajectory.reward for trajectory in batch], dtype=np.float64)
        advantages = rewards - float(np.mean(rewards))
        advantages /= float(np.std(advantages) + 1e-6)

        prox_snapshot = self.policy.get_snapshot()
        current_snapshot = prox_snapshot.copy()

        importance_values: list[float] = []
        clip_values: list[float] = []

        for _ in range(self.config.update_epochs):
            grad_w = np.zeros_like(current_snapshot.weights)
            grad_b = np.zeros_like(current_snapshot.bias)
            token_count = 0

            for trajectory, advantage in zip(batch, advantages, strict=True):
                for token in trajectory.tokens:
                    token_grad_w, token_grad_b, importance, clip_value = self._token_gradient(
                        token=token,
                        advantage=float(advantage),
                        current_snapshot=current_snapshot,
                        prox_snapshot=prox_snapshot,
                    )
                    grad_w += token_grad_w
                    grad_b += token_grad_b
                    importance_values.append(importance)
                    clip_values.append(clip_value)
                    token_count += 1

            scale = self.config.learning_rate / max(token_count, 1)
            current_snapshot.weights += scale * grad_w
            current_snapshot.bias += scale * grad_b

        self.policy.load_snapshot(current_snapshot)

        reward_mean = float(np.mean(rewards))
        batch_accuracy = float(np.mean([trajectory.accuracy for trajectory in batch]))
        avg_behavior_version = float(np.mean([trajectory.mean_behavior_version for trajectory in batch]))
        clip_fraction = float(np.mean(clip_values)) if clip_values else 0.0
        avg_importance = float(np.mean(importance_values)) if importance_values else 1.0
        avg_reasoning_steps = float(np.mean([trajectory.reasoning_steps for trajectory in batch]))
        avg_decode_len = float(np.mean([trajectory.decode_len for trajectory in batch]))
        avg_traj_time_sec = float(np.mean([trajectory.wall_time for trajectory in batch]))
        avg_reward_latency_sec = float(np.mean([trajectory.reward_latency_sec for trajectory in batch]))
        avg_buffer_wait_sec = float(np.mean([trajectory.buffer_wait_sec for trajectory in batch]))
        avg_staleness = float(np.mean([trajectory.sampled_staleness for trajectory in batch]))
        avg_version_span = float(np.mean([trajectory.version_span for trajectory in batch]))
        interruption_rate = float(np.mean([1.0 if trajectory.interrupted else 0.0 for trajectory in batch]))
        hard_prompt_fraction = float(
            np.mean([1.0 if trajectory.prompt.required_reasoning_steps > 0 else 0.0 for trajectory in batch])
        )

        return UpdateStats(
            reward_mean=reward_mean,
            batch_accuracy=batch_accuracy,
            avg_behavior_version=avg_behavior_version,
            clip_fraction=clip_fraction,
            avg_importance=avg_importance,
            avg_reasoning_steps=avg_reasoning_steps,
            avg_decode_len=avg_decode_len,
            avg_traj_time_sec=avg_traj_time_sec,
            avg_reward_latency_sec=avg_reward_latency_sec,
            avg_buffer_wait_sec=avg_buffer_wait_sec,
            avg_staleness=avg_staleness,
            avg_version_span=avg_version_span,
            interruption_rate=interruption_rate,
            hard_prompt_fraction=hard_prompt_fraction,
        )


def evaluate_policy(
    policy: ToyAutoregressivePolicy,
    env: ToyReasoningEnv,
    config: ExperimentConfig,
) -> dict[str, object]:
    prompts = env.build_eval_set(config.eval_size)
    accuracies: list[float] = []
    rewards: list[float] = []
    lengths: list[float] = []
    reasoning_steps_list: list[float] = []
    easy_accuracies: list[float] = []
    hard_accuracies: list[float] = []
    example_traces: list[dict[str, object]] = []

    snapshot = policy.get_snapshot()
    for prompt in prompts:
        tokens, predicted_answer = policy.greedy_decode(prompt, config.max_decode_steps, snapshot)
        reasoning_steps = sum(1 for token in tokens if token in REASONING_TOKENS)
        reward = env.score(prompt, predicted_answer, reasoning_steps)
        accuracy = 1.0 if predicted_answer == prompt.target else 0.0

        rewards.append(reward)
        accuracies.append(accuracy)
        lengths.append(float(len(tokens)))
        reasoning_steps_list.append(float(reasoning_steps))
        if prompt.required_reasoning_steps > 0:
            hard_accuracies.append(accuracy)
        else:
            easy_accuracies.append(accuracy)

        if len(example_traces) < config.trace_examples:
            example_traces.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "prompt": prompt.text,
                    "target": prompt.target,
                    "required_reasoning_steps": prompt.required_reasoning_steps,
                    "tokens": [token_name(token) for token in tokens],
                    "predicted_answer": predicted_answer,
                    "accuracy": accuracy,
                    "reward": round(reward, 4),
                }
            )

    return {
        "eval_accuracy": float(np.mean(accuracies)),
        "eval_reward": float(np.mean(rewards)),
        "avg_decode_len": float(np.mean(lengths)),
        "eval_avg_reasoning_steps": float(np.mean(reasoning_steps_list)),
        "eval_easy_accuracy": float(np.mean(easy_accuracies)) if easy_accuracies else 0.0,
        "eval_hard_accuracy": float(np.mean(hard_accuracies)) if hard_accuracies else 0.0,
        "sample_eval_traces": example_traces,
    }


def _summarize_workers(workers: list[InterruptibleRolloutWorker]) -> dict[str, object]:
    worker_stats = [worker.stats() for worker in workers]
    return {
        "num_workers": float(len(workers)),
        "total_completed_rollouts": float(sum(worker["rollouts_completed"] for worker in worker_stats)),
        "total_refresh_count": float(sum(worker["refresh_count"] for worker in worker_stats)),
        "avg_worker_rollout_time_sec": float(np.mean([worker["avg_rollout_time_sec"] for worker in worker_stats])),
        "per_worker": worker_stats,
    }


def _sample_traces(batch: list[Trajectory], trace_examples: int) -> list[dict[str, object]]:
    ordered = sorted(batch, key=lambda trajectory: trajectory.prompt.prompt_id)
    return [trajectory.to_trace() for trajectory in ordered[:trace_examples]]


def _build_summary(
    *,
    mode: str,
    config: ExperimentConfig,
    last_update_stats: UpdateStats,
    wall_time: float,
    completed_rollouts: int,
    interrupted_rollouts: int,
    eval_stats: dict[str, object],
    train_history: list[dict[str, float]],
    sample_batch: list[Trajectory],
    parameter_service: ParameterService,
    reward_service: RewardService,
    workers: list[InterruptibleRolloutWorker],
    replay_buffer: ReplayBuffer | None = None,
    controller: RolloutController | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "mode": mode,
        "wall_time_sec": wall_time,
        "throughput_traj_per_sec": completed_rollouts / wall_time,
        "updates": float(config.num_updates),
        "final_train_reward": last_update_stats.reward_mean,
        "final_batch_accuracy": last_update_stats.batch_accuracy,
        "avg_behavior_version": last_update_stats.avg_behavior_version,
        "clip_fraction": last_update_stats.clip_fraction,
        "avg_importance": last_update_stats.avg_importance,
        "avg_reasoning_steps": last_update_stats.avg_reasoning_steps,
        "avg_train_decode_len": last_update_stats.avg_decode_len,
        "avg_batch_staleness": last_update_stats.avg_staleness,
        "completed_rollouts": float(completed_rollouts),
        "interrupted_rollouts": float(interrupted_rollouts),
        "train_history": train_history,
        "sample_trajectories": _sample_traces(sample_batch, config.trace_examples),
        "parameter_service": parameter_service.stats(),
        "reward_service": reward_service.stats(),
        "workers": _summarize_workers(workers),
        **eval_stats,
    }

    if replay_buffer is not None:
        summary["replay_buffer"] = replay_buffer.stats()
    if controller is not None:
        summary["controller"] = controller.stats()

    return summary


def _build_workers(
    policy: ToyAutoregressivePolicy,
    parameter_service: ParameterService,
    reward_service: RewardService,
    config: ExperimentConfig,
    seed_offset: int,
) -> list[InterruptibleRolloutWorker]:
    return [
        InterruptibleRolloutWorker(
            worker_id=worker_id,
            policy=policy,
            parameter_service=parameter_service,
            reward_service=reward_service,
            config=config,
            seed=seed_offset + worker_id,
        )
        for worker_id in range(config.num_workers)
    ]


async def run_async_experiment(config: ExperimentConfig) -> dict[str, object]:
    env = ToyReasoningEnv(seed=config.seed)
    eval_env = ToyReasoningEnv(seed=config.seed + 101)
    policy = ToyAutoregressivePolicy(seed=config.seed)
    parameter_service = ParameterService(policy.get_snapshot(), config.history_limit)
    reward_service = RewardService(env=env, base_delay=config.reward_delay)
    replay_buffer = ReplayBuffer(capacity=config.buffer_capacity, prioritize_oldest=config.prioritize_oldest)
    trainer = DecoupledPPOTrainer(policy=policy, parameter_service=parameter_service, config=config)
    workers = _build_workers(
        policy=policy,
        parameter_service=parameter_service,
        reward_service=reward_service,
        config=config,
        seed_offset=config.seed + 1000,
    )
    controller = RolloutController(
        env=env,
        workers=workers,
        replay_buffer=replay_buffer,
        parameter_service=parameter_service,
        config=config,
    )

    stop_event = asyncio.Event()
    controller_task = asyncio.create_task(controller.run(stop_event))

    start = time.perf_counter()
    last_update_stats: UpdateStats | None = None
    train_history: list[dict[str, float]] = []
    last_batch: list[Trajectory] = []

    for update_idx in range(config.num_updates):
        batch = await replay_buffer.get_many(
            config.train_batch_size,
            current_version=parameter_service.current_version,
        )
        last_batch = batch
        last_update_stats = trainer.update(batch)
        new_version = await parameter_service.publish(policy.get_snapshot())
        train_history.append(
            last_update_stats.as_dict(
                update_index=update_idx + 1,
                policy_version=new_version,
                buffer_size=replay_buffer.size(),
            )
        )

        if config.verbose and (update_idx + 1) % max(1, config.num_updates // 6) == 0:
            print(
                f"[async] update={update_idx + 1:02d} "
                f"version={new_version:02d} "
                f"reward={last_update_stats.reward_mean:.3f} "
                f"batch_acc={last_update_stats.batch_accuracy:.3f} "
                f"stale={last_update_stats.avg_staleness:.2f} "
                f"buffer={replay_buffer.size():03d}"
            )

    stop_event.set()
    await controller_task
    wall_time = time.perf_counter() - start
    eval_stats = evaluate_policy(policy, eval_env, config)

    assert last_update_stats is not None
    return _build_summary(
        mode="async",
        config=config,
        last_update_stats=last_update_stats,
        wall_time=wall_time,
        completed_rollouts=controller.completed_rollouts,
        interrupted_rollouts=controller.interrupted_rollouts,
        eval_stats=eval_stats,
        train_history=train_history,
        sample_batch=last_batch,
        parameter_service=parameter_service,
        reward_service=reward_service,
        workers=workers,
        replay_buffer=replay_buffer,
        controller=controller,
    )


async def run_sync_experiment(config: ExperimentConfig) -> dict[str, object]:
    env = ToyReasoningEnv(seed=config.seed)
    eval_env = ToyReasoningEnv(seed=config.seed + 101)
    policy = ToyAutoregressivePolicy(seed=config.seed)
    parameter_service = ParameterService(policy.get_snapshot(), config.history_limit)
    reward_service = RewardService(env=env, base_delay=config.reward_delay)
    trainer = DecoupledPPOTrainer(policy=policy, parameter_service=parameter_service, config=config)
    workers = _build_workers(
        policy=policy,
        parameter_service=parameter_service,
        reward_service=reward_service,
        config=config,
        seed_offset=config.seed + 2024,
    )

    start = time.perf_counter()
    last_update_stats: UpdateStats | None = None
    completed_rollouts = 0
    train_history: list[dict[str, float]] = []
    last_batch: list[Trajectory] = []

    for update_idx in range(config.num_updates):
        version, frozen_snapshot = await parameter_service.get_latest()
        prompts = [env.sample_prompt() for _ in range(config.train_batch_size)]
        batch = await asyncio.gather(
            *(
                workers[prompt_idx % len(workers)].rollout(
                    prompt,
                    frozen_snapshot=frozen_snapshot,
                    frozen_version=version,
                )
                for prompt_idx, prompt in enumerate(prompts)
            )
        )
        last_batch = batch
        completed_rollouts += len(batch)
        last_update_stats = trainer.update(batch)
        new_version = await parameter_service.publish(policy.get_snapshot())
        train_history.append(
            last_update_stats.as_dict(
                update_index=update_idx + 1,
                policy_version=new_version,
                buffer_size=0,
            )
        )

        if config.verbose and (update_idx + 1) % max(1, config.num_updates // 6) == 0:
            print(
                f"[sync ] update={update_idx + 1:02d} "
                f"version={new_version:02d} "
                f"reward={last_update_stats.reward_mean:.3f} "
                f"batch_acc={last_update_stats.batch_accuracy:.3f}"
            )

    wall_time = time.perf_counter() - start
    eval_stats = evaluate_policy(policy, eval_env, config)

    assert last_update_stats is not None
    return _build_summary(
        mode="sync",
        config=config,
        last_update_stats=last_update_stats,
        wall_time=wall_time,
        completed_rollouts=completed_rollouts,
        interrupted_rollouts=0,
        eval_stats=eval_stats,
        train_history=train_history,
        sample_batch=last_batch,
        parameter_service=parameter_service,
        reward_service=reward_service,
        workers=workers,
    )
