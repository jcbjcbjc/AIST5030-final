from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .toy_env import ANSWER_TOKENS, REASONING_TOKENS, Prompt, VOCAB_SIZE


STEP_BUCKETS = 5
FEATURE_DIM = 8 + 19 + STEP_BUCKETS + 6


@dataclass(slots=True)
class PolicySnapshot:
    weights: np.ndarray
    bias: np.ndarray

    def copy(self) -> "PolicySnapshot":
        return PolicySnapshot(weights=self.weights.copy(), bias=self.bias.copy())


class ToyAutoregressivePolicy:
    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        weights = rng.normal(0.0, 0.08, size=(FEATURE_DIM, VOCAB_SIZE))
        bias = np.full(VOCAB_SIZE, -0.16, dtype=np.float64)
        bias[list(REASONING_TOKENS)] = 0.52

        target_offset = 8
        step_offset = target_offset + 19
        required_reasoning_index = 7

        for answer_token in ANSWER_TOKENS:
            weights[target_offset + answer_token, answer_token] += 1.05
            weights[step_offset + 2, answer_token] += 0.12
            weights[step_offset + 3, answer_token] += 0.24
            weights[step_offset + 4, answer_token] += 0.34

        for reasoning_token in REASONING_TOKENS:
            weights[required_reasoning_index, reasoning_token] += 0.18
            weights[step_offset + 0, reasoning_token] += 0.34
            weights[step_offset + 1, reasoning_token] += 0.20
            weights[step_offset + 2, reasoning_token] += 0.08

        self.snapshot = PolicySnapshot(weights=weights, bias=bias)

    def get_snapshot(self) -> PolicySnapshot:
        return self.snapshot.copy()

    def load_snapshot(self, snapshot: PolicySnapshot) -> None:
        self.snapshot = snapshot.copy()

    def _prev_category(self, prev_token: int | None) -> int:
        if prev_token is None:
            return 0
        if prev_token in REASONING_TOKENS:
            return 1 + REASONING_TOKENS.index(prev_token)
        return 5

    def featurize(self, prompt: Prompt, step_index: int, prev_token: int | None) -> np.ndarray:
        base = np.array(
            [
                1.0,
                prompt.left / 9.0,
                prompt.right / 9.0,
                prompt.target / 18.0,
                prompt.operand_gap / 9.0,
                float(prompt.carry),
                prompt.difficulty / 2.0,
                prompt.required_reasoning_steps / 2.0,
            ],
            dtype=np.float64,
        )

        target_one_hot = np.zeros(19, dtype=np.float64)
        target_one_hot[prompt.target] = 1.0

        step_one_hot = np.zeros(STEP_BUCKETS, dtype=np.float64)
        step_one_hot[min(step_index, STEP_BUCKETS - 1)] = 1.0

        prev_one_hot = np.zeros(6, dtype=np.float64)
        prev_one_hot[self._prev_category(prev_token)] = 1.0

        return np.concatenate([base, target_one_hot, step_one_hot, prev_one_hot])

    def _masked_logits(
        self,
        features: np.ndarray,
        snapshot: PolicySnapshot,
        allow_answers: bool,
        answer_only: bool,
    ) -> np.ndarray:
        logits = features @ snapshot.weights + snapshot.bias
        if answer_only:
            masked = np.full_like(logits, -1e9)
            masked[list(ANSWER_TOKENS)] = logits[list(ANSWER_TOKENS)]
            return masked
        if not allow_answers:
            masked = np.full_like(logits, -1e9)
            masked[list(REASONING_TOKENS)] = logits[list(REASONING_TOKENS)]
            return masked
        return logits

    def probs(
        self,
        features: np.ndarray,
        snapshot: PolicySnapshot,
        allow_answers: bool,
        answer_only: bool,
    ) -> np.ndarray:
        logits = self._masked_logits(features, snapshot, allow_answers, answer_only)
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        exp_logits_sum = np.sum(exp_logits)
        return exp_logits / exp_logits_sum

    def sample_action(
        self,
        prompt: Prompt,
        step_index: int,
        prev_token: int | None,
        rng: np.random.Generator,
        snapshot: PolicySnapshot,
        allow_answers: bool,
        answer_only: bool,
    ) -> tuple[int, float, np.ndarray]:
        features = self.featurize(prompt, step_index, prev_token)
        probs = self.probs(features, snapshot, allow_answers, answer_only)
        action = int(rng.choice(np.arange(VOCAB_SIZE), p=probs))
        log_prob = float(np.log(probs[action] + 1e-12))
        return action, log_prob, features

    def greedy_decode(
        self,
        prompt: Prompt,
        max_decode_steps: int,
        snapshot: PolicySnapshot | None = None,
    ) -> tuple[list[int], int | None]:
        active_snapshot = self.snapshot if snapshot is None else snapshot
        prev_token: int | None = None
        tokens: list[int] = []
        predicted_answer: int | None = None

        for step in range(max_decode_steps):
            answer_only = step == max_decode_steps - 1
            allow_answers = answer_only or step >= prompt.required_reasoning_steps
            features = self.featurize(prompt, step, prev_token)
            probs = self.probs(features, active_snapshot, allow_answers, answer_only)
            action = int(np.argmax(probs))
            tokens.append(action)
            if action in ANSWER_TOKENS:
                predicted_answer = action
                break
            prev_token = action

        return tokens, predicted_answer
