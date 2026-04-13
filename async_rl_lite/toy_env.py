from __future__ import annotations

from dataclasses import dataclass


ANSWER_TOKENS = tuple(range(19))
REASONING_TOKENS = (19, 20, 21, 22)
VOCAB_SIZE = len(ANSWER_TOKENS) + len(REASONING_TOKENS)


def token_name(token_id: int) -> str:
    if token_id in ANSWER_TOKENS:
        return f"ANS_{token_id}"
    return f"R_{token_id - ANSWER_TOKENS[-1] - 1}"


@dataclass(slots=True)
class Prompt:
    prompt_id: int
    left: int
    right: int

    @property
    def target(self) -> int:
        return self.left + self.right

    @property
    def carry(self) -> int:
        return 1 if self.target >= 10 else 0

    @property
    def operand_gap(self) -> int:
        return abs(self.left - self.right)

    @property
    def difficulty(self) -> int:
        return self.carry + int(self.operand_gap >= 5)

    @property
    def required_reasoning_steps(self) -> int:
        if self.carry:
            return 2
        if self.target >= 9 or self.operand_gap >= 6:
            return 1
        return 0

    @property
    def estimated_cost(self) -> float:
        return 1.0 + 0.35 * self.difficulty + 0.40 * self.required_reasoning_steps

    @property
    def text(self) -> str:
        return f"What is {self.left} + {self.right}?"


class ToyReasoningEnv:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._counter = 0

    def sample_prompt(self) -> Prompt:
        index = (self._counter * 37 + self.seed) % 100
        left = index // 10
        right = index % 10
        prompt = Prompt(prompt_id=self._counter, left=left, right=right)
        self._counter += 1
        return prompt

    def build_eval_set(self, size: int) -> list[Prompt]:
        prompts: list[Prompt] = []
        for idx in range(size):
            index = (idx * 29 + self.seed + 11) % 100
            left = index // 10
            right = index % 10
            prompts.append(Prompt(prompt_id=10_000 + idx, left=left, right=right))
        return prompts

    def score(self, prompt: Prompt, predicted_answer: int | None, reasoning_steps: int) -> float:
        if predicted_answer is None:
            reward = -0.35
        else:
            distance = abs(predicted_answer - prompt.target)
            if distance == 0:
                reward = 1.0
            elif distance == 1:
                reward = 0.32
            elif distance == 2:
                reward = 0.16
            else:
                reward = max(-0.10, 0.26 - 0.05 * distance)

        matched_reasoning = min(reasoning_steps, prompt.required_reasoning_steps)
        reward += 0.08 * matched_reasoning

        if reasoning_steps < prompt.required_reasoning_steps:
            reward -= 0.12 * (prompt.required_reasoning_steps - reasoning_steps)
        elif reasoning_steps > prompt.required_reasoning_steps + 1:
            reward -= 0.04 * (reasoning_steps - prompt.required_reasoning_steps - 1)

        if prompt.carry and predicted_answer == prompt.target and reasoning_steps >= prompt.required_reasoning_steps:
            reward += 0.05

        return max(-0.50, min(1.20, reward))
