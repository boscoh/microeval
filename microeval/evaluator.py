import json
import logging
import re
import textwrap
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from pydantic import BaseModel

if TYPE_CHECKING:
    from microeval.schemas import RunConfig

logger = logging.getLogger(__name__)

EVALUATOR_REGISTRY: Dict[str, Type["BaseEvaluator"]] = {}


def register_evaluator(name: str):
    def decorator(cls: Type["BaseEvaluator"]) -> Type["BaseEvaluator"]:
        EVALUATOR_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_available_evaluators() -> list[str]:
    return list(EVALUATOR_REGISTRY.keys())


class EvalResult(BaseModel):
    score: float = 0.5
    reasoning: str = ""
    elapsed_ms: int = 0
    token_count: int = 0


def parse_json_score(response_text: str) -> tuple[float, str]:
    """Parse JSON response, falls back to regex extraction if JSON fails."""
    try:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data = json.loads(text)
        score = float(data.get("score", 0.5))
        reasoning = data.get("reasoning", "")
        return max(0.0, min(1.0, score)), reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    numbers = re.findall(r"\b0?\.\d+\b|\b1(?:\.0+)?\b|\b0\b", response_text.strip())
    if numbers:
        return max(0.0, min(1.0, float(numbers[0]))), ""

    return 0.5, ""


class BaseEvaluator(ABC):
    def __init__(
        self, run_config: Any, llm: Any = None, params: Optional[Dict[str, Any]] = None
    ):
        self.run_config = run_config
        self.llm = llm
        self.params = params or {}

    @abstractmethod
    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        pass

    def _empty_result(self, score: float = 0.5, reasoning: str = "") -> Dict[str, Any]:
        return {
            "score": score,
            "reasoning": reasoning,
            "elapsed_ms": 0,
            "token_count": 0,
        }


class LLMEvaluator(BaseEvaluator):
    @abstractmethod
    def build_prompt(self, response_text: str) -> str:
        pass

    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        if not response_text.strip():
            return self._empty_result(
                score=0.0, reasoning="Empty response text provided"
            )

        prompt = self.build_prompt(response_text)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful evaluation assistant. Always respond with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.get_completion(messages)

        if "error" in response.get("metadata", {}):
            error_msg = response["metadata"]["error"]
            logger.error(f"LLM error in evaluation: {error_msg}")
            raise RuntimeError(f"LLM error: {error_msg}")

        response_text = response.get("text", "")
        score, reasoning = parse_json_score(response_text)

        return {
            "score": score,
            "reasoning": reasoning or response_text,
            "elapsed_ms": response.get("elapsed_ms", 0),
            "token_count": response.get("token_count", 0),
        }


@register_evaluator("coherence")
class CoherenceEvaluator(LLMEvaluator):
    def build_prompt(self, response_text: str) -> str:
        question = self.run_config.input or ""
        return textwrap.dedent(f"""
            Evaluate the coherence of the following answer to the given question.
            
            Coherence means:
            - The answer is logically structured
            - Ideas flow naturally from one to another
            - The response is internally consistent
            - The language is clear and well-organized
            
            Question: {question}
            Answer: {response_text}
            
            Respond with JSON in this exact format:
            {{"score": <number between 0.0 and 1.0>, "reasoning": "<brief explanation>"}}
        """).strip()


@register_evaluator("equivalence")
class EquivalenceEvaluator(LLMEvaluator):
    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        if not response_text.strip():
            return self._empty_result(
                score=0.0, reasoning="Empty response text provided"
            )

        if not self.run_config.output:
            return self._empty_result(
                score=0.0, reasoning="No expected answer provided for comparison"
            )

        answer = self.run_config.output
        if not answer.strip():
            return self._empty_result(score=0.0, reasoning="Empty expected answer")

        if answer.strip().lower() == response_text.strip().lower():
            return self._empty_result(
                score=1.0, reasoning="Response exactly matches expected answer"
            )

        return await super().evaluate(response_text)

    def build_prompt(self, response_text: str) -> str:
        answer = self.run_config.output
        return textwrap.dedent(f"""
            Rate the semantic equivalence of these two answers.

            Expected: {answer}
            Actual: {response_text}

            Score must be exactly 0.0 (wrong or unrelated), 0.5 (correct but incomplete or imprecise), or 1.0 (fully equivalent).

            Respond: {{"score": <0.0|0.5|1.0>, "reasoning": "<brief explanation>"}}
        """).strip()


@register_evaluator("word_count")
class WordCountEvaluator(BaseEvaluator):
    """Params: min_words, max_words, target_words (takes precedence)."""

    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        if not response_text.strip():
            return self._empty_result(
                score=0.0, reasoning="Empty response text provided"
            )

        min_words = self.params.get("min_words")
        max_words = self.params.get("max_words")
        target_words = self.params.get("target_words")

        word_count = len(response_text.split())

        if target_words is not None:
            if word_count == 0:
                return self._empty_result(score=0.0, reasoning="No words in response")
            distance = abs(word_count - target_words)
            if distance >= target_words:
                score = 0.5 * (1 - (distance - target_words) / (target_words + 1))
            else:
                score = 1.0 - (0.5 * (distance / target_words))
            return self._empty_result(
                score=score,
                reasoning=f"Word count: {word_count}, target: {target_words}",
            )

        if min_words is not None and word_count < min_words:
            score = 0.5 + (0.5 * min(1.0, word_count / max(1, min_words)))
            return self._empty_result(
                score=score,
                reasoning=f"Word count {word_count} below minimum {min_words}",
            )

        if max_words is not None and word_count > max_words:
            excess = word_count - max_words
            score = max(0.5, 1.0 - (0.5 * min(1.0, excess / max(1, max_words))))
            return self._empty_result(
                score=score,
                reasoning=f"Word count {word_count} exceeds maximum {max_words}",
            )

        return self._empty_result(score=1.0, reasoning=f"Word count: {word_count}")


class EvaluationRunner:
    def __init__(self, llm, run_config):
        self.llm = llm
        self.run_config: RunConfig = run_config
        self._evaluators: Dict[str, BaseEvaluator] = {}

        for evaluator_config in self.run_config.evaluators:
            if isinstance(evaluator_config, str):
                name = evaluator_config.lower()
                params = {}
            elif isinstance(evaluator_config, dict):
                name = evaluator_config.get("name", "").lower()
                params = evaluator_config.get("params", {})
            elif hasattr(evaluator_config, "name"):
                name = evaluator_config.name.lower()
                params = (
                    evaluator_config.params
                    if hasattr(evaluator_config, "params")
                    else {}
                )
            else:
                continue

            if name in EVALUATOR_REGISTRY:
                evaluator_cls = EVALUATOR_REGISTRY[name]
                self._evaluators[name] = evaluator_cls(
                    run_config=run_config, llm=llm, params=params
                )

    async def evaluate_response(self, response: Any) -> Dict[str, dict]:
        results = {}
        response_text = response.get("text", "")

        for evaluator_config in self.run_config.evaluators:
            if isinstance(evaluator_config, str):
                evaluator_name = evaluator_config.lower()
            elif isinstance(evaluator_config, dict):
                evaluator_name = evaluator_config.get("name", "").lower()
            elif hasattr(evaluator_config, "name"):
                evaluator_name = evaluator_config.name.lower()
            else:
                continue

            try:
                if evaluator_name in self._evaluators:
                    evaluator = self._evaluators[evaluator_name]
                    result = await evaluator.evaluate(response_text)
                    results[evaluator_name] = result
                else:
                    results[evaluator_name] = {
                        "score": 1.0,
                        "reasoning": f"Unknown evaluator: {evaluator_name}",
                        "elapsed_ms": 0,
                        "token_count": 0,
                    }
            except Exception as e:
                logging.error(
                    f"Error in {evaluator_name} evaluation: {e}", exc_info=True
                )
                results[evaluator_name] = {
                    "score": 0.5,
                    "reasoning": str(e),
                    "elapsed_ms": 0,
                    "token_count": 0,
                }

        return results
