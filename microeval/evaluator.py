import json
import logging
import math
import re
import textwrap
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import pydash

from microeval.utils import parse_json, snap_score

if TYPE_CHECKING:
    from microeval.schemas import RunConfig

logger = logging.getLogger(__name__)

_VALID_SCORES = (0.0, 0.5, 1.0)

EVALUATOR_REGISTRY: Dict[str, Type["BaseEvaluator"]] = {}


def register_evaluator(name: str):
    def decorator(cls: Type["BaseEvaluator"]) -> Type["BaseEvaluator"]:
        EVALUATOR_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_available_evaluators() -> list[str]:
    return list(EVALUATOR_REGISTRY.keys())


class BaseEvaluator(ABC):
    def __init__(
        self, run_config: Any, llm: Any = None, params: Optional[Dict[str, Any]] = None
    ):
        self.run_config = run_config
        self.llm = llm
        self.params = params or {}

    @abstractmethod
    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        """Evaluate a response and return a result dictionary.

        :param response_text: The text response to evaluate.
        :return: Evaluation result dictionary. Example::

                {
                  "score": 0.95,
                  "reasoning": "Response meets evaluation criteria",
                  "elapsed_ms": 1200,
                  "token_count": 150
                }
        """
        pass

    def _empty_result(self, score: float = 0.5, reasoning: str = "") -> Dict[str, Any]:
        """Create an empty evaluation result dictionary.

        :param score: Evaluation score (0.0-1.0).
        :param reasoning: Explanation for the score.
        :return: Evaluation result dictionary. Example::

                {
                  "score": 0.5,
                  "reasoning": "Response is partially relevant",
                  "elapsed_ms": 0,
                  "token_count": 0
                }
        """
        return {
            "score": score,
            "reasoning": reasoning,
            "elapsed_ms": 0,
            "token_count": 0,
        }


class LLMEvaluator(BaseEvaluator):
    valid_scores: tuple = _VALID_SCORES

    @abstractmethod
    def build_prompt(self, response_text: str) -> str:
        pass

    def _score_tool(self) -> Dict[str, Any]:
        """Get the tool definition for structured scoring.

        :return: Tool definition dictionary. Example::

                {
                  "type": "function",
                  "function": {
                    "name": "submit_score",
                    "description": "Submit the evaluation score and reasoning.",
                    "parameters": {
                      "type": "object",
                      "properties": {
                        "score": {
                          "type": "number",
                          "enum": [0.0, 0.5, 1.0],
                          "description": "Numeric score from the allowed values."
                        },
                        "reasoning": {
                          "type": "string",
                          "description": "Brief explanation of the score."
                        }
                      },
                      "required": ["score", "reasoning"]
                    }
                  }
                }
        """
        return {
            "type": "function",
            "function": {
                "name": "submit_score",
                "description": "Submit the evaluation score and reasoning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "enum": list(self.valid_scores),
                            "description": "Numeric score from the allowed values.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of the score.",
                        },
                    },
                    "required": ["score", "reasoning"],
                },
            },
        }

    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        if not response_text.strip():
            return self._empty_result(
                score=0.0, reasoning="Empty response text provided"
            )

        prompt = self.build_prompt(response_text)
        messages = [
            {"role": "system", "content": "You are an evaluation assistant."},
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.get_completion(messages, tools=[self._score_tool()])

        error_msg = pydash.get(response, "metadata.error")
        if error_msg:
            logger.error(f"LLM error in evaluation: {error_msg}")
            raise RuntimeError(f"LLM error: {error_msg}")

        tool_calls = response.get("tool_calls") or []
        if tool_calls:
            arguments = pydash.get(tool_calls, "[0].function.arguments")
            if isinstance(arguments, str):
                try:
                    args = json.loads(arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse tool arguments as JSON: {e}")
                    args = {}
            elif isinstance(arguments, dict):
                args = arguments
            else:
                args = {}

            score_value = pydash.get(args, "score", 0.5)
            if isinstance(score_value, dict):
                score_value = pydash.get(
                    score_value, "value", pydash.get(score_value, "score", 0.5)
                )
            if not isinstance(score_value, (int, float)):
                try:
                    score_value = float(score_value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid score value: {score_value}, using default 0.5"
                    )
                    score_value = 0.5
            score = snap_score(float(score_value), self.valid_scores)
            reasoning = pydash.get(args, "reasoning", "") or pydash.get(
                response, "text", ""
            )
        else:
            text = response.get("text", "")
            data = parse_json(text)
            if data is not None:
                score = snap_score(float(data.get("score", 0.5)), self.valid_scores)
                reasoning = data.get("reasoning", "") or text
            else:
                numbers = re.findall(r"\b0?\.\d+\b|\b1(?:\.0+)?\b|\b0\b", text)
                score = (
                    snap_score(float(numbers[0]), self.valid_scores) if numbers else 0.5
                )
                reasoning = text

        return {
            "score": score,
            "reasoning": reasoning,
            "elapsed_ms": response.get("elapsed_ms", 0),
            "token_count": response.get("token_count", 0),
        }


@register_evaluator("equivalence")
class EquivalenceEvaluator(LLMEvaluator):
    """Evaluate semantic equivalence between response and expected output.

    Expected params: None
    """

    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        """Evaluate semantic equivalence between response and expected output.

        :param response_text: The text response to evaluate.
        :return: Evaluation result dictionary. Example::

                {
                  "score": 0.95,
                  "reasoning": "The response is semantically equivalent to the expected output.",
                  "elapsed_ms": 1200,
                  "token_count": 150
                }
        """
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
            Compare these two answers for semantic equivalence. Ignore differences in formatting, capitalization, or phrasing that preserve the same meaning.

            Answer A: {answer}
            Answer B: {response_text}

            First, briefly explain what each answer says and whether the core meaning matches.

            Then assign a score:
            - 1.0: fully equivalent (same meaning, same key facts, no meaningful omissions)
            - 0.5: partially correct (right topic/direction but missing key details, imprecise, or adds significant unsupported claims)
            - 0.0: incorrect or unrelated (wrong answer, contradicts Answer A, or off-topic)
        """).strip()


@register_evaluator("relevance_llm")
class RelevanceLLMEvaluator(LLMEvaluator):
    """Rate how well the response addresses the question.

    Expected params: None
    """
    valid_scores = (0.0, 0.25, 0.5, 0.75, 1.0)

    def build_prompt(self, response_text: str) -> str:
        question = self.run_config.input or ""
        return textwrap.dedent(f"""
            Rate how well the following response addresses the question. Consider whether it stays on topic, covers the key aspects of the question, and avoids irrelevant content.

            Question: {question}
            Response: {response_text}

            First, briefly identify what the question is asking and whether the response addresses those points.

            Then assign a score:
            - 1.0: fully addresses the question with no irrelevant content
            - 0.75: mostly addresses the question with minor gaps or slight tangents
            - 0.5: partially addresses the question but misses significant aspects or includes substantial irrelevant content
            - 0.25: tangentially related but largely fails to address the question
            - 0.0: completely off-topic or does not engage with the question at all
        """).strip()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


@register_evaluator("relevance_embedding")
class RelevanceEmbeddingEvaluator(BaseEvaluator):
    """Calculate relevance using cosine similarity of embeddings between question and response.

    Expected params: None
    """

    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        if not response_text.strip():
            return self._empty_result(
                score=0.0, reasoning="Empty response text provided"
            )

        question = self.run_config.input or ""
        if not question.strip():
            return self._empty_result(
                score=0.5, reasoning="No question provided for comparison"
            )

        embed_client = self.llm
        if not embed_client:
            return self._empty_result(
                score=0.5, reasoning="No embedding client available"
            )

        try:
            if not hasattr(embed_client, "get_embedding"):
                return self._empty_result(
                    score=0.5,
                    reasoning="Embedding client does not support get_embedding",
                )
        except Exception:
            return self._empty_result(
                score=0.5, reasoning="Cannot check embedding support"
            )

        try:
            import time

            start_time = time.time()

            question_embedding = await embed_client.get_embedding(question)
            response_embedding = await embed_client.get_embedding(response_text)

            similarity = cosine_similarity(question_embedding, response_embedding)

            elapsed_ms = int((time.time() - start_time) * 1000)

            score = max(0.0, min(1.0, similarity))

            reasoning = f"Cosine similarity: {similarity:.4f} (score: {score:.4f})"

            return {
                "score": score,
                "reasoning": reasoning,
                "elapsed_ms": elapsed_ms,
                "token_count": 0,
            }
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {e}", exc_info=True)
            return self._empty_result(
                score=0.5, reasoning=f"Error calculating similarity: {str(e)}"
            )


@register_evaluator("word_count")
class WordCountEvaluator(BaseEvaluator):
    """Evaluate response length against word count parameters.

    Expected params:

    .. code-block:: json

        {
          "min_words": 50,
          "max_words": 200,
          "target_words": 100
        }

    Note: ``target_words`` takes precedence over ``min_words`` and ``max_words``.
    """

    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        """Evaluate response length against word count parameters.

        :param response_text: The text response to evaluate.
        :return: Evaluation result dictionary. Example::

                {
                  "score": 1.0,
                  "reasoning": "Word count: 150 (target: 100, range: 50-200)",
                  "elapsed_ms": 0,
                  "token_count": 0
                }
        """
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
    def __init__(self, llm, run_config, embed_client=None):
        self.llm = llm
        self.embed_client = embed_client
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
                evaluator_llm = (
                    embed_client
                    if name == "relevance_embedding" and embed_client
                    else llm
                )
                self._evaluators[name] = evaluator_cls(
                    run_config=run_config, llm=evaluator_llm, params=params
                )

    async def evaluate_response(self, response: Any) -> Dict[str, dict]:
        """Evaluate a single response using all registered evaluators.

        :param response: Response dictionary from LLM client. Example::

                {
                  "text": "The weather is sunny",
                  "metadata": {"usage": {...}}
                }

        :return: Dictionary mapping evaluator names to their results. Example::

                {
                  "equivalence": {
                    "score": 0.95,
                    "reasoning": "Very similar to expected output",
                    "elapsed_ms": 1200,
                    "token_count": 150
                  },
                  "relevance_llm": {
                    "score": 1.0,
                    "reasoning": "Highly relevant to the question",
                    "elapsed_ms": 1100,
                    "token_count": 140
                  }
                }
        """
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
