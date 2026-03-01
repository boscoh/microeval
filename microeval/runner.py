import asyncio
import logging
from statistics import mean, stdev
from typing import TYPE_CHECKING, Optional

import pydash
from path import Path

from microeval.config import _get_connected_llm_client, _get_default_model
from microeval.evaluator import EvaluationRunner
from microeval.schemas import RunConfig, RunResult, evals_dir
from microeval.utils import save_yaml

if TYPE_CHECKING:
    from microeval.llm import SimpleLLMClient

logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, config: RunConfig):
        """Initialize Runner with config.

        Clients are created and connected when connect() is called.

        :param config: RunConfig with evaluation settings
        """
        self._config = config
        self._main_llm_client = None
        self._eval_llm_client = None
        self._eval_embed_llm_client = None
        self._evaluation_runner = None
        self._connected = False

    def _resolve_service_model(
        self, service: str, model: str, model_type: str = "chat_models"
    ) -> tuple[str, str]:
        """Resolve service and model with defaults.

        :param service: Service name
        :param model: Model name (may be empty)
        :param model_type: Type of models to look up ("chat_models" or "embed_models")
        :return: Tuple of (service, model) with defaults applied
        """
        if not model:
            model = _get_default_model(service, model_type) or "default"
        return service, model

    async def _get_main_llm_client(self) -> "SimpleLLMClient":
        """Get and connect main LLM client for the query being tested."""
        service, model = self._resolve_service_model(
            self._config.chat_service, self._config.model, "chat_models"
        )
        return await _get_connected_llm_client(service, model)

    async def _get_eval_llm_client(self) -> "SimpleLLMClient":
        """Get and connect LLM client for LLM-based evaluators."""
        if self._config.eval_chat_service:
            service = self._config.eval_chat_service
            model = self._config.eval_chat_model
        else:
            service = self._config.chat_service
            model = self._config.model

        service, model = self._resolve_service_model(service, model, "chat_models")
        return await _get_connected_llm_client(service, model)

    async def _get_embed_client(self) -> "SimpleLLMClient":
        """Get and connect embedding client for embedding-based evaluators.

        Selection priority:
        1. eval_embed_service/eval_embed_model if specified
        2. Default embed model from eval_chat_service or chat_service
        3. Use eval_chat_service or chat_service client if it supports embeddings
        4. Fall back to OpenAI text-embedding-3-small
        """
        if self._config.eval_embed_service:
            service, model = self._resolve_service_model(
                self._config.eval_embed_service,
                self._config.eval_embed_model,
                "embed_models",
            )
            return await _get_connected_llm_client(service, model)

        eval_chat_service = self._config.eval_chat_service or self._config.chat_service

        if eval_chat_service and eval_chat_service != "groq":
            client = await self._try_get_embed_client_from_service(eval_chat_service)
            if client:
                return client

        if self._config.chat_service and self._config.chat_service != "groq":
            client = await self._try_get_embed_client_from_service(
                self._config.chat_service
            )
            if client:
                return client

        return await _get_connected_llm_client("openai", "text-embedding-3-small")

    async def _try_get_embed_client_from_service(
        self, service: str
    ) -> Optional["SimpleLLMClient"]:
        """Try to get an embedding client from a service.

        :param service: Service name to try
        :return: Embedding client if found, None otherwise
        """
        default_embed_model = _get_default_model(service, "embed_models")
        if default_embed_model:
            return await _get_connected_llm_client(service, default_embed_model)

        if self._config.eval_chat_service and service == self._config.eval_chat_service:
            model = self._config.eval_chat_model or "default"
        else:
            model = self._config.model or "default"

        client = await _get_connected_llm_client(service, model)
        if hasattr(client, "get_embedding"):
            return client

        return None

    async def connect(self) -> bool:
        """Get and connect all LLM clients (idempotent - clients are cached by service/model).

        Clients are automatically connected and cached, so multiple calls with the same
        config return the same client instances.
        """
        self._main_llm_client = await self._get_main_llm_client()
        self._eval_llm_client = await self._get_eval_llm_client()
        self._eval_embed_llm_client = await self._get_embed_client()

        self._evaluation_runner = EvaluationRunner(
            self._eval_llm_client, self._config, self._eval_embed_llm_client
        )

        self._connected = True
        return True

    async def run(self):
        try:
            evals_dir.results.makedirs_p()
            results_filename = Path(self._config.file_path).stem + ".yaml"
            results_path = evals_dir.results / results_filename
            if results_path.exists():
                results_path.remove()
                logger.info(f"Removed existing results file '{results_path}'")

            if not self._connected:
                await self.connect()

            fields = self._config.evaluators + [
                "elapsed_seconds",
                "token_count",
                "cost",
            ]
            eval_results_dict = {f: RunResult(name=f) for f in fields}

            response_texts = []
            run_id = Path(self._config.file_path).stem
            n = self._config.repeat
            for i in range(n):
                q_svc = self._main_llm_client.service
                q_model = self._main_llm_client.model or "default"
                logger.info(
                    f"> Eval {i + 1}/{n} '{run_id}' query: '{q_svc}:{q_model}'"
                )

                response = await self._main_llm_client.get_completion(
                    messages=[
                        {"role": "system", "content": self._config.prompt},
                        {"role": "user", "content": self._config.input},
                    ],
                    temperature=self._config.temperature,
                )

                response_texts.append(response["text"])

                elapsed_seconds = pydash.get(response, "metadata.usage.elapsed_seconds")
                logger.debug(f"ElapsedSeconds: {elapsed_seconds}")

                usage = pydash.get(response, "metadata.usage")
                token_count = pydash.get(usage, "prompt_tokens", 0) + pydash.get(
                    usage, "completion_tokens", 0
                )
                cost_value = self._main_llm_client.get_token_cost(
                    pydash.get(usage, "prompt_tokens", 0),
                    pydash.get(usage, "completion_tokens", 0),
                )
                logger.debug(f"TokenCount: {token_count}")

                eval_results_dict["elapsed_seconds"].values.append(elapsed_seconds)
                eval_results_dict["token_count"].values.append(token_count)
                eval_results_dict["cost"].values.append(cost_value)

                e_svc = self._eval_llm_client.service
                e_model = self._eval_llm_client.model or "default"
                logger.info(
                    f"> Eval {i + 1}/{n} '{run_id}' eval: '{e_svc}:{e_model}'"
                )
                results = await self._evaluation_runner.evaluate_response(response)
                for evaluator_name, value in results.items():
                    eval_results_dict[evaluator_name].values.append(value["score"])

            for eval_result in eval_results_dict.values():
                valid_values = [v for v in eval_result.values if v is not None]
                if valid_values:
                    eval_result.average = mean(valid_values)
                    eval_result.standard_deviation = (
                        stdev(valid_values) if len(valid_values) > 1 else 0.0
                    )

            evaluations = [result.model_dump() for result in eval_results_dict.values()]

            eval_results = {
                "texts": response_texts,
                "evaluations": evaluations,
                "eval_models": {
                    "eval_chat_service": self._eval_llm_client.service,
                    "eval_chat_model": self._eval_llm_client.model or "default",
                    "eval_embed_service": self._eval_embed_llm_client.service,
                    "eval_embed_model": self._eval_embed_llm_client.model or "default",
                },
            }
            save_yaml(eval_results, results_path)

            logger.info(f"Results saved to '{results_path}'")
        except Exception as e:
            logger.error(f"Error during run: {e}")
            raise


def create_runner(file_path: str) -> Runner:
    """Create a Runner instance from config file.

    Clients are created and connected when connect() is called.

    :param file_path: Path to run configuration YAML file
    :return: Runner instance (call connect() to initialize clients)
    """
    config = RunConfig.read_from_yaml(file_path)
    return Runner(config)


async def _run_one(file_path: str) -> None:
    """Create runner, connect, and run a single evaluation. Raises on failure."""
    runner = create_runner(file_path)
    await runner.connect()
    await runner.run()


async def run_all(file_paths):
    """Run all evaluations in parallel.

    Each run uses its own Runner. LLM clients are cached by (service, model) so
    connections are shared. Rate limiting (e.g. OPENAI_RPM) applies across runs.
    """
    if not file_paths:
        return
    tasks = [_run_one(p) for p in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for file_path, result in zip(file_paths, results):
        if isinstance(result, Exception):
            logger.error(f"Run failed '{file_path}': {result}")
