import logging
from statistics import mean, stdev

from path import Path

from microeval.evaluator import EvaluationRunner
from microeval.llm import get_llm_client
from microeval.schemas import RunConfig, RunResult, evals_dir
from microeval.utils import save_yaml

logger = logging.getLogger(__name__)


def get_eval_llm_client(config: RunConfig):
    """
    Get LLM client for evaluations.
    Uses eval_service/eval_model if specified, otherwise falls back to the tested LLM.
    """
    if config.eval_service:
        kwargs = {}
        if config.eval_model:
            kwargs["model"] = config.eval_model
        logger.info(f"Using separate LLM for evaluation: {config.eval_service}")
        return get_llm_client(config.eval_service, **kwargs)

    return get_llm_client(config.service, model=config.model)


class Runner:
    def __init__(self, file_path: str):
        self._config = RunConfig.read_from_yaml(file_path)
        self._llm = get_llm_client(self._config.service, model=self._config.model)
        self._eval_llm = get_eval_llm_client(self._config)
        self._evaluation_runner = EvaluationRunner(self._eval_llm, self._config)
        self._connected = False

    async def connect(self) -> bool:
        await self._llm.connect()
        if self._eval_llm is not self._llm:
            await self._eval_llm.connect()
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
            for i in range(self._config.repeat):
                logger.info(
                    f">>> Evaluate iteration {i + 1}/{self._config.repeat} {run_id}"
                )

                response = await self._llm.get_completion(
                    messages=[
                        {"role": "system", "content": self._config.prompt},
                        {"role": "user", "content": self._config.input},
                    ],
                    temperature=self._config.temperature,
                )

                response_texts.append(response["text"])

                elapsed_seconds = response["metadata"]["usage"]["elapsed_seconds"]
                logger.debug(f"ElapsedSeconds: {elapsed_seconds}")

                usage = response["metadata"]["usage"]
                token_count = usage.get("total_tokens", 0)
                cost_value = self._llm.get_token_cost(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )
                logger.debug(f"TokenCount: {token_count}")

                eval_results_dict["elapsed_seconds"].values.append(elapsed_seconds)
                eval_results_dict["token_count"].values.append(token_count)
                eval_results_dict["cost"].values.append(cost_value)

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

            eval_results = {"texts": response_texts, "evaluations": evaluations}
            save_yaml(eval_results, results_path)

            logger.info(f"Results saved to '{results_path}'")
        except Exception as e:
            logger.error(f"Error during run: {e}")
            raise
        finally:
            await self._llm.close()
            if self._eval_llm is not self._llm:
                await self._eval_llm.close()


async def run_all(file_paths):
    for run_config in file_paths:
        try:
            runner = Runner(run_config)
            await runner.connect()
        except Exception as e:
            logger.error(f"Failed to connect to LLM for '{run_config}': {e}")
            continue
        try:
            await runner.run()
        except Exception as e:
            logger.error(f"Job failed: '{run_config}' - {e}")
