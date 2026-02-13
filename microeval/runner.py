import asyncio
import logging
from statistics import mean, stdev

from path import Path

from microeval.evaluator import EvaluationRunner
from microeval.llm import get_llm_client
from microeval.schemas import RunConfig, RunResult, evals_dir
from microeval.yamlx import save_yaml

logger = logging.getLogger(__name__)


def get_eval_llm_client(config: RunConfig, model_to_test: str):
    """
    Get LLM client for evaluations.
    Uses eval_service/eval_model if specified, otherwise falls back to the tested LLM.

    :param config: RunConfig with evaluation settings
    :param model_to_test: The model being tested (used as fallback if eval_model not specified)
    """
    if config.eval_service:
        kwargs = {}
        if config.eval_model:
            kwargs["model"] = config.eval_model
        logger.info(f"Using separate LLM for evaluation: {config.eval_service}")
        return get_llm_client(config.eval_service, **kwargs)

    return get_llm_client(config.service, model=model_to_test)


class Runner:
    def __init__(self, file_path: str):
        self._config = RunConfig.read_from_yaml(file_path)

    async def run(self):
        """Run evaluations for all configured models."""
        evals_dir.results.makedirs_p()
        base_results_filename = Path(self._config.file_path).stem

        # Run evaluations for each model
        for model in self._config.model:
            await self._run_single_model(model, base_results_filename)

    async def _run_single_model(self, model: str, base_results_filename: str):
        """Run evaluation for a single model."""
        try:
            # Create model-specific results filename
            if len(self._config.model) > 1:
                results_filename = f"{base_results_filename}-{model}.yaml"
            else:
                results_filename = f"{base_results_filename}.yaml"
            results_path = evals_dir.results / results_filename

            if results_path.exists():
                results_path.remove()
                logger.info(f"Removed existing results file '{results_path}'")

            logger.info(f"Running evaluation for model: {model}")

            # Create LLM clients for this model
            llm = get_llm_client(self._config.service, model=model)
            eval_llm = get_eval_llm_client(self._config, model)
            cost_per_token = llm.get_token_cost()
            evaluation_runner = EvaluationRunner(eval_llm, self._config)

            await llm.connect()
            if eval_llm is not llm:
                await eval_llm.connect()

            fields = self._config.evaluators + [
                "elapsed_seconds",
                "token_count",
                "cost",
            ]
            eval_results_dict = {f: RunResult(name=f) for f in fields}

            response_texts = []
            for i in range(self._config.repeat):
                logger.info(f">>> Evaluate iteration {i + 1}/{self._config.repeat} for model {model}")

                response = await llm.get_completion(
                    messages=[
                        {"role": "system", "content": self._config.prompt},
                        {"role": "user", "content": self._config.input},
                    ],
                    temperature=self._config.temperature,
                )

                # Check if the response contains an error
                if "error" in response.get("metadata", {}):
                    error_msg = response["metadata"]["error"]
                    logger.error(f"Chat client error: {error_msg}")
                    raise RuntimeError(f"Chat client error: {error_msg}")

                response_texts.append(response["text"])

                elapsed_seconds = response["metadata"]["usage"]["elapsed_seconds"]
                logger.debug(f"ElapsedSeconds: {elapsed_seconds}")

                token_count = response["metadata"]["usage"].get("total_tokens", 0)
                cost_value = (
                    token_count * cost_per_token / 1000
                    if token_count is not None
                    else None
                )
                logger.debug(f"TokenCount: {token_count}")

                eval_results_dict["elapsed_seconds"].values.append(elapsed_seconds)
                eval_results_dict["token_count"].values.append(token_count)
                eval_results_dict["cost"].values.append(cost_value)

                results = await evaluation_runner.evaluate_response(response)
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
                "model": model,
                "texts": response_texts,
                "evaluations": evaluations
            }
            save_yaml(eval_results, results_path)

            logger.info(f"Results saved to '{results_path}'")
        except Exception as e:
            logger.error(f"Error during run for model {model}: {e}")
            raise
        finally:
            await llm.close()
            if eval_llm is not llm:
                await eval_llm.close()


async def run_all(file_paths):
    for run_config in file_paths:
        try:
            await Runner(run_config).run()
        except Exception as e:
            logger.error(f"Job failed: {run_config} - {e}")


def main():
    import argparse

    from microeval.logger import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Run LLM evaluations")
    parser.add_argument(
        "evals_dir",
        help="Base directory for evals (e.g., my-evals)",
    )
    args = parser.parse_args()

    evals_dir.set_base(args.evals_dir)

    logger.info(f"Running all configs in `./{evals_dir.runs}/*.yaml`")
    file_paths = list(evals_dir.runs.glob("*.yaml"))

    if not file_paths:
        logger.warning(f"No config files found in {evals_dir.runs}")
        return

    asyncio.run(run_all(file_paths))


if __name__ == "__main__":
    main()
