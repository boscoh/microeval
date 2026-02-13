import copy
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

from path import Path
from pydantic import BaseModel, Field

from microeval.llm import LLMService
from microeval.yamlx import load_yaml, save_yaml

logger = logging.getLogger(__name__)


class EvaluatorConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


TableType = Literal["result", "run", "prompt", "query"]

ext_from_table = {
    "result": ".yaml",
    "run": ".yaml",
    "prompt": ".txt",
    "query": ".yaml",
}


class EvalsDir:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.getenv("EVALS_DIR", "evals")
        self._base = Path(base_dir)

    @property
    def name(self) -> str:
        return str(self._base)

    @property
    def prompts(self) -> Path:
        return self._base / "prompts"

    @property
    def queries(self) -> Path:
        return self._base / "queries"

    @property
    def results(self) -> Path:
        return self._base / "results"

    @property
    def runs(self) -> Path:
        return self._base / "runs"

    def get_dir(self, table: TableType) -> Path:
        return {
            "result": self.results,
            "run": self.runs,
            "prompt": self.prompts,
            "query": self.queries,
        }[table]

    def set_base(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.getenv("EVALS_DIR", "evals")
        self._base = Path(base_dir)
        for d in [self.prompts, self.queries, self.results, self.runs]:
            d.makedirs_p()
        logger.info(f"Evals directory set to: {self._base}")


evals_dir = EvalsDir()


class RunConfig(BaseModel):
    file_path: Optional[str] = None
    query_ref: Optional[str] = None
    prompt_ref: Optional[str] = None
    prompt: str = ""
    input: str = ""
    output: str = ""
    service: LLMService
    model: str = ""
    repeat: int = 1
    temperature: float = 0.0
    evaluators: List[Union[str, EvaluatorConfig, Dict[str, Any]]] = Field(
        default_factory=lambda: ["coherence"]
    )
    eval_service: Optional[LLMService] = None
    eval_model: Optional[str] = None

    @staticmethod
    def read_from_yaml(file_path: str) -> "RunConfig":
        data = load_yaml(file_path)
        result = RunConfig(**data)
        result.file_path = file_path
        logger.info(f"Loaded run config from '{file_path}'")

        system_prompt_path = evals_dir.prompts / f"{result.prompt_ref}.txt"
        if system_prompt_path.exists():
            result.prompt = system_prompt_path.read_text()
            logger.info(f"Loaded system prompt from '{system_prompt_path}'")
        else:
            logger.warning(f"System prompt file not found: {system_prompt_path}")

        query_path = evals_dir.queries / f"{result.query_ref}.yaml"
        try:
            query = load_yaml(query_path)
        except FileNotFoundError:
            raise ValueError(f"Query file not found: {query_path}")
        if "input" not in query:
            raise ValueError(f"Query file must contain a 'input' key: {query_path}")
        if "output" not in query:
            raise ValueError(f"Query file must contain a 'output' key: {query_path}")
        logger.info(f"Loaded query from '{query_path}'")
        result.input = query["input"]
        result.output = query["output"]
        logger.info(f"Loaded run config from '{query_path}'")

        return result

    def save(self, file_path: str):
        save_config = copy.deepcopy(self.model_dump())
        del save_config["input"]
        del save_config["output"]
        del save_config["prompt"]
        save_yaml(save_config, file_path)
        logger.info(f"Saved test config to '{file_path}'")


class RunResult(BaseModel):
    name: str
    values: List[Optional[float]] = Field(default_factory=list)
    average: Optional[float] = None
    standard_deviation: Optional[float] = None
