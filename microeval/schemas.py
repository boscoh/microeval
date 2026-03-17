import copy
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

from path import Path
from pydantic import BaseModel, Field

from microeval.llm import LLMService
from microeval.utils import load_yaml, save_yaml

logger = logging.getLogger(__name__)


# Directory and File Management
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
        if base_dir is not None:
            os.environ["EVALS_DIR"] = base_dir

        base_dir = os.getenv("EVALS_DIR", "evals")
        self._base = Path(base_dir)
        for d in [self.prompts, self.queries, self.results, self.runs]:
            d.makedirs_p()
        logger.info(f"Evals directory set to: {self._base}")


evals_dir = EvalsDir()


# Evaluator Domain
class EvaluatorConfig(BaseModel):
    """Configuration for an evaluator with optional parameters.

    Example::

        {
          "name": "word_count",
          "params": {
            "min_words": 50,
            "max_words": 200
          }
        }
    """

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    score: float = 0.5
    reasoning: str = ""
    elapsed_ms: int = 0
    token_count: int = 0


# Run Domain
class RunConfig(BaseModel):
    """Configuration for a single evaluation run.
    
    Supports eval.yaml global config with field names:
    - eval_chat_service / eval_chat_model (for LLM-based evaluators)
    - eval_embed_service / eval_embed_model (for embedding-based evaluators)
    """

    file_path: Optional[str] = None
    query_ref: Optional[str] = None
    prompt_ref: Optional[str] = None
    prompt: str = ""
    input: str = ""
    output: str = ""
    chat_service: LLMService
    model: str = ""
    repeat: int = 1
    temperature: float = 0.0
    evaluators: List[Union[str, EvaluatorConfig, Dict[str, Any]]] = Field(
        default_factory=lambda: ["equivalence"]
    )
    eval_chat_service: Optional[LLMService] = None
    eval_chat_model: Optional[str] = None
    eval_embed_service: Optional[LLMService] = None
    eval_embed_model: Optional[str] = None

    @staticmethod
    def read_from_yaml(file_path: str) -> "RunConfig":
        """Load RunConfig from YAML file with support for eval.yaml global config.

        The eval.yaml file uses: eval_chat_service, eval_chat_model,
        eval_embed_service, eval_embed_model. Env vars EVAL_CHAT_SERVICE,
        EVAL_CHAT_MODEL, EVAL_EMBED_SERVICE, EVAL_EMBED_MODEL override.
        CamelCase keys (evalChatService, etc.) are normalized to snake_case.
        """
        data = load_yaml(file_path)

        _alias_to_eval_key = {
            "evalChatService": "eval_chat_service",
            "evalChatModel": "eval_chat_model",
            "evalEmbedService": "eval_embed_service",
            "evalEmbedModel": "eval_embed_model",
        }
        for alias, key in _alias_to_eval_key.items():
            if alias in data and key not in data:
                data[key] = data.pop(alias)

        global_config = {}
        global_config_path = evals_dir._base / "eval.yaml"
        if global_config_path.exists():
            try:
                global_config = load_yaml(str(global_config_path))
                logger.info(f"Loaded global config from '{global_config_path}'")
            except Exception as e:
                logger.warning(
                    f"Failed to load global config from '{global_config_path}': {e}"
                )

        eval_keys = (
            "eval_chat_service",
            "eval_chat_model",
            "eval_embed_service",
            "eval_embed_model",
        )
        env_key_mapping = {
            "eval_chat_service": "EVAL_CHAT_SERVICE",
            "eval_chat_model": "EVAL_CHAT_MODEL",
            "eval_embed_service": "EVAL_EMBED_SERVICE",
            "eval_embed_model": "EVAL_EMBED_MODEL",
        }
        for key in eval_keys:
            env_key = env_key_mapping[key]
            if env_key in os.environ:
                data[key] = os.environ[env_key]
            elif (key not in data or data.get(key) is None) and key in global_config:
                data[key] = global_config[key]

        if "service" in data and "chat_service" not in data:
            data["chat_service"] = data.pop("service")
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
        if "service" in save_config:
            save_config["chat_service"] = save_config.pop("service")

        # Remove None values so defaults can be resolved when reading back
        save_config = {k: v for k, v in save_config.items() if v is not None}

        save_yaml(save_config, file_path)
        logger.info(f"Saved test config to '{file_path}'")


class RunResult(BaseModel):
    name: str
    values: List[Optional[float]] = Field(default_factory=list)
    average: Optional[float] = None
    standard_deviation: Optional[float] = None
