import logging
from functools import lru_cache

import pydash
from dotenv import load_dotenv
from path import Path

from microeval.llm import SimpleLLMClient, get_llm_client, load_models_config

logger = logging.getLogger(__name__)

_env_loaded = False


def load_env() -> bool:
    """Load environment variables from .env file (idempotent).

    Searches for .env file in:
    1. Current working directory
    2. Module parent directory

    Only loads once per process unless force=True is specified.
    Logs the location of the .env file when found.

    :return: True if .env file was found and loaded (or already loaded)
    """
    global _env_loaded

    if _env_loaded:
        return True

    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        logger.info(f"Loading .env from: {cwd_env}")
        load_dotenv(cwd_env, verbose=True)
        _env_loaded = True
        return True

    module_dir = Path(__file__).parent.parent
    module_env = module_dir / ".env"
    if module_env.exists():
        logger.info(f"Loading .env from: {module_env}")
        load_dotenv(module_env, verbose=True)
        _env_loaded = True
        return True

    return False


@lru_cache()
def _get_models_for_service(service: str, model_type: str) -> tuple:
    """Return the list of model IDs configured for a service in models.yaml."""
    config = load_models_config()
    models = pydash.get(config, f"{model_type}.{service}", [])
    if isinstance(models, list):
        return tuple(models)
    return (models,) if models else ()


@lru_cache()
def _get_default_model(service: str, model_type: str) -> str:
    """Get the default model for a service.

    :param service: Service name
    :param model_type: 'chat_models' or 'embed_models'
    :return: Default model name or empty string
    """
    models = _get_models_for_service(service, model_type)
    return models[0] if models else ""  # type: ignore[return-value]


@lru_cache(maxsize=128)
def _get_cached_llm_client_sync(service: str, model: str) -> SimpleLLMClient:
    """Get cached LLM client instance by service and model (idempotent, sync).

    Returns unconnected client instance.

    :param service: Service name (openai, ollama, bedrock, groq)
    :param model: Model name or "default"
    :return: Cached SimpleLLMClient instance (not connected)
    """
    kwargs = {}
    if model != "default":
        kwargs["model"] = model
    return get_llm_client(service, **kwargs)


async def _get_connected_llm_client(service: str, model: str) -> SimpleLLMClient:
    """Get cached and connected LLM client by service and model (idempotent).

    :param service: Service name (openai, ollama, bedrock, groq)
    :param model: Model name or "default"
    :return: Cached and connected SimpleLLMClient instance
    """
    client = _get_cached_llm_client_sync(service, model)
    await client.connect()
    return client
