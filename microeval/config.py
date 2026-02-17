import logging

from dotenv import load_dotenv
from path import Path

logger = logging.getLogger(__name__)


def load_env() -> bool:
    """Load environment variables from .env file (idempotent).

    Searches for .env file in:
    1. Current working directory
    2. Module parent directory (chatboti package location)

    Only loads once per process unless force=True is specified.
    Logs the location of the .env file when found.

    :return: True if .env file was found and loaded (or already loaded)
    """
    global _env_loaded

    # Skip if already loaded (unless forced)
    if _env_loaded:
        return True

    # Try current working directory first
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        logger.info(f"Loading .env from: {cwd_env}")
        load_dotenv(cwd_env, verbose=True)
        _env_loaded = True
        return True

    # Try module parent directory
    module_dir = Path(__file__).parent.parent
    module_env = module_dir / ".env"
    if module_env.exists():
        logger.info(f"Loading .env from: {module_env}")
        load_dotenv(module_env, verbose=True)
        _env_loaded = True
        return True

    return False


_env_loaded = False
