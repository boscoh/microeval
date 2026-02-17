import importlib.metadata
import logging

logger = logging.getLogger(__name__)

try:
    __version__ = importlib.metadata.version("microeval")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
