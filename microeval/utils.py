import json
import logging
import re
from typing import Any, Dict, Optional

import yaml
from path import Path

logger = logging.getLogger(__name__)

_MAX_LEN_LINE = 80


# --- YAML ---

def _folded_str_yaml_representer(dumper, data):
    if isinstance(data, str):
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        if len(data) > 60:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="")


yaml.add_representer(str, _folded_str_yaml_representer)


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, file_path: str):
    """Write a Python object to a YAML file with block-style for long strings."""
    parent = Path(file_path).parent
    if parent:
        parent.makedirs_p()
    with open(file_path, "w") as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=_MAX_LEN_LINE,
            indent=2,
            default_style=None,
            explicit_start=True,
        )


# --- JSON ---

def snap_score(value: float, valid_scores: tuple = (0.0, 0.5, 1.0)) -> float:
    return min(valid_scores, key=lambda s: abs(s - value))


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from text, handling markdown code blocks and embedded JSON.

    :param text: Text that may contain JSON (e.g., "{\"score\": 0.5}" or "```json\n{\"score\": 0.5}\n```").
    :return: Parsed dictionary if found, None otherwise. Example::

            {
              "score": 0.5,
              "reasoning": "Partial match"
            }
    """
    if not text:
        return None

    def try_parse(s):
        try:
            result = json.loads(s)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return None

    if parsed := try_parse(text.strip()):
        return parsed

    patterns = [
        r"```(?:json|python)?\s*([\s\S]*?)\s*```",
        r"\{[\s\S]*\}",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            if parsed := try_parse(match):
                return parsed

    return None
