"""Generate graph data from evaluation results for visualization."""

import logging
from typing import Dict, List

import pydash
from path import Path

from microeval.utils import load_yaml

logger = logging.getLogger(__name__)


def parse_service_model_from_basename(basename: str) -> tuple[str, str]:
    """Extract service and model from result filename."""
    parts = basename.split("-")
    if len(parts) < 2:
        return "unknown", "unknown"

    service = parts[1] if len(parts) > 1 else "unknown"
    model = "-".join(parts[2:]) if len(parts) > 2 else "unknown"

    return service, model


def extract_evaluation_data(results_dir: Path) -> Dict[str, List[Dict]]:
    """Extract all evaluation metrics from result files, organized by evaluator type.

    :param results_dir: Directory containing result YAML files.
    :return: Dictionary mapping evaluator names to lists of evaluation data. Example::

            {
              "equivalence": [
                {
                  "basename": "summarize-gpt4o",
                  "service": "openai",
                  "model": "gpt-4o",
                  "average": 0.95,
                  "standardDeviation": 0.03
                }
              ],
              "relevance_llm": [
                {
                  "basename": "summarize-gpt4o",
                  "service": "openai",
                  "model": "gpt-4o",
                  "average": 0.92,
                  "standardDeviation": 0.05
                }
              ]
            }
    """
    evaluators_data = {}

    for result_file in results_dir.glob("*.yaml"):
        try:
            data = load_yaml(result_file)
            basename = result_file.stem
            service, model = parse_service_model_from_basename(basename)

            for evaluation in pydash.get(data, "evaluations", []):
                eval_name = pydash.get(evaluation, "name")
                average = pydash.get(evaluation, "average")

                if eval_name and average is not None:
                    if eval_name not in evaluators_data:
                        evaluators_data[eval_name] = []

                    evaluators_data[eval_name].append(
                        {
                            "basename": basename,
                            "service": service,
                            "model": model,
                            "average": average,
                            "standardDeviation": pydash.get(evaluation, "standard_deviation", 0.0),
                        }
                    )
        except Exception as e:
            logger.warning(f"Could not parse {result_file}: {e}")

    for eval_name in evaluators_data:
        evaluators_data[eval_name] = sorted(
            evaluators_data[eval_name], key=lambda x: x["basename"]
        )

    return evaluators_data


def get_color_for_basename(basename: str) -> str:
    """Get a consistent light color for each basename."""
    light_colors = [
        "#b3d9ff",  # light blue
        "#ffd9b3",  # light orange
        "#c2f0c2",  # light green
        "#ffb3d9",  # light pink
        "#d9b3ff",  # light purple
        "#b3fff0",  # light cyan
        "#fff0b3",  # light yellow
        "#f0b3c2",  # light rose
        "#d9e6b3",  # light lime
        "#b3d9e6",  # light sky
    ]

    hash_value = sum(ord(c) for c in basename)
    return light_colors[hash_value % len(light_colors)]


def generate_plotly_graph(
    performance_data: List[Dict], graph_id: str, x_axis_label: str
) -> Dict:
    """Generate Plotly graph configuration from performance data.

    :param performance_data: List of performance data dictionaries. Example::

            [
              {
                "basename": "summarize-gpt4o",
                "average": 0.95,
                "standardDeviation": 0.03
              }
            ]

    :param graph_id: Unique identifier for the graph (e.g., "equivalence-graph").
    :param x_axis_label: Label for the x-axis (e.g., "Equivalence Score").
    :return: Plotly graph configuration dictionary. Example::

            {
              "id": "equivalence-graph",
              "data": [{
                "y": ["summarize-gpt4o"],
                "x": [0.95],
                "error_x": {"array": [0.03], "visible": true},
                "type": "bar",
                "orientation": "h"
              }],
              "layout": {
                "xaxis": {"title": {"text": "Equivalence Score"}},
                "height": 200
              }
            }
    """
    y_labels = [d["basename"] for d in performance_data]
    x_values = [d["average"] for d in performance_data]
    error_values = [d["standardDeviation"] for d in performance_data]

    colors = [get_color_for_basename(d["basename"]) for d in performance_data]

    num_items = len(performance_data)
    bar_height_px = 28
    base_margin = 120
    height = max(200, num_items * bar_height_px + base_margin)

    max_value_with_error = max(
        x_values[i] + error_values[i] for i in range(len(x_values))
    )
    x_range_max = max_value_with_error * 1.15

    return {
        "id": graph_id,
        "data": [
            {
                "y": y_labels,
                "x": x_values,
                "error_x": {
                    "type": "data",
                    "array": error_values,
                    "visible": True,
                    "color": "#444",
                    "thickness": 1,
                    "width": 6,
                },
                "type": "bar",
                "orientation": "h",
                "width": 0.8,
                "marker": {"color": colors},
            }
        ],
        "layout": {
            "xaxis": {
                "title": {"text": x_axis_label, "font": {"size": 12, "weight": "bold"}},
                "range": [0, x_range_max],
            },
            "yaxis": {"title": "", "ticklen": 10, "tickcolor": "transparent"},
            "bargap": 0.1,
            "margin": {"t": 20, "b": 100, "l": 250, "r": 40},
            "height": height,
        },
    }
