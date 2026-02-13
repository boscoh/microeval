import json
import logging
import os
import time
import webbrowser
from contextlib import asynccontextmanager
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from path import Path
from pydantic import BaseModel

from microeval.evaluator import get_available_evaluators
from microeval.llm import load_config
from microeval.logger import setup_logging
from microeval.runner import Runner
from microeval.schemas import RunConfig, TableType, evals_dir, ext_from_table
from microeval.yamlx import load_yaml, save_yaml

logger = logging.getLogger(__name__)

setup_logging()

config = load_config()
chat_models = config["chat_models"]


async def get_json_from_request(request) -> Dict[str, Any]:
    """Returns parsed json from request"""
    return (
        await request.json()
        if hasattr(request, "json")
        else json.loads(await request.body())
    )


def read_content(file_path: str):
    """Returns a JSON object for ext='.yaml', or a string if ext='.txt" """
    file_path = Path(file_path)
    ext = file_path.suffix
    if ext == ".yaml":
        content = load_yaml(file_path)
    else:
        content = file_path.read_text(encoding="utf-8")
    return content


def save_content(content, file_path):
    if file_path.parent:
        file_path.parent.makedirs_p()
    ext = file_path.suffix
    if ext == ".yaml":
        save_yaml(content, file_path)
    else:
        file_path.write_text(content)


@asynccontextmanager
async def lifespan(app: FastAPI):
    base_dir = os.getenv("EVALS_DIR", "evals")
    evals_dir.set_base(base_dir)
    logger.info(f"Server initialized with evals_dir: {base_dir}")
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if not any(
        ext in str(request.url) for ext in [".js", ".css", ".ico", ".png", ".jpg"]
    ):
        logger.info(f"{request.method} {request.url.path}")
    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serves the main index page (index.html)"""
    index_path = Path(__file__).parent / "index.html"
    logger.info(f"Serving index page from: {index_path}")
    try:
        html_content = index_path.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)
    except Exception as ex:
        logger.error(f"Error serving index.html: {ex}")
        raise HTTPException(status_code=500, detail=f"Error serving index.html: {ex}")


@app.get("/api/graph-data")
def get_graph_data():
    """Dynamically generates graph data from current results directory"""
    try:
        from microeval.graph import extract_evaluation_data, generate_plotly_graph

        logger.info(f"Generating dynamic graph data from {evals_dir.results}")

        results_dir = evals_dir.results
        evaluators_data = extract_evaluation_data(results_dir)

        if not evaluators_data:
            return {"graphs": [], "evaluationData": {}}

        eval_type = "Evaluation"
        if "consultant" in str(results_dir).lower():
            eval_type = "Consultant"
        elif "engineer" in str(results_dir).lower():
            eval_type = "Engineer"

        evaluator_labels = {
            "elapsed_seconds": "Elapsed Time (seconds)",
            "token_count": "Token Count",
            "cost": "Cost ($)",
            "coherence": "Coherence Score (0-1)",
            "equivalence": "Equivalence Score (0-1)",
            "word_count": "Word Count Score (0-1)",
        }

        graphs = []
        for eval_name, data in evaluators_data.items():
            if not data:
                continue

            graph_id = f"{eval_type.lower()}-{eval_name.replace('_', '-')}-graph"
            eval_display_name = eval_name.replace("_", " ").title()
            x_axis_label = evaluator_labels.get(eval_name, eval_display_name)

            graph = generate_plotly_graph(data, graph_id, x_axis_label)
            graphs.append(graph)

        return {"graphs": graphs, "evaluationData": evaluators_data}
    except Exception as ex:
        logger.error(f"Error generating graph data: {ex}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating graph data: {str(ex)}"
        )


@app.get("/defaults")
def get_defaults():
    """
    Response: { "content": object }
    """
    default_service = "openai"
    default_model = chat_models[default_service][0] if chat_models[default_service] else ""

    return {
        "content": {
            "evals_dir": evals_dir.name,
            "evaluators": get_available_evaluators(),
            "run_config": {
                "promptRef": "",
                "queryRef": "",
                "prompt": "",
                "input": "",
                "output": "",
                "service": default_service,
                "model": default_model,
                "repeat": 1,
                "temperature": 0.2,
                "evaluators": ["coherence"],
            },
            "services": list(chat_models.keys()),
            "models": chat_models,
        }
    }


class ContentResponse(BaseModel):
    content: Any


@app.get("/list/{table}")
async def list_objects(table):
    """Response: { "content": ["string"] }"""
    try:
        table_dir = evals_dir.get_dir(table)
        ext = ext_from_table[table]
        logger.info(f"Request for names in: {table_dir}")
        basenames = [f.stem for f in table_dir.iterdir() if f.suffix == ext]
        logger.info(f"Found: {len(basenames)} names in {table_dir} with {ext}")
        return ContentResponse(content=basenames)
    except Exception as ex:
        logger.error(f"Error listing basenames: {ex}")
        raise HTTPException(status_code=500, detail=f"Error listing basenames: {ex}")


class FetchObjectRequest(BaseModel):
    table: TableType
    basename: str


@app.post("/fetch", response_model=ContentResponse)
async def fetch_object(request: FetchObjectRequest):
    try:
        logger.info(f"Request to fetch {request.table}/{request.basename}")
        table_dir = evals_dir.get_dir(request.table)
        ext = ext_from_table[request.table]
        f = Path(request.basename)
        file_path = (table_dir / f) + ext
        content = read_content(file_path)
        logger.info(f"Read content from '{file_path}'")
        return ContentResponse(content=content)

    except KeyError as ke:
        error_msg = f"Invalid table or basename: {ke}"
        logger.error(error_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
    except FileNotFoundError as fnf:
        error_msg = f"File not found: {fnf}"
        logger.error(error_msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)
    except Exception as ex:
        error_msg = f"Error reading object: {ex}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )


class SaveObjectRequest(BaseModel):
    table: TableType
    basename: str
    content: Any


class MessageResponse(BaseModel):
    message: str


@app.post("/save", response_model=MessageResponse)
async def save_object(request: SaveObjectRequest):
    try:
        logger.info(f"Request to save {request.table}/{request.basename}")
        table = request.table
        table_dir = evals_dir.get_dir(table)
        ext = ext_from_table[table]
        file_path = table_dir / f"{request.basename}{ext}"
        save_content(request.content, file_path)
        logger.info(f"Successfully saved to '{file_path}'")
        return MessageResponse(
            message=f"Successfully saved {request.table}/{request.basename}"
        )

    except Exception as ex:
        error_msg = f"Error saving object: {ex}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )


class EvaluateRequest(BaseModel):
    basename: str
    content: Any


@app.post("/evaluate", response_model=MessageResponse)
async def evaluate(request: EvaluateRequest):
    try:
        basename = request.basename
        config = request.content
        config_path = evals_dir.runs / f"{basename}.yaml"
        logger.info(f"Running evaluation of {basename}")
        run_config = RunConfig(**config)
        run_config.save(config_path)
        await Runner(config_path).run()
        return MessageResponse(message=f"Successfully evaluated {request.basename}")

    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error during evaluation: {str(e)}"
        )


class DeleteRequest(BaseModel):
    table: TableType
    basename: str


@app.post("/delete", response_model=MessageResponse)
async def delete_object(request: DeleteRequest):
    try:
        logger.info(f"Request to delete {request.table}/{request.basename}")
        table_dir = evals_dir.get_dir(request.table)
        ext = ext_from_table[request.table]
        file_path = table_dir / f"{request.basename}{ext}"

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_path.remove()
        logger.info(f"Successfully deleted '{file_path}'")
        return MessageResponse(
            message=f"Successfully deleted {request.table}/{request.basename}"
        )

    except KeyError as ke:
        error_msg = f"Invalid table or basename: {ke}"
        logger.error(error_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
    except FileNotFoundError as fnf:
        error_msg = f"File not found: {fnf}"
        logger.error(error_msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)
    except Exception as ex:
        error_msg = f"Error deleting object: {ex}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )


class RenameRequest(BaseModel):
    table: TableType
    basename: str
    newBasename: str


@app.post("/rename", response_model=MessageResponse)
async def rename_object(request: RenameRequest):
    try:
        logger.info(
            f"Request to rename {request.table}/{request.basename} to {request.newBasename}"
        )
        table_dir = evals_dir.get_dir(request.table)
        ext = ext_from_table[request.table]
        old_path = table_dir / f"{request.basename}{ext}"
        new_path = table_dir / f"{request.newBasename}{ext}"

        if not old_path.exists():
            raise FileNotFoundError(f"File not found: {old_path}")

        if new_path.exists():
            raise FileExistsError(f"File already exists: {new_path}")

        old_path.rename(new_path)
        logger.info(f"Successfully renamed '{old_path}' to '{new_path}'")
        return MessageResponse(
            message=f"Successfully renamed {request.table}/{request.basename} to {request.newBasename}"
        )

    except KeyError as ke:
        error_msg = f"Invalid table or basename: {ke}"
        logger.error(error_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
    except FileNotFoundError as fnf:
        error_msg = f"File not found: {fnf}"
        logger.error(error_msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)
    except FileExistsError as fee:
        error_msg = f"File already exists: {fee}"
        logger.error(error_msg)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error_msg)
    except Exception as ex:
        error_msg = f"Error renaming object: {ex}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )


def is_in_container() -> bool:
    """Check if running inside a container (Docker, Podman, Kubernetes, ECS, etc.)."""
    if os.path.exists("/.dockerenv"):
        return True
    if os.path.exists("/run/.containerenv"):
        return True
    container_indicators = [
        "docker",
        "containerd",
        "kubepods",
        "crio",
        "libpod",
        "ecs",
    ]
    for cgroup_file in ["/proc/1/cgroup", "/proc/self/cgroup"]:
        if os.path.exists(cgroup_file):
            try:
                with open(cgroup_file, "r") as f:
                    content = f.read()
                    if any(indicator in content for indicator in container_indicators):
                        return True
            except (OSError, IOError):
                pass
    return False


def poll_and_open_browser(
    port: int, timeout_seconds: int = 300, interval_seconds: int = 1
) -> None:
    start_time = time.time()
    url = f"http://localhost:{port}"

    while time.time() - start_time < timeout_seconds:
        try:
            response = httpx.get(url, timeout=2)
            if response.status_code == 200:
                logger.info(f"Server is live at {url}, opening browser...")
                webbrowser.open(url)
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass

        time.sleep(interval_seconds)

    logger.warning(f"Server did not respond within {timeout_seconds} seconds")
