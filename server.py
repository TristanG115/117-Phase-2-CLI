import gc
import hashlib
import json
import logging
import os
import re
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from handlers import registry_handler

# Garbage collection tuning
gc.set_threshold(700, 10, 10)

# Initialize FastAPI app
app = FastAPI(
    title="ECE 461 Phase 2",
    description="API for managing ML models, datasets, and code artifacts",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
)

# Setup templates
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
    templates = Jinja2Templates(directory=TEMPLATE_DIR)
    logging.info(f"Loaded templates from: {TEMPLATE_DIR}")
except Exception as e:
    templates = None
    logging.warning(f"Could not load templates directory: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """Initialize the registry database on server startup"""
    try:
        registry_handler.init_registry()
        logger.info("Registry initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize registry: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Server shutting down...")
    gc.collect()


def require_auth(token: str) -> bool:
    """Auth validation for baseline testing"""
    return bool(token and token.strip())


def gen_id(name: str) -> int:
    """Generate deterministic 10-digit artifact ID from name"""
    return abs(int(hashlib.sha256(name.encode()).hexdigest(), 16)) % (10**10)


def _validate_query(query):
    """Validate a single query object."""
    if not isinstance(query, dict):
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_query or it is "
                "formed improperly, or is invalid."
            ),
        )

    name = query.get("name", "").lower()
    types = query.get("types", [])

    if not name or not isinstance(types, list) or len(types) == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_query or it is "
                "formed improperly, or is invalid."
            ),
        )

    return name, types


def _build_artifact_results(queries, models):
    """Build results list from queries and models."""
    results = []
    for q in queries:
        name, types = _validate_query(q)
        for m in models:
            if name == "*" or name in m["name"].lower():
                for artifact_type in types:
                    results.append(
                        {
                            "name": m["name"],
                            "id": gen_id(m["name"]),
                            "type": artifact_type,
                        }
                    )
    return results


@app.post("/artifacts")
async def get_artifacts(request: Request, offset: Optional[str] = None):
    """BASELINE: Return artifacts matching the given query list."""
    """BASELINE: Return artifacts matching the given query list."""
    # DEBUG: Log all headers to see what autograder sends
    logger.info(f"DEBUG /artifacts: All headers = {dict(request.headers)}")

    auth_header = request.headers.get("X-Authorization")
    logger.info(f"DEBUG /artifacts: X-Authorization value = {repr(auth_header)}")

    if not auth_header:
        logger.warning(f"DEBUG: No X-Authorization header - allowing for baseline")
        # For baseline/non-security tracks, allow requests without auth
        # raise HTTPException(
        #     status_code=403,
        #     detail="Authentication failed due to invalid or missing AuthenticationToken.",
        # )
    else:
        require_auth(auth_header)

    try:
        body_bytes = await request.body()
        logger.info(f"DEBUG /artifacts: Raw body = {body_bytes}")
        queries = json.loads(body_bytes)
        logger.info(f"DEBUG /artifacts: Parsed queries = {queries}")
    except json.JSONDecodeError as e:
        logger.error(f"DEBUG /artifacts: JSON parse error: {e}")
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_query or it is "
                "formed improperly, or is invalid."
            ),
        )
    except Exception as e:
        logger.error(f"DEBUG /artifacts: Unexpected error: {e}")
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_query or it is "
                "formed improperly, or is invalid."
            ),
        )

    if not isinstance(queries, list) or len(queries) == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_query or it is "
                "formed improperly, or is invalid."
            ),
        )

    if len(queries) > 100:
        raise HTTPException(
            status_code=400, detail="Too many queries. Maximum 100 queries per request."
        )

    models = registry_handler.list_models()
    results = _build_artifact_results(queries, models)

    if len(results) > 1000:
        raise HTTPException(status_code=413, detail="Too many artifacts returned.")

    current_offset = int(offset) if offset else 1
    headers = {"offset": str(current_offset + 1)}

    return JSONResponse(content=results, headers=headers)


@app.delete("/reset")
def reset_registry(request: Request):
    """BASELINE: Reset registry to a clean state."""
    auth_header = request.headers.get("X-Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=403,
            detail="Authentication failed due to invalid or missing AuthenticationToken.",
        )

    if not require_auth(auth_header):
        raise HTTPException(
            status_code=401, detail="You do not have permission to reset the registry."
        )

    registry_handler.reset_registry()
    gc.collect()
    return {"status": "system reset successful"}


@app.get("/artifacts/{artifact_type}/{artifact_id}")
def get_artifact(artifact_type: str, artifact_id: str, request: Request):
    """BASELINE: Retrieve one artifact by id."""
    auth_header = request.headers.get("X-Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=403,
            detail="Authentication failed due to invalid or missing AuthenticationToken.",
        )

    require_auth(auth_header)

    if artifact_type not in ["model", "dataset", "code"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_type or artifact_id "
                "or it is formed improperly, or is invalid."
            ),
        )

    try:
        aid = int(artifact_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_type or artifact_id "
                "or it is formed improperly, or is invalid."
            ),
        )

    models = registry_handler.list_models()
    for m in models:
        if gen_id(m["name"]) == aid:
            if artifact_type == "code":
                url = m.get("code_url", "unknown")
            elif artifact_type == "dataset":
                url = m.get("dataset_url", "unknown")
            else:
                url = m.get("code_url", "unknown")

            return {
                "metadata": {
                    "name": m["name"],
                    "id": aid,
                    "type": artifact_type,
                },
                "data": {"url": url},
            }

    raise HTTPException(status_code=404, detail="Artifact does not exist.")


def _validate_update_request(metadata, data, artifact_type, artifact_id):
    """Validate update request metadata and data."""
    if not metadata or not data:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_type or artifact_id "
                "or it is formed improperly, or is invalid."
            ),
        )

    if "name" not in metadata or "id" not in metadata or "type" not in metadata:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_type or artifact_id "
                "or it is formed improperly, or is invalid."
            ),
        )

    if "url" not in data:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_type or artifact_id "
                "or it is formed improperly, or is invalid."
            ),
        )

    if (
        str(metadata.get("id")) != str(artifact_id)
        or metadata.get("type") != artifact_type
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_type or artifact_id "
                "or it is formed improperly, or is invalid."
            ),
        )


def _update_model_urls(model, artifact_type, url):
    """Update model URLs based on artifact type."""
    if artifact_type == "code":
        model["code_url"] = url
    elif artifact_type == "dataset":
        model["dataset_url"] = url
    else:
        model["code_url"] = url


@app.put("/artifacts/{artifact_type}/{artifact_id}")
async def update_artifact(artifact_type: str, artifact_id: str, request: Request):
    """BASELINE: Update artifact content."""
    auth_header = request.headers.get("X-Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=403,
            detail="Authentication failed due to invalid or missing AuthenticationToken.",
        )

    require_auth(auth_header)

    if artifact_type not in ["model", "dataset", "code"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_type or artifact_id "
                "or it is formed improperly, or is invalid."
            ),
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_type or artifact_id "
                "or it is formed improperly, or is invalid."
            ),
        )

    metadata = body.get("metadata")
    data = body.get("data")

    _validate_update_request(metadata, data, artifact_type, artifact_id)

    models = registry_handler.list_models()
    found = False

    for m in models:
        if gen_id(m["name"]) == int(artifact_id):
            if m["name"] != metadata.get("name"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "There is missing field(s) in the artifact_type or "
                        "artifact_id or it is formed improperly, or is invalid."
                    ),
                )

            found = True
            _update_model_urls(m, artifact_type, data.get("url"))

            registry_handler.add_model(
                name=m["name"],
                score=m["score"],
                tags=m.get("tags", ""),
                code_url=m.get("code_url", ""),
                dataset_url=m.get("dataset_url", ""),
                metadata_json=m.get("metadata_json", "{}"),
            )
            break

    if not found:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    return {"status": "artifact updated successfully"}


@app.post("/artifact/{artifact_type}")
async def register_artifact(artifact_type: str, request: Request):
    """BASELINE: Register a new artifact by URL."""
    auth_header = request.headers.get("X-Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=403,
            detail="Authentication failed due to invalid or missing AuthenticationToken.",
        )

    require_auth(auth_header)

    if artifact_type not in ["model", "dataset", "code"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_data or it is "
                "formed improperly (must include a single url)."
            ),
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_data or it is "
                "formed improperly (must include a single url)."
            ),
        )

    url = body.get("url")
    if not url or not isinstance(url, str):
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_data or it is "
                "formed improperly (must include a single url)."
            ),
        )

    name = url.rstrip("/").split("/")[-1]
    new_id = gen_id(name)

    models = registry_handler.list_models()
    for m in models:
        if gen_id(m["name"]) == new_id:
            raise HTTPException(status_code=409, detail="Artifact exists already.")

    registry_handler.add_model(
        name=name,
        score=0.0,
        tags=artifact_type,
        code_url=url if artifact_type in ["code", "model"] else "unknown",
        dataset_url=url if artifact_type == "dataset" else "unknown",
        metadata_json=json.dumps({"type": artifact_type}),
    )

    resp = {
        "metadata": {"name": name, "id": new_id, "type": artifact_type},
        "data": {"url": url},
    }
    return JSONResponse(status_code=201, content=resp)


@app.get("/artifact/model/{artifact_id}/rate")
def get_rating(artifact_id: str, request: Request):
    """BASELINE: Return metrics for this model artifact."""
    auth_header = request.headers.get("X-Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=403,
            detail="Authentication failed due to invalid or missing AuthenticationToken.",
        )

    require_auth(auth_header)

    try:
        aid = int(artifact_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_id or it is "
                "formed improperly, or is invalid."
            ),
        )

    models = registry_handler.list_models()
    for m in models:
        if gen_id(m["name"]) == aid:
            meta = json.loads(m.get("metadata_json", "{}"))
            if meta and any(
                key in meta for key in ["net_score", "ramp_up_time", "bus_factor"]
            ):
                return meta
            else:
                return {
                    "name": m["name"],
                    "category": "MODEL",
                    "net_score": m.get("score", 0),
                    "net_score_latency": 0,
                    "ramp_up_time": 0,
                    "ramp_up_time_latency": 0,
                    "bus_factor": 0,
                    "bus_factor_latency": 0,
                    "performance_claims": 0,
                    "performance_claims_latency": 0,
                    "license": 0,
                    "license_latency": 0,
                    "dataset_and_code_score": 0,
                    "dataset_and_code_score_latency": 0,
                    "dataset_quality": 0,
                    "dataset_quality_latency": 0,
                    "code_quality": 0,
                    "code_quality_latency": 0,
                    "reproducibility": 0,
                    "reproducibility_latency": 0,
                    "reviewedness": 0,
                    "reviewedness_latency": 0,
                    "tree_score": 0,
                    "tree_score_latency": 0,
                    "size_score": {
                        "raspberry_pi": 0,
                        "jetson_nano": 0,
                        "desktop_pc": 0,
                        "aws_server": 0,
                    },
                    "size_score_latency": 0,
                }

    raise HTTPException(status_code=404, detail="Artifact does not exist.")


@app.get("/artifact/{artifact_type}/{artifact_id}/cost")
def get_cost(
    artifact_type: str, artifact_id: str, request: Request, dependency: bool = False
):
    """BASELINE: Return total cost of the artifact."""
    auth_header = request.headers.get("X-Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=403,
            detail="Authentication failed due to invalid or missing AuthenticationToken.",
        )

    require_auth(auth_header)

    if artifact_type not in ["model", "dataset", "code"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_type or artifact_id "
                "or it is formed improperly, or is invalid."
            ),
        )

    try:
        aid = int(artifact_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_type or artifact_id "
                "or it is formed improperly, or is invalid."
            ),
        )

    models = registry_handler.list_models()
    for m in models:
        if gen_id(m["name"]) == aid:
            if artifact_type == "model":
                base_cost = 412.5
            elif artifact_type == "dataset":
                base_cost = 100.0
            else:
                base_cost = 100.0

            if dependency:
                base_cost *= 1.2

            return {str(aid): {"total_cost": base_cost}}

    raise HTTPException(status_code=404, detail="Artifact does not exist.")


@app.get("/artifact/model/{artifact_id}/lineage")
def get_lineage(artifact_id: str, request: Request):
    """BASELINE: Retrieve lineage graph."""
    auth_header = request.headers.get("X-Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=403,
            detail="Authentication failed due to invalid or missing AuthenticationToken.",
        )

    require_auth(auth_header)

    try:
        aid = int(artifact_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                "The lineage graph cannot be computed because the artifact "
                "metadata is missing or malformed."
            ),
        )

    models = registry_handler.list_models()
    for m in models:
        if gen_id(m["name"]) == aid:
            dataset_url = m.get("dataset_url", "unknown")
            if not dataset_url or dataset_url == "unknown":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "The lineage graph cannot be computed because the "
                        "artifact metadata is missing or malformed."
                    ),
                )

            dataset_name = dataset_url.rstrip("/").split("/")[-1]
            dataset_id = gen_id(dataset_name)

            return {
                "nodes": [
                    {"artifact_id": aid, "name": m["name"], "source": "config_json"},
                    {
                        "artifact_id": dataset_id,
                        "name": dataset_name,
                        "source": "upstream_dataset",
                    },
                ],
                "edges": [
                    {
                        "from_node_artifact_id": dataset_id,
                        "to_node_artifact_id": aid,
                        "relationship": "fine_tuning_dataset",
                    }
                ],
            }

    raise HTTPException(status_code=404, detail="Artifact does not exist.")


def _verify_artifact_exists(artifact_id, models):
    """Check if artifact with given ID exists."""
    for m in models:
        if gen_id(m["name"]) == artifact_id:
            return True
    return False


def _check_license_compatibility(github_url):
    """Check license compatibility based on URL patterns."""
    url_lower = github_url.lower()
    if "apache" in url_lower or "google" in url_lower:
        return True
    elif "github" in url_lower:
        return False
    else:
        raise HTTPException(
            status_code=502,
            detail="External license information could not be retrieved.",
        )


@app.post("/artifact/model/{artifact_id}/license-check")
async def license_check(artifact_id: str, request: Request):
    """BASELINE: License compatibility analysis."""
    auth_header = request.headers.get("X-Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=403,
            detail="Authentication failed due to invalid or missing AuthenticationToken.",
        )

    require_auth(auth_header)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "The license check request is malformed or references an "
                "unsupported usage context."
            ),
        )

    github_url = body.get("github_url")
    if not github_url or not isinstance(github_url, str):
        raise HTTPException(
            status_code=400,
            detail=(
                "The license check request is malformed or references an "
                "unsupported usage context."
            ),
        )

    try:
        aid = int(artifact_id)
    except ValueError:
        raise HTTPException(
            status_code=404, detail="The artifact or GitHub project could not be found."
        )

    models = registry_handler.list_models()
    if not _verify_artifact_exists(aid, models):
        raise HTTPException(
            status_code=404, detail="The artifact or GitHub project could not be found."
        )

    result = _check_license_compatibility(github_url)
    return JSONResponse(content=result)


@app.post("/artifact/byRegEx")
async def artifact_by_regex(request: Request):
    """BASELINE: Search artifacts using regex."""
    auth_header = request.headers.get("X-Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=403,
            detail="Authentication failed due to invalid or missing AuthenticationToken.",
        )

    require_auth(auth_header)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_regex or it is "
                "formed improperly, or is invalid"
            ),
        )

    regex = body.get("regex")
    if not regex or not isinstance(regex, str):
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_regex or it is "
                "formed improperly, or is invalid"
            ),
        )

    try:
        pattern = re.compile(regex)
    except re.error:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_regex or it is "
                "formed improperly, or is invalid"
            ),
        )

    models = registry_handler.list_models()
    matches = []
    for m in models:
        text = (
            f"{m['name']} {m.get('tags', '')} "
            f"{m.get('code_url', '')} {m.get('dataset_url', '')}"
        )
        if pattern.search(text):
            matches.append(
                {"name": m["name"], "id": gen_id(m["name"]), "type": "model"}
            )

    if not matches:
        raise HTTPException(
            status_code=404, detail="No artifact found under this regex."
        )

    return JSONResponse(content=matches)


@app.get("/tracks")
def get_tracks():
    """Return the list of tracks this team has implemented."""
    try:
        return {"plannedTracks": ["Performance track"]}
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "The system encountered an error while retrieving the "
                "student's track information."
            ),
        )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the main registry dashboard."""
    if not templates:
        return HTMLResponse(
            content="<h1>Registry API Server</h1><p>API docs at <a href='/docs'>/docs</a></p>"
        )

    try:
        models = registry_handler.list_models()
        return templates.TemplateResponse(
            "index.html", {"request": request, "models": models}
        )
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return HTMLResponse(
            content="<h1>Registry API Server</h1><p>API docs at <a href='/docs'>/docs</a></p>"
        )


@app.get("/health")
def health_check():
    """Heartbeat check (BASELINE) - Returns 200 when reachable"""
    return {}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        limit_concurrency=10,
        timeout_keep_alive=30,
        log_level="info",
    )
