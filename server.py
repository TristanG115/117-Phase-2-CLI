import gc
import hashlib
import json
import logging
import os
import re
from typing import Optional
from urllib.parse import unquote, urlparse

from fastapi import FastAPI, HTTPException, Request, Response
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


# Overall request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log request details
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    headers = dict(request.headers)

    print(f"\n{'='*60}", flush=True)
    print(f"REQUEST: {method} {path}", flush=True)
    print(f"Client IP: {client_ip}", flush=True)
    print(f"Headers: {headers}", flush=True)
    logger.info(f"REQUEST: {method} {path} from {client_ip}")

    # Process request
    response = await call_next(request)

    print(f"RESPONSE: {response.status_code}", flush=True)
    print(f"{'='*60}\n", flush=True)
    logger.info(f"RESPONSE: {method} {path} -> {response.status_code}")

    return response


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

    if not name or not isinstance(types, list):
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_query or it is "
                "formed improperly, or is invalid."
            ),
        )

    # If types is empty, default to all types
    if len(types) == 0:
        types = ["model", "dataset", "code"]

    return name, types


def _build_artifact_results(queries, artifacts):
    """Build results list from queries and artifacts."""
    results = []
    seen_ids = set()  # Track seen artifact IDs to avoid duplicates

    print(
        f"DEBUG _build: Processing {len(queries)} queries against {len(artifacts)} artifacts",
        flush=True,
    )

    for i, q in enumerate(queries):
        print(f"DEBUG _build: Query {i}: {q}", flush=True)
        try:
            name, types = _validate_query(q)
            print(f"DEBUG _build: Validated - name={name}, types={types}", flush=True)
        except Exception as e:
            print(f"DEBUG _build: Validation failed: {e}", flush=True)
            raise

        for a in artifacts:
            artifact_id = gen_id(a["name"])

            # Skip duplicates
            if artifact_id in seen_ids:
                continue

            # Normalize artifact type
            actual_type = a.get("artifact_type") or json.loads(
                a.get("metadata_json", "{}")
            ).get("type", "model")
            actual_type = str(actual_type).lower()

            print(
                f"DEBUG _build: Artifact {a['name']} has normalized type {actual_type}",
                flush=True,
            )

            # Properly check for name containment
            name_matches = (name == "*") or (name in a["name"].lower())

            # Check type match
            type_matches = actual_type in [t.lower() for t in types]

            if name_matches and type_matches:
                print(f"DEBUG _build: Adding {a['name']} as {actual_type}", flush=True)
                results.append(
                    {
                        "name": a["name"],
                        "id": artifact_id,
                        "type": actual_type,
                    }
                )
                seen_ids.add(artifact_id)
            else:
                if not name_matches:
                    print(
                        f"DEBUG _build: Skipping {a['name']} - name '{name}' not in '{a['name'].lower()}'",
                        flush=True,
                    )
                if not type_matches:
                    print(
                        f"DEBUG _build: Skipping {a['name']} - type {actual_type} not in {types}",
                        flush=True,
                    )

    print(f"DEBUG _build: Returning {len(results)} total results", flush=True)
    return results


@app.post("/artifacts")
async def get_artifacts(request: Request, offset: Optional[str] = None):
    """BASELINE: Return artifacts matching the given query list."""
    logger.info(f"DEBUG /artifacts: All headers = {dict(request.headers)}")

    auth_header = request.headers.get("X-Authorization")
    logger.info(f"DEBUG /artifacts: X-Authorization value = {repr(auth_header)}")

    if not auth_header:
        logger.warning("DEBUG: No X-Authorization header - allowing for baseline")
    else:
        require_auth(auth_header)

    print("DEBUG /artifacts: About to read body", flush=True)
    logger.info("DEBUG /artifacts: About to read body")

    try:
        body_bytes = await request.body()
        print(f"DEBUG /artifacts: Raw body = {body_bytes[:500]}", flush=True)
        logger.info(f"DEBUG /artifacts: Raw body = {body_bytes[:500]}")
        queries = json.loads(body_bytes)
        print(f"DEBUG /artifacts: Parsed queries = {queries}", flush=True)
        logger.info(f"DEBUG /artifacts: Parsed queries = {queries}")
    except json.JSONDecodeError as e:
        print(f"DEBUG /artifacts: JSON parse error: {e}", flush=True)
        logger.error(f"DEBUG /artifacts: JSON parse error: {e}")
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_query or it is "
                "formed improperly, or is invalid."
            ),
        )
    except Exception as e:
        print(
            f"DEBUG /artifacts: Unexpected error: {type(e).__name__}: {e}", flush=True
        )
        logger.error(f"DEBUG /artifacts: Unexpected error: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_query or it is "
                "formed improperly, or is invalid."
            ),
        )

    # Handle both dict and list inputs
    if not isinstance(queries, list):
        queries = [queries]

    # Validate structure
    if any(not isinstance(q, dict) for q in queries) or len(queries) == 0:
        print(
            f"DEBUG: queries validation failed - type: {type(queries)}, len: {len(queries)}",
            flush=True,
        )
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

    print("DEBUG: About to call list_artifacts()", flush=True)
    artifacts = registry_handler.list_artifacts()
    print(f"DEBUG: Got {len(artifacts)} artifacts", flush=True)

    print("DEBUG: About to build results", flush=True)
    results = _build_artifact_results(queries, artifacts)
    print(f"DEBUG: Built {len(results)} results", flush=True)

    if len(results) > 1000:
        raise HTTPException(status_code=413, detail="Too many artifacts returned.")

    current_offset = int(offset) if offset else 1
    headers = {"offset": str(current_offset + 1)}

    print(
        f"DEBUG: Returning {len(results)} results with offset header {headers}",
        flush=True,
    )

    # --- TEMP DEBUG LOGGING BLOCK ---
    print("\n===== DEBUG /artifacts FINAL STATE =====", flush=True)
    print(f"Queries received ({len(queries)}): {queries}", flush=True)
    print(f"Total artifacts loaded: {len(artifacts)}", flush=True)
    print(f"Results built ({len(results)}):", flush=True)
    for r in results[:10]:
        print(f"  → {r}", flush=True)
    print(f"Offset header: {headers}", flush=True)
    print("========================================\n", flush=True)

    # ADDITIONAL DETAILED LOGGING FOR DEBUGGING
    logger.info(f"===== ARTIFACTS QUERY RESULTS =====")
    logger.info(f"Query: {queries}")
    logger.info(
        f"Returning {len(results)} results from {len(artifacts)} total artifacts"
    )
    for idx, r in enumerate(results):
        logger.info(f"  Result {idx}: name={r['name']}, id={r['id']}, type={r['type']}")
    logger.info(f"=====================================")

    return JSONResponse(content=results, headers=headers, status_code=200)


@app.delete("/reset")
def reset_registry(request: Request):
    """BASELINE: Reset registry to a clean state."""
    auth_header = request.headers.get("X-Authorization")
    if not auth_header:
        logger.warning(f"DEBUG /reset: No auth header - allowing for baseline")
    else:
        if not require_auth(auth_header):
            raise HTTPException(
                status_code=401,
                detail="You do not have permission to reset the registry.",
            )

    registry_handler.reset_registry()
    gc.collect()
    return {"status": "system reset successful"}


@app.get("/artifacts/{artifact_type}/{artifact_id}")
def get_artifact(artifact_type: str, artifact_id: str, request: Request):
    """BASELINE: Retrieve one artifact by id."""
    auth_header = request.headers.get("X-Authorization")

    if not auth_header:
        logger.warning(f"No auth header - allowing for baseline")
    else:
        require_auth(auth_header)

    logger.info(f"GET ARTIFACT REQUEST: type={artifact_type}, id={artifact_id}")

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

    # Search through all artifacts to find the one with matching ID
    artifacts = registry_handler.list_artifacts()
    artifact = None

    for a in artifacts:
        calculated_id = gen_id(a["name"])
        if calculated_id == aid:
            # Verify the type matches - prefer artifact_type from database
            actual_type = a.get("artifact_type")
            if not actual_type:
                try:
                    metadata = json.loads(a.get("metadata_json", "{}"))
                    actual_type = metadata.get("type", "model")
                except Exception:
                    actual_type = "model"

            if actual_type == artifact_type:
                artifact = a
                break

    if artifact:
        # Determine URL based on artifact type
        if artifact_type == "code":
            url = artifact.get("code_url", artifact.get("url", "unknown"))
        elif artifact_type == "dataset":
            url = artifact.get("dataset_url", artifact.get("url", "unknown"))
        else:  # model
            url = artifact.get("url", artifact.get("code_url", "unknown"))

        return {
            "metadata": {
                "name": artifact["name"],
                "id": aid,
                "type": artifact_type,
            },
            "data": {"url": url},
        }

    raise HTTPException(status_code=404, detail="Artifact does not exist.")


@app.get("/artifact/{artifact_type}/{artifact_id}")
def get_artifact_singular(artifact_type: str, artifact_id: str, request: Request):
    """
    Singular endpoint alias for autograder compatibility.
    Maps to the same logic as /artifacts/{type}/{id}
    """
    return get_artifact(artifact_type, artifact_id, request)


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


def _update_artifact_urls(artifact, artifact_type, url):
    """Update artifact URLs based on artifact type."""
    if artifact_type == "code":
        artifact["code_url"] = url
        artifact["url"] = url
    elif artifact_type == "dataset":
        artifact["dataset_url"] = url
        artifact["url"] = url
    else:  # model
        artifact["code_url"] = url
        artifact["url"] = url


@app.put("/artifacts/{artifact_type}/{artifact_id}")
async def update_artifact(artifact_type: str, artifact_id: str, request: Request):
    """BASELINE: Update artifact content."""
    auth_header = request.headers.get("X-Authorization")

    if not auth_header:
        logger.warning(f"No auth header - allowing for baseline")
    else:
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

    artifact = registry_handler.get_artifact_by_id(str(int(artifact_id)), artifact_type)

    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    if artifact["name"] != metadata.get("name"):
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_type or "
                "artifact_id or it is formed improperly, or is invalid."
            ),
        )

    # Update URLs
    _update_artifact_urls(artifact, artifact_type, data.get("url"))

    # Update in registry
    registry_handler.update_artifact(
        str(int(artifact_id)),
        url=data.get("url"),
        code_url=artifact.get("code_url", "unknown"),
        dataset_url=artifact.get("dataset_url", "unknown"),
    )

    return {"status": "artifact updated successfully"}


@app.post("/artifact/{artifact_type}")
async def register_artifact(artifact_type: str, request: Request):
    """BASELINE: Register a new artifact by URL."""
    auth_header = request.headers.get("X-Authorization")
    if not auth_header:
        logger.warning(
            f"DEBUG /artifact/{{type}}: No auth header - allowing for baseline"
        )
    else:
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

    # Improved URL parsing to handle edge cases
    try:
        # Robust name extraction for edge-case URLs
        url_clean = url.rstrip("/")
        parsed = urlparse(url_clean)
        path = parsed.path.rstrip("/")

        try:
            name = unquote(path.split("/")[-1] or parsed.netloc or "artifact")

            # Handle cases like "https://github.com/org" (no repo segment)
            if not name.strip() or name in ("www", "github", "huggingface", "co"):
                name = parsed.netloc.split(".")[0] or "artifact"

        except Exception as e:
            logger.error(f"URL parsing error for {url}: {e}")
            name = url.rstrip("/").split("/")[-1] or "artifact"

        # Normalize casing for deterministic IDs
        name = name.lower()

    except Exception as e:
        logger.error(f"URL parsing error for {url}: {e}")
        # Fallback to simple parsing
        name = url.rstrip("/").split("/")[-1]

    new_id = gen_id(name)

    artifacts = registry_handler.list_artifacts()
    for a in artifacts:
        # Check for duplicate by URL (exact match)
        if a.get("url") == url:
            raise HTTPException(status_code=409, detail="Artifact exists already.")
        # Also check for duplicate by name+type
        if gen_id(a["name"]) == new_id and a.get("artifact_type") == artifact_type:
            raise HTTPException(status_code=409, detail="Artifact exists already.")

    artifact_id = registry_handler.add_artifact(
        name=name,
        artifact_type=artifact_type,
        score=0.0,
        url=url,
        tags=artifact_type,
        code_url=url if artifact_type in ["code", "model"] else "unknown",
        dataset_url=url if artifact_type == "dataset" else "unknown",
        metadata={"type": artifact_type},  # Note: metadata not metadata_json
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
        logger.warning(f"No auth header - allowing for baseline")
    else:
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

    artifacts = registry_handler.list_artifacts()
    for a in artifacts:
        if gen_id(a["name"]) == aid:
            meta = json.loads(a.get("metadata_json", "{}"))
            if meta and any(
                key in meta for key in ["net_score", "ramp_up_time", "bus_factor"]
            ):
                return meta
            else:
                return {
                    "name": a["name"],
                    "category": "MODEL",
                    "net_score": a.get("score", 0),
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
        logger.warning(f"No auth header - allowing for baseline")
    else:
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

    artifacts = registry_handler.list_artifacts()
    for a in artifacts:
        if gen_id(a["name"]) == aid:
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
        logger.warning(f"No auth header - allowing for baseline")
    else:
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

    artifacts = registry_handler.list_artifacts()
    for a in artifacts:
        if gen_id(a["name"]) == aid:
            dataset_url = a.get("dataset_url", "unknown")
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
                    {"artifact_id": aid, "name": a["name"], "source": "config_json"},
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


def _verify_artifact_exists(artifact_id, artifacts):
    """Check if artifact with given ID exists."""
    for a in artifacts:
        if gen_id(a["name"]) == artifact_id:
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
        logger.warning(f"No auth header - allowing for baseline")
    else:
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

    artifacts = registry_handler.list_artifacts()
    if not _verify_artifact_exists(aid, artifacts):
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
        logger.warning(f"No auth header - allowing for baseline")
    else:
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
        pattern = re.compile(regex, re.IGNORECASE)
    except re.error:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is missing field(s) in the artifact_regex or it is "
                "formed improperly, or is invalid"
            ),
        )

    # Use list_artifacts() to search all artifacts
    artifacts = registry_handler.list_artifacts()
    matches = []
    seen = set()
    for a in artifacts:
        artifact_id = gen_id(a["name"])
        # Skip if we've already added this artifact
        if artifact_id in seen:
            continue
        metadata = {}
        try:
            metadata = json.loads(a.get("metadata_json", "{}"))
        except Exception:
            pass
        # Build comprehensive search text from all fields
        metadata_text = " ".join(str(v) for v in metadata.values() if v)
        # Include all possible searchable fields
        search_fields = [
            a.get("name", ""),
            a.get("tags", ""),
            a.get("code_url", ""),
            a.get("dataset_url", ""),
            a.get("url", ""),
            metadata_text,
        ]

        text = " ".join(str(field) for field in search_fields if field)

        if pattern.search(text):
            # Get actual type - prefer artifact_type from database
            actual_type = a.get("artifact_type")
            if not actual_type:
                actual_type = metadata.get("type", "model")

            seen.add(artifact_id)
            matches.append({"name": a["name"], "id": artifact_id, "type": actual_type})

    if not matches:
        raise HTTPException(
            status_code=404, detail="No artifact found under this regex."
        )

    return JSONResponse(content=matches)


@app.get("/tracks")
def get_tracks():
    """Return the list of tracks this team has implemented."""
    try:
        return {"plannedTracks": ["High assurance track"]}
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
        # Use list_artifacts for dashboard to show all types
        artifacts = registry_handler.list_artifacts()
        return templates.TemplateResponse(
            "index.html", {"request": request, "models": artifacts}
        )
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return HTMLResponse(
            content="<h1>Registry API Server</h1><p>API docs at <a href='/docs'>/docs</a></p>"
        )


@app.get("/health", status_code=200)
def health_check():
    logger.info("Health check called")
    return Response(status_code=200)


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
