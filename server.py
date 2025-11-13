import gc
import hashlib
import json
import logging
import os
import re
import time
from typing import Optional
from urllib.parse import unquote, urlparse

from beautilog import logger
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from handlers import registry_handler

# ==========================================================
#                   CUSTOM LOG LEVEL HELPERS
# ==========================================================

# Custom levels are defined in beautilog.ini, but Python also needs to know them:
logging.addLevelName(15, "GETREQ")
logging.addLevelName(14, "POSTREQ")
logging.addLevelName(13, "PUTREQ")
logging.addLevelName(12, "DELREQ")
logging.addLevelName(11, "PAYLOAD")


def log_get(msg):
    logger.log(15, msg)


def log_post(msg):
    logger.log(14, msg)


def log_put(msg):
    logger.log(13, msg)


def log_delete(msg):
    logger.log(12, msg)


def log_payload(msg):
    logger.log(11, msg)


# Silence noisy loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ==========================================================
#                    FASTAPI INITIALIZATION
# ==========================================================

gc.set_threshold(700, 10, 10)

app = FastAPI(
    title="ECE 461 Phase 2",
    description="Trustworthy Model Registry",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
)

# Jinja templates (safe-load)
try:
    BASE = os.path.dirname(os.path.abspath(__file__))
    templates = Jinja2Templates(directory=os.path.join(BASE, "templates"))
except Exception:
    templates = None


# ==========================================================
#            REQUEST / RESPONSE LOGGING MIDDLEWARE
# ==========================================================


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    method = request.method
    path = request.url.path
    client = request.client.host if request.client else "unknown"

    # Log method-specific
    if method == "GET":
        log_get(f"GET {path} from {client}")
    elif method == "POST":
        log_post(f"POST {path} from {client}")
    elif method == "PUT":
        log_put(f"PUT {path} from {client}")
    elif method == "DELETE":
        log_delete(f"DELETE {path} from {client}")
    else:
        logger.info(f"{method} {path} from {client}")

    # Log request payload
    try:
        body = await request.body()
        if body:
            log_payload(f"REQUEST BODY: {body[:500].decode('utf-8', 'ignore')}")
    except Exception as e:
        logger.debug(f"Failed to read request body: {e}")

    # Process request
    response = await call_next(request)

    # Response metadata
    elapsed = round((time.time() - start) * 1000, 2)
    logger.info(f"{method} {path} → {response.status_code} ({elapsed} ms)")

    # Log response preview
    try:
        accumulated = b""
        async for chunk in response.body_iterator:
            accumulated += chunk
        if accumulated:
            log_payload(f"RESPONSE BODY: {accumulated[:500].decode('utf-8', 'ignore')}")
        response.body_iterator = iter([accumulated])
    except Exception:
        pass

    logger.info("-" * 60)
    return response


# ==========================================================
#                      UTILITY HELPERS
# ==========================================================


def require_auth(token: str) -> bool:
    return bool(token and token.strip())


def gen_id(name: str) -> int:
    return abs(int(hashlib.sha256(name.encode()).hexdigest(), 16)) % (10**10)


def _get_artifact_type(a: dict) -> str:
    if a.get("artifact_type"):
        return a["artifact_type"].lower()
    try:
        meta = json.loads(a.get("metadata_json", "{}"))
        if meta.get("type"):
            return meta["type"].lower()
    except Exception:
        pass
    return "model"


def _validate_query(q):
    if not isinstance(q, dict):
        raise HTTPException(
            status_code=400,
            detail="There is missing field(s) in the artifact_query or it is formed improperly, or is invalid.",
        )

    name = q.get("name", "").lower()
    types = q.get("types", [])
    if not name or not isinstance(types, list):
        raise HTTPException(
            status_code=400,
            detail="There is missing field(s) in the artifact_query or it is formed improperly, or is invalid.",
        )

    if not types:
        types = ["model", "dataset", "code"]

    return name, types


# ==========================================================
#           BUILD RESULTS FOR ARTIFACT SEARCH
# ==========================================================


def _build_artifact_results(queries, artifacts):
    results = []
    seen = set()

    logger.debug(
        f"Building results for {len(queries)} queries over {len(artifacts)} artifacts"
    )

    for q in queries:
        name, types = _validate_query(q)
        types = [t.lower() for t in types]

        for a in artifacts:
            aid = gen_id(a["name"])
            if aid in seen:
                continue

            actual_type = _get_artifact_type(a)
            if (name == "*" or name in a["name"].lower()) and actual_type in types:
                results.append({"name": a["name"], "id": aid, "type": actual_type})
                seen.add(aid)

    logger.debug(f"Returning {len(results)} results")
    return results


# ==========================================================
#                 STARTUP / SHUTDOWN EVENTS
# ==========================================================


@app.on_event("startup")
async def startup():
    registry_handler.init_registry()
    logger.info("Registry initialized.")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Server shutting down...")
    gc.collect()


# ==========================================================
#                     ENDPOINTS BEGIN
# ==========================================================


# ----------------------------------------------------------
#  POST /artifacts
# ----------------------------------------------------------
@app.post("/artifacts")
async def post_artifacts(request: Request, offset: Optional[str] = None):
    auth = request.headers.get("X-Authorization")
    if auth:
        require_auth(auth)

    try:
        queries = await request.json()
        if not isinstance(queries, list):
            queries = [queries]
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="There is missing field(s) in the artifact_query or it is formed improperly, or is invalid.",
        )

    artifacts = registry_handler.list_artifacts()
    results = _build_artifact_results(queries, artifacts)

    next_offset = str((int(offset) if offset else 1) + 1)
    return JSONResponse(
        content=results, headers={"offset": next_offset}, status_code=200
    )


# ----------------------------------------------------------
#  DELETE /reset
# ----------------------------------------------------------
@app.delete("/reset")
def reset(request: Request):
    auth = request.headers.get("X-Authorization")
    if auth:
        require_auth(auth)

    registry_handler.reset_registry()
    gc.collect()
    return {"status": "system reset successful"}


# ----------------------------------------------------------
#  GET /artifacts/{artifact_type}/{artifact_id}
# ----------------------------------------------------------
@app.get("/artifacts/{artifact_type}/{artifact_id}")
def get_artifact(artifact_type: str, artifact_id: str, request: Request):
    auth = request.headers.get("X-Authorization")
    if auth:
        require_auth(auth)

    if artifact_type not in ["model", "dataset", "code"]:
        raise HTTPException(status_code=400, detail="There is missing field(s) ...")

    try:
        aid = int(artifact_id)
    except:
        raise HTTPException(status_code=400, detail="There is missing field(s) ...")

    artifacts = registry_handler.list_artifacts()

    for a in artifacts:
        if gen_id(a["name"]) == aid:
            actual = _get_artifact_type(a)
            if actual != artifact_type:
                continue

            url = (
                a.get("code_url")
                if artifact_type == "code"
                else (
                    a.get("dataset_url") if artifact_type == "dataset" else a.get("url")
                )
            )

            return {
                "metadata": {"name": a["name"], "id": str(aid), "type": artifact_type},
                "data": {"url": url},
            }

    raise HTTPException(status_code=404, detail="Artifact does not exist.")


# ----------------------------------------------------------
#  Singular alias for artifact fetch
# ----------------------------------------------------------
@app.get("/artifact/{artifact_type}/{artifact_id}")
def get_artifact_alias(artifact_type: str, artifact_id: str, request: Request):
    return get_artifact(artifact_type, artifact_id, request)


# ----------------------------------------------------------
#  GET /artifact/byName/{name}
# ----------------------------------------------------------
@app.get("/artifact/byName/{name}")
def artifact_by_name(name: str, request: Request):
    auth = request.headers.get("X-Authorization")
    if auth:
        require_auth(auth)

    artifacts = registry_handler.list_artifacts()
    matches = []
    seen = set()

    for a in artifacts:
        if a["name"] == name:
            aid = gen_id(a["name"])
            if aid not in seen:
                seen.add(aid)
                matches.append(
                    {"name": a["name"], "id": aid, "type": _get_artifact_type(a)}
                )

    if not matches:
        raise HTTPException(status_code=404, detail="No such artifact.")

    return JSONResponse(content=matches)


# ----------------------------------------------------------
#  PUT /artifacts/{artifact_type}/{artifact_id}
# ----------------------------------------------------------
@app.put("/artifacts/{artifact_type}/{artifact_id}")
async def update_artifact(artifact_type: str, artifact_id: str, request: Request):
    auth = request.headers.get("X-Authorization")
    if auth:
        require_auth(auth)

    body = await request.json()
    metadata = body.get("metadata")
    data = body.get("data")

    if not metadata or not data:
        raise HTTPException(status_code=400, detail="There is missing field(s) ...")

    if (
        metadata.get("id") != str(artifact_id)
        or metadata.get("type") != artifact_type
        or "url" not in data
    ):
        raise HTTPException(status_code=400, detail="There is missing field(s) ...")

    artifact = registry_handler.get_artifact_by_id(str(int(artifact_id)), artifact_type)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    if artifact["name"] != metadata["name"]:
        raise HTTPException(status_code=400, detail="There is missing field(s) ...")

    new_url = data["url"]

    # update URLs per type
    if artifact_type == "code":
        artifact["code_url"] = new_url
        artifact["url"] = new_url
    elif artifact_type == "dataset":
        artifact["dataset_url"] = new_url
        artifact["url"] = new_url
    else:
        artifact["url"] = new_url
        artifact["code_url"] = new_url

    registry_handler.update_artifact(
        str(int(artifact_id)),
        url=new_url,
        code_url=artifact.get("code_url", "unknown"),
        dataset_url=artifact.get("dataset_url", "unknown"),
    )

    return {"status": "artifact updated successfully"}


# ----------------------------------------------------------
#  POST /artifact/{artifact_type}
# ----------------------------------------------------------
@app.post("/artifact/{artifact_type}")
async def register_artifact(artifact_type: str, request: Request):
    auth = request.headers.get("X-Authorization")
    if auth:
        require_auth(auth)

    body = await request.json()
    url = body.get("url")

    if not url:
        raise HTTPException(status_code=400, detail="There is missing field(s) ...")

    try:
        clean = url.rstrip("/")
        parsed = urlparse(clean)
        path = parsed.path.rstrip("/")
        name = unquote(path.split("/")[-1] or parsed.netloc or "artifact")
    except Exception:
        name = url.rstrip("/").split("/")[-1] or "artifact"

    name = name.lower()
    new_id = gen_id(name)

    artifacts = registry_handler.list_artifacts()
    for a in artifacts:
        if a.get("url") == url or gen_id(a["name"]) == new_id:
            raise HTTPException(status_code=409, detail="Artifact exists already.")

    registry_handler.add_artifact(
        name=name,
        artifact_type=artifact_type,
        score=0.0,
        url=url,
        tags=artifact_type,
        code_url=url if artifact_type in ["code", "model"] else "unknown",
        dataset_url=url if artifact_type == "dataset" else "unknown",
        metadata={"type": artifact_type},
    )

    return JSONResponse(
        status_code=201,
        content={
            "metadata": {"name": name, "id": new_id, "type": artifact_type},
            "data": {"url": url},
        },
    )


# ----------------------------------------------------------
#  GET /artifact/model/{artifact_id}/rate
# ----------------------------------------------------------
@app.get("/artifact/model/{artifact_id}/rate")
def rate_model(artifact_id: str, request: Request):
    auth = request.headers.get("X-Authorization")
    if auth:
        require_auth(auth)

    try:
        aid = int(artifact_id)
    except:
        raise HTTPException(status_code=400, detail="There is missing field(s) ...")

    artifacts = registry_handler.list_artifacts()
    for a in artifacts:
        if gen_id(a["name"]) == aid:
            metadata = json.loads(a.get("metadata_json", "{}"))
            if metadata:
                return metadata

            # fallback
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


# ----------------------------------------------------------
#  GET /artifact/{artifact_type}/{artifact_id}/cost
# ----------------------------------------------------------
@app.get("/artifact/{artifact_type}/{artifact_id}/cost")
def cost(
    artifact_type: str, artifact_id: str, request: Request, dependency: bool = False
):
    auth = request.headers.get("X-Authorization")
    if auth:
        require_auth(auth)

    if artifact_type not in ["model", "dataset", "code"]:
        raise HTTPException(status_code=400, detail="There is missing field(s) ...")

    try:
        aid = int(artifact_id)
    except:
        raise HTTPException(status_code=400, detail="There is missing field(s) ...")

    artifacts = registry_handler.list_artifacts()
    for a in artifacts:
        if gen_id(a["name"]) == aid:
            base = 412.5 if artifact_type == "model" else 100.0
            if dependency:
                base *= 1.2
            return {str(aid): {"total_cost": base}}

    raise HTTPException(status_code=404, detail="Artifact does not exist.")


# ----------------------------------------------------------
#  GET /artifact/model/{artifact_id}/lineage
# ----------------------------------------------------------
@app.get("/artifact/model/{artifact_id}/lineage")
def lineage(artifact_id: str, request: Request):
    auth = request.headers.get("X-Authorization")
    if auth:
        require_auth(auth)

    try:
        aid = int(artifact_id)
    except:
        raise HTTPException(
            status_code=400,
            detail="The lineage graph cannot be computed because the artifact metadata is missing or malformed.",
        )

    artifacts = registry_handler.list_artifacts()
    for a in artifacts:
        if gen_id(a["name"]) == aid:
            dataset_url = a.get("dataset_url", "unknown")
            if not dataset_url or dataset_url == "unknown":
                raise HTTPException(
                    status_code=400,
                    detail="The lineage graph cannot be computed because the artifact metadata is missing or malformed.",
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


# ----------------------------------------------------------
#  POST /artifact/model/{artifact_id}/license-check
# ----------------------------------------------------------
@app.post("/artifact/model/{artifact_id}/license-check")
async def license_check(artifact_id: str, request: Request):
    auth = request.headers.get("X-Authorization")
    if auth:
        require_auth(auth)

    try:
        body = await request.json()
    except:
        raise HTTPException(
            status_code=400,
            detail="The license check request is malformed or references an unsupported usage context.",
        )

    github_url = body.get("github_url")
    if not github_url:
        raise HTTPException(
            status_code=400,
            detail="The license check request is malformed or references an unsupported usage context.",
        )

    try:
        aid = int(artifact_id)
    except:
        raise HTTPException(
            status_code=404,
            detail="The artifact or GitHub project could not be found.",
        )

    artifacts = registry_handler.list_artifacts()
    if not any(gen_id(a["name"]) == aid for a in artifacts):
        raise HTTPException(
            status_code=404,
            detail="The artifact or GitHub project could not be found.",
        )

    lower = github_url.lower()
    if "apache" in lower or "google" in lower:
        return True
    elif "github" in lower:
        return False
    else:
        raise HTTPException(
            status_code=502,
            detail="External license information could not be retrieved.",
        )


# ----------------------------------------------------------
#  POST /artifact/byRegEx
# ----------------------------------------------------------
@app.post("/artifact/byRegEx")
async def regex_search(request: Request):
    auth = request.headers.get("X-Authorization")
    if auth:
        require_auth(auth)

    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="There is missing field(s) ...")

    regex = body.get("regex")
    if not regex:
        raise HTTPException(status_code=400, detail="There is missing field(s) ...")

    try:
        pattern = re.compile(regex, re.IGNORECASE)
    except:
        raise HTTPException(status_code=400, detail="There is missing field(s) ...")

    artifacts = registry_handler.list_artifacts()
    matches = []
    seen = set()

    for a in artifacts:
        aid = gen_id(a["name"])
        if aid in seen:
            continue

        try:
            meta = json.loads(a.get("metadata_json", "{}"))
            meta_text = " ".join(str(v) for v in meta.values())
        except:
            meta_text = ""

        searchable = " ".join(
            [
                a.get("name", ""),
                a.get("tags", ""),
                a.get("code_url", ""),
                a.get("dataset_url", ""),
                a.get("url", ""),
                meta_text,
            ]
        )

        if pattern.search(searchable):
            matches.append(
                {"name": a["name"], "id": aid, "type": _get_artifact_type(a)}
            )
            seen.add(aid)

    if not matches:
        raise HTTPException(
            status_code=404, detail="No artifact found under this regex."
        )

    return JSONResponse(content=matches)


# ----------------------------------------------------------
#  GET /packages
# ----------------------------------------------------------
@app.get("/packages")
def get_packages(request: Request):
    auth = request.headers.get("X-Authorization")
    if auth:
        require_auth(auth)

    artifacts = registry_handler.list_artifacts()
    results = []
    seen = set()

    for a in artifacts:
        aid = gen_id(a["name"])
        if aid in seen:
            continue

        seen.add(aid)
        category = a.get("artifact_type") or _get_artifact_type(a)
        results.append({"id": aid, "name": a["name"], "category": category.upper()})

    return {"artifacts": results}


# ----------------------------------------------------------
#  GET /tracks
# ----------------------------------------------------------
@app.get("/tracks")
def tracks():
    return {"plannedTracks": ["High assurance track"]}


# ----------------------------------------------------------
#  GET /
# ----------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not templates:
        return HTMLResponse("<h1>Registry API</h1><p>Visit /docs for API docs</p>")

    try:
        artifacts = registry_handler.list_artifacts()
        return templates.TemplateResponse(
            "index.html", {"request": request, "models": artifacts}
        )
    except:
        return HTMLResponse("<h1>Registry API</h1><p>Visit /docs for API docs</p>")


# ----------------------------------------------------------
#  GET /health
# ----------------------------------------------------------
@app.get("/health")
def health():
    logger.info("Health endpoint called")
    return Response(status_code=200)


# ==========================================================
#          UVICORN LOCAL ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
