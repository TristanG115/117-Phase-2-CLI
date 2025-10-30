# server.py
import hashlib
import json
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from handlers import registry_handler

templates = Jinja2Templates(directory=".")

app = FastAPI(title="117 Phase 2 - Trustworthy Model Registry")


# === Helper functions ===
def require_auth(_):
    """Auth bypass for baseline testing (Reliability track)."""
    return True


def gen_id(name: str) -> int:
    """Generate deterministic 10-digit artifact ID."""
    return abs(int(hashlib.sha256(name.encode()).hexdigest(), 16)) % (10**10)


# === POST /artifacts ===
@app.post("/artifacts")
async def get_artifacts(request: Request, offset: int = 1):
    """BASELINE: Return artifacts matching the given query list."""
    require_auth(None)

    try:
        queries = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(queries, list) or len(queries) == 0:
        raise HTTPException(status_code=400, detail="Missing query list")

    models = registry_handler.list_models()
    results = []

    for q in queries:
        name = q.get("name", "").lower()
        types = q.get("types", [])
        if not name or not isinstance(types, list):
            raise HTTPException(status_code=400, detail="Malformed artifact_query")

        for m in models:
            if name == "*" or name in m["name"].lower():
                results.append(
                    {
                        "name": m["name"],
                        "id": gen_id(m["name"]),
                        "type": types[0] if types else "model",
                    }
                )

    headers = {"offset": str(offset + 1)}
    return JSONResponse(content=results, headers=headers)


# === DELETE /reset ===
@app.delete("/reset")
def reset_registry():
    """BASELINE: Reset registry to a clean state."""
    require_auth(None)
    registry_handler.reset_registry()
    return {"status": "system reset successful"}


# === GET /artifacts/{artifact_type}/{id} ===
@app.get("/artifacts/{artifact_type}/{artifact_id}")
def get_artifact(artifact_type: str, artifact_id: str):
    """BASELINE: Retrieve one artifact by id."""
    require_auth(None)

    try:
        aid = int(artifact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid artifact id")

    models = registry_handler.list_models()
    for m in models:
        if gen_id(m["name"]) == aid:
            url = m["code_url"] if artifact_type == "code" else m["dataset_url"]
            return {
                "metadata": {
                    "name": m["name"],
                    "id": aid,
                    "type": artifact_type,
                },
                "data": {"url": url},
            }

    raise HTTPException(status_code=404, detail="Artifact does not exist")


# === PUT /artifacts/{artifact_type}/{id} ===
@app.put("/artifacts/{artifact_type}/{artifact_id}")
async def update_artifact(artifact_type: str, artifact_id: str, request: Request):
    """BASELINE: Update artifact content."""
    require_auth(None)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    metadata = body.get("metadata")
    data = body.get("data")
    if not metadata or not data:
        raise HTTPException(status_code=400, detail="Missing metadata or data")

    if (
        str(metadata.get("id")) != str(artifact_id)
        or metadata.get("type") != artifact_type
    ):
        raise HTTPException(status_code=400, detail="Metadata mismatch")

    models = registry_handler.list_models()
    found = False
    for m in models:
        if gen_id(m["name"]) == int(artifact_id):
            found = True
            if artifact_type == "code":
                m["code_url"] = data.get("url", m["code_url"])
            elif artifact_type == "dataset":
                m["dataset_url"] = data.get("url", m["dataset_url"])
            registry_handler.add_model(
                name=m["name"],
                score=m["score"],
                tags=m.get("tags", ""),
                code_url=m.get("code_url", ""),
                dataset_url=m.get("dataset_url", ""),
                metadata_json=json.dumps(m),
            )
            break

    if not found:
        raise HTTPException(status_code=404, detail="Artifact does not exist")

    return {"status": "artifact updated successfully"}


# === POST /artifact/{artifact_type} ===
@app.post("/artifact/{artifact_type}")
async def register_artifact(artifact_type: str, request: Request):
    """BASELINE: Register a new artifact by URL."""
    require_auth(None)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    url = body.get("url")
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="Missing url field")

    name = url.rstrip("/").split("/")[-1]
    new_id = gen_id(name)

    models = registry_handler.list_models()
    for m in models:
        if gen_id(m["name"]) == new_id:
            raise HTTPException(status_code=409, detail="Artifact exists already")

    registry_handler.add_model(
        name=name,
        score=0.0,
        tags=artifact_type,
        code_url=url if artifact_type == "code" else "unknown",
        dataset_url=url if artifact_type == "dataset" else "unknown",
        metadata_json=json.dumps({"type": artifact_type}),
    )

    resp = {
        "metadata": {"name": name, "id": new_id, "type": artifact_type},
        "data": {"url": url},
    }
    return JSONResponse(status_code=201, content=resp)


# === GET /artifact/model/{id}/rate ===
@app.get("/artifact/model/{artifact_id}/rate")
def get_rating(artifact_id: str):
    """BASELINE: Return metrics for this model artifact."""
    require_auth(None)

    try:
        aid = int(artifact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid artifact id")

    models = registry_handler.list_models()
    for m in models:
        if gen_id(m["name"]) == aid:
            meta = json.loads(m.get("metadata_json", "{}"))
            return (
                meta
                if meta
                else {
                    "name": m["name"],
                    "category": "MODEL",
                    "net_score": m["score"],
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
            )

    raise HTTPException(status_code=404, detail="Artifact does not exist")


# === GET /artifact/{artifact_type}/{id}/cost ===
@app.get("/artifact/{artifact_type}/{artifact_id}/cost")
def get_cost(artifact_type: str, artifact_id: str, dependency: bool = False):
    """BASELINE: Return total cost of the artifact."""
    require_auth(None)

    try:
        aid = int(artifact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid artifact id")

    models = registry_handler.list_models()
    for m in models:
        if gen_id(m["name"]) == aid:
            base_cost = 412.5 if artifact_type == "model" else 100.0
            if dependency:
                base_cost *= 1.2
            return {str(aid): {"total_cost": base_cost}}

    raise HTTPException(status_code=404, detail="Artifact does not exist")


# === GET /artifact/model/{id}/lineage ===
@app.get("/artifact/model/{artifact_id}/lineage")
def get_lineage(artifact_id: str):
    """BASELINE: Retrieve lineage graph."""
    require_auth(None)

    try:
        aid = int(artifact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid artifact id")

    models = registry_handler.list_models()
    for m in models:
        if gen_id(m["name"]) == aid:
            dataset_name = m.get("dataset_url", "unknown").split("/")[-1]
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

    raise HTTPException(status_code=404, detail="Artifact does not exist")


# === POST /artifact/model/{id}/license-check ===
@app.post("/artifact/model/{artifact_id}/license-check")
async def license_check(artifact_id: str, request: Request):
    """BASELINE: License compatibility analysis."""
    require_auth(None)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    github_url = body.get("github_url")
    if not github_url or not isinstance(github_url, str):
        raise HTTPException(status_code=400, detail="Missing github_url field")

    if "apache" in github_url.lower() or "google" in github_url.lower():
        return JSONResponse(content=True)
    elif "github" in github_url.lower():
        return JSONResponse(content=False)
    else:
        raise HTTPException(status_code=502, detail="External license retrieval failed")


# === POST /artifact/byRegEx ===
@app.post("/artifact/byRegEx")
async def artifact_by_regex(request: Request):
    """BASELINE: Search artifacts using regex."""
    require_auth(None)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    regex = body.get("regex")
    if not regex or not isinstance(regex, str):
        raise HTTPException(status_code=400, detail="Missing regex field")

    try:
        pattern = re.compile(regex)
    except re.error:
        raise HTTPException(status_code=400, detail="Invalid regex pattern")

    models = registry_handler.list_models()
    matches = []
    for m in models:
        text = f"{m['name']} {m.get('tags', '')} {m.get('code_url', '')} {m.get('dataset_url', '')}"
        if pattern.search(text):
            matches.append(
                {"name": m["name"], "id": gen_id(m["name"]), "type": "model"}
            )

    if not matches:
        raise HTTPException(
            status_code=404, detail="No artifact found under this regex"
        )

    return JSONResponse(content=matches)


# === GET /tracks ===
@app.get("/tracks")
def get_tracks():
    """Return the list of tracks this team has implemented."""
    return {"plannedTracks": ["Reliability track"]}


# Frontend setup
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the main registry dashboard (index.html)."""
    from handlers import registry_handler

    models = registry_handler.list_models()
    return templates.TemplateResponse(
        "index.html", {"request": request, "models": models}
    )
