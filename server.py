"""
Simple FastAPI server
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Model Packages API", version="1.0.0")

# Set up templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# In-memory package storage
packages_db = {}

# Read results from this file
RESULTS_FILE = Path("evaluation_results.ndjson") 

logger = logging.getLogger(__name__)


class Package(BaseModel):
    """Package model matching evaluation output"""
    name: str
    category: str
    net_score: float
    ramp_up_time: float
    bus_factor: float
    performance_claims: float
    license: float
    dataset_and_code_score: float
    dataset_quality: float
    code_quality: float


def load_packages_from_ndjson():
    global packages_db
    packages_db = {}
    
    if not RESULTS_FILE.exists():
        logger.warning(f"Results file not found: {RESULTS_FILE}")
        logger.info("Run: python model_evaluator.py test_urls.txt > evaluation_results.ndjson")
        return
    
    try:
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    pkg = json.loads(line)
                    package_name = pkg.get("name", "unknown")
                    packages_db[package_name] = pkg
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line: {e}")
                    continue
        
        logger.info(f"Loaded {len(packages_db)} packages from {RESULTS_FILE}")
    except Exception as e:
        logger.error(f"Error loading packages: {e}")
        packages_db = {}


@app.on_event("startup")
async def startup():
    """Load packages on server startup"""
    load_packages_from_ndjson()


@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Model Packages API",
        "packages_loaded": len(packages_db),
        "endpoints": {
            "packages": "/packages",
            "dashboard": "/dashboard",
            "reload": "/reload"
        }
    }


@app.get("/packages", response_model=List[Package])
async def get_packages(
    name: Optional[str] = Query(None, description="Search by package name"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=1000, description="Max results to return")
):
    """
    GET /packages: List all packages, optional name search
    
    Query Parameters:
    name: Filter packages by name (case-insensitive substring match)
    offset: Pagination offset (default: 0)
    limit: Maximum number of results (default: 100)
    
    Returns:
        List of Package objects
    """
    # Get all packages
    results = list(packages_db.values())
    
    # Filter by name if provided
    if name:
        name_lower = name.lower()
        results = [
            pkg for pkg in results 
            if name_lower in pkg.get("name", "").lower()
        ]
    
    # Sort by net_score (highest first)
    results.sort(key=lambda x: x.get("net_score", 0), reverse=True)
    
    # Apply pagination
    paginated_results = results[offset:offset + limit]
    
    return paginated_results


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    HTML Bootstrap based dashboard UI for browsing and searching packages
    Serves index.html template
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "total_packages": len(packages_db)}
    )


@app.post("/reload")
async def reload_packages():
    """
    Reload packages from NDJSON file
    In case of updates to evaluation results
    """
    load_packages_from_ndjson()
    return {
        "message": "Packages reloaded",
        "count": len(packages_db)
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Model Packages API Server")
    print("=" * 60)
    print(f"\nLooking for evaluation results in: {RESULTS_FILE}")
    
    if not RESULTS_FILE.exists():
        print("\nNo evaluation results found")
        print("\nTo create evaluation results, run:")
        print("  1. export GITHUB_TOKEN='your_token'")
        print("  2. export LOG_FILE='evaluation.log'")
        print("  3. export LOG_LEVEL='1'")
        print("  4. echo 'https://huggingface.co/google-bert/bert-base-uncased' > test_urls.txt")
        print("  5. python model_evaluator.py test_urls.txt > evaluation_results.ndjson")
    
    # Check if templates/index.html exists
    template_file = templates_dir / "index.html"
    if not template_file.exists():
        print(f"\nFile not found: {template_file}")
    
    print("\nDashboard: http://localhost:8000/dashboard")
    print("\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")