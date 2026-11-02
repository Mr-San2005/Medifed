"""Local-only FastAPI application for MediFed results and image diagnosis."""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.services import available_variants, load_results, run_prediction
from medifed.config import HOSPITALS, PROJECT_ROOT


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
MAX_IMAGE_BYTES = 10 * 1024 * 1024

app = FastAPI(title="MediFed", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/")
async def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/diagnose")
async def diagnosis_page():
    return FileResponse(STATIC_DIR / "diagnose.html")


@app.get("/network")
async def network_page():
    return FileResponse(STATIC_DIR / "network.html")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/results")
async def results():
    return load_results()


@app.get("/api/variants")
async def variants():
    return {"variants": available_variants()}


async def _read_bounded_body(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_IMAGE_BYTES:
        raise OverflowError
    chunks = bytearray()
    async for chunk in request.stream():
        chunks.extend(chunk)
        if len(chunks) > MAX_IMAGE_BYTES:
            raise OverflowError
    return bytes(chunks)


@app.post("/api/predict")
async def predict(request: Request, variant: str):
    response_headers = {"Cache-Control": "no-store"}
    content_type = request.headers.get("content-type", "").lower()
    if not (content_type.startswith("image/") or content_type == "application/octet-stream"):
        return JSONResponse({"error": "Send the image bytes with an image/* content type."}, status_code=400, headers=response_headers)
    try:
        image_bytes = await _read_bounded_body(request)
    except OverflowError:
        return JSONResponse({"error": "Image exceeds the 10 MB limit."}, status_code=413, headers=response_headers)

    known_variants = {
        "fedavg_global",
        *(f"fedbn_hospital{h}" for h in HOSPITALS),
        *(f"local_hospital{h}" for h in HOSPITALS),
    }
    if variant not in known_variants:
        return JSONResponse({"error": f"Unknown model variant: {variant}"}, status_code=404, headers=response_headers)
    if variant not in available_variants():
        return JSONResponse({"error": f"Required checkpoint is missing for model variant: {variant}"}, status_code=503, headers=response_headers)
    try:
        result = run_prediction(variant, image_bytes)
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400, headers=response_headers)
    except FileNotFoundError as error:
        return JSONResponse({"error": str(error)}, status_code=503, headers=response_headers)

    payload = {
        "ranked": result["ranked"],
        "top_class": result["top_class"],
        "milliseconds": result["milliseconds"],
        "original_png": base64.b64encode(result["original_png"]).decode("ascii"),
        "heatmap_png": base64.b64encode(result["heatmap_png"]).decode("ascii"),
    }
    return JSONResponse(payload, headers=response_headers)