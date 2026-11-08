"""Local service connecting the website to the analysis pipeline.

    pip install -r requirements-server.txt
    python server.py

Then open http://localhost:8000. Run it on the machine that has the GPU and model
weights. Results go to SpacetimeDB through `main.py run --stdb`, which uses that
machine's SpacetimeDB login.

There is no user authentication, so it only listens on this computer by default.
Requests must come from the site it serves; there is deliberately no CORS.
"""

import argparse
import asyncio
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import ClientDisconnect

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
JOBS_DIR = ROOT / "jobs"
OUTPUTS_DIR = ROOT / "outputs"

MAX_BYTES = 4 * 10**9
VIDEO_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime"}
JOB_ID = re.compile(r"^job_[0-9]{10}_[0-9a-f]{8}$")
ACTIVE = {"queued", "running"}

# A line from the pipeline's log moves the job to the stage named here.
STAGE_MARKERS = (
    ("Inference devices", "court"),
    ("Fixed H calibrated", "tracking"),
    ("Detecting hits", "hits"),
    ("Classifying shots", "shots"),
    ("Rallies:", "publishing"),
)

LOGGER = logging.getLogger("goodminton.server")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()
_process: tuple[str, subprocess.Popen] | None = None


def pipeline_command(video: Path, output: Path) -> list[str]:
    return [sys.executable, "-u", str(ROOT / "main.py"), "run", str(video), "--output", str(output), "--stdb"]


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _save(job: dict) -> None:
    path = _job_dir(job["id"]) / "job.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(job), encoding="utf-8")
    os.replace(temporary, path)


def _update(job_id: str, **changes) -> None:
    with _lock:
        job = _jobs[job_id]
        job.update(changes)
        _save(job)


def _video_path(job: dict) -> Path:
    return _job_dir(job["id"]) / f"video{job['extension']}"


def _public(job: dict) -> dict:
    has_video = _video_path(job).is_file()
    fields = ("id", "status", "stage", "filename", "near", "far", "error", "warnings", "rallies", "created_at", "finished_at")
    return {**{k: job.get(k) for k in fields}, "video_url": f"/api/videos/{job['id']}" if has_video else None}


def _load_jobs() -> None:
    """Restore finished jobs. Anything mid-flight died with the previous server."""
    JOBS_DIR.mkdir(exist_ok=True)
    for path in JOBS_DIR.glob("job_*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if job.get("status") in ACTIVE:
            job.update(status="failed", error="The server restarted during this analysis. Please upload the video again.")
            _save(job)
        _jobs[job["id"]] = job


def _run_job(job_id: str) -> None:
    global _process
    job = _jobs[job_id]
    if job["status"] != "queued":
        return
    output = OUTPUTS_DIR / job_id
    OUTPUTS_DIR.mkdir(exist_ok=True)
    _update(job_id, status="running", stage="checking")
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    last_error, warnings = None, []
    with (_job_dir(job_id) / "pipeline.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            pipeline_command(_video_path(job), output), cwd=ROOT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        with _lock:
            _process = (job_id, process)
        for line in process.stdout:
            log.write(line)
            log.flush()
            for marker, stage in STAGE_MARKERS:
                if marker in line:
                    _update(job_id, stage=stage)
            if line.startswith("ERROR "):
                last_error = line[6:].strip()
            elif "SpacetimeDB push" in line and "failed" in line:
                warnings.append(line.split(" ", 1)[-1].strip())
        code = process.wait()
    with _lock:
        _process = None
    finished = time.time()
    if _jobs[job_id]["status"] == "cancelled":
        shutil.rmtree(output, ignore_errors=True)
    elif code == 0:
        rallies = None
        try:
            rallies = json.loads((output / "metadata.json").read_text(encoding="utf-8")).get("rallies")
        except (OSError, ValueError):
            pass
        _update(job_id, status="complete", stage="done", rallies=rallies, warnings=warnings, finished_at=finished)
    else:
        _update(job_id, status="failed", finished_at=finished, warnings=warnings,
                error=last_error or "The analysis stopped unexpectedly. See pipeline.log in this job's folder.")


def _worker() -> None:
    # One at a time: a second GPU pipeline would only make both fail.
    while True:
        job_id = _queue.get()
        try:
            _run_job(job_id)
        except Exception as exc:
            LOGGER.exception("Job %s crashed", job_id)
            _update(job_id, status="failed", error=f"The analysis could not start: {exc}", finished_at=time.time())


@asynccontextmanager
async def lifespan(_app):
    _load_jobs()
    threading.Thread(target=_worker, daemon=True, name="pipeline-worker").start()
    yield
    with _lock:
        running = _process
    if running:
        running[1].terminate()


app = FastAPI(title="Goodminton", lifespan=lifespan, docs_url=None, redoc_url=None)


def same_site(request: Request) -> None:
    """Stop other websites from driving this service through the visitor's browser."""
    origin = request.headers.get("origin")
    if origin is not None and urlsplit(origin).netloc != request.headers.get("host"):
        raise HTTPException(403, "Requests must come from this site.")
    if request.headers.get("x-goodminton") != "1":
        raise HTTPException(403, "Missing request header.")


def _clean_name(value: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f<>&\"']", "", value).strip()[:60]


def _get_job(job_id: str) -> dict:
    job = _jobs.get(job_id) if JOB_ID.match(job_id) else None
    if job is None:
        raise HTTPException(404, "No such analysis.")
    return job


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/jobs", status_code=201, dependencies=[Depends(same_site)])
async def create_job(request: Request, filename: str, near: str = "", far: str = ""):
    extension = Path(filename).suffix.lower()
    if extension not in VIDEO_TYPES:
        raise HTTPException(415, "Choose an MP4 or MOV file.")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BYTES:
        raise HTTPException(413, "This file is larger than the 4 GB limit.")

    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    folder = _job_dir(job_id)
    folder.mkdir(parents=True)
    partial = folder / "upload.partial"
    size, header = 0, b""
    try:
        with partial.open("wb") as stream:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "This file is larger than the 4 GB limit.")
                if len(header) < 12:
                    header += chunk[:12]
                await asyncio.to_thread(stream.write, chunk)
        # Both MP4 and MOV carry an 'ftyp', 'moov', 'mdat', 'free' or 'wide' box at the start.
        if size == 0 or header[4:8] not in {b"ftyp", b"moov", b"mdat", b"free", b"wide"}:
            raise HTTPException(415, "That file doesn't look like a video.")
    except ClientDisconnect:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, "The upload was interrupted.") from None
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    job = {
        "id": job_id, "status": "queued", "stage": "queued", "extension": extension,
        "filename": _clean_name(Path(filename).name), "near": _clean_name(near), "far": _clean_name(far),
        "size": size, "error": None, "warnings": [], "rallies": None,
        "created_at": time.time(), "finished_at": None,
    }
    os.replace(partial, _video_path(job))
    with _lock:
        _jobs[job_id] = job
        _save(job)
    _queue.put(job_id)
    return _public(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _public(_get_job(job_id))


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(same_site)])
def cancel_job(job_id: str):
    job = _get_job(job_id)
    with _lock:
        if job["status"] not in ACTIVE:
            raise HTTPException(409, "This analysis has already finished.")
        job.update(status="cancelled", finished_at=time.time())
        _save(job)
        running = _process
    if running and running[0] == job_id:
        running[1].terminate()
    _video_path(job).unlink(missing_ok=True)
    return _public(job)


@app.get("/api/videos/{job_id}")
def get_video(job_id: str):
    job = _get_job(job_id)
    path = _video_path(job)
    if not path.is_file():
        raise HTTPException(404, "This video is no longer stored.")
    return FileResponse(path, media_type=VIDEO_TYPES[job["extension"]])  # Handles Range, so seeking works.


@app.get("/")
def home():
    return RedirectResponse("/strung/index.html")


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="site")


def main() -> None:
    parser = argparse.ArgumentParser(description="Goodminton upload and analysis service")
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on (default: this computer only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    hosts = ["localhost", "127.0.0.1"] + ([args.host] if args.host != "0.0.0.0" else ["*"])
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    if args.host not in {"127.0.0.1", "localhost"}:
        LOGGER.warning("Listening on %s: anyone who can reach it can upload videos.", args.host)
    print(f"Goodminton is running: http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
