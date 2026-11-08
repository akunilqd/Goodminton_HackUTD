"""Upload a pipeline run directory to SpacetimeDB via HTTP REST.

Calls reducers defined in rallymirror-backend/spacetimedb/src/index.ts.
Deploy the module first:
    cd rallymirror-backend
    spacetime publish badminton --server maincloud --yes

Then upload a run:
    python main.py stdb-upload outputs/run_5class --module badminton
"""

import csv
import json
import os
import time
from math import hypot, isfinite
from pathlib import Path

try:
    import requests
except ImportError as exc:
    raise ImportError("pip install requests") from exc

MAINCLOUD = "https://maincloud.spacetimedb.com"
COURT_W = 6.10
COURT_L = 13.40
POSITION_SAMPLE_EVERY = 5
SPEED_SAMPLE_EVERY = 1


def _read_token(server: str) -> str | None:
    """Try to read the auth token from the SpacetimeDB CLI config."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # pip install tomli on 3.10
        except ImportError:
            return None
    cfg = Path(os.environ.get("LOCALAPPDATA", "")) / "SpacetimeDB" / "config" / "cli.toml"
    if not cfg.exists():
        cfg = Path.home() / ".spacetime" / "config.toml"
    if not cfg.exists():
        return None
    try:
        data = tomllib.loads(cfg.read_text(encoding="utf-8"))
        # SpacetimeDB 2.x stores tokens per server nickname
        for srv in data.get("server", []):
            if isinstance(srv, dict):
                nick = srv.get("nickname", "")
                url = srv.get("host", srv.get("url", ""))
                if "maincloud" in nick or "maincloud" in url or server in url:
                    return srv.get("token")
        # Fallback: first token found
        for srv in data.get("server", []):
            if isinstance(srv, dict) and srv.get("token"):
                return srv["token"]
    except Exception:
        pass
    return None


def _post(base_url: str, module: str, reducer: str, args: list, token: str | None = None):
    url = f"{base_url.rstrip('/')}/v1/database/{module}/call/{reducer}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(url, data=json.dumps(args), headers=headers, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"[{reducer}] HTTP {resp.status_code}: {resp.text[:300]}")
    return resp


def upload_run(
    run_dir: Path,
    *,
    module: str = "badminton",
    server: str = MAINCLOUD,
    token: str | None = None,
    clear_first: bool = True,
) -> dict:
    run_dir = Path(run_dir).resolve()
    run_id = run_dir.name

    # Try to get token from CLI config if not provided
    if token is None:
        token = _read_token(server)
        if token:
            print(f"  using token from SpacetimeDB CLI config")
        else:
            print("  WARNING: no auth token found — requests may fail on maincloud")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta_path = run_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    video_info = metadata.get("video", {})
    if isinstance(video_info, dict):
        video = str(video_info.get("source", ""))
        fps = float(video_info.get("fps", 0))
    else:
        video = str(video_info)
        fps = float(metadata.get("fps", 0))
    total_frames = int(metadata.get("processed_frames", metadata.get("total_frames", 0)))
    created_at = int(time.time() * 1000)

    if clear_first:
        try:
            _post(server, module, "clear_run", [run_id], token)
            print(f"  clearRun OK")
        except RuntimeError as e:
            print(f"  clearRun skipped ({e})")

    _post(server, module, "upsert_run", [run_id, video, total_frames, fps, created_at], token)
    print(f"  upsertRun OK  ({total_frames} frames, {fps} fps)")

    # ── Shots ─────────────────────────────────────────────────────────────────
    shots_path = run_dir / "shots.jsonl"
    n_shots = 0
    if shots_path.exists():
        with shots_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                s = json.loads(line)
                _post(server, module, "add_shot", [
                    run_id,
                    int(s["shot_id"]),
                    int(s.get("hitter_id") or 0),
                    s.get("player_side") or "",
                    s.get("shot_type_en", ""),
                    float(s.get("confidence", 0)),
                    int(s.get("start_frame", 0)),
                    int(s.get("end_frame", 0)),
                ], token)
                n_shots += 1
    print(f"  addShot OK    ({n_shots} shots)")

    # ── Speed CSV ─────────────────────────────────────────────────────────────
    speed_path = run_dir / "speed.csv"
    n_speed = 0
    if speed_path.exists():
        with speed_path.open(encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i % SPEED_SAMPLE_EVERY != 0:
                    continue
                def _fv(key):
                    v = row.get(key, "")
                    try:
                        return float(v) if v not in ("", None) else 0.0
                    except ValueError:
                        return 0.0
                _post(server, module, "add_frame_speed", [
                    run_id,
                    int(row["frame_index"]),
                    _fv("timestamp_s"),
                    _fv("shuttle_speed_px_s"),
                    _fv("shuttle_speed_m_s"),
                    _fv("player1_speed_m_s"),
                    _fv("player2_speed_m_s"),
                ], token)
                n_speed += 1
    print(f"  addFrameSpeed OK ({n_speed} rows)")

    # ── Player positions (batched) ─────────────────────────────────────────────
    frames_path = run_dir / "frames.jsonl"
    pos_by_tid: dict[int, tuple[list, list, list]] = {}
    if frames_path.exists():
        with frames_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip() or i % POSITION_SAMPLE_EVERY != 0:
                    continue
                row = json.loads(line)
                for p in row.get("players") or []:
                    tid = p.get("track_id")
                    cxy = p.get("court_xy_m")
                    if tid is None or not cxy or len(cxy) < 2:
                        continue
                    nx = max(0.0, min(1.0, float(cxy[0]) / COURT_W))
                    ny = max(0.0, min(1.0, float(cxy[1]) / COURT_L))
                    if not (isfinite(nx) and isfinite(ny)):
                        continue
                    if tid not in pos_by_tid:
                        pos_by_tid[tid] = ([], [], [])
                    pos_by_tid[tid][0].append(int(row["frame_index"]))
                    pos_by_tid[tid][1].append(nx)
                    pos_by_tid[tid][2].append(ny)

    n_pos = 0
    BATCH = 150
    for tid, (fis, nxs, nys) in pos_by_tid.items():
        for start in range(0, len(fis), BATCH):
            _post(server, module, "add_player_positions_batch", [
                run_id,
                int(tid),
                fis[start:start + BATCH],
                nxs[start:start + BATCH],
                nys[start:start + BATCH],
            ], token)
            n_pos += len(fis[start:start + BATCH])
    print(f"  addPlayerPositionsBatch OK ({n_pos} positions)")

    # ── Player stats ──────────────────────────────────────────────────────────
    from module7_summary.analytics import (
        _load_frames, _load_shots, _load_speed_csv,
        compute_player_distance, compute_avg_shuttle_speed, compute_shot_distribution,
    )
    frames_rows = _load_frames(run_dir)
    shots_rows = _load_shots(run_dir)
    speed_map = _load_speed_csv(run_dir)
    distances = compute_player_distance(frames_rows)
    avg_speeds = compute_avg_shuttle_speed(shots_rows, speed_map)
    shot_dist = compute_shot_distribution(shots_rows)

    for tid in sorted(distances.keys() | shot_dist.keys()):
        _post(server, module, "set_player_stat", [
            run_id,
            int(tid),
            float(distances.get(tid, 0.0)),
            float(avg_speeds.get(tid, 0.0)),
            int(sum(shot_dist.get(tid, {}).values())),
        ], token)
    print(f"  setPlayerStat OK ({len(distances)} players)")

    return {
        "run_id": run_id,
        "module": module,
        "server": server,
        "shots": n_shots,
        "speed_rows": n_speed,
        "positions": n_pos,
    }
