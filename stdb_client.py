"""SpacetimeDB HTTP client for the badminton pipeline.

Wraps push (reducer calls) and fetch (SQL queries) against the
'badminton' module.  All push functions are fire-and-forget: they raise
RuntimeError on HTTP failure so the caller can decide to log-and-continue
rather than crashing the pipeline.

Usage:
    client = StdbClient(module="badminton", server="https://maincloud.spacetimedb.com")
    client.push_run_meta(run_id, video_source, fps, total_frames)
    client.push_running_details_batch(run_id, frames_list)   # list of FrameResult-dicts
    client.push_shot_classes(run_id, hits_list, shots_list)
    client.push_summary(run_id, analytics_dict)
"""

import json
import os
import time
from math import hypot, isfinite
from pathlib import Path

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

MAINCLOUD = "https://maincloud.spacetimedb.com"
COURT_W = 6.10
COURT_L = 13.40
BATCH_SIZE = 100   # frames per reducer call


def _read_token() -> str | None:
    try:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return None
        cfg = Path(os.environ.get("LOCALAPPDATA", "")) / "SpacetimeDB" / "config" / "cli.toml"
        if not cfg.exists():
            cfg = Path.home() / ".spacetime" / "config.toml"
        if not cfg.exists():
            return None
        data = tomllib.loads(cfg.read_text(encoding="utf-8"))
        for srv in data.get("server", []):
            if isinstance(srv, dict) and srv.get("token"):
                return srv["token"]
    except Exception:
        pass
    return None


class StdbClient:
    def __init__(
        self,
        module: str = "badminton",
        server: str = MAINCLOUD,
        token: str | None = None,
    ):
        if not _HAS_REQUESTS:
            raise ImportError("pip install requests")
        self.module = module
        self.server = server.rstrip("/")
        self.token = token or _read_token()

    def _post(self, reducer: str, args: list):
        url = f"{self.server}/v1/database/{self.module}/call/{reducer}"
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = requests.post(url, data=json.dumps(args), headers=headers, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"[{reducer}] HTTP {resp.status_code}: {resp.text[:200]}")

    def _sql(self, query: str) -> list[dict]:
        url = f"{self.server}/v1/database/{self.module}/sql"
        headers = {"Content-Type": "text/plain"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = requests.get(url, data=query, headers=headers, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"[sql] HTTP {resp.status_code}: {resp.text[:200]}")
        rows = resp.json()
        # SpacetimeDB returns [{"schema": {...}, "rows": [...]}]
        if isinstance(rows, list) and rows and "rows" in rows[0]:
            schema_cols = [c["name"] for c in rows[0]["schema"]["elements"]]
            return [dict(zip(schema_cols, r)) for r in rows[0]["rows"]]
        return []

    # ── Push: run metadata ────────────────────────────────────────────────────

    def push_run_meta(self, run_id: str, video_source: str, fps: float, total_frames: int):
        self._post("upsert_run_meta", [
            run_id, video_source, float(fps), int(total_frames), int(time.time() * 1000)
        ])

    def clear_run(self, run_id: str):
        try:
            self._post("clear_run", [run_id])
        except RuntimeError:
            pass  # May not exist yet

    # ── Push: running_details (batch from frames.jsonl rows) ──────────────────

    def push_running_details_batch(self, run_id: str, frame_rows: list[dict]):
        """
        frame_rows: list of dicts parsed from frames.jsonl.
        Speeds are computed here via 3-frame centered derivative.
        Sends BATCH_SIZE rows per reducer call.
        """
        rows = sorted(frame_rows, key=lambda r: r["frame_index"])
        n = len(rows)

        def _p_xy(row, track_id):
            for p in row.get("players") or []:
                if p.get("track_id") == track_id:
                    cxy = p.get("court_xy_m")
                    if cxy and len(cxy) >= 2:
                        x, y = float(cxy[0]), float(cxy[1])
                        if isfinite(x) and isfinite(y):
                            return x, y
            return 0.0, 0.0

        def _sh_xy(row):
            sh = row.get("shuttle") or {}
            if sh.get("visible"):
                cp = sh.get("court_projection_m")
                if cp and len(cp) >= 2:
                    x, y = float(cp[0]), float(cp[1])
                    if isfinite(x) and isfinite(y):
                        return x, y, True
            return 0.0, 0.0, False

        def _speed(prev_row, cur_row, next_row, pos_fn):
            if (prev_row["frame_index"] + 1 != cur_row["frame_index"] or
                    cur_row["frame_index"] + 1 != next_row["frame_index"]):
                return 0.0
            p0, p1, p2 = pos_fn(prev_row), pos_fn(cur_row), pos_fn(next_row)
            if p0 == (0.0, 0.0) or p2 == (0.0, 0.0):
                return 0.0
            dt_b = cur_row["timestamp_s"] - prev_row["timestamp_s"]
            dt_a = next_row["timestamp_s"] - cur_row["timestamp_s"]
            span = dt_b + dt_a
            if dt_b <= 0 or dt_a <= 0:
                return 0.0
            vx = (dt_a / span) * (p1[0] - p0[0]) / dt_b + (dt_b / span) * (p2[0] - p1[0]) / dt_a
            vy = (dt_a / span) * (p1[1] - p0[1]) / dt_b + (dt_b / span) * (p2[1] - p1[1]) / dt_a
            s = hypot(vx, vy)
            return s if isfinite(s) else 0.0

        # Build column arrays
        frame_indices, timestamps = [], []
        p1xs, p1ys, p2xs, p2ys = [], [], [], []
        sh_xs, sh_ys, sh_vis = [], [], []
        p1speeds, p2speeds, sh_speeds = [], [], []

        for i, row in enumerate(rows):
            frame_indices.append(int(row["frame_index"]))
            timestamps.append(float(row["timestamp_s"]))
            p1x, p1y = _p_xy(row, 1)
            p2x, p2y = _p_xy(row, 2)
            sx, sy, sv = _sh_xy(row)
            p1xs.append(p1x); p1ys.append(p1y)
            p2xs.append(p2x); p2ys.append(p2y)
            sh_xs.append(sx); sh_ys.append(sy); sh_vis.append(sv)

            if 0 < i < n - 1:
                prev, nxt = rows[i - 1], rows[i + 1]
                p1speeds.append(_speed(prev, row, nxt, lambda r: _p_xy(r, 1)))
                p2speeds.append(_speed(prev, row, nxt, lambda r: _p_xy(r, 2)))
                sh_speeds.append(_speed(prev, row, nxt, lambda r: _sh_xy(r)[:2]))
            else:
                p1speeds.append(0.0); p2speeds.append(0.0); sh_speeds.append(0.0)

        # Send in batches
        for start in range(0, n, BATCH_SIZE):
            sl = slice(start, start + BATCH_SIZE)
            self._post("add_running_details_batch", [
                run_id,
                frame_indices[sl], timestamps[sl],
                p1xs[sl], p1ys[sl],
                p2xs[sl], p2ys[sl],
                sh_xs[sl], sh_ys[sl], sh_vis[sl],
                p1speeds[sl], p2speeds[sl], sh_speeds[sl],
            ])

    # ── Push: shot_class (M4 hits + M5 shot types combined) ───────────────────

    def push_shot_classes(self, run_id: str, hits: list[dict], shots: list[dict]):
        """
        hits: list of HitEvent dicts (from hits.jsonl)
        shots: list of ShotEvent dicts (from shots.jsonl); matched by start_frame == hit frame_index
        """
        # Build frame→shot lookup
        shot_by_frame = {s["start_frame"]: s for s in shots}

        for hit in hits:
            fi = hit["frame_index"]
            shot = shot_by_frame.get(fi, {})
            self._post("add_shot_class", [
                run_id,
                int(hit["event_id"]),
                int(fi),
                float(hit["timestamp_s"]),
                int(hit.get("hitter_id") or 0),
                float(hit.get("hit_score", 0)),
                shot.get("shot_type_en", ""),
                float(shot.get("confidence", 0)),
                shot.get("player_side", ""),
            ])

    # ── Push: rallies (M7) ────────────────────────────────────────────────────

    def clear_rallies(self, run_id: str):
        self._post("clear_rallies", [run_id])

    def push_rallies(self, run_id: str, rallies: list[dict]):
        for rally in rallies:
            landing = rally.get("landing_xy_m") or (0.0, 0.0)
            self._post("add_rally", [
                run_id,
                int(rally["rally_index"]),
                int(rally["game"]),
                int(rally["start_frame"]),
                int(rally["end_frame"]),
                float(rally["start_s"]),
                float(rally["end_s"]),
                int(rally["shot_count"]),
                int(rally["first_event_id"]),
                int(rally["last_event_id"]),
                int(rally["last_hitter_id"]),
                int(rally["winner_id"]),
                rally["outcome"],
                float(landing[0]),
                float(landing[1]),
                int(rally["score_after"]["1"]),
                int(rally["score_after"]["2"]),
                bool(rally["score_reliable"]),
            ])

    # ── Push: summary (M7 analytics) ─────────────────────────────────────────

    def push_summary(
        self, run_id: str, frame_rows: list[dict], hits: list[dict], shots: list[dict]
    ):
        from module7_summary.analytics import (
            compute_player_distance, compute_avg_shuttle_speed,
            compute_shot_distribution, _load_speed_csv,
        )
        # Compute stats inline
        distances = compute_player_distance(frame_rows)
        shot_dist = compute_shot_distribution(shots)

        # Build speed lookup from inline computation (re-use push_running_details logic)
        # Shuttle speed at hit frames
        rows_sorted = sorted(frame_rows, key=lambda r: r["frame_index"])
        frame_by_fi = {r["frame_index"]: r for r in rows_sorted}

        def _sh_speed_at(fi):
            idxs = [fi - 1, fi, fi + 1]
            if not all(i in frame_by_fi for i in idxs):
                return None
            prev, cur, nxt = frame_by_fi[fi - 1], frame_by_fi[fi], frame_by_fi[fi + 1]
            if (prev["frame_index"] + 1 != cur["frame_index"] or
                    cur["frame_index"] + 1 != nxt["frame_index"]):
                return None
            def _sh_xy(r):
                sh = r.get("shuttle") or {}
                if sh.get("visible"):
                    cp = sh.get("court_projection_m")
                    if cp and len(cp) >= 2:
                        return float(cp[0]), float(cp[1])
                return None
            p0, p1, p2 = _sh_xy(prev), _sh_xy(cur), _sh_xy(nxt)
            if p0 is None or p2 is None:
                return None
            dt_b = cur["timestamp_s"] - prev["timestamp_s"]
            dt_a = nxt["timestamp_s"] - cur["timestamp_s"]
            if dt_b <= 0 or dt_a <= 0:
                return None
            span = dt_b + dt_a
            vx = (dt_a / span) * (p1[0] - p0[0]) / dt_b + (dt_b / span) * (p2[0] - p1[0]) / dt_a
            vy = (dt_a / span) * (p1[1] - p0[1]) / dt_b + (dt_b / span) * (p2[1] - p1[1]) / dt_a
            s = hypot(vx, vy)
            return s if isfinite(s) else None

        shot_by_frame = {s["start_frame"]: s for s in shots}
        speed_sums: dict[int, float] = {}
        speed_counts: dict[int, int] = {}
        for hit in hits:
            fi = hit["frame_index"]
            hid = hit.get("hitter_id")
            if hid is None:
                continue
            v = _sh_speed_at(fi)
            if v is not None:
                speed_sums[hid] = speed_sums.get(hid, 0.0) + v
                speed_counts[hid] = speed_counts.get(hid, 0) + 1
        avg_speeds = {tid: speed_sums[tid] / speed_counts[tid]
                      for tid in speed_sums if speed_counts[tid] > 0}
        overall_avg = sum(avg_speeds.values()) / len(avg_speeds) if avg_speeds else 0.0

        # Heatmaps (every 5th frame, normalised)
        SAMPLE = 5
        heatmaps: dict[int, list] = {1: [], 2: []}
        for i, row in enumerate(rows_sorted):
            if i % SAMPLE != 0:
                continue
            for p in row.get("players") or []:
                tid = p.get("track_id")
                cxy = p.get("court_xy_m")
                if tid not in heatmaps or not cxy or len(cxy) < 2:
                    continue
                nx = max(0.0, min(1.0, float(cxy[0]) / COURT_W))
                ny = max(0.0, min(1.0, float(cxy[1]) / COURT_L))
                if isfinite(nx) and isfinite(ny):
                    heatmaps[tid].append([round(nx, 4), round(ny, 4)])

        self._post("set_summary", [
            run_id,
            float(distances.get(1, 0)),
            float(distances.get(2, 0)),
            float(overall_avg),
            json.dumps({k: sum(v.values()) for k, v in shot_dist.items()}),
            json.dumps(heatmaps.get(1, [])),
            json.dumps(heatmaps.get(2, [])),
        ])

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def fetch_running_details(self, run_id: str) -> list[dict]:
        return self._sql(
            f"SELECT * FROM running_details WHERE run_id = '{run_id}' ORDER BY frame_index ASC"
        )

    def fetch_shot_classes(self, run_id: str) -> list[dict]:
        return self._sql(
            f"SELECT * FROM shot_class WHERE run_id = '{run_id}' ORDER BY frame_index ASC"
        )
