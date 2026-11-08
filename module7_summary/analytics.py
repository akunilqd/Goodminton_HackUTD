"""Module 7 – match summary analytics and HTML report generation.

Reads frames.jsonl, shots.jsonl, and speed.csv from a run directory and
produces a single self-contained HTML report containing:
  - Total distance travelled per player (metres)
  - Average shuttlecock speed at each player's hits (m/s)
  - Pie chart of shot-type distribution per player
  - Position heatmap per player (court coordinates)
"""

import csv
import json
from math import hypot
from pathlib import Path

COURT_W = 6.10   # metres, full width
COURT_L = 13.40  # metres, full length

SHOT_COLORS = {
    "Smash": "#ef4444",
    "Clear": "#3b82f6",
    "Drive": "#f59e0b",
    "Net":   "#10b981",
    "Serve": "#8b5cf6",
}


def _load_frames(run_dir: Path) -> list[dict]:
    rows = []
    with (run_dir / "frames.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r["frame_index"])


def _load_shots(run_dir: Path) -> list[dict]:
    rows = []
    path = run_dir / "shots.jsonl"
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_speed_csv(run_dir: Path) -> dict[int, dict]:
    """Return {frame_index: row_dict} from speed.csv."""
    result: dict[int, dict] = {}
    path = run_dir / "speed.csv"
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fi = int(row["frame_index"])
            result[fi] = row
    return result


def compute_player_distance(frames: list[dict]) -> dict[int, float]:
    """Sum Euclidean court distances between consecutive frames per player."""
    last_pos: dict[int, tuple[float, float]] = {}
    totals: dict[int, float] = {}
    for row in frames:
        for p in row.get("players") or []:
            tid = p.get("track_id")
            cxy = p.get("court_xy_m")
            if tid is None or not cxy or len(cxy) < 2:
                continue
            x, y = float(cxy[0]), float(cxy[1])
            if tid in last_pos:
                lx, ly = last_pos[tid]
                totals[tid] = totals.get(tid, 0.0) + hypot(x - lx, y - ly)
            last_pos[tid] = (x, y)
            if tid not in totals:
                totals[tid] = 0.0
    return totals


def compute_avg_shuttle_speed(
    shots: list[dict], speed_by_frame: dict[int, dict]
) -> dict[int, float]:
    """Average shuttle_speed_m_s at each shot's start_frame, grouped by hitter."""
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for shot in shots:
        hid = shot.get("hitter_id")
        if hid is None:
            continue
        frame = shot.get("start_frame")
        row = speed_by_frame.get(frame)
        if row is None:
            continue
        val = row.get("shuttle_speed_m_s", "")
        if val == "" or val is None:
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        sums[hid] = sums.get(hid, 0.0) + v
        counts[hid] = counts.get(hid, 0) + 1
    return {tid: sums[tid] / counts[tid] for tid in sums if counts[tid] > 0}


def compute_shot_distribution(shots: list[dict]) -> dict[int, dict[str, int]]:
    """Count shot types per player: {player_id: {shot_type: count}}."""
    dist: dict[int, dict[str, int]] = {}
    for shot in shots:
        hid = shot.get("hitter_id")
        stype = shot.get("shot_type_en", "Unknown")
        if hid is None:
            continue
        dist.setdefault(hid, {})
        dist[hid][stype] = dist[hid].get(stype, 0) + 1
    return dist


def compute_player_heatmap_positions(
    frames: list[dict],
) -> dict[int, list[tuple[float, float]]]:
    """Collect normalised [0,1] court positions per player."""
    positions: dict[int, list[tuple[float, float]]] = {}
    for row in frames:
        for p in row.get("players") or []:
            tid = p.get("track_id")
            cxy = p.get("court_xy_m")
            if tid is None or not cxy or len(cxy) < 2:
                continue
            nx = max(0.0, min(1.0, float(cxy[0]) / COURT_W))
            ny = max(0.0, min(1.0, float(cxy[1]) / COURT_L))
            positions.setdefault(tid, []).append((nx, ny))
    return positions


def _html_report(
    player_ids: list[int],
    distances: dict[int, float],
    avg_speeds: dict[int, float],
    shot_dist: dict[int, dict[str, int]],
    heatmap_pos: dict[int, list[tuple[float, float]]],
    metadata: dict,
) -> str:
    video_info = metadata.get("video", {})
    if isinstance(video_info, dict):
        video_name = Path(str(video_info.get("source", ""))).name
        fps = video_info.get("fps", "?")
    else:
        video_name = str(video_info)
        fps = metadata.get("fps", "?")
    total_frames = metadata.get("processed_frames", metadata.get("total_frames", "?"))

    player_colors = {player_ids[0]: "#3b82f6", player_ids[1]: "#f97316"} if len(player_ids) >= 2 else {}

    def player_label(tid):
        side = "Top" if tid == player_ids[0] else "Bottom"
        return f"Player {tid} ({side})"

    # Build JSON data for JS
    pie_data = {}
    for tid in player_ids:
        d = shot_dist.get(tid, {})
        pie_data[tid] = {
            "labels": list(d.keys()),
            "values": list(d.values()),
            "colors": [SHOT_COLORS.get(k, "#94a3b8") for k in d],
        }

    heatmap_json = {
        str(tid): heatmap_pos.get(tid, [])
        for tid in player_ids
    }

    summary_rows = ""
    for tid in player_ids:
        dist = distances.get(tid, 0.0)
        avg_v = avg_speeds.get(tid)
        avg_v_str = f"{avg_v:.1f} m/s" if avg_v is not None else "—"
        shots_total = sum(shot_dist.get(tid, {}).values())
        pc = player_colors.get(tid, "#94a3b8")
        summary_rows += f"""
        <tr>
          <td><span class="dot" style="background:{pc}"></span>{player_label(tid)}</td>
          <td>{dist:.1f} m</td>
          <td>{avg_v_str}</td>
          <td>{shots_total}</td>
        </tr>"""

    pie_canvases = ""
    for tid in player_ids:
        pie_canvases += f'<div class="chart-box"><h3>{player_label(tid)}</h3><canvas id="pie{tid}" width="260" height="260"></canvas></div>\n'

    heatmap_canvases = ""
    for tid in player_ids:
        heatmap_canvases += f'<div class="chart-box"><h3>{player_label(tid)}</h3><canvas id="hm{tid}" width="244" height="536" style="border:1px solid #334155;border-radius:4px"></canvas></div>\n'

    pie_js = ""
    for tid in player_ids:
        d = pie_data[tid]
        labels_json = json.dumps(d["labels"])
        values_json = json.dumps(d["values"])
        colors_json = json.dumps(d["colors"])
        pie_js += f"""
      new Chart(document.getElementById('pie{tid}'), {{
        type: 'doughnut',
        data: {{
          labels: {labels_json},
          datasets: [{{ data: {values_json}, backgroundColor: {colors_json}, borderWidth: 2, borderColor: '#1e293b' }}]
        }},
        options: {{
          plugins: {{
            legend: {{ labels: {{ color: '#cbd5e1', font: {{ size: 13 }} }} }},
            tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.label}}: ${{ctx.parsed}}` }} }}
          }}
        }}
      }});"""

    heatmap_js = f"""
      const hmData = {json.dumps(heatmap_json)};
      const COURT_W_M = {COURT_W};
      const COURT_L_M = {COURT_L};

      function drawHeatmap(canvasId, points) {{
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        // Court background
        ctx.fillStyle = '#0f4c2a';
        ctx.fillRect(0, 0, W, H);

        // Court lines
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        const pad = 12;
        const cw = W - pad * 2, ch = H - pad * 2;

        // Outer boundary
        ctx.strokeRect(pad, pad, cw, ch);

        // Net (midline)
        ctx.beginPath(); ctx.moveTo(pad, pad + ch / 2); ctx.lineTo(pad + cw, pad + ch / 2); ctx.stroke();

        // Service lines (1.98m from net = 29.5% of half)
        const svc = ch / 2 * 0.295;
        ctx.beginPath(); ctx.moveTo(pad, pad + ch / 2 - svc); ctx.lineTo(pad + cw, pad + ch / 2 - svc); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad, pad + ch / 2 + svc); ctx.lineTo(pad + cw, pad + ch / 2 + svc); ctx.stroke();

        // Centre service line
        ctx.beginPath(); ctx.moveTo(pad + cw / 2, pad + ch / 2 - svc); ctx.lineTo(pad + cw / 2, pad + ch / 2 + svc); ctx.stroke();

        // Back service lines (0.76m from back = 11.3% of half)
        const back = ch / 2 * 0.113;
        ctx.beginPath(); ctx.moveTo(pad, pad + back); ctx.lineTo(pad + cw, pad + back); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad, pad + ch - back); ctx.lineTo(pad + cw, pad + ch - back); ctx.stroke();

        if (!points || points.length === 0) return;

        // Density grid
        const GW = 24, GH = 54;
        const grid = new Float32Array(GW * GH);
        let maxVal = 0;
        for (const [nx, ny] of points) {{
          const gx = Math.min(GW - 1, Math.floor(nx * GW));
          const gy = Math.min(GH - 1, Math.floor(ny * GH));
          grid[gy * GW + gx]++;
          if (grid[gy * GW + gx] > maxVal) maxVal = grid[gy * GW + gx];
        }}
        if (maxVal === 0) return;

        const cellW = cw / GW, cellH = ch / GH;
        for (let gy = 0; gy < GH; gy++) {{
          for (let gx = 0; gx < GW; gx++) {{
            const v = grid[gy * GW + gx] / maxVal;
            if (v === 0) continue;
            const alpha = Math.pow(v, 0.5) * 0.85;
            const r = Math.round(255 * Math.min(1, v * 2));
            const g = Math.round(255 * Math.max(0, 1 - Math.abs(v - 0.5) * 2));
            const b = Math.round(255 * Math.max(0, 1 - v * 2));
            ctx.fillStyle = `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
            ctx.fillRect(pad + gx * cellW, pad + gy * cellH, cellW + 0.5, cellH + 0.5);
          }}
        }}
      }}
"""

    for tid in player_ids:
        heatmap_js += f"      drawHeatmap('hm{tid}', hmData['{tid}']);\n"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Match Summary</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #f1f5f9; --muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; padding: 32px 24px; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }}
  .meta {{ color: var(--muted); font-size: .85rem; margin-bottom: 32px; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
  h3 {{ font-size: .95rem; font-weight: 600; margin-bottom: 12px; color: var(--text); }}
  section {{ margin-bottom: 48px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: .9rem; }}
  th {{ color: var(--muted); font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }}
  tr:last-child td {{ border-bottom: none; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }}
  .cards {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .chart-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px; display: flex; flex-direction: column; align-items: center; }}
  .table-wrap {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
</style>
</head>
<body>
<h1>Match Summary</h1>
<p class="meta">{video_name} &nbsp;·&nbsp; {total_frames} frames &nbsp;·&nbsp; {fps} fps</p>

<section>
  <h2>Overview</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Player</th><th>Distance</th><th>Avg Shuttle Speed</th><th>Shots</th></tr></thead>
      <tbody>{summary_rows}</tbody>
    </table>
  </div>
</section>

<section>
  <h2>Shot Distribution</h2>
  <div class="cards">{pie_canvases}</div>
</section>

<section>
  <h2>Position Heatmap</h2>
  <div class="cards">{heatmap_canvases}</div>
</section>

<script>
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = '#334155';
  (function () {{
    {pie_js}
    {heatmap_js}
  }})();
</script>
</body>
</html>"""


def generate_report(run_dir: Path, output: Path) -> dict:
    """Compute all analytics from run_dir and write HTML report to output."""
    run_dir = Path(run_dir).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    frames = _load_frames(run_dir)
    shots = _load_shots(run_dir)
    speed_by_frame = _load_speed_csv(run_dir)

    meta_path = run_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    distances = compute_player_distance(frames)
    avg_speeds = compute_avg_shuttle_speed(shots, speed_by_frame)
    shot_dist = compute_shot_distribution(shots)
    heatmap_pos = compute_player_heatmap_positions(frames)

    # Stable player order (by track_id)
    player_ids = sorted(distances.keys() | shot_dist.keys() | heatmap_pos.keys())

    html = _html_report(player_ids, distances, avg_speeds, shot_dist, heatmap_pos, metadata)
    output.write_text(html, encoding="utf-8")

    return {
        "output": str(output),
        "players": player_ids,
        "distances_m": {str(k): round(v, 2) for k, v in distances.items()},
        "avg_shuttle_speed_m_s": {str(k): round(v, 2) for k, v in avg_speeds.items()},
        "shot_counts": {str(k): dict(v) for k, v in shot_dist.items()},
    }
