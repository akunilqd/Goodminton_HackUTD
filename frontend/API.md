# Goodminton backend API

This repo has no HTTP server of its own. `main.py` is a command-line pipeline that analyses a video and publishes the results to **SpacetimeDB**. The frontend reads from SpacetimeDB directly.

| | |
| --- | --- |
| Base URL | `https://maincloud.spacetimedb.com` (local: `http://localhost:3000` after `spacetime start`) |
| Database | `badminton` |
| Frontend config | `assets/js/api-config.js` |
| Frontend client | `assets/js/api.js` (`GM.api`), UI in `assets/js/runs.js` |
| CORS | `Access-Control-Allow-Origin: *`, so `file://` pages work too |

## Endpoints

| Method and path | Used by | Auth | Body |
| --- | --- | --- | --- |
| `POST /v1/database/badminton/sql` | Frontend (read) | None for public tables | SQL as `text/plain` |
| `POST /v1/database/badminton/call/{reducer}` | Pipeline only (write) | `Authorization: Bearer <token>` | JSON array of positional arguments |

The token comes from the SpacetimeDB CLI config on the analysis machine. It must never reach the frontend.

**SQL response**

```json
[{ "schema": { "elements": [{ "name": { "some": "run_id" }, "algebraic_type": { "String": [] } }] },
   "rows": [["my_run"]], "total_duration_micros": 164, "stats": {} }]
```

**SQL limits, confirmed against the live server:** `ORDER BY` returns 400, so sort on the client. `LIMIT` works; there is no offset, so page by `frame_index` range. Column names are snake_case with digits split out (`p1X` becomes `p_1_x`).

**Errors are plain text, not JSON:** 400 for an unknown table or column or unsupported SQL, 404 for an unknown database.

## Tables

All four are public. A `player_id` of 1 or 2 is a track ID. Court coordinates are metres, with x from 0 to 6.1 and y from 0 (far baseline) to 13.4 (near baseline).

**`run_meta`**, one row per run: `run_id` string (primary key), `video_source` string (absolute path on the analysis machine), `fps` f32, `total_frames` u32, `created_at` u64 (ms since epoch).

**`shot_class`**, one row per hit: `id` u64, `run_id` string, `event_id` u32, `frame_index` u32, `timestamp_s` f32, `player_id` u32 (0 means unknown), `hit_score` f32 (heuristic, not a probability), `hit_type` string (`Smash`, `Clear`, `Drop`, `Drive`, `Net`, `Serve`, or empty if unclassified), `confidence` f32, `player_side` string (`Top` is the far end, `Bottom` the near end, or empty).

**`running_details`**, one row per frame: `id` u64, `run_id` string, `frame_index` u32, `timestamp_s` f32, `p_1_x`, `p_1_y`, `p_2_x`, `p_2_y`, `shuttle_x`, `shuttle_y` f32 (`0, 0` means missing), `shuttle_visible` bool, `p_1_speed`, `p_2_speed`, `shuttle_speed` f32 in m/s (0 means not computable).

**`summary`**, one row per finished run: `run_id` string (primary key), `p_1_total_distance_m`, `p_2_total_distance_m`, `avg_shuttle_speed_ms` f32, `shot_counts_json` string (actual shape `{"1": 6, "2": 7}`, hits per player), `heatmap_p_1_json`, `heatmap_p_2_json` string (`[[x, y], …]` normalised 0 to 1).

**`rally`**, one row per rally (needs the module re-published): `id` u64, `run_id` string, `rally_index`, `game`, `start_frame`, `end_frame` u32, `start_s`, `end_s` f32, `shot_count`, `first_event_id`, `last_event_id`, `last_hitter_id`, `winner_id` u32 (0 means no result), `outcome` string (`winner`, `net`, `out`, `unknown`), `landing_x`, `landing_y` f32 (`0, 0` means none seen), `score_p_1`, `score_p_2` u32 (after this rally), `score_reliable` bool (false once any earlier rally had no result).

## Reducers (pipeline writes)

`upsert_run_meta`, `add_running_details_batch`, `add_shot_class`, `set_summary`, `add_rally`, `clear_rallies`, `clear_run`. Definitions are in `../rallymirror-backend/spacetimedb/src/index.ts`.

## Upload and analysis service (`server.py`)

Runs on the analysis computer at `http://localhost:8000` and also serves this website. It only listens on that computer by default, and it has no user accounts. Writes need the `X-Goodminton: 1` header and must come from the same site.

| Method and path | Purpose |
| --- | --- |
| `POST /api/jobs?filename=&near=&far=` | Upload one MP4 or MOV as the raw request body (up to 4 GB). Returns the job, status 201 |
| `GET /api/jobs/{id}` | `status` (`queued`, `running`, `complete`, `failed`, `cancelled`), `stage`, `filename`, `near`, `far`, `error`, `warnings`, `rallies`, `video_url` |
| `POST /api/jobs/{id}/cancel` | Stop the analysis and delete the video |
| `GET /api/videos/{id}` | The original video, with seeking |
| `GET /api/health` | Liveness check |

The job ID is also the run ID in SpacetimeDB. Jobs run one at a time. Videos and logs stay in `jobs/`.

## Client error codes

| Code | Cause | Retried automatically | "Try again" shown |
| --- | --- | --- | --- |
| `timeout` | No complete response in 15 s | Yes | Yes |
| `network` | Request failed | Yes | Yes |
| `server` | HTTP 5xx | Yes | Yes |
| `rate_limited` | HTTP 429, honours `Retry-After` | Yes | Yes |
| `offline` | Browser is offline; recovers on reconnect | No | Yes |
| `malformed` | Response isn't the expected JSON | No | Yes |
| `contract` | HTTP 400 or a missing column | No | No |
| `auth` | HTTP 401 or 403 | No | No |
| `db_not_found` | HTTP 404 | No | No |
| `run_not_found` | No such run | No | No |
| `invalid_id` | Bad run ID in the link | No | No |
| `no_rallies` | The run has no rally with a result | No | No |
