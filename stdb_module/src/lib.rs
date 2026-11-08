use spacetimedb::{reducer, table, ReducerContext, Table};

// ── Tables ────────────────────────────────────────────────────────────────────

/// One analysis run (maps to a run directory).
#[table(name = run_meta, public)]
pub struct RunMeta {
    #[primary_key]
    pub run_id: String,
    pub video: String,
    pub total_frames: u32,
    pub fps: f32,
    pub created_at: u64, // Unix ms
}

/// A classified shot event.
#[table(name = shot_record, public)]
pub struct ShotRecord {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub run_id: String,
    pub shot_id: u32,
    pub hitter_id: u32,
    pub player_side: String,
    pub shot_type: String,
    pub confidence: f32,
    pub start_frame: u32,
    pub end_frame: u32,
}

/// Per-frame shuttle and player speeds (sampled: every Nth frame to stay lean).
#[table(name = frame_speed, public)]
pub struct FrameSpeed {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub run_id: String,
    pub frame_index: u32,
    pub timestamp_s: f32,
    pub shuttle_speed_px_s: f32,  // NaN → 0 (not visible)
    pub shuttle_speed_m_s: f32,
    pub player1_speed_m_s: f32,
    pub player2_speed_m_s: f32,
}

/// Player court positions for heatmap (normalised 0–1 per court dimension).
#[table(name = player_position, public)]
pub struct PlayerPosition {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub run_id: String,
    pub track_id: u32,
    pub frame_index: u32,
    pub norm_x: f32, // 0 = left edge, 1 = right edge
    pub norm_y: f32, // 0 = top baseline, 1 = bottom baseline
}

/// Aggregate stats per player (pre-computed by the Python uploader).
#[table(name = player_stat, public)]
pub struct PlayerStat {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub run_id: String,
    pub track_id: u32,
    pub total_distance_m: f32,
    pub avg_shuttle_speed_m_s: f32,
    pub total_shots: u32,
}

// ── Reducers ──────────────────────────────────────────────────────────────────

#[reducer]
pub fn upsert_run(
    _ctx: &ReducerContext,
    run_id: String,
    video: String,
    total_frames: u32,
    fps: f32,
    created_at: u64,
) {
    RunMeta::insert(RunMeta { run_id, video, total_frames, fps, created_at });
}

#[reducer]
pub fn add_shot(
    _ctx: &ReducerContext,
    run_id: String,
    shot_id: u32,
    hitter_id: u32,
    player_side: String,
    shot_type: String,
    confidence: f32,
    start_frame: u32,
    end_frame: u32,
) {
    ShotRecord::insert(ShotRecord {
        id: 0,
        run_id,
        shot_id,
        hitter_id,
        player_side,
        shot_type,
        confidence,
        start_frame,
        end_frame,
    });
}

#[reducer]
pub fn add_frame_speed(
    _ctx: &ReducerContext,
    run_id: String,
    frame_index: u32,
    timestamp_s: f32,
    shuttle_speed_px_s: f32,
    shuttle_speed_m_s: f32,
    player1_speed_m_s: f32,
    player2_speed_m_s: f32,
) {
    FrameSpeed::insert(FrameSpeed {
        id: 0,
        run_id,
        frame_index,
        timestamp_s,
        shuttle_speed_px_s,
        shuttle_speed_m_s,
        player1_speed_m_s,
        player2_speed_m_s,
    });
}

#[reducer]
pub fn add_player_positions_batch(
    _ctx: &ReducerContext,
    run_id: String,
    track_id: u32,
    frame_indices: Vec<u32>,
    norm_xs: Vec<f32>,
    norm_ys: Vec<f32>,
) {
    for ((fi, nx), ny) in frame_indices
        .into_iter()
        .zip(norm_xs)
        .zip(norm_ys)
    {
        PlayerPosition::insert(PlayerPosition {
            id: 0,
            run_id: run_id.clone(),
            track_id,
            frame_index: fi,
            norm_x: nx,
            norm_y: ny,
        });
    }
}

#[reducer]
pub fn set_player_stat(
    _ctx: &ReducerContext,
    run_id: String,
    track_id: u32,
    total_distance_m: f32,
    avg_shuttle_speed_m_s: f32,
    total_shots: u32,
) {
    PlayerStat::insert(PlayerStat {
        id: 0,
        run_id,
        track_id,
        total_distance_m,
        avg_shuttle_speed_m_s,
        total_shots,
    });
}

#[reducer]
pub fn clear_run(_ctx: &ReducerContext, run_id: String) {
    // Delete all data for a run_id so it can be re-uploaded cleanly.
    for row in ShotRecord::iter().filter(|r| r.run_id == run_id) {
        ShotRecord::delete_by_id(&row.id);
    }
    for row in FrameSpeed::iter().filter(|r| r.run_id == run_id) {
        FrameSpeed::delete_by_id(&row.id);
    }
    for row in PlayerPosition::iter().filter(|r| r.run_id == run_id) {
        PlayerPosition::delete_by_id(&row.id);
    }
    for row in PlayerStat::iter().filter(|r| r.run_id == run_id) {
        PlayerStat::delete_by_id(&row.id);
    }
    for row in RunMeta::iter().filter(|r| r.run_id == run_id) {
        RunMeta::delete_by_run_id(&row.run_id);
    }
}
