import { schema, table, t, SenderError } from 'spacetimedb/server';

// ── Tables ─────────────────────────────────────────────────────────────────

/**
 * Per-video run metadata.
 * One row per video analysed.
 */
const runMeta = table(
  { name: 'run_meta', public: true },
  {
    runId: t.string().primaryKey(),
    videoSource: t.string(),
    fps: t.f32(),
    totalFrames: t.u32(),
    createdAt: t.u64(),
    nearName: t.string().default(""),
    farName: t.string().default(""),
  }
);

/**
 * Per-frame tracking data (M1 + M2 + M3 + M6 speeds).
 * One row per frame per video.
 */
const runningDetails = table(
  { name: 'running_details', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    runId: t.string().index('btree'),
    frameIndex: t.u32(),
    timestampS: t.f32(),
    // Player positions after homography (court coords in metres)
    p1X: t.f32(),
    p1Y: t.f32(),
    p2X: t.f32(),
    p2Y: t.f32(),
    // Shuttle position after homography (court coords in metres; 0 if not visible)
    shuttleX: t.f32(),
    shuttleY: t.f32(),
    shuttleVisible: t.bool(),
    // Speeds in m/s (0 if not computable)
    p1Speed: t.f32(),
    p2Speed: t.f32(),
    shuttleSpeed: t.f32(),
  }
);

/**
 * Shot/hit classification (M4 hit_score + M5 hit_type combined).
 * One row per detected hit.
 */
const shotClass = table(
  { name: 'shot_class', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    runId: t.string().index('btree'),
    eventId: t.u32(),
    frameIndex: t.u32(),
    timestampS: t.f32(),
    playerId: t.u32(),
    hitScore: t.f32(),
    hitType: t.string(),   // e.g. "Smash", "Clear", "Drive", "Net", "Serve"
    confidence: t.f32(),
    playerSide: t.string(), // "Top" or "Bottom"
  }
);

/**
 * Match summary analytics (M7).
 * One row per video run.
 */
const summary = table(
  { name: 'summary', public: true },
  {
    runId: t.string().primaryKey(),
    p1TotalDistanceM: t.f32(),
    p2TotalDistanceM: t.f32(),
    avgShuttleSpeedMS: t.f32(),
    shotCountsJson: t.string(),  // JSON: {"Smash":3,"Clear":2,...}
    heatmapP1Json: t.string(),   // JSON: [[normX,normY],...] sampled positions
    heatmapP2Json: t.string(),
  }
);

/**
 * One row per rally (M7). winnerId 0 and outcome "unknown" mean no landing was
 * seen, so no point was awarded. landingX/Y are 0 when there was no landing.
 */
const rally = table(
  { name: 'rally', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    runId: t.string().index('btree'),
    rallyIndex: t.u32(),
    game: t.u32(),
    startFrame: t.u32(),
    endFrame: t.u32(),
    startS: t.f32(),
    endS: t.f32(),
    shotCount: t.u32(),
    firstEventId: t.u32(),
    lastEventId: t.u32(),
    lastHitterId: t.u32(),
    winnerId: t.u32(),
    outcome: t.string(),   // "winner", "net", "out" or "unknown"
    landingX: t.f32(),
    landingY: t.f32(),
    scoreP1: t.u32(),      // Score after this rally.
    scoreP2: t.u32(),
    scoreReliable: t.bool(),
  }
);

/**
 * The analysed video (M5 overlay), split into chunks by the pipeline and
 * reassembled by appendAnalyzedVideo. One row per run.
 */
const analyzedVideo = table(
  { name: 'analyzed_video', public: true },
  {
    runId: t.string().primaryKey(),
    video: t.byteArray(),
  }
);

// ── Schema ─────────────────────────────────────────────────────────────────

const spacetimedb = schema({ runMeta, runningDetails, shotClass, summary, rally, analyzedVideo });
export default spacetimedb;

// ── Reducers ───────────────────────────────────────────────────────────────

export const upsertRunMeta = spacetimedb.reducer(
  { runId: t.string(), videoSource: t.string(), fps: t.f32(), totalFrames: t.u32(), createdAt: t.u64(), nearName: t.string(), farName: t.string() },
  (ctx, { runId, videoSource, fps, totalFrames, createdAt, nearName, farName }) => {
    if (ctx.db.runMeta.runId.find(runId)) {
      ctx.db.runMeta.runId.update({ runId, videoSource, fps, totalFrames, createdAt, nearName, farName });
    } else {
      ctx.db.runMeta.insert({ runId, videoSource, fps, totalFrames, createdAt, nearName, farName });
    }
  }
);

export const addRunningDetailsBatch = spacetimedb.reducer(
  {
    runId: t.string(),
    frameIndices: t.array(t.u32()),
    timestampsS: t.array(t.f32()),
    p1Xs: t.array(t.f32()),
    p1Ys: t.array(t.f32()),
    p2Xs: t.array(t.f32()),
    p2Ys: t.array(t.f32()),
    shuttleXs: t.array(t.f32()),
    shuttleYs: t.array(t.f32()),
    shuttleVisibles: t.array(t.bool()),
    p1Speeds: t.array(t.f32()),
    p2Speeds: t.array(t.f32()),
    shuttleSpeeds: t.array(t.f32()),
  },
  (ctx, { runId, frameIndices, timestampsS, p1Xs, p1Ys, p2Xs, p2Ys,
           shuttleXs, shuttleYs, shuttleVisibles, p1Speeds, p2Speeds, shuttleSpeeds }) => {
    for (let i = 0; i < frameIndices.length; i++) {
      ctx.db.runningDetails.insert({
        id: 0n,
        runId,
        frameIndex: frameIndices[i],
        timestampS: timestampsS[i],
        p1X: p1Xs[i], p1Y: p1Ys[i],
        p2X: p2Xs[i], p2Y: p2Ys[i],
        shuttleX: shuttleXs[i], shuttleY: shuttleYs[i],
        shuttleVisible: shuttleVisibles[i],
        p1Speed: p1Speeds[i], p2Speed: p2Speeds[i], shuttleSpeed: shuttleSpeeds[i],
      });
    }
  }
);

export const addShotClass = spacetimedb.reducer(
  {
    runId: t.string(), eventId: t.u32(), frameIndex: t.u32(), timestampS: t.f32(),
    playerId: t.u32(), hitScore: t.f32(), hitType: t.string(),
    confidence: t.f32(), playerSide: t.string(),
  },
  (ctx, args) => {
    ctx.db.shotClass.insert({ id: 0n, ...args });
  }
);

export const setSummary = spacetimedb.reducer(
  {
    runId: t.string(),
    p1TotalDistanceM: t.f32(), p2TotalDistanceM: t.f32(),
    avgShuttleSpeedMS: t.f32(),
    shotCountsJson: t.string(),
    heatmapP1Json: t.string(), heatmapP2Json: t.string(),
  },
  (ctx, args) => {
    if (ctx.db.summary.runId.find(args.runId)) {
      ctx.db.summary.runId.update(args);
    } else {
      ctx.db.summary.insert(args);
    }
  }
);

export const addRally = spacetimedb.reducer(
  {
    runId: t.string(), rallyIndex: t.u32(), game: t.u32(),
    startFrame: t.u32(), endFrame: t.u32(), startS: t.f32(), endS: t.f32(),
    shotCount: t.u32(), firstEventId: t.u32(), lastEventId: t.u32(),
    lastHitterId: t.u32(), winnerId: t.u32(), outcome: t.string(),
    landingX: t.f32(), landingY: t.f32(),
    scoreP1: t.u32(), scoreP2: t.u32(), scoreReliable: t.bool(),
  },
  (ctx, args) => {
    ctx.db.rally.insert({ id: 0n, ...args });
  }
);

export const clearRallies = spacetimedb.reducer(
  { runId: t.string() },
  (ctx, { runId }) => {
    for (const r of [...ctx.db.rally.runId.filter(runId)]) {
      ctx.db.rally.id.delete(r.id);
    }
  }
);

export const beginAnalyzedVideo = spacetimedb.reducer(
  { runId: t.string() },
  (ctx, { runId }) => {
    if (ctx.db.analyzedVideo.runId.find(runId)) ctx.db.analyzedVideo.runId.delete(runId);
    ctx.db.analyzedVideo.insert({ runId, video: new Uint8Array(0) });
  }
);

export const appendAnalyzedVideo = spacetimedb.reducer(
  { runId: t.string(), chunk: t.byteArray() },
  (ctx, { runId, chunk }) => {
    const row = ctx.db.analyzedVideo.runId.find(runId);
    if (!row) throw new SenderError('no analyzed video started for this run');
    const merged = new Uint8Array(row.video.length + chunk.length);
    merged.set(row.video, 0);
    merged.set(chunk, row.video.length);
    ctx.db.analyzedVideo.runId.update({ runId: row.runId, video: merged });
  }
);

export const clearRun = spacetimedb.reducer(
  { runId: t.string() },
  (ctx, { runId }) => {
    for (const r of [...ctx.db.runningDetails.runId.filter(runId)]) {
      ctx.db.runningDetails.id.delete(r.id);
    }
    for (const r of [...ctx.db.shotClass.runId.filter(runId)]) {
      ctx.db.shotClass.id.delete(r.id);
    }
    for (const r of [...ctx.db.rally.runId.filter(runId)]) {
      ctx.db.rally.id.delete(r.id);
    }
    if (ctx.db.summary.runId.find(runId)) ctx.db.summary.runId.delete(runId);
    if (ctx.db.analyzedVideo.runId.find(runId)) ctx.db.analyzedVideo.runId.delete(runId);
    if (ctx.db.runMeta.runId.find(runId)) ctx.db.runMeta.runId.delete(runId);
  }
);
