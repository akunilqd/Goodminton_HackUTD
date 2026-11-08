/* Goodminton — backend client.
 * Talks to the SpacetimeDB HTTP API (POST /v1/database/:db/sql). Read-only by design: the site
 * never calls reducers. Every failure surfaces as an ApiError with a stable `code`, a message
 * safe to show people, and a `retryable` flag. The contract is documented in API.md.
 */
(function () {
  'use strict';
  var GM = window.GM = window.GM || {};
  var cfg = Object.assign({
    baseUrl: 'https://maincloud.spacetimedb.com', database: 'badminton',
    timeoutMs: 15000, retries: 2, retryBaseMs: 400, framePageSize: 3000
  }, window.GM_API_CONFIG || {});

  var MESSAGES = {
    offline: 'You appear to be offline. We’ll try again when your connection returns.',
    timeout: 'The analysis service took too long to respond.',
    network: 'We couldn’t reach the analysis service.',
    rate_limited: 'The analysis service is busy right now.',
    server: 'The analysis service had a problem.',
    auth: 'The analysis service refused this request.',
    db_not_found: 'The analysis database couldn’t be found. Check the address in api-config.js.',
    run_not_found: 'This run doesn’t exist, or it has been removed.',
    invalid_id: 'That link doesn’t point to a valid run.',
    no_rallies: 'This run has no rallies with a clear result yet, so there’s no scored report to show.',
    contract: 'The analysis service sent a different format from the one this site expects.',
    malformed: 'The analysis service sent a response we couldn’t read.',
    aborted: 'The request was cancelled.',
    unknown: 'Something went wrong while loading this data.'
  };
  var RETRYABLE = { offline: 1, timeout: 1, network: 1, rate_limited: 1, server: 1, malformed: 1, unknown: 1 };
  var AUTO_RETRY = { timeout: 1, network: 1, rate_limited: 1, server: 1 };

  function ApiError(code, opts) {
    opts = opts || {};
    this.name = 'ApiError';
    this.code = MESSAGES[code] ? code : 'unknown';
    this.message = MESSAGES[this.code];
    this.retryable = !!RETRYABLE[this.code];
    this.status = opts.status || null;
    this.detail = opts.detail || null; // for developers and the console; never rendered as HTML
    this.retryAfterMs = opts.retryAfterMs || 0;
  }
  ApiError.prototype = Object.create(Error.prototype);
  ApiError.prototype.constructor = ApiError;

  function wait(ms, signal) {
    return new Promise(function (resolve, reject) {
      if (signal && signal.aborted) { reject(new ApiError('aborted')); return; }
      function onAbort() { clearTimeout(t); reject(new ApiError('aborted')); }
      var t = setTimeout(function () { if (signal) signal.removeEventListener('abort', onAbort); resolve(); }, ms);
      if (signal) signal.addEventListener('abort', onAbort, { once: true });
    });
  }

  function errorFromResponse(res, body) {
    var s = res.status, opts = { status: s, detail: String(body || '').slice(0, 300) };
    if (s === 401 || s === 403) return new ApiError('auth', opts);
    if (s === 404) return new ApiError('db_not_found', opts);
    if (s === 429) {
      var after = parseFloat(res.headers.get('Retry-After'));
      if (after > 0) opts.retryAfterMs = Math.min(after * 1000, 5000);
      return new ApiError('rate_limited', opts);
    }
    if (s === 400) return new ApiError('contract', opts); // unknown table/column or unsupported SQL
    return new ApiError('server', opts);
  }

  // [{ schema: { elements: [{ name: { some: 'run_id' } }] }, rows: [[…]] }] → [{ run_id: … }]
  function parseRows(text) {
    var data;
    try { data = JSON.parse(text); } catch (e) { throw new ApiError('malformed', { detail: 'Invalid JSON' }); }
    var first = Array.isArray(data) ? data[0] : null;
    if (!first || !first.schema || !Array.isArray(first.schema.elements) || !Array.isArray(first.rows)) {
      throw new ApiError('malformed', { detail: 'Unexpected result envelope' });
    }
    var names = first.schema.elements.map(function (e) {
      return e && e.name && typeof e.name === 'object' ? e.name.some : e && e.name;
    });
    return first.rows.map(function (row) {
      var out = {};
      names.forEach(function (n, i) { if (n) out[n] = row[i]; });
      return out;
    });
  }

  async function attempt(query, signal) {
    var ctl = new AbortController(), timedOut = false;
    var timer = setTimeout(function () { timedOut = true; ctl.abort(); }, cfg.timeoutMs);
    function forward() { ctl.abort(); }
    if (signal) {
      if (signal.aborted) ctl.abort(); else signal.addEventListener('abort', forward, { once: true });
    }
    var url = cfg.baseUrl.replace(/\/+$/, '') + '/v1/database/' + encodeURIComponent(cfg.database) + '/sql';
    try {
      var res, body;
      try {
        res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: query, cache: 'no-store', signal: ctl.signal });
        body = await res.text(); // inside the timeout: a stalled body must fail too
      } catch (e) {
        if (signal && signal.aborted) throw new ApiError('aborted');
        if (timedOut) throw new ApiError('timeout');
        throw new ApiError(navigator.onLine === false ? 'offline' : 'network', { detail: e && e.message });
      }
      if (!res.ok) throw errorFromResponse(res, body);
      return parseRows(body);
    } finally {
      clearTimeout(timer);
      if (signal) signal.removeEventListener('abort', forward);
    }
  }

  async function sql(query, opts) {
    var signal = opts && opts.signal;
    for (var i = 0; ; i++) {
      try { return await attempt(query, signal); }
      catch (e) {
        if (!AUTO_RETRY[e.code] || i >= cfg.retries) throw e;
        var backoff = cfg.retryBaseMs * Math.pow(2, i) + Math.random() * cfg.retryBaseMs;
        await wait(Math.max(backoff, e.retryAfterMs), signal);
      }
    }
  }

  // ---------- safe query building and value cleaning
  function quoteId(id) {
    if (typeof id !== 'string' || !id || id.length > 256 || /[\u0000-\u001f\u007f]/.test(id)) throw new ApiError('invalid_id');
    return "'" + id.replace(/'/g, "''") + "'";
  }
  function num(v) { var n = typeof v === 'number' ? v : parseFloat(v); return isFinite(n) ? n : null; }
  function text(v) { return typeof v === 'string' ? v : ''; }
  function need(rows, columns) {
    if (rows.length && columns.some(function (c) { return !(c in rows[0]); })) throw new ApiError('contract', { detail: 'Missing column' });
    return rows;
  }
  function select(cols, table, where, opts) {
    var q = 'SELECT ' + cols.join(', ') + ' FROM ' + table + (where ? ' WHERE ' + where : '');
    return sql(q, opts).then(function (rows) { return need(rows, cols); });
  }
  // video_source is the absolute path on the analysis machine: only the file name is meaningful here
  function baseName(p) { return text(p).split(/[\\/]/).pop(); }

  // ---------- normalisers: backend rows → the shapes the UI uses
  var META = ['run_id', 'video_source', 'fps', 'total_frames', 'created_at'];
  var SUMMARY = ['run_id', 'p_1_total_distance_m', 'p_2_total_distance_m', 'avg_shuttle_speed_ms', 'heatmap_p_1_json', 'heatmap_p_2_json'];
  var SHOTS = ['event_id', 'frame_index', 'timestamp_s', 'player_id', 'hit_score', 'hit_type', 'confidence', 'player_side'];
  var RALLIES = ['rally_index', 'game', 'start_frame', 'end_frame', 'start_s', 'end_s', 'shot_count', 'first_event_id', 'last_event_id', 'last_hitter_id', 'winner_id', 'outcome', 'landing_x', 'landing_y', 'score_p_1', 'score_p_2', 'score_reliable'];
  var FRAMES = ['frame_index', 'timestamp_s', 'p_1_x', 'p_1_y', 'p_2_x', 'p_2_y', 'shuttle_x', 'shuttle_y', 'shuttle_visible', 'p_1_speed', 'p_2_speed', 'shuttle_speed'];

  function normMeta(r) {
    var fps = num(r.fps), frames = num(r.total_frames) || 0, ms = num(r.created_at);
    return { id: text(r.run_id), videoName: baseName(r.video_source), fps: fps, totalFrames: frames, durationS: fps > 0 ? frames / fps : null, createdAt: ms ? new Date(ms) : null };
  }
  function normHeat(json) {
    var pts;
    try { pts = JSON.parse(json); } catch (e) { return []; }
    if (!Array.isArray(pts)) return [];
    return pts.filter(function (p) { return Array.isArray(p) && num(p[0]) !== null && num(p[1]) !== null; });
  }
  function normSummary(r) {
    return {
      distanceM: { 1: num(r.p_1_total_distance_m), 2: num(r.p_2_total_distance_m) },
      avgShuttleSpeedMs: num(r.avg_shuttle_speed_ms),
      heatmaps: { 1: normHeat(r.heatmap_p_1_json), 2: normHeat(r.heatmap_p_2_json) }
    };
  }
  function normShot(r) {
    var id = num(r.player_id);
    return {
      id: num(r.event_id), frame: num(r.frame_index), t: num(r.timestamp_s),
      playerId: id > 0 ? id : null,                 // the pipeline writes 0 for "hitter unknown"
      side: text(r.player_side) || null,            // 'Top' (far) or 'Bottom' (near)
      type: text(r.hit_type) || null,               // '' when the hit was never classified
      confidence: r.hit_type ? num(r.confidence) : null,
      hitScore: num(r.hit_score)
    };
  }
  function normRally(r) {
    var winner = num(r.winner_id);
    return {
      index: num(r.rally_index), game: num(r.game), startFrame: num(r.start_frame), endFrame: num(r.end_frame), startS: num(r.start_s), endS: num(r.end_s),
      shotCount: num(r.shot_count), firstEventId: num(r.first_event_id), lastEventId: num(r.last_event_id),
      winnerId: winner > 0 ? winner : null, outcome: text(r.outcome) || 'unknown', landing: xy(r.landing_x, r.landing_y),
      scoreP1: num(r.score_p_1) || 0, scoreP2: num(r.score_p_2) || 0, scoreReliable: r.score_reliable === true
    };
  }
  function xy(x, y) { var a = num(x), b = num(y); return a === null || b === null || (a === 0 && b === 0) ? null : [a, b]; } // 0,0 means "missing"
  function normFrame(r) {
    return {
      frame: num(r.frame_index), t: num(r.timestamp_s),
      p1: xy(r.p_1_x, r.p_1_y), p2: xy(r.p_2_x, r.p_2_y),
      shuttle: r.shuttle_visible ? xy(r.shuttle_x, r.shuttle_y) : null,
      p1Speed: num(r.p_1_speed) || null, p2Speed: num(r.p_2_speed) || null, shuttleSpeed: num(r.shuttle_speed) || null // 0 means "not computable"
    };
  }
  function byCreatedDesc(a, b) { return (b.createdAt || 0) - (a.createdAt || 0); }

  // ---------- public reads
  async function listRuns(opts) {
    var res = await Promise.all([select(META, 'run_meta', '', opts), select(['run_id'], 'summary', '', opts)]);
    var done = {};
    res[1].forEach(function (r) { done[r.run_id] = true; });
    return res[0].map(function (r) {
      var m = normMeta(r); m.complete = !!done[m.id]; return m;
    }).sort(byCreatedDesc);
  }

  // null means this backend hasn't deployed the rally table yet: older data must keep working.
  function getRallies(id, opts) {
    return select(RALLIES, 'rally', 'run_id = ' + quoteId(id), opts).then(function (rows) {
      return rows.map(normRally).sort(function (a, b) { return a.index - b.index; });
    }, function (e) { if (e.code === 'contract') return null; throw e; });
  }

  async function getRun(id, opts) {
    var where = 'run_id = ' + quoteId(id);
    var res = await Promise.all([select(META, 'run_meta', where, opts), select(SUMMARY, 'summary', where, opts), select(SHOTS, 'shot_class', where, opts), getRallies(id, opts)]);
    if (!res[0].length) throw new ApiError('run_not_found');
    var run = normMeta(res[0][0]);
    run.summary = res[1].length ? normSummary(res[1][0]) : null;
    run.complete = !!run.summary;
    run.rallies = res[3];
    run.shots = res[2].map(normShot).sort(function (a, b) { return a.frame - b.frame; }); // the server has no ORDER BY
    return run;
  }

  // Per-frame data can be very large, so it is paged by frame range (the server has no OFFSET).
  async function getFrames(run, opts) {
    opts = opts || {};
    var frames = [], size = cfg.framePageSize, base = 'run_id = ' + quoteId(run.id);
    for (var from = 0; from < run.totalFrames; from += size) {
      var rows = await select(FRAMES, 'running_details', base + ' AND frame_index >= ' + from + ' AND frame_index < ' + (from + size), opts);
      for (var i = 0; i < rows.length; i++) frames.push(normFrame(rows[i]));
      if (opts.onProgress) opts.onProgress(Math.min(from + size, run.totalFrames), run.totalFrames);
    }
    return frames.sort(function (a, b) { return a.frame - b.frame; });
  }

  GM.api = { config: cfg, ApiError: ApiError, sql: sql, listRuns: listRuns, getRun: getRun, getRallies: getRallies, getFrames: getFrames };
})();
