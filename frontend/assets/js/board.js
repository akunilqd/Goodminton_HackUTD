/* Goodminton — analysis boards (homepage sample + full report) */
(function () {
  'use strict';
  var GM = window.GM, M = window.GM_MATCH, h = GM.h, P = GM.P, fmt = GM.fmt, score = GM.score;

  function scoreline() {
    var box = h('div', { class: 'scoreline', role: 'table', 'aria-label': 'Match score' });
    ['A', 'B'].forEach(function (id, i) {
      if (i === 1) box.appendChild(h('div', { class: 'div', role: 'presentation' }));
      var who = h('div', { class: 'who', role: 'rowheader' }, [P(id).name]);
      if (M.winner === id) who.appendChild(h('span', { class: 'win-mark', title: 'Match winner', 'aria-label': '(winner)' }));
      box.appendChild(who);
      M.games.forEach(function (g) {
        var won = g.winner === id;
        box.appendChild(h('div', { class: 'g ' + (won ? 'won' : 'lost'), role: 'cell', 'aria-label': 'Game ' + g.n + ': ' + g.score[id] }, [String(g.score[id])]));
      });
    });
    return box;
  }

  function segmented(label, options, value, onChange) {
    var group = h('div', { class: 'filter-group' });
    group.appendChild(h('span', { class: 'label', id: 'lbl-' + label.replace(/\W/g, '') }, [label]));
    var seg = h('div', { class: 'seg', role: 'group', 'aria-labelledby': 'lbl-' + label.replace(/\W/g, '') });
    var btns = options.map(function (o) {
      var b = h('button', { type: 'button', 'aria-pressed': String(o.value === value) }, [o.label]);
      b.addEventListener('click', function () {
        btns.forEach(function (x) { x.setAttribute('aria-pressed', 'false'); });
        b.setAttribute('aria-pressed', 'true');
        onChange(o.value);
      });
      seg.appendChild(b);
      return b;
    });
    group.appendChild(seg);
    group.setValue = function (v) { options.forEach(function (o, i) { btns[i].setAttribute('aria-pressed', String(o.value === v)); }); };
    return group;
  }

  function legend() {
    return h('div', { class: 'map-legend', html:
      '<span><svg viewBox="0 0 16 12"><line x1="1" y1="6" x2="15" y2="6" style="stroke:var(--sel, var(--green))" stroke-width="2"/></svg>Selected rally</span>' +
      '<span><svg viewBox="0 0 16 12"><line x1="1" y1="6" x2="15" y2="6" style="stroke:var(--sel, var(--green))" stroke-width="2.4" stroke-dasharray="1 3" stroke-linecap="round"/></svg>Estimated (tracking lost)</span>' +
      '<span><svg viewBox="0 0 16 12"><circle cx="8" cy="6" r="4" style="fill:var(--sel, var(--green))"/></svg>Winner</span>' +
      '<span><svg viewBox="0 0 16 12"><path d="M4 2l8 8M12 2l-8 8" style="stroke:var(--ink)" stroke-width="2"/></svg>Error</span>' +
      '<span><svg viewBox="0 0 16 12"><circle cx="8" cy="6" r="2.6" style="fill:var(--ink)" opacity=".4"/></svg>Other rally endings</span>'
    });
  }

  function gameOptions() {
    return [{ label: 'All', value: 'all' }].concat(M.games.map(function (g) { return { label: 'G' + g.n, value: g.n }; }));
  }

  function tpList(store, opts) {
    var wrap = h('div', { class: 'tps' });
    var items = M.turningPoints.map(function (tp) {
      var r = M.rallies[tp.rally];
      var btn = h('button', { class: 'btn btn-small', type: 'button' });
      btn.innerHTML = GM.ICON.clip + '<span>Play <span class="num">' + fmt(r.start) + '</span></span>';
      btn.addEventListener('click', function () {
        if (store.get().game !== 'all' && store.get().game !== tp.game) { store.set({ game: tp.game }); if (opts && opts.onGame) opts.onGame(tp.game); }
        GM.select(store, tp.rally);
        if (opts && opts.scrollTo) opts.scrollTo.scrollIntoView({ behavior: GM.reduceMotion ? 'auto' : 'smooth', block: 'start' });
      });
      var art = h('article', { class: 'tp' }, [
        h('div', { class: 'tp-head' }, [h('span', { class: 'tp-num', 'aria-hidden': 'true' }, [tp.label]), h('h3', null, [tp.title])]),
        h('p', { class: 'fact' }, [tp.fact]),
        h('p', { class: 'suggest' }, [tp.suggestion]),
        btn
      ]);
      return { tp: tp, el: art };
    });
    items.forEach(function (i) { wrap.appendChild(i.el); });
    store.on(function (st) {
      items.forEach(function (i) { i.el.classList.toggle('active', st.selected != null && st.selected >= i.tp.from && st.selected <= i.tp.to); });
    });
    return wrap;
  }

  // ------------------------------------------------------------ tracking-derived analysis
  // Everything below is computed from the tracked 2D positions in the match data.
  // Speeds and distances are ESTIMATES (single fixed camera, no height measurement);
  // the UI labels them that way and keeps them apart from counted facts.
  var C = M.court, SHOT_NAME = GM.SHOT_NAME, TYPES = ['serve', 'clear', 'lift', 'drop', 'smash', 'net', 'drive', 'push', 'block'];
  var DECAY = { smash: 1.6, clear: 2.2, lift: 2.0, serve: 1.4, drop: 1.3, drive: 1.1, push: 1.0, block: 0.9, net: 0.8 };
  var KMH = 3.6;

  // shuttle: average speed over the flight from path length / flight time, then a
  // drag-shaped profile (fast off the racket, slow at landing) whose mean equals that average
  function shotSpeed(s) {
    var d = Math.hypot(s.to[0] - s.from[0], s.to[1] - s.from[1]);
    var hpk = 6.2 * (GM.PEAK[s.type] || 0.3);
    var len = Math.sqrt(d * d + 5.33 * hpk * hpk);
    var avg = len / Math.max(0.2, s.dt), k = DECAY[s.type] || 1;
    var v0 = avg * k / (1 - Math.exp(-k));
    return { avg: avg * KMH, launch: v0 * KMH, at: function (p) { return v0 * Math.exp(-k * p) * KMH; } };
  }
  function playerSpeed(r, who, t) {
    // half-second window: a camera track can't resolve faster changes reliably
    var a = GM.playerPos(r, who, t - 0.25), b = GM.playerPos(r, who, t + 0.25);
    return Math.hypot(b[0] - a[0], b[1] - a[1]) / 0.5; // m/s
  }
  // distance a player covered from just before the serve up to time t
  function distanceTo(r, who, t) {
    var d = 0, p = GM.playerPos(r, who, r.start - 1);
    for (var x = r.start - 0.9; x <= t; x += 0.1) { var q = GM.playerPos(r, who, x); d += Math.hypot(q[0] - p[0], q[1] - p[1]); p = q; }
    return d;
  }
  function depthName(who, y) {
    var d = who === 'A' ? y - C.NET : C.NET - y;
    return d < 1.98 ? 'Front court' : (d < 4.6 ? 'Mid court' : 'Rear court');
  }

  // per-player match summary, cached
  var SUMMARY = null;
  function summary() {
    if (SUMMARY) return SUMMARY;
    var out = {};
    ['A', 'B'].forEach(function (id) {
      out[id] = { points: 0, winners: 0, errNet: 0, errOut: 0, shots: 0, mix: {}, smashes: [], dist: 0, peak: 0, speeds: [], run: 0, zones: {}, rallies: 0 };
      TYPES.forEach(function (t) { out[id].mix[t] = 0; });
    });
    var runWho = null, runN = 0;
    M.rallies.forEach(function (r) {
      var w = out[r.winner]; w.points++;
      runN = runWho === r.winner ? runN + 1 : 1; runWho = r.winner; if (runN > w.run) w.run = runN;
      var last = r.shots[r.shots.length - 1];
      if (r.outcome === 'winner') out[last.by].winners++;
      else if (last.result === 'net') out[last.by].errNet++;
      else out[last.by].errOut++;
      r.shots.forEach(function (s) {
        var o = out[s.by]; o.shots++; o.mix[s.type]++;
        if (s.type === 'smash' && s.tracked) o.smashes.push(shotSpeed(s).launch);
        if (s.tracked && s.result !== 'net' && s.result !== 'out' || (s.tracked && s.i < r.shots.length - 1)) {
          var half = s.by === 'A' ? 'far' : 'near', z = GM.zoneOf(s.to[0], s.to[1], half);
          var key = ['Front', 'Mid', 'Rear'][z[1]] + ' ' + ['left', 'centre', 'right'][z[0]];
          o.zones[key] = (o.zones[key] || 0) + 1;
        }
      });
      ['A', 'B'].forEach(function (id) {
        var o = out[id], p = GM.playerPos(r, id, r.start - 1);
        o.rallies++;
        for (var x = r.start - 0.9; x <= r.end; x += 0.1) {
          var q = GM.playerPos(r, id, x), step = Math.hypot(q[0] - p[0], q[1] - p[1]);
          o.dist += step; p = q;
        }
        for (var y = r.start; y <= r.end; y += 0.5) o.speeds.push(playerSpeed(r, id, y));
      });
    });
    ['A', 'B'].forEach(function (id) { var sp = out[id].speeds.sort(function (a, b) { return a - b; }); out[id].peak = sp[Math.floor(sp.length * 0.95)] || 0; delete out[id].speeds; });
    SUMMARY = out;
    return out;
  }

  // ---------- heatmap layer on the court map
  var HEAT_RAMP = [[0, [120, 40, 255, 0]], [0.18, [120, 40, 255, 110]], [0.5, [190, 40, 240, 170]], [0.78, [255, 46, 230, 205]], [1, [255, 214, 246, 235]]];
  function ramp(v) {
    for (var i = 1; i < HEAT_RAMP.length; i++) {
      if (v <= HEAT_RAMP[i][0]) {
        var a = HEAT_RAMP[i - 1], b = HEAT_RAMP[i], t = (v - a[0]) / (b[0] - a[0]);
        return a[1].map(function (c, k) { return c + (b[1][k] - c) * t; });
      }
    }
    return HEAT_RAMP[HEAT_RAMP.length - 1][1];
  }
  var X0 = -0.6, Y0 = -0.6, XW = C.W + 1.2, YH = C.L + 1.2, RES = 22; // px per metre
  function heatImage(points) {
    var w = Math.round(XW * RES), hh = Math.round(YH * RES);
    var cv = document.createElement('canvas'); cv.width = w; cv.height = hh;
    var cx = cv.getContext('2d');
    var acc = new Float32Array(w * hh);
    points.forEach(function (p) {
      var px = Math.round((p[0] - X0) * RES), py = Math.round((p[1] - Y0) * RES);
      if (px >= 0 && py >= 0 && px < w && py < hh) acc[py * w + px] += p[2] || 1;
    });
    // separable box blur x3 ≈ gaussian
    function blur(src, rad) {
      var tmp = new Float32Array(src.length), x, y, s2, k;
      for (y = 0; y < hh; y++) { s2 = 0; for (x = -rad; x <= rad; x++) s2 += src[y * w + Math.min(w - 1, Math.max(0, x))];
        for (x = 0; x < w; x++) { tmp[y * w + x] = s2 / (2 * rad + 1); k = Math.min(w - 1, x + rad + 1); s2 += src[y * w + k] - src[y * w + Math.max(0, x - rad)]; } }
      var out = new Float32Array(src.length);
      for (x = 0; x < w; x++) { s2 = 0; for (y = -rad; y <= rad; y++) s2 += tmp[Math.min(hh - 1, Math.max(0, y)) * w + x];
        for (y = 0; y < hh; y++) { out[y * w + x] = s2 / (2 * rad + 1); k = Math.min(hh - 1, y + rad + 1); s2 += tmp[k * w + x] - tmp[Math.max(0, y - rad) * w + x]; } }
      return out;
    }
    acc = blur(blur(blur(acc, 7), 7), 7);
    var mx = 0, i; for (i = 0; i < acc.length; i++) if (acc[i] > mx) mx = acc[i];
    var img = cx.createImageData(w, hh);
    for (i = 0; i < acc.length; i++) {
      var c = ramp(mx ? Math.pow(acc[i] / mx, 0.7) : 0);
      img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = c[3];
    }
    cx.putImageData(img, 0, 0);
    return cv.toDataURL();
  }
  var MOVE_CACHE = {};
  function movementPoints(game, who) {
    var key = game + '|' + who;
    if (MOVE_CACHE[key]) return MOVE_CACHE[key];
    var pts = [];
    GM.ralliesFor(game).forEach(function (r) {
      (who === 'both' ? ['A', 'B'] : [who]).forEach(function (id) {
        for (var x = r.start - 0.5; x <= r.end; x += 0.2) { var q = GM.playerPos(r, id, x); pts.push([q[0], q[1], 1]); }
      });
    });
    return (MOVE_CACHE[key] = pts);
  }
  function landingPoints(game, who) {
    var pts = [];
    GM.ralliesFor(game).forEach(function (r) {
      r.shots.forEach(function (s) { if (s.tracked && (who === 'both' || s.by === who) && s.result !== 'out') pts.push([s.to[0], s.to[1], 1]); });
    });
    return pts;
  }

  function attachHeat(svg) {
    var NS = 'http://www.w3.org/2000/svg';
    var layer = document.createElementNS(NS, 'g'); layer.setAttribute('class', 'heat');
    svg.insertBefore(layer, svg.firstChild); // floor → heat → lines → paths
    var floor = svg.querySelector('.court-floor');
    if (floor) svg.insertBefore(floor, layer);
    var image = document.createElementNS(NS, 'image');
    image.setAttribute('x', X0); image.setAttribute('y', Y0); image.setAttribute('width', XW); image.setAttribute('height', YH);
    image.setAttribute('preserveAspectRatio', 'none');
    layer.appendChild(image);
    return { layer: layer, paint: function (pts) { image.setAttribute('href', heatImage(pts)); } };
  }

  function courtPanel(host, store) {
    var mode = 'move';
    var head = h('div', { class: 'chart-head' }, [h('h3', null, ['Court map']), h('span', { class: 'small muted' }, ['Top-down · metres'])]);
    host.appendChild(head);
    var modes = segmented('Heatmap', [{ label: 'Movement', value: 'move' }, { label: 'Landings', value: 'land' }, { label: 'Off', value: 'off' }], mode, function (v) { mode = v; paint(store.get()); });
    modes.classList.add('heat-modes');
    host.appendChild(modes);
    var box = h('div', { class: 'court-box' });
    host.appendChild(box);
    var svg = GM.CourtMap(box, store);
    var heat = attachHeat(svg), layer = heat.layer;
    var scale = h('div', { class: 'heat-scale' });
    host.appendChild(scale);
    function paint(st) {
      if (mode === 'off') { layer.style.display = 'none'; scale.style.visibility = 'hidden'; return; }
      layer.style.display = ''; scale.style.visibility = 'visible';
      var pts = mode === 'move' ? movementPoints(st.game, st.player) : landingPoints(st.game, st.player);
      heat.paint(pts);
      var g = st.game === 'all' ? 'all games' : 'game ' + st.game;
      scale.innerHTML = '<span class="small">' + (mode === 'move' ? 'Where players stood' : 'Where shots landed') + ', ' + g + '</span><span class="ramp" aria-hidden="true"><i>Less</i><b></b><i>More</i></span>';
    }
    store.on(function (st, prev) { if (st.game !== prev.game || st.player !== prev.player) paint(st); });
    paint(store.get());
    host.appendChild(legend());
    host.appendChild(h('p', { class: 'how-read', html: '<b>How to read it.</b> The highlighted path is the selected rally, shot by shot. The heat underneath shows ' +
      'where both players spent time (Movement) or where tracked shots landed (Landings) across the games you picked. Brighter means more.' }));
  }

  // Real footage uses timestamped running_details, never the sample's simulated flights.
  GM.trackedMetrics = function (run, input) {
    var frames = input.slice().sort(function (a, b) { return a.t - b.t; });
    var totals = [], shots = (run.shots || []).slice().sort(function (a, b) { return a.t - b.t; });
    frames.forEach(function (f, i) {
      var prev = frames[i - 1], d = i ? totals[i - 1].slice() : [0, 0];
      ['p1', 'p2'].forEach(function (key, j) {
        if (!prev || f.frame !== prev.frame + 1 || !f[key] || !prev[key]) return;
        var dt = f.t - prev.t, step = Math.hypot(f[key][0] - prev[key][0], f[key][1] - prev[key][1]);
        if (dt > 0 && step / dt <= 12) d[j] += step;
      });
      totals.push(d);
    });
    function lastAt(list, t) {
      var lo = 0, hi = list.length;
      while (lo < hi) { var mid = (lo + hi) >>> 1; if (list[mid].t <= t + 0.00001) lo = mid + 1; else hi = mid; }
      return lo - 1;
    }
    return function (t) {
      var i = lastAt(frames, t), f = frames[i];
      if (!f || t - f.t > Math.max(0.1, 2 / (run.fps || 30))) f = null;
      var rally = (run.rallies || []).find(function (r) { return t >= r.startS && t <= r.endS; });
      var si = lastAt(shots, t), shot = shots[si];
      if (shot && (rally ? shot.t < rally.startS : t - shot.t > 1.5)) shot = null;
      var base = rally ? lastAt(frames, rally.startS) : -1;
      var result = { frame: f, shot: shot, shotNumber: shot ? si + 1 : null, rally: rally, players: {} };
      [1, 2].forEach(function (id) {
        var pos = f && f['p' + id], speed = pos && f['p' + id + 'Speed'];
        var depth = pos ? (id === 2 ? pos[1] - 6.7 : 6.7 - pos[1]) : null;
        var zone = !pos ? null : pos[0] < 0 || pos[0] > 6.1 || pos[1] < 0 || pos[1] > 13.4 ? 'Outside' : depth < 1.98 ? 'Front' : depth < 4.6 ? 'Mid' : 'Rear';
        result.players[id] = { speed: speed != null && speed >= 0 && speed <= 12 ? speed * 3.6 : null,
          distance: f && pos ? totals[i][id - 1] - (base >= 0 ? totals[base][id - 1] : 0) : null, zone: zone };
      });
      result.shuttleSpeed = f && f.shuttle && f.shuttleSpeed != null ? f.shuttleSpeed * 3.6 : null;
      return result;
    };
  };

  GM.TrackedLivePanel = function (host, run) {
    var box = h('section', { class: 'hud-live tracked-live', 'aria-label': 'Live analysis' });
    box.innerHTML = '<div class="hl-head"><span class="live-dot" aria-hidden="true"></span><span class="hl-title">Live analysis</span><span class="hl-at num" data-at></span></div>' +
      '<div class="hl-shot"><span class="hl-k" data-k></span><span class="hl-by" data-by></span></div><div class="hl-type" data-type></div>' +
      '<div class="hl-speed"><span class="hl-k">Shuttle <span class="est">Est.</span></span><span class="hl-v num"><b data-sv></b><span>km/h</span></span></div>' +
      '<div class="gauge" aria-hidden="true"><i data-sg></i></div>' +
      '<table class="hl-table"><thead><tr><th scope="col">Metric</th><th scope="col" data-name="2"></th><th scope="col" data-name="1"></th></tr></thead><tbody>' +
      ['Speed', 'Covered', 'Zone'].map(function (label, i) { return '<tr><th scope="row">' + label + '</th><td data-value="2-' + i + '"></td><td data-value="1-' + i + '"></td></tr>'; }).join('') +
      '</tbody></table><p class="hl-foot" data-foot></p><p class="hl-foot" data-status role="status"></p>';
    host.appendChild(box);
    var q = function (sel) { return box.querySelector(sel); }, read = null, time = 0, playing = false;
    var names = { 1: run.farName || 'Far player', 2: run.nearName || 'Near player' };
    [1, 2].forEach(function (id) { q('[data-name="' + id + '"]').textContent = names[id]; });
    function value(v) { return v == null ? '\u2014' : v.toFixed(1); }
    function render(t, active) {
      time = t; playing = active;
      var d = read ? read(t) : { players: {1: {}, 2: {}} }, shot = d.shot;
      box.classList.toggle('is-playing', !!active);
      q('[data-at]').textContent = fmt(t);
      q('[data-k]').textContent = shot ? 'Shot ' + d.shotNumber + ' / ' + run.shots.length : '';
      q('[data-by]').textContent = shot ? names[shot.playerId] || '' : '';
      q('[data-type]').textContent = shot ? shot.type || 'Unclassified' : '\u2014';
      q('[data-sv]').textContent = value(d.shuttleSpeed);
      q('[data-sg]').style.transform = 'scaleX(' + Math.min(1, (d.shuttleSpeed || 0) / 160) + ')';
      [1, 2].forEach(function (id) {
        var p = d.players[id];
        [value(p.speed), value(p.distance), p.zone || '\u2014'].forEach(function (v, i) { q('[data-value="' + id + '-' + i + '"]').textContent = v; });
        q('[data-name="' + id + '"]').classList.toggle('on', !!shot && shot.playerId === id);
      });
      q('[data-foot]').textContent = 'Speed km/h | covered m ' + (d.rally ? 'this rally' : 'since start') + ' | 2D estimates';
      if (read) q('[data-status]').textContent = !d.frame ? 'No tracking at this time' : shot && shot.confidence != null ? 'Class confidence ' + Math.round(shot.confidence * 100) + '%' : '';
    }
    render(0, false);
    q('[data-status]').textContent = 'Loading tracking data...';
    return { render: render, setStatus: function (message) { q('[data-status]').textContent = message; },
      setFrames: function (frames) { read = GM.trackedMetrics(run, frames); render(time, playing); } };
  };

  function realLivePanel(host, store) {
    var panel = GM.TrackedLivePanel(host, M.source);
    panel.setFrames(M.trackingFrames || []);
    store.on(function (st) { panel.render(st.time, st.playing); });
    panel.render(store.get().time, store.get().playing);
    return panel;
  }

  // ---------- live analysis of the moment on screen
  function LivePanel(host, store, opts) {
    if (M.isLive) return realLivePanel(host, store);
    opts = opts || {};
    var box = h('section', { class: 'live' + (opts.compact ? ' live-hud' : ''), 'aria-label': 'Analysis of the moment on screen' });
    box.innerHTML =
      '<header class="live-head"><span class="live-dot" aria-hidden="true"></span><h3>Live analysis</h3><span class="live-at num" data-at></span></header>' +
      '<div class="live-shot">' +
        '<div class="ls-meta"><span class="ls-k" data-shotk></span><span class="ls-by" data-by></span><span class="ls-trk" data-trk></span></div>' +
        '<div class="ls-type" data-type></div>' +
        '<ul class="ls-types" aria-label="Shot classes">' + TYPES.map(function (t) { return '<li data-t="' + t + '">' + SHOT_NAME[t].replace(' shot', '') + '</li>'; }).join('') + '</ul>' +
      '</div>' +
      '<div class="live-speed">' +
        '<div class="lsp-k">Shuttle speed <span class="est">Estimated</span></div>' +
        '<div class="lsp-v num"><b data-sv>—</b><span>km/h</span></div>' +
        '<div class="gauge" aria-hidden="true"><i data-sg></i><span class="tick" style="left:25%"></span><span class="tick" style="left:50%"></span><span class="tick" style="left:75%"></span></div>' +
        '<div class="lsp-sub num" data-ssub></div>' +
      '</div>' +
      '<div class="live-players">' + ['A', 'B'].map(function (id) {
        return '<div class="lp lp-' + id.toLowerCase() + '" data-p="' + id + '">' +
          '<div class="lp-name"><span class="lp-code">' + P(id).code + '</span>' + P(id).name + '</div>' +
          '<dl>' +
          '<div><dt>Speed <span class="est">Est.</span></dt><dd class="num"><b data-v></b> km/h<span class="bar"><i data-vb></i></span></dd></div>' +
          '<div><dt>Covered this rally</dt><dd class="num"><b data-d></b> m</dd></div>' +
          '<div><dt>Position</dt><dd data-z></dd></div>' +
          '</dl></div>';
      }).join('') + '</div>' +
      '<p class="how-read live-note">Shot class comes from the shuttle’s flight and landing. Speeds are estimated from 2D tracking on one fixed camera, so treat them as approximate and compare them within this match, not across matches.</p>';
    host.appendChild(box);
    var q = function (s) { return box.querySelector(s); };
    var at = q('[data-at]'), shotk = q('[data-shotk]'), by = q('[data-by]'), trk = q('[data-trk]'), type = q('[data-type]'),
        sv = q('[data-sv]'), sg = q('[data-sg]'), ssub = q('[data-ssub]'), chips = box.querySelectorAll('.ls-types li');
    var lastType = null;
    var GAUGE_MAX = 160; // km/h, the gauge's full scale
    function render(st) {
      var t = st.time, r = GM.rallyAt(t);
      box.classList.toggle('is-playing', !!st.playing);
      if (!r) { at.textContent = fmt(t); type.textContent = 'Between rallies'; return; }
      at.textContent = fmt(t) + ' · Rally ' + r.id.replace('-', '·');
      var cur = null, idx = -1;
      r.shots.forEach(function (s, i) { if (t >= s.t) { idx = i; cur = s; } });
      var inFlight = cur && t <= cur.t + cur.dt;
      var shot = cur || r.shots[0];
      if (!cur) {
        shotk.textContent = 'Ready'; by.textContent = P(r.server).short + ' to serve';
        trk.textContent = ''; trk.className = 'ls-trk';
      } else {
        shotk.textContent = 'Shot ' + (idx + 1) + ' of ' + r.length;
        by.textContent = P(cur.by).short;
        trk.textContent = cur.tracked ? 'Tracked' : 'Estimated'; trk.className = 'ls-trk' + (cur.tracked ? '' : ' is-est');
      }
      var name = cur ? SHOT_NAME[cur.type] : 'Serve';
      if (name !== lastType) { type.textContent = name; lastType = name; if (!GM.reduceMotion && type.animate) type.animate([{ opacity: 0.2, filter: 'blur(3px)' }, { opacity: 1, filter: 'blur(0)' }], { duration: 180, easing: 'ease-out' }); }
      Array.prototype.forEach.call(chips, function (li) { li.classList.toggle('on', !!cur && li.getAttribute('data-t') === cur.type); });
      if (cur) {
        var sp = shotSpeed(cur), v = inFlight ? sp.at((t - cur.t) / cur.dt) : sp.launch;
        sv.textContent = Math.round(v);
        sg.style.transform = 'scaleX(' + Math.min(1, v / GAUGE_MAX) + ')';
        box.classList.toggle('speed-live', !!inFlight);
        ssub.textContent = (inFlight ? 'In flight · ' : '') + 'off the racket ' + Math.round(sp.launch) + ' · average ' + Math.round(sp.avg) + ' km/h over ' + cur.dt.toFixed(2) + ' s';
      } else { sv.textContent = '—'; sg.style.transform = 'scaleX(0)'; ssub.textContent = 'Waiting for the serve'; box.classList.remove('speed-live'); }
      ['A', 'B'].forEach(function (id) {
        var row = box.querySelector('[data-p="' + id + '"]');
        var v = playerSpeed(r, id, t) * KMH, pos = GM.playerPos(r, id, t);
        row.querySelector('[data-v]').textContent = v.toFixed(1);
        row.querySelector('[data-vb]').style.transform = 'scaleX(' + Math.min(1, v / 22) + ')';
        row.querySelector('[data-d]').textContent = distanceTo(r, id, Math.min(t, r.end)).toFixed(1);
        row.querySelector('[data-z]').textContent = depthName(id, pos[1]);
        row.classList.toggle('is-hitting', !!(inFlight && cur.by === id));
      });
    }
    store.on(function (st, prev) { if (st.time !== prev.time || st.selected !== prev.selected || st.playing !== prev.playing) render(st); });
    render(store.get());
  }

  // ---------- compact overlay for full screen: only what you read at a glance, fixed widths so nothing shifts
  function HudPanel(host, store) {
    if (M.isLive) return realLivePanel(host, store);
    var box = h('section', { class: 'hud-live', 'aria-label': 'Live analysis' });
    box.innerHTML =
      '<div class="hl-head"><span class="live-dot" aria-hidden="true"></span><span class="hl-title">Live</span><span class="hl-at num" data-at></span></div>' +
      '<div class="hl-shot"><span class="hl-k num" data-k></span><span class="hl-by" data-by></span></div>' +
      '<div class="hl-type" data-type></div>' +
      '<div class="hl-speed"><span class="hl-k">Shuttle <span class="est">Est.</span></span><span class="hl-v num"><b data-sv></b><span>km/h</span></span></div>' +
      '<div class="gauge" aria-hidden="true"><i data-sg></i></div>' +
      '<table class="hl-table"><colgroup><col class="c0"><col><col></colgroup>' +
      '<thead><tr><th scope="col"><span class="sr-only">Metric</span></th>' + ['A', 'B'].map(function (id) { return '<th scope="col" data-hit="' + id + '"><i aria-hidden="true"></i>' + P(id).code + '</th>'; }).join('') + '</tr></thead>' +
      '<tbody>' +
      '<tr><th scope="row">Speed</th><td class="num" data-v="A"></td><td class="num" data-v="B"></td></tr>' +
      '<tr><th scope="row">Covered</th><td class="num" data-d="A"></td><td class="num" data-d="B"></td></tr>' +
      '<tr><th scope="row">Zone</th><td data-z="A"></td><td data-z="B"></td></tr>' +
      '</tbody></table>' +
      '<p class="hl-foot">Speed km/h · covered m this rally · estimated</p>';
    host.appendChild(box);
    var q = function (sel) { return box.querySelector(sel); };
    var at = q('[data-at]'), k = q('[data-k]'), by = q('[data-by]'), type = q('[data-type]'), sv = q('[data-sv]'), sg = q('[data-sg]');
    var ZONE = { 'Front court': 'Front', 'Mid court': 'Mid', 'Rear court': 'Rear' };
    function render(st) {
      var t = st.time, r = GM.rallyAt(t);
      box.classList.toggle('is-playing', !!st.playing);
      if (!r) return;
      at.textContent = fmt(t);
      var cur = null, idx = -1;
      r.shots.forEach(function (s, i) { if (t >= s.t) { idx = i; cur = s; } });
      var inFlight = cur && t <= cur.t + cur.dt;
      k.textContent = cur ? 'Shot ' + (idx + 1) + '/' + r.length : 'Serve';
      by.textContent = cur ? P(cur.by).code : P(r.server).code;
      type.textContent = cur ? SHOT_NAME[cur.type] : 'Ready';
      if (cur) {
        var sp = shotSpeed(cur), v = inFlight ? sp.at((t - cur.t) / cur.dt) : sp.launch;
        sv.textContent = Math.round(v); sg.style.transform = 'scaleX(' + Math.min(1, v / 160) + ')';
      } else { sv.textContent = '–'; sg.style.transform = 'scaleX(0)'; }
      ['A', 'B'].forEach(function (id) {
        q('[data-v="' + id + '"]').textContent = (playerSpeed(r, id, t) * KMH).toFixed(1);
        q('[data-d="' + id + '"]').textContent = distanceTo(r, id, Math.min(t, r.end)).toFixed(1);
        q('[data-z="' + id + '"]').textContent = ZONE[depthName(id, GM.playerPos(r, id, t)[1])];
        q('[data-hit="' + id + '"]').classList.toggle('on', !!(inFlight && cur.by === id));
      });
    }
    store.on(function (st, prev) { if (st.time !== prev.time || st.selected !== prev.selected || st.playing !== prev.playing) render(st); });
    render(store.get());
  }

  // ---------- full-screen footage with the analysis laid over it
  // map on the left, live analysis on the right, like a broadcast analysis overlay
  var FS_ICON = {
    enter: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 7.5V3h4.5M12.5 3H17v4.5M17 12.5V17h-4.5M7.5 17H3v-4.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="square"/></svg>',
    exit: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7.5 3v4.5H3M17 7.5h-4.5V3M12.5 17v-4.5H17M3 12.5h4.5V17" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="square"/></svg>',
    hud: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 3.2l7 3.8-7 3.8-7-3.8zM3 11l7 3.8 7-3.8" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>'
  };
  var EASE_FS = 'cubic-bezier(0.32, 0.72, 0, 1)';
  function Fullscreen(playerApi, store) {
    var wrap = playerApi.el, frame = wrap.querySelector('.player-frame'), controls = wrap.querySelector('.player-controls');
    var hudBtn = h('button', { class: 'ctl ctl-hud', type: 'button', 'aria-pressed': 'true', 'aria-label': 'Analysis overlays', title: 'Analysis overlays (H)' });
    hudBtn.innerHTML = FS_ICON.hud;
    var btn = h('button', { class: 'ctl ctl-fs', type: 'button', 'aria-label': 'Full screen', title: 'Full screen (F)' });
    btn.innerHTML = FS_ICON.enter;
    controls.appendChild(hudBtn); controls.appendChild(btn);
    var hud = null, mapBox = null, liveBox = null, shell = null, holder = null, on = false, busy = false;
    var savedY = 0;
    function restoreScroll() { if (Math.abs(window.scrollY - savedY) > 1) window.scrollTo({ top: savedY, behavior: 'instant' }); }

    function buildHud() {
      hud = h('div', { class: 'fs-hud' });
      mapBox = h('div', { class: 'fs-map' });
      mapBox.appendChild(h('div', { class: 'fs-player' }, [P('A').name]));
      mapBox.appendChild(h('div', { class: 'fs-k' }, ['Court map']));
      var svg = GM.CourtMap(mapBox, store), heat = attachHeat(svg);
      function paint(st) { heat.paint(movementPoints(st.game, st.player)); }
      store.on(function (st, prev) { if (st.game !== prev.game || st.player !== prev.player) paint(st); });
      paint(store.get());
      liveBox = h('div', { class: 'fs-live' });
      liveBox.appendChild(h('div', { class: 'fs-player fs-player-far' }, [P('B').name]));
      HudPanel(liveBox, store);
      hud.appendChild(mapBox); hud.appendChild(liveBox);
      frame.appendChild(hud);
    }
    function label() {
      btn.innerHTML = on ? FS_ICON.exit : FS_ICON.enter;
      btn.setAttribute('aria-label', on ? 'Exit full screen' : 'Full screen');
      btn.title = on ? 'Exit full screen (Esc)' : 'Full screen (F)';
    }
    function fsEl() { return document.fullscreenElement || document.webkitFullscreenElement; }

    function enter() {
      if (on || busy) return;
      on = true; busy = true;
      if (!hud) buildHud();
      savedY = window.scrollY;
      var from = frame.getBoundingClientRect();
      holder = h('div', { class: 'fs-holder', 'aria-hidden': 'true' });
      holder.style.height = wrap.offsetHeight + 'px';
      wrap.parentNode.insertBefore(holder, wrap);
      shell = h('div', { class: 'fs-shell', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Full-screen footage with live analysis' });
      shell.style.visibility = 'hidden';
      shell.appendChild(wrap);
      document.body.appendChild(shell);
      wrap.classList.add('is-fs');
      document.documentElement.classList.add('fs-lock');
      label();
      var started = false;
      function go() {
        if (started) return; started = true;
        shell.style.visibility = '';
        btn.focus({ preventScroll: true });
        if (GM.reduceMotion || !wrap.animate) { busy = false; return; }
        var to = frame.getBoundingClientRect();
        var dx = from.left - to.left, dy = from.top - to.top, k = from.width / to.width;
        shell.animate([{ backgroundColor: 'rgba(0,0,0,0)' }, { backgroundColor: 'rgba(0,0,0,1)' }], { duration: 320, easing: 'ease-out' });
        var a = wrap.animate([{ transform: 'translate(' + dx + 'px,' + dy + 'px) scale(' + k + ')' }, { transform: 'none' }], { duration: 520, easing: EASE_FS });
        controls.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 200, delay: 360, fill: 'backwards', easing: 'ease-out' });
        mapBox.animate([{ opacity: 0, transform: 'translateX(-28px)', filter: 'blur(4px)' }, { opacity: 1, transform: 'none', filter: 'blur(0)' }], { duration: 340, delay: 380, fill: 'backwards', easing: 'cubic-bezier(0.23, 1, 0.32, 1)' });
        liveBox.animate([{ opacity: 0, transform: 'translateX(28px)', filter: 'blur(4px)' }, { opacity: 1, transform: 'none', filter: 'blur(0)' }], { duration: 340, delay: 440, fill: 'backwards', easing: 'cubic-bezier(0.23, 1, 0.32, 1)' });
        a.finished.then(function () { busy = false; }, function () { busy = false; });
      }
      // real full screen where the browser allows it; the overlay alone everywhere else (e.g. iPhone)
      var req = shell.requestFullscreen || shell.webkitRequestFullscreen;
      if (req) {
        var pr;
        try { pr = req.call(shell); } catch (e) { pr = null; }
        if (pr && pr.then) pr.then(function () { requestAnimationFrame(function () { requestAnimationFrame(go); }); }, go);
        else requestAnimationFrame(function () { requestAnimationFrame(go); });
        setTimeout(go, 600); // never leave the shell hidden if the browser is slow to answer
      } else go();
    }

    function exit() {
      if (!on || busy) return;
      busy = true;
      function back() {
        function finish() {
          holder.parentNode.insertBefore(wrap, holder);
          holder.remove(); shell.remove();
          wrap.classList.remove('is-fs');
          document.documentElement.classList.remove('fs-lock');
          restoreScroll(); requestAnimationFrame(restoreScroll); setTimeout(restoreScroll, 120);
          on = false; busy = false; label();
          btn.focus({ preventScroll: true });
        }
        // full screen can shift the page underneath; put it back so the video lands where it left
        restoreScroll();
        if (GM.reduceMotion || !wrap.animate) { finish(); restoreScroll(); return; }
        var from = frame.getBoundingClientRect(), to = holder.getBoundingClientRect();
        var dx = to.left - from.left, dy = to.top - from.top, k = to.width / from.width;
        // exit is faster than entry: the system is responding, not presenting
        [mapBox, liveBox, controls].forEach(function (n) { n.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 120, fill: 'forwards', easing: 'ease-out' }); });
        shell.animate([{ backgroundColor: 'rgba(0,0,0,1)' }, { backgroundColor: 'rgba(0,0,0,0)' }], { duration: 300, fill: 'forwards', easing: 'ease-out' });
        var a = wrap.animate([{ transform: 'none' }, { transform: 'translate(' + dx + 'px,' + dy + 'px) scale(' + k + ')' }], { duration: 320, easing: EASE_FS, fill: 'forwards' });
        a.finished.then(function () {
          finish();
          [wrap, mapBox, liveBox, controls].forEach(function (n) { n.getAnimations().forEach(function (x) { x.cancel(); }); });
        }, finish);
      }
      var exitFs = document.exitFullscreen || document.webkitExitFullscreen;
      if (fsEl() === shell && exitFs) {
        var pr = exitFs.call(document);
        if (pr && pr.then) pr.then(function () { requestAnimationFrame(back); }, back); else requestAnimationFrame(back);
      } else back();
    }

    function toggle() { on ? exit() : enter(); }
    btn.addEventListener('click', toggle);
    hudBtn.addEventListener('click', function () {
      var off = wrap.classList.toggle('hud-off');
      hudBtn.setAttribute('aria-pressed', String(!off));
    });
    // Esc in real full screen: the browser leaves first, then we animate home
    ['fullscreenchange', 'webkitfullscreenchange'].forEach(function (ev) {
      document.addEventListener(ev, function () { if (!fsEl() && on && !busy) exit(); });
    });
    document.addEventListener('keydown', function (e) {
      var tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 'f') { e.preventDefault(); toggle(); }
      else if (e.key === 'h' && on) hudBtn.click();
      else if (e.key === 'Escape' && on) { e.preventDefault(); exit(); }
    });
    return { enter: enter, exit: exit };
  }

  // ---------- the two players, side by side, after the analysis
  function playersSection(host) {
    var S = summary();
    var wrap = h('div', { class: 'players' });
    ['A', 'B'].forEach(function (id) {
      var o = S[id], pl = P(id);
      var gw = M.games.filter(function (g) { return g.winner === id; }).length;
      var mix = TYPES.map(function (t) { return [t, o.mix[t]]; }).filter(function (x) { return x[1]; }).sort(function (a, b) { return b[1] - a[1]; });
      var zone = Object.keys(o.zones).sort(function (a, b) { return o.zones[b] - o.zones[a]; })[0];
      var zoneShare = Math.round(o.zones[zone] / Object.keys(o.zones).reduce(function (s, k) { return s + o.zones[k]; }, 0) * 100);
      var smashAvg = o.smashes.length ? o.smashes.reduce(function (a, b) { return a + b; }, 0) / o.smashes.length : 0;
      var smashTop = o.smashes.length ? Math.max.apply(null, o.smashes) : 0;
      var col = h('article', { class: 'pc pc-' + id.toLowerCase() + (M.winner === id ? ' is-winner' : '') });
      col.innerHTML =
        '<header class="pc-head">' +
          '<span class="pc-plate" aria-hidden="true">' + pl.code + '</span>' +
          '<div><h3>' + pl.name + '</h3><p class="small">' + (M.isLive ? '' : (pl.hand === 'left' ? 'Left' : 'Right') + '-handed · ') + (id === 'A' ? 'near' : 'far') + ' end on camera' + (M.isLive ? '' : ' · fictional player') + '</p></div>' +
          '<div class="pc-result num"><b>' + gw + '</b><span>games</span></div>' +
        '</header>' +
        '<div class="pc-group"><h4>' + (M.isLive ? 'Detected <span>model output, check against the footage</span>' : 'Measured <span>counted from the match</span>') + '</h4>' +
          '<dl class="pc-grid">' +
            '<div><dt>Points won</dt><dd class="num">' + o.points + '</dd></div>' +
            '<div><dt>Winners</dt><dd class="num">' + o.winners + '</dd></div>' +
            '<div><dt>Errors</dt><dd class="num">' + (o.errNet + o.errOut) + '<small>' + o.errNet + ' net · ' + o.errOut + ' out</small></dd></div>' +
            '<div><dt>Longest run</dt><dd class="num">' + o.run + '<small>points in a row</small></dd></div>' +
            '<div><dt>Shots played</dt><dd class="num">' + o.shots + '</dd></div>' +
            '<div><dt>Most-used target</dt><dd>' + (zone || '—') + '<small>' + (zone ? zoneShare + '% of tracked landings, camera view' : 'No tracked landings') + '</small></dd></div>' +
          '</dl>' +
          '<div class="pc-mix"><div class="pc-mix-k">Shot mix</div>' + mix.map(function (x) {
            var pct = x[1] / o.shots * 100;
            return '<div class="mx"><span>' + SHOT_NAME[x[0]] + '</span><span class="mx-bar"><i style="width:' + pct.toFixed(1) + '%"></i></span><span class="num">' + Math.round(pct) + '%</span></div>';
          }).join('') + '</div>' +
        '</div>' +
        '<div class="pc-group pc-est"><h4>Estimated <span>from 2D tracking, approximate</span></h4>' +
          '<dl class="pc-grid">' +
            '<div><dt>Smash, off the racket</dt><dd class="num">' + (o.smashes.length ? Math.round(smashAvg) : '—') + '<small>' + (o.smashes.length ? 'km/h average · top ' + Math.round(smashTop) : 'No tracked smashes') + '</small></dd></div>' +
            '<div><dt>Distance covered</dt><dd class="num">' + (o.dist / 1000).toFixed(2) + '<small>km · ' + Math.round(o.dist / o.rallies) + ' m per rally</small></dd></div>' +
            '<div><dt>Fast movement</dt><dd class="num">' + (o.peak * KMH).toFixed(1) + '<small>km/h, 95th percentile over half-second windows</small></dd></div>' +
          '</dl>' +
        '</div>';
      wrap.appendChild(col);
    });
    host.appendChild(wrap);
  }
  GM.playersSection = playersSection;
  GM.attachHeat = attachHeat;

  // ------------------------------------------------------------ homepage sample
  GM.mountSample = function (root) {
    var store = GM.newStore();
    var top = h('div', { class: 'board-top' });
    var meta = h('div', { class: 'meta' }, [
      scoreline(),
      h('div', { class: 'small muted', html: '<span class="tag"><span class="dot"></span>Illustrative data</span>&nbsp; Fictional players · ' + M.rallies.length + ' rallies · ' + M.util.fmtDuration(M.duration) })
    ]);
    var gsel = segmented('Game', gameOptions(), 'all', function (v) { store.set({ game: v }); });
    top.appendChild(meta); top.appendChild(gsel);
    root.appendChild(top);

    var board = h('div', { class: 'board-v2 board-home' });
    var video = h('div', { class: 'bv-video' });
    var row = h('div', { class: 'bv-row' });
    var left = h('div', { class: 'bv-court' });
    var right = h('div', { class: 'bv-live' });
    var wide = h('div', { class: 'board-wide board-tl' });
    row.appendChild(left); row.appendChild(right);
    board.appendChild(video); board.appendChild(row); board.appendChild(wide);
    root.appendChild(board);

    var player = GM.Player(video, store);
    Fullscreen(player, store);
    video.appendChild(h('p', { class: 'how-read', style: 'margin:10px 0 0', html: '<b>The footage.</b> Selecting a rally jumps the video to it. Here, a placeholder court is driven by the sample’s tracked positions until real footage is supplied.' }));

    courtPanel(left, store);
    LivePanel(right, store);
    GM.RallyDetail(right, store);

    wide.appendChild(h('div', { class: 'chart-head' }, [h('h3', null, ['Rally timeline']), h('span', { class: 'small muted' }, ['Click a bar, or focus it and use ← →'])]));
    GM.Timeline(wide, store);
    wide.appendChild(h('p', { class: 'how-read', style: 'margin-top:6px', html: '<b>How to read it.</b> One bar per rally: up for points won by ' + P('A').short + ' (solid), down for ' + P('B').short + ' (hatched). Bar height is the number of shots. The line above tracks who led, and brackets mark runs of four or more points.' }));
    wide.appendChild(h('div', { class: 'chart-head', style: 'margin-top:36px' }, [h('h3', null, ['Key moments']), h('span', { class: 'small muted' }, ['The runs that decided each game'])]));
    wide.appendChild(tpList(store, { scrollTo: board, onGame: function (g) { gsel.setValue(g); } }));

    var pw = h('div', { class: 'board-wide board-players' });
    pw.appendChild(h('div', { class: 'chart-head' }, [h('h3', null, ['The players']), h('span', { class: 'small muted' }, ['Whole match · illustrative'])]));
    playersSection(pw);
    board.appendChild(pw);

    // start on the second turning point so the board is never empty
    var tp = M.turningPoints[1], r0 = M.rallies[tp.rally], s0 = r0.shots[Math.min(2, r0.shots.length - 1)];
    store.set({ selected: tp.rally, time: s0.t + s0.dt * 0.3 });
    GM.bindKeys(store, player);
    return store;
  };

  // ------------------------------------------------------------ full report
  GM.mountReport = function (root) {
    var store = GM.newStore();
    var filters = root.querySelector('[data-filters]');
    var gsel = segmented('Game', gameOptions(), 'all', function (v) { store.set({ game: v }); });
    var psel = segmented('Player', [{ label: 'Both', value: 'both' }, { label: P('A').short, value: 'A' }, { label: P('B').short, value: 'B' }], 'both', function (v) { store.set({ player: v }); });
    filters.appendChild(gsel); filters.appendChild(psel);
    filters.appendChild(h('span', { class: 'spacer' }));
    filters.appendChild(h('span', { class: 'small muted hide-s', html: '<span class="kbd">J</span><span class="kbd">K</span> next / previous rally' }));

    var main = root.querySelector('[data-main]');
    var side = root.querySelector('[data-side]');
    var liveHost = root.querySelector('[data-live]');
    var tlRow = root.querySelector('[data-timeline]') || main;
    var player = GM.Player(main, store);
    Fullscreen(player, store);

    var tlBox = h('div', { class: 'rp-timeline' });
    tlRow.appendChild(tlBox);
    tlBox.appendChild(h('div', { class: 'chart-head' }, [h('h3', null, ['Where the match turned']), h('span', { class: 'small muted' }, ['Rally timeline'])]));
    var tlGameWrap = h('div', { class: 'tl-mobile-scope' });
    tlBox.appendChild(tlGameWrap);
    GM.Timeline(tlGameWrap, store);
    tlBox.appendChild(h('p', { class: 'how-read', style: 'margin-top:6px', html: 'Bars: points won by <b>' + P('A').short + '</b> (up, solid) and <b>' + P('B').short + '</b> (down, hatched); height is shots per rally. Line: score margin. Brackets: runs of 4+. Numbered markers: turning points below.' }));

    if (liveHost) {
      courtPanel(side, store);
      LivePanel(liveHost, store);
      GM.RallyDetail(liveHost, store);
    } else {
      side.appendChild(h('div', { class: 'chart-head' }, [h('h3', null, ['Court map']), h('span', { class: 'small muted' }, ['Top-down'])]));
      GM.CourtMap(side, store);
      side.appendChild(legend());
      GM.RallyDetail(side, store);
    }
    var playersHost = root.querySelector('[data-players]');
    if (playersHost) playersSection(playersHost);

    var markBtn = h('button', { class: 'btn btn-small', type: 'button', 'aria-pressed': 'false' });
    (liveHost || side).appendChild(markBtn);

    // patterns
    GM.Placement(root.querySelector('[data-placement]'), store);
    GM.RallyLengths(root.querySelector('[data-lengths]'), store);
    GM.Ledger(root.querySelector('[data-ledger]'), store);
    root.querySelector('[data-tps]').appendChild(tpList(store, { scrollTo: main, onGame: function (g) { gsel.setValue(g); } }));

    // review list (per-viewer convenience; a backend would store this per user)
    var KEY = 'gm-review-' + M.id;
    var marked = [];
    try { marked = JSON.parse(localStorage.getItem(KEY) || '[]'); } catch (e) { marked = []; }
    function save() { try { localStorage.setItem(KEY, JSON.stringify(marked)); } catch (e) { } }
    function isMarked(i) { return marked.indexOf(i) !== -1; }
    function toggle(i) { if (isMarked(i)) marked.splice(marked.indexOf(i), 1); else marked.push(i); marked.sort(function (a, b) { return a - b; }); save(); renderMine(); syncMark(); }
    function syncMark() {
      var st = store.get();
      if (st.selected == null) { markBtn.disabled = true; return; }
      markBtn.disabled = false;
      var on = isMarked(st.selected);
      markBtn.setAttribute('aria-pressed', String(on));
      markBtn.innerHTML = (on ? GM.ICON.marked : GM.ICON.mark) + '<span>' + (on ? 'In your review list' : 'Add rally to review list') + '</span>';
    }
    markBtn.addEventListener('click', function () { var s = store.get().selected; if (s != null) toggle(s); });

    var suggested = root.querySelector('[data-suggested]');
    var mine = root.querySelector('[data-mine]');
    function clipRow(r, title, note, opts) {
      opts = opts || {};
      var li = h('li', { class: 'clip', 'data-i': r.index });
      var play = h('button', { class: 'btn btn-small clip-time', type: 'button', 'aria-label': 'Play rally ' + r.id + ' at ' + fmt(r.start) });
      play.innerHTML = '<span class="num">' + fmt(r.start) + '</span>';
      play.addEventListener('click', function () {
        if (store.get().game !== 'all' && store.get().game !== r.game) { store.set({ game: r.game }); gsel.setValue(r.game); }
        GM.select(store, r.index);
        main.scrollIntoView({ behavior: GM.reduceMotion ? 'auto' : 'smooth', block: 'start' });
      });
      var body = h('div', null, [h('h4', null, [title]), h('p', null, [note])]);
      if (opts.suggest) body.appendChild(h('p', { class: 'suggest' }, [opts.suggest]));
      li.appendChild(play); li.appendChild(body);
      if (opts.removable) {
        var rm = h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Remove rally ' + r.id + ' from review list', 'aria-pressed': 'true' });
        rm.innerHTML = GM.ICON.marked;
        rm.addEventListener('click', function () { toggle(r.index); });
        li.appendChild(rm);
      } else li.appendChild(h('span'));
      return li;
    }
    function renderSuggested(st) {
      suggested.innerHTML = '';
      var list = GM.ralliesFor(st.game).filter(function (r) { return st.player === 'both' || r.winner === st.player; });
      var rows = [];
      M.turningPoints.forEach(function (tp) {
        var r = M.rallies[tp.rally];
        if (list.indexOf(r) !== -1) rows.push([r, 'Turning point ' + tp.label + ' · ' + tp.title, tp.fact, { suggest: tp.suggestion }]);
      });
      var longest = list.slice().sort(function (a, b) { return b.length - a.length; })[0];
      if (longest) rows.push([longest, 'Longest rally · ' + longest.length + ' shots', 'Game ' + longest.game + ', ' + score(longest.scoreBefore) + ' → ' + score(longest.scoreAfter) + '. ' + GM.endingText(longest) + '.']);
      // late-game net errors: either player on 18 or more
      var nets = list.filter(function (r) {
        return r.outcome === 'error' && r.shots[r.shots.length - 1].result === 'net' && Math.max(r.scoreBefore.A, r.scoreBefore.B) >= 18;
      });
      nets.slice(0, 2).forEach(function (r) { rows.push([r, 'Late net error at ' + score(r.scoreBefore), 'Game ' + r.game + '. ' + GM.endingText(r) + '.']); });
      if (!rows.length) { suggested.appendChild(h('li', { class: 'empty', html: '<b>Nothing suggested for this filter.</b>Try “All” games or “Both” players.' })); return; }
      rows.sort(function (a, b) { return a[0].index - b[0].index; }).forEach(function (row) { suggested.appendChild(clipRow(row[0], row[1], row[2], row[3])); });
    }
    function renderMine() {
      mine.innerHTML = '';
      if (!marked.length) {
        mine.appendChild(h('li', { class: 'empty', html: '<b>Your review list is empty.</b>Select a rally on the timeline, then choose “Add rally to review list”. Saved in this browser only.' }));
        return;
      }
      marked.forEach(function (i) {
        var r = M.rallies[i];
        mine.appendChild(clipRow(r, 'Rally ' + r.id.replace('-', '·') + ' · ' + score(r.scoreBefore) + ' → ' + score(r.scoreAfter), GM.plural(r.length, 'shot') + '. ' + GM.endingText(r) + '.', { removable: true }));
      });
    }
    function syncActive(st) {
      root.querySelectorAll('.clip').forEach(function (li) { li.classList.toggle('active', +li.getAttribute('data-i') === st.selected); });
    }
    store.on(function (st, prev) {
      if (st.game !== prev.game || st.player !== prev.player) renderSuggested(st);
      if (st.selected !== prev.selected) { syncMark(); syncActive(st); }
    });
    renderSuggested(store.get()); renderMine();

    var start = M.turningPoints[1] || M.turningPoints[0] || { rally: 0 }, sr = M.rallies[start.rally], ss = sr.shots[Math.min(2, sr.shots.length - 1)]; // a short, clean rally reads best on first load
    store.set({ selected: start.rally, time: ss.t + ss.dt * 0.3 });
    syncMark(); syncActive(store.get());
    GM.bindKeys(store, player);
    return store;
  };
})();
