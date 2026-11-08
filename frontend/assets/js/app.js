/* Goodminton — shared UI engine
 * Court rendering (top-down + broadcast projection), the footage player,
 * the score-margin timeline, pattern charts, and the store linking them.
 * Classic script (no modules) so pages also work opened from disk.
 */
(function () {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';
  var M = window.GM_MATCH;
  var C = M.court;
  var fmt = M.util.fmtTime;
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------------------------------------------------------------- utils
  function el(tag, attrs, parent) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function h(tag, attrs, children) {
    var n = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === 'text') n.textContent = attrs[k];
      else if (k === 'html') n.innerHTML = attrs[k];
      else if (k.slice(0, 2) === 'on') n.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== undefined && attrs[k] !== null && attrs[k] !== false) n.setAttribute(k, attrs[k] === true ? '' : attrs[k]);
    }
    (children || []).forEach(function (c) { if (c != null) n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return n;
  }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function lerp(a, b, p) { return a + (b - a) * p; }
  function ease(p) { return p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2; }
  function P(id) { return M.players[id]; }
  function other(id) { return id === 'A' ? 'B' : 'A'; }
  function score(s) { return s.A + '–' + s.B; }
  function plural(n, w) { return n + ' ' + w + (n === 1 ? '' : 's'); }
  var SHOT_NAME = { serve: 'Serve', clear: 'Clear', lift: 'Lift', drop: 'Drop', smash: 'Smash', net: 'Net shot', drive: 'Drive', push: 'Push', block: 'Block', unknown: 'Unclassified' };

  function endingText(r) {
    var last = r.shots[r.shots.length - 1];
    if (r.outcome === 'winner') return SHOT_NAME[last.type] + ' winner · ' + P(r.winner).short;
    return (last.result === 'net' ? 'Into the net' : 'Out') + ' · ' + P(last.by).short + ' error';
  }

  // ---------------------------------------------------------------- store
  function Store(init) {
    var state = init, subs = [];
    return {
      get: function () { return state; },
      set: function (patch, meta) {
        var prev = state; state = Object.assign({}, state, patch);
        subs.forEach(function (f) { f(state, prev, meta || {}); });
      },
      on: function (f) { subs.push(f); }
    };
  }

  function ralliesFor(game) {
    return game === 'all' ? M.rallies : M.rallies.filter(function (r) { return r.game === game; });
  }
  function rallyAt(t) {
    for (var i = 0; i < M.rallies.length; i++) {
      var r = M.rallies[i];
      var next = M.rallies[i + 1];
      if (t >= r.start - 1.2 && (!next || t < next.start - 1.2)) return r;
    }
    return null;
  }

  // ---------------------------------------------------------------- top-down court
  function drawCourtLines(g, opts) {
    opts = opts || {};
    var lw = opts.lw || 0.045;
    var cls = 'court-line';
    function line(x1, y1, x2, y2) { el('line', { x1: x1, y1: y1, x2: x2, y2: y2, class: cls, 'stroke-width': lw }, g); }
    el('rect', { x: 0, y: 0, width: C.W, height: C.L, class: 'court-floor' }, g);
    // outer (doubles) boundary
    el('rect', { x: 0, y: 0, width: C.W, height: C.L, class: cls, 'stroke-width': lw, fill: 'none' }, g);
    // singles sidelines
    line(C.SINGLES_IN, 0, C.SINGLES_IN, C.L);
    line(C.W - C.SINGLES_IN, 0, C.W - C.SINGLES_IN, C.L);
    // doubles long service lines
    line(0, C.DOUBLES_LONG, C.W, C.DOUBLES_LONG);
    line(0, C.L - C.DOUBLES_LONG, C.W, C.L - C.DOUBLES_LONG);
    // short service lines
    line(0, C.NET - C.SHORT, C.W, C.NET - C.SHORT);
    line(0, C.NET + C.SHORT, C.W, C.NET + C.SHORT);
    // centre lines
    line(C.W / 2, 0, C.W / 2, C.NET - C.SHORT);
    line(C.W / 2, C.NET + C.SHORT, C.W / 2, C.L);
    // net
    el('line', { x1: -0.25, y1: C.NET, x2: C.W + 0.25, y2: C.NET, class: 'court-net', 'stroke-width': lw * 1.8 }, g);
    el('circle', { cx: -0.25, cy: C.NET, r: 0.07, class: 'court-post' }, g);
    el('circle', { cx: C.W + 0.25, cy: C.NET, r: 0.07, class: 'court-post' }, g);
  }

  function arrowHead(g, x1, y1, x2, y2, size, cls) {
    var a = Math.atan2(y2 - y1, x2 - x1), s = size;
    var p1 = [x2 - s * Math.cos(a - 0.45), y2 - s * Math.sin(a - 0.45)];
    var p2 = [x2 - s * Math.cos(a + 0.45), y2 - s * Math.sin(a + 0.45)];
    return el('path', { d: 'M' + p1 + 'L' + x2 + ',' + y2 + 'L' + p2, class: cls, fill: 'none' }, g);
  }
  function cross(g, x, y, s, cls) {
    return el('path', { d: 'M' + (x - s) + ',' + (y - s) + 'L' + (x + s) + ',' + (y + s) + 'M' + (x + s) + ',' + (y - s) + 'L' + (x - s) + ',' + (y + s), class: cls }, g);
  }

  function CourtMap(host, store, opts) {
    opts = opts || {};
    var svg = el('svg', { viewBox: '-1.1 -1.2 8.3 15.9', class: 'court-map', role: 'img', 'aria-label': 'Top-down court map' });
    host.appendChild(svg);
    var base = el('g', null, svg);
    drawCourtLines(base);
    // half labels
    var tA = el('text', { x: C.W + 0.55, y: C.L - 0.2, class: 'court-label', transform: 'rotate(-90 ' + (C.W + 0.55) + ' ' + (C.L - 0.2) + ')' }, base);
    tA.textContent = P('A').code + ' · near';
    var tB = el('text', { x: C.W + 0.55, y: C.NET - 0.35, class: 'court-label', transform: 'rotate(-90 ' + (C.W + 0.55) + ' ' + (C.NET - 0.35) + ')' }, base);
    tB.textContent = P('B').code + ' · far';
    var endings = el('g', { class: 'endings' }, svg);
    var path = el('g', { class: 'rally-path' }, svg);

    function drawEndings(st) {
      clear(endings);
      ralliesFor(st.game).forEach(function (r) {
        var last = r.shots[r.shots.length - 1];
        if (st.player !== 'both' && last.by !== st.player) return;
        if (st.selected === r.index) return;
        var x = last.to[0], y = last.to[1];
        if (r.outcome === 'winner') el('circle', { cx: x, cy: y, r: 0.12, class: 'end-winner' }, endings);
        else cross(endings, x, y, 0.1, 'end-error');
      });
    }

    var lastDrawn = { idx: null, upto: null };
    function drawPath(r, t) {
      clear(path);
      if (!r) return;
      var shots = r.shots;
      // long rallies: keep the closing exchange readable, fade what came before
      var KEEP = 5, n = shots.length;
      var lastDone = t == null ? n - 1 : shots.reduce(function (m, s, i) { return t >= s.t ? i : m; }, -1);
      shots.forEach(function (s, i) {
        var done = t == null || t >= s.t + s.dt;
        var cur = !done && t >= s.t;
        if (!done && !cur && t != null) {
          // upcoming: faint guide only
          el('line', { x1: s.from[0], y1: s.from[1], x2: s.to[0], y2: s.to[1], class: 'shot shot-future' }, path);
          return;
        }
        var old = n > 8 && i < lastDone - (KEEP - 1);
        var p = cur ? Math.min(1, (t - s.t) / s.dt) : 1;
        var x2 = lerp(s.from[0], s.to[0], p), y2 = lerp(s.from[1], s.to[1], p);
        var cls = 'shot' + (s.tracked ? '' : ' shot-untracked') + (s.by === 'A' ? ' by-a' : ' by-b') + (old ? ' shot-old' : '');
        if (!old) el('line', { x1: s.from[0], y1: s.from[1], x2: x2, y2: y2, class: 'shot-case' }, path);
        el('line', { x1: s.from[0], y1: s.from[1], x2: x2, y2: y2, class: cls }, path);
        if (old) return;
        if (p === 1 && i < shots.length - 1) {
          el('circle', { cx: s.to[0], cy: s.to[1], r: 0.085, class: 'shot-land' }, path);
          // number the recent shots so the closing exchange can be followed
          var lbl = el('text', { x: s.to[0] + 0.16, y: s.to[1] - 0.12, class: 'shot-num' }, path);
          lbl.textContent = String(i + 1);
        }
        if (p > 0.2) arrowHead(path, s.from[0], s.from[1], x2, y2, 0.22, 'shot-head');
        if (cur) el('circle', { cx: x2, cy: y2, r: 0.13, class: 'shot-shuttle' }, path);
      });
      // ending mark once rally completed
      var last = shots[shots.length - 1];
      if (t == null || t >= last.t + last.dt) {
        if (r.outcome === 'winner') {
          el('circle', { cx: last.to[0], cy: last.to[1], r: 0.26, class: 'mark-winner-ring' }, path);
          el('circle', { cx: last.to[0], cy: last.to[1], r: 0.13, class: 'mark-winner' }, path);
        } else {
          cross(path, last.to[0], last.to[1], 0.2, 'mark-error');
        }
      }
      // first shot origin
      el('circle', { cx: shots[0].from[0], cy: shots[0].from[1], r: 0.11, class: 'shot-origin' }, path);
    }

    function traceAnimate() {
      if (reduceMotion || !path.animate) return;
      var lines = path.querySelectorAll('line.shot');
      var delay = 0;
      Array.prototype.forEach.call(lines, function (ln) {
        var len = Math.hypot(ln.x2.baseVal.value - ln.x1.baseVal.value, ln.y2.baseVal.value - ln.y1.baseVal.value);
        ln.style.strokeDasharray = len; ln.style.strokeDashoffset = len;
        ln.animate([{ strokeDashoffset: len }, { strokeDashoffset: 0 }], { duration: 140, delay: delay, fill: 'forwards', easing: 'cubic-bezier(.2,.7,.2,1)' });
        delay += 95;
      });
      Array.prototype.forEach.call(path.querySelectorAll('.shot-head,.shot-land,.mark-winner,.mark-winner-ring,.mark-error'), function (n, i) {
        n.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160, delay: Math.min(delay, 95 * (i + 1)), fill: 'backwards' });
      });
    }

    store.on(function (st, prev, meta) {
      if (st.game !== prev.game || st.player !== prev.player || st.selected !== prev.selected) drawEndings(st);
      var r = st.selected != null ? M.rallies[st.selected] : null;
      if (st.selected !== prev.selected || st.time !== prev.time || st.playing !== prev.playing) {
        var live = r && st.playing && st.time >= r.start - 1.2 && st.time <= r.end + 0.1;
        var t = live ? st.time : null;
        if (lastDrawn.idx === st.selected && t == null && lastDrawn.upto == null && !meta.trace) return;
        drawPath(r, t);
        lastDrawn = { idx: st.selected, upto: t };
        if (meta.trace && !live) traceAnimate();
      }
    });
    drawEndings(store.get());
    return svg;
  }

  // ---------------------------------------------------------------- broadcast projection
  var BC = { w: 1600, h: 900, nearY: 846, farY: 262, nearW: 1120, farW: 660, cx: 800, vert: 0.42 };
  BC.k = BC.nearW / BC.farW - 1;
  function project(x, y, hM) {
    // y: far baseline 0 → near baseline L. depth fraction t: 0 near → 1 far
    var t = (C.L - y) / C.L;
    var u = t * (1 + BC.k) / (1 + BC.k * t);
    var w = BC.nearW / (1 + BC.k * t);
    var sx = BC.cx + (x / C.W - 0.5) * w;
    var sy = BC.nearY - u * (BC.nearY - BC.farY);
    var ppm = w / C.W; // pixels per metre at this depth
    if (hM) sy -= hM * ppm * BC.vert;
    return [sx, sy, ppm];
  }

  function drawBroadcast(svg) {
    // scene: back wall, perimeter boards, floor, court
    var defs = el('defs', null, svg);
    var pat = el('pattern', { id: 'grain', width: 6, height: 6, patternUnits: 'userSpaceOnUse' }, defs);
    el('rect', { width: 6, height: 6, fill: 'transparent' }, pat);
    el('circle', { cx: 1, cy: 1, r: 0.6, fill: 'rgba(0,0,0,.08)' }, pat);
    el('rect', { x: 0, y: 0, width: BC.w, height: BC.h, class: 'bc-floor' }, svg);
    el('rect', { x: 0, y: 0, width: BC.w, height: 205, class: 'bc-wall' }, svg);
    // perimeter boards carrying the placeholder label
    el('rect', { x: 0, y: 160, width: BC.w, height: 46, class: 'bc-board' }, svg);
    // one centred board label, clear of the score bug (left) and the placeholder tag (right)
    var tx = el('text', { x: BC.w / 2, y: 191, class: 'bc-board-text', 'text-anchor': 'middle' }, svg);
    tx.textContent = 'FOOTAGE PLACEHOLDER';
    // court surface (mat) — projected polygon of the doubles court plus run-off
    function poly(pts, cls) {
      el('polygon', { points: pts.map(function (p) { var q = project(p[0], p[1]); return q[0] + ',' + q[1]; }).join(' '), class: cls }, svg);
    }
    poly([[-1.6, -1.6], [C.W + 1.6, -1.6], [C.W + 1.6, C.L + 2.2], [-1.6, C.L + 2.2]], 'bc-mat-runoff');
    poly([[0, 0], [C.W, 0], [C.W, C.L], [0, C.L]], 'bc-mat');
    el('rect', { x: 0, y: 0, width: BC.w, height: BC.h, fill: 'url(#grain)' }, svg);
    var lines = el('g', { class: 'bc-lines' }, svg);
    function ln(x1, y1, x2, y2) {
      var a = project(x1, y1), b = project(x2, y2);
      el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], 'stroke-width': Math.max(1.6, (a[2] + b[2]) / 2 * 0.045) }, lines);
    }
    ln(0, 0, C.W, 0); ln(0, C.L, C.W, C.L); ln(0, 0, 0, C.L); ln(C.W, 0, C.W, C.L);
    ln(C.SINGLES_IN, 0, C.SINGLES_IN, C.L); ln(C.W - C.SINGLES_IN, 0, C.W - C.SINGLES_IN, C.L);
    ln(0, C.DOUBLES_LONG, C.W, C.DOUBLES_LONG); ln(0, C.L - C.DOUBLES_LONG, C.W, C.L - C.DOUBLES_LONG);
    ln(0, C.NET - C.SHORT, C.W, C.NET - C.SHORT); ln(0, C.NET + C.SHORT, C.W, C.NET + C.SHORT);
    ln(C.W / 2, 0, C.W / 2, C.NET - C.SHORT); ln(C.W / 2, C.NET + C.SHORT, C.W / 2, C.L);
    var layers = { far: el('g', null, svg), net: el('g', null, svg), near: el('g', null, svg), air: el('g', null, svg) };
    // net
    var nl = project(-0.15, C.NET), nr = project(C.W + 0.15, C.NET);
    var nlt = project(-0.15, C.NET, 1.55), nrt = project(C.W + 0.15, C.NET, 1.524);
    el('polygon', { points: [nl, nr, nrt, nlt].map(function (p) { return p[0] + ',' + p[1]; }).join(' '), class: 'bc-net' }, layers.net);
    el('line', { x1: nlt[0], y1: nlt[1], x2: nrt[0], y2: nrt[1], class: 'bc-net-tape' }, layers.net);
    el('line', { x1: nl[0], y1: nl[1], x2: nlt[0], y2: nlt[1] - 4, class: 'bc-post' }, layers.net);
    el('line', { x1: nr[0], y1: nr[1], x2: nrt[0], y2: nrt[1] - 4, class: 'bc-post' }, layers.net);
    return layers;
  }

  function figure(g, x, y, who, label) {
    var p = project(x, y), s = p[2] / 205; // ~1.75 m tall at the court's vertical scale
    var gg = el('g', { class: 'bc-player bc-player-' + who.toLowerCase(), transform: 'translate(' + p[0] + ',' + p[1] + ') scale(' + s + ')' }, g);
    el('ellipse', { cx: 0, cy: 0, rx: 34, ry: 9, class: 'bc-shadow' }, gg);
    el('rect', { x: -15, y: -128, width: 30, height: 92, rx: 13, class: 'bc-body' }, gg);
    el('rect', { x: -12, y: -42, width: 10, height: 42, rx: 5, class: 'bc-leg' }, gg);
    el('rect', { x: 2, y: -42, width: 10, height: 42, rx: 5, class: 'bc-leg' }, gg);
    el('circle', { cx: 0, cy: -146, r: 15, class: 'bc-head' }, gg);
    var t = el('text', { x: 0, y: -176, class: 'bc-tag' }, gg); t.textContent = label;
    return gg;
  }

  // positions of a player at time t inside a rally
  function playerPos(r, who, t) {
    var ready = who === 'A' ? [C.W / 2, C.NET + 3.4] : [C.W / 2, C.NET - 3.4];
    if (!r) return ready;
    var keys = [];
    var recv = r.shots[0];
    // receiver starts in the diagonal service court
    if (r.server !== who) keys.push({ t: r.start - 1, p: [C.W - recv.from[0], who === 'A' ? C.NET + 3.1 : C.NET - 3.1] });
    else keys.push({ t: r.start - 1, p: recv.from });
    r.shots.forEach(function (s) { if (s.by === who) keys.push({ t: s.t, p: s.from }); });
    // after hitting, drift back to base
    var lastK = keys[keys.length - 1];
    keys.push({ t: r.end + 2.5, p: ready });
    if (t <= keys[0].t) return keys[0].p;
    for (var i = 0; i < keys.length - 1; i++) {
      var a = keys[i], b = keys[i + 1];
      if (t >= a.t && t <= b.t) {
        var p = ease((t - a.t) / Math.max(0.01, b.t - a.t));
        return [lerp(a.p[0], b.p[0], p), lerp(a.p[1], b.p[1], p)];
      }
    }
    return keys[keys.length - 1].p || lastK.p;
  }
  var PEAK = { serve: 0.9, clear: 1, lift: 0.95, drop: 0.45, smash: 0.12, net: 0.12, drive: 0.12, push: 0.18, block: 0.1, unknown: 0.3 };
  function shuttlePos(r, t) {
    if (!r) return null;
    for (var i = 0; i < r.shots.length; i++) {
      var s = r.shots[i];
      if (t >= s.t && t <= s.t + s.dt) {
        var p = (t - s.t) / s.dt;
        var last = i === r.shots.length - 1;
        var hgt = 6.2 * PEAK[s.type] * Math.sin(Math.PI * p) + (s.type === 'smash' ? 2.6 * (1 - p) + 0.9 * p : 1.1) * (last ? (1 - p) : 1);
        return { x: lerp(s.from[0], s.to[0], p), y: lerp(s.from[1], s.to[1], p), h: Math.max(0, hgt), shot: s, i: i };
      }
    }
    var lastS = r.shots[r.shots.length - 1];
    if (t > lastS.t + lastS.dt && t < r.end + 2.5) return { x: lastS.to[0], y: lastS.to[1], h: 0, shot: lastS, landed: true };
    return null;
  }

  // ---------------------------------------------------------------- footage player
  function Player(host, store, opts) {
    opts = opts || {};
    var wrap = h('div', { class: 'player' });
    var frame = h('div', { class: 'player-frame' });
    var svg = el('svg', { viewBox: '0 0 1600 900', class: 'bc', role: 'img', 'aria-label': 'Footage placeholder: broadcast view of the court with player and shuttle positions from the sample data' });
    frame.appendChild(svg);
    var layers = drawBroadcast(svg);
    var scene = el('g', null, layers.near);
    var air = layers.air;

    // overlay: score bug, timestamp, label
    var bug = h('div', { class: 'bug', 'aria-hidden': 'true' });
    var stamp = h('div', { class: 'stamp', 'aria-hidden': 'true' });
    var label = h('div', { class: 'placeholder-label' }, ['Footage placeholder · sample still']);
    frame.appendChild(bug); frame.appendChild(stamp); frame.appendChild(label);
    wrap.appendChild(frame);

    // controls
    var playBtn = h('button', { class: 'ctl ctl-play', type: 'button', 'aria-label': 'Play' });
    playBtn.innerHTML = ICON.play;
    var prevBtn = h('button', { class: 'ctl', type: 'button', 'aria-label': 'Previous rally', title: 'Previous rally (K)' });
    prevBtn.innerHTML = ICON.prev;
    var nextBtn = h('button', { class: 'ctl', type: 'button', 'aria-label': 'Next rally', title: 'Next rally (J)' });
    nextBtn.innerHTML = ICON.next;
    var time = h('span', { class: 'ctl-time num' });
    var scrub = h('input', { type: 'range', class: 'scrub', min: 0, max: M.video.length, step: 0.1, value: 0, 'aria-label': 'Seek' });
    var gaps = h('label', { class: 'ctl-toggle' }, [h('input', { type: 'checkbox', checked: true }), ' Rallies only']);
    var controls = h('div', { class: 'player-controls' }, [playBtn, prevBtn, nextBtn, time, scrub, gaps]);
    wrap.appendChild(controls);
    host.appendChild(wrap);

    var skipGaps = true;
    gaps.querySelector('input').addEventListener('change', function (e) { skipGaps = e.target.checked; });

    var raf = null, lastTs = null;
    function tick(ts) {
      var st = store.get();
      if (!st.playing) { raf = null; lastTs = null; return; }
      var dt = lastTs == null ? 0 : (ts - lastTs) / 1000;
      lastTs = ts;
      var t = st.time + dt;
      if (skipGaps) {
        var r = rallyAt(t);
        if (r && t > r.end + 1.8) {
          var nx = M.rallies[r.index + 1];
          if (nx) t = nx.start - 1.0;
        }
      }
      if (st.stopAt != null && t >= st.stopAt) { store.set({ time: st.stopAt, playing: false, stopAt: null }); raf = null; return; }
      if (t >= M.video.length) { store.set({ time: M.video.length, playing: false }); raf = null; return; }
      var cur = rallyAt(t);
      var patch = { time: t };
      if (cur && cur.index !== st.selected && t >= cur.start - 0.5) patch.selected = cur.index;
      store.set(patch);
      raf = requestAnimationFrame(tick);
    }

    function render(st) {
      var t = st.time;
      var r = rallyAt(t);
      clear(scene); clear(air); clear(layers.far);
      var posA = playerPos(r, 'A', t), posB = playerPos(r, 'B', t);
      figure(layers.far, posB[0], posB[1], 'B', P('B').code);
      figure(scene, posA[0], posA[1], 'A', P('A').code);
      var sh = shuttlePos(r, t);
      if (sh) {
        var g = project(sh.x, sh.y), a = project(sh.x, sh.y, sh.h);
        el('ellipse', { cx: g[0], cy: g[1], rx: 5 * g[2] / 92, ry: 2 * g[2] / 92, class: 'bc-shuttle-shadow' }, air);
        el('circle', { cx: a[0], cy: a[1], r: Math.max(4, 7 * a[2] / 92), class: 'bc-shuttle' }, air);
        if (sh.landed) {
          if (r.outcome === 'winner') el('ellipse', { cx: g[0], cy: g[1], rx: 30 * g[2] / 92, ry: 11 * g[2] / 92, class: 'bc-land-ring' }, air);
        }
      }
      // score bug
      var sb = r ? (t > r.end + 0.8 ? r.scoreAfter : r.scoreBefore) : { A: 0, B: 0 };
      var g2 = r ? r.game : 1;
      var srv = r ? (t > r.end + 0.8 ? r.winner : r.server) : 'A';
      var prevGames = M.games.filter(function (x) { return x.n < g2; });
      var gwA = prevGames.filter(function (x) { return x.winner === 'A'; }).length;
      var gwB = prevGames.length - gwA;
      bug.innerHTML =
        '<span class="bug-g">G' + g2 + '</span>' +
        '<span class="bug-row' + (srv === 'A' ? ' srv' : '') + '"><b>' + P('A').code + '</b><i>' + gwA + '</i><em>' + sb.A + '</em></span>' +
        '<span class="bug-row' + (srv === 'B' ? ' srv' : '') + '"><b>' + P('B').code + '</b><i>' + gwB + '</i><em>' + sb.B + '</em></span>';
      stamp.textContent = fmt(t) + (r ? '  ·  Rally ' + r.id.replace('-', '·') : '');
      time.textContent = fmt(t) + ' / ' + fmt(M.video.length);
      if (document.activeElement !== scrub) scrub.value = t;
      playBtn.innerHTML = st.playing ? ICON.pause : ICON.play;
      playBtn.setAttribute('aria-label', st.playing ? 'Pause' : 'Play');
    }

    function play() {
      var st = store.get();
      store.set({ playing: true });
      if (!raf) raf = requestAnimationFrame(tick);
    }
    function pause() { store.set({ playing: false }); }
    playBtn.addEventListener('click', function () { store.get().playing ? pause() : play(); });
    function step(d) {
      var st = store.get();
      var list = ralliesFor(st.game);
      var curIdx = st.selected == null ? -1 : list.findIndex(function (r) { return r.index === st.selected; });
      var nx = list[Math.max(0, Math.min(list.length - 1, curIdx + d))];
      if (nx) GM.select(store, nx.index);
    }
    prevBtn.addEventListener('click', function () { step(-1); });
    nextBtn.addEventListener('click', function () { step(1); });
    scrub.addEventListener('input', function () {
      var t = +scrub.value; var r = rallyAt(t);
      store.set({ time: t, selected: r ? r.index : null, stopAt: null });
    });

    store.on(function (st, prev) {
      render(st);
      if (st.playing && !raf) raf = requestAnimationFrame(tick);
    });
    render(store.get());
    return { play: play, pause: pause, step: step, el: wrap };
  }

  // ---------------------------------------------------------------- timeline
  function Timeline(host, store, opts) {
    opts = opts || {};
    var wrap = h('div', { class: 'timeline', tabindex: 0, role: 'group', 'aria-roledescription': 'rally timeline', 'aria-label': 'Rally timeline. Use left and right arrow keys to move between rallies, Enter to play.' });
    var live = h('div', { class: 'sr-only', 'aria-live': 'polite' });
    var tip = h('div', { class: 'tl-tip', role: 'presentation' });
    host.style.position = 'relative';
    host.appendChild(wrap); host.appendChild(live); host.appendChild(tip);
    var svg = el('svg', { class: 'tl-svg' }, wrap);
    var W = 0, geo = null;

    function layout(st) {
      var list = ralliesFor(st.game);
      var games = st.game === 'all' ? M.games : M.games.filter(function (g) { return g.n === st.game; });
      W = Math.max(320, wrap.clientWidth);
      var compact = W < 640;
      var Hm = compact ? 34 : 44, gap = compact ? 10 : 22;
      var maxLen = 34;
      var barH = compact ? 56 : 78;
      var top = 22, mid = top + Hm + 26 + barH, H = mid + barH + 34;
      var count = list.length;
      var gutter = compact ? 54 : 74; // right gutter carries the axis labels
      var usable = W - 8 - gutter - gap * (games.length - 1);
      var bw = usable / count;
      // phones: never squeeze a rally below a tappable slot; the strip scrolls instead
      var MIN_BW = compact ? 11 : 0;
      if (bw < MIN_BW) { bw = MIN_BW; W = 8 + gutter + gap * (games.length - 1) + bw * count; }
      var xs = {}, x = 4, gx = [];
      games.forEach(function (g, gi) {
        var start = x;
        for (var i = g.first; i <= g.last; i++) { xs[i] = x; x += bw; }
        gx.push({ g: g, x0: start, x1: x });
        x += gap;
      });
      return { list: list, games: games, gx: gx, xs: xs, bw: bw, Hm: Hm, top: top, mid: mid, barH: barH, H: H, maxLen: maxLen, compact: compact };
    }

    function draw() {
      var st = store.get();
      geo = layout(st);
      clear(svg);
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + geo.H);
      svg.setAttribute('width', W); svg.setAttribute('height', geo.H);
      var g = geo, mTop = g.top, mH = g.Hm;
      // per-game frames
      g.gx.forEach(function (gg) {
        var game = gg.g;
        var lbl = el('text', { x: gg.x0, y: 12, class: 'tl-game' }, svg);
        lbl.textContent = 'G' + game.n + '  ' + game.score.A + '–' + game.score.B;
        // margin strip: step line of A minus B
        var mid0 = mTop + mH / 2;
        el('line', { x1: gg.x0, y1: mid0, x2: gg.x1, y2: mid0, class: 'tl-zero' }, svg);
        var d = 'M' + gg.x0 + ',' + mid0, m = 0;
        for (var i = game.first; i <= game.last; i++) {
          var r = M.rallies[i];
          m = r.scoreAfter.A - r.scoreAfter.B;
          var y = mid0 - Math.max(-8, Math.min(8, m)) / 8 * (mH / 2);
          d += 'H' + (g.xs[i] + g.bw / 2) + 'V' + y;
        }
        d += 'H' + gg.x1;
        el('path', { d: d, class: 'tl-margin' }, svg);
        // baseline for bars
        el('line', { x1: gg.x0, y1: g.mid, x2: gg.x1, y2: g.mid, class: 'tl-base' }, svg);
        // interval tick at 11
        for (var j = game.first; j <= game.last; j++) {
          if (M.rallies[j].interval) {
            var ix = g.xs[j] + g.bw;
            el('line', { x1: ix, y1: g.mid - g.barH - 4, x2: ix, y2: g.mid + g.barH + 4, class: 'tl-interval' }, svg);
            var it = el('text', { x: ix + 3, y: g.mid + g.barH + 2, class: 'tl-note' }, svg); it.textContent = '11';
          }
        }
      });
      // axis legends (left)
      var gxe = W - 2;
      var la = el('text', { x: gxe, y: g.mid - g.barH + 8, class: 'tl-axis', 'text-anchor': 'end' }, svg); la.textContent = P('A').code + ' won ↑';
      var lb = el('text', { x: gxe, y: g.mid + g.barH - 2, class: 'tl-axis', 'text-anchor': 'end' }, svg); lb.textContent = P('B').code + ' won ↓';
      var lm = el('text', { x: gxe, y: mTop + 9, class: 'tl-axis', 'text-anchor': 'end' }, svg); lm.textContent = P('A').code + ' ahead';
      var lm2 = el('text', { x: gxe, y: mTop + mH - 2, class: 'tl-axis', 'text-anchor': 'end' }, svg); lm2.textContent = P('B').code + ' ahead';
      // runs brackets
      M.runs.forEach(function (run) {
        if (g.xs[run.from] == null) return;
        var x0 = g.xs[run.from], x1 = g.xs[run.to] + g.bw - 1;
        var up = run.player === 'A';
        var tallest = 0;
        for (var q = run.from; q <= run.to; q++) tallest = Math.max(tallest, Math.min(1, M.rallies[q].length / g.maxLen) * g.barH);
        var y = up ? g.mid - tallest - 7 : g.mid + tallest + 7;
        el('path', { d: 'M' + x0 + ',' + (y + (up ? 5 : -5)) + 'V' + y + 'H' + x1 + 'V' + (y + (up ? 5 : -5)), class: 'tl-run' }, svg);
        if (x1 - x0 > 22) {
          var t = el('text', { x: (x0 + x1) / 2, y: up ? y - 4 : y + 12, class: 'tl-run-label', 'text-anchor': 'middle' }, svg);
          t.textContent = run.len + '–0';
        }
      });
      // bars
      g.list.forEach(function (r) {
        var x = g.xs[r.index];
        var hgt = Math.max(3, Math.min(1, r.length / g.maxLen) * g.barH);
        var up = r.winner === 'A';
        var dim = st.player !== 'both' && r.winner !== st.player;
        var cls = 'tl-bar' + (up ? ' a' : ' b') + (r.index === st.selected ? ' sel' : '') + (dim ? ' dim' : '') + (r.tracking === 'partial' ? ' partial' : '');
        var w = Math.max(1.5, g.bw - (g.bw > 6 ? 2 : 0.8));
        el('rect', { x: x, y: up ? g.mid - hgt : g.mid, width: w, height: hgt, class: cls, 'data-i': r.index }, svg);
      });
      // turning points
      M.turningPoints.forEach(function (tp) {
        if (g.xs[tp.from] == null) return;
        var cx = (g.xs[tp.from] + g.xs[tp.to] + g.bw) / 2;
        var cy = mTop + mH + 13;
        var mk = el('g', { class: 'tl-tp' }, svg);
        el('circle', { cx: cx, cy: cy, r: 8.5, class: 'tl-tp-dot' }, mk);
        var tt = el('text', { x: cx, y: cy + 3.6, class: 'tl-tp-num', 'text-anchor': 'middle' }, mk); tt.textContent = tp.label;
      });
      // selected rally pointer
      if (st.selected != null && g.xs[st.selected] != null) {
        var sx = g.xs[st.selected] + g.bw / 2;
        if (wrap.scrollWidth > wrap.clientWidth + 2) {
          var vis0 = wrap.scrollLeft, vis1 = vis0 + wrap.clientWidth;
          if (sx < vis0 + 40 || sx > vis1 - 80) wrap.scrollLeft = Math.max(0, sx - wrap.clientWidth / 2);
        }
        el('line', { x1: sx, y1: mTop, x2: sx, y2: g.H - 18, class: 'tl-cursor' }, svg);
        var r = M.rallies[st.selected];
        var lab = el('text', { x: Math.min(W - 4, Math.max(4, sx)), y: g.H - 4, class: 'tl-cursor-label', 'text-anchor': sx < 60 ? 'start' : (sx > W - 60 ? 'end' : 'middle') }, svg);
        lab.textContent = score(r.scoreBefore) + ' → ' + score(r.scoreAfter) + ' · ' + fmt(r.start);
      }
    }

    function idxAt(clientX) {
      var rect = svg.getBoundingClientRect();
      var x = clientX - rect.left;
      var best = null, bd = 1e9;
      geo.list.forEach(function (r) {
        var cx = geo.xs[r.index] + geo.bw / 2, d = Math.abs(cx - x);
        if (d < bd) { bd = d; best = r; }
      });
      return bd < Math.max(10, geo.bw) ? best : null;
    }
    svg.addEventListener('pointermove', function (e) {
      var r = idxAt(e.clientX);
      if (!r) { tip.classList.remove('on'); return; }
      tip.innerHTML = '<b class="num">' + r.id.replace('-', '·') + '</b> ' + score(r.scoreBefore) + ' → <b class="num">' + score(r.scoreAfter) + '</b><br>' + plural(r.length, 'shot') + ' · ' + endingText(r) + '<br><span class="num">' + fmt(r.start) + '</span>' + (r.tracking === 'partial' ? ' · partial tracking' : '');
      var rect = host.getBoundingClientRect(), wr = wrap.getBoundingClientRect();
      var x = e.clientX - rect.left;
      tip.style.left = Math.max(0, Math.min(rect.width - 230, x - 115)) + 'px';
      tip.style.top = (wr.top - rect.top - 8) + 'px';
      tip.classList.add('on');
    });
    svg.addEventListener('pointerleave', function () { tip.classList.remove('on'); });
    svg.addEventListener('click', function (e) {
      var r = idxAt(e.clientX);
      if (r) GM.select(store, r.index);
    });
    wrap.addEventListener('keydown', function (e) {
      var st = store.get(), list = geo.list;
      var cur = list.findIndex(function (r) { return r.index === st.selected; });
      var nx = null;
      if (e.key === 'ArrowRight') nx = list[Math.min(list.length - 1, cur + 1)];
      else if (e.key === 'ArrowLeft') nx = list[Math.max(0, cur < 0 ? 0 : cur - 1)];
      else if (e.key === 'Home') nx = list[0];
      else if (e.key === 'End') nx = list[list.length - 1];
      else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); if (st.selected != null) GM.select(store, st.selected); return; }
      if (nx) { e.preventDefault(); store.set({ selected: nx.index, time: nx.start - 1, playing: false }, { trace: true }); }
    });

    var ro = new ResizeObserver(function () { draw(); });
    ro.observe(wrap);
    var lastSel = null;
    store.on(function (st, prev) {
      if (st.game !== prev.game || st.player !== prev.player || st.selected !== prev.selected) draw();
      if (st.selected !== lastSel && st.selected != null) {
        var r = M.rallies[st.selected]; lastSel = st.selected;
        live.textContent = 'Rally ' + r.id.replace('-', ' point ') + ', ' + score(r.scoreBefore) + ' to ' + score(r.scoreAfter) + ', ' + plural(r.length, 'shot') + ', ' + endingText(r) + ', at ' + fmt(r.start);
      }
    });
    draw();
    return { redraw: draw };
  }

  // ---------------------------------------------------------------- rally detail
  function RallyDetail(host, store, opts) {
    var box = h('div', { class: 'rally-detail', 'aria-live': 'polite' });
    host.appendChild(box);
    function render(st) {
      var r = st.selected != null ? M.rallies[st.selected] : null;
      if (!r) { box.innerHTML = '<p class="muted">Select a rally on the timeline to see its path on the court.</p>'; return; }
      var last = r.shots[r.shots.length - 1];
      var types = {};
      r.shots.forEach(function (s) { types[s.type] = (types[s.type] || 0) + 1; });
      var seq = r.shots.map(function (s) { return '<span class="seq-' + s.by.toLowerCase() + (s.tracked ? '' : ' seq-untracked') + '" title="' + SHOT_NAME[s.type] + ' by ' + P(s.by).short + (s.tracked ? '' : ' (not tracked; estimated)') + '">' + SHOT_NAME[s.type].charAt(0) + '</span>'; }).join('');
      box.innerHTML =
        '<div class="rd-head"><span class="rd-id num">Rally ' + r.id.replace('-', '·') + '</span>' +
        '<span class="rd-score num">' + score(r.scoreBefore) + ' <span class="arrow-ic">' + ICON.arrow + '</span> ' + score(r.scoreAfter) + '</span></div>' +
        '<dl class="rd-facts">' +
        '<div><dt>Point</dt><dd>' + P(r.winner).short + '</dd></div>' +
        '<div><dt>Shots</dt><dd class="num">' + r.length + '</dd></div>' +
        '<div><dt>Ended</dt><dd>' + endingText(r).split(' · ')[0] + '</dd></div>' +
        '<div><dt>Starts</dt><dd class="num">' + fmt(r.start) + '</dd></div>' +
        '</dl>' +
        '<div class="rd-seq" aria-label="Shot sequence">' + seq + '</div>' +
        (r.tracking === 'partial' ? '<p class="rd-warn">Tracking lost the shuttle for part of this rally. Dotted segments are estimated.</p>' : '');
    }
    store.on(function (st, prev) { if (st.selected !== prev.selected) render(st); });
    render(store.get());
  }

  // ---------------------------------------------------------------- patterns
  function zoneOf(x, y, half) {
    // returns [col 0..2 from screen-left, depth 0 front..2 rear] within half
    var col = x < 2.03 ? 0 : (x < 4.07 ? 1 : 2);
    var d = half === 'far' ? C.NET - y : y - C.NET;
    var dep = d < 1.98 ? 0 : (d < 4.6 ? 1 : 2);
    return [col, dep];
  }

  function Placement(host, store) {
    var wrap = h('div', { class: 'placement' });
    host.appendChild(wrap);
    function render(st) {
      clear(wrap);
      var who = st.player === 'both' ? ['A', 'B'] : [st.player];
      who.forEach(function (id) {
        var half = id === 'A' ? 'far' : 'near';
        var grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], total = 0, wins = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        ralliesFor(st.game).forEach(function (r) {
          r.shots.forEach(function (s, i) {
            if (s.by !== id || s.result === 'net' || s.result === 'out' || !s.tracked) return;
            var z = zoneOf(s.to[0], s.to[1], half);
            grid[z[1]][z[0]]++; total++;
            if (i === r.shots.length - 1 && r.outcome === 'winner') wins[z[1]][z[0]]++;
          });
        });
        var max = Math.max.apply(null, grid.map(function (row) { return Math.max.apply(null, row); }));
        var fig = h('figure', { class: 'pl-fig' });
        var svg = el('svg', { viewBox: '-0.3 -0.9 6.7 8.2', class: 'pl-svg', role: 'img', 'aria-label': P(id).short + ' shot landings by zone' });
        // draw opponent half (always shown with net at top for reading)
        var g = el('g', null, svg);
        for (var dep = 0; dep < 3; dep++) for (var col = 0; col < 3; col++) {
          var x = col * (C.W / 3), y = [0, 1.98, 4.6][dep], hh = [1.98, 2.62, 2.1][dep];
          var v = grid[dep][col], a = max ? v / max : 0;
          el('rect', { x: x, y: y, width: C.W / 3, height: hh, class: 'pl-cell', style: 'fill-opacity:' + (0.06 + a * 0.62).toFixed(3) }, g);
          var pct = total ? Math.round(v / total * 100) : 0;
          var tx = el('text', { x: x + C.W / 6, y: y + hh / 2 + 0.05, class: 'pl-pct' + (a > 0.62 ? ' on-dark' : ''), 'text-anchor': 'middle' }, g);
          tx.textContent = pct + '%';
          if (wins[dep][col]) {
            var wt = el('text', { x: x + C.W / 6, y: y + hh / 2 + 0.5, class: 'pl-win' + (a > 0.62 ? ' on-dark' : ''), 'text-anchor': 'middle' }, g);
            wt.textContent = plural(wins[dep][col], 'winner');
          }
        }
        el('rect', { x: 0, y: 0, width: C.W, height: 6.7, class: 'pl-outline' }, g);
        el('line', { x1: -0.2, y1: 0, x2: C.W + 0.2, y2: 0, class: 'pl-net' }, g);
        var nt = el('text', { x: C.W / 2, y: -0.3, class: 'pl-cap', 'text-anchor': 'middle' }, g); nt.textContent = 'NET';
        var bt = el('text', { x: C.W / 2, y: 7.2, class: 'pl-cap', 'text-anchor': 'middle' }, g); bt.textContent = 'BACK OF ' + P(other(id)).code + ' COURT';
        fig.appendChild(svg);
        fig.appendChild(h('figcaption', { html: '<b>' + P(id).short + '</b> · ' + total + ' tracked landings' }));
        wrap.appendChild(fig);
      });
    }
    store.on(function (st, prev) { if (st.game !== prev.game || st.player !== prev.player) render(st); });
    render(store.get());
  }

  function RallyLengths(host, store) {
    var wrap = h('div', { class: 'lengths' });
    host.appendChild(wrap);
    var buckets = [[1, 4], [5, 8], [9, 12], [13, 16], [17, 20], [21, 99]];
    function render(st) {
      clear(wrap);
      var list = ralliesFor(st.game);
      var data = buckets.map(function (b) {
        var inB = list.filter(function (r) { return r.length >= b[0] && r.length <= b[1]; });
        return { b: b, A: inB.filter(function (r) { return r.winner === 'A'; }).length, B: inB.filter(function (r) { return r.winner === 'B'; }).length };
      });
      var max = Math.max.apply(null, data.map(function (d) { return Math.max(d.A, d.B); })) || 1;
      var W = 400, rowH = 74, bw = W / buckets.length;
      var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + (rowH * 2 + 40), class: 'len-svg', role: 'img', 'aria-label': 'Points won by rally length' });
      var mid = rowH + 6;
      el('line', { x1: 0, y1: mid, x2: W, y2: mid, class: 'tl-base' }, svg);
      data.forEach(function (d, i) {
        var x = i * bw + 10, w = bw - 20;
        var ha = d.A / max * (rowH - 16), hb = d.B / max * (rowH - 16);
        var dimA = st.player === 'B', dimB = st.player === 'A';
        el('rect', { x: x, y: mid - ha, width: w, height: ha, class: 'tl-bar a' + (dimA ? ' dim' : '') }, svg);
        el('rect', { x: x, y: mid, width: w, height: hb, class: 'tl-bar b' + (dimB ? ' dim' : '') }, svg);
        var ta = el('text', { x: x + w / 2, y: mid - ha - 5, class: 'len-n', 'text-anchor': 'middle' }, svg); ta.textContent = d.A;
        var tb = el('text', { x: x + w / 2, y: mid + hb + 15, class: 'len-n', 'text-anchor': 'middle' }, svg); tb.textContent = d.B;
        var tl = el('text', { x: x + w / 2, y: rowH * 2 + 36, class: 'len-cap', 'text-anchor': 'middle' }, svg);
        tl.textContent = d.b[1] > 90 ? d.b[0] + '+' : d.b[0] + '–' + d.b[1];
      });
      var l1 = el('text', { x: 0, y: 12, class: 'tl-axis' }, svg); l1.textContent = P('A').code + ' won ↑';
      var l2 = el('text', { x: 0, y: rowH * 2 + 8, class: 'tl-axis' }, svg); l2.textContent = P('B').code + ' won ↓';
      wrap.appendChild(svg);
      wrap.appendChild(h('p', { class: 'chart-x', text: 'Shots per rally' }));
    }
    store.on(function (st, prev) { if (st.game !== prev.game || st.player !== prev.player) render(st); });
    render(store.get());
  }

  function Ledger(host, store) {
    var wrap = h('div', { class: 'ledger' });
    host.appendChild(wrap);
    function render(st) {
      var list = ralliesFor(st.game);
      function stat(id) {
        var won = list.filter(function (r) { return r.winner === id; });
        var winners = won.filter(function (r) { return r.outcome === 'winner'; }).length;
        var mine = list.filter(function (r) { return r.outcome === 'error' && r.endedBy === id; });
        var net = mine.filter(function (r) { return r.shots[r.shots.length - 1].result === 'net'; }).length;
        var longest = won.reduce(function (m, r) { return Math.max(m, r.length); }, 0);
        var serveWon = list.filter(function (r) { return r.server === id; });
        var sw = serveWon.filter(function (r) { return r.winner === id; }).length;
        return { pts: won.length, winners: winners, err: mine.length, net: net, out: mine.length - net, longest: longest, serve: serveWon.length ? Math.round(sw / serveWon.length * 100) : 0 };
      }
      var a = stat('A'), b = stat('B');
      var rows = [
        ['Points won', a.pts, b.pts],
        ['Winners', a.winners, b.winners],
        ['Errors', a.err, b.err],
        ['— into the net', a.net, b.net],
        ['— out', a.out, b.out],
        ['Points won on own serve', a.serve + '%', b.serve + '%'],
        ['Longest rally won', a.longest + ' shots', b.longest + ' shots']
      ];
      var dimA = st.player === 'B' ? ' dim' : '', dimB = st.player === 'A' ? ' dim' : '';
      wrap.innerHTML = '<table class="ledger-t"><caption class="sr-only">Measured statistics</caption><thead><tr><th scope="col">' + (M.isLive ? 'Estimated' : 'Measured') + '</th><th scope="col" class="num' + dimA + '">' + P('A').code + '</th><th scope="col" class="num' + dimB + '">' + P('B').code + '</th></tr></thead><tbody>' +
        rows.map(function (r) { return '<tr' + (r[0].charAt(0) === '—' ? ' class="sub"' : '') + '><th scope="row">' + r[0] + '</th><td class="num' + dimA + '">' + r[1] + '</td><td class="num' + dimB + '">' + r[2] + '</td></tr>'; }).join('') +
        '</tbody></table>';
    }
    store.on(function (st, prev) { if (st.game !== prev.game || st.player !== prev.player) render(st); });
    render(store.get());
  }

  // ---------------------------------------------------------------- icons (authored, 1.6 stroke on 20 grid)
  var ICON = {
    play: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6 4.5v11l9-5.5z" fill="currentColor"/></svg>',
    pause: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6 4.5h2.6v11H6zM11.4 4.5H14v11h-2.6z" fill="currentColor"/></svg>',
    prev: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 4.5v11M15 4.5l-7.5 5.5L15 15.5z" fill="currentColor" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>',
    next: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M15 4.5v11M5 4.5l7.5 5.5L5 15.5z" fill="currentColor" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>',
    clip: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 5.5h14v9H3z M8.5 8l3.5 2-3.5 2z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>',
    mark: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5.5 3.5h9v13l-4.5-3.2-4.5 3.2z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>',
    marked: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5.5 3.5h9v13l-4.5-3.2-4.5 3.2z" fill="currentColor" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>',
    arrow: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3.5 10h12.6M11 5.2l5 4.8-5 4.8" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="square" stroke-linejoin="round"/></svg>'
  };

  // shared print-style hatch for the far player's bars (solid ink = LIN, hatch = OKA)
  (function defs() {
    var s = el('svg', { width: 0, height: 0, 'aria-hidden': 'true', style: 'position:absolute' });
    var d = el('defs', null, s);
    var p = el('pattern', { id: 'gm-hatch', width: 4, height: 4, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, d);
    el('rect', { width: 4, height: 4, style: 'fill:var(--hatch-bg, var(--paper))' }, p);
    el('line', { x1: 0, y1: 0, x2: 0, y2: 4, style: 'stroke:var(--hatch-line, var(--ink))', 'stroke-width': 1.6 }, p);
    var add = function () { document.body.insertBefore(s, document.body.firstChild); };
    if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
  })();

  // ---------------------------------------------------------------- selection
  var GM = window.GM = window.GM || {};
  GM.ICON = ICON;
  GM.h = h; GM.el = el; GM.fmt = fmt; GM.score = score; GM.P = P; GM.plural = plural; GM.endingText = endingText;
  GM.reduceMotion = reduceMotion;
  GM.Store = Store; GM.CourtMap = CourtMap; GM.Player = Player; GM.Timeline = Timeline; GM.RallyDetail = RallyDetail;
  GM.Placement = Placement; GM.RallyLengths = RallyLengths; GM.Ledger = Ledger; GM.drawCourtLines = drawCourtLines;
  GM.project = project; GM.drawBroadcast = drawBroadcast; GM.figure = figure; GM.ralliesFor = ralliesFor;
  GM.playerPos = playerPos; GM.shuttlePos = shuttlePos; GM.rallyAt = rallyAt; GM.SHOT_NAME = SHOT_NAME; GM.zoneOf = zoneOf; GM.PEAK = PEAK;

  /* Select a rally: seek the clip to it, light its path, and play it
   * (paused under reduced motion so nothing moves on its own). */
  GM.select = function (store, index, opts) {
    opts = opts || {};
    var r = M.rallies[index];
    if (!r) return;
    var autoplay = opts.play !== false && !reduceMotion;
    store.set({ selected: index, time: r.start - 1.0, playing: autoplay, stopAt: autoplay ? r.end + 1.6 : null }, { trace: !autoplay });
  };

  GM.newStore = function (init) {
    return Store(Object.assign({ game: 'all', player: 'both', selected: null, time: M.rallies[0].start - 1, playing: false, stopAt: null }, init || {}));
  };

  // global keyboard: J/K next/prev rally, space play when not in a control
  GM.bindKeys = function (store, player) {
    document.addEventListener('keydown', function (e) {
      var tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 'j') { player.step(1); }
      else if (e.key === 'k') { player.step(-1); }
    });
  };
})();
