/* Goodminton — sample match data
 * ---------------------------------------------------------------
 * ILLUSTRATIVE DATA. Fictional players, generated deterministically.
 * Shape mirrors what a backend should return from GET /api/matches/:id
 * so this whole file can be replaced by a fetch() later:
 *
 *   window.GM_MATCH = await (await fetch('/api/matches/sample')).json();
 *
 * Coordinates are metres on a doubles court: x 0–6.1 (left→right as seen
 * from the camera), y 0–13.4 (far baseline → near baseline). Net at y 6.7.
 * Player A plays the near half (y > 6.7), player B the far half.
 * --------------------------------------------------------------- */
(function () {
  'use strict';

  var COURT = {
    W: 6.1, L: 13.4, NET: 6.7,
    SINGLES_IN: 0.46, SHORT: 1.98, DOUBLES_LONG: 0.76
  };

  /* Upload & processing limits. PROVISIONAL until the backend measures them. */
  var LIMITS = {
    provisional: true,
    formats: ['MP4', 'MOV'],
    mime: ['video/mp4', 'video/quicktime'],
    maxSizeGB: 4,
    maxDurationMin: 120,
    processingEstimate: null, // unknown until measured — UI must say so
    recording: [
      'One fixed camera, raised, behind a baseline',
      'All four corners of the court in frame',
      '720p or higher, 30 fps or higher',
      'Singles matches only in this version'
    ]
  };

  // ---------- deterministic PRNG ----------
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  var rnd = mulberry32(20261107);
  function r(a, b) { return a + (b - a) * rnd(); }
  function ri(a, b) { return Math.floor(r(a, b + 1)); }
  function pick(arr) { return arr[Math.floor(rnd() * arr.length)]; }
  function weighted(pairs) {
    var total = 0, i;
    for (i = 0; i < pairs.length; i++) total += pairs[i][1];
    var x = rnd() * total;
    for (i = 0; i < pairs.length; i++) { x -= pairs[i][1]; if (x <= 0) return pairs[i][0]; }
    return pairs[pairs.length - 1][0];
  }
  function shuffle(arr) {
    for (var i = arr.length - 1; i > 0; i--) {
      var j = Math.floor(rnd() * (i + 1)); var t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }
  // interleave points so no accidental run longer than 3 competes with
  // the authored turning points
  function fill(nA, nB) {
    var a = [], i, s, tries = 0;
    for (i = 0; i < nA; i++) a.push('A');
    for (i = 0; i < nB; i++) a.push('B');
    do { s = shuffle(a).join(''); tries++; } while (/AAAA|BBBB/.test(s) && tries < 500);
    return s;
  }
  function repeat(c, n) { return new Array(n + 1).join(c); }

  // ---------- point-winner sequences (authored structure) ----------
  // G1: A 21–18. Lindqvist closes from 17–18 with four straight.
  var g1 = fill(17, 17) + 'B' + repeat('A', 4);
  // G2: B 21–17. Okafor's six straight from 11–11.
  var g2 = fill(10, 11) + 'A' + repeat('B', 6) + 'A' + fill(5, 3) + 'B';
  // G3: A 21–16. Lindqvist's seven straight from 8–12.
  var g3 = fill(8, 11) + 'B' + repeat('A', 7) + 'B' + fill(5, 3) + 'A';

  // segments that get biased shot patterns (index ranges per game are
  // resolved by score state below)
  var BIAS = {
    g2run: { game: 2, from: [11, 11], to: [11, 17] },
    g3run: { game: 3, from: [8, 12], to: [15, 12] }
  };

  // ---------- shot model ----------
  var SHOT_TIME = { serve: 1.0, clear: 1.35, lift: 1.15, drop: 1.0, smash: 0.55, net: 0.8, drive: 0.6, push: 0.75, block: 0.7 };
  var SHOT_PEAK = { serve: 0.9, clear: 1, lift: 0.95, drop: 0.45, smash: 0.18, net: 0.12, drive: 0.15, push: 0.2, block: 0.12 };

  // depth bands within a half, measured from the net
  function depthY(half, band) {
    // half: 'far' (y<6.7) or 'near' (y>6.7); band: front | mid | rear
    var d;
    if (band === 'front') d = r(0.25, 1.9);
    else if (band === 'mid') d = r(2.2, 4.4);
    else d = r(4.9, 6.45);
    return half === 'far' ? COURT.NET - d : COURT.NET + d;
  }
  function inX() { return r(0.75, 5.35); }

  function nextType(prevBand, isLast) {
    if (prevBand === 'rear') return weighted([['clear', 4], ['drop', 4], ['smash', isLast ? 5 : 3]]);
    if (prevBand === 'front') return weighted([['net', 4], ['lift', 5], ['push', 2]]);
    return weighted([['drive', 3], ['block', 3], ['push', 2], ['lift', 2]]);
  }
  function bandFor(type) {
    switch (type) {
      case 'clear': case 'lift': return 'rear';
      case 'drop': case 'net': case 'block': return 'front';
      case 'smash': return pick(['mid', 'rear', 'mid']);
      default: return 'mid';
    }
  }

  function buildRally(ctx) {
    // ctx: server, winner, scoreBefore, bias, game
    var server = ctx.server, receiver = server === 'A' ? 'B' : 'A';
    var outcome, n;
    if (ctx.bias === 'g2run') outcome = 'winner';
    else outcome = rnd() < 0.46 ? 'winner' : 'error';

    if (ctx.bias === 'g3run') n = ri(13, 23);
    else if (ctx.bias === 'g2run') n = ri(5, 9);
    else n = Math.max(1, Math.round(weighted([[3, 2], [5, 4], [7, 5], [9, 5], [11, 4], [13, 3], [17, 2], [23, 1], [31, 0.4]]) + r(-1.4, 1.4)));

    // parity: last hitter = server when n is odd
    var wantLast = outcome === 'winner' ? ctx.winner : (ctx.winner === 'A' ? 'B' : 'A');
    var lastIsServer = (n % 2) === 1;
    if ((lastIsServer ? server : receiver) !== wantLast) n += (n > 2 && rnd() < 0.5) ? -1 : 1;
    if (n < 2 && outcome === 'winner' && wantLast !== server) n = 2;

    var shots = [], hitter = server;
    var serverScore = ctx.scoreBefore[server];
    // right service court when server's score is even
    var fromRight = serverScore % 2 === 0;
    var half = server === 'A' ? 'near' : 'far';
    // "right" for near player (facing up) is large x; for far player, small x
    var sx = (server === 'A') === fromRight ? r(3.4, 4.6) : r(1.5, 2.7);
    var from = [sx, half === 'near' ? COURT.NET + r(2.1, 2.6) : COURT.NET - r(2.1, 2.6)];
    var prevBand = 'mid';

    for (var i = 0; i < n; i++) {
      var isLast = i === n - 1;
      var type, band, to, result = 'in';
      var targetHalf = hitter === 'A' ? 'far' : 'near';
      if (i === 0) {
        type = 'serve';
        band = rnd() < 0.7 ? 'rear' : 'front';
        // diagonal: lands on the opposite side in screen x
        var tx = sx > 3.05 ? r(0.7, 2.7) : r(3.4, 5.4);
        to = [tx, depthY(targetHalf, band)];
      } else {
        type = nextType(prevBand, isLast);
        band = bandFor(type);
        to = [inX(), depthY(targetHalf, band)];
      }

      if (isLast && ctx.bias === 'g2run' && hitter === 'B' && ctx.scoreBefore.B > 11) {
        // Okafor's run: finishing to Lindqvist's rear backhand (screen-left, near rear)
        type = i === 0 ? 'serve' : pick(['smash', 'clear', 'smash', 'drop']);
        if (type === 'drop') type = 'smash';
        to = [r(0.6, 2.0), COURT.NET + r(4.7, 6.4)];
        band = 'rear';
      }

      if (isLast && outcome === 'error') {
        if (rnd() < 0.45) {
          result = 'net';
          to = [r(1, 5.1), hitter === 'A' ? COURT.NET + 0.12 : COURT.NET - 0.12];
        } else {
          result = 'out';
          var side = rnd();
          if (side < 0.45) to = [to[0], targetHalf === 'far' ? r(-0.55, -0.12) : COURT.L + r(0.12, 0.55)];
          else to = [rnd() < 0.5 ? r(0.02, 0.36) : r(5.74, 6.05), to[1]];
        }
      }

      var tracked = !(rnd() < 0.009 && n > 4 && !isLast);
      shots.push({
        i: i, by: hitter, type: type,
        from: [round(from[0]), round(from[1])],
        to: [round(to[0]), round(to[1])],
        result: isLast ? (outcome === 'winner' ? 'winner' : result) : 'in',
        tracked: tracked,
        dt: SHOT_TIME[type] * r(0.88, 1.12)
      });
      prevBand = band;
      from = [to[0] + r(-0.25, 0.25), to[1] + (hitter === 'A' ? -r(0, 0.35) : r(0, 0.35))];
      hitter = hitter === 'A' ? 'B' : 'A';
    }
    return { shots: shots, outcome: outcome, endedBy: wantLast };
  }
  function round(v) { return Math.round(v * 100) / 100; }

  // ---------- assemble the match ----------
  var players = {
    A: { id: 'A', name: 'Mira Lindqvist', short: 'Lindqvist', code: 'LIN', hand: 'right' },
    B: { id: 'B', name: 'Ada Okafor', short: 'Okafor', code: 'OKA', hand: 'right' }
  };

  var sequences = [g1, g2, g3];
  var rallies = [], games = [];
  var t = 130; // footage starts warm-up; first serve at 02:10
  var server = 'A';

  sequences.forEach(function (seq, gi) {
    var game = gi + 1;
    var score = { A: 0, B: 0 };
    var intervalDone = false;
    var start = t, firstIndex = rallies.length;
    for (var k = 0; k < seq.length; k++) {
      var w = seq[k];
      var bias = null;
      Object.keys(BIAS).forEach(function (key) {
        var b = BIAS[key];
        if (b.game !== game) return;
        // inside the run: score has reached `from` but not yet `to`
        var inA = score.A >= b.from[0] && score.A <= b.to[0] && score.B >= b.from[1] && score.B <= b.to[1];
        var notDone = !(score.A === b.to[0] && score.B === b.to[1]);
        if (inA && notDone) bias = key;
      });
      var before = { A: score.A, B: score.B };
      var built = buildRally({ server: server, winner: w, scoreBefore: before, bias: bias, game: game });
      var dur = 0, st = t + r(0.4, 0.9);
      built.shots.forEach(function (s) { s.t = round(st + dur); dur += s.dt; s.dt = round(s.dt); });
      score[w]++;
      var rally = {
        id: game + '-' + (k + 1),
        game: game, n: k + 1, index: rallies.length,
        server: server, winner: w,
        outcome: built.outcome, endedBy: built.endedBy,
        scoreBefore: before, scoreAfter: { A: score.A, B: score.B },
        start: round(st), end: round(st + dur),
        length: built.shots.length,
        shots: built.shots,
        tracking: built.shots.every(function (s) { return s.tracked; }) ? 'full' : 'partial',
        bias: bias
      };
      rallies.push(rally);
      server = w;
      t = st + dur + r(14, 26);
      if (!intervalDone && (score.A === 11 || score.B === 11)) { t += 60; intervalDone = true; rally.interval = true; }
    }
    var winner = score.A > score.B ? 'A' : 'B';
    games.push({ n: game, score: { A: score.A, B: score.B }, winner: winner, start: round(start), end: round(rallies[rallies.length - 1].end), first: firstIndex, last: rallies.length - 1 });
    server = winner;
    t += 120;
  });

  var match = {
    id: 'sample',
    illustrative: true,
    title: 'Sample match — club singles',
    date: '2026-10-18',
    venue: 'Illustrative sample · fictional players',
    players: players,
    winner: 'A',
    games: games,
    rallies: rallies,
    duration: round(rallies[rallies.length - 1].end - rallies[0].start),
    video: {
      src: null, // set to a real file URL when footage is supplied
      poster: null,
      length: round(rallies[rallies.length - 1].end + 40)
    },
    court: COURT
  };

  // ---------- derived analysis (all measured, computed from rallies) ----------
  function runs(minLen) {
    var out = [];
    games.forEach(function (g) {
      var cur = null;
      for (var i = g.first; i <= g.last; i++) {
        var rl = rallies[i];
        if (cur && cur.player === rl.winner) { cur.to = i; cur.len++; }
        else { if (cur && cur.len >= minLen) out.push(cur); cur = { game: g.n, player: rl.winner, from: i, to: i, len: 1 }; }
      }
      if (cur && cur.len >= minLen) out.push(cur);
    });
    return out;
  }

  function findRally(game, a, b) {
    for (var i = 0; i < rallies.length; i++) {
      var rl = rallies[i];
      if (rl.game === game && rl.scoreBefore.A === a && rl.scoreBefore.B === b) return rl;
    }
    return null;
  }

  function slice(from, to) { return rallies.slice(from.index, to.index + 1); }
  function avg(list, f) { return list.length ? list.reduce(function (s, x) { return s + f(x); }, 0) / list.length : 0; }

  var matchAvg = avg(rallies, function (x) { return x.length; });

  // Turning points: located by score state, facts computed, suggestions authored.
  var tp1Start = findRally(1, 17, 18), tp1End = rallies[games[0].last];
  var tp2Start = findRally(2, 11, 11), tp2End = findRally(2, 11, 16);
  var tp3Start = findRally(3, 8, 12), tp3End = findRally(3, 14, 12);

  var tp2Set = slice(tp2Start, tp2End);
  var tp2Corner = tp2Set.filter(function (x) {
    var last = x.shots[x.shots.length - 1];
    return x.outcome === 'winner' && last.to[0] < 2.2 && last.to[1] > COURT.NET + 4.5;
  }).length;
  var tp3Set = slice(tp3Start, tp3End);
  var tp3Avg = avg(tp3Set, function (x) { return x.length; });

  match.turningPoints = [
    {
      id: 'tp1', label: '1', game: 1,
      from: tp1Start.index, to: tp1End.index, rally: tp1Start.index,
      title: 'Lindqvist closes Game 1 from 17–18',
      fact: 'Four straight points, 17–18 to 21–18. ' + slice(tp1Start, tp1End).filter(function (x) { return x.outcome === 'error'; }).length + ' of the 4 ended on an Okafor error.',
      suggestion: 'Worth rewatching Okafor’s shot choices from 18–18. Consider whether she played safer or riskier than earlier in the game.'
    },
    {
      id: 'tp2', label: '2', game: 2,
      from: tp2Start.index, to: tp2End.index, rally: tp2Start.index + 2,
      title: 'Okafor’s six-point run from 11–11',
      fact: 'Six straight points, 11–11 to 11–17. ' + tp2Corner + ' of 6 were winners landing in Lindqvist’s rear backhand corner.',
      suggestion: 'Suggest drilling the round-the-head recovery to the rear backhand corner; it was targeted repeatedly in this run.'
    },
    {
      id: 'tp3', label: '3', game: 3,
      from: tp3Start.index, to: tp3End.index, rally: tp3End.index,
      title: 'Lindqvist’s seven straight from 8–12',
      fact: 'Seven straight points, 8–12 to 15–12. Average rally ' + tp3Avg.toFixed(1) + ' shots, against ' + matchAvg.toFixed(1) + ' for the match.',
      suggestion: 'Longer rallies seemed to favour Lindqvist here. Consider whether extending rallies is a deliberate plan worth repeating.'
    }
  ];

  match.runs = runs(4);

  match.util = {
    avg: avg,
    fmtTime: function (s) {
      s = Math.max(0, Math.round(s));
      var h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), sec = s % 60;
      var mm = (h ? String(m).padStart(2, '0') : String(m).padStart(2, '0'));
      return (h ? h + ':' : '') + mm + ':' + String(sec).padStart(2, '0');
    },
    fmtDuration: function (s) {
      var m = Math.round(s / 60), h = Math.floor(m / 60);
      return h ? h + '\u00a0h\u00a0' + String(m % 60).padStart(2, '0') + '\u00a0min' : m + '\u00a0min';
    }
  };

  /* Demo library rows — ILLUSTRATIVE. A backend returns GET /api/matches. */
  var LIBRARY = [
    { id: 'sample', date: '2026-10-18', a: 'Mira Lindqvist', b: 'Ada Okafor', result: '21–18 17–21 21–16', winner: 'a', duration: match.duration, status: 'ready' },
    { id: 'm-1029', date: '2026-10-29', a: 'Mira Lindqvist', b: 'Sana Iqbal', result: null, winner: null, duration: 2710, status: 'processing', progress: 0.62 },
    { id: 'm-1024', date: '2026-10-24', a: 'Jonas Brekke', b: 'Mira Lindqvist', result: '15–21 21–19 18–21', winner: 'b', duration: 3895, status: 'ready' },
    { id: 'm-1012', date: '2026-10-12', a: 'Ada Okafor', b: 'Lena Hart', result: '21–12 21–15', winner: 'a', duration: 2140, status: 'ready' },
    { id: 'm-1003', date: '2026-10-03', a: 'Mira Lindqvist', b: 'Theo Vance', result: null, winner: null, duration: 1580, status: 'failed', reason: 'Court corners not visible — camera too low' },
    { id: 'm-0927', date: '2026-09-27', a: 'Priya Nair', b: 'Ada Okafor', result: '19–21 21–17 22–20', winner: 'a', duration: 4210, status: 'ready' },
    { id: 'm-0919', date: '2026-09-19', a: 'Mira Lindqvist', b: 'Jonas Brekke', result: '21–14 21–16', winner: 'a', duration: 2380, status: 'ready' }
  ];

  window.GM_MATCH = match;
  window.GM_LIMITS = LIMITS;
  window.GM_LIBRARY = LIBRARY;
})();
