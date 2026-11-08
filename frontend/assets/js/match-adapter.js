/* Goodminton — turns a published run into the match shape the full report renders (see data.js).
 * Only decided rallies become report rallies, because the report is built around who won each
 * point. Undecided rallies are counted in `undecided` so the page can say what was left out.
 */
(function () {
  'use strict';
  var GM = window.GM, M = window.GM_MATCH, api = GM.api;
  var C = M.court, KNOWN = { serve: 1, clear: 1, drop: 1, smash: 1, net: 1, drive: 1, lift: 1, push: 1, block: 1 };

  function nearId(run) {
    var votes = {};
    run.shots.forEach(function (s) { if (s.playerId) votes[s.playerId] = (votes[s.playerId] || 0) + (s.side === 'Bottom' ? 1 : s.side === 'Top' ? -1 : 0); });
    return (votes[1] || 0) > (votes[2] || 0) ? 1 : 2;
  }
  function round(v) { return Math.round(v * 100) / 100; }

  function toMatch(run, frames) {
    var decided = (run.rallies || []).filter(function (r) { return r.winnerId; });
    if (!decided.length) throw new api.ApiError('no_rallies');
    var near = nearId(run), side = function (id) { return id === near ? 'A' : 'B'; };
    var byFrame = {};
    frames.forEach(function (f) { byFrame[f.frame] = f; });
    function pos(frame, id) { var f = byFrame[frame]; return f ? (id === 1 ? f.p1 : f.p2) : null; }
    var homeOf = function (who) { return [C.W / 2, who === 'A' ? C.NET + 3 : C.NET - 3]; };

    var rallies = [], games = [], game = null;
    decided.forEach(function (r) {
      var list = run.shots.filter(function (s) { return s.id >= r.firstEventId && s.id <= r.lastEventId; });
      if (!game || game.n !== r.game) {
        game = { n: r.game, first: rallies.length, last: rallies.length, score: { A: 0, B: 0 }, start: list[0].t };
        games.push(game);
      }
      var shots = list.map(function (s, i) {
        var next = list[i + 1], who = s.playerId ? side(s.playerId) : (i ? (list[i - 1].playerId ? (side(list[i - 1].playerId) === 'A' ? 'B' : 'A') : 'A') : 'A');
        var opp = who === 'A' ? 'B' : 'A';
        var from = s.playerId ? pos(s.frame, s.playerId) : null;
        var to = next ? (next.playerId ? pos(next.frame, next.playerId) : null) : r.landing;
        var type = (s.type || '').toLowerCase();
        return {
          i: i, by: who, type: KNOWN[type] ? type : 'unknown',
          from: from || homeOf(who), to: to || homeOf(opp), t: s.t,
          dt: next ? Math.max(0.2, round(next.t - s.t)) : 0.8,
          result: next ? 'in' : (r.outcome === 'winner' ? 'winner' : r.outcome),
          tracked: !!(from && to)
        };
      });
      var w = side(r.winnerId), after = { A: near === 1 ? r.scoreP1 : r.scoreP2, B: near === 1 ? r.scoreP2 : r.scoreP1 };
      var before = { A: after.A - (w === 'A' ? 1 : 0), B: after.B - (w === 'B' ? 1 : 0) };
      var n = rallies.length - game.first + 1, last = shots[shots.length - 1];
      rallies.push({
        id: r.game + '-' + n, game: r.game, n: n, index: rallies.length,
        server: shots[0].by, winner: w, outcome: r.outcome === 'winner' ? 'winner' : 'error', endedBy: last.by,
        scoreBefore: before, scoreAfter: after, start: shots[0].t, end: round(last.t + last.dt), length: shots.length, shots: shots,
        tracking: shots.every(function (s) { return s.tracked; }) ? 'full' : 'partial'
      });
      game.last = rallies.length - 1; game.score = after; game.end = round(last.t + last.dt);
    });
    games.forEach(function (g) { g.winner = g.score.A === g.score.B ? null : (g.score.A > g.score.B ? 'A' : 'B'); });
    var wa = games.filter(function (g) { return g.winner === 'A'; }).length, wb = games.filter(function (g) { return g.winner === 'B'; }).length;

    function streaks(min) {
      var out = [];
      games.forEach(function (g) {
        var cur = null;
        for (var i = g.first; i <= g.last; i++) {
          if (cur && cur.player === rallies[i].winner) { cur.to = i; cur.len++; continue; }
          if (cur && cur.len >= min) out.push(cur);
          cur = { game: g.n, player: rallies[i].winner, from: i, to: i, len: 1 };
        }
        if (cur && cur.len >= min) out.push(cur);
      });
      return out;
    }
    var players = {
      A: { id: 'A', name: 'Near player', short: 'Near', code: 'P' + near, hand: null, trackId: near },
      B: { id: 'B', name: 'Far player', short: 'Far', code: 'P' + (3 - near), hand: null, trackId: 3 - near }
    };
    var turning = streaks(3).sort(function (a, b) { return b.len - a.len; }).slice(0, 3).sort(function (a, b) { return a.from - b.from; }).map(function (x, i) {
      var first = rallies[x.from], end = rallies[x.to], who = players[x.player].short;
      return {
        id: 'tp' + (i + 1), label: String(i + 1), game: x.game, from: x.from, to: x.to, rally: x.from,
        title: who + ' player, ' + x.len + ' straight points',
        fact: x.len + ' straight points, ' + GM.score(first.scoreBefore) + ' to ' + GM.score(end.scoreAfter) + '.',
        suggestion: 'Worth rewatching to see what changed during this run.'
      };
    });
    return {
      isLive: true, id: run.id, title: run.id, players: players, games: games, rallies: rallies,
      winner: wa === wb ? null : (wa > wb ? 'A' : 'B'), duration: round(rallies[rallies.length - 1].end - rallies[0].start),
      video: { src: null, poster: null, length: run.durationS || round(rallies[rallies.length - 1].end + 5) },
      turningPoints: turning, runs: streaks(4),
      undecided: run.rallies.length - decided.length, scoreReliable: run.rallies.every(function (r) { return r.scoreReliable; }),
      source: run
    };
  }

  // The engine holds a reference to GM_MATCH, so real data replaces its contents in place.
  function hydrate(match) {
    Object.keys(M).forEach(function (k) { if (k !== 'court' && k !== 'util') delete M[k]; });
    Object.assign(M, match);
    return M;
  }

  GM.loadLiveMatch = async function (id, signal, onProgress) {
    var run = await api.getRun(id, { signal: signal });
    if (!run.rallies || !run.rallies.some(function (r) { return r.winnerId; })) throw new api.ApiError('no_rallies');
    var frames = await api.getFrames(run, { signal: signal, onProgress: onProgress });
    return hydrate(toMatch(run, frames));
  };
  GM.toLiveMatch = toMatch;
})();
