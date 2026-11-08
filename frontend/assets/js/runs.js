/* Goodminton — real analysed runs from the backend: the library list and the run page.
 * Everything from the backend is untrusted text (the database is publicly writable until its
 * reducers are locked down), so it only ever reaches the DOM as text nodes.
 */
(function () {
  'use strict';
  var GM = window.GM, h = GM.h, api = GM.api;
  var TITLES = {
    offline: 'You’re offline', timeout: 'That took too long', network: 'Can’t reach the analysis service',
    rate_limited: 'The service is busy', server: 'The analysis service had a problem', auth: 'Access refused',
    db_not_found: 'Analysis database not found', run_not_found: 'Run not found', no_rallies: 'No scored rallies yet', invalid_id: 'Invalid link',
    contract: 'This site and the backend are out of step', malformed: 'Unreadable response', unknown: 'Something went wrong'
  };

  function replace(host, node) { while (host.firstChild) host.removeChild(host.firstChild); if (node) host.appendChild(node); }
  function errorNotice(err, retry, opts) {
    var box = h('div', { class: 'notice', role: 'alert' }, [h('h3', null, [opts && opts.title || TITLES[err.code] || TITLES.unknown]), h('p', null, [err.message])]);
    var actions = h('div', { class: 'load-actions' });
    if (err.retryable) {
      var btn = h('button', { class: 'btn btn-small', type: 'button' }, ['Try again']);
      btn.addEventListener('click', function () { btn.disabled = true; btn.textContent = 'Trying…'; retry(); });
      actions.appendChild(btn);
    }
    if (opts && opts.link) actions.appendChild(h('a', { class: 'btn btn-small', href: opts.link.href }, [opts.link.label]));
    if (actions.firstChild) box.appendChild(actions);
    box.appendChild(h('p', { class: 'small muted' }, ['Error: ' + err.code + (err.status ? ' (HTTP ' + err.status + ')' : '')]));
    return box;
  }

  /* One data exchange bound to one region of the page: latest request wins, the skeleton waits
     120 ms so fast responses never flash, and a retryable failure recovers when the network returns. */
  function Loader(host, opts) {
    var ctl = null, timer = null, last = null;
    function fail(me, err) {
      if (me !== ctl || err.code === 'aborted') return;
      if (!(err instanceof api.ApiError)) { console.error(err); err = new api.ApiError('unknown'); }
      else console.warn('[api]', err.code, err.status || '', err.detail || '');
      clearTimeout(timer); last = err;
      host.removeAttribute('aria-busy');
      replace(host, errorNotice(err, run, opts.error && opts.error(err)));
    }
    function run() {
      if (ctl) ctl.abort();
      var me = ctl = new AbortController();
      last = null;
      clearTimeout(timer);
      host.setAttribute('aria-busy', 'true');
      timer = setTimeout(function () { if (me === ctl) replace(host, opts.skeleton()); }, 120);
      Promise.resolve().then(function () { return opts.fetch(me.signal); }).then(function (data) {
        if (me !== ctl) return;
        clearTimeout(timer);
        host.removeAttribute('aria-busy');
        replace(host, null);
        opts.render(data, host);
      }).catch(function (err) { fail(me, err); });
    }
    window.addEventListener('online', function () { if (last && last.retryable) run(); });
    run();
    return { reload: run };
  }

  function skeleton(lines, label) {
    var box = h('div', { class: 'sk-group', role: 'status' }, [h('span', { class: 'sr-only' }, [label || 'Loading'])]);
    for (var i = 0; i < lines; i++) box.appendChild(h('div', { class: 'sk', 'aria-hidden': 'true' }));
    return box;
  }
  function dateText(d) { return d ? d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }) : '—'; }
  function lengthText(s) {
    if (s == null) return '—';
    s = Math.round(s);
    return s < 60 ? s + ' s' : Math.floor(s / 60) + ' min ' + (s % 60) + ' s';
  }
  function clock(t) {
    if (t == null) return '—';
    var m = Math.floor(t / 60), s = t - m * 60;
    return m + ':' + (s < 10 ? '0' : '') + s.toFixed(1);
  }
  function runHref(id) { return 'run.html?id=' + encodeURIComponent(id); }

  // ---------- library list
  GM.mountRunsList = function (host) {
    return Loader(host, {
      fetch: function (signal) { return api.listRuns({ signal: signal }); },
      skeleton: function () { return skeleton(3, 'Loading analysed runs'); },
      render: function (runs) {
        if (!runs.length) {
          replace(host, h('div', { class: 'empty' }, [h('b', null, ['No analysed runs yet.']), 'Runs appear here once the pipeline publishes them with --stdb.']));
          return;
        }
        var tbody = h('tbody');
        runs.forEach(function (r) {
          var href = runHref(r.id);
          var tr = h('tr', null, [
            h('td', { class: 'num c-date' }, [dateText(r.createdAt)]),
            h('td', { class: 'players c-players' }, [h('a', { class: 'rowlink', href: href }, [r.id])]),
            h('td', null, [r.videoName || '—']),
            h('td', { class: 'num c-dur' }, [lengthText(r.durationS)]),
            h('td', null, [h('span', { class: 'status ' + (r.complete ? 'ready' : 'processing') }, [h('i'), r.complete ? 'Ready' : 'Incomplete'])])
          ]);
          tr.addEventListener('click', function (e) { if (!e.target.closest('a')) location.href = href; });
          tbody.appendChild(tr);
        });
        var head = h('tr');
        ['Date', 'Run', 'Source video', 'Length', 'Analysis'].forEach(function (t) { head.appendChild(h('th', { scope: 'col' }, [t])); });
        replace(host, h('table', { class: 'lib' }, [h('caption', { class: 'sr-only' }, ['Runs published by the analysis pipeline']), h('thead', null, [head]), tbody]));
      }
    });
  };

  // ---------- run page
  var NAMES = { near: 'Near player', far: 'Far player' };
  function playersOf(run) {
    var seen = {};
    run.shots.forEach(function (s) {
      if (!s.playerId) return;
      var p = seen[s.playerId] = seen[s.playerId] || { id: s.playerId, top: 0, bottom: 0, shots: [] };
      p.shots.push(s);
      if (s.side === 'Top') p.top++; else if (s.side === 'Bottom') p.bottom++;
    });
    [1, 2].forEach(function (id) { if (!seen[id]) seen[id] = { id: id, top: 0, bottom: 0, shots: [] }; });
    return Object.keys(seen).map(function (k) {
      var p = seen[k];
      p.end = p.bottom > p.top ? 'near' : (p.top > p.bottom ? 'far' : null);
      p.name = p.end ? NAMES[p.end] : 'Player ' + p.id;
      return p;
    }).sort(function (a, b) { return (a.end === 'near' ? 0 : 1) - (b.end === 'near' ? 0 : 1) || a.id - b.id; });
  }
  function fact(label, value) { return h('span', null, [label ? label + ' ' : '', h('b', { class: 'num' }, [value])]); }
  function metric(label, value, note) {
    var dd = h('dd', { class: 'num' }, [value]);
    if (note) dd.appendChild(h('small', null, [note]));
    return h('div', null, [h('dt', null, [label]), dd]);
  }
  function p95(values) {
    if (!values.length) return null;
    var v = values.slice().sort(function (a, b) { return a - b; });
    return v[Math.floor(v.length * 0.95)];
  }

  function playerCard(run, p, frameSlot) {
    var mix = {}, total = p.shots.length;
    p.shots.forEach(function (s) { var k = s.type || 'Unclassified'; mix[k] = (mix[k] || 0) + 1; });
    var rows = Object.keys(mix).sort(function (a, b) { return mix[b] - mix[a]; });
    var dist = run.summary ? run.summary.distanceM[p.id] : null;
    var head = h('header', { class: 'pc-head' }, [
      h('span', { class: 'pc-plate', 'aria-hidden': 'true' }, ['P' + p.id]),
      h('div', null, [h('h3', null, [p.name]), h('p', { class: 'small' }, ['Tracked as player ' + p.id + (p.end ? ', ' + p.end + ' end on camera' : '')])])
    ]);
    var mixBox = h('div', { class: 'pc-mix' }, [h('div', { class: 'pc-mix-k' }, ['Shot mix'])]);
    if (!rows.length) mixBox.appendChild(h('p', { class: 'small muted' }, ['No hits attributed to this player.']));
    rows.forEach(function (k) {
      var pct = mix[k] / total * 100;
      mixBox.appendChild(h('div', { class: 'mx' }, [h('span', null, [k]), h('span', { class: 'mx-bar' }, [h('i', { style: 'width:' + pct.toFixed(1) + '%' })]), h('span', { class: 'num' }, [Math.round(pct) + '%'])]));
    });
    return h('article', { class: 'pc' }, [
      head,
      h('div', { class: 'pc-group' }, [
        h('h4', null, ['Detected ', h('span', null, ['model output, check against the footage'])]),
        h('dl', { class: 'pc-grid' }, [metric('Hits', String(total))]),
        mixBox
      ]),
      h('div', { class: 'pc-group pc-est' }, [
        h('h4', null, ['Estimated ', h('span', null, ['from 2D tracking, approximate'])]),
        h('dl', { class: 'pc-grid' }, [metric('Distance covered', dist == null ? '—' : dist.toFixed(1), dist == null ? 'Not published yet' : 'metres')]),
        frameSlot
      ])
    ]);
  }

  function courtHeat(run) {
    var C = GM_MATCH.court, panel = h('div', { class: 'panel' }, [h('div', { class: 'chart-head' }, [h('h3', null, ['Court map']), h('span', { class: 'small muted' }, ['Where players stood'])])]);
    var pts = run.summary ? run.summary.heatmaps[1].concat(run.summary.heatmaps[2]) : [];
    if (!pts.length) { panel.appendChild(h('div', { class: 'empty' }, [h('b', null, ['No position data yet.']), 'The heatmap appears once the run’s summary is published.'])); return panel; }
    var svg = GM.el('svg', { viewBox: '-1.1 -1.2 8.3 15.9', class: 'court-map', role: 'img', 'aria-label': 'Heatmap of player positions on the court' });
    GM.drawCourtLines(GM.el('g', null, svg));
    GM.attachHeat(svg).paint(pts.map(function (p) { return [p[0] * C.W, p[1] * C.L, 1]; }));
    panel.appendChild(h('div', { class: 'court-box' }, [svg]));
    panel.appendChild(h('p', { class: 'how-read' }, ['Sampled from tracked positions. Brighter means more time spent there.']));
    return panel;
  }

  function shotTable(run, players) {
    var names = {};
    players.forEach(function (p) { names[p.id] = p.name; });
    var panel = h('section', { class: 'panel run-shots', 'aria-labelledby': 'run-shots-h' }, [h('div', { class: 'chart-head' }, [h('h3', { id: 'run-shots-h' }, ['Detected hits']), h('span', { class: 'small muted' }, [run.shots.length + ' in this run'])])]);
    if (!run.shots.length) { panel.appendChild(h('div', { class: 'empty' }, [h('b', null, ['No hits were detected.']), 'The clip may be too short, or the shuttle wasn’t tracked.'])); return panel; }
    var tbody = h('tbody');
    run.shots.forEach(function (s, i) {
      tbody.appendChild(h('tr', null, [
        h('td', { class: 'num' }, [String(i + 1)]), h('td', { class: 'num' }, [clock(s.t)]),
        h('td', null, [s.playerId ? names[s.playerId] || 'Player ' + s.playerId : 'Unknown']),
        h('td', null, [s.type || 'Unclassified']),
        h('td', { class: 'num' }, [s.confidence == null ? '—' : Math.round(s.confidence * 100) + '%'])
      ]));
    });
    var head = h('tr');
    ['#', 'Time', 'Player', 'Shot', 'Classifier score'].forEach(function (t) { head.appendChild(h('th', { scope: 'col' }, [t])); });
    panel.appendChild(h('div', { class: 'run-table' }, [h('table', { class: 'lib run-lib' }, [h('caption', { class: 'sr-only' }, ['Hits detected in this run']), h('thead', null, [head]), tbody])]));
    panel.appendChild(h('p', { class: 'how-read' }, ['The classifier score is the model’s own output, not a calibrated probability.']));
    return panel;
  }

  var OUTCOMES = { winner: 'Landed in', net: 'Net', out: 'Out', unknown: 'No clear result' };
  function rallyTable(run, players) {
    var names = {};
    players.forEach(function (p) { names[p.id] = p.name; });
    var n = run.rallies.length;
    var panel = h('section', { class: 'panel run-shots', 'aria-labelledby': 'run-rallies-h' }, [h('div', { class: 'chart-head' }, [h('h3', { id: 'run-rallies-h' }, ['Rallies']), h('span', { class: 'small muted' }, [n + (n === 1 ? ' in this run' : ' in this run')])])]);
    if (!n) { panel.appendChild(h('div', { class: 'empty' }, [h('b', null, ['No rallies were found.']), 'A rally needs at least two detected hits.'])); return panel; }
    var tbody = h('tbody');
    run.rallies.forEach(function (r) {
      tbody.appendChild(h('tr', null, [
        h('td', { class: 'num' }, [String(r.index)]), h('td', { class: 'num' }, [clock(r.startS) + '–' + clock(r.endS)]), h('td', { class: 'num' }, [String(r.shotCount)]),
        h('td', null, [OUTCOMES[r.outcome] || OUTCOMES.unknown]), h('td', null, [r.winnerId ? names[r.winnerId] || 'Player ' + r.winnerId : '—']),
        h('td', { class: 'num' }, [r.scoreP1 + '–' + r.scoreP2])
      ]));
    });
    var head = h('tr');
    ['#', 'Time', 'Shots', 'Ending', 'Point to', 'Score (P1–P2)'].forEach(function (t) { head.appendChild(h('th', { scope: 'col' }, [t])); });
    panel.appendChild(h('div', { class: 'run-table' }, [h('table', { class: 'lib run-lib' }, [h('caption', { class: 'sr-only' }, ['Rallies in this run']), h('thead', null, [head]), tbody])]));
    panel.appendChild(h('p', { class: 'how-read' }, ['A rally with no clear result usually means the clip ended, or the shuttle wasn’t seen landing. It adds no point to the score.']));
    return panel;
  }

  // Speeds need the per-frame table, which can be large: it loads on its own and can fail on its own.
  function frameMetrics(run, players, slots, status) {
    Loader(status, {
      fetch: function (signal) {
        return api.getFrames(run, { signal: signal, onProgress: function (done, all) {
          var n = status.querySelector('[data-frames-progress]');
          if (n) n.textContent = 'Loading tracking data · ' + done.toLocaleString() + ' of ' + all.toLocaleString() + ' frames';
        } });
      },
      skeleton: function () { return h('p', { class: 'small muted', role: 'status', 'data-frames-progress': '' }, ['Loading tracking data…']); },
      error: function () { return { title: 'Tracking data didn’t load' }; },
      render: function (frames) {
        players.forEach(function (p) {
          var key = p.id === 1 ? 'p1Speed' : 'p2Speed';
          var speeds = frames.map(function (f) { return f[key]; }).filter(function (v) { return v && v <= 12; }); // above 12 m/s is a tracking jump, not footwork
          var fast = p95(speeds);
          replace(slots[p.id], h('dl', { class: 'pc-grid' }, [metric('Fast movement', fast == null ? '—' : (fast * 3.6).toFixed(1), fast == null ? 'Not enough tracking' : 'km/h, 95th percentile')]));
        });
        replace(status, null);
      }
    });
  }

  GM.Loader = Loader;
  GM.loadSkeleton = skeleton;

  GM.mountRun = function (root) {
    var id = new URLSearchParams(location.search).get('id') || '';
    var link = { href: 'library.html', label: 'Back to library' };
    return Loader(root, {
      fetch: function (signal) {
        return Promise.all([api.getRun(id, { signal: signal }), GM.pipeline.jobIfAny(id)]).then(function (res) { res[0].job = res[1]; return res[0]; });
      },
      skeleton: function () { return h('div', { class: 'wrap run-loading' }, [skeleton(5, 'Loading run')]); },
      error: function () { return { link: link }; },
      render: function (run) {
        document.title = run.id + ' — Goodminton';
        var players = playersOf(run), wrap = h('div', { class: 'wrap' });
        var facts = h('div', { class: 'facts' }, [fact('', dateText(run.createdAt)), fact('Length', lengthText(run.durationS)), fact('', run.totalFrames.toLocaleString() + ' frames'), fact('', run.shots.length + ' hits')]);
        if (run.summary && run.summary.avgShuttleSpeedMs != null) facts.appendChild(fact('Avg shuttle at hit (est.)', Math.round(run.summary.avgShuttleSpeedMs * 3.6) + ' km/h'));
        wrap.appendChild(h('header', { class: 'report-head' }, [h('div', null, [h('h1', null, [run.id]), facts])]));
        var decided = (run.rallies || []).filter(function (r) { return r.winnerId; }).length;
        if (run.rallies) facts.appendChild(fact('', run.rallies.length + (run.rallies.length === 1 ? ' rally' : ' rallies') + (run.rallies.length ? ', ' + decided + ' with a clear result' : '')));
        var videoUrl = run.job && run.job.video_url;
        var note = 'Source: ' + (run.job && run.job.filename || run.videoName || 'unknown') + '. ' + (videoUrl ? '' : 'The original video isn’t available for this run. ') +
          (run.rallies ? 'Results are estimated from where the shuttle was seen landing; a rally with no clear landing has no result.' : 'Rally results haven’t been published for this run.');
        wrap.appendChild(h('p', { class: 'how-read run-note' }, [note]));
        if (decided) wrap.appendChild(h('p', { class: 'run-open' }, [h('a', { class: 'btn btn-primary', href: 'report.html?run=' + encodeURIComponent(run.id) }, ['Open the full report'])]));
        if (!run.complete) wrap.appendChild(h('div', { class: 'notice', role: 'status' }, [h('h3', null, ['Analysis incomplete']), h('p', null, ['This run’s summary hasn’t been published, so some figures are missing.'])]));
        if (videoUrl) wrap.appendChild(h('video', { class: 'run-video', src: videoUrl, controls: true, preload: 'metadata', playsinline: true }));
        var slots = {}, status = h('div', { class: 'run-frames-status', 'aria-live': 'polite' }), cards = h('div', { class: 'players' });
        players.forEach(function (p) { slots[p.id] = h('div', { class: 'run-slot' }); cards.appendChild(playerCard(run, p, slots[p.id])); });
        wrap.appendChild(h('div', { class: 'run-grid' }, [courtHeat(run), h('div', { class: 'run-main' }, [cards, status])]));
        if (run.rallies) wrap.appendChild(rallyTable(run, players));
        wrap.appendChild(shotTable(run, players));
        root.appendChild(wrap);
        if (run.totalFrames > 0) frameMetrics(run, players, slots, status);
      }
    });
  };
})();
