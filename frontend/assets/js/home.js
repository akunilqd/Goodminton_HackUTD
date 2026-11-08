/* Goodminton — homepage: the hero still, then the live sample board */
(function () {
  'use strict';
  var GM = window.GM, M = window.GM_MATCH, el = GM.el, P = GM.P;

  // The hero frame: the last point of Okafor's six-point run, marked on the footage.
  var rally = M.rallies.filter(function (r) { return r.game === 2 && r.scoreBefore.A === 11 && r.scoreBefore.B === 16; })[0];

  function hero() {
    var host = document.querySelector('[data-still]');
    if (!host || !rally) return;
    var svg = el('svg', { class: 'bc', role: 'img', preserveAspectRatio: 'xMidYMid slice',
      'aria-label': 'Footage placeholder: broadcast view of game 2 at 11–16. The path of the rally is traced on the court, ending in a winner in the near rear corner.' });
    host.insertBefore(svg, host.firstChild);
    var layers = GM.drawBroadcast(svg);
    var last = rally.shots[rally.shots.length - 1];
    var t = last.t + last.dt * 0.62;

    // earlier shots: faint ground lines, hit → landing
    var trace = el('g', null, layers.far);
    var pts = [rally.shots[0].from].concat(rally.shots.map(function (s) { return s.to; }));
    var d = pts.slice(0, -1).map(function (p, i) { var q = GM.project(p[0], p[1]); return (i ? 'L' : 'M') + q[0].toFixed(1) + ',' + q[1].toFixed(1); }).join('');
    el('path', { d: d, class: 'bc-trace-ground' }, trace);
    pts.slice(0, -1).forEach(function (p) {
      var q = GM.project(p[0], p[1]);
      el('circle', { cx: q[0], cy: q[1], r: 5, class: 'bc-trace-dot' }, trace);
    });
    // final shot: a flight tracer through the air, sampled along its arc
    var arc = '';
    for (var k = 0; k <= 40; k++) {
      var pp = k / 40;
      var hx = last.from[0] + (last.to[0] - last.from[0]) * pp, hy = last.from[1] + (last.to[1] - last.from[1]) * pp;
      var hh = (last.type === 'smash' ? 2.7 * (1 - pp) : 2.4 * (1 - pp) + 4.2 * Math.sin(Math.PI * pp));
      var q2 = GM.project(hx, hy, Math.max(0, hh));
      arc += (k ? 'L' : 'M') + q2[0].toFixed(1) + ',' + q2[1].toFixed(1);
    }
    var tr = el('g', null, layers.air);
    el('path', { d: arc, class: 'bc-trace-shadow' }, tr);
    var main = el('path', { d: arc, class: 'bc-trace' }, tr);
    var end = GM.project(last.to[0], last.to[1]);
    el('ellipse', { cx: end[0], cy: end[1], rx: 30, ry: 11, class: 'bc-trace-end' }, trace);
    el('ellipse', { cx: end[0], cy: end[1], rx: 52, ry: 19, class: 'bc-trace-end', opacity: '.5' }, trace);

    // players at that moment (positions from the sample data)
    var a = last.to, b = last.from;
    GM.figure(layers.far, b[0], b[1] - 0.1, 'B', P('B').code);
    // Lindqvist caught at mid-court, wrong-footed by the shot to her rear backhand
    GM.figure(layers.near, a[0] < 3.05 ? 4.1 : 2.0, 9.6, 'A', P('A').code);
    // the shuttle, just before landing
    var p = 0.9, sx = last.from[0] + (last.to[0] - last.from[0]) * p, sy = last.from[1] + (last.to[1] - last.from[1]) * p;
    var sh = last.type === 'smash' ? 2.7 * (1 - p) : 2.4 * (1 - p) + 4.2 * Math.sin(Math.PI * p);
    var g0 = GM.project(sx, sy), s0 = GM.project(sx, sy, sh);
    el('ellipse', { cx: g0[0], cy: g0[1], rx: 7, ry: 2.6, class: 'bc-shuttle-shadow' }, layers.air);
    el('circle', { cx: s0[0], cy: s0[1], r: 8, class: 'bc-shuttle' }, layers.air);

    // trace the path once on load — it explains what the frame is showing
    if (!GM.reduceMotion && main.getTotalLength) {
      var len = main.getTotalLength();
      main.style.strokeDasharray = len; main.style.strokeDashoffset = len;
      main.animate([{ strokeDashoffset: len }, { strokeDashoffset: 0 }], { duration: 1400, delay: 350, easing: 'cubic-bezier(.3,.1,.2,1)', fill: 'forwards' });
    }

    // overlays
    var bug = host.querySelector('.bug');
    var prev = M.games.filter(function (g) { return g.n < 2; });
    var gwA = prev.filter(function (g) { return g.winner === 'A'; }).length;
    bug.innerHTML = '<span class="bug-g">G2</span>' +
      '<span class="bug-row"><b>' + P('A').code + '</b><i>' + gwA + '</i><em>' + rally.scoreBefore.A + '</em></span>' +
      '<span class="bug-row srv"><b>' + P('B').code + '</b><i>' + (prev.length - gwA) + '</i><em>' + rally.scoreBefore.B + '</em></span>';
    host.querySelector('.stamp').textContent = GM.fmt(t) + '  ·  Rally ' + rally.id.replace('-', '·');

    var card = host.querySelector('.callout-card');
    card.innerHTML = '<b>' + P('B').short + '’s sixth straight point</b>' +
      GM.score(rally.scoreBefore) + ' → ' + GM.score(rally.scoreAfter) + ' · ' + rally.length + ' shots · ' + GM.endingText(rally).split(' · ')[0].toLowerCase() + ' to the rear backhand corner';

    function fit() {
      var wide = host.clientWidth > 700;
      svg.setAttribute('viewBox', wide ? '0 150 1600 700' : '160 150 1280 960');
      if (!wide) { card.style.left = card.style.top = ''; return; }
      // anchor the callout next to the landing mark
      var pt = svg.createSVGPoint(); pt.x = end[0]; pt.y = end[1];
      var m = svg.getScreenCTM(); if (!m) return;
      var sp = pt.matrixTransform(m), box = host.getBoundingClientRect();
      var x = sp.x - box.left, y = sp.y - box.top;
      var cw = card.offsetWidth, ch = card.offsetHeight;
      var left = x + 46; if (left + cw > box.width - 12) left = x - cw - 46;
      card.style.left = Math.max(12, left) + 'px';
      card.style.top = Math.max(12, Math.min(box.height - ch - 12, y - ch - 20)) + 'px';
    }
    fit();
    new ResizeObserver(fit).observe(host);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(fit);
  }

  function steps() {
    // small court diagrams that illustrate each step, drawn from the same sample
    document.querySelectorAll('[data-step]').forEach(function (fig) {
      var kind = fig.getAttribute('data-step');
      var svg = el('svg', { viewBox: kind === 'track' ? '-0.3 -0.3 14 6.7' : '0 0 330 90', class: 'step-svg', 'aria-hidden': 'true' });
      fig.appendChild(svg);
      if (kind === 'upload') {
        el('rect', { x: 1, y: 8, width: 120, height: 70, fill: 'none', stroke: '#141a24', 'stroke-width': 1.5 }, svg);
        el('path', { d: 'M48 32l26 11-26 11z', fill: '#141a24' }, svg);
        el('path', { d: 'M138 43h56', stroke: '#1d6b49', 'stroke-width': 2, 'stroke-dasharray': '4 4' }, svg);
        el('path', { d: 'M190 37l7 6-7 6', fill: 'none', stroke: '#1d6b49', 'stroke-width': 2 }, svg);
        var t1 = el('text', { x: 210, y: 40, class: 'step-cap' }, svg); t1.textContent = 'MP4 · MOV';
        var t2 = el('text', { x: 210, y: 56, class: 'step-cap muted' }, svg); t2.textContent = 'one fixed camera';
      } else if (kind === 'track') {
        // the whole court, turned on its side, with one tracked rally drawn on it
        var r = M.rallies[M.turningPoints[2].rally];
        var g = el('g', { transform: 'translate(0 6.1) rotate(-90)' }, svg);
        GM.drawCourtLines(g, { lw: 0.05 });
        var d = [r.shots[0].from].concat(r.shots.map(function (s) { return s.to; }))
          .map(function (p, i) { return (i ? 'L' : 'M') + p[0] + ',' + p[1]; }).join('');
        el('path', { d: d, fill: 'none', stroke: '#1d6b49', 'stroke-width': 0.06, 'stroke-linejoin': 'round' }, g);
        r.shots.forEach(function (s) { el('circle', { cx: s.to[0], cy: s.to[1], r: 0.12, fill: s.by === 'A' ? '#141a24' : '#f1ede5', stroke: '#141a24', 'stroke-width': 0.04 }, g); });
      } else if (kind === 'review') {
        var list = M.rallies.filter(function (x) { return x.game === 2; }).slice(14, 40);
        var bw = 300 / list.length;
        el('line', { x1: 0, y1: 45, x2: 300, y2: 45, stroke: '#141a24', 'stroke-width': 1 }, svg);
        list.forEach(function (x, i) {
          var hh = Math.min(40, x.length * 1.6);
          var up = x.winner === 'A';
          var hl = x.index === M.turningPoints[1].rally;
          el('rect', { x: i * bw + 1, y: up ? 45 - hh : 45, width: bw - 2, height: hh, fill: hl ? '#1d6b49' : (up ? '#141a24' : 'url(#gm-hatch)'), stroke: hl ? '#1d6b49' : '#141a24', 'stroke-width': up ? 0 : 0.8 }, svg);
        });
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    hero();
    steps();
    GM.mountSample(document.querySelector('[data-sample]'));
    GM.mountUpload(document.querySelector('[data-upload]'), { id: 'home' });
  });
})();
