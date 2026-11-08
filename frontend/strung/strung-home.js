/* Goodminton · STRUNG — homepage
 * A live string bed (spring mesh) in a night sky. The cursor plucks it, a click "serves" into it,
 * and once you scroll it straightens into the court lines, carrying a real sample rally.
 * The racket → court morph runs on a clock (fixed duration), not on scroll distance, so it plays
 * at the same speed however fast you scroll; the stage stays pinned until it has finished.
 */
(function () {
  'use strict';
  var GM = window.GM, M = window.GM_MATCH, C = M.court, P = GM.P;
  var RM = GM.reduceMotion;

  // the hero rally: the last point of Okafor's six-point run
  var rally = M.rallies.filter(function (r) { return r.game === 2 && r.scoreBefore.A === 11 && r.scoreBefore.B === 16; })[0];

  var MORPH_MS = 1900;        // racket → court, at any scroll speed
  var UNMORPH_MS = 1100;      // court → racket on the way back up (exit faster than enter)
  var HEAD_MM = 230;          // a racket head is ~230 mm across: scales the HUD readouts

  function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function smooth(a, b, x) { var t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function ease(t) { return t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function rgba(hex, a) {
    var h = hex.replace('#', ''); if (h.length === 3) h = h.replace(/./g, '$&$&');
    var n = parseInt(h, 16); return 'rgba(' + (n >> 16 & 255) + ',' + (n >> 8 & 255) + ',' + (n & 255) + ',' + a + ')';
  }

  // ------------------------------------------------------------------ night sky (drawn once per layout)
  function Sky(host) {
    var a = document.createElement('canvas'), b = document.createElement('canvas');
    host.appendChild(a); host.appendChild(b);
    function rnd(seed) { return function () { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; }; }
    function paint() {
      var dpr = Math.min(2, window.devicePixelRatio || 1), W = host.clientWidth, H = host.clientHeight;
      [a, b].forEach(function (c) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); });
      var ca = a.getContext('2d'), cb = b.getContext('2d'), r = rnd(1907);
      ca.setTransform(dpr, 0, 0, dpr, 0, 0); cb.setTransform(dpr, 0, 0, dpr, 0, 0);
      var n = Math.round(W * H / 4200);
      for (var i = 0; i < n; i++) {
        var x = r() * W, y = Math.pow(r(), 1.5) * H * 0.82, s = r() < 0.9 ? 0.7 : 1.3;
        var fade = 1 - y / (H * 0.9), c = r() < 0.25 ? cb : ca;
        c.globalAlpha = (0.35 + r() * 0.65) * fade;
        c.fillStyle = r() < 0.12 ? '#dbc4ff' : (r() < 0.08 ? '#22e0ff' : '#ffffff');
        c.fillRect(x, y, s, s);
      }
      // a handful of four-point sparkles
      for (var k = 0; k < Math.max(5, Math.round(W / 220)); k++) {
        var sx = r() * W, sy = r() * H * 0.55, L = 3 + r() * 4, c2 = k % 2 ? cb : ca;
        c2.globalAlpha = 0.8; c2.strokeStyle = k % 3 ? '#ffffff' : '#dbc4ff'; c2.lineWidth = 1;
        c2.beginPath(); c2.moveTo(sx - L, sy); c2.lineTo(sx + L, sy); c2.moveTo(sx, sy - L); c2.lineTo(sx, sy + L); c2.stroke();
      }
    }
    paint();
    return { resize: paint };
  }

  // ------------------------------------------------------------------ string bed
  function StringBed(canvas, hero, hooks) {
    var ctx = canvas.getContext('2d');
    // the frame and throat live on a canvas behind the strings
    var back = document.createElement('canvas');
    back.className = 'bed-back'; back.setAttribute('aria-hidden', 'true');
    canvas.parentNode.insertBefore(back, canvas);
    var bctx = back.getContext('2d');
    var headEl = hero.querySelector('.ink h1');
    var W = 0, H = 0, dpr = 1, geo = null;
    var mains = [], crosses = [];
    var pointer = { x: -9999, y: -9999, on: false };
    var shuttles = [];
    var e = 0;           // morph progress 0 racket → 1 court
    var running = false, lastT = 0, maxD = 0;
    var COL = {};

    function readColors() {
      COL = {
        fg: css('--fg') || '#f4f2ff', night: css('--night-2') || '#05030a', violet: css('--violet') || '#7828ff',
        magenta: css('--magenta') || '#ff2ee6', cyan: css('--cyan') || '#22e0ff', lavender: css('--lavender') || '#dbc4ff',
        muted: css('--muted') || '#948eb4', ice: css('--cyan-ice') || '#d4faff'
      };
    }

    function layout() {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      W = canvas.clientWidth; H = canvas.clientHeight;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      back.width = canvas.width; back.height = canvas.height;
      // the bed fills the viewport: the head bleeds past the top edge (and the sides on narrow screens)
      var ry = H * 0.6, rx = ry * 0.86;
      var cx = W / 2, cy = H * 0.4;
      var s = (2 * rx) / 23;
      var narrow = W < 860;
      // phones: a smaller court high in the frame, leaving the lower band for the caption
      var Hc = narrow ? Math.min(H * 0.52, (W - 32) * 13.4 / 6.1 * 0.9) : Math.min(H * 0.74, (W - 32) * 13.4 / 6.1 * 0.9), Wc = Hc * C.W / C.L;
      var cx0 = cx - Wc / 2, cy0 = narrow ? 120 : (H - Hc) / 2 + 26;
      geo = { cx: cx, cy: cy, rx: rx, ry: ry, s: s, court: { x0: cx0, y0: cy0, w: Wc, h: Hc } };
      geo.mmPerPx = HEAD_MM / (2 * rx);

      // where the headline sits: the strings part around it so it reads cleanly
      var cr = canvas.getBoundingClientRect(), hr = headEl.getBoundingClientRect();
      geo.head = { x: hr.left - cr.left + hr.width / 2, y: hr.top - cr.top + hr.height / 2, rx: hr.width / 2 + 60, ry: hr.height / 2 + 46 };

      var fw = Math.max(10, ry * 0.075) * 0.5, irx = rx - fw, iry = ry - fw;
      function onEllipseY(x) { var u = (x - cx) / irx; return iry * Math.sqrt(Math.max(0, 1 - u * u)); }
      function onEllipseX(y) { var u = (y - cy) / iry; return irx * Math.sqrt(Math.max(0, 1 - u * u)); }
      function courtX(m) { return cx0 + m * (Wc / C.W); }
      function courtY(m) { return cy0 + m * (Hc / C.L); }

      var LONG = [0, C.SINGLES_IN, C.W / 2, C.W - C.SINGLES_IN, C.W];
      var TRANS = [0, C.DOUBLES_LONG, C.NET - C.SHORT, C.NET, C.NET + C.SHORT, C.L - C.DOUBLES_LONG, C.L];

      var N = 22, i, j;
      mains = [];
      for (j = 0; j < N; j++) {
        var x = cx - rx + s * (j + 1), h = onEllipseY(x);
        mains.push({ kind: 'm', u: (j + 1) / (N + 1), a0: [x, cy - h], b0: [x, cy + h], d: new Float32Array(20), v: new Float32Array(20) });
      }
      var NC = Math.max(9, Math.floor((2 * ry) / s) - 1);
      crosses = [];
      for (i = 0; i < NC; i++) {
        var y = cy - ry + (2 * ry) * (i + 1) / (NC + 1), w = onEllipseX(y);
        crosses.push({ kind: 'c', u: (i + 1) / (NC + 1), a0: [cx - w, y], b0: [cx + w, y], d: new Float32Array(20), v: new Float32Array(20) });
      }
      geo.counts = { mains: N, crosses: NC };
      function assign(list, targets, span) {
        list.forEach(function (st) { st.keep = false; });
        targets.forEach(function (t) {
          var best = null, bd = 9;
          list.forEach(function (st) { var d = Math.abs(st.u * span - t); if (d < bd && !st.keep) { bd = d; best = st; } });
          if (best) { best.keep = true; best.target = t; }
        });
      }
      assign(mains, LONG, C.W);
      assign(crosses, TRANS, C.L);
      mains.forEach(function (st) {
        var m = st.keep ? st.target : st.u * C.W;
        st.a1 = [courtX(m), courtY(0)]; st.b1 = [courtX(m), courtY(C.L)];
      });
      crosses.forEach(function (st) {
        var m = st.keep ? st.target : st.u * C.L;
        st.a1 = [courtX(0), courtY(m)]; st.b1 = [courtX(C.W), courtY(m)];
        st.net = st.keep && Math.abs(m - C.NET) < 0.01;
      });
      geo.selMain = mains[11];
      geo.courtX = courtX; geo.courtY = courtY;
    }

    // ---- physics
    function stepPhysics() {
      var energy = 0, k = 0.32, damp = 0.9, peak = 0;
      var R = Math.max(60, geo.s * 2.2);
      function run(st, normalAxis) {
        var n = st.d.length, ax = st.a0[0], ay = st.a0[1], bx = st.b0[0], by = st.b0[1];
        for (var i = 1; i < n - 1; i++) {
          var t = i / (n - 1);
          var px = lerp(ax, bx, t) + (normalAxis === 0 ? st.d[i] : 0);
          var py = lerp(ay, by, t) + (normalAxis === 1 ? st.d[i] : 0);
          var lap = st.d[i - 1] + st.d[i + 1] - 2 * st.d[i];
          var f = lap * k - st.d[i] * 0.012;
          if (pointer.on) {
            var dx = px - pointer.x, dy = py - pointer.y, dist = Math.hypot(dx, dy);
            if (dist < R) f += (normalAxis === 0 ? Math.sign(dx || 1) : Math.sign(dy || 1)) * (1 - dist / R) * 2.2;
          }
          st.v[i] = (st.v[i] + f) * damp;
        }
        for (var q = 1; q < n - 1; q++) {
          st.d[q] += st.v[q]; energy += Math.abs(st.v[q]) + Math.abs(st.d[q]) * 0.02;
          var ad = Math.abs(st.d[q]); if (ad > peak) peak = ad;
        }
      }
      mains.forEach(function (st) { run(st, 0); });
      crosses.forEach(function (st) { run(st, 1); });
      maxD = peak;
      return energy;
    }

    function impact(x, y, strength) {
      var R = geo.s * 6;
      function hit(st, axis) {
        for (var i = 1; i < st.d.length - 1; i++) {
          var t = i / (st.d.length - 1);
          var px = lerp(st.a0[0], st.b0[0], t), py = lerp(st.a0[1], st.b0[1], t);
          var dx = px - x, dy = py - y, dist = Math.hypot(dx, dy);
          if (dist < R) st.v[i] += (axis === 0 ? Math.sign(dx || 1) : Math.sign(dy || 1)) * (1 - dist / R) * strength;
        }
      }
      mains.forEach(function (st) { hit(st, 0); });
      crosses.forEach(function (st) { hit(st, 1); });
    }

    function serve(x, y) {
      shuttles.push({ x: x, y: y, t: 0, hit: false, fromX: x - W * 0.42, fromY: -60, toX: x + W * 0.3, toY: -80 });
      kick();
    }

    // ---- drawing
    function stringPath(st, axis) {
      var n = st.d.length, bend = 1 - smooth(0, 0.18, e), E = ease(e);
      var ax = lerp(st.a0[0], st.a1[0], E), ay = lerp(st.a0[1], st.a1[1], E);
      var bx = lerp(st.b0[0], st.b1[0], E), by = lerp(st.b0[1], st.b1[1], E);
      ctx.beginPath();
      for (var i = 0; i < n; i++) {
        var t = i / (n - 1);
        var x = lerp(ax, bx, t) + (axis === 0 ? st.d[i] * bend : 0);
        var y = lerp(ay, by, t) + (axis === 1 ? st.d[i] * bend : 0);
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }
    }
    // neon string: a wide low-alpha glow under a crisp core
    function neon(st, axis, alpha, width, core, glow, glowAlpha) {
      stringPath(st, axis);
      ctx.globalAlpha = alpha * glowAlpha; ctx.strokeStyle = glow; ctx.lineWidth = width * 4.2; ctx.stroke();
      ctx.globalAlpha = alpha; ctx.strokeStyle = core; ctx.lineWidth = width; ctx.stroke();
    }

    function draw() {
      var g = geo, E = ease(e);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';

      // ---------- back canvas: throat, shaft, rim, court floor
      var fctx = ctx; ctx = bctx;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      var drop = E * H * 0.6;
      ctx.globalAlpha = 1 - smooth(0.1, 0.5, e);
      var by0 = g.cy + g.ry * 0.93 + drop;
      ctx.beginPath();
      ctx.moveTo(g.cx - g.rx * 0.32, by0 - g.ry * 0.08);
      ctx.quadraticCurveTo(g.cx, by0 + g.ry * 0.2, g.cx - 7, by0 + g.ry * 0.36);
      ctx.lineTo(g.cx - 7, H + 20); ctx.lineTo(g.cx + 7, H + 20); ctx.lineTo(g.cx + 7, by0 + g.ry * 0.36);
      ctx.quadraticCurveTo(g.cx, by0 + g.ry * 0.2, g.cx + g.rx * 0.32, by0 - g.ry * 0.08);
      ctx.closePath();
      ctx.fillStyle = COL.night; ctx.fill();
      ctx.strokeStyle = COL.violet; ctx.lineWidth = 1.5; ctx.stroke();

      var fa = 1 - smooth(0.05, 0.45, e);
      if (fa > 0.01 && g.rx <= W * 0.5) {
        var rw = Math.max(10, g.ry * 0.075), grow = 1 + E * .4, ex = g.rx * grow, ey = g.ry * grow, ecy = g.cy + drop * 0.3;
        ctx.globalAlpha = fa * 0.35; ctx.strokeStyle = COL.violet; ctx.lineWidth = rw * 2.4;
        ctx.beginPath(); ctx.ellipse(g.cx, ecy, ex, ey, 0, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = fa; ctx.strokeStyle = COL.night; ctx.lineWidth = rw;
        ctx.beginPath(); ctx.ellipse(g.cx, ecy, ex, ey, 0, 0, Math.PI * 2); ctx.stroke();
        ctx.lineWidth = 1.6; ctx.strokeStyle = COL.violet;
        ctx.beginPath(); ctx.ellipse(g.cx, ecy, ex + rw / 2, ey + rw / 2, 0, 0, Math.PI * 2); ctx.stroke();
        ctx.strokeStyle = COL.magenta; ctx.globalAlpha = fa * 0.8;
        ctx.beginPath(); ctx.ellipse(g.cx, ecy, ex - rw / 2, ey - rw / 2, 0, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = COL.lavender; ctx.globalAlpha = fa * 0.9;
        for (var k = 0; k < 64; k++) {
          var ang = k / 64 * Math.PI * 2;
          ctx.beginPath(); ctx.arc(g.cx + Math.cos(ang) * ex, ecy + Math.sin(ang) * ey, 1.5, 0, Math.PI * 2); ctx.fill();
        }
      }
      var inn = smooth(0.62, 0.95, e);
      if (inn > 0) {
        var c = g.court;
        ctx.globalAlpha = inn; ctx.fillStyle = rgba(COL.violet, 0.13);
        ctx.fillRect(c.x0, c.y0, c.w, c.h);
      }
      ctx.globalAlpha = 1;
      ctx = fctx;

      // ---------- strings: mains behind, crosses over; non-court strings fade during the morph
      var sw = Math.max(1.4, g.s * 0.075);
      mains.forEach(function (st) {
        var a = st.keep ? 1 : 1 - smooth(0.15, 0.7, e);
        if (a <= 0.01) return;
        var sel = st === g.selMain && e < 0.5;
        var w = sel ? sw * 1.8 : (st.keep ? lerp(sw, 2.2, E) : sw);
        neon(st, 0, a * (st.keep ? 1 : .8), w, sel ? COL.cyan : COL.fg, sel ? COL.magenta : COL.violet, sel ? 0.45 : 0.22);
      });
      crosses.forEach(function (st) {
        var a = st.keep ? 1 : 1 - smooth(0.15, 0.7, e);
        if (a <= 0.01) return;
        var net = st.net && e > 0.5;
        neon(st, 1, a, st.net ? lerp(sw, 4, E) : (st.keep ? lerp(sw, 2.2, E) : sw), net ? COL.magenta : COL.fg, net ? COL.magenta : COL.violet, net ? 0.4 : 0.22);
      });
      // the weave: at alternate crossings the main passes over the cross (cut the cross either side)
      var wv = 1 - smooth(0.05, 0.25, e);
      if (wv > 0.02) {
        crosses.forEach(function (cr, ci) {
          var y = cr.a0[1];
          mains.forEach(function (mn, mi) {
            if ((ci + mi) % 2) return;
            var x = mn.a0[0], top = mn.a0[1], bot = mn.b0[1];
            if (y < top + 2 || y > bot - 2 || x < cr.a0[0] + 2 || x > cr.b0[0] - 2) return;
            var k = Math.round((y - top) / (bot - top) * (mn.d.length - 1));
            var kc = Math.round((x - cr.a0[0]) / (cr.b0[0] - cr.a0[0]) * (cr.d.length - 1));
            var X = x + mn.d[k], Y = y + cr.d[kc], gap = sw * 1.8;
            ctx.globalCompositeOperation = 'destination-out'; ctx.globalAlpha = wv;
            ctx.lineWidth = sw + 2.4; ctx.beginPath(); ctx.moveTo(X - gap, Y); ctx.lineTo(X + gap, Y); ctx.stroke();
            ctx.globalCompositeOperation = 'source-over';
            var sel = mn === g.selMain;
            ctx.strokeStyle = sel ? COL.cyan : COL.fg; ctx.lineWidth = sel ? sw * 1.8 : sw;
            ctx.beginPath(); ctx.moveTo(X, Y - gap * 1.4); ctx.lineTo(X, Y + gap * 1.4); ctx.stroke();
          });
        });
        ctx.globalAlpha = 1;
      }

      // the strings part around the headline so it reads cleanly
      var hm = 1 - smooth(0, 0.3, e);
      if (hm > 0.01 && g.head) {
        var hh = g.head;
        ctx.save();
        ctx.globalCompositeOperation = 'destination-out';
        ctx.translate(hh.x, hh.y); ctx.scale(hh.rx, hh.ry);
        var grd = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
        grd.addColorStop(0, 'rgba(0,0,0,' + (0.86 * hm) + ')');
        grd.addColorStop(0.62, 'rgba(0,0,0,' + (0.78 * hm) + ')');
        grd.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(0, 0, 1, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
      }

      if (e > 0.6) {
        var gp = smooth(0.6, 0.9, e);
        // centre-line gap through the service area
        var xC = g.courtX(C.W / 2);
        ctx.globalCompositeOperation = 'destination-out'; ctx.globalAlpha = gp; ctx.lineWidth = 9;
        ctx.beginPath(); ctx.moveTo(xC, g.courtY(C.NET - C.SHORT) + 3); ctx.lineTo(xC, g.courtY(C.NET + C.SHORT) - 3); ctx.stroke();
        ctx.globalCompositeOperation = 'source-over';
        // the net: a magenta tube past the sidelines, with posts
        var yN = g.courtY(C.NET), xL = g.courtX(0) - 14, xR = g.courtX(C.W) + 14;
        ctx.strokeStyle = COL.magenta; ctx.globalAlpha = gp * 0.35; ctx.lineWidth = 14;
        ctx.beginPath(); ctx.moveTo(xL, yN); ctx.lineTo(xR, yN); ctx.stroke();
        ctx.globalAlpha = gp; ctx.lineWidth = 4;
        ctx.beginPath(); ctx.moveTo(xL, yN); ctx.lineTo(xR, yN); ctx.stroke();
        ctx.fillStyle = COL.magenta; ctx.fillRect(xL - 3, yN - 6, 6, 12); ctx.fillRect(xR - 3, yN - 6, 6, 12);
      }

      // measured annotations: court dimensions and the net label
      if (inn > 0.01) annotate(inn);

      // the rally, strung across the court
      if (e > 0.55 && rally) {
        var pr = smooth(0.58, 0.97, e);
        var pts = [rally.shots[0].from].concat(rally.shots.map(function (s) { return s.to; }));
        var segs = pts.length - 1, total = pr * segs;
        var path = function () {
          ctx.beginPath();
          for (var q = 0; q <= Math.min(segs, Math.ceil(total)); q++) {
            var p = pts[q], frac = Math.min(1, total - (q - 1));
            var X = g.courtX(p[0]), Y = g.courtY(p[1]);
            if (q === 0) ctx.moveTo(X, Y);
            else {
              var pp = pts[q - 1], X0 = g.courtX(pp[0]), Y0 = g.courtY(pp[1]);
              ctx.lineTo(lerp(X0, X, Math.max(0, frac)), lerp(Y0, Y, Math.max(0, frac)));
            }
          }
          ctx.stroke();
        };
        ctx.globalAlpha = 0.4; ctx.strokeStyle = COL.magenta; ctx.lineWidth = 16; path();
        ctx.globalAlpha = 1; ctx.strokeStyle = '#000'; ctx.lineWidth = 8; path();
        ctx.strokeStyle = COL.cyan; ctx.lineWidth = 4; path();
        ctx.strokeStyle = COL.ice; ctx.lineWidth = 1.2; path();
        ctx.font = '800 13px ' + (css('--f-display') || 'sans-serif'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        for (var q = 0; q < Math.floor(total) + 1 && q < pts.length; q++) {
          var X = g.courtX(pts[q][0]), Y = g.courtY(pts[q][1]);
          ctx.fillStyle = '#000'; ctx.beginPath(); ctx.arc(X, Y, 7, 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = COL.cyan; ctx.lineWidth = 2.4; ctx.stroke();
          if (q > 0) {
            var lx = X + (pts[q][0] > C.W / 2 ? -18 : 18), ly = Y - 14;
            ctx.lineWidth = 4; ctx.strokeStyle = '#000'; ctx.strokeText(String(q), lx, ly);
            ctx.fillStyle = COL.fg; ctx.fillText(String(q), lx, ly);
          }
        }
        if (total >= segs - 0.02) {
          var L = pts[pts.length - 1], LX = g.courtX(L[0]), LY = g.courtY(L[1]);
          ctx.strokeStyle = COL.cyan; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.arc(LX, LY, 16, 0, Math.PI * 2); ctx.stroke();
          ctx.globalAlpha = .5; ctx.strokeStyle = COL.magenta;
          ctx.beginPath(); ctx.arc(LX, LY, 26, 0, Math.PI * 2); ctx.stroke(); ctx.globalAlpha = 1;
        }
      }

      // shuttles in flight
      shuttles.forEach(function (sh) {
        var t = sh.t, x, y, sc;
        if (t < 1) { x = lerp(sh.fromX, sh.x, t); y = lerp(sh.fromY, sh.y, t * t); sc = lerp(1.9, 1, t); }
        else { var u = t - 1; x = lerp(sh.x, sh.toX, u); y = lerp(sh.y, sh.toY, Math.sqrt(u)); sc = lerp(1, 1.6, u); }
        var ang = t < 1 ? Math.atan2(sh.y - sh.fromY, sh.x - sh.fromX) : Math.atan2(sh.toY - sh.y, sh.toX - sh.x) + Math.PI;
        drawShuttle(x, y, sc * Math.max(10, geo.s * 0.55), ang, t < 1 ? 1 : 1 - (t - 1));
      });
      ctx.globalAlpha = 1;
    }

    function annotate(a) {
      var c = geo.court, x0 = c.x0, x1 = c.x0 + c.w, y0 = c.y0, y1 = c.y0 + c.h;
      ctx.save();
      ctx.globalAlpha = a; ctx.strokeStyle = COL.muted; ctx.fillStyle = COL.lavender; ctx.lineWidth = 1;
      ctx.font = '700 12px ' + (css('--f-display') || 'sans-serif'); ctx.textBaseline = 'middle';
      // width, above the far baseline
      var yw = y0 - 20;
      ctx.beginPath(); ctx.moveTo(x0, yw); ctx.lineTo(x1, yw); ctx.moveTo(x0, yw - 5); ctx.lineTo(x0, yw + 5); ctx.moveTo(x1, yw - 5); ctx.lineTo(x1, yw + 5); ctx.stroke();
      var tw = C.W.toFixed(2) + ' M';
      ctx.textAlign = 'center';
      var mw = ctx.measureText(tw).width + 12;
      ctx.clearRect((x0 + x1) / 2 - mw / 2, yw - 8, mw, 16);
      ctx.fillText(tw, (x0 + x1) / 2, yw + 1);
      // length, beside the right sideline
      var xl = x1 + 30;
      ctx.beginPath(); ctx.moveTo(xl, y0); ctx.lineTo(xl, y1); ctx.moveTo(xl - 5, y0); ctx.lineTo(xl + 5, y0); ctx.moveTo(xl - 5, y1); ctx.lineTo(xl + 5, y1); ctx.stroke();
      ctx.translate(xl, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2);
      var tl = C.L.toFixed(2) + ' M', ml = ctx.measureText(tl).width + 12;
      ctx.clearRect(-ml / 2, -8, ml, 16);
      ctx.fillText(tl, 0, 1);
      ctx.restore();
      ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = COL.magenta; ctx.font = '800 12px ' + (css('--f-display') || 'sans-serif');
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText('NET', x0 - 24, geo.courtY(C.NET) + 1);
      ctx.restore();
    }

    function drawShuttle(x, y, s, ang, alpha) {
      ctx.save(); ctx.globalAlpha = alpha; ctx.translate(x, y); ctx.rotate(ang - Math.PI / 2);
      ctx.shadowColor = rgba(COL.violet, 0.9); ctx.shadowBlur = 14;
      ctx.fillStyle = COL.fg; ctx.strokeStyle = COL.night; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(-s * 0.32, -s * 0.15); ctx.lineTo(-s * 0.85, -s * 1.9); ctx.lineTo(s * 0.85, -s * 1.9); ctx.lineTo(s * 0.32, -s * 0.15); ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.beginPath(); for (var i = -3; i <= 3; i++) { ctx.moveTo(i * s * 0.1, -s * 0.15); ctx.lineTo(i * s * 0.26, -s * 1.9); } ctx.stroke();
      ctx.fillStyle = COL.magenta; ctx.beginPath(); ctx.arc(0, 0, s * 0.36, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.restore();
    }

    function frame(now) {
      var dt = Math.min(48, now - (lastT || now)); lastT = now;
      var energy = stepPhysics();
      var alive = shuttles.length > 0;
      shuttles.forEach(function (sh) {
        sh.t += dt / (sh.t < 1 ? 300 : 520);
        if (!sh.hit && sh.t >= 1) { sh.hit = true; impact(sh.x, sh.y, 9); if (hooks.contact) hooks.contact(sh.x, sh.y); }
      });
      shuttles = shuttles.filter(function (sh) { return sh.t < 2; });
      draw();
      if (hooks.deflect) hooks.deflect(maxD * geo.mmPerPx);
      if (energy > 0.4 || alive || pointer.on) requestAnimationFrame(frame);
      else { running = false; lastT = 0; if (hooks.deflect) hooks.deflect(0); }
    }
    function kick() { if (!running) { running = true; requestAnimationFrame(frame); } }

    // ---- input
    function local(ev) { var r = canvas.getBoundingClientRect(); return { x: ev.clientX - r.left, y: ev.clientY - r.top }; }
    if (!RM) {
      canvas.addEventListener('pointermove', function (ev) {
        if (e > 0.2) { pointer.on = false; return; }
        var p = local(ev); pointer.x = p.x; pointer.y = p.y; pointer.on = ev.pointerType === 'mouse'; kick();
      });
      canvas.addEventListener('pointerleave', function () { pointer.on = false; });
      canvas.addEventListener('pointerdown', function (ev) {
        if (e > 0.2) return;
        var p = local(ev);
        var dx = (p.x - geo.cx) / geo.rx, dy = (p.y - geo.cy) / geo.ry;
        if (dx * dx + dy * dy <= 1) serve(p.x, p.y); else impact(p.x, p.y, 3);
        kick();
      });
    }

    readColors();
    layout();
    draw();
    return {
      geo: function () { return geo; },
      resize: function () { layout(); draw(); },
      setMorph: function (v) { e = v; if (!running) draw(); },
      serve: function () { serve(geo.cx + geo.rx * 0.12, geo.cy + geo.ry * 0.12); }
    };
  }

  // ------------------------------------------------------------------ step diagrams
  function steps() {
    var NS = 'http://www.w3.org/2000/svg';
    function el(tag, a, p) { var n = document.createElementNS(NS, tag); for (var k in a) n.setAttribute(k, a[k]); p.appendChild(n); return n; }
    document.querySelectorAll('[data-step]').forEach(function (svg) {
      var k = svg.getAttribute('data-step');
      if (k === 'upload') {
        svg.setAttribute('viewBox', '0 0 300 110');
        var g = el('g', { stroke: 'var(--fg)', fill: 'none', 'stroke-width': 2 }, svg);
        el('rect', { x: 2, y: 8, width: 150, height: 96, 'stroke-width': 2, stroke: 'var(--violet)' }, g);
        for (var x = 20; x < 150; x += 18) el('line', { x1: x, y1: 8, x2: x, y2: 104, opacity: .3 }, g);
        for (var y = 22; y < 104; y += 18) el('line', { x1: 2, y1: y, x2: 152, y2: y, opacity: .3 }, g);
        el('path', { d: 'M62 38l34 18-34 18z', fill: 'var(--magenta)', stroke: 'none' }, g);
        el('path', { d: 'M168 56h70M226 44l14 12-14 12', 'stroke-width': 3, stroke: 'var(--cyan)' }, g);
        var t = el('text', { x: 252, y: 62, fill: 'var(--fg)', stroke: 'none', 'font-family': 'var(--f-display)', 'font-weight': 800, 'font-size': 18, 'letter-spacing': 2 }, g); t.textContent = 'GO';
      } else if (k === 'track') {
        svg.setAttribute('viewBox', '-0.4 -0.4 14.2 6.9');
        var g2 = el('g', { transform: 'translate(0 6.1) rotate(-90)' }, svg);
        el('rect', { x: 0, y: 0, width: C.W, height: C.L, fill: 'rgba(120,40,255,.16)', stroke: 'var(--fg)', 'stroke-width': .06 }, g2);
        el('path', { d: 'M0.46 0V13.4M5.64 0V13.4M0 0.76H6.1M0 12.64H6.1M0 4.72H6.1M0 8.68H6.1M3.05 0V4.72M3.05 8.68V13.4', stroke: 'var(--fg)', 'stroke-width': .045, fill: 'none', opacity: .8 }, g2);
        el('line', { x1: -0.3, y1: C.NET, x2: C.W + 0.3, y2: C.NET, stroke: 'var(--magenta)', 'stroke-width': .12 }, g2);
        var r = M.rallies[M.turningPoints[2].rally];
        var d = [r.shots[0].from].concat(r.shots.map(function (s) { return s.to; })).map(function (p, i) { return (i ? 'L' : 'M') + p[0] + ',' + p[1]; }).join('');
        el('path', { d: d, stroke: '#000', 'stroke-width': .18, fill: 'none', 'stroke-linejoin': 'round' }, g2);
        el('path', { d: d, stroke: 'var(--cyan)', 'stroke-width': .08, fill: 'none', 'stroke-linejoin': 'round' }, g2);
      } else {
        svg.setAttribute('viewBox', '0 0 300 110');
        var list = M.rallies.filter(function (x) { return x.game === 2; }).slice(14, 40);
        var bw = 300 / list.length;
        el('line', { x1: 0, y1: 56, x2: 300, y2: 56, stroke: 'var(--fg)', 'stroke-width': 2 }, svg);
        list.forEach(function (x, i) {
          var hh = Math.min(48, x.length * 2), up = x.winner === 'A', hl = x.index === M.turningPoints[1].rally + 3;
          el('rect', { x: i * bw + 1.5, y: up ? 56 - hh : 56, width: bw - 3, height: hh, fill: hl ? 'var(--cyan)' : (up ? 'var(--lavender)' : 'transparent'), stroke: hl ? 'var(--cyan)' : 'var(--violet-soft)', 'stroke-width': up && !hl ? 0 : 1.4 }, svg);
        });
      }
    });
  }

  // ------------------------------------------------------------------ ticker: measured sample facts
  function tickers() {
    var longest = M.rallies.reduce(function (m, r) { return Math.max(m, r.length); }, 0);
    var items = [
      '<i>Sample</i> illustrative · fictional players',
      '<b>' + M.rallies.length + '</b> rallies',
      M.games.map(function (g, i) { return 'Game ' + (i + 1) + ' <b>' + g.score.A + '–' + g.score.B + '</b>'; }).join('</span><span>'),
      'Longest rally <b>' + longest + '</b> shots',
      '<b>' + M.turningPoints.length + '</b> turning points',
      '<b>' + M.util.fmtDuration(M.duration) + '</b> of play',
      'Formats <b>MP4 · MOV</b>'
    ];
    var run = '<span>' + items.join('</span><span>') + '</span>';
    document.querySelectorAll('[data-ticker]').forEach(function (t) { t.innerHTML = run + run + run + run; });
  }

  // ------------------------------------------------------------------ boot
  document.addEventListener('DOMContentLoaded', function () {
    var hero = document.querySelector('[data-hero]'), canvas = document.querySelector('[data-bed]');
    var nav = document.querySelector('[data-nav]');
    var plate = document.querySelector('[data-plate]');
    plate.innerHTML = '<i>Sample</i> ' + P('A').code + ' ' + M.games.map(function (g) { return g.score.A + '–' + g.score.B; }).join(' ') + ' ' + P('B').code + ' <b>/</b> ' + M.rallies.length + ' rallies';
    document.querySelector('[data-sample-plate]').innerHTML = '<i>' + M.rallies.length + '</i> rallies <b>/</b> ' + M.util.fmtDuration(M.duration) + ' <b>/</b> 3 games';
    document.querySelector('[data-cap-text]').textContent = 'Game 2, ' + GM.score(rally.scoreBefore) + ' → ' + GM.score(rally.scoreAfter) + ': ' + rally.length + ' shots, ending with ' + GM.endingText(rally).split(' · ')[0].toLowerCase() + ' into Lindqvist’s rear backhand corner. Okafor’s sixth straight point.';

    // ---- HUD
    var hud = {
      modeK: hero.querySelector('[data-hud-mode-k]'), bed: hero.querySelector('[data-hud-bed]'),
      defK: hero.querySelector('[data-hud-def-k]'), def: hero.querySelector('[data-hud-def]'), meter: hero.querySelector('[data-hud-meter]'),
      hitK: hero.querySelector('[data-hud-hit-k]'), hit: hero.querySelector('[data-hud-hit]'), cue: hero.querySelector('[data-hud-cue]')
    };
    var courtMode = null, lastHit = 'Click the strings';
    var bed;
    function setHudMode(court) {
      if (court === courtMode) return; courtMode = court;
      var gm = bed.geo();
      hero.querySelector('.hud').classList.toggle('court', court);
      if (court) {
        hud.modeK.textContent = 'Court'; hud.bed.innerHTML = '<b>' + C.W.toFixed(2) + '</b> × <b>' + C.L.toFixed(2) + '</b> m';
        hud.defK.textContent = 'Rally'; hud.def.innerHTML = '<b>' + rally.length + '</b> shots'; hud.meter.style.transform = 'scaleX(' + Math.min(1, rally.length / 20) + ')';
        hud.hitK.textContent = 'Score'; hud.hit.innerHTML = GM.score(rally.scoreBefore) + ' → <b>' + GM.score(rally.scoreAfter) + '</b>';
        hud.cue.firstChild.textContent = 'To the sample ';
      } else {
        hud.modeK.textContent = 'String bed'; hud.bed.innerHTML = '<b>' + gm.counts.mains + '</b> mains × <b>' + gm.counts.crosses + '</b> crosses';
        hud.defK.textContent = 'Max deflection'; hud.def.innerHTML = '<b>0.0</b> mm'; hud.meter.style.transform = 'scaleX(0)';
        hud.hitK.textContent = 'Last contact'; hud.hit.innerHTML = lastHit;
        hud.cue.firstChild.textContent = 'String the court ';
      }
    }

    var sky = Sky(hero.querySelector('[data-sky]'));
    bed = StringBed(canvas, hero, {
      deflect: function (mm) {
        if (courtMode) return;
        hud.def.firstChild.textContent = mm.toFixed(1);
        hud.meter.style.transform = 'scaleX(' + Math.min(1, mm / 12) + ')';
      },
      contact: function (x, y) {
        var g = bed.geo(), off = Math.hypot(x - g.cx, y - (g.cy + g.ry * 0.1)) * g.mmPerPx;
        lastHit = off < 18 ? '<b>Sweet spot</b>' : '<b>' + Math.round(off) + '</b> mm off the sweet spot';
        if (!courtMode) hud.hit.innerHTML = lastHit;
      }
    });
    setHudMode(false);

    var grip = document.querySelector('[data-grip]'), ink = hero.querySelector('.ink'), lede = hero.querySelector('.lede-wrap'),
        cap = document.querySelector('[data-cap]'), key = document.querySelector('[data-key]'), plateTop = hero.querySelector('.plate-top'),
        hudEl = hero.querySelector('.hud');

    // ---- the morph, driven by a clock
    function apply(e) {
      bed.setMorph(e);
      var out = 1 - smooth(0, 0.28, e);
      ink.style.opacity = out; ink.style.transform = 'translateY(' + (-e * 80) + 'px)';
      lede.style.opacity = out; plateTop.style.opacity = out;
      grip.style.transform = 'translate(-50%, ' + (smooth(0, 0.4, e) * 120) + '%)';
      grip.style.visibility = e > 0.42 ? 'hidden' : 'visible';
      // caption and key arrive in sequence once the strings have landed
      var c1 = smooth(0.74, 0.94, e), c2 = smooth(0.82, 1, e);
      cap.style.opacity = c1; cap.style.transform = (window.innerWidth < 860 ? '' : 'translateY(-50%) ') + 'translateY(' + ((1 - c1) * 16) + 'px)';
      key.style.opacity = c2; key.style.transform = 'translateY(-50%) translateY(' + ((1 - c2) * 16) + 'px)';
      cap.style.pointerEvents = c1 > 0.5 ? 'auto' : 'none';
      var hf = Math.abs(e - 0.5) * 2;  // HUD dips while it swaps from racket to court readings
      hudEl.style.opacity = smooth(0.2, 0.9, hf);
      setHudMode(e > 0.5);
    }
    var mo = { e: 0, target: 0, raf: 0, last: 0 };
    function tick(now) {
      var dt = mo.last ? Math.min(64, now - mo.last) : 16; mo.last = now;
      var dir = mo.target > mo.e ? 1 : -1;
      mo.e = clamp(mo.e + dir * dt / (dir > 0 ? MORPH_MS : UNMORPH_MS), 0, 1);
      apply(mo.e);
      if (mo.e !== mo.target) mo.raf = requestAnimationFrame(tick);
      else { mo.raf = 0; mo.last = 0; }
    }
    function setTarget(t, instant) {
      mo.target = t;
      if (instant) { if (mo.raf) cancelAnimationFrame(mo.raf); mo.raf = 0; mo.last = 0; mo.e = t; apply(t); return; }
      if (!mo.raf && mo.e !== t) mo.raf = requestAnimationFrame(tick);
    }

    // ---- scroll: crossing the trigger starts the morph; the stage stays pinned until it finishes
    var TRIGGER = 24, jumpUntil = 0;
    function pinEnd() { return hero.offsetTop + hero.offsetHeight - window.innerHeight; }
    function playing() { return !RM && mo.target === 1 && mo.e < 1; }
    function onScroll() {
      var y = window.scrollY, want = y > hero.offsetTop + TRIGGER ? 1 : 0;
      if (RM || performance.now() < jumpUntil) setTarget(want, true);
      else if (want !== mo.target) {
        // landing far past the hero (restored scroll position, End key): nothing to watch, so settle at once
        if (want === 1 && mo.e === 0 && y > pinEnd() + window.innerHeight) setTarget(1, true);
        else setTarget(want);
      }
      if (playing() && y > pinEnd()) window.scrollTo({ top: pinEnd(), behavior: 'instant' });
      nav.classList.toggle('solid', y > hero.offsetHeight - window.innerHeight - 10);
    }
    // hold the pin against wheel, touch and keyboard while the morph is playing
    function atPin() { return playing() && window.scrollY >= pinEnd() - 1; }
    window.addEventListener('wheel', function (ev) { if (ev.deltaY > 0 && atPin()) ev.preventDefault(); }, { passive: false });
    var touchY = 0;
    window.addEventListener('touchstart', function (ev) { touchY = ev.touches[0].clientY; }, { passive: true });
    window.addEventListener('touchmove', function (ev) { if (ev.touches[0].clientY < touchY && atPin()) ev.preventDefault(); }, { passive: false });
    window.addEventListener('keydown', function (ev) {
      if (/^(ArrowDown|PageDown|End| )$/.test(ev.key) && atPin() && !ev.target.closest('input, textarea, select, [contenteditable]')) ev.preventDefault();
    });
    // in-page links jump straight past: settle the morph instantly instead of pinning them
    document.addEventListener('click', function (ev) { if (ev.target.closest('a[href^="#"]')) jumpUntil = performance.now() + 1400; });

    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', function () { bed.resize(); sky.resize(); courtMode = null; setHudMode(mo.e > 0.5); onScroll(); });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { bed.resize(); });
    apply(0);
    onScroll();
    // one serve on arrival shows the bed is live (not under reduced motion)
    if (!RM) setTimeout(function () { if (mo.e === 0) bed.serve(); }, 900);

    steps();
    tickers();
    GM.mountSample(document.querySelector('[data-sample]'));
    GM.mountUpload(document.querySelector('[data-upload]'), { id: 'home' });
  });
})();
