/* Goodminton — upload flow
 * Sends the file to the local analysis service (backend/server.py).
 */
(function () {
  'use strict';
  var GM = window.GM = window.GM || {};
  var L = window.GM_LIMITS;

  function h(tag, attrs, children) { return GM.h(tag, attrs, children); }
  function bytes(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2).replace(/\.?0+$/, '') + ' GB';
    if (n >= 1e6) return Math.round(n / 1e6) + ' MB';
    return Math.max(1, Math.round(n / 1e3)) + ' KB';
  }
  var ERR_ICON = '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M10 5.5v5.5M10 13.4v1.2" stroke="currentColor" stroke-width="1.8"/></svg>';
  var UP_ICON = '<svg viewBox="0 0 40 40" width="40" height="40" aria-hidden="true"><path d="M20 27V9M12.5 16.5L20 9l7.5 7.5" fill="none" style="stroke:var(--green)" stroke-width="2.4" stroke-linecap="square"/><path d="M8 26v6h24v-6" fill="none" style="stroke:var(--ink)" stroke-width="2.4"/></svg>';

  GM.mountUpload = function (root, opts) {
    opts = opts || {};
    var state = { file: null, duration: null, upload: null };

    var input = h('input', { type: 'file', id: (opts.id || 'up') + '-file', accept: L.mime.join(',') + ',.mp4,.mov' });
    var errBox = h('div', { class: 'err', role: 'alert', hidden: true });
    var limits = h('div', { class: 'file-limits', id: (opts.id || 'up') + '-limits' }, [
      h('span', { html: '<b>Formats</b> ' + L.formats.join(' · ') }),
      h('span', { html: '<b>Size</b> up to ' + L.maxSizeGB + ' GB' }),
      h('span', { html: '<b>Length</b> up to ' + L.maxDurationMin + ' min' })
    ]);
    if (L.provisional) limits.appendChild(h('span', { class: 'tag', title: 'Limits will be confirmed once the processing service is measured' }, ['Provisional']));
    input.setAttribute('aria-describedby', limits.id);

    var court = '<svg class="drop-court" viewBox="0 0 134 61" preserveAspectRatio="none" aria-hidden="true"><g fill="none" style="stroke:var(--green)" stroke-width="1" vector-effect="non-scaling-stroke"><rect x=".5" y=".5" width="133" height="60" vector-effect="non-scaling-stroke"/><path d="M.5 5.1h133M.5 55.9h133M7.6 .5v60M126.4 .5v60M47.2 .5v60M86.8 .5v60M7.6 30.5h39.6M86.8 30.5h39.6" vector-effect="non-scaling-stroke"/></g><path d="M67 0v61" style="stroke:var(--ink)" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>';
    var drop = h('label', { class: 'drop', for: input.id });
    drop.innerHTML = court;
    var inner = h('div', { class: 'drop-inner' });
    inner.innerHTML = UP_ICON;
    inner.appendChild(h('h3', null, ['Drop a match recording here']));
    inner.appendChild(h('span', { class: 'btn btn-primary', 'aria-hidden': 'true' }, ['Choose a video']));
    inner.appendChild(limits);
    drop.appendChild(inner);
    drop.appendChild(input);

    var idle = h('div', { class: 'up-idle', style: 'display:flex;flex-direction:column;gap:12px' }, [drop, errBox]);

    // selected / uploading state
    var card = h('div', { class: 'filecard', hidden: true });
    var name = h('div', { class: 'name' });
    var meta = h('div', { class: 'meta num' });
    var remove = h('button', { class: 'btn btn-small', type: 'button' }, ['Remove']);
    card.appendChild(h('div', { class: 'filecard-row' }, [h('div', null, [name, meta]), remove]));

    var form = h('form', { class: 'up-form', style: 'display:grid;gap:14px', novalidate: true });
    function field(id, label, hint, attrs) {
      var f = h('div', { class: 'field' }, [h('label', { for: id }, [label]), h('input', Object.assign({ id: id, name: id, autocomplete: 'off' }, attrs || {}))]);
      if (hint) f.appendChild(h('span', { class: 'hint' }, [hint]));
      return f;
    }
    var pfx = opts.id || 'up';
    form.appendChild(h('div', { class: 'form-row' }, [
      field(pfx + '-near', 'Near-side player', 'Closest to the camera at the start', { placeholder: 'e.g. M. Lindqvist', required: true }),
      field(pfx + '-far', 'Far-side player', null, { placeholder: 'e.g. A. Okafor', required: true })
    ]));
    var formErr = h('div', { class: 'err', role: 'alert', hidden: true });
    var startArrow = h('span', { class: 'arrow-ic', 'aria-hidden': 'true' });
    startArrow.innerHTML = GM.ICON.arrow;
    var start = h('button', { class: 'btn btn-primary', type: 'submit' }, ['Upload and analyse ', startArrow]);
    form.appendChild(formErr);
    form.appendChild(h('div', null, [start]));
    card.appendChild(form);

    var progress = h('div', { class: 'up-progress', hidden: true, style: 'display:grid;gap:8px' });
    var meter = h('div', { class: 'meter', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': 0, 'aria-label': 'Upload progress' }, [h('i')]);
    var ptext = h('div', { class: 'small num', 'aria-live': 'polite' });
    var cancel = h('button', { class: 'btn btn-small', type: 'button' }, ['Cancel upload']);
    progress.appendChild(meter);
    progress.appendChild(h('div', { style: 'display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap' }, [ptext, cancel]));
    card.appendChild(progress);

    var mainCol = h('div', { class: 'up-main', style: 'display:flex;flex-direction:column;gap:16px;min-width:0' }, [idle, card]);

    var privacy = h('aside', { class: 'privacy', 'aria-label': 'Privacy and recording guidance' });
    privacy.innerHTML =
      '<h3>Private by default <span class="tag" title="Policy to be confirmed before launch">Provisional</span></h3>' +
      '<p>Your recording is visible only to your account. It is used to analyse this match and nothing else. You can delete the video and its analysis at any time.</p>' +
      '<h3 style="margin-top:8px">For a reliable analysis <span class="tag" title="Requirements to be confirmed on real recordings">Provisional</span></h3>' +
      '<ul>' + L.recording.map(function (x) { return '<li>' + x + '</li>'; }).join('') + '</ul>';

    root.classList.add('upload');
    root.appendChild(mainCol);
    root.appendChild(privacy);

    function showError(msg) {
      errBox.innerHTML = ERR_ICON + '<span>' + msg + '</span>';
      errBox.hidden = false;
      drop.classList.add('is-error');
    }
    function clearError() { errBox.hidden = true; drop.classList.remove('is-error'); }

    function validate(file) {
      var ext = (file.name.split('.').pop() || '').toLowerCase();
      var okType = L.mime.indexOf(file.type) !== -1 || ext === 'mp4' || ext === 'mov';
      if (!okType) return '“' + file.name + '” is not a supported video. Choose an ' + L.formats.join(' or ') + ' file.';
      if (file.size > L.maxSizeGB * 1e9) return 'This file is ' + bytes(file.size) + '. The limit is ' + L.maxSizeGB + ' GB; trim the warm-up or export at a lower bitrate.';
      return null;
    }

    function choose(file) {
      clearError();
      if (!file) return;
      var err = validate(file);
      if (err) { showError(err); return; }
      state.file = file; state.duration = null;
      name.textContent = file.name;
      meta.textContent = bytes(file.size) + ' · reading length…';
      idle.hidden = true; card.hidden = false; form.hidden = false; progress.hidden = true;
      // read duration from metadata when the browser can
      try {
        var v = document.createElement('video');
        v.preload = 'metadata';
        var url = URL.createObjectURL(file);
        v.onloadedmetadata = function () {
          URL.revokeObjectURL(url);
          state.duration = v.duration;
          if (v.duration > L.maxDurationMin * 60) {
            reset();
            showError('This recording is ' + Math.round(v.duration / 60) + ' minutes long. The limit is ' + L.maxDurationMin + ' minutes; trim it to the match itself.');
            return;
          }
          meta.textContent = bytes(file.size) + ' · ' + GM_MATCH.util.fmtTime(v.duration) + (v.videoHeight ? ' · ' + v.videoHeight + 'p' : '');
        };
        v.onerror = function () { URL.revokeObjectURL(url); meta.textContent = bytes(file.size) + ' · length unknown (checked during processing)'; };
        v.src = url;
      } catch (e) { meta.textContent = bytes(file.size); }
      var first = form.querySelector('input');
      if (first) setTimeout(function () { first.focus(); }, 30);
    }

    function reset() {
      if (state.upload) state.upload.abort();
      state = { file: null, duration: null, upload: null };
      input.value = '';
      card.hidden = true; idle.hidden = false;
      start.disabled = false;
      form.querySelectorAll('input').forEach(function (i) { i.disabled = false; });
    }

    input.addEventListener('change', function () { choose(input.files[0]); });
    ['dragenter', 'dragover'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add('is-over'); }); });
    ['dragleave', 'dragend', 'drop'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove('is-over'); }); });
    drop.addEventListener('drop', function (e) { var f = e.dataTransfer && e.dataTransfer.files[0]; choose(f); });
    remove.addEventListener('click', function () { reset(); drop.focus && input.focus(); });

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var near = form.querySelector('#' + pfx + '-near'), far = form.querySelector('#' + pfx + '-far');
      var missing = [near, far].filter(function (i) { return !i.value.trim(); });
      [near, far].forEach(function (i) { i.setAttribute('aria-invalid', String(!i.value.trim())); });
      if (missing.length) {
        formErr.style.color = '';
        formErr.innerHTML = ERR_ICON + '<span>Name both players so the report can label each side of the court.</span>';
        formErr.hidden = false; missing[0].focus(); return;
      }
      formErr.hidden = true;
      transfer(near.value.trim(), far.value.trim());
    });

    function restoreForm(message) {
      state.upload = null;
      progress.hidden = true; form.hidden = false; remove.hidden = false;
      meter.querySelector('i').style.width = '0';
      ptext.textContent = '';
      formErr.textContent = message;
      formErr.hidden = false;
    }

    function transfer(a, b) {
      var f = state.file;
      form.hidden = true; progress.hidden = false; remove.hidden = true; formErr.hidden = true; formErr.style.color = '';
      ptext.textContent = 'Starting upload…';
      state.upload = GM.pipeline.upload(f, { near: a, far: b }, function (sent, total) {
        var p = sent / total;
        meter.querySelector('i').style.width = (p * 100).toFixed(1) + '%';
        meter.setAttribute('aria-valuenow', Math.round(p * 100));
        ptext.textContent = Math.round(p * 100) + '% · ' + bytes(sent) + ' of ' + bytes(total);
      });
      state.upload.promise.then(function (job) {
        location.href = '/strung/processing.html?job=' + encodeURIComponent(job.id);
      }, function (err) {
        if (err.message === 'cancelled') { formErr.style.color = 'var(--ink-2)'; restoreForm('Upload cancelled. Nothing was kept. Start again, or remove the file.'); start.focus(); return; }
        restoreForm(err.message);
      });
    }
    cancel.addEventListener('click', function () { if (state.upload) state.upload.abort(); });
  };
})();
