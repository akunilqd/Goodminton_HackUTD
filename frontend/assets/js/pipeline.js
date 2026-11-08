/* Goodminton — client for the local analysis service (backend/server.py). */
(function () {
  'use strict';
  var GM = window.GM = window.GM || {};
  var served = /^https?:$/.test(location.protocol);
  var HEADERS = { 'X-Goodminton': '1' };

  function PipelineError(message, status) { this.name = 'PipelineError'; this.message = message; this.status = status || null; }
  PipelineError.prototype = Object.create(Error.prototype);
  var NOT_RUNNING = 'The analysis server isn’t reachable. On the analysis computer run “python server.py”, then open http://localhost:8000.';

  async function request(path, options) {
    if (!served) throw new PipelineError(NOT_RUNNING);
    var res;
    try { res = await fetch(path, Object.assign({ cache: 'no-store' }, options)); }
    catch (e) { throw new PipelineError(NOT_RUNNING); }
    var body = null;
    try { body = await res.json(); } catch (e) { /* non-JSON error page */ }
    if (!res.ok) throw new PipelineError((body && body.detail) || 'The analysis server had a problem.', res.status);
    return body;
  }

  // XHR rather than fetch: only XHR reports upload progress.
  function upload(file, fields, onProgress) {
    var xhr = new XMLHttpRequest();
    var promise = new Promise(function (resolve, reject) {
      if (!served) { reject(new PipelineError(NOT_RUNNING)); return; }
      var q = new URLSearchParams({ filename: file.name, near: fields.near, far: fields.far });
      xhr.open('POST', '/api/jobs?' + q.toString());
      xhr.setRequestHeader('X-Goodminton', '1');
      xhr.setRequestHeader('Content-Type', 'application/octet-stream');
      xhr.upload.onprogress = function (e) { if (e.lengthComputable) onProgress(e.loaded, e.total); };
      xhr.onload = function () {
        var body = null;
        try { body = JSON.parse(xhr.responseText); } catch (e) { /* handled below */ }
        if (xhr.status === 201 && body) resolve(body);
        else reject(new PipelineError((body && body.detail) || 'The upload failed. Please try again.', xhr.status));
      };
      xhr.onerror = function () { reject(new PipelineError(NOT_RUNNING)); };
      xhr.onabort = function () { reject(new PipelineError('cancelled', 0)); };
      xhr.send(file);
    });
    return { promise: promise, abort: function () { xhr.abort(); } };
  }

  GM.pipeline = {
    served: served,
    upload: upload,
    job: function (id) { return request('/api/jobs/' + encodeURIComponent(id)); },
    cancel: function (id) { return request('/api/jobs/' + encodeURIComponent(id) + '/cancel', { method: 'POST', headers: HEADERS }); },
    // Names and the original video exist only for runs uploaded through this service.
    jobIfAny: function (id) { return /^job_/.test(id || '') && served ? this.job(id).catch(function () { return null; }) : Promise.resolve(null); },
    NOT_RUNNING: NOT_RUNNING
  };
})();
