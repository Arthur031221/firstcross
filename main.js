(() => {
'use strict';
const RM = matchMedia('(prefers-reduced-motion: reduce)').matches;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

requestAnimationFrame(() => requestAnimationFrame(() => document.documentElement.classList.add('go')));

// Title motion after the OpenMotion "Point Cloud" style card (github.com/nexu-io/openmotion, Apache-2.0):
// single photons land at random over the title and only the ones on a letter show, at a flux that rises
// tenfold, then tenfold again, until the letters are solid. Plays once; the ray and its first crossing follow.
(() => {
  const h = $('.wordmark');
  if (!h) return;
  const show = () => h.classList.add('ph-on', 'lit');
  if (RM || matchMedia('(forced-colors: active)').matches) { show(); return; }
  const run = () => {
    let bg = 'rgb(255, 255, 255)';
    for (let e = h.parentElement; e; e = e.parentElement) {
      const c = getComputedStyle(e).backgroundColor;
      if (c && c !== 'transparent' && !/rgba\(.*,\s*0\)$/.test(c)) { bg = c; break; }
    }
    const r = h.getBoundingClientRect(), pad = 14;
    const rg = document.createRange(); rg.selectNodeContents(h);
    const right = rg.getBoundingClientRect().right;
    const W = Math.ceil(right - r.left + 2 * pad), H = Math.ceil(r.height + 2 * pad);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cv = document.createElement('canvas');
    cv.className = 'ph-cv'; cv.setAttribute('aria-hidden', 'true');
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    cv.style.cssText = `width:${W}px;height:${H}px;left:${-pad}px;top:${-pad}px`;
    const ctx = cv.getContext('2d');
    if (!ctx) { show(); return; }
    ctx.scale(dpr, dpr); ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    ctx.globalCompositeOperation = 'destination-out'; ctx.fillStyle = '#000';
    h.appendChild(cv); h.classList.add('ph-on');
    // about 92% of the area is hit by the end, the rest goes with the fade
    const T = 1800, N = Math.round(W * H / 1.6);
    let n = 0, t0 = 0, stop = false;
    const end = () => { stop = true; h.classList.add('lit'); cv.style.opacity = '0'; setTimeout(() => cv.remove(), 220); };
    addEventListener('resize', end, { once: true });
    const step = (now) => {
      if (stop) return;
      if (!t0) t0 = now;
      const u = Math.min(1, (now - t0) / T), goal = Math.round(N * (0.04 + 0.96 * (Math.pow(16, u) - 1) / 15));
      ctx.beginPath();
      for (; n < goal; n++) ctx.rect(Math.random() * W, Math.random() * H, 2, 2);
      ctx.fill();
      if (u < 1) { requestAnimationFrame(step); return; }
      removeEventListener('resize', end); end();
    };
    requestAnimationFrame(step);
  };
  // starts with the page's tm-go signal (parsed page, fonts and an idle main thread, at most 2 s)
  new Promise((ok) => { if (document.documentElement.classList.contains('tm-go')) ok(); else document.addEventListener('tm-go', ok, { once: true }); }).then(run, show);
})();


/* result videos: play while at least 40% is visible, pause when it leaves */
(() => {
  const vs = $$('video[muted]');
  if (!vs.length || RM || !('IntersectionObserver' in window)) return;
  const hand = new WeakSet(), autoOff = new WeakSet(), autoOn = new WeakSet();
  vs.forEach(v => {
    v.addEventListener('pause', () => { if (autoOff.has(v)) autoOff.delete(v); else hand.add(v); });
    v.addEventListener('play', () => { if (autoOn.has(v)) autoOn.delete(v); else hand.delete(v); });
  });
  const io = new IntersectionObserver(es => es.forEach(e => {
    const v = e.target;
    if (e.isIntersecting) {
      if (hand.has(v) || !v.paused) return;
      autoOn.add(v); v.play().catch(() => { autoOn.delete(v); });
    } else if (!v.paused) { autoOff.add(v); v.pause(); }
  }), { threshold: 0.4 });
  vs.forEach(v => io.observe(v));
})();

  /* contents: mark the section being read */
  (() => {
    const toc = document.querySelector('.toc');
    if (!toc) return;
    const list = toc.querySelector('ol');
    const links = [...toc.querySelectorAll('a[href^="#"]')];
    const secs = links.map((a) => document.getElementById(a.getAttribute('href').slice(1)));
    if (secs.some((s) => !s)) return;
    const still = matchMedia('(prefers-reduced-motion: reduce)');
    let cur = -1, queued = false;
    const set = (i) => {
      if (i === cur) return;
      cur = i;
      links.forEach((a, k) => { if (k === i) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current'); });
      // the phone strip scrolls sideways: keep the current label in view
      if (i >= 0 && list.scrollWidth > list.clientWidth + 1) {
        const a = links[i], left = a.offsetLeft - (parseFloat(getComputedStyle(list).paddingLeft) || 0);
        if (a.offsetLeft < list.scrollLeft || a.offsetLeft + a.offsetWidth > list.scrollLeft + list.clientWidth) list.scrollTo({ left, behavior: still.matches ? 'auto' : 'smooth' });
      }
    };
    const pick = () => {
      queued = false;
      const line = innerHeight * 0.35;
      let i = -1;
      secs.forEach((s, k) => { if (s.getBoundingClientRect().top <= line) i = k; });
      if (innerHeight + scrollY >= document.documentElement.scrollHeight - 40) i = secs.length - 1;
      set(i);
    };
    const ask = () => { if (!queued) { queued = true; requestAnimationFrame(pick); } };
    addEventListener('scroll', ask, { passive: true });
    addEventListener('resize', ask);
    pick();
  })();

  /* chef meshes at 1 PPP: one WebGL canvas, two panes, one camera; drag turns, wheel or pinch zooms */
  (() => {
    const fig = $('#m3d'); if (!fig) return;
    const stage = $('.m3-stage', fig), cv = $('canvas', stage), pre = $('.m3-pre', fig), hint = $('.m3-hint', fig), labs = $$('.m3-labs span', fig);
    let gl = null;
    try { gl = cv.getContext('webgl', { antialias: true, alpha: false, preserveDrawingBuffer: true }); } catch (e) { gl = null; }
    if (!gl) return;
    const sh = (t, src) => { const o = gl.createShader(t); gl.shaderSource(o, src); gl.compileShader(o); if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error('shader'); return o; };
    const P = gl.createProgram();
    try {
      gl.attachShader(P, sh(gl.VERTEX_SHADER, 'attribute vec3 q;attribute vec3 n;uniform mat4 M;uniform float B;varying vec3 vn;varying vec3 vp;void main(){vp=(q*2.-1.)*B;vn=n;gl_Position=M*vec4(vp,1.);}'));
      gl.attachShader(P, sh(gl.FRAGMENT_SHADER, '#ifdef GL_FRAGMENT_PRECISION_HIGH\nprecision highp float;\n#else\nprecision mediump float;\n#endif\nvarying vec3 vn;varying vec3 vp;uniform vec3 E,L1,L2;void main(){vec3 N=normalize(vn);if(dot(N,E-vp)<0.)N=-N;float l=.2+.66*max(dot(N,L1),0.)+.24*max(dot(N,L2),0.);gl_FragColor=vec4(vec3(.74*l),1.);}'));
      gl.linkProgram(P); if (!gl.getProgramParameter(P, gl.LINK_STATUS)) throw new Error('link');
    } catch (e) { return; }
    const A = { q: gl.getAttribLocation(P, 'q'), n: gl.getAttribLocation(P, 'n') };
    const U = {}; ['M', 'B', 'E', 'L1', 'L2'].forEach((k) => { U[k] = gl.getUniformLocation(P, k); });
    let dead = false;
    cv.addEventListener('webglcontextlost', (e) => { e.preventDefault(); dead = true; cv.hidden = true; hint.hidden = true; pre.hidden = false; });

    /* data: per mesh, uint16 positions over the crop box and uint16 triangle indices; normals are built here */
    let G = null;
    const cam = { az: 0, el: 0, zoom: 1, az0: 0, el0: 0 };
    const reset = () => { cam.az = cam.az0; cam.el = cam.el0; cam.zoom = 1; };
    async function load() {
      const ok = (r) => { if (!r.ok) throw new Error('http'); return r; };
      const d = await fetch('media/3d/chef1.json').then(ok).then((r) => r.json());
      const ab = await fetch('media/3d/' + d.bin).then(ok).then((r) => r.arrayBuffer());
      const meshes = ['ta', 'ours'].map((k) => {
        const m = d[k], q = new Uint16Array(ab, m.pos[0], m.pos[1] / 2), f = new Uint16Array(ab, m.idx[0], m.idx[1] / 2), n = new Float32Array(q.length);
        for (let i = 0; i < f.length; i += 3) {
          const a = 3 * f[i], b = 3 * f[i + 1], c = 3 * f[i + 2];
          const ux = q[b] - q[a], uy = q[b + 1] - q[a + 1], uz = q[b + 2] - q[a + 2], vx = q[c] - q[a], vy = q[c + 1] - q[a + 1], vz = q[c + 2] - q[a + 2];
          const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
          for (const j of [a, b, c]) { n[j] += nx; n[j + 1] += ny; n[j + 2] += nz; }
        }
        for (let j = 0; j < n.length; j += 3) { const l = Math.hypot(n[j], n[j + 1], n[j + 2]) || 1; n[j] /= l; n[j + 1] /= l; n[j + 2] /= l; }
        const pb = gl.createBuffer(), nb = gl.createBuffer(), ib = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, q, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, nb); gl.bufferData(gl.ARRAY_BUFFER, n, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, f, gl.STATIC_DRAW);
        return { pb, nb, ib, cnt: f.length };
      });
      /* first view: the training camera that the turntable starts from, turned about the vertical axis */
      const e = d.eye, c = d.cen, dx = e[0] - c[0], dy = e[1] - c[1], dz = e[2] - c[2];
      cam.az0 = Math.atan2(dx, dz); cam.el0 = Math.atan2(dy, Math.hypot(dx, dz)); reset();
      G = { meshes, cen: c, box: d.box };
    }

    /* camera: orbit about the object's centre, +y up */
    const HF = Math.tan(0.3), R = 0.22;
    const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const mix = (a, b, c, x, y, z) => norm([a[0] * x + b[0] * y + c[0] * z, a[1] * x + b[1] * y + c[1] * z, a[2] * x + b[2] * y + c[2] * z]);
    function view(aspect) {
      const c = G.cen, ce = Math.cos(cam.el), se = Math.sin(cam.el);
      const dist = 1.08 * R / Math.sin(Math.atan(HF * Math.min(1, aspect))) * cam.zoom;
      const eye = [c[0] + dist * ce * Math.sin(cam.az), c[1] + dist * se, c[2] + dist * ce * Math.cos(cam.az)];
      const f = norm([c[0] - eye[0], c[1] - eye[1], c[2] - eye[2]]), s = norm(cross(f, [0, 1, 0])), u = cross(s, f);
      const near = dist * 0.05, far = dist * 4, fy = 1 / HF, fx = fy / aspect, a = (far + near) / (near - far), b = 2 * far * near / (near - far);
      const tx = -dot(s, eye), ty = -dot(u, eye), tz = dot(f, eye);
      const M = new Float32Array([fx * s[0], fy * u[0], -a * f[0], f[0], fx * s[1], fy * u[1], -a * f[1], f[1], fx * s[2], fy * u[2], -a * f[2], f[2], fx * tx, fy * ty, a * tz + b, -tz]);
      /* lights: a key from above left of the camera and a fill from its right */
      return { M, eye, l1: mix(f, u, s, -0.8, 0.55, -0.35), l2: mix(f, u, s, -0.4, -0.1, 0.9) };
    }

    /* drawing: each pane is the box of its label */
    let W = 0, H = 0, dirty = true, raf = 0, vis = false, turn = false, spin = !RM, last = 0;
    function panes() {
      const sr = stage.getBoundingClientRect(), k = W / Math.max(1, sr.width);
      return labs.map((l) => { const r = l.getBoundingClientRect(); return { x: Math.round((r.left - sr.left) * k), y: Math.round((r.top - sr.top) * k), w: Math.round(r.width * k), h: Math.round(r.height * k) }; });
    }
    function draw() {
      dirty = false;
      if (!G || !W) return;
      gl.disable(gl.SCISSOR_TEST); gl.clearColor(0.961, 0.961, 0.961, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.SCISSOR_TEST); gl.enable(gl.DEPTH_TEST); gl.useProgram(P); gl.uniform1f(U.B, G.box);
      panes().forEach((p, i) => {
        const y = H - p.y - p.h, m = G.meshes[i], V = view(p.w / Math.max(1, p.h));
        gl.viewport(p.x, y, p.w, p.h); gl.scissor(p.x, y, p.w, p.h); gl.clear(gl.DEPTH_BUFFER_BIT);
        gl.uniformMatrix4fv(U.M, false, V.M); gl.uniform3fv(U.E, V.eye); gl.uniform3fv(U.L1, V.l1); gl.uniform3fv(U.L2, V.l2);
        gl.bindBuffer(gl.ARRAY_BUFFER, m.pb); gl.enableVertexAttribArray(A.q); gl.vertexAttribPointer(A.q, 3, gl.UNSIGNED_SHORT, true, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, m.nb); gl.enableVertexAttribArray(A.n); gl.vertexAttribPointer(A.n, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.ib); gl.drawElements(gl.TRIANGLES, m.cnt, gl.UNSIGNED_SHORT, 0);
      });
    }
    const req = () => { if (!raf) raf = requestAnimationFrame(tick); };
    function tick(now) {
      raf = 0;
      if (dead || !vis || document.hidden) { last = 0; return; }
      /* until the reader takes over, the meshes turn once every 30 s while at least 40% is on screen */
      if (spin && turn && G) { if (last) { cam.az += Math.min(50, now - last) * 2 * Math.PI / 30000; dirty = true; } last = now; } else last = 0;
      if (dirty) draw();
      if (spin && turn) req();
    }
    function resize() {
      const dpr = Math.min(2, window.devicePixelRatio || 1), r = stage.getBoundingClientRect();
      const w = Math.max(2, Math.round(r.width * dpr)), h = Math.max(2, Math.round(r.height * dpr));
      if (w === W && h === H) return;
      W = w; H = h; cv.width = w; cv.height = h; dirty = true; req();
    }
    new ResizeObserver(resize).observe(stage);

    /* controls: drag or arrows turn, wheel or pinch zoom, double click resets */
    const ptrs = new Map();
    let pinch = 0;
    const clampZ = () => { cam.zoom = Math.max(0.35, Math.min(2.5, cam.zoom)); };
    const touched = () => { spin = false; };
    stage.addEventListener('pointerdown', (e) => {
      touched();
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, touch: e.pointerType !== 'mouse' });
      try { stage.setPointerCapture(e.pointerId); } catch (x) { /* ignore */ }
      if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch = Math.hypot(a.x - b.x, a.y - b.y); }
    });
    stage.addEventListener('pointermove', (e) => {
      const p = ptrs.get(e.pointerId); if (!p || !G) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (ptrs.size >= 2) {
        const [a, b] = [...ptrs.values()], dd = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch > 0 && dd > 0) { cam.zoom *= pinch / dd; clampZ(); }
        pinch = dd;
      } else {
        cam.az -= dx * 0.0085;
        if (!p.touch) cam.el = Math.max(-1.45, Math.min(1.45, cam.el + dy * 0.006));
      }
      dirty = true; req();
    });
    const end = (e) => { ptrs.delete(e.pointerId); pinch = 0; };
    stage.addEventListener('pointerup', end); stage.addEventListener('pointercancel', end);
    stage.addEventListener('wheel', (e) => {
      if (!G) return; e.preventDefault(); touched();
      cam.zoom *= Math.exp(Math.max(-100, Math.min(100, e.deltaY)) * 0.0015); clampZ(); dirty = true; req();
    }, { passive: false });
    stage.addEventListener('dblclick', () => { if (G) { reset(); dirty = true; req(); } });
    stage.addEventListener('keydown', (e) => {
      const k = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, -0.08], ArrowDown: [0, 0.08] }[e.key];
      if (!k || !G) return; e.preventDefault(); touched();
      cam.az += k[0]; cam.el = Math.max(-1.45, Math.min(1.45, cam.el + k[1])); dirty = true; req();
    });
    document.addEventListener('visibilitychange', () => { if (!document.hidden) { last = 0; dirty = true; req(); } });

    /* load when the block comes near; the still images stay if anything fails */
    let asked = false;
    new IntersectionObserver((es) => {
      if (!es[0].isIntersecting || asked) return;
      asked = true;
      load().then(() => { if (dead) return; cv.hidden = false; hint.hidden = false; resize(); dirty = true; req(); requestAnimationFrame(() => { pre.hidden = true; }); }).catch(() => {});
    }, { rootMargin: '600px 0px' }).observe(fig);
    new IntersectionObserver((es) => {
      const e = es[es.length - 1];
      vis = e.isIntersecting; turn = e.intersectionRatio >= 0.4;
      if (vis) { dirty = true; req(); }
    }, { threshold: [0, 0.4] }).observe(stage);
  })();
})();
