/*
 * lm-ix.js: shared kit for the companion site's interactive figures.
 *
 * Plain JavaScript, no dependencies, no network requests. Each figure is a raw-HTML
 * block whose root element carries class "lm-ix"; the figure's own script calls
 * LMIX.* helpers. Styling lives in theme.scss (.lm-ix); colors are read from the
 * --ix-* CSS variables so drawing code and stylesheet agree.
 *
 * Helpers:
 *   el(tag, attrs, parent)          create an SVG element
 *   colors(root)                    {blue, fs, red, cyan, orange, muted, grid, axis, rule, plane, bg}
 *   fmt(v, d)                       fixed decimals, with -0.00 shown as 0.00
 *   camera()                        orthographic 3-D projector with yaw/pitch, scale, center, offset
 *   rotator(svg, cam, redraw)       drag-to-rotate on an SVG
 *   slider(input, output, fmtFn, onInput)   bind a range input to its readout
 *   presets(container, list, apply) buttons from [{label, action}]
 *   stat(tile, value, isZero, fmtFn) set a stat tile, toggling the "condition met" highlight
 *   contours(f, xr, yr, levels, n)  marching-squares iso-lines of f on a grid: [{level, segs:[[x1,y1,x2,y2],...]}]
 *   arrow(svg, p, q, color, width)  2-D arrow from screen point p to q
 */
(function () {
  if (window.LMIX) return;
  const NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function colors(root) {
    const cs = getComputedStyle(root);
    const g = n => cs.getPropertyValue('--ix-' + n).trim();
    return { blue: g('blue'), fs: g('fs'), red: g('red'), cyan: g('cyan'), orange: g('orange'),
             muted: g('muted'), grid: g('grid'), axis: g('axis'), rule: g('rule'),
             plane: g('plane'), bg: g('bg'), ink: g('ink') };
  }

  function fmt(v, d) {
    d = d === undefined ? 2 : d;
    const s = (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : v).toFixed(d);
    return s === '-' + (0).toFixed(d) ? (0).toFixed(d) : s;
  }

  // Orthographic camera: rotate about the vertical axis (yaw), then tilt (pitch).
  function camera(opts) {
    const c = Object.assign({ yaw: 0, pitch: 0.4, cx: 260, cy: 220, scale: 30, offset: [0, 0, 0] }, opts || {});
    c.project = function (v) {
      const x = v[0] - c.offset[0], y = v[1] - c.offset[1], z = v[2] - c.offset[2];
      const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw), cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
      const x1 = cy * x - sy * y, y1 = sy * x + cy * y;
      return [c.cx + c.scale * x1, c.cy - c.scale * (z * cp - y1 * sp), y1 * cp + z * sp];
    };
    return c;
  }

  function rotator(svg, cam, redraw) {
    let drag = null;
    svg.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, yaw: cam.yaw, pitch: cam.pitch }; svg.setPointerCapture(e.pointerId); });
    svg.addEventListener('pointermove', e => {
      if (!drag) return;
      cam.yaw = drag.yaw + (e.clientX - drag.x) * 0.01;
      cam.pitch = Math.max(-1.2, Math.min(1.4, drag.pitch + (e.clientY - drag.y) * 0.01));
      redraw();
    });
    svg.addEventListener('pointerup', () => { drag = null; });
  }

  function slider(input, output, fmtFn, onInput) {
    const f = fmtFn || (v => fmt(v));
    const go = () => { output.textContent = f(+input.value); onInput(+input.value); };
    input.addEventListener('input', go);
    return { set(v) { input.value = v; go(); }, get() { return +input.value; }, refresh: go };
  }

  function presets(container, list, apply) {
    list.forEach(p => {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = p.label;
      b.addEventListener('click', () => (apply || (a => a()))(p.action));
      container.appendChild(b);
    });
  }

  function stat(tile, value, isZero, fmtFn) {
    tile.querySelector('.v').textContent = (fmtFn || fmt)(value);
    tile.classList.toggle('zero', !!isZero);
  }

  // Marching squares on an n x n grid over xr x yr; returns segments per level.
  function contours(f, xr, yr, levels, n) {
    n = n || 60;
    const xs = [], ys = [], v = [];
    for (let i = 0; i <= n; i++) { xs.push(xr[0] + (xr[1] - xr[0]) * i / n); ys.push(yr[0] + (yr[1] - yr[0]) * i / n); }
    for (let j = 0; j <= n; j++) { v.push([]); for (let i = 0; i <= n; i++) v[j].push(f(xs[i], ys[j])); }
    const lerp = (a, b, va, vb, L) => a + (b - a) * (L - va) / (vb - va || 1e-12);
    return levels.map(L => {
      const segs = [];
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const a = v[j][i], b = v[j][i + 1], c = v[j + 1][i + 1], d = v[j + 1][i];
        const pts = [];
        if ((a < L) !== (b < L)) pts.push([lerp(xs[i], xs[i + 1], a, b, L), ys[j]]);
        if ((b < L) !== (c < L)) pts.push([xs[i + 1], lerp(ys[j], ys[j + 1], b, c, L)]);
        if ((d < L) !== (c < L)) pts.push([lerp(xs[i], xs[i + 1], d, c, L), ys[j + 1]]);
        if ((a < L) !== (d < L)) pts.push([xs[i], lerp(ys[j], ys[j + 1], a, d, L)]);
        if (pts.length >= 2) segs.push([pts[0][0], pts[0][1], pts[1][0], pts[1][1]]);
        if (pts.length === 4) segs.push([pts[2][0], pts[2][1], pts[3][0], pts[3][1]]);
      }
      return { level: L, segs };
    });
  }

  function arrow(svg, p, q, color, width) {
    el('line', { x1: p[0], y1: p[1], x2: q[0], y2: q[1], stroke: color, 'stroke-width': width || 2 }, svg);
    const dx = q[0] - p[0], dy = q[1] - p[1], L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L, s = 8;
    el('polygon', { points: `${q[0]},${q[1]} ${q[0] - s * ux - s * 0.5 * uy},${q[1] - s * uy + s * 0.5 * ux} ${q[0] - s * ux + s * 0.5 * uy},${q[1] - s * uy - s * 0.5 * ux}`, fill: color }, svg);
  }

  window.LMIX = { el, colors, fmt, camera, rotator, slider, presets, stat, contours, arrow };
})();
