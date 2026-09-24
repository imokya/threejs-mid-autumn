import * as THREE from 'three/webgpu';
import {
  Fn, uniform, time, uv, vec2, vec3, vec4, float, color, mix, smoothstep, positionLocal, positionWorld,
  cameraPosition, normalize, dot, pow, max, sin, cos, fract, floor, hash, length, step, exp, instanceIndex,
  mx_noise_float, mx_fractal_noise_float, mx_worley_noise_float, reflector, reflect, pass, screenUV, texture, normalWorld,
  getViewPosition, select
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import moonMapUrl from '../assets/moon-lroc-2k.jpg';
import sushiUrl from '../assets/sushi.glb?url';

const $ = (s) => document.querySelector(s);
const loadMsg = (t) => { $('#loadmsg').textContent = t; };
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

/* =========================================================================
   画质档位
   ?quality=low|high  手动指定；缺省 auto（移动端与 4 核以下设备自动走低档）
   低档：像素比 1.0（下限 0.85）、无多重采样、896 阴影贴图、反射 0.24 每 3 帧
   高档：像素比 1.75（下限 1.0）、2x 多重采样、1280 阴影贴图、反射 0.32 每 2 帧
   运行时若帧率持续偏低，会自动下调像素比（高档下限 1.0，低档 0.85），恢复后升回。
   ?quality=low|high 之外还有几个调试开关：?stats ?lights=lean
   ========================================================================= */
const QS = new URLSearchParams(location.search);
const QUALITY = (() => {
  const q = QS.get('quality');
  if (q === 'low' || q === 'high') return q;
  const mobile = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
  return mobile || (navigator.hardwareConcurrency || 8) <= 4 ? 'low' : 'high';
})();
const LOW = QUALITY === 'low';
const SHOW_STATS = QS.has('stats');
const LEAN_LIGHTS = QS.get('lights') === 'lean';
const FOG_OFF = QS.get('fog') === 'off';          // A/B：关掉后期高度雾
const TUNE = {
  pixelRatioCap: LOW ? 1.0 : 1.75,        // 上限，实际还会按帧率自适应
  pixelRatioMin: LOW ? 0.85 : 1.0,        // 自适应下限
  samples: LOW ? 0 : 2,                   // 后期链路多重采样；WebGPU 下会提到 4（见下）
  shadowMapSize: LOW ? 896 : 1280,
  shadowEveryFrames: LOW ? 4 : 3,         // 月光固定，阴影无需逐帧重绘
  reflectionScale: LOW ? 0.24 : 0.32,     // 水面反射分辨率
  reflectionEveryFrames: LOW ? 3 : 2,     // 反射隔帧重算，中间帧复用上一张
  mistCount: LOW ? 20 : 34,               // 雾片数量
  cloudCount: LOW ? 6 : 9,                // 远景云数量
};
const T0 = performance.now();
const phases = [];
const phase = (name) => phases.push({ 阶段: name, 毫秒: Math.round(performance.now() - T0) });

/* =========================================================================
   基础工具：随机、噪声、画布纹理、几何合并
   ========================================================================= */
// 随机流：state()/set() 用于在「只做兜底」的耗时生成前后对齐序列，保证场景随机细节不变
function mulberry32(a) {
  const f = function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  f.state = () => a;
  f.set = (v) => { a = v | 0; };
  return f;
}
const rand = mulberry32(915);
const R = (a = 0, b = 1) => a + (b - a) * rand();
const lerp = (a, b, t) => a + (b - a) * t;
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const ease = (t) => t * t * t * (t * (t * 6 - 15) + 10);
const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);

const perm = new Uint8Array(512);
{ const p = [...Array(256).keys()]; for (let i = 255; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } for (let i = 0; i < 512; i++) perm[i] = p[i & 255]; }
function grad(h, x, y) { switch (h & 7) { case 0: return x + y; case 1: return -x + y; case 2: return x - y; case 3: return -x - y; case 4: return x; case 5: return -x; case 6: return y; default: return -y; } }
function noise2(x, y) {
  // 每个采样点原本要调 4 次 Math.floor（X/Y 各两次），这里复用取整结果，输出完全一致
  const fx = Math.floor(x), fy = Math.floor(y);
  const X = fx & 255, Y = fy & 255; x -= fx; y -= fy;
  const u = x * x * x * (x * (x * 6 - 15) + 10), v = y * y * y * (y * (y * 6 - 15) + 10);
  const a = perm[X] + Y, b = perm[X + 1] + Y;
  return lerp(lerp(grad(perm[a], x, y), grad(perm[b], x - 1, y), u), lerp(grad(perm[a + 1], x, y - 1), grad(perm[b + 1], x - 1, y - 1), u), v);
}
function fbm(x, y, o = 5) { let s = 0, a = 0.5, f = 1; for (let i = 0; i < o; i++) { s += a * noise2(x * f, y * f); f *= 2.03; a *= 0.5; } return s; }
// 可平铺噪声
function tileNoise(w, h, scale, oct = 4, seed = 0) {
  const F = (x, y) => fbm(x * scale / w + seed, y * scale / h + seed * 1.37, oct);
  return (x, y) => { const fx = x / w, fy = y / h; return F(x, y) * (1 - fx) * (1 - fy) + F(x - w, y) * fx * (1 - fy) + F(x, y - h) * (1 - fx) * fy + F(x - w, y - h) * fx * fy; };
}
function canvasTex(w, h, draw, { repeat = [1, 1], srgb = true } = {}) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const g = c.getContext('2d', { willReadFrequently: true }); draw(g, w, h);
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t;
}
function pixels(g, w, h, fn) {
  const img = g.getImageData(0, 0, w, h), d = img.data;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const c = fn(x, y); const i = (y * w + x) * 4; d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = c[3] ?? 255; }
  g.putImageData(img, 0, 0);
}

// 将同材质几何合并，减少绘制调用
class Merger {
  constructor() { this.map = new Map(); }
  add(geo, mat, p = [0, 0, 0], r = [0, 0, 0], s = [1, 1, 1]) {
    const m = new THREE.Matrix4().compose(V(...p), new THREE.Quaternion().setFromEuler(new THREE.Euler(...r)), V(...s));
    return this.addM(geo, mat, m);
  }
  addM(geo, mat, m) {
    let g = geo.index ? geo.toNonIndexed() : geo.clone();
    for (const k of Object.keys(g.attributes)) if (!['position', 'normal', 'uv'].includes(k)) g.deleteAttribute(k);
    if (!g.attributes.normal) g.computeVertexNormals();
    if (!g.attributes.uv) g.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(g.attributes.position.count * 2), 2));
    g.applyMatrix4(m);
    if (!this.map.has(mat)) this.map.set(mat, []);
    this.map.get(mat).push(g);
  }
  build({ cast = true, receive = true } = {}) {
    const grp = new THREE.Group();
    for (const [mat, list] of this.map) {
      const mesh = new THREE.Mesh(mergeGeometries(list, false), mat);
      mesh.castShadow = cast && !mat.userData.noShadow; mesh.receiveShadow = receive;
      grp.add(mesh);
    }
    return grp;
  }
}

// WebGL2 回退时 UBO 上限可能只有 16KB：实例数 >1000 则改走顶点属性，统一扩容
function Inst(geo, mat, count) { const m = new THREE.InstancedMesh(geo, mat, Math.max(count, 1001)); m.count = count; return m; }

/* =========================================================================
   材质纹理（全部程序生成）
   ========================================================================= */

// 程序月面（兜底：NASA 影像不可用时才真正生成）
function drawProceduralMoon(g, w, h) {
  const n = tileNoise(w, h, 7, 5, 21), n2 = tileNoise(w, h, 60, 3, 4);
  pixels(g, w, h, (x, y) => { const k = 0.82 + n(x, y) * 0.24 + n2(x, y) * 0.10; return [222 * k, 218 * k, 207 * k]; });
  const P = (lat, lon) => [(0.25 + lon / 360) * w, (0.5 - lat / 180) * h];
  const deg = w / 360;
  const mare = (lat, lon, r, a = 0.5, sx = 1, sy = 1) => {
    const [x, y] = P(lat, lon); const cs = 1 / Math.max(0.3, Math.cos(lat * Math.PI / 180));
    for (let i = 0; i < 70; i++) {
      const ox = R(-0.7, 0.7) * r * deg * sx * cs, oy = R(-0.7, 0.7) * r * deg * sy, rr = r * deg * R(0.35, 0.8);
      const gr = g.createRadialGradient(x + ox, y + oy, rr * 0.1, x + ox, y + oy, rr);
      gr.addColorStop(0, `rgba(118,120,126,${a * 0.31})`); gr.addColorStop(0.6, `rgba(122,124,130,${a * 0.19})`); gr.addColorStop(1, 'rgba(90,92,100,0)');
      g.fillStyle = gr; g.beginPath(); g.ellipse(x + ox, y + oy, rr * sx * cs, rr * sy, 0, 0, 7); g.fill();
    }
  };
  g.filter = 'blur(5px)';
  [[18, -57, 24, 0.5, 1.0, 1.4], [33, -16, 16], [28, 17, 9.5], [8.5, 31, 11.5], [17, 59, 7.5, 0.6], [-8, 51, 8.5], [-15, 35, 5.5], [-21, -17, 9.5],
   [-24, -39, 6], [56, 0, 5, 0.4, 5, 0.7], [13, 4, 4], [-10, -23, 6, 0.4], [7, -31, 7, 0.4], [-2, -12, 4, 0.35], [44, 30, 5, 0.35]].forEach((m) => mare(...m));
  g.filter = 'none';
  // 陨坑
  for (let i = 0; i < 1800; i++) {
    const x = R(0, w), y = R(h * 0.05, h * 0.95), r = Math.pow(R(), 3.5) * 12 + 1.0;
    g.fillStyle = `rgba(90,88,86,${R(0.05, 0.14)})`; g.beginPath(); g.arc(x + r * 0.15, y + r * 0.15, r, 0, 7); g.fill();
    g.strokeStyle = `rgba(255,252,242,${R(0.1, 0.22)})`; g.lineWidth = Math.max(0.6, r * 0.18); g.beginPath(); g.arc(x - r * 0.1, y - r * 0.1, r, 3.4, 5.9); g.stroke();
  }
  // 辐射纹亮坑：第谷、哥白尼、开普勒、阿利斯塔克
  const rays = (lat, lon, len, n, a) => {
    const [x, y] = P(lat, lon);
    for (let i = 0; i < n; i++) { const ang = R(0, 6.28), l = len * deg * R(0.4, 1); const gr = g.createLinearGradient(x, y, x + Math.cos(ang) * l, y + Math.sin(ang) * l); gr.addColorStop(0, `rgba(255,252,240,${a})`); gr.addColorStop(1, 'rgba(255,252,240,0)'); g.strokeStyle = gr; g.lineWidth = R(1.5, 4); g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(ang) * l, y + Math.sin(ang) * l); g.stroke(); }
    const gr = g.createRadialGradient(x, y, 0, x, y, 1.6 * deg); gr.addColorStop(0, 'rgba(255,255,248,.9)'); gr.addColorStop(1, 'rgba(255,255,248,0)'); g.fillStyle = gr; g.beginPath(); g.arc(x, y, 1.6 * deg, 0, 7); g.fill();
  };
  rays(-43, -11, 38, 46, 0.22); rays(10, -20, 16, 30, 0.2); rays(8, -38, 10, 20, 0.18); rays(24, -47, 6, 12, 0.2);
}

async function makeTextures() {
  const T = {};
  T.stone = canvasTex(512, 512, (g, w, h) => {
    const n = tileNoise(w, h, 5, 4, 1), n2 = tileNoise(w, h, 26, 3, 7);
    pixels(g, w, h, (x, y) => { const v = 112 + n(x, y) * 60 + n2(x, y) * 26; return [v * 0.9, v * 0.93, v]; });
    g.strokeStyle = 'rgba(18,20,26,.6)'; g.lineWidth = 3;
    for (let r = 0; r < 4; r++) { const y = r * 128; g.beginPath(); g.moveTo(0, y + 1); g.lineTo(w, y + 1); g.stroke();
      for (let k = 0; k < 4; k++) { const x = (k * 160 + (r % 2) * 80) % w; g.beginPath(); g.moveTo(x, y); g.lineTo(x, y + 128); g.stroke(); } }
  });
  T.lacquer = canvasTex(256, 512, (g, w, h) => {
    const n = tileNoise(w, h, 3, 4, 3), n2 = tileNoise(w, h, 40, 2, 9);
    pixels(g, w, h, (x, y) => { const gr = Math.sin(x * 0.19 + n(x, y) * 9) * 0.5 + 0.5; const k = 1 - gr * 0.18 + n2(x, y) * 0.12; return [128 * k, 34 * k, 26 * k]; });
  });
  T.wood = canvasTex(256, 512, (g, w, h) => {
    const n = tileNoise(w, h, 3, 4, 5);
    pixels(g, w, h, (x, y) => { const gr = Math.sin(x * 0.23 + n(x, y) * 10) * 0.5 + 0.5; const k = 0.8 + gr * 0.25; return [74 * k, 52 * k, 40 * k]; });
    g.fillStyle = 'rgba(20,12,8,.5)'; for (let x = 0; x < w; x += 64) g.fillRect(x, 0, 2, h);
  });
  T.tile = canvasTex(512, 512, (g, w, h) => {
    const n = tileNoise(w, h, 8, 3, 11);
    pixels(g, w, h, (x, y) => {
      const s = Math.abs(Math.cos(Math.PI * x / 32)); let k = 0.35 + 0.65 * Math.pow(s, 0.7);
      if (y % 42 < 5) k *= 0.55; k *= 1 + n(x, y) * 0.25;
      return [60 * k, 64 * k, 74 * k];
    });
  });
  T.lattice = canvasTex(256, 256, (g, w, h) => {
    g.clearRect(0, 0, w, h); g.strokeStyle = '#3a211a'; g.lineCap = 'square';
    g.lineWidth = 14; g.strokeRect(7, 7, w - 14, h - 14);
    g.lineWidth = 6;
    for (let i = 1; i < 4; i++) { g.beginPath(); g.moveTo(i * w / 4, 0); g.lineTo(i * w / 4, h); g.stroke(); g.beginPath(); g.moveTo(0, i * h / 4); g.lineTo(w, i * h / 4); g.stroke(); }
    g.lineWidth = 4;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { const x = i * w / 4, y = j * h / 4; g.strokeRect(x + 16, y + 16, w / 4 - 32, h / 4 - 32); g.beginPath(); g.moveTo(x + w / 8, y); g.lineTo(x + w / 8, y + 16); g.moveTo(x + w / 8, y + h / 4 - 16); g.lineTo(x + w / 8, y + h / 4); g.moveTo(x, y + h / 8); g.lineTo(x + 16, y + h / 8); g.moveTo(x + w / 4 - 16, y + h / 8); g.lineTo(x + w / 4, y + h / 8); g.stroke(); }
  });
  T.paper = canvasTex(256, 256, (g, w, h) => {
    const n = tileNoise(w, h, 10, 3, 13);
    pixels(g, w, h, (x, y) => { const v = y / h; const edge = Math.pow(Math.sin(Math.PI * v), 0.5); let k = 0.55 + 0.45 * edge + n(x, y) * 0.08; if (y % 32 < 3) k *= 0.72; return [255 * k, 205 * k, 140 * k]; });
    g.strokeStyle = 'rgba(120,30,20,.35)'; g.lineWidth = 3; g.beginPath(); g.moveTo(30, 200); g.bezierCurveTo(80, 150, 90, 110, 160, 70); g.moveTo(90, 140); g.quadraticCurveTo(120, 140, 140, 120); g.stroke();
    g.fillStyle = 'rgba(160,40,40,.4)'; for (let i = 0; i < 9; i++) { g.beginPath(); g.arc(60 + i * 12, 170 - i * 11 + (i % 2) * 8, 5, 0, 7); g.fill(); }
  });
  // 月面优先用 NASA 影像（加载见「启动」段）。程序月面只在影像失败时才真正生成，
  // 这里用小画布空跑同一段绘制：省掉约 65% 的纹理耗时，同时消耗等量随机数，
  // 使后续场景（林木、山石、灯影）的随机细节与优化前完全一致。
  await nextFrame();  // 分帧让出主线程，片头动效不卡死
  T.moon = canvasTex(256, 128, drawProceduralMoon);
  T.tileN = canvasTex(512, 512, (g, w, h) => {
    const H = (x, y) => { const s2 = Math.abs(Math.cos(Math.PI * x / 32)); return Math.sqrt(s2) * 0.9 + (((y % 42) + 42) % 42) / 42 * 0.55; };
    pixels(g, w, h, (x, y) => { const dx = (H(x + 1, y) - H(x - 1, y)) * 2.5, dy = (H(x, y + 1) - H(x, y - 1)) * 2.5; const l = Math.hypot(dx, dy, 1); return [(-dx / l * 0.5 + 0.5) * 255, (dy / l * 0.5 + 0.5) * 255, (1 / l * 0.5 + 0.5) * 255]; });
  }, { srgb: false });
  T.win = canvasTex(128, 128, (g, w, h) => {
    const n = tileNoise(w, h, 6, 3, 51);
    pixels(g, w, h, (x, y) => { const cx = x / w - 0.5, cy = y / h - 0.5; const v = Math.max(0.35, 1 - (cx * cx + cy * cy) * 1.6 + n(x, y) * 0.14); return [255 * v, 196 * v, 122 * v]; });
  });
  T.flower = canvasTex(128, 128, (g, w, h) => {
    g.clearRect(0, 0, w, h); g.translate(64, 64);
    for (let k = 0; k < 5; k++) {
      g.rotate(Math.PI * 2 / 5);
      const gr = g.createRadialGradient(0, -8, 2, 0, -26, 34); gr.addColorStop(0, '#f7c7d4'); gr.addColorStop(0.55, '#fff0f3'); gr.addColorStop(1, '#ffffff');
      g.fillStyle = gr; g.beginPath(); g.ellipse(0, -30, 21, 30, 0, 0, 7); g.fill();
    }
    g.fillStyle = '#c96078'; g.beginPath(); g.arc(0, 0, 9, 0, 7); g.fill();
    g.fillStyle = '#f6d67a'; for (let k = 0; k < 14; k++) { const a = k / 14 * 7; g.beginPath(); g.arc(Math.cos(a) * 14, Math.sin(a) * 14, 2.4, 0, 7); g.fill(); }
  }, { repeat: [1, 1] });
  T.petal = canvasTex(64, 64, (g) => {
    g.clearRect(0, 0, 64, 64); const gr = g.createRadialGradient(32, 50, 2, 32, 30, 34); gr.addColorStop(0, '#f2b3c4'); gr.addColorStop(1, '#fff5f7');
    g.fillStyle = gr; g.beginPath(); g.moveTo(32, 60); g.bezierCurveTo(2, 40, 8, 6, 26, 6); g.quadraticCurveTo(32, 12, 38, 6); g.bezierCurveTo(56, 6, 62, 40, 32, 60); g.fill();
  });
  T.needle = canvasTex(256, 256, (g, w, h) => {
    const n = tileNoise(w, h, 6, 4, 17);
    pixels(g, w, h, (x, y) => { const k = 0.75 + n(x, y) * 0.6; return [30 * k, 52 * k, 38 * k]; });
    g.lineWidth = 1.2; for (let i = 0; i < 2600; i++) { const x = R(0, w), y = R(0, h), a = R(0, 6.28), l = R(4, 10); g.strokeStyle = `rgba(${R(60, 110) | 0},${R(100, 150) | 0},${R(70, 100) | 0},.45)`; g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l); g.stroke(); }
  });
  await nextFrame();  // 分帧让出主线程，片头动效不卡死
  T.rock = canvasTex(512, 512, (g, w, h) => {
    const n = tileNoise(w, h, 4, 5, 23), n2 = tileNoise(w, h, 22, 3, 29);
    pixels(g, w, h, (x, y) => { const a = n(x, y), b = n2(x, y); const moss = clamp01((a - 0.12) * 3); let k = 0.85 + a * 0.5 + b * 0.25; return [lerp(96, 52, moss) * k, lerp(100, 70, moss) * k, lerp(108, 50, moss) * k]; });
    g.strokeStyle = 'rgba(15,16,20,.45)'; g.lineWidth = 1.6;
    for (let i = 0; i < 40; i++) { let x = R(0, w), y = R(0, h); g.beginPath(); g.moveTo(x, y); for (let k = 0; k < 6; k++) { x += R(-20, 20); y += R(4, 26); g.lineTo(x, y); } g.stroke(); }
  });
  await nextFrame();  // 分帧让出主线程，片头动效不卡死
  T.ground = canvasTex(512, 512, (g, w, h) => {
    const n = tileNoise(w, h, 6, 5, 31), n2 = tileNoise(w, h, 30, 3, 37);
    pixels(g, w, h, (x, y) => { const k = 0.8 + n(x, y) * 0.6 + n2(x, y) * 0.3; return [44 * k, 54 * k, 40 * k]; });
  });
  T.canopy = canvasTex(256, 256, (g, w, h) => {
    g.fillStyle = '#2a1c10'; g.fillRect(0, 0, w, h);
    const cell = 32;
    for (let i = 0; i < w / cell; i++) for (let j = 0; j < h / cell; j++) {
      const x = i * cell, y = j * cell, horiz = (i + j) % 2 === 0;
      for (let k = 0; k < 3; k++) {
        const o = 2 + k * 10;
        const gr = horiz ? g.createLinearGradient(0, y + o, 0, y + o + 8) : g.createLinearGradient(x + o, 0, x + o + 8, 0);
        const tone = R(0.85, 1.1);
        gr.addColorStop(0, `rgb(${92 * tone | 0},${68 * tone | 0},${40 * tone | 0})`); gr.addColorStop(0.5, `rgb(${168 * tone | 0},${132 * tone | 0},${84 * tone | 0})`); gr.addColorStop(1, `rgb(${80 * tone | 0},${58 * tone | 0},${34 * tone | 0})`);
        g.fillStyle = gr;
        if (horiz) g.fillRect(x + 1, y + o, cell - 2, 8); else g.fillRect(x + o, y + 1, 8, cell - 2);
      }
    }
  });
  T.willow = canvasTex(64, 512, (g, w, h) => {
    g.clearRect(0, 0, w, h);
    g.strokeStyle = 'rgba(78,74,40,1)'; g.lineWidth = 1.4; g.beginPath(); g.moveTo(32, 0); g.lineTo(32, h); g.stroke();
    for (let y = 3; y < h - 4; y += 4.2) for (const sd of [-1, 1]) {
      const len = R(12, 21) * (1 - y / h * 0.3);
      g.save(); g.translate(32 + sd * 0.8, y); g.rotate(-sd * R(0.35, 0.7));
      g.fillStyle = `hsl(${R(62, 92) | 0},${R(38, 58) | 0}%,${R(30, 50) | 0}%)`;
      g.beginPath(); g.ellipse(0, len / 2, R(1.8, 2.8), len / 2, 0, 0, 7); g.fill(); g.restore();
    }
  });
  T.cloth = canvasTex(256, 256, (g, w, h) => {
    const n = tileNoise(w, h, 12, 3, 41);
    pixels(g, w, h, (x, y) => { const k = 0.92 + n(x, y) * 0.1 + ((x + y) % 4 < 1 ? -0.03 : 0); return [226 * k, 224 * k, 214 * k]; });
    g.strokeStyle = 'rgba(120,130,150,.18)'; g.lineWidth = 1; for (let i = 0; i < 18; i++) { g.beginPath(); const x = R(0, w); g.moveTo(x, 0); g.bezierCurveTo(x + R(-30, 30), 80, x + R(-30, 30), 170, x + R(-20, 20), h); g.stroke(); }
  });
  T.plank = canvasTex(256, 256, (g, w, h) => {
    const n = tileNoise(w, h, 4, 4, 43);
    pixels(g, w, h, (x, y) => { const gr = Math.sin(y * 0.3 + n(x, y) * 8) * 0.5 + 0.5; let k = 0.75 + gr * 0.2 + n(x, y) * 0.2; if (x % 32 < 2) k *= 0.45; return [84 * k, 62 * k, 46 * k]; });
  });
  T.silk = canvasTex(128, 256, (g, w, h) => {
    const gr = g.createLinearGradient(0, 0, 0, h); gr.addColorStop(0, 'rgba(250,244,230,.95)'); gr.addColorStop(1, 'rgba(240,232,214,.75)');
    g.fillStyle = gr; g.fillRect(0, 0, w, h);
    g.strokeStyle = 'rgba(90,110,90,.35)'; g.lineWidth = 2;
    g.beginPath(); g.moveTo(40, h); g.bezierCurveTo(50, 180, 30, 120, 70, 60); g.moveTo(52, 150); g.quadraticCurveTo(80, 140, 96, 120); g.moveTo(45, 110); g.quadraticCurveTo(20, 96, 16, 80); g.stroke();
    g.fillStyle = 'rgba(90,120,95,.28)'; for (let i = 0; i < 16; i++) { g.save(); g.translate(R(20, 110), R(50, 200)); g.rotate(R(0, 6)); g.beginPath(); g.ellipse(0, 0, 3, 10, 0, 0, 7); g.fill(); g.restore(); }
  });
  await nextFrame();  // 分帧让出主线程，片头动效不卡死
  T.wall = canvasTex(256, 256, (g, w, h) => {
    const n = tileNoise(w, h, 5, 4, 47);
    pixels(g, w, h, (x, y) => { const k = 0.86 + n(x, y) * 0.25; return [214 * k, 206 * k, 190 * k]; });
  });
  return T;
}

/* =========================================================================
   启动
   ========================================================================= */
const canvas = $('#c');
loadMsg('启画 · 初始化图形');
// 场景走后期链路，抗锯齿由后期通道的多重采样负责，这里关掉画布自带的 MSAA 省一份显存与带宽
const renderer = new THREE.WebGPURenderer({ canvas, antialias: false, forceWebGL: !QS.has('webgpu') });
renderer.setPixelRatio(Math.min(devicePixelRatio, TUNE.pixelRatioCap));
renderer.setSize(innerWidth, innerHeight, false);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = LOW ? THREE.PCFShadowMap : THREE.PCFSoftShadowMap;
// 注意：r180 里阴影走 ShadowNode，刷新闸门在 LightShadow 上（见 moonLight.shadow），
// renderer.shadowMap.autoUpdate 已不被读取，直接设置它不会有任何效果。
await renderer.init();
$('#backend').textContent = renderer.backend.isWebGPUBackend ? 'WEBGPU' : 'WEBGL2 · 兼容模式';
// WebGPU 的 sampleCount 只接受 1 或 4（three 会把它直接写进纹理描述符），WebGL2 没有这个限制
if (TUNE.samples > 0 && renderer.backend.isWebGPUBackend) TUNE.samples = 4;
phase('渲染器就绪');

loadMsg('描山画水 · 生成纹理');
await nextFrame();
const TX = await makeTextures();
phase('纹理生成');
// NASA Scientific Visualization Studio · CGI Moon Kit / LROC color mosaic.
try {
  const lunar = await new THREE.TextureLoader().loadAsync(moonMapUrl);
  lunar.colorSpace = THREE.SRGBColorSpace;
  lunar.wrapS = THREE.RepeatWrapping;
  lunar.offset.x = 0.25;
  lunar.anisotropy = 8;
  TX.moon.dispose();
  TX.moon = lunar;
} catch (error) {
  // 影像不可用：这时才真正生成 2048x1024 的程序月面，生成后把随机流拨回原位，
  // 避免多消耗的一万多注随机数打乱后续场景的随机细节
  console.warn('月面影像未加载，使用程序月面', error);
  const seedAtSceneStart = rand.state();
  TX.moon.dispose();
  TX.moon = canvasTex(2048, 1024, drawProceduralMoon);
  rand.set(seedAtSceneStart);
}
phase('月面影像');


const FOG_COL = 0x152b36;
const MOON_DIR = V(0.13, 0.168, -1).normalize();
const MOON_POS = MOON_DIR.clone().multiplyScalar(1500);

const scene = new THREE.Scene();
scene.background = new THREE.Color(FOG_COL);
scene.fog = new THREE.FogExp2(FOG_COL, 0.00095);
// 近裁面 0.35（原 0.1）：远/近比从 7 万降到 2 万，远处岸线与水面、山体交叠处的深度精度提高约 3.5 倍，
// 消除镜头移动时远景的 z-fighting 闪烁。全片最近的镜位离物体也在半米以上，不会被裁切。
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.35, 7000);
camera.position.set(0, 7, 72);

/* ---------- 共享 uniform ---------- */
const uMoonDir = uniform(MOON_DIR.clone());
const uBoat = uniform(V());
const uCold = uniform(0);
const uSea = uniform(0);
const uPassOpacity = uniform(0.9);

/* ---------- 材质库 ---------- */
const STD = (o) => new THREE.MeshStandardNodeMaterial({ roughness: 0.85, metalness: 0, ...o });
const texR = (t, rx, ry = rx) => { const c = t.clone(); c.repeat.set(rx, ry); c.needsUpdate = true; return c; };
const M = {
  stone: STD({ map: texR(TX.stone, 0.25), roughness: 0.95 }),
  stoneStep: STD({ map: texR(TX.stone, 0.35), roughness: 0.9, color: 0xc9ccd4 }),
  lacquer: new THREE.MeshPhysicalNodeMaterial({ map: TX.lacquer, roughness: 0.45, clearcoat: 0.7, clearcoatRoughness: 0.2 }),
  wood: STD({ map: TX.wood, roughness: 0.75 }),
  plank: STD({ map: texR(TX.plank, 0.5), roughness: 0.8 }),
  tile: STD({ map: TX.tile, normalMap: TX.tileN, normalScale: new THREE.Vector2(1.3, 1.3), roughness: 0.55, metalness: 0.15 }),
  copper: STD({ map: TX.tile, normalMap: TX.tileN, normalScale: new THREE.Vector2(1.3, 1.3), color: 0xc49a66, roughness: 0.42, metalness: 0.6 }),
  bracket: STD({ color: 0x2e5a55, roughness: 0.6 }),
  pagodaWall: new THREE.MeshPhysicalNodeMaterial({ map: TX.lacquer, color: 0xb07060, roughness: 0.5, clearcoat: 0.4 }),
  under: STD({ map: TX.wood, roughness: 0.9, color: 0x7a5646, side: THREE.BackSide }),
  ridge: STD({ color: 0x2b2f38, roughness: 0.6 }),
  lattice: STD({ map: TX.lattice, alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.7 }),
  rock: STD({ map: texR(TX.rock, 0.5), roughness: 0.95 }),
  cliff: STD({ map: texR(TX.rock, 3, 2), roughness: 0.95 }),
  ground: STD({ map: texR(TX.ground, 0.05), roughness: 1 }),
  bark: STD({ map: texR(TX.wood, 1, 3), color: 0x6b5a50, roughness: 0.95 }),
  needle: STD({ map: TX.needle, roughness: 1, color: 0xbfd0c4 }),
  wall: STD({ map: TX.wall, roughness: 0.95 }),
  gold: STD({ color: 0xb08a4a, roughness: 0.35, metalness: 0.8 }),
  canopy: STD({ map: texR(TX.canopy, 5, 3), roughness: 0.8, side: THREE.DoubleSide }),
  hull: STD({ map: texR(TX.plank, 1, 3), roughness: 0.75, side: THREE.DoubleSide }),
  bridge: STD({ map: texR(TX.stone, 0.08), roughness: 0.95, color: 0xb8bcc6 }),
  foliage: STD({ map: TX.needle, color: 0x7f988a, roughness: 1, flatShading: true }),
  // 远景林木用：少 3/4 的面，改用平滑着色——低面数配 flatShading 会结晶体一样炸开
  foliageFar: STD({ map: TX.needle, color: 0x7f988a, roughness: 1 }),
};
const glowMat = (r, g, b) => { const m = new THREE.MeshBasicNodeMaterial(); m.colorNode = vec3(r, g, b); m.userData.noShadow = true; return m; };
M.window = new THREE.MeshBasicNodeMaterial(); M.window.colorNode = texture(TX.win).rgb.mul(3.3); M.window.userData.noShadow = true;
M.windowDim = new THREE.MeshBasicNodeMaterial(); M.windowDim.colorNode = texture(TX.win).rgb.mul(1.35); M.windowDim.userData.noShadow = true;
M.eaveLight = glowMat(1.7, 0.95, 0.32);
M.lamp = glowMat(4.5, 2.3, 0.8);
M.paper = new THREE.MeshBasicNodeMaterial(); M.paper.colorNode = texture(TX.paper).rgb.mul(vec3(2.9, 1.9, 1.05)); M.paper.userData.noShadow = true;
M.redCap = STD({ color: 0x6e1e18, roughness: 0.5 });

/* ---------- 建筑提亮 ----------
   夜里月光很弱，瓦面、漆柱、白墙在镜头里常常只剩一圈轮廓。这里只抬「建筑」类材质：
     gain  基色增益：把反照率整体上抬。瓦、木原本压得很暗，靠它才看得出材质。
     glow  自发光底：取材质自身反照率做一层底光，把暗部托起来。不吃灯光、不产生额外开销。
   山水草木（hill0/1/2、foliage、needle、bark）、水面、天空、乌篷船一律不动，夜色不变。
   想更亮或更暗，只改下面这张表里的数字即可。gain=1、glow=0 就等于关掉。 */
const ARCH = [
  // [材质,          基色增益, 自发光底, 底色（无贴图者）]
  [M.tile,          1.62, 0.090, 0],        // 筒瓦屋面
  [M.copper,        1.55, 0.085, 0],        // 铜脊、铜瓦
  [M.ridge,         1.60, 0.075, 0x2b2f38], // 正脊、檐口
  [M.bracket,       1.70, 0.075, 0x2e5a55], // 斗拱
  [M.under,         1.55, 0.080, 0],        // 檐下（背面木）
  [M.lacquer,       1.26, 0.105, 0],        // 朱漆柱、栏杆
  [M.pagodaWall,    1.35, 0.100, 0],        // 塔身
  [M.wood,          1.30, 0.080, 0],        // 木构、梁枋
  [M.plank,         1.32, 0.080, 0],        // 板壁、铺板
  [M.wall,          1.22, 0.105, 0],        // 白墙
  [M.stoneStep,     1.20, 0.055, 0],        // 踏步
  [M.stone,         1.30, 0.050, 0],        // 铺地、台基
  [M.bridge,        1.18, 0.048, 0],        // 石桥
  [M.lattice,       1.40, 0.080, 0],        // 门窗棂
  [M.canopy,        1.32, 0.065, 0],        // 席帘、蒲团
  [M.gold,          1.18, 0.055, 0xb08a4a], // 金饰
  [M.redCap,        1.40, 0.055, 0x6e1e18], // 灯罩红顶
];
// 第四轮：整体再抬一档（基色 ×1.15、底光 ×1.4），只作用于上表的建筑材质
const ARCH_BOOST = { gain: 1.15, glow: 1.4 };
for (const [m, gain, glow, base] of ARCH) {
  m.color.multiplyScalar(gain * ARCH_BOOST.gain);
  if (glow > 0) m.emissiveNode = (m.map ? texture(m.map).rgb : color(base)).mul(glow * ARCH_BOOST.glow);
}
M.paperRed = new THREE.MeshBasicNodeMaterial(); M.paperRed.colorNode = texture(TX.paper).rgb.mul(vec3(3.4, 0.62, 0.3)); M.paperRed.userData.noShadow = true;
function terrainMat(base, rock, hi) {
  const m = new THREE.MeshStandardNodeMaterial({ roughness: 1 });
  const slope = float(1).sub(normalWorld.y).clamp(0.0, 1.0);
  const n = mx_fractal_noise_float(positionWorld.mul(0.035), 3, 2.0, 0.5).mul(0.5).add(0.5);
  const n2 = mx_noise_float(positionWorld.mul(0.4)).mul(0.5).add(0.5);
  const c = mix(color(base), color(rock), smoothstep(0.28, 0.6, slope.add(n.sub(0.5).mul(0.35))));
  m.colorNode = mix(c, color(hi), smoothstep(0.55, 0.95, n).mul(0.4)).mul(n2.mul(0.35).add(0.8));
  return m;
}
M.hill0 = terrainMat(0x1c2b22, 0x3b4048, 0x2e3d31);
M.hill1 = terrainMat(0x1b2433, 0x2e3542, 0x263246);
M.hill2 = terrainMat(0x1d2740, 0x2a3148, 0x28324a);
M.willow = new THREE.MeshStandardNodeMaterial({ map: TX.willow, color: 0x7d8466, alphaTest: 0.35, side: THREE.DoubleSide, roughness: 0.9 });
M.willow.emissiveNode = texture(TX.willow).rgb.mul(0.05);
{
  const k = positionLocal.y.negate().clamp(0.0, 1.0), sw = k.mul(k), ph = hash(instanceIndex).mul(6.283);
  M.willow.positionNode = positionLocal.add(vec3(sin(time.mul(0.9).add(ph).add(k.mul(1.8))).mul(0.2).mul(sw), 0, cos(time.mul(0.7).add(ph.mul(1.3))).mul(0.14).mul(sw)));
}

/* =========================================================================
   天空 · 明月 · 星辰 · 云
   ========================================================================= */
/* 天穹云层：把视线投到一张高空平面上取噪声 —— 近处云大、近地平线处压扁变密，透视自然。
   形体 = 域扭曲 fbm（层云的絮状）+ 反 Worley（积云的团块）；
   光照 = 朝月亮方向偏移再取一次密度，两者之差即「迎月面」：迎月的云边发亮，背月的云腹偏暗；
   薄处靠近月亮时透出银边与月晕。月亮本身也读同一层云，云会真实地飘过月面。 */
const skyClouds = Fn(([d]) => {
  const h = d.y;
  const cp = d.xz.div(max(h, 0.04)).mul(0.8);
  const q0 = cp.add(vec2(time.mul(0.006), time.mul(0.0022)));
  const warp = mx_noise_float(vec3(q0.mul(0.3), time.mul(0.01)));
  const q = q0.add(vec2(warp, warp.mul(0.6)).mul(0.5));
  const puff = float(1).sub(mx_worley_noise_float(vec3(q.mul(1.6), time.mul(0.01))));
  const n = mx_fractal_noise_float(vec3(q.mul(0.85), time.mul(0.006)), 4, 2.1, 0.5).mul(0.45).add(0.5).mul(0.75).add(puff.mul(0.35));
  const mP = uMoonDir.xz.div(max(uMoonDir.y, 0.05)).mul(0.8);
  const toward = normalize(mP.sub(cp));
  const n2 = mx_fractal_noise_float(vec3(q.add(toward.mul(0.07)).mul(0.85), time.mul(0.006)), 3, 2.1, 0.5).mul(0.45).add(0.5).mul(0.75).add(puff.mul(0.35));
  const md = max(dot(d, uMoonDir), 0.0);
  // 少量、疏朗：阈值抬高只留零散云团；月亮周围约 10°~20° 留出一片晴空，云绝不遮月
  const cover = smoothstep(0.64, 0.88, n).mul(smoothstep(0.03, 0.2, h)).mul(float(1).sub(smoothstep(0.94, 0.985, md)));
  const light = n.sub(n2).mul(5.0).add(0.45).clamp(0.0, 1.0);
  const glow = pow(md, 12.0).mul(0.3).add(pow(md, 90.0).mul(0.4)).mul(float(1).sub(cover.mul(0.55)));
  const col = mix(color(0x0a1120), color(0x56668c), light).add(color(0xc8d2ee).mul(glow));
  return vec4(col, cover.mul(0.93));
});
const sky = new THREE.Mesh(new THREE.SphereGeometry(4000, 48, 24), new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide, depthWrite: false, fog: false }));
sky.material.colorNode = Fn(() => {
  const d = normalize(positionLocal).toVar();
  const h = d.y;
  const c = mix(color(FOG_COL), color(0x131d36), smoothstep(0.0, 0.16, h)).toVar();
  c.assign(mix(c, color(0x03060d), smoothstep(0.12, 0.95, h)));
  const md = max(dot(d, uMoonDir), 0.0);
  // 月晕收窄减弱：只留一圈很淡的外晕，月轮边缘清清楚楚
  c.addAssign(color(0xb8c8ff).mul(pow(md, 10).mul(0.07)));
  c.addAssign(color(0xffe0b0).mul(pow(md, 90).mul(0.06).add(pow(md, 900).mul(0.08))));
  c.addAssign(color(0x3b2a2c).mul(float(1).sub(smoothstep(0.0, 0.14, h)).mul(0.45)));
  const sp = d.mul(240.0);
  const cell = floor(sp);
  const f = fract(sp).sub(0.5);
  const hs = hash(cell.x.add(cell.y.mul(157.31)).add(cell.z.mul(113.97)).add(1000.0));
  const tw = sin(time.mul(hs.mul(3.0).add(0.6)).add(hs.mul(80.0))).mul(0.4).add(0.6);
  const star = step(0.991, hs).mul(float(1).sub(smoothstep(0.0, 0.32, length(f)))).mul(tw).mul(smoothstep(0.04, 0.3, h)).mul(float(1).sub(pow(md, 6)));
  c.addAssign(vec3(star.mul(1.6)));
  const cl = skyClouds(d);
  c.assign(mix(c, cl.rgb, cl.a));
  return c;
})();
sky.renderOrder = -10;
scene.add(sky);

// 月亮
const moon = new THREE.Mesh(new THREE.SphereGeometry(92, 128, 96), new THREE.MeshBasicNodeMaterial({ fog: false }));
{
  const vd = normalize(cameraPosition.sub(positionWorld));
  const limb = pow(max(dot(normalWorld, vd), 0.0), 0.35);
  moon.material.colorNode = texture(TX.moon).rgb.mul(limb.mul(0.22).add(0.78)).mul(vec3(1.04, 1.01, 0.95));   // 略压亮度，少吃辉光，月面细节更清
}
moon.position.copy(MOON_POS);
moon.lookAt(0, 0, 0);
scene.add(moon);

// 云
// 云：oct 为程序噪声的阶数。fbm 的每一阶振幅是上一阶的一半，
// 最高阶对总方差的贡献在千分位（5 阶→3 阶只让噪声标准差变动 0.7%），
// 却要占掉这个着色器四成以上的运算量，故默认只留 3 阶。
// 过月之云是单一主角，单独保留 5 阶。
function cloudMaterial(seed, opacityU, scale = [3, 1.4], oct = 3) {
  const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, fog: false, side: THREE.DoubleSide });
  const u = uv();
  const p0 = vec3(u.x.mul(scale[0]).add(seed).add(time.mul(0.006)), u.y.mul(scale[1]), time.mul(0.012).add(seed));
  // 域扭曲：一阶低频噪声把采样点推歪，云边出现卷曲、拉丝，而不是均匀的棉絮
  const warp = mx_noise_float(p0.mul(0.55).add(3.7));
  const p = p0.add(vec3(warp.mul(0.45), warp.mul(0.25), 0));
  const n = mx_fractal_noise_float(p, oct, 2.0, 0.5).mul(0.6).add(0.5);
  const er = length(u.sub(0.5).mul(vec2(2.0, 2.6))).add(n.sub(0.5).mul(0.55));
  const edge = float(1).sub(smoothstep(0.5, 1.0, er));
  const dens = smoothstep(0.42, 0.85, n);
  const a = dens.mul(edge).mul(opacityU);
  const vd = normalize(positionWorld.sub(cameraPosition));
  const md = max(dot(vd, uMoonDir), 0.0);
  const lit = pow(md, 24.0);
  // 夜云逆光：厚处吃掉月光偏暗，薄边透光发亮（银边），离月越近越明显
  const thin = float(1).sub(dens);
  const rim = thin.mul(pow(md, 10.0).mul(1.6).add(lit.mul(2.4)));
  const body = mix(color(0x3a4766), color(0x121a2c), dens.mul(0.85));
  m.colorNode = body.add(color(0x9aabd2).mul(lit.mul(0.5))).add(vec3(rim.mul(0.9), rim.mul(0.95), rim));
  m.opacityNode = a;
  return m;
}
const clouds = [];
function dirFrom(az, el) { const b = MOON_DIR.clone(); const yaw = Math.atan2(b.x, -b.z) + az; const e = Math.asin(b.y) + el; return V(Math.sin(yaw) * Math.cos(e), Math.sin(e), -Math.cos(yaw) * Math.cos(e)); }
for (let i = 0; i < 9; i++) {
  const az = R(-0.9, 0.9), el = R(-0.16, 0.22);
  if (Math.abs(az) < 0.3 && Math.abs(el) < 0.22) continue;   // 月亮周围不放云
  const w = R(500, 1100), h = w * R(0.28, 0.4);
  // 随机数与造面片分开：低档减少云量时，随机流照常推进，场景其余部分的细节一个都不变
  const seed = R(0, 50), opacity = R(0.55, 0.9), dist = R(1150, 1350);
  if (clouds.length >= TUNE.cloudCount) continue;
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), cloudMaterial(seed, uniform(opacity)));
  m.position.copy(dirFrom(az, el).multiplyScalar(dist));
  m.lookAt(0, 0, 0); m.renderOrder = -5;
  scene.add(m); clouds.push(m);
}
// 过月之云（月有阴晴）：全场唯一的主角云，噪声阶数不动
const passCloud = new THREE.Mesh(new THREE.PlaneGeometry(520, 190), cloudMaterial(7.3, uPassOpacity, [2.2, 1.1], 5));
passCloud.renderOrder = -4; scene.add(passCloud);
// 从月亮下方掠过（云顶低于月轮下缘），只在月下拖出一缕，不遮月面
function placePassCloud(p) { passCloud.position.copy(dirFrom(p * 0.34, -0.17).multiplyScalar(1200)); passCloud.lookAt(0, 0, 0); }
placePassCloud(2);

// 云海（高处不胜寒）
const sea = new THREE.Mesh(new THREE.PlaneGeometry(1800, 1800), new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, fog: false }));
{
  const wp = positionWorld.xz.mul(0.006);
  const n = mx_fractal_noise_float(vec3(wp.add(vec2(time.mul(0.01), 0)), time.mul(0.02)), 3, 2.0, 0.5).mul(0.6).add(0.5);
  const d = length(uv().sub(0.5)).mul(2.0);
  const vd = normalize(positionWorld.sub(cameraPosition));
  const lit = pow(max(dot(vd, uMoonDir), 0.0), 6.0);
  sea.material.colorNode = mix(color(0x3a4a6e), color(0xb7c6e8), n.mul(0.5).add(lit.mul(0.6)));
  sea.material.opacityNode = smoothstep(0.35, 0.8, n).mul(float(1).sub(smoothstep(0.6, 1.0, d))).mul(uSea).mul(0.92);
}
sea.rotation.x = -Math.PI / 2; sea.position.set(100, 58, -240); sea.visible = false;
scene.add(sea);

/* =========================================================================
   灯光
   ========================================================================= */
const moonLight = new THREE.DirectionalLight(0xcad0ea, 1.45);
moonLight.position.copy(MOON_DIR).multiplyScalar(160).add(V(-12, 0, 16));
moonLight.target.position.set(-12, 0, 16);
moonLight.castShadow = true;
moonLight.shadow.mapSize.set(TUNE.shadowMapSize, TUNE.shadowMapSize);
Object.assign(moonLight.shadow.camera, { left: -70, right: 70, top: 70, bottom: -70, near: 10, far: 400 });
moonLight.shadow.bias = -0.0004; moonLight.shadow.normalBias = 0.04;
moonLight.shadow.autoUpdate = false;   // 月位固定，阴影改由主循环按帧隔刷新
scene.add(moonLight, moonLight.target);
scene.add(new THREE.HemisphereLight(0x55618e, 0x3a2616, 1.3));
const fill = new THREE.DirectionalLight(0x6d82b8, 0.35); fill.position.set(60, 40, 160); scene.add(fill);

/* =========================================================================
   飞鸟：三群掠过月与水面
   ========================================================================= */
// 一具极简的鸟：锥形身体 + 两片后掠翼，局部 +Z 为前，单翼展 0.78（再乘实例缩放）。
// 身体与左右翼各用一个实例网格：扇翅是绕肩（翼几何的原点）转，CPU 每帧写 27 个矩阵，
// 比塞进顶点着色器更稳——实例网格里 positionNode 拿到的是「已乘实例矩阵」的坐标。
const birdBody = new THREE.ConeGeometry(0.055, 0.36, 5).rotateX(Math.PI / 2);
const birdWing = (sx) => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([0, 0.004, 0.13, sx * 0.78, 0.02, -0.03, 0, 0.004, -0.15], 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 1, 0, 1], 2));
  g.computeVertexNormals();
  return g;
};
const birdMat = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide, fog: false });
birdMat.colorNode = color(0x0a0f1c);   // 逆光剪影：比天光暗一档，过月时是干净的黑色
const FLOCKS = [
  // [只数, 圈心 x/z, 半径 rx/rz, 高度, 周期秒, 起始相位, 体型, 环绕方向]
  // 第一群的椭圆正好穿过「月面视线」上的一点，所以每圈都会有一次横越月亮
  [4, [76.4, -249.4], [44, 30], 41.9, 96, 0.00, 1.45, 1],
  [3, [-14, -132], [46, 32], 54, 74, 0.50, 1.50, -1],
  [2, [52, -196], [34, 26], 62, 84, 1.60, 1.55, 1],
];
const birds = { list: [] };
for (const [n, c, r, y, per, ph0, s, dir] of FLOCKS) {
  for (let i = 0; i < n; i++) birds.list.push({
    c, r, dir, y: y + i * 1.5, w: (Math.PI * 2 / per) * dir, ph: ph0 + i * 0.22 * dir,
    s: s * (1 + i * 0.06), bob: 0.9 + i * 0.35, flap: 13.0 + (i % 3) * 1.6, fp: i * 1.9,
  });
}
birds.body = new THREE.InstancedMesh(birdBody, birdMat, birds.list.length);
birds.wingR = new THREE.InstancedMesh(birdWing(1), birdMat, birds.list.length);
birds.wingL = new THREE.InstancedMesh(birdWing(-1), birdMat, birds.list.length);
for (const m of [birds.body, birds.wingR, birds.wingL]) { m.frustumCulled = false; scene.add(m); }
const _bd = new THREE.Object3D(), _bw = new THREE.Object3D(), _bwR = new THREE.Matrix4(), _bwS = new THREE.Matrix4();
function updateBirds(t) {
  const n = birds.list.length;
  for (let i = 0; i < n; i++) {
    const b = birds.list[i], a = b.ph + t * b.w;
    _bd.position.set(b.c[0] + Math.cos(a) * b.r[0], b.y + Math.sin(a * 1.7 + b.ph) * b.bob, b.c[1] + Math.sin(a) * b.r[1]);
    // 航向取椭圆切线；dir 为 -1 时整只鸟掉头飞
    _bd.rotation.set(0, Math.atan2(-Math.sin(a) * b.r[0], Math.cos(a) * b.r[1]) + (b.dir > 0 ? 0 : Math.PI), 0);
    _bd.rotateZ(0.34 * b.dir);   // 转弯侧倾
    _bd.scale.setScalar(b.s);
    _bd.updateMatrix();
    birds.body.setMatrixAt(i, _bd.matrix);
    // 双翼绕肩对称摆动：翼尖上抬约 ±0.22 个身位，起落之间略微收拢
    const f = Math.sin(t * b.flap + b.fp) * 0.52;
    _bw.matrix.copy(_bd.matrix);
    _bw.matrix.multiply(_bwR.makeRotationZ(f));
    _bw.matrix.multiply(_bwS.makeScale(1 - Math.abs(f) * 0.16, 1, 1));
    birds.wingR.setMatrixAt(i, _bw.matrix);
    _bw.matrix.copy(_bd.matrix);
    _bw.matrix.multiply(_bwR.makeRotationZ(-f));
    _bw.matrix.multiply(_bwS.makeScale(1 - Math.abs(f) * 0.16, 1, 1));
    birds.wingL.setMatrixAt(i, _bw.matrix);
  }
  birds.body.instanceMatrix.needsUpdate = birds.wingR.instanceMatrix.needsUpdate = birds.wingL.instanceMatrix.needsUpdate = true;
}

/* =========================================================================
   水面（TSL 反射 + 噪声扰动 + 舟行涟漪）
   ========================================================================= */
{
  const reflection = reflector({ resolutionScale: TUNE.reflectionScale, bounces: false });
  reflection.target.rotateX(-Math.PI / 2);
  scene.add(reflection.target);
  // 反射本质是「把整个场景换个机位再画一遍」，是全场最贵的一笔。
  // 水面被三层噪声扭曲、再加菲涅尔混色，反射贴图隔帧复用肉眼看不出差别，
  // 却直接砍掉约一半的反射开销。
  // 节拍由主循环的 frameNo 驱动（每动画帧 +1），而不是数 updateBefore 被调用的次数：
  // 后者一帧内可能被多个绘制对象触发，次数不稳定。
  // 另外这里必须返回 undefined 而非 false —— 返回 false 会让上层认为「没执行」，
  // 本帧下一个绘制对象还会再调一次，等于白省。
  {
    const rb = reflection.reflector;
    const updateOnce = rb.updateBefore.bind(rb);
    let lastTick = -1;
    rb.updateBefore = (frameData) => {
      // 镜头在动时复用旧反射，会让灯影、月路在新旧两帧间跳动（闪烁）；画布刚改尺寸时旧反射贴图已被清空（闪一帧黑）。
      // 所以只在镜头近乎静止、且尺寸未变时才隔帧复用。
      const skip = frameNo % TUNE.reflectionEveryFrames !== 0 && !camMotion.moved && !camMotion.resized;
      if (skip || lastTick === frameNo) return;
      lastTick = frameNo;
      return updateOnce(frameData);
    };
  }
  const wp = positionWorld.xz;
  const n1 = mx_noise_float(vec3(wp.mul(vec2(0.05, 0.3)), time.mul(0.25)));
  const n2 = mx_noise_float(vec3(wp.mul(vec2(0.22, 1.3)).add(13.7), time.mul(0.6)));
  const n3 = mx_noise_float(vec3(wp.mul(vec2(0.9, 3.4)).add(3.1), time.mul(1.2)));
  const bd = length(wp.sub(uBoat.xz));
  const ripple = sin(bd.mul(3.2).sub(time.mul(3.0))).mul(exp(bd.mul(-0.28))).mul(0.8);
  const dist = length(positionWorld.sub(cameraPosition));
  const fade = float(1).div(dist.mul(0.004).add(1.0)).mul(smoothstep(0.0, 6.0, dist).mul(0.7).add(0.3));
  const off = vec2(
    n1.mul(0.3).add(n2.mul(0.25)).add(n3.mul(0.15)).add(ripple.mul(0.3)).mul(0.02),
    n1.mul(0.35).add(n2.mul(0.5)).add(n3.mul(0.4)).add(ripple.mul(0.6)).mul(0.14)
  ).mul(fade);
  reflection.uvNode = reflection.uvNode.add(off);
  const vd = normalize(cameraPosition.sub(positionWorld));
  const fres = pow(float(1).sub(max(vd.y, 0.0)), 4.0).mul(0.62).add(0.3);
  const water = new THREE.Mesh(new THREE.PlaneGeometry(9000, 9000), new THREE.MeshBasicNodeMaterial());
  // 月光碎影：扰动法线的高光形成粼粼月路
  const nrm = normalize(vec3(n2.mul(0.16).add(n3.mul(0.14)).add(ripple.mul(0.08)), 1.0, n1.mul(0.1).add(n3.mul(0.16)).add(n2.mul(0.08))));
  const rv = reflect(vd.negate(), nrm);
  const glit = pow(max(dot(rv, uMoonDir), 0.0), 420.0).mul(1.25).add(pow(max(dot(rv, uMoonDir), 0.0), 40.0).mul(0.12));
  water.material.colorNode = mix(color(0x060a15), reflection.rgb.mul(0.8), fres).add(vec3(1.0, 0.88, 0.7).mul(glit));
  water.rotation.x = -Math.PI / 2;
  scene.add(water);
}

/* =========================================================================
   屋顶生成器（庑殿 / 攒尖，起翘飞檐）
   ========================================================================= */
function roofGeometry(verts, topFn, h, upturn, segPerEdge = 16, rings = 14) {
  const perim = []; let acc = 0;
  const n = verts.length;
  for (let e = 0; e < n; e++) {
    const a = verts[e], b = verts[(e + 1) % n];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    let nx = (b[1] - a[1]) / len, nz = -(b[0] - a[0]) / len; if (nx * (a[0] + b[0]) + nz * (a[1] + b[1]) < 0) { nx = -nx; nz = -nz; }
    for (let s = 0; s < segPerEdge; s++) { const k = s / segPerEdge; perim.push({ x: lerp(a[0], b[0], k), z: lerp(a[1], b[1], k), cf: Math.pow(Math.abs(2 * k - 1), 7), u: acc + len * k, vtx: s === 0 ? e : -1, nx, nz }); }
    acc += len;
  }
  perim.push({ ...perim[0], u: acc, vtx: -1 });
  const M2 = perim.length, pos = [], uvs = [], corners = verts.map(() => []), eave = [];
  for (let r = 0; r <= rings; r++) {
    const t = r / rings;
    for (const p of perim) {
      const top = topFn(p.x, p.z), ext = 1 + 0.14 * p.cf * Math.pow(1 - t, 2);
      const x = lerp(p.x * ext, top.x, t), z = lerp(p.z * ext, top.z, t);
      const y = h * (0.28 * t + 0.72 * t * t) + upturn * p.cf * Math.pow(1 - t, 3);
      pos.push(x, y, z); uvs.push(p.u * 0.5, t * (h + 4) * 0.35);
      if (p.vtx >= 0) corners[p.vtx].push(V(x, y + 0.08, z));
      if (r === 0 && p !== perim[M2 - 1]) eave.push({ p: V(x, y, z), cf: p.cf, nx: p.nx, nz: p.nz });
    }
  }
  const idx = [];
  for (let r = 0; r < rings; r++) for (let j = 0; j < M2 - 1; j++) { const a = r * M2 + j, b = a + 1, c = a + M2, d = c + 1; idx.push(a, b, c, b, d, c); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  g.setIndex(idx); g.computeVertexNormals();
  // 保证法线朝外朝上
  const nrm = g.attributes.normal; let sy = 0; for (let i = 0; i < nrm.count; i++) sy += nrm.getY(i);
  if (sy < 0) { const ix = g.index.array; for (let i = 0; i < ix.length; i += 3) { const t = ix[i + 1]; ix[i + 1] = ix[i + 2]; ix[i + 2] = t; } g.computeVertexNormals(); }
  return { geo: g, corners, eave };
}
function addRoof(mg, { type = 'rect', hw, hd, rw = 0, h, upturn, n = 8, r = 1, pos = [0, 0, 0], ridge = 0.14, mat = M.tile, topScale = 0, eaveLight = null, finial = true }) {
  let verts, topFn;
  if (type === 'rect') { verts = [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]]; topFn = (x) => ({ x: Math.max(-rw, Math.min(rw, x)), z: 0 }); }
  else { verts = []; for (let i = 0; i < n; i++) { const a = (i + 0.5) / n * Math.PI * 2; verts.push([Math.cos(a) * r, Math.sin(a) * r]); } topFn = (x, z) => ({ x: x * topScale, z: z * topScale }); }
  const { geo, corners, eave } = roofGeometry(verts, topFn, h, upturn);
  const m = new THREE.Matrix4().makeTranslation(...pos);
  mg.addM(geo, mat, m); mg.addM(geo, M.under, m);
  const sc = Math.max(0.03, ridge);
  for (const c of corners) { if (c.length < 2) continue; const tube = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(c), 20, sc * 0.6, 6); mg.addM(tube, M.ridge, m); }
  const size = type === 'rect' ? Math.min(hw, hd) : r;
  if (size > 1.4) {
    // 封檐板 + 椽头 + 檐角风铃
    const ec = new THREE.CatmullRomCurve3(eave.map((e) => e.p), true);
    mg.addM(new THREE.TubeGeometry(ec, eave.length * 3, sc * 0.5, 5, true), M.wood, m);
    if (eaveLight) mg.addM(new THREE.TubeGeometry(ec, eave.length * 3, sc * 0.26, 4, true), eaveLight, new THREE.Matrix4().makeTranslation(pos[0], pos[1] - sc * 0.55, pos[2]));
    for (const e of eave) {
      if (e.cf > 0.2) continue;
      mg.add(new THREE.BoxGeometry(sc * 0.7, sc * 0.7, sc * 6), M.lacquer, [pos[0] + e.p.x - e.nx * sc * 2.4, pos[1] + e.p.y - sc * 0.8, pos[2] + e.p.z - e.nz * sc * 2.4], [0, Math.atan2(e.nx, e.nz), 0]);
    }
    for (const c of corners) { const q = c[0]; mg.add(new THREE.ConeGeometry(sc * 0.9, sc * 2.4, 8), M.gold, [pos[0] + q.x, pos[1] + q.y - sc * 2.2, pos[2] + q.z]); }
  }
  if (type === 'rect') {
    mg.add(new THREE.BoxGeometry(rw * 2 + 0.3, sc * 3.2, sc * 2.4), M.ridge, [pos[0], pos[1] + h + sc * 0.8, pos[2]]);
    mg.add(new THREE.BoxGeometry(rw * 2 + 0.3, sc * 0.8, sc * 2.8), M.ridge, [pos[0], pos[1] + h + sc * 2.6, pos[2]]);
    for (const sx of [-1, 1]) { const orn = new THREE.TorusGeometry(sc * 3, sc * 0.9, 6, 12, Math.PI * 1.1); mg.add(orn, M.ridge, [pos[0] + sx * (rw + 0.1), pos[1] + h + sc * 3.2, pos[2]], [0, 0, sx > 0 ? -0.3 : Math.PI + 0.3 - Math.PI * 0.1]); }
  } else if (topScale === 0 && finial) {
    const fy = pos[1] + h;
    mg.add(new THREE.SphereGeometry(sc * 3, 12, 8), M.gold, [pos[0], fy + sc * 2, pos[2]]);
    mg.add(new THREE.ConeGeometry(sc * 1.6, sc * 14, 8), M.gold, [pos[0], fy + sc * 10, pos[2]]);
    mg.add(new THREE.SphereGeometry(sc * 1.8, 10, 8), M.gold, [pos[0], fy + sc * 7, pos[2]]);
  }
}

/* =========================================================================
   灯笼
   ========================================================================= */
function lanternParts(mg, x, y, z, s = 1, mat = M.paper) {
  const prof = []; for (let i = 0; i <= 12; i++) { const t = i / 12; prof.push(new THREE.Vector2(Math.sin(Math.PI * t) * 0.34 * s + 0.12 * s, (t - 0.5) * 0.82 * s)); }
  mg.add(new THREE.LatheGeometry(prof, 20), mat, [x, y, z]);
  mg.add(new THREE.CylinderGeometry(0.15 * s, 0.17 * s, 0.1 * s, 16), M.redCap, [x, y + 0.44 * s, z]);
  mg.add(new THREE.CylinderGeometry(0.17 * s, 0.15 * s, 0.1 * s, 16), M.redCap, [x, y - 0.44 * s, z]);
  mg.add(new THREE.CylinderGeometry(0.012 * s, 0.012 * s, 0.6 * s, 4), M.ridge, [x, y + 0.78 * s, z]);
  mg.add(new THREE.ConeGeometry(0.06 * s, 0.4 * s, 8), M.redCap, [x, y - 0.68 * s, z], [Math.PI, 0, 0]);
}
function pointLight(parent, x, y, z, intensity = 14, dist = 22, col = 0xffa860) {
  const l = new THREE.PointLight(col, intensity, dist, 2); l.position.set(x, y, z); parent.add(l); return l;
}

/* =========================================================================
   岸 · 水榭 · 石阶 · 栈桥 · 石灯
   ========================================================================= */
phase('天空 · 水面 · 灯光');
loadMsg('筑台起榭 · 朱阁绮户');
await nextFrame();
const shoreX = (z) => -22.5 + fbm(z * 0.035, 3.3, 3) * 6 - Math.max(0, -z - 4) * 0.32 - Math.max(0, z - 70) * 0.35 + (z > 19 && z < 31 ? 1.5 : 0);
{
  const s = new THREE.Shape(); s.moveTo(-340, -95);
  for (let z = -95; z <= 150; z += 2) s.lineTo(shoreX(z), z);
  s.lineTo(-340, 150); s.lineTo(-340, -95);
  const geo = new THREE.ExtrudeGeometry(s, { depth: 4, bevelEnabled: true, bevelThickness: 0.7, bevelSize: 1.2, bevelSegments: 3, curveSegments: 4 });
  geo.rotateX(Math.PI / 2); geo.translate(0, 1.5 - 0.7, 0);
  const shore = new THREE.Mesh(geo, M.ground); shore.receiveShadow = true; scene.add(shore);
}

// 右岸
const rightX = (z) => { let x = 41 + fbm(z * 0.04, 9.1, 3) * 6 + Math.max(0, -z - 60) * 0.2 + Math.max(0, z - 80) * 0.3; if (z > -46 && z < 36) x = Math.max(x, 39.5); return x; };
{
  const s = new THREE.Shape(); s.moveTo(380, -175);
  for (let z = -175; z <= 150; z += 2) s.lineTo(rightX(z), z);
  s.lineTo(380, 150); s.lineTo(380, -175);
  const geo = new THREE.ExtrudeGeometry(s, { depth: 4, bevelEnabled: true, bevelThickness: 0.7, bevelSize: 1.2, bevelSegments: 3, curveSegments: 4 });
  geo.rotateX(Math.PI / 2); geo.translate(0, 1.5 - 0.7, 0);
  const shore = new THREE.Mesh(geo, M.ground); shore.receiveShadow = true; scene.add(shore);
}

// 岩石
function rockGeo(seed, detail = 3) {
  let g = new THREE.IcosahedronGeometry(1, detail); g.deleteAttribute('uv'); g.deleteAttribute('normal'); g = mergeVertices(g);
  const p = g.attributes.position, v = V();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i).normalize();
    let d = 1 + 0.38 * fbm(v.x * 1.6 + seed, v.y * 1.6 + v.z * 1.3 + seed, 4) + 0.12 * noise2(v.x * 5 + v.z * 3, v.y * 5 + seed);
    const q = v.clone().multiplyScalar(d); if (q.y < -0.25) q.y = -0.25 + (q.y + 0.25) * 0.3;
    p.setXYZ(i, q.x, q.y, q.z);
  }
  g.computeVertexNormals();
  const uvA = new Float32Array(p.count * 2); for (let i = 0; i < p.count; i++) { uvA[i * 2] = (p.getX(i) + p.getZ(i)) * 0.5; uvA[i * 2 + 1] = p.getY(i) * 0.5; }
  g.setAttribute('uv', new THREE.BufferAttribute(uvA, 2));
  return g;
}
const ROCKS = [0, 1, 2, 3, 4].map((i) => rockGeo(i * 7.7 + 1));
{
  const mg = new Merger();
  for (let z = -60; z < 125; z += R(2.2, 4)) {
    if (z > 19.5 && z < 30.5) continue;
    const x = shoreX(z) + R(-1, 1.8), s = R(0.9, 2.4);
    mg.add(ROCKS[(rand() * 5) | 0], M.rock, [x, R(0.2, 1.2), z], [R(0, 3), R(0, 6), R(0, 3) * 0.2], [s * R(1, 1.6), s * R(0.6, 1), s]);
  }
  // 阶前大石
  mg.add(ROCKS[1], M.rock, [-19, 0.9, 18], [0, 1, 0], [2.6, 2.0, 2.4]);
  mg.add(ROCKS[3], M.rock, [-18.5, 0.5, 33], [0, 2, 0], [2.4, 1.6, 2.2]);
  mg.add(ROCKS[2], M.rock, [-24, 1.8, 16.5], [0, 0.4, 0], [1.8, 1.5, 1.6]);
  for (let z = -170; z < 140; z += R(2.4, 4.2)) { const x = rightX(z) - R(-0.5, 1.6), s = R(0.9, 2.2); mg.add(ROCKS[(rand() * 5) | 0], M.rock, [x, R(0.2, 1.1), z], [R(0, 3), R(0, 6), 0], [s * R(1, 1.5), s * R(0.6, 1), s]); }
  // 右前水石
  mg.add(ROCKS[4], M.rock, [14, 0.0, 20], [0, 0.5, 0], [2.4, 1.2, 2.0]);
  mg.add(ROCKS[0], M.rock, [16.8, -0.1, 21.5], [0, 2, 0], [1.7, 0.9, 1.5]);
  mg.add(ROCKS[3], M.rock, [12.5, -0.2, 22.5], [0, 1, 0], [1.1, 0.6, 1.0]);
  const g = mg.build(); scene.add(g);
}

// 水榭
const pavilion = new THREE.Group();
pavilion.position.set(-34, 1.5, 22); pavilion.rotation.y = Math.PI / 2;
{
  const mg = new Merger();
  const W = 14, D = 9, H = 5.4, FY = 1.6;
  mg.add(new THREE.BoxGeometry(W + 3, FY, D + 3), M.stone, [0, FY / 2, 0]);
  mg.add(new THREE.BoxGeometry(W + 3.3, 0.18, D + 3.3), M.stoneStep, [0, FY + 0.02, 0]);
  mg.add(new THREE.BoxGeometry(W + 1, 0.08, D + 1), M.plank, [0, FY + 0.14, 0]);
  const cx = [-6.3, -2.1, 2.1, 6.3], cz = [-4.2, 0, 4.2];
  for (const x of cx) for (const z of cz) {
    if (z === 0 && Math.abs(x) < 3) continue;
    mg.add(new THREE.CylinderGeometry(0.2, 0.23, H, 14), M.lacquer, [x, FY + H / 2, z]);
    mg.add(new THREE.CylinderGeometry(0.36, 0.4, 0.3, 14), M.stoneStep, [x, FY + 0.3, z]);
  }
  const TOP = FY + H;
  for (const z of [-4.2, 4.2]) mg.add(new THREE.BoxGeometry(W + 1.2, 0.42, 0.32), M.lacquer, [0, TOP - 0.1, z]);
  for (const x of [-6.3, 6.3]) mg.add(new THREE.BoxGeometry(0.32, 0.42, D + 0.4), M.lacquer, [x, TOP - 0.1, 0]);
  mg.add(new THREE.BoxGeometry(W + 2.2, 0.3, D + 2.2), M.wood, [0, TOP + 0.22, 0]);
  for (let i = -8; i <= 8; i++) { mg.add(new THREE.BoxGeometry(0.2, 0.22, 0.5), M.wood, [i * 0.9, TOP + 0.08, D / 2 + 0.95]); mg.add(new THREE.BoxGeometry(0.2, 0.22, 0.5), M.wood, [i * 0.9, TOP + 0.08, -D / 2 - 0.95]); }
  // 挂落（前、左右侧）
  for (let b = 0; b < 3; b++) { const x0 = cx[b] + 0.22, x1 = cx[b + 1] - 0.22; mg.add(new THREE.PlaneGeometry(x1 - x0, 0.6), M.lattice, [(x0 + x1) / 2, TOP - 0.62, 4.2]); }
  for (const z of [-2.1, 2.1]) mg.add(new THREE.PlaneGeometry(4.0, 0.6), M.lattice, [6.3, TOP - 0.62, z], [0, Math.PI / 2, 0]);
  // 后墙：槛墙 + 花窗 + 透光窗纸
  for (let b = 0; b < 3; b++) {
    const x0 = cx[b] + 0.22, x1 = cx[b + 1] - 0.22, xm = (x0 + x1) / 2, bw = x1 - x0;
    mg.add(new THREE.BoxGeometry(bw, 1.0, 0.3), M.wall, [xm, FY + 0.5 + 0.1, -4.2]);
    mg.add(new THREE.PlaneGeometry(bw, H - 1.9), M.lattice, [xm, FY + 1.1 + (H - 1.9) / 2, -4.05]);
    mg.add(new THREE.PlaneGeometry(bw, H - 1.9), b === 1 ? M.window : M.windowDim, [xm, FY + 1.1 + (H - 1.9) / 2, -4.25]);
    mg.add(new THREE.BoxGeometry(bw, 0.12, 0.2), M.lacquer, [xm, FY + 1.1, -4.1]);
  }
  // 左侧墙（-x）带圆窗感的花窗
  for (const z of [-2.1, 2.1]) {
    mg.add(new THREE.BoxGeometry(0.3, 1.0, 4.0), M.wall, [-6.3, FY + 0.6, z]);
    mg.add(new THREE.PlaneGeometry(4.0, H - 1.9), M.lattice, [-6.15, FY + 1.1 + (H - 1.9) / 2, z], [0, Math.PI / 2, 0]);
    mg.add(new THREE.PlaneGeometry(4.0, H - 1.9), M.windowDim, [-6.35, FY + 1.1 + (H - 1.9) / 2, z], [0, Math.PI / 2, 0]);
  }
  // 美人靠栏杆：前檐两侧、右侧（面月）
  const rail = (x0, z0, x1, z1) => {
    const len = Math.hypot(x1 - x0, z1 - z0), ang = Math.atan2(z1 - z0, x1 - x0), mx = (x0 + x1) / 2, mz = (z0 + z1) / 2;
    mg.add(new THREE.BoxGeometry(len, 0.1, 0.34), M.lacquer, [mx, FY + 1.02, mz], [0, -ang, 0]);
    mg.add(new THREE.BoxGeometry(len, 0.08, 0.1), M.lacquer, [mx, FY + 0.35, mz], [0, -ang, 0]);
    mg.add(new THREE.BoxGeometry(len, 0.3, 0.06), M.wood, [mx, FY + 0.2, mz], [0, -ang, 0]);
    const nb = Math.floor(len / 0.32);
    for (let i = 1; i < nb; i++) { const t = i / nb; mg.add(new THREE.BoxGeometry(0.05, 0.62, 0.05), M.lacquer, [lerp(x0, x1, t), FY + 0.68, lerp(z0, z1, t)], [0, 0, 0]); }
  };
  rail(-6.1, 4.2, -2.3, 4.2); rail(2.3, 4.2, 6.1, 4.2);
  rail(6.3, -4.0, 6.3, -0.2); rail(6.3, 0.2, 6.3, 4.0);
  // 屋顶
  addRoof(mg, { type: 'rect', hw: W / 2 + 2.3, hd: D / 2 + 2.3, rw: 4.2, h: 3.4, upturn: 1.35, pos: [0, TOP + 0.36, 0], ridge: 0.16 });
  // 室内：案几、坐墩、茶具
  mg.add(new THREE.BoxGeometry(2.0, 0.08, 1.0), M.wood, [0.6, FY + 0.82, -0.6]);
  for (const [lx, lz] of [[-0.35, -0.4], [1.55, -0.4], [-0.35, -0.8], [1.55, -0.8]]) mg.add(new THREE.BoxGeometry(0.08, 0.66, 0.08), M.wood, [lx, FY + 0.45, lz]);
  for (const sx of [-0.8, 2.0]) mg.add(new THREE.CylinderGeometry(0.28, 0.24, 0.5, 14), M.lacquer, [sx, FY + 0.39, -0.6]);
  mg.add(new THREE.SphereGeometry(0.13, 14, 10), M.gold, [0.4, FY + 0.97, -0.6], [0, 0, 0], [1, 0.8, 1]);
  for (const cxp of [0.85, 1.05]) mg.add(new THREE.CylinderGeometry(0.05, 0.035, 0.07, 10), M.wall, [cxp, FY + 0.9, -0.5]);
  // 吊灯（前檐两盏 + 内一盏）
  lanternParts(mg, -6.95, TOP - 1.05, 4.9, 1);
  lanternParts(mg, 6.95, TOP - 1.05, 4.9, 1);
  lanternParts(mg, 0.6, TOP - 1.4, -1.5, 0.8);
  mg.add(new THREE.CylinderGeometry(0.02, 0.02, 0.9, 4), M.ridge, [-6.95, TOP - 0.3, 4.9]);
  mg.add(new THREE.CylinderGeometry(0.02, 0.02, 0.9, 4), M.ridge, [6.95, TOP - 0.3, 4.9]);
  pavilion.add(mg.build());
  pointLight(pavilion, -6.95, TOP - 1.1, 5.4, 16, 24);
  pointLight(pavilion, 6.95, TOP - 1.1, 5.4, 16, 24);
  pointLight(pavilion, 0.6, TOP - 1.6, -1.2, 10, 14);
  // 纱帘（风动）
  const curtainMat = new THREE.MeshStandardNodeMaterial({ map: TX.silk, transparent: true, side: THREE.DoubleSide, roughness: 1 });
  curtainMat.emissiveNode = texture(TX.silk).rgb.mul(0.28);
  const topPin = positionLocal.y.add(H / 2 - 0.4).div(H - 0.8);
  curtainMat.positionNode = positionLocal.add(vec3(0, 0, sin(time.mul(1.1).add(positionLocal.y.mul(1.6)).add(positionLocal.x.mul(3.0))).mul(0.12).mul(float(1).sub(topPin).add(0.05))));
  for (const [x, z, w] of [[-5.4, -3.9, 1.4], [-3.0, -3.9, 1.2], [3.0, -3.9, 1.2], [5.4, -3.9, 1.4], [5.7, 3.9, 0.9], [-5.7, 3.9, 0.9]]) {
    const c = new THREE.Mesh(new THREE.PlaneGeometry(w, H - 0.8, 8, 14), curtainMat);
    c.position.set(x, FY + H / 2 - 0.1, z); c.castShadow = true; pavilion.add(c);
  }
}
scene.add(pavilion);

// 后院廊房（衬景）
{
  const mg = new Merger();
  const bx = -54, bz = 12;
  mg.add(new THREE.BoxGeometry(10, 5, 26), M.wall, [bx, 4, bz]);
  for (let i = 0; i < 5; i++) mg.add(new THREE.PlaneGeometry(3, 2.4), i % 2 ? M.window : M.windowDim, [bx + 5.02, 4.4, bz - 10 + i * 5], [0, Math.PI / 2, 0]);
  for (let i = 0; i < 6; i++) mg.add(new THREE.CylinderGeometry(0.18, 0.2, 5, 10), M.lacquer, [bx + 5.4, 4, bz - 12.5 + i * 5]);
  addRoof(mg, { type: 'rect', hw: 7.2, hd: 15.5, rw: 0.01, h: 3.2, upturn: 0.9, pos: [bx, 6.6, bz], ridge: 0.14 });
  const g = mg.build(); g.children.forEach((m) => { m.geometry.rotateY(0); }); scene.add(g);
}

// 两岸临水楼阁
function mergeInto(mg, L, m) { for (const [mat, list] of L.map) for (const g of list) mg.addM(g, mat, m); }
function house(mg, x, gy, z, rot, { w = 10, d = 7, floors = 2, h = 3.0, base = 0.9 } = {}) {
  const L = new Merger();
  const Bx = (a, b, c) => new THREE.BoxGeometry(a, b, c);
  // 石台基 + 条石压沿 + 临水踏步
  L.add(Bx(w + 1.2, base + 3, d + 1.2), M.stone, [0, base - (base + 3) / 2, 0]);
  L.add(Bx(w + 1.5, 0.16, d + 1.5), M.stoneStep, [0, base + 0.02, 0]);
  for (let k = 0; k < 4; k++) L.add(Bx(3.2 - k * 0.1, 0.3, 0.6), M.stoneStep, [w * 0.18, base - 0.3 - k * 0.3, d / 2 + 0.9 + k * 0.6]);
  let fy = base;
  for (let k = 0; k < floors; k++) {
    const s = 1 - k * 0.1, W = w * s, D = d * s, top = k === floors - 1;
    L.add(Bx(W + 0.5, 0.2, D + 0.5 + (k > 0 ? 0.8 : 0)), M.wood, [0, fy + 0.1, k > 0 ? 0.4 : 0]);
    // 墙：后墙、山墙（带木框分格）
    L.add(Bx(W, h, 0.2), M.wall, [0, fy + h / 2, -D / 2]);
    for (const sx of [-1, 1]) {
      L.add(Bx(0.2, h, D), M.wall, [sx * W / 2, fy + h / 2, 0]);
      for (const zz of [-D / 2, 0, D / 2 - 0.1]) L.add(Bx(0.26, h, 0.18), M.wood, [sx * (W / 2 + 0.02), fy + h / 2, zz]);
      L.add(Bx(0.26, 0.16, D), M.wood, [sx * (W / 2 + 0.02), fy + h * 0.33, 0]);
      L.add(new THREE.PlaneGeometry(D * 0.36, h * 0.4), M.lattice, [sx * (W / 2 + 0.13), fy + h * 0.66, -D * 0.22], [0, sx * Math.PI / 2, 0]);
      L.add(new THREE.PlaneGeometry(D * 0.36, h * 0.4), M.windowDim, [sx * (W / 2 + 0.05), fy + h * 0.66, -D * 0.22], [0, sx * Math.PI / 2, 0]);
    }
    const nb = Math.max(2, Math.round(W / 2.4)), bw = W / nb;
    for (let i = 0; i <= nb; i++) {
      const cx = -W / 2 + i * bw;
      L.add(new THREE.CylinderGeometry(0.13, 0.15, h, 12), M.lacquer, [cx, fy + h / 2, D / 2]);
      L.add(new THREE.CylinderGeometry(0.22, 0.24, 0.18, 12), M.stoneStep, [cx, fy + 0.29, D / 2]);
      // 雀替
      if (i > 0) L.add(Bx(0.5, 0.16, 0.12), M.lacquer, [cx - 0.3, fy + h - 0.42, D / 2 + 0.02]);
      if (i < nb) L.add(Bx(0.5, 0.16, 0.12), M.lacquer, [cx + 0.3, fy + h - 0.42, D / 2 + 0.02]);
    }
    for (let i = 0; i < nb; i++) {
      const bx = -W / 2 + (i + 0.5) * bw, r = rand(), wh = h - 1.3, wy = fy + 0.95 + wh / 2;
      L.add(Bx(bw - 0.26, 0.9, 0.16), M.wall, [bx, fy + 0.55, D / 2 - 0.05]);
      L.add(Bx(bw - 0.5, 0.55, 0.05), M.lacquer, [bx, fy + 0.55, D / 2 + 0.05]);
      L.add(new THREE.PlaneGeometry(bw - 0.26, wh), M.lattice, [bx, wy, D / 2 + 0.02]);
      L.add(new THREE.PlaneGeometry(bw - 0.26, wh), r < 0.62 ? M.window : (r < 0.86 ? M.windowDim : M.wood), [bx, wy, D / 2 - 0.1]);
      L.add(Bx(0.07, wh, 0.1), M.lacquer, [bx, wy, D / 2 + 0.03]);
      L.add(new THREE.PlaneGeometry(bw - 0.3, 0.5), M.lattice, [bx, fy + h - 0.55, D / 2 + 0.04]);
    }
    // 额枋 + 斗拱
    L.add(Bx(W + 0.3, 0.3, 0.26), M.lacquer, [0, fy + h - 0.1, D / 2]);
    L.add(Bx(W + 0.3, 0.14, 0.3), M.bracket, [0, fy + h + 0.1, D / 2 + 0.02]);
    for (let xx = -W / 2; xx <= W / 2 + 0.01; xx += 0.62) { L.add(Bx(0.22, 0.2, 0.42), M.wood, [xx, fy + h + 0.27, D / 2 + 0.08]); L.add(Bx(0.42, 0.1, 0.2), M.bracket, [xx, fy + h + 0.42, D / 2 + 0.12]); }
    if (k > 0) {
      const zr = D / 2 + 0.7;
      L.add(Bx(W + 0.3, 0.09, 0.28), M.lacquer, [0, fy + 0.95, zr]);
      L.add(Bx(W + 0.3, 0.07, 0.08), M.lacquer, [0, fy + 0.35, zr]);
      for (let xx = -W / 2; xx <= W / 2 + 0.01; xx += 0.36) L.add(Bx(0.05, 0.6, 0.05), M.lacquer, [xx, fy + 0.65, zr]);
      for (let xx = -W / 2; xx <= W / 2 + 0.01; xx += bw) L.add(Bx(0.12, 1.05, 0.12), M.lacquer, [xx, fy + 0.62, zr]);
    }
    fy += h;
    if (!top) {
      addRoof(L, { type: 'rect', hw: W / 2 + 1.0, hd: D / 2 + 1.1, rw: W / 2 * 0.7, h: 0.95, upturn: 0.35, pos: [0, fy + 0.35, 0], ridge: 0.08 });
      for (let i = 0; i < nb; i += 2) lanternParts(L, -W / 2 + (i + 0.5) * bw, fy - 0.3, D / 2 + 0.85, 0.7);
      fy += 1.02;
    } else {
      addRoof(L, { type: 'rect', hw: W / 2 + 1.5, hd: D / 2 + 1.5, rw: W / 2 * 0.6, h: h * 0.85, upturn: 0.95, pos: [0, fy + 0.45, 0], ridge: 0.13 });
      for (const sx of [-1, 1]) lanternParts(L, sx * (W / 2 + 0.6), fy - 0.35, D / 2 + 1.25, 0.8);
      // 匾额
      L.add(Bx(1.8, 0.62, 0.08), M.ridge, [0, fy - 0.05, D / 2 + 0.3]);
      L.add(Bx(1.95, 0.08, 0.1), M.gold, [0, fy + 0.27, D / 2 + 0.3]); L.add(Bx(1.95, 0.08, 0.1), M.gold, [0, fy - 0.37, D / 2 + 0.3]);
      L.add(Bx(1.1, 0.26, 0.02), M.gold, [0, fy - 0.05, D / 2 + 0.35]);
    }
  }
  mergeInto(mg, L, new THREE.Matrix4().compose(V(x, gy, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(0, rot, 0)), V(1, 1, 1)));
}
{
  const mg = new Merger();
  // 水榭望月的视线上留出空当（月低时不被楼阁遮挡）
  const left = [[-13, 10, 7, 2, 6], [-25, 12, 8, 3, 0], [-40, 9, 7, 2, 0], [-55, 11, 8, 2, 0]];
  for (const [z, w, d, f, back] of left) house(mg, shoreX(z) - d / 2 - 2.6 - back, 1.5, z, Math.PI / 2, { w, d, floors: f, h: R(2.8, 3.3) });
  const right = [[40, 12, 8, 2], [25, 10, 7, 2], [9, 12, 8, 3], [-7, 9, 7, 1], [-22, 12, 8, 2], [-38, 10, 7, 2], [-56, 13, 9, 3], [-78, 11, 8, 2], [-100, 12, 8, 2], [-126, 14, 9, 3], [-150, 11, 8, 2]];
  for (const [z, w, d, f] of right) house(mg, rightX(z) + d / 2 + 3.2, 1.5, z, -Math.PI / 2, { w, d, floors: f, h: R(2.8, 3.3) });
  // 右岸临水廊
  const Bx = (a, b, c) => new THREE.BoxGeometry(a, b, c);
  mg.add(Bx(2.8, 0.16, 78), M.plank, [37.4, 1.05, -5]);
  for (let z = 33; z >= -43; z -= 2.6) { mg.add(new THREE.CylinderGeometry(0.12, 0.14, 2.4, 8), M.wood, [36.1, -0.1, z]); mg.add(Bx(0.1, 1.0, 0.1), M.wood, [36.1, 1.6, z]); }
  mg.add(Bx(0.12, 0.1, 78), M.wood, [36.1, 2.12, -5]); mg.add(Bx(0.08, 0.07, 78), M.wood, [36.1, 1.6, -5]);
  for (let z = 30; z >= -42; z -= 9) { mg.add(new THREE.CylinderGeometry(0.06, 0.07, 2.2, 6), M.wood, [36.1, 2.2, z]); mg.add(Bx(0.5, 0.05, 0.05), M.wood, [35.9, 3.28, z]); lanternParts(mg, 35.7, 2.72, z, 0.55); }
  mg.add(Bx(9, 0.16, 2.6), M.plank, [32.5, 0.75, -12]);
  for (let x = 28.5; x <= 36; x += 2.5) for (const z of [-13.2, -10.8]) mg.add(new THREE.CylinderGeometry(0.12, 0.14, 2, 8), M.wood, [x, -0.2, z]);
  scene.add(mg.build());
  // 每一盏点光都会进入所有标准材质的逐像素光照循环，代价是按盏数线性叠加的。
  // ?lights=lean 跳过两岸民居这 6 盏（19→13），用来实测光照循环到底占多少帧时间。
  if (!LEAN_LIGHTS) for (const [x, y, z] of [[-31, 5.2, -13], [-28, 5.2, -25], [35.7, 3.2, 21], [35.7, 3.2, -6], [37, 5.2, 9], [36.5, 5.2, -22]]) pointLight(scene, x, y, z, 30, 30);
}

// 石阶
{
  const mg = new Merger();
  for (let i = 0; i < 7; i++) {
    const top = 3.1 - 0.45 * (i + 1) + 0.45, x = -28 + 1.15 * (i + 0.5);
    mg.add(new THREE.BoxGeometry(1.25, top + 1, 6.2), M.stoneStep, [x, (top - 1) / 2, 25]);
  }
  for (const z of [21.6, 28.4]) mg.add(new THREE.BoxGeometry(8.6, 0.7, 0.6), M.stone, [-24, 1.55, z], [0, 0, 0.36]);
  // 栈桥
  for (let x = -20; x <= -2; x += 3) for (const z of [23.6, 26.4]) mg.add(new THREE.CylinderGeometry(0.14, 0.16, 2.2, 8), M.wood, [x, -0.3, z]);
  mg.add(new THREE.BoxGeometry(19, 0.16, 3.2), M.plank, [-11, 0.62, 25]);
  for (let x = -19.5; x <= -2.5; x += 1.2) mg.add(new THREE.BoxGeometry(0.06, 0.02, 3.2), M.ridge, [x, 0.71, 25]);
  for (let x = -20; x <= -2; x += 3) mg.add(new THREE.BoxGeometry(0.12, 1.1, 0.12), M.wood, [x, 1.2, 23.5]);
  mg.add(new THREE.BoxGeometry(18.2, 0.1, 0.1), M.wood, [-11, 1.72, 23.5]);
  mg.add(new THREE.BoxGeometry(18.2, 0.07, 0.07), M.wood, [-11, 1.2, 23.5]);
  // 石灯笼
  const sl = (x, z, s) => {
    mg.add(new THREE.CylinderGeometry(0.45 * s, 0.55 * s, 0.3 * s, 8), M.stoneStep, [x, 1.5 + 0.15 * s, z]);
    mg.add(new THREE.CylinderGeometry(0.16 * s, 0.2 * s, 0.9 * s, 8), M.stoneStep, [x, 1.5 + 0.75 * s, z]);
    mg.add(new THREE.BoxGeometry(0.8 * s, 0.14 * s, 0.8 * s), M.stoneStep, [x, 1.5 + 1.25 * s, z]);
    mg.add(new THREE.BoxGeometry(0.56 * s, 0.55 * s, 0.56 * s), M.stoneStep, [x, 1.5 + 1.6 * s, z]);
    mg.add(new THREE.BoxGeometry(0.34 * s, 0.34 * s, 0.6 * s), M.lamp, [x, 1.5 + 1.6 * s, z]);
    mg.add(new THREE.BoxGeometry(0.6 * s, 0.34 * s, 0.34 * s), M.lamp, [x, 1.5 + 1.6 * s, z]);
    addRoof(mg, { type: 'poly', n: 4, r: 0.75 * s, h: 0.5 * s, upturn: 0.12 * s, pos: [x, 1.5 + 1.88 * s, z], ridge: 0.03 * s });
  };
  sl(-21.5, 19.2, 1.3); sl(-21.5, 31, 1.1);
  const g = mg.build(); scene.add(g);
  pointLight(scene, -21.5, 3.6, 19.2, 7, 12); pointLight(scene, -21.5, 3.3, 31, 5, 10);
}

/* =========================================================================
   树：梅/海棠（花）、松
   ========================================================================= */
/* ---------- 远景减模 ----------
   地景里的林木是全场最多面的东西（三片林共五千多棵，一棵 80 面）。
   这里按「到舟行中心的距离」分两级：圈内的近树保持 80 面，圈外的换 20 面。
   半径取 210：全部镜位都在这个圈内 60 米以内，所以圈外的树恒定在 200 米开外，
   那个角径上低面数几何分辨不出来。
   关键：只有几何换了，随机数照旧按原顺序抽完 —— 每一棵树的位置、大小、
   朝向，以及下游所有场景的随机细节，都与减模前逐位一致。 */
const LOD_R2 = 210 * 210;
const farAway = (x, z) => (x - 14) ** 2 + (z + 6) ** 2 > LOD_R2;   // 圈心＝舟行中心
const FOLIAGE_HI = new THREE.IcosahedronGeometry(1, 1);   // 80 面
const FOLIAGE_LO = new THREE.IcosahedronGeometry(1, 0);   // 20 面

phase('岸 · 水榭 · 栈桥');
loadMsg('植松种花 · 疏影横斜');
await nextFrame();
function taperTube(pts, r0, r1, radial = 7) {
  const curve = new THREE.CatmullRomCurve3(pts), segs = Math.max(4, pts.length * 3);
  const g = new THREE.TubeGeometry(curve, segs, 1, radial, false);
  const p = g.attributes.position, c = V(), v = V();
  for (let i = 0; i <= segs; i++) {
    curve.getPointAt(i / segs, c); const r = lerp(r0, r1, i / segs);
    for (let j = 0; j <= radial; j++) { const k = i * (radial + 1) + j; v.fromBufferAttribute(p, k).sub(c).multiplyScalar(r).add(c); p.setXYZ(k, v.x, v.y, v.z); }
  }
  g.computeVertexNormals(); return { geo: g, curve };
}
function growTree(root, dir, len, rad, depth, opt, out) {
  const pts = [root.clone()]; const cd = dir.clone().normalize(); let cur = root.clone();
  for (let i = 1; i <= 4; i++) { cd.add(V(R(-1, 1), R(-0.6, 1) * opt.up, R(-1, 1)).multiplyScalar(opt.wiggle)).normalize(); cd.y -= opt.droop * (1 - depth / opt.maxDepth); cd.normalize(); cur = cur.clone().addScaledVector(cd, len / 4); pts.push(cur); }
  const { geo, curve } = taperTube(pts, rad, rad * 0.62, depth > 2 ? 8 : 5);
  out.geos.push(geo);
  if (depth <= 2) for (let i = 0; i < 4; i++) out.tips.push(curve.getPointAt(R(0.3, 1)));
  if (depth === 0) { out.tips.push(cur.clone()); out.ends.push({ p: cur.clone(), d: cd.clone() }); return; }
  const kids = depth > 3 ? 2 : 3;
  for (let k = 0; k < kids; k++) {
    const t = k === 0 ? 1 : R(0.45, 0.9), p = curve.getPointAt(t), tan = curve.getTangentAt(t);
    const nd = tan.clone().add(V(R(-1, 1), R(-0.2, 0.8), R(-1, 1)).multiplyScalar(opt.spread)).normalize();
    growTree(p, nd, len * R(0.62, 0.82), rad * 0.6, depth - 1, opt, out);
  }
}
// 花树
const blossom = { tips: [], geos: [], ends: [] };
{
  growTree(V(-44, 1.3, 36), V(0.35, 1, -0.15), 7.5, 0.62, 5, { wiggle: 0.35, droop: 0.12, spread: 0.9, up: 0.6, maxDepth: 5 }, blossom);
  growTree(V(-44, 1.3, 36), V(0.9, 0.55, -0.3), 7, 0.45, 4, { wiggle: 0.35, droop: 0.18, spread: 0.9, up: 0.5, maxDepth: 4 }, blossom);
  const bo = { wiggle: 0.35, droop: 0.12, spread: 0.9, up: 0.6, maxDepth: 4 };
  for (const [x, z, l] of [[-22.5, -6, 4.2], [-33, -33, 4.8], [-30, -62, 4.4], [46, 33, 4.4], [47, -14, 4.8], [48, -46, 4.2], [50, -88, 5]]) growTree(V(x, 1.3, z), V(R(-0.3, 0.3), 1, R(-0.3, 0.3)), l, 0.34, 4, bo, blossom);
  const mg = new Merger(); for (const g of blossom.geos) mg.addM(g, M.bark, new THREE.Matrix4());
  scene.add(mg.build());
  const fmat = new THREE.MeshStandardNodeMaterial({ map: TX.flower, alphaTest: 0.45, side: THREE.DoubleSide, roughness: 0.8 });
  fmat.emissiveNode = texture(TX.flower).rgb.mul(0.22);
  const count = blossom.tips.length * 9;
  const flowers = Inst(new THREE.PlaneGeometry(0.42, 0.42), fmat, count);
  const d = new THREE.Object3D(), col = new THREE.Color(); let k = 0;
  for (const tp of blossom.tips) for (let i = 0; i < 9; i++) {
    d.position.copy(tp).add(V(R(-1, 1), R(-0.6, 0.9), R(-1, 1)).multiplyScalar(0.95));
    d.rotation.set(R(0, 6.3), R(0, 6.3), R(0, 6.3)); d.scale.setScalar(R(0.7, 1.25)); d.updateMatrix();
    flowers.setMatrixAt(k, d.matrix); col.setHSL(0.95, R(0.1, 0.5), R(0.82, 1)); flowers.setColorAt(k, col); k++;
  }
  flowers.castShadow = true; scene.add(flowers);
}
// 松
function pine(root, dir, h, rad, seed, pads = 7, spread = 1) {
  const out = { geos: [] }, foliage = [];
  const pts = [root.clone()]; let cur = root.clone(); const cd = dir.clone().normalize();
  for (let i = 1; i <= 6; i++) { cd.add(V(R(-0.4, 0.4), 0.15, R(-0.4, 0.4))).normalize(); cur = cur.clone().addScaledVector(cd, h / 6); pts.push(cur); }
  const trunk = taperTube(pts, rad, rad * 0.35, 9); out.geos.push(trunk.geo);
  for (let i = 0; i < pads; i++) {
    const t = lerp(0.45, 1, i / (pads - 1)), p = trunk.curve.getPointAt(t);
    const a = R(0, 6.28), bl = h * R(0.22, 0.42) * spread * (1.2 - t * 0.5);
    const bd = V(Math.cos(a), R(-0.05, 0.25), Math.sin(a)).normalize();
    const bp = [p.clone()]; let c = p.clone(); for (let j = 1; j <= 3; j++) { c = c.clone().addScaledVector(bd, bl / 3).add(V(0, R(-0.2, 0.3), 0)); bp.push(c); }
    out.geos.push(taperTube(bp, rad * 0.35, rad * 0.12, 5).geo);
    foliage.push({ p: c.clone().add(V(0, 0.25, 0)), s: h * R(0.12, 0.17) * spread });
  }
  foliage.push({ p: trunk.curve.getPointAt(1).add(V(0, 0.3, 0)), s: h * 0.16 * spread });
  return { out, foliage };
}
const pineFoliage = [];
const pineMerger = new Merger();
function addPine(...a) { const { out, foliage } = pine(...a); for (const g of out.geos) pineMerger.addM(g, M.bark, new THREE.Matrix4()); pineFoliage.push(...foliage); }
addPine(V(-19.5, 1.5, 9.5), V(0.6, 1, -0.2), 7.5, 0.32, 2, 6, 1.0);
addPine(V(-44, 1.5, 4), V(0.1, 1, 0.1), 11, 0.4, 3, 6, 0.9);
addPine(V(-48, 1.5, 44), V(0.2, 1, 0.1), 14, 0.5, 4, 7, 1.0);
{
  scene.add(pineMerger.build());
  const pad = new THREE.SphereGeometry(1, 14, 9);
  const inst = Inst(pad, M.needle, pineFoliage.length * 4);
  const d = new THREE.Object3D(); let k = 0;
  for (const f of pineFoliage) for (let i = 0; i < 4; i++) {
    d.position.copy(f.p).add(V(R(-1, 1) * f.s * 0.8, R(-0.2, 0.3) * f.s, R(-1, 1) * f.s * 0.8));
    d.scale.set(f.s * R(0.7, 1.1), f.s * R(0.26, 0.36), f.s * R(0.7, 1.1)); d.rotation.y = R(0, 6); d.updateMatrix(); inst.setMatrixAt(k++, d.matrix);
  }
  inst.castShadow = true; inst.receiveShadow = true; scene.add(inst);
}

// 垂柳
{
  const bark = new Merger(), anchors = [], counts = [];
  const willow = (root, dir, len, rad, n) => {
    const out = { tips: [], geos: [], ends: [] };
    growTree(root, dir, len, rad, 3, { wiggle: 0.22, droop: -0.05, spread: 0.75, up: 1.0, maxDepth: 3 }, out);
    for (const g of out.geos) bark.addM(g, M.bark, new THREE.Matrix4());
    for (let i = 0; i < n; i++) anchors.push({ p: out.tips[(rand() * out.tips.length) | 0], len: R(3, 8.5) * len / 7 });
  };
  willow(V(45, 1.3, 13), V(-1, 1.1, -0.15), 7.5, 0.55, 420);
  willow(V(47, 1.3, -30), V(-0.6, 1, 0.1), 6.5, 0.45, 220);
  willow(V(-25, 1.3, -18), V(0.5, 1, 0), 6, 0.42, 200);
  willow(V(-36, 1.3, -47), V(0.4, 1, 0), 6, 0.42, 180);
  scene.add(bark.build());
  const g1 = new THREE.PlaneGeometry(0.6, 1, 1, 12); g1.translate(0, -0.5, 0);
  const sg = mergeGeometries([g1, g1.clone().rotateY(Math.PI / 2)]);
  const inst = Inst(sg, M.willow, anchors.length), d = new THREE.Object3D();
  anchors.forEach((a, i) => { d.position.copy(a.p).add(V(R(-0.3, 0.3), 0, R(-0.3, 0.3))); d.rotation.set(R(-0.08, 0.08), R(0, 6.3), R(-0.08, 0.08)); d.scale.set(R(0.8, 1.2), Math.min(a.len, a.p.y - 0.3), 1); d.updateMatrix(); inst.setMatrixAt(i, d.matrix); });
  inst.castShadow = true; scene.add(inst);
}

// 岸上林木
{
  const inst = Inst(FOLIAGE_HI, M.foliage, 1600), far = Inst(FOLIAGE_LO, M.foliageFar, 1600);
  const d = new THREE.Object3D(); let k = 0, kf = 0;
  const skip = (x, z) => (x > -42 && x < -26 && z > 11 && z < 33) || (x > -61 && x < -46 && z > -3 && z < 27);
  while (k + kf < 1600) {
    const z = R(-170, 140), left = rand() < 0.5;
    const x = left ? shoreX(z) - R(14, 100) : rightX(z) + R(17, 100);
    if (skip(x, z)) continue;
    d.position.set(x, 1.5 + R(0, 0.8), z); const sc = R(1.4, 3.4); d.scale.set(sc, sc * R(1.1, 2.3), sc); d.rotation.y = R(0, 6); d.updateMatrix();
    if (farAway(x, z)) far.setMatrixAt(kf++, d.matrix); else inst.setMatrixAt(k++, d.matrix);
  }
  inst.count = k; far.count = kf;
  inst.receiveShadow = true; far.receiveShadow = true; scene.add(inst, far);
}

/* =========================================================================
   远山 · 峭壁 · 宝塔 · 石桥
   ========================================================================= */
phase('花木 · 松');
loadMsg('远山叠翠 · 琼楼玉宇');
await nextFrame();
const ridged = (x, z, o = 5) => { let s2 = 0, a = 0.5, f = 1; for (let i = 0; i < o; i++) { const n = 1 - Math.abs(noise2(x * f, z * f)); s2 += a * n * n; f *= 2.07; a *= 0.5; } return s2; };
function makeRange({ x0, x1, z0, z1, peaks, hmin, hmax, smin, smax, seed, mat, forced = [], segX = 300, segZ = 90 }) {
  const pk = [...forced];
  for (let i = 0; i < peaks; i++) pk.push({ x: R(x0 + 30, x1 - 30), z: R(lerp(z0, z1, 0.25), lerp(z0, z1, 0.7)), h: R(hmin, hmax), s: R(smin, smax) });
  const H = (x, z) => {
    let h = 0;
    for (const p of pk) { const dx = (x - p.x) / p.s, dz = (z - p.z) / (p.s * 1.25); const g = Math.exp(-(dx * dx + dz * dz)); h = Math.max(h, p.h * Math.pow(g, 0.62)); }
    h *= 0.5 + 0.65 * ridged(x * 0.011 + seed, z * 0.011);
    h += ridged(x * 0.042, z * 0.042 + seed, 4) * 8 * clamp01(h / 12);
    h += fbm(x * 0.16, z * 0.16 + seed, 2) * 0.9;
    const e = clamp01((z1 - z) / 22) * clamp01((z - z0) / 30);
    // 为月出方向留出自然山口，保留两侧山势。
    const moonCorridor = Math.exp(-Math.pow((x + z * 0.13) / (55 + Math.abs(z) * 0.10), 2));
    return h * e * (1 - moonCorridor * 0.8) - 3 * (1 - e) - 1;
  };
  const geo = new THREE.PlaneGeometry(x1 - x0, z1 - z0, segX, segZ); geo.rotateX(-Math.PI / 2); geo.translate((x0 + x1) / 2, 0, (z0 + z1) / 2);
  const p = geo.attributes.position; for (let i = 0; i < p.count; i++) p.setY(i, H(p.getX(i), p.getZ(i)));
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, mat); m.receiveShadow = false; scene.add(m);
  return H;
}
// 山体细分按「离镜位活动区的远近」递减：近山细、远山粗。
// 最远一档在 600~1000 米外、雾里已吃掉七八成对比，粗一点的脊线看不出来。
// 注意 makeRange 只改网格密度、不改峰位，下面的随机流一点没动。
const nearH = makeRange({ x0: -300, x1: -5, z0: -175, z1: -55, peaks: 8, hmin: 24, hmax: 52, smin: 16, smax: 30, seed: 3, mat: M.hill0, forced: [{ x: -72, z: -125, h: 55, s: 24 }], segX: 280, segZ: 112 });
makeRange({ x0: -700, x1: 700, z0: -500, z1: -300, peaks: 16, hmin: 60, hmax: 130, smin: 40, smax: 80, seed: 9, mat: M.hill1, segX: 240, segZ: 64 });
makeRange({ x0: -1400, x1: 1400, z0: -950, z1: -600, peaks: 22, hmin: 120, hmax: 240, smin: 60, smax: 120, seed: 17, mat: M.hill2, segX: 130, segZ: 32 });
makeRange({ x0: 175, x1: 520, z0: -320, z1: -110, peaks: 5, hmin: 40, hmax: 80, smin: 22, smax: 38, seed: 21, mat: M.hill0, segX: 150, segZ: 84 });
// 近山林木
{
  // 远近两级：圈内的树用 80 面球，圈外的树换 20 面球 + 平滑着色，轮廓才不炸。
  // 随机数照旧按原顺序抽完，所以减模不影响任何细节。
  const inst = Inst(FOLIAGE_HI, M.foliage, 2600), far = Inst(FOLIAGE_LO, M.foliageFar, 2600);
  const cone = Inst(new THREE.ConeGeometry(1, 2.4, 7), M.needle, 867);
  const d = new THREE.Object3D(); let k = 0, kf = 0; scene.add(cone, far);
  while (k + kf < 2600) {
    const x = R(-290, -10), z = R(-165, -60), y = nearH(x, z); if (y < 2) continue;
    d.position.set(x, y + 0.3, z); const s = R(0.8, 1.8); d.scale.set(s, s * R(1.2, 2.2), s); d.rotation.y = R(0, 6);
    if ((k + kf) % 3 === 0) { cone.setMatrixAt((k + kf) / 3, new THREE.Matrix4().compose(V(x, y + 1.2, z), new THREE.Quaternion(), V(s * 0.8, s * 1.6, s * 0.8))); } d.updateMatrix();
    if (farAway(x, z)) far.setMatrixAt(kf++, d.matrix); else inst.setMatrixAt(k++, d.matrix);
  }
  inst.count = k; far.count = kf;
  scene.add(inst);
}

// 夕照山 + 雷峰塔
const PG = V(122, 30, -205);
const PAGODA_TOP = V(PG.x, 90, PG.z);
const hillH = (x, z) => {
  const r = Math.hypot(x - PG.x, z - PG.z);
  let h = r < 17 ? PG.y : PG.y * Math.pow(Math.exp(-Math.pow((r - 17) / 40, 2)), 0.9);
  h += (ridged(x * 0.05, z * 0.05 + 4, 4) * 5 + fbm(x * 0.2, z * 0.2, 2) * 0.6) * clamp01((r - 17) / 12) * clamp01(h / 8);
  return h - 1.2;
};
{
  // 夕照山本体：镜位最近也在 46 米外、且多在俯视角度，网格 200→160 已足够；
  // 树一棵没减——第 4、5 镜会贴到山腰，低面数的树在那里看得出来。
  const geo = new THREE.PlaneGeometry(240, 240, 160, 160); geo.rotateX(-Math.PI / 2); geo.translate(PG.x, 0, PG.z);
  const p = geo.attributes.position; for (let i = 0; i < p.count; i++) p.setY(i, hillH(p.getX(i), p.getZ(i)));
  geo.computeVertexNormals();
  const hill = new THREE.Mesh(geo, M.hill0); hill.receiveShadow = true; scene.add(hill);
  // 山林
  const trees = Inst(new THREE.IcosahedronGeometry(1, 1), M.foliage, 1400), cones = Inst(new THREE.ConeGeometry(1, 2.6, 7), M.needle, 500), d = new THREE.Object3D();
  let k = 0, c = 0;
  while (k < 1400) {
    const a = R(0, 6.28), r = R(19, 95), x = PG.x + Math.cos(a) * r, z = PG.z + Math.sin(a) * r, y = hillH(x, z);
    if (y < 2.5) continue;
    const sc = R(1.3, 3); d.position.set(x, y + sc * 0.6, z); d.scale.set(sc, sc * R(1.1, 1.8), sc); d.rotation.y = R(0, 6); d.updateMatrix(); trees.setMatrixAt(k++, d.matrix);
    if (c < 500 && rand() < 0.4) { d.position.y += sc * 0.8; d.scale.set(sc * 0.6, sc * 1.4, sc * 0.6); d.updateMatrix(); cones.setMatrixAt(c++, d.matrix); }
  }
  cones.count = c; scene.add(trees, cones);
  // 盘山灯路
  const lamps = Inst(new THREE.SphereGeometry(0.3, 8, 6), M.lamp, 70);
  for (let i = 0; i < 70; i++) { const t = i / 70, a = 2.4 + t * 8.5, r = lerp(80, 20, t), x = PG.x + Math.cos(a) * r, z = PG.z + Math.sin(a) * r; d.position.set(x, hillH(x, z) + 1.4, z); d.scale.setScalar(1); d.rotation.set(0, 0, 0); d.updateMatrix(); lamps.setMatrixAt(i, d.matrix); }
  scene.add(lamps);
}
function leifeng(mg, cx, cy, cz) {
  const Bx = (a, b, c) => new THREE.BoxGeometry(a, b, c);
  const Cy = (r0, r1, hh, open = false) => new THREE.CylinderGeometry(r0, r1, hh, 8, 1, open, Math.PI / 8);
  const vtx = (r, k) => [Math.cos(Math.PI / 8 + k * Math.PI / 4) * r, Math.sin(Math.PI / 8 + k * Math.PI / 4) * r];
  const alongSides = (r, step, fn) => { for (let k = 0; k < 8; k++) { const [x0, z0] = vtx(r, k), [x1, z1] = vtx(r, k + 1), n = Math.max(1, Math.round(Math.hypot(x1 - x0, z1 - z0) / step)), th = (k + 1) * Math.PI / 4; for (let i = 0; i < n; i++) { const t = (i + 0.5) / n; fn(lerp(x0, x1, t), lerp(z0, z1, t), th); } } };
  const rail = (r, y, hgt = 1.05, skip = -1) => {
    for (let k = 0; k < 8; k++) {
      const [x0, z0] = vtx(r, k), [x1, z1] = vtx(r, k + 1), len = Math.hypot(x1 - x0, z1 - z0), th = (k + 1) * Math.PI / 4, ry = -(th + Math.PI / 2);
      mg.add(Bx(0.2, hgt + 0.25, 0.2), M.lacquer, [cx + x0, y + (hgt + 0.25) / 2, cz + z0]);
      if (k === skip) continue;
      const mx = cx + (x0 + x1) / 2, mz = cz + (z0 + z1) / 2;
      mg.add(Bx(len, 0.12, 0.22), M.lacquer, [mx, y + hgt, mz], [0, ry, 0]);
      mg.add(Bx(len, 0.08, 0.1), M.lacquer, [mx, y + 0.28, mz], [0, ry, 0]);
      const nb = Math.round(len / 0.42);
      for (let i = 1; i < nb; i++) { const t = i / nb; mg.add(Bx(0.06, hgt - 0.28, 0.06), M.lacquer, [cx + lerp(x0, x1, t), y + 0.28 + (hgt - 0.28) / 2, cz + lerp(z0, z1, t)]); }
    }
  };
  let y = cy;
  // 两层须弥座台基 + 石栏 + 面河石阶
  mg.add(Cy(19, 19.6, 2), M.stone, [cx, y + 1, cz]); mg.add(Cy(19.5, 19.5, 0.25), M.stoneStep, [cx, y + 2.1, cz]); rail(19.2, y + 2.22, 1.0, 3); y += 2.22;
  mg.add(Cy(15.5, 16, 1.8), M.stone, [cx, y + 0.9, cz]); mg.add(Cy(16, 16, 0.25), M.stoneStep, [cx, y + 1.9, cz]); rail(15.7, y + 2.02, 1.0, 3); y += 2.02;
  for (let i = 0; i < 9; i++) mg.add(Bx(0.75, 0.46, 6), M.stoneStep, [cx - 15.9 - i * 0.72, cy + 4.1 - i * 0.46 - 0.23, cz]);
  const RB = [9.6, 8.8, 8.0, 7.2, 6.4], HF = [7.4, 6.2, 5.8, 5.4, 5.0];
  for (let f = 0; f < 5; f++) {
    const rb = RB[f], hf = HF[f];
    if (f > 0) {
      mg.add(Cy(rb + 2.4, rb + 2.1, 0.5), M.wood, [cx, y + 0.25, cz]);
      alongSides(rb + 1.6, 0.85, (x, z, th) => { mg.add(Bx(0.3, 0.36, 0.9), M.bracket, [cx + x, y - 0.16, cz + z], [0, Math.PI / 2 - th, 0]); });
      rail(rb + 2.2, y + 0.5, 1.0); y += 0.5;
    }
    mg.add(Cy(rb * 0.96, rb * 0.96, hf), M.pagodaWall, [cx, y + hf / 2, cz]);
    for (let k = 0; k < 8; k++) {
      const [x0, z0] = vtx(rb, k), [x1, z1] = vtx(rb, k + 1), th = (k + 1) * Math.PI / 4, ox = Math.cos(th), oz = Math.sin(th), bl = Math.hypot(x1 - x0, z1 - z0) / 3 - 0.62, ry = Math.PI / 2 - th;
      for (let j = 0; j < 3; j++) {
        const px = lerp(x0, x1, j / 3), pz = lerp(z0, z1, j / 3);
        mg.add(new THREE.CylinderGeometry(0.3, 0.34, hf, 12), M.lacquer, [cx + px, y + hf / 2, cz + pz]);
        mg.add(new THREE.CylinderGeometry(0.46, 0.5, 0.3, 12), M.stoneStep, [cx + px, y + 0.15, cz + pz]);
        const tc = (j + 0.5) / 3, bx = cx + lerp(x0, x1, tc), bz = cz + lerp(z0, z1, tc), dh = hf - 2.0, dy = y + 0.35 + dh / 2;
        mg.add(new THREE.PlaneGeometry(bl, dh), M.lattice, [bx + ox * 0.16, dy, bz + oz * 0.16], [0, ry, 0]);
        mg.add(new THREE.PlaneGeometry(bl, dh), (j === 1 || (k + f) % 3 === 0) ? M.window : M.windowDim, [bx + ox * 0.05, dy, bz + oz * 0.05], [0, ry, 0]);
        mg.add(new THREE.PlaneGeometry(bl, 0.7), M.lattice, [bx + ox * 0.16, y + hf - 1.05, bz + oz * 0.16], [0, ry, 0]);
        mg.add(new THREE.PlaneGeometry(bl, 0.7), M.windowDim, [bx + ox * 0.05, y + hf - 1.05, bz + oz * 0.05], [0, ry, 0]);
        mg.add(Bx(bl + 0.1, 0.14, 0.12), M.lacquer, [bx + ox * 0.18, y + hf - 1.5, bz + oz * 0.18], [0, ry, 0]);
      }
    }
    // 额枋 + 两跳斗拱
    mg.add(Cy(rb + 0.2, rb + 0.2, 0.7, true), M.lacquer, [cx, y + hf - 0.35, cz]);
    mg.add(Cy(rb + 0.35, rb + 0.35, 0.32, true), M.bracket, [cx, y + hf + 0.16, cz]);
    alongSides(rb + 0.55, 0.7, (x, z, th) => { mg.add(Bx(0.34, 0.3, 1.1), M.wood, [cx + x, y + hf + 0.48, cz + z], [0, Math.PI / 2 - th, 0]); mg.add(Bx(0.9, 0.18, 0.34), M.bracket, [cx + x, y + hf + 0.72, cz + z], [0, Math.PI / 2 - th, 0]); });
    alongSides(rb + 1.2, 0.7, (x, z, th) => { mg.add(Bx(0.3, 0.28, 1.0), M.wood, [cx + x, y + hf + 0.95, cz + z], [0, Math.PI / 2 - th, 0]); mg.add(Bx(0.8, 0.16, 0.3), M.bracket, [cx + x, y + hf + 1.15, cz + z], [0, Math.PI / 2 - th, 0]); });
    y += hf + 1.25;
    if (f < 4) { const re = rb + 4.4; addRoof(mg, { type: 'poly', n: 8, r: re, h: 2.7, upturn: 1.35, pos: [cx, y, cz], ridge: 0.32, mat: M.copper, topScale: (RB[f + 1] + 1.1) / re, eaveLight: M.eaveLight }); y += 2.35; }
  }
  // 顶檐 + 塔刹
  addRoof(mg, { type: 'poly', n: 8, r: RB[4] + 4.6, h: 6.4, upturn: 1.7, pos: [cx, y, cz], ridge: 0.36, mat: M.copper, eaveLight: M.eaveLight, finial: false });
  y += 6.1;
  mg.add(new THREE.CylinderGeometry(1.3, 1.7, 1.2, 16), M.gold, [cx, y + 0.6, cz]);
  const lotus = []; for (let i = 0; i <= 10; i++) { const t = i / 10; lotus.push(new THREE.Vector2(0.9 + Math.sin(t * Math.PI * 0.6) * 1.1, t * 1.2)); }
  mg.add(new THREE.LatheGeometry(lotus, 24), M.gold, [cx, y + 1.2, cz]);
  mg.add(new THREE.CylinderGeometry(0.2, 0.26, 10, 10), M.gold, [cx, y + 7, cz]);
  for (let i = 0; i < 7; i++) mg.add(new THREE.TorusGeometry(1.1 - i * 0.08, 0.14, 8, 24), M.gold, [cx, y + 3.2 + i * 0.85, cz], [Math.PI / 2, 0, 0]);
  mg.add(new THREE.SphereGeometry(0.85, 16, 12), M.gold, [cx, y + 10.3, cz], [0, 0, 0], [1, 1.15, 1]);
  mg.add(new THREE.SphereGeometry(0.4, 12, 10), M.gold, [cx, y + 11.5, cz]);
  mg.add(new THREE.ConeGeometry(0.16, 1.6, 8), M.gold, [cx, y + 12.6, cz]);
  return y + 6;
}
{
  const mg = new Merger();
  PAGODA_TOP.y = leifeng(mg, PG.x, PG.y - 1.2, PG.z);
  scene.add(mg.build({ cast: false, receive: true }));
  pointLight(scene, PG.x - 30, PG.y + 16, PG.z + 12, 1100, 110);
  pointLight(scene, PG.x - 22, PG.y + 46, PG.z + 22, 700, 90);
}

// 石拱桥
{
  const L = 170, s = new THREE.Shape();
  s.moveTo(0, -3); s.lineTo(L, -3); s.lineTo(L, 8); s.quadraticCurveTo(L / 2, 20, 0, 8); s.lineTo(0, -3);
  const arches = [[20, 5], [48, 7], [85, 9], [122, 7], [150, 5]];
  for (const [cx, r] of arches) { const hp = new THREE.Path(); hp.moveTo(cx + r, -2.5); hp.lineTo(cx + r, 1); hp.absarc(cx, 1, r, 0, Math.PI, false); hp.lineTo(cx - r, -2.5); hp.lineTo(cx + r, -2.5); s.holes.push(hp); }
  const geo = new THREE.ExtrudeGeometry(s, { depth: 7, bevelEnabled: false, curveSegments: 28 }); geo.translate(0, 0, -3.5);
  const mg = new Merger(); mg.addM(geo, M.bridge, new THREE.Matrix4());
  const topY = (x) => { const t = x / L; return (1 - t) * (1 - t) * 8 + 2 * (1 - t) * t * 20 + t * t * 8; };
  for (let x = 2; x < L; x += 4) for (const z of [-3.3, 3.3]) {
    const y = topY(x); mg.add(new THREE.BoxGeometry(0.4, 1.3, 0.4), M.stoneStep, [x, y + 0.65, z]);
    const y2 = topY(x + 4); const a = Math.atan2(y2 - y, 4); mg.add(new THREE.BoxGeometry(4.1, 0.25, 0.25), M.stoneStep, [x + 2, (y + y2) / 2 + 1.2, z], [0, 0, a]);
    if ((x - 2) % 4 === 0) { const oz = Math.sign(z) * 0.9; mg.add(new THREE.BoxGeometry(0.1, 1.5, 0.1), M.wood, [x, y + 2.0, z]); mg.add(new THREE.BoxGeometry(0.07, 0.07, 1.0), M.wood, [x, y + 2.72, z + oz * 0.5]); lanternParts(mg, x, y + 1.75, z + oz, 1.55, M.paperRed); }
  }
  const cy = topY(L / 2);
  for (const [px, pz] of [[-2, -2.4], [2, -2.4], [-2, 2.4], [2, 2.4]]) mg.add(new THREE.CylinderGeometry(0.2, 0.2, 3.4, 8), M.lacquer, [L / 2 + px, cy + 1.7, pz]);
  addRoof(mg, { type: 'poly', n: 4, r: 4.4, h: 2.2, upturn: 0.8, pos: [L / 2, cy + 3.4, 0], ridge: 0.12 });
  lanternParts(mg, L / 2 - 2.5, cy + 2.6, 2.9, 0.8); lanternParts(mg, L / 2 + 2.5, cy + 2.6, 2.9, 0.8);
  const g = mg.build({ cast: false, receive: true }); g.position.set(-94, 0, -236); scene.add(g);
  for (const bx of [30, 85, 140]) pointLight(scene, -94 + bx, topY(bx) + 4, -236 + 6, 160, 50, 0xff6a40);
}

/* =========================================================================
   雾（薄雾浮水）
   ========================================================================= */
const mists = [];
{
  for (let i = 0; i < 34; i++) {
    const w = R(90, 190), h = R(10, 22), op = R(0.1, 0.24) * 0.35;
    // 先把随机数抽完再决定造不造面片，低档减面时下游场景的随机细节不受影响
    const seed = R(0, 100), px = R(-200, 240), pz = R(-330, -45);
    if (i >= TUNE.mistCount) continue;
    const u = uv();
    const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, fog: false });
    const n = mx_fractal_noise_float(vec3(u.x.mul(3).add(time.mul(0.01)).add(seed), u.y.mul(0.9), time.mul(0.03)), 3, 2, 0.5).mul(0.6).add(0.5);
    const mask = smoothstep(0.0, 0.25, u.x).mul(float(1).sub(smoothstep(0.75, 1.0, u.x))).mul(smoothstep(0.0, 0.3, u.y)).mul(float(1).sub(smoothstep(0.3, 1.0, u.y)));
    m.colorNode = color(0x7385ad);
    m.opacityNode = smoothstep(0.3, 0.8, n).mul(mask).mul(op);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), m);
    mesh.position.set(px, h * 0.38 - 1, pz); mesh.renderOrder = 2;
    scene.add(mesh); mists.push(mesh);
  }
  // 贴水薄雾
  const lo = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, fog: false });
  const wp = positionWorld.xz.mul(0.02);
  const n = mx_fractal_noise_float(vec3(wp.add(vec2(time.mul(0.03), 0)), time.mul(0.05)), 3, 2, 0.5).mul(0.6).add(0.5);
  lo.colorNode = color(0x8193ba);
  lo.opacityNode = smoothstep(0.35, 0.9, n).mul(0.22);
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(1400, 1400), lo); plane.rotation.x = -Math.PI / 2; plane.position.set(0, 0.35, -300); plane.renderOrder = 1;
  scene.add(plane);
}

/* =========================================================================
   扁舟与东坡
   ========================================================================= */
phase('远山 · 宝塔 · 石桥');
loadMsg('一叶扁舟 · 把酒问天');
await nextFrame();
const boat = new THREE.Group();
const poet = { root: new THREE.Group(), bags: [] };
{
  /* ---------- 船材 ---------- */
  const plankTex = canvasTex(512, 256, (g, w, h) => {
    const n = tileNoise(w, h, 5, 4, 61), n2 = tileNoise(w, h, 40, 2, 63);
    pixels(g, w, h, (x, y) => { const row = Math.floor(y / 32), gr = Math.sin(x * 0.08 + n(x, y) * 7 + row * 3.1) * 0.5 + 0.5; let k = 0.7 + gr * 0.2 + n(x, y) * 0.22 + n2(x, y) * 0.08 + ((row * 37) % 7) * 0.025; if (y % 32 < 2) k *= 0.35; if (x % 170 < 2) k *= 0.5; return [100 * k, 68 * k, 44 * k]; });
    g.fillStyle = 'rgba(20,14,10,.8)'; for (let r = 0; r < 8; r++) for (let x = 20 + (r % 2) * 40; x < w; x += 85) { g.beginPath(); g.arc(x, r * 32 + 16, 1.6, 0, 7); g.fill(); }
  });
  const plankN = canvasTex(512, 256, (g, w, h) => { pixels(g, w, h, (x, y) => { const m = y % 32; const dy = m < 2 ? -0.7 : (m > 29 ? 0.45 : 0); return [128, (dy * 0.5 + 0.5) * 255, 235]; }); }, { srgb: false });
  const hullMat = new THREE.MeshStandardNodeMaterial({ map: plankTex, normalMap: plankN, roughness: 0.62, color: 0xd8c0a6 });
  const hullIn = new THREE.MeshStandardNodeMaterial({ map: plankTex, normalMap: plankN, roughness: 0.8, color: 0x8a7462, side: THREE.BackSide });
  const darkWood = STD({ map: TX.wood, color: 0x9a8474, roughness: 0.7 });
  const darkLacquer = new THREE.MeshPhysicalNodeMaterial({ color: 0x24110d, roughness: 0.35, clearcoat: 0.9, clearcoatRoughness: 0.15 });
  const celadon = new THREE.MeshPhysicalNodeMaterial({ color: 0x9fc2ad, roughness: 0.18, clearcoat: 1, clearcoatRoughness: 0.08 });
  const rope = STD({ color: 0x8c7552, roughness: 0.95 });
  const bamboo = STD({ color: 0x7d7a48, roughness: 0.55 });
  const mg = new Merger();
  const stick = (a, b, r, mat, seg = 8) => { const d = b.clone().sub(a), len = d.length(); mg.addM(new THREE.CylinderGeometry(r, r, len, seg), mat, new THREE.Matrix4().compose(a.clone().add(b).multiplyScalar(0.5), new THREE.Quaternion().setFromUnitVectors(V(0, 1, 0), d.normalize()), V(1, 1, 1))); };

  /* ---------- 船身：放样壳体 + 板缝 + 船肋 ---------- */
  const L = 7.4, W = 1.8, N = 72, MS = 24;
  const hw = (u) => u < 0 ? W / 2 * Math.pow(Math.max(0, 1 - Math.pow(-u, 2.1)), 0.6) + 0.015 : (W / 2 * Math.pow(Math.max(0, 1 - Math.pow(u, 4)), 0.5)) * 0.78 + W * 0.13;
  const top = (u) => 0.5 + 0.58 * Math.pow(Math.max(0, -u), 3) + 0.22 * Math.pow(Math.max(0, u), 3);
  const bot = (u) => -0.16 + 0.34 * Math.pow(Math.abs(u), 3);
  const sec = (u, s) => { const w = hw(u), yt = top(u), yb = bot(u); return [w * Math.sign(s) * Math.pow(Math.abs(s), 0.75), yb + (yt - yb) * Math.pow(Math.abs(s), 2.0)]; };
  const zOf = (u) => u * L / 2;
  {
    const pos = [], uvs = [], idx = [];
    for (let i = 0; i <= N; i++) { const u = -1 + 2 * i / N; for (let j = 0; j <= MS; j++) { const s = -1 + 2 * j / MS, [x, y] = sec(u, s); pos.push(x, y, zOf(u)); uvs.push(zOf(u) / L * 3, j / MS); } }
    for (let i = 0; i < N; i++) for (let j = 0; j < MS; j++) { const a = i * (MS + 1) + j, b = a + 1, c = a + MS + 1, d = c + 1; idx.push(a, c, b, b, c, d); }
    const hull = new THREE.BufferGeometry(); hull.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); hull.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2)); hull.setIndex(idx); hull.computeVertexNormals();
    const mid = (N / 2) * (MS + 1) + MS / 2;
    if (hull.attributes.normal.getY(mid) > 0) { const ix = hull.index.array; for (let k = 0; k < ix.length; k += 3) { const t = ix[k + 1]; ix[k + 1] = ix[k + 2]; ix[k + 2] = t; } hull.computeVertexNormals(); }
    mg.addM(hull, hullMat, new THREE.Matrix4()); mg.addM(hull, hullIn, new THREE.Matrix4());
    // 船尾封板
    const sh = new THREE.Shape(); for (let j = 0; j <= MS; j++) { const [x, y] = sec(1, -1 + 2 * j / MS); j ? sh.lineTo(x, y) : sh.moveTo(x, y); }
    const tr = new THREE.ShapeGeometry(sh); tr.translate(0, 0, L / 2 - 0.005); mg.addM(tr, hullMat, new THREE.Matrix4());
    const tr2 = tr.clone(); mg.addM(tr2, hullIn, new THREE.Matrix4());
  }
  for (const sd of [-1, 1]) {
    const rim = [], rub = [];
    for (let i = 0; i <= 40; i++) { const u = -0.99 + 1.99 * i / 40, [x, y] = sec(u, sd), [x2, y2] = sec(u, sd * 0.84); rim.push(V(x, y + 0.02, zOf(u))); rub.push(V(x2 * 1.03, y2, zOf(u))); }
    mg.addM(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(rim), 80, 0.045, 8), darkWood, new THREE.Matrix4());
    mg.addM(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(rub), 80, 0.028, 6), darkWood, new THREE.Matrix4());
  }
  for (let u = -0.7; u <= 0.9; u += 0.11) { const pts = []; for (let j = 0; j <= 14; j++) { const [x, y] = sec(u, -0.97 + 1.94 * j / 14); pts.push(V(x * 0.955, y + 0.02, zOf(u))); } mg.addM(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 24, 0.022, 5), darkWood, new THREE.Matrix4()); }
  for (let k = -2; k <= 2; k++) mg.add(new THREE.BoxGeometry(0.16, 0.025, 4.5), hullMat, [k * 0.185, 0.0, 0.3]);
  for (let z = -1.9; z <= 2.6; z += 0.9) mg.add(new THREE.BoxGeometry(0.9, 0.03, 0.06), darkWood, [0, -0.03, z]);
  // 船头甲板、船尾甲板、坐板
  { const u0 = -0.6, s = new THREE.Shape(); s.moveTo(-hw(u0) * 0.86, zOf(u0)); s.lineTo(hw(u0) * 0.86, zOf(u0)); s.lineTo(0.05, zOf(-0.97)); s.lineTo(-0.05, zOf(-0.97)); const g = new THREE.ExtrudeGeometry(s, { depth: 0.05, bevelEnabled: false }); g.rotateX(Math.PI / 2); mg.addM(g, hullMat, new THREE.Matrix4().makeTranslation(0, 0.44, 0)); }
  mg.add(new THREE.BoxGeometry(0.1, 0.16, 0.5), darkWood, [0, 1.02, zOf(-0.985)], [0.5, 0, 0]);
  mg.add(new THREE.BoxGeometry(hw(0.82) * 1.7, 0.05, 0.62), hullMat, [0, 0.5, zOf(0.91)]);
  for (const u of [0.1, 0.72]) mg.add(new THREE.BoxGeometry(hw(u) * 1.85, 0.05, 0.26), darkWood, [0, 0.36, zOf(u)]);
  /* ---------- 竹编乌篷（双节） ---------- */
  const cz = 1.7, clen = 1.9, cr = 0.74;
  const can = new THREE.CylinderGeometry(cr, cr, clen, 48, 8, true, -Math.PI / 2, Math.PI); can.rotateX(-Math.PI / 2);
  mg.add(can, M.canopy, [0, 0.2, cz]);
  const can2 = new THREE.CylinderGeometry(cr - 0.05, cr - 0.05, 0.75, 44, 3, true, -Math.PI / 2, Math.PI); can2.rotateX(-Math.PI / 2);
  mg.add(can2, M.canopy, [0, 0.2, cz - clen / 2 - 0.2]);
  for (let i = 0; i <= 5; i++) mg.add(new THREE.TorusGeometry(cr + 0.012, 0.024, 6, 32, Math.PI), darkWood, [0, 0.2, cz - clen / 2 + i * clen / 5]);
  mg.add(new THREE.TorusGeometry(cr - 0.035, 0.022, 6, 32, Math.PI), darkWood, [0, 0.2, cz - clen / 2 - 0.57]);
  for (const sx of [-1, 1]) mg.add(new THREE.BoxGeometry(0.05, 0.1, clen + 0.8), darkWood, [sx * cr, 0.22, cz - 0.38]);
  mg.add(new THREE.CylinderGeometry(0.05, 0.05, 1.3, 12), bamboo, [0, 0.86, cz - clen / 2 - 0.6], [0, 0, Math.PI / 2]);
  for (const sx of [-0.4, 0.4]) mg.add(new THREE.TorusGeometry(0.058, 0.008, 5, 12), rope, [sx, 0.86, cz - clen / 2 - 0.6], [0, Math.PI / 2, 0]);
  // 竹篙
  { const a = V(0.28, 0.96, -0.6), b = V(0.1, 0.94, 3.6); stick(a, b, 0.028, bamboo); for (let t = 0.05; t < 1; t += 0.09) mg.add(new THREE.TorusGeometry(0.03, 0.007, 5, 10), bamboo, [lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t)]); }
  /* ---------- 橹与缆绳 ---------- */
  { const a = V(0.28, 0.66, 3.35), b = V(1.05, -0.35, 5.5); stick(a, b, 0.038, darkWood); mg.addM(new THREE.BoxGeometry(0.03, 0.18, 0.95), darkWood, new THREE.Matrix4().compose(b.clone().add(V(0.12, -0.1, 0.35)), new THREE.Quaternion().setFromUnitVectors(V(0, 0, 1), b.clone().sub(a).normalize()), V(1, 1, 1)));
    stick(V(0.28, 0.66, 3.35), V(0.2, 0.5, 3.1), 0.05, darkWood); mg.addM(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([V(0.45, 0.45, 3.7), V(0.3, 0.2, 3.3), V(0.1, 0.5, 3.0)]), 20, 0.012, 5), rope, new THREE.Matrix4()); }
  for (let i = 0; i < 3; i++) mg.add(new THREE.TorusGeometry(0.11 - i * 0.01, 0.02, 6, 20), rope, [0.08, 0.47 + i * 0.032, zOf(-0.9)], [Math.PI / 2, 0, 0]);
  /* ---------- 船头方灯 ---------- */
  const LX = -0.45, LZ = -1.55, LY = 0.02;
  for (const [a, b] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) mg.add(new THREE.BoxGeometry(0.03, 0.5, 0.03), darkLacquer, [LX + a * 0.15, LY + 0.25, LZ + b * 0.15]);
  mg.add(new THREE.BoxGeometry(0.42, 0.05, 0.42), darkLacquer, [LX, LY + 0.52, LZ]); mg.add(new THREE.BoxGeometry(0.2, 0.05, 0.2), darkLacquer, [LX, LY + 0.57, LZ]);
  mg.add(new THREE.BoxGeometry(0.38, 0.05, 0.38), darkLacquer, [LX, LY + 0.02, LZ]);
  mg.add(new THREE.TorusGeometry(0.09, 0.012, 6, 14, Math.PI), darkLacquer, [LX, LY + 0.6, LZ]);
  mg.add(new THREE.BoxGeometry(0.28, 0.42, 0.28), M.paper, [LX, LY + 0.27, LZ]);
  // 篷口小灯
  lanternParts(mg, 0, 0.68, cz - clen / 2 - 0.5, 0.32);
  /* ---------- 案、酒壶、杯、古琴 ---------- */
  const tx = 0.42, tz = -1.75, ty = 0.2;
  mg.add(new THREE.BoxGeometry(0.34, 0.03, 0.56), darkLacquer, [tx, ty, tz]);
  for (const [a, b] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) mg.add(new THREE.BoxGeometry(0.03, ty, 0.03), darkLacquer, [tx + a * 0.14, ty / 2, tz + b * 0.24]);
  const ewer = [[0.001, 0], [0.045, 0], [0.065, 0.04], [0.075, 0.09], [0.06, 0.14], [0.026, 0.18], [0.02, 0.23], [0.03, 0.245], [0.001, 0.245]].map(([r, y]) => new THREE.Vector2(r, y));
  mg.add(new THREE.LatheGeometry(ewer, 24), celadon, [tx, ty + 0.015, tz - 0.1]);
  stick(V(tx, ty + 0.12, tz - 0.16), V(tx, ty + 0.19, tz - 0.24), 0.009, celadon);
  mg.add(new THREE.TorusGeometry(0.045, 0.008, 6, 14, Math.PI), celadon, [tx, ty + 0.13, tz - 0.03], [0, Math.PI / 2, -Math.PI / 2]);
  const cupP = [[0.001, 0], [0.018, 0], [0.03, 0.03], [0.034, 0.035], [0.028, 0.034], [0.001, 0.008]].map(([r, y]) => new THREE.Vector2(r, y));
  mg.add(new THREE.LatheGeometry(cupP, 16), celadon, [tx, ty + 0.015, tz + 0.12]);
  // 古琴
  { const qx = -0.5, qy = 0.07, qz = -0.75; mg.add(new THREE.BoxGeometry(0.19, 0.045, 1.2), darkLacquer, [qx, qy, qz]); mg.add(new THREE.BoxGeometry(0.15, 0.02, 1.16), darkLacquer, [qx, qy + 0.03, qz]);
    for (let k = 0; k < 7; k++) mg.add(new THREE.BoxGeometry(0.003, 0.003, 1.12), M.gold, [qx - 0.055 + k * 0.018, qy + 0.045, qz]);
    for (let k = 0; k < 13; k++) mg.add(new THREE.SphereGeometry(0.005, 6, 4), M.gold, [qx - 0.075, qy + 0.042, qz - 0.5 + k * 0.08]);
    mg.addM(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([V(qx, qy, qz - 0.6), V(qx + 0.05, qy - 0.02, qz - 0.66), V(qx + 0.07, qy - 0.05, qz - 0.62)]), 10, 0.008, 4), M.redCap, new THREE.Matrix4()); }
  const iron = STD({ color: 0x383c38, metalness: 0.7, roughness: 0.52 });
  for (const sd of [-1, 1]) {
    for (const f of [0.42, 0.60, 0.76, 0.91]) {
      const pts = [];
      for (let i = 0; i <= 72; i++) { const u = -0.97 + 1.94 * i / 72; const [x, y] = sec(u, sd * f); pts.push(V(x * 1.006, y, zOf(u))); }
      mg.addM(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 96, 0.005, 4), darkWood, new THREE.Matrix4());
    }
    for (let u = -0.8; u < 0.92; u += 0.13) { const [x, y] = sec(u, sd * 0.93); mg.add(new THREE.SphereGeometry(0.009, 8, 6), iron, [x * 1.012, y, zOf(u)]); }
  }
  for (let j = 1; j < 23; j++) {
    const a = Math.PI * j / 23;
    stick(V(Math.cos(a) * (cr + 0.007), 0.2 + Math.sin(a) * (cr + 0.007), cz - clen / 2), V(Math.cos(a) * (cr + 0.007), 0.2 + Math.sin(a) * (cr + 0.007), cz + clen / 2), 0.006, bamboo, 5);
  }
  boat.add(mg.build());
  pointLight(boat, LX, LY + 0.3, LZ, 7, 13);
  pointLight(boat, 0, 0.6, cz - clen / 2 - 0.5, 2.5, 5);

  /* ---------- 东坡：外部模型 assets/sushi.glb，立于船头，举杯问天 ----------
     模型为单网格静模（2210 顶点、一张烘焙贴图），高 1 个单位（含举起的手），面朝 +Z。
     这里统一约定人物面朝 -Z（PF 即脸的方向），故绕 Y 转半圈。 */
  const P = poet.root;
  const gltf = await new GLTFLoader().loadAsync(sushiUrl);
  let src = null; gltf.scene.traverse((o) => { if (!src && o.isMesh) src = o; });
  const map = src.material.map; map.anisotropy = 8;
  const robe = new THREE.MeshPhysicalNodeMaterial({ map, roughness: 0.78, sheen: 0.55, sheenRoughness: 0.5, sheenColor: new THREE.Color(0xdfe6ff), side: THREE.DoubleSide });
  robe.emissiveNode = texture(map).rgb.mul(0.07);   // 夜里托住衣袍暗部，背影不至于糊成一团
  // 衣摆随风：只动膝下到脚面上一段（模型空间高度 0~1），脚底不滑动
  const yN = positionLocal.y;
  const hem = smoothstep(0.03, 0.12, yN).mul(float(1).sub(smoothstep(0.12, 0.5, yN)));
  robe.positionNode = positionLocal.add(vec3(sin(time.mul(1.4).add(yN.mul(9.0)).add(positionLocal.z.mul(6.0))).mul(0.012), 0, cos(time.mul(1.1).add(yN.mul(7.0))).mul(0.016)).mul(hem));
  const SUSHI_H = 1.85;                                // 头顶约 0.95 → 身高约 1.76m
  const body = new THREE.Mesh(src.geometry, robe);
  body.scale.setScalar(SUSHI_H); body.rotation.y = Math.PI;
  body.castShadow = true; body.receiveShadow = true;
  P.add(body); poet.model = body;
  poet.hand = V(0.22, 0.99, -0.14).multiplyScalar(SUSHI_H);   // 举杯之手在 P 局部坐标中的位置
  P.position.set(0, 0.44, -2.45);                      // 船头甲板面 y=0.44
  boat.add(P);
  boat.traverse((o) => { if (o.isMesh) o.receiveShadow = true; });
}
scene.add(boat);

// 舟行轨迹：绕水面缓行一周
const BOAT_C = V(14, 0, -6), BOAT_R = [12, 20];
const boatAt = (t, out = V()) => { const a = Math.PI * 0.64 + t * (Math.PI * 2 / 320); return out.set(BOAT_C.x + BOAT_R[0] * Math.cos(a), 0, BOAT_C.z + BOAT_R[1] * Math.sin(a)); };
const boatDir = (t) => { const a = Math.PI * 0.64 + t * (Math.PI * 2 / 320); return V(-BOAT_R[0] * Math.sin(a), 0, BOAT_R[1] * Math.cos(a)).normalize(); };

/* =========================================================================
   花瓣飘零 · 河灯 · 孔明灯
   ========================================================================= */
const PETALS = 800, CAM_PETALS = 240;
const petals = { mesh: null, s: [] };
const _fw = V(), _rt = V();
{
  const pm = new THREE.MeshStandardNodeMaterial({ map: TX.petal, alphaTest: 0.4, side: THREE.DoubleSide, roughness: 0.8 });
  pm.emissiveNode = texture(TX.petal).rgb.mul(0.55);
  petals.mesh = Inst(new THREE.PlaneGeometry(0.17, 0.17), pm, PETALS);
  petals.mesh.frustumCulled = false;
  const spawn = (s, init) => {
    s.v = V(R(0.15, 0.5), -R(0.3, 0.6), R(-0.2, 0.1)); s.r = V(R(0, 6), R(0, 6), R(0, 6)); s.w = V(R(-2, 2), R(-2, 2), R(-2, 2));
    s.float = 0; s.ph = R(0, 6.28);
    if (s.cam) {
      camera.getWorldDirection(_fw); _fw.y = 0; _fw.normalize(); _rt.set(-_fw.z, 0, _fw.x);
      s.p = camera.position.clone().addScaledVector(_fw, R(2.5, 30)).addScaledVector(_rt, R(-14, 14)).add(V(0, init ? R(-3, 9) : R(4, 10), 0));
      return;
    }
    const tp = blossom.tips[(rand() * blossom.tips.length) | 0];
    s.p = tp.clone().add(V(R(-1, 1), R(-0.5, 0.5), R(-1, 1)));
    if (init === 'water') { const z = R(-60, 40); s.p.set(R(shoreX(z) + 1, rightX(z) - 1), 0.03, z); s.float = R(0, 40); }
    else if (init) s.p.y -= R(0, 12);
  };
  for (let i = 0; i < PETALS; i++) { const s = { cam: i < CAM_PETALS }; spawn(s, i < CAM_PETALS ? true : (i < CAM_PETALS + 260 ? 'water' : true)); petals.s.push(s); }
  petals.spawn = spawn;
  scene.add(petals.mesh);
}
const dummy = new THREE.Object3D();
function updatePetals(dt, t, wind) {
  const { s, mesh, spawn } = petals;
  const cp = camera.position;
  for (let i = 0; i < s.length; i++) {
    const q = s[i];
    const onLand = q.p.x < shoreX(q.p.z) + 0.5 || q.p.x > rightX(q.p.z) - 0.5;
    if (q.float > 0) {
      if (onLand) { spawn(q, false); continue; }
      q.float += dt; q.p.y = 0.03 + Math.sin(t * 1.3 + q.ph) * 0.01;
      q.p.x += Math.sin(q.ph + t * 0.1) * 0.06 * dt; q.p.z += (-0.1 + Math.cos(q.ph + t * 0.13) * 0.05) * dt;
      dummy.rotation.set(-Math.PI / 2, 0, q.ph + t * 0.05);
      if (q.float > 70 || (q.cam && q.p.distanceToSquared(cp) > 3600)) { spawn(q, false); continue; }
    } else {
      q.p.x += (q.v.x * (1 + wind * 5) + Math.sin(t * 1.7 + q.ph) * 0.35) * dt;
      q.p.y += (q.v.y * (1 + wind * 0.5)) * dt;
      q.p.z += (q.v.z + Math.cos(t * 1.3 + q.ph) * 0.3) * dt;
      if (q.p.y < 1.5 && onLand) { spawn(q, false); continue; }
      if (q.cam && (q.p.y < cp.y - 12 || q.p.distanceToSquared(cp) > 2500)) { spawn(q, false); continue; }
      q.r.addScaledVector(q.w, dt); dummy.rotation.set(q.r.x, q.r.y, q.r.z);
      if (q.p.y <= 0.03) { q.p.y = 0.03; q.float = 0.001; }
    }
    dummy.position.copy(q.p); dummy.scale.setScalar(1); dummy.updateMatrix(); mesh.setMatrixAt(i, dummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
}

// 水上宫灯
const RIVER = 52;
const river = { s: [] };
{
  const fm = new Merger(), Bx = (a, b, c) => new THREE.BoxGeometry(a, b, c);
  for (const [a, b] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) fm.add(Bx(0.035, 0.5, 0.035), M.wood, [a * 0.15, 0.25, b * 0.15]);
  fm.add(Bx(0.42, 0.045, 0.42), M.wood, [0, 0.52, 0]); fm.add(Bx(0.38, 0.05, 0.38), M.wood, [0, 0.02, 0]); fm.add(Bx(0.16, 0.06, 0.16), M.wood, [0, 0.57, 0]);
  const frameGeo = mergeGeometries(fm.map.get(M.wood));
  const gm = new THREE.MeshBasicNodeMaterial(); gm.colorNode = texture(TX.paper).rgb.mul(vec3(3.0, 1.75, 0.8));
  river.frame = Inst(frameGeo, M.wood, RIVER); river.glow = Inst(new THREE.BoxGeometry(0.28, 0.44, 0.28), gm, RIVER);
  river.frame.frustumCulled = river.glow.frustumCulled = false;
  const spawn = (s, z) => { s.z = z; s.x = R(shoreX(z) + 2.5, rightX(z) - 2.5); s.v = R(0.12, 0.35); s.ph = R(0, 6.3); s.sc = R(0.9, 1.35); };
  for (let i = 0; i < RIVER; i++) { const s = {}; spawn(s, R(-200, 46)); river.s.push(s); }
  river.spawn = spawn;
  scene.add(river.frame, river.glow);
}
function updateRiver(dt, t) {
  for (let i = 0; i < RIVER; i++) {
    const s = river.s[i];
    s.z -= s.v * dt; s.x += Math.sin(t * 0.15 + s.ph) * 0.04 * dt;
    if (s.z < -200) river.spawn(s, 46);
    const dx = s.x - B.x, dz = s.z - B.z, dd = dx * dx + dz * dz; if (dd < 9) { const k = (3 - Math.sqrt(dd)) * 0.5 * dt; s.x += dx * k; s.z += dz * k; }
    dummy.position.set(s.x, Math.sin(t * 1.4 + s.ph) * 0.02 - 0.03, s.z);
    dummy.rotation.set(Math.sin(t * 0.9 + s.ph) * 0.04, s.ph, Math.cos(t * 0.8 + s.ph) * 0.04); dummy.scale.setScalar(s.sc); dummy.updateMatrix();
    river.frame.setMatrixAt(i, dummy.matrix); river.glow.setMatrixAt(i, dummy.matrix);
  }
  river.frame.instanceMatrix.needsUpdate = river.glow.instanceMatrix.needsUpdate = true;
}

// 孔明灯：满河放灯 + 东坡手中一盏
const SKYL = 150;
const skyl = { s: [], t: -1 };
const lanGeo = new THREE.CylinderGeometry(0.52, 0.38, 1.1, 4, 6, true, Math.PI / 4);
const makeLanMat = (instanced) => {
  const m = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide });
  const vy = uv().y.clamp(0.0, 1.0);
  const seed = floor(positionWorld.x.mul(0.4)).add(floor(positionWorld.z.mul(0.4)).mul(1.7));
  const fl = sin(time.mul(7.0).add(seed)).mul(0.07).add(0.93);
  const rib = texture(TX.paper).g.mul(0.35).add(0.75);
  m.colorNode = mix(vec3(3.3, 1.5, 0.42), vec3(0.95, 0.28, 0.08), pow(vy, 0.8)).mul(fl).mul(rib);
  return m;
};
{
  skyl.mesh = Inst(lanGeo, makeLanMat(true), SKYL); skyl.mesh.frustumCulled = false;
  for (let i = 0; i < SKYL; i++) {
    const kind = i < 85 ? 'river' : i < 125 ? 'bank' : 'far';
    let x, z, y0 = 0.6;
    if (kind === 'river') { z = R(-190, 40); x = R(shoreX(z) + 3, rightX(z) - 3); }
    else if (kind === 'bank') { z = R(-150, 40); x = rand() < 0.5 ? shoreX(z) - R(3, 12) : rightX(z) + R(3, 12); y0 = R(3, 9); }
    else { x = R(-400, 400); z = R(-650, -230); y0 = R(5, 40); }
    const dd = Math.hypot(x - 8, z - 5);
    skyl.s.push({ x, z, y0, delay: kind === 'far' ? R(0, 10) : dd * 0.045 + R(0, 5), vy: R(1.0, 1.8), ph: R(0, 6.3), sc: kind === 'far' ? R(2.4, 4) : R(0.9, 1.3) });
  }
  scene.add(skyl.mesh);
}
const myLan = new THREE.Group();
{
  const body = new THREE.Mesh(lanGeo, makeLanMat(false)); myLan.add(body);
  myLan.scale.setScalar(1.25); myLan.visible = false;
  const l = new THREE.PointLight(0xffa050, 5, 12, 2); l.position.y = -0.3; myLan.add(l);
  scene.add(myLan);
}
const myLanState = { released: false, p: V(), v: 0 };
function updateSkyLanterns(dt, t, active, shotU, lanternShot) {
  if (active) { if (skyl.t < 0) skyl.t = 0; skyl.t += dt; } else if (skyl.t >= 0 && !active) { skyl.t += dt; if (skyl.t > 90) skyl.t = -1; }
  const lt = skyl.t;
  for (let i = 0; i < SKYL; i++) {
    const s = skyl.s[i], a = lt - s.delay;
    if (lt < 0 || a < 0) { dummy.scale.setScalar(0.0001); dummy.position.set(s.x, -5, s.z); }
    else {
      const y = s.y0 + a * s.vy + a * a * 0.012;
      dummy.position.set(s.x + Math.sin(t * 0.4 + s.ph) * 0.8 + a * 0.35, y, s.z - a * 0.5);
      dummy.rotation.set(Math.sin(t * 0.9 + s.ph) * 0.05, s.ph, Math.cos(t * 0.8 + s.ph) * 0.05);
      dummy.scale.setScalar(y > 320 ? 0.0001 : s.sc * Math.min(1, a * 0.8));
    }
    dummy.updateMatrix(); skyl.mesh.setMatrixAt(i, dummy.matrix);
  }
  skyl.mesh.instanceMatrix.needsUpdate = true;
  // 东坡手中灯：前 22% 捧于手，其后放飞
  if (lanternShot) {
    myLan.visible = true;
    const hold = poet.root.localToWorld(poet.hand.clone()).add(V(0, 0.8 + Math.sin(t * 2) * 0.03, 0));
    if (shotU < 0.22) { myLanState.released = false; myLanState.p.copy(hold); myLanState.v = 0; }
    else { if (!myLanState.released) { myLanState.released = true; } myLanState.v = Math.min(1.15, myLanState.v + dt * 0.5); myLanState.p.y += myLanState.v * dt; myLanState.p.addScaledVector(PF, dt * 0.35); myLanState.p.x += Math.sin(t * 0.7) * 0.15 * dt; }
    myLan.position.copy(myLanState.p);
  } else if (myLan.visible) { myLanState.p.y += 1.1 * dt; myLan.position.copy(myLanState.p); if (myLanState.p.y > 200) myLan.visible = false; }
}

/* =========================================================================
   后期：辉光 · 暗角 · 冷调 · 胶片颗粒
   ========================================================================= */
const post = new THREE.PostProcessing(renderer);
{
  const sp = pass(scene, camera, { samples: TUNE.samples });
  const col0 = sp.getTextureNode('output');
  /* 高度雾（后期一次全屏计算，代替堆叠透明雾片）
     用深度重建每个像素的世界坐标，沿视线对「随高度指数衰减」的雾密度做解析积分：
       ρ(y) = a·e^(−b·y)      ∫ρ ds = a·(e^(−b·y0) − e^(−b·y1)) / (b·dy)
     两层：一层铺满河谷的薄霭（b 小，山脚被吞、峰顶露出），一层贴水的浓雾（b 大，只在水面上几米）。
     雾色随视线与月亮的夹角加入前向散射——朝月看雾会亮起一片冷白的月晕。 */
  const depth = sp.getTextureNode('depth').r;
  const uProjInv = uniform(camera.projectionMatrixInverse), uCamWorld = uniform(camera.matrixWorld), uCamPos = uniform(camera.position);
  const col = Fn(() => {
    const vp = getViewPosition(screenUV, depth, uProjInv);
    const wp = uCamWorld.mul(vec4(vp, 1.0)).xyz;
    const sky = step(0.999999, depth);
    const rv = wp.sub(uCamPos);
    const L = mix(length(rv), float(2600.0), sky);
    const rd = normalize(rv);
    const y0 = uCamPos.y, y1 = max(y0.add(rd.y.mul(L)), -2.0);
    const layer = (a, b) => {
      const e0 = exp(y0.mul(b).negate()), e1 = exp(y1.mul(b).negate()), k = rd.y.mul(b).mul(L);
      return select(k.abs().lessThan(1e-3), e0.mul(L), e0.sub(e1).div(rd.y.mul(b))).mul(a);
    };
    // 雾团：在命中点取一次低频噪声，缓慢漂移，让雾有浓淡而非一层均匀的灰
    const drift = mx_noise_float(vec3(wp.xz.mul(0.012).add(vec2(time.mul(0.012), time.mul(-0.006))), time.mul(0.02))).mul(0.45).add(1.0);
    const amt = layer(float(0.0011), float(0.075)).add(layer(float(0.009), float(0.6)).mul(drift)).mul(mix(drift, float(1.0), sky));
    const f = float(1).sub(exp(amt.negate())).mul(mix(float(1.0), float(0.35), sky));
    const md = max(dot(rd, uMoonDir), 0.0);
    const fogC = mix(color(0x243650), color(0x5d6f96), smoothstep(-0.05, 0.25, rd.y).oneMinus().mul(0.4))
      .add(color(0xc9d4f0).mul(pow(md, 8.0).mul(0.12).add(pow(md, 60.0).mul(0.1))));
    return mix(col0.rgb, fogC, f);
  })();
  const colOut = FOG_OFF ? col0.rgb : col;
  const bl = bloom(colOut, 0.28, 0.55, 0.95);
  const v = length(screenUV.sub(0.5).mul(vec2(1.1, 1.0)));
  const vign = float(1).sub(smoothstep(0.35, 0.95, v).mul(0.32));
  const c0 = colOut.add(bl.rgb);
  const lum0 = dot(c0, vec3(0.299, 0.587, 0.114));
  const c = c0.mul(mix(vec3(0.91, 1.02, 1.06), vec3(1.08, 1.01, 0.90), smoothstep(0.04, 0.9, lum0)));
  const lum = dot(c, vec3(0.299, 0.587, 0.114));
  const coldC = mix(vec3(lum), c, 0.55).mul(vec3(0.82, 0.95, 1.3));
  const graded = mix(c, coldC, uCold);
  const grain = mx_noise_float(vec3(screenUV.mul(vec2(900, 520)), time.mul(24))).mul(0.004);
  post.outputNode = vec4(graded.mul(vign).add(grain), 1.0);
}

/* =========================================================================
   镜头脚本：一句一镜
   ========================================================================= */
const B = V(), F = V(), S = V(), PF = V(0, 0, -1), PB = V();
// 镜头运动量：反射、阴影是否可以跳帧复用，由它决定
const camMotion = { moved: true, resized: true, p: V(), q: new THREE.Quaternion() };
function trackCamera() {
  const dp = camMotion.p.distanceTo(camera.position), dq = 1 - Math.abs(camMotion.q.dot(camera.quaternion));
  camMotion.moved = dp > 0.02 || dq > 2e-7;
  camMotion.p.copy(camera.position); camMotion.q.copy(camera.quaternion);
}
const L3 = (a, b, u) => a.clone().lerp(b, u);
const moonAim = (from, d = 600) => from.clone().addScaledVector(MOON_DIR, d);
// 始终从东坡身后取景：d 为后退距离，side 为侧移，up 为高度
const behind = (d, side = 0, up = 2) => PB.clone().addScaledVector(PF, -d).addScaledVector(V(-PF.z, 0, PF.x), side).add(V(0, up, 0));
const ahead = (d, up = 1) => PB.clone().addScaledVector(PF, d).add(V(0, up, 0));
const lanPos = () => myLanState.p.clone();
const SHOTS = [
  { title: true, text: '水调歌头', pre: '丙辰中秋，欢饮达旦，大醉，作此篇，兼怀子由。', dur: 15, pose: 'idle',
    cam: (u) => ({ p: L3(V(-6, 50, 165), behind(15, 2.5, 4.2), u), t: L3(V(10, 59, -220), V(12, 39, -150), u), fov: 50 }) },
  { text: '明月几时有？把酒问青天。', dur: 13, pose: 'toast',
    cam: (u) => ({ p: L3(behind(15, 2.5, 4.2), behind(4.4, 1.0, 1.15), u), t: L3(ahead(4, 1.3), moonAim(PB.clone().add(V(0, 1.5, 0)), 14), ease(clamp01(u * 1.4 - 0.25))), fov: lerp(46, 42, u) }) },
  { text: '不知天上宫阙，今夕是何年。', dur: 13, pose: 'look',
    cam: (u) => ({ p: L3(behind(4.4, 1.0, 1.15), behind(7, -1.5, 1.0), u), t: MOON_POS.clone().add(V(lerp(-40, 30, u), lerp(-60, 0, u), 0)), fov: lerp(30, 12, u) }) },
  { text: '我欲乘风归去，', dur: 12, pose: 'wind', wind: 1,
    cam: (u) => ({ p: L3(behind(7, 2, 2.4), V(PG.x - 80, 70, PG.z + 75), ease(u)), t: L3(ahead(6, 3), PAGODA_TOP.clone().add(V(0, -18, 0)), ease(u)), fov: lerp(46, 50, u) }) },
  { text: '又恐琼楼玉宇，高处不胜寒。', dur: 15, pose: 'look', cold: 1, sea: 1,
    cam: (u) => { const a = 2.35 - u * 0.9, r = lerp(70, 46, u); return { p: V(PG.x + Math.cos(a) * r, lerp(92, 84, u), PG.z + Math.sin(a) * r), t: PAGODA_TOP.clone().add(V(0, lerp(-16, -22, u), 0)), fov: 44 }; } },
  { text: '起舞弄清影，何似在人间。', dur: 17, pose: 'dance',
    // 取景规矩：东坡的机位一律落在他的背面半球。
    // 上一镜在雷峰塔（他正面那一侧），所以这里起幅放在他身后 20 米，
    // 让整段飞行都在他背后完成，再缓缓推近、沿后肩横移半程。
    cam: (u) => ({ p: behind(lerp(20, 4.6, u), lerp(-7, -3.2, u), lerp(5.5, 2.0, u)), t: ahead(1.2, 1.0), fov: lerp(34, 42, u) }) },
  { text: '转朱阁，低绮户，照无眠。', dur: 15, pose: 'idle',
    cam: (u) => ({ p: L3(V(-10, 6, 34), V(-18, 5.4, 6), u), t: L3(V(-30, 6.4, 30), V(-33, 5, 14), u), fov: 40 }) },
  { text: '不应有恨，何事长向别时圆？', dur: 14, pose: 'look',
    cam: (u) => { const p = L3(V(-24, 4.8, 24), V(-33.5, 4.7, 24), ease(u)); return { p, t: p.clone().addScaledVector(MOON_DIR, 60).add(V(0, -7, 0)), fov: 48 }; } },
  { text: '人有悲欢离合，月有阴晴圆缺，此事古难全。', dur: 17, pose: 'look', pass: true,
    cam: (u) => ({ p: L3(V(-10, 5, -30), V(12, 6, -70), u), t: MOON_POS.clone().add(V(0, -40, 0)), fov: lerp(34, 24, u) }) },
  { text: '但愿人长久，', dur: 17, pose: 'release', sky: true, lantern: true,
    cam: (u) => { const lp = lanPos(); const side = V(-PF.z, 0, PF.x); const p = lp.clone().addScaledVector(PF, -lerp(6.5, 11, u)).addScaledVector(side, lerp(2.4, 3, u)).add(V(0, -lerp(0.5, 4, u), 0)); p.y = Math.max(p.y, 1.4); return { p, t: lp.clone().addScaledVector(PF, 1.5).add(V(0, lerp(-0.7, 2.5, u), 0)), fov: lerp(44, 50, u) }; } },
  { text: '千里共婵娟。', dur: 20, pose: 'look', sky: true,
    cam: (u) => ({ p: L3(lanPos().addScaledVector(PF, -11).addScaledVector(V(-PF.z, 0, PF.x), 3).add(V(0, -4, 0)), V(-14, 62, 170), ease(u)), t: L3(lanPos().addScaledVector(PF, 3).add(V(0, 2, 0)), V(14, 65, -240), ease(u)), fov: lerp(50, 52, u) }) },
];

/* =========================================================================
   界面
   ========================================================================= */
const verseEl = $('#verse');
const ticks = $('#ticks');
SHOTS.forEach((s, i) => {
  const b = document.createElement('button'); b.className = 'tick';
  b.innerHTML = `<span>${s.title ? '题 · 水调歌头' : s.text.split(/[，。？]/)[0]}</span>`;
  b.onclick = () => { exitFree(); goto(i); };
  ticks.appendChild(b);
});
function renderVerse(shot) {
  $("#chapter").textContent = shot.title ? "序 · 江上月" : `${String(SHOTS.indexOf(shot)).padStart(2, "0")} / 10 · ${shot.text.split(/[，。？]/)[0]}`;
  const cols = [];
  if (shot.title) {
    cols.push({ cls: 'title', text: shot.text });
    cols.push({ cls: 'small', text: shot.pre.replace(/[，。]/g, ' ') });
  } else {
    shot.text.split(/[，。？]/).filter(Boolean).forEach((t) => cols.push({ cls: '', text: t }));
  }
  let delay = 0.35;
  verseEl.classList.remove('out');
  verseEl.innerHTML = cols.map((c) => `<div class="col ${c.cls}">${[...c.text].map((ch) => { const d = delay; delay += ch === ' ' ? 0.3 : (c.cls === 'small' ? 0.05 : 0.16); return ch === ' ' ? '<span class="ch">&nbsp;</span>' : `<span class="ch" style="animation-delay:${d.toFixed(2)}s">${ch}</span>`; }).join('')}</div>`).join('')
    + (shot.title ? `<div class="col author"><span class="ch" style="animation-delay:${(delay + 0.2).toFixed(2)}s">宋 · 苏轼</span></div>` : '');
}
function markTicks() { [...ticks.children].forEach((t, i) => { t.classList.toggle('on', i === dir.idx); t.classList.toggle('done', i < dir.idx); }); }

/* =========================================================================
   导演：自动播放 / 自由观赏 / 镜头跟随
   ========================================================================= */
const controls = new OrbitControls(camera, canvas);
controls.enabled = false; controls.enableDamping = true; controls.dampingFactor = 0.06; controls.rotateSpeed = 0.55;
controls.minDistance = 1.5; controls.maxDistance = 600; controls.maxPolarAngle = Math.PI * 0.495;

const dir = { idx: 0, t: 0, playing: true, free: false, blend: null, look: V(0, 17, -120), fov: 50 };
// 一镜到底：切换时从当前机位平滑飞向下一句的机位（带弧线抬升），无黑场硬切
function goto(i) {
  i = (i + SHOTS.length) % SHOTS.length;
  const d = V(); camera.getWorldDirection(d);
  dir.blend = { p: camera.position.clone(), d, fov: camera.fov, k: 0, dur: null };
  dir.idx = i; dir.t = 0; markTicks();
  verseEl.classList.add('out'); clearTimeout(dir.timer);
  dir.timer = setTimeout(() => renderVerse(SHOTS[i]), 650);
}
function setPlaying(p) { dir.playing = p; $('#playIcon').setAttribute('d', p ? 'M8 5h3v14H8zM13 5h3v14h-3z' : 'M8 5v14l11-7z'); }
function enterFree() {
  if (dir.free || window.__noFree) return; dir.free = true; document.body.classList.add('free');
  const fwd = dir.look.clone().sub(camera.position); const d = Math.min(fwd.length(), 24);
  controls.target.copy(camera.position).addScaledVector(fwd.normalize(), d); controls.enabled = true; controls.update();
  $('#free').textContent = '续播';
}
function exitFree() {
  if (!dir.free) return; dir.free = false; document.body.classList.remove('free'); controls.enabled = false;
  $('#free').textContent = '自由观赏'; setPlaying(true);
}
window.addEventListener('pointerdown', (e) => { if (e.target === canvas) enterFree(); }, true);
canvas.addEventListener('wheel', () => enterFree(), { capture: true, passive: true });
$('#free').onclick = () => (dir.free ? exitFree() : enterFree());
$('#play').onclick = () => { if (dir.free) exitFree(); else setPlaying(!dir.playing); };
$('#prev').onclick = () => { exitFree(); goto(dir.idx - 1); };
$('#next').onclick = () => { exitFree(); goto(dir.idx + 1); };
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); $('#play').click(); }
  else if (e.code === 'ArrowRight') $('#next').click();
  else if (e.code === 'ArrowLeft') $('#prev').click();
  else if (e.code === 'KeyF') $('#free').click();
  else if (e.code === 'KeyM') $('#sound').click();
});
let idleTimer; const wake = () => { document.body.classList.remove('idle'); clearTimeout(idleTimer); idleTimer = setTimeout(() => document.body.classList.add('idle'), 3500); };
window.addEventListener('pointermove', wake); wake();

/* =========================================================================
   琴音（程序合成：五声音阶拨弦 + 水声）
   ========================================================================= */
const audio = { ctx: null, on: false };
function startAudio() {
  const ctx = new AudioContext(); audio.ctx = ctx;
  const master = ctx.createGain(); master.gain.value = 0.55; master.connect(ctx.destination); audio.master = master;
  const conv = ctx.createConvolver(); const len = ctx.sampleRate * 4.2, ir = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let c = 0; c < 2; c++) { const d = ir.getChannelData(c); for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3.2); }
  conv.buffer = ir; const wet = ctx.createGain(); wet.gain.value = 0.5; conv.connect(wet).connect(master);
  const nb = ctx.createBuffer(1, ctx.sampleRate * 3, ctx.sampleRate), nd = nb.getChannelData(0); let last = 0;
  for (let i = 0; i < nd.length; i++) { last = (last + 0.02 * (Math.random() * 2 - 1)) / 1.02; nd[i] = last * 3.2; }
  const ns = ctx.createBufferSource(); ns.buffer = nb; ns.loop = true; const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 520; const ng = ctx.createGain(); ng.gain.value = 0.16;
  ns.connect(lp).connect(ng).connect(master); ns.start();
  const scale = [146.83, 164.81, 185.0, 220.0, 246.94, 293.66, 329.63, 369.99, 440.0, 493.88];
  const pluck = (f, t0, v = 0.22) => {
    const g = ctx.createGain(); g.gain.setValueAtTime(0.0001, t0); g.gain.exponentialRampToValueAtTime(v, t0 + 0.008); g.gain.exponentialRampToValueAtTime(0.0001, t0 + 4.2);
    const lpf = ctx.createBiquadFilter(); lpf.type = 'lowpass'; lpf.frequency.setValueAtTime(3200, t0); lpf.frequency.exponentialRampToValueAtTime(500, t0 + 2.5);
    g.connect(lpf); lpf.connect(master); lpf.connect(conv);
    [[1, 1], [2, 0.35], [3, 0.12], [4.02, 0.05]].forEach(([h, a]) => { const o = ctx.createOscillator(); o.type = h === 1 ? 'triangle' : 'sine'; o.frequency.setValueAtTime(f * h * 1.012, t0); o.frequency.exponentialRampToValueAtTime(f * h, t0 + 0.12); const og = ctx.createGain(); og.gain.value = a; o.connect(og).connect(g); o.start(t0); o.stop(t0 + 4.4); });
  };
  let idx = 4;
  const phrase = () => {
    if (!audio.on) return;
    let t = ctx.currentTime + 0.05; const n = 2 + ((Math.random() * 4) | 0);
    for (let i = 0; i < n; i++) { idx = Math.max(0, Math.min(scale.length - 1, idx + [-2, -1, -1, 1, 1, 2][(Math.random() * 6) | 0])); pluck(scale[idx], t, 0.14 + Math.random() * 0.1); if (Math.random() < 0.3) pluck(scale[idx] / 2, t, 0.1); t += [0.35, 0.6, 0.9, 1.2][(Math.random() * 4) | 0]; }
    audio.timer = setTimeout(phrase, (t - ctx.currentTime) * 1000 + 1400 + Math.random() * 2600);
  };
  audio.phrase = phrase;
}
$('#sound').onclick = () => {
  audio.on = !audio.on;
  if (audio.on) { if (!audio.ctx) startAudio(); audio.ctx.resume(); audio.master.gain.setTargetAtTime(0.55, audio.ctx.currentTime, 0.5); audio.phrase(); }
  else { clearTimeout(audio.timer); audio.master.gain.setTargetAtTime(0, audio.ctx.currentTime, 0.4); }
  $('#soundWave').setAttribute('opacity', audio.on ? '1' : '.25');
};

/* =========================================================================
   主循环
   ========================================================================= */
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); applySize(); });

// 像素比自适应：帧率持续偏低就逐级降采样，长期充裕再升回（上限为画质档位）。
// 取样窗口 0.7 秒（原 1 秒）：一两拍卡顿就能触发降档；升档要更长的稳定期，避免在阈值上反复抖。
const prState = { cur: Math.min(devicePixelRatio, TUNE.pixelRatioCap), acc: 0, n: 0, ms: 0, low: 0, high: 0, cool: 0 };
function applySize() { renderer.setPixelRatio(prState.cur); renderer.setSize(innerWidth, innerHeight, false); syncAspect(); camMotion.resized = true; prState.cool = 10; }
// 画布尺寸有多条变更路径（resize 事件、像素比自适应、内嵌浏览器改视口时可能不发 resize），
// 相机宽高比必须跟着画布走，否则整幅画面被横向拉伸（月亮成了椭圆）。逐帧比对一次，代价可忽略。
function syncAspect() { const a = innerWidth / innerHeight; if (Math.abs(camera.aspect - a) > 1e-4) { camera.aspect = a; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight, false); camMotion.resized = true; } }

/* 取景守卫：东坡的机位必须落在他的背面半球。
   他始终朝着月亮（PF 就是脸的方向），机位一旦越过侧面，五官就亮在轮廓上。
   镜头将来再改，只要把机位摆到他正面，控制台会直接点名，?stats 里也能实时看到。
   face = 机位方向与 PF 的点积：-1 正后方、0 正侧面、+1 正前方。 */
const rear = { face: -2, d: 0, warned: -1 };
const _ndc = V(), _rearV = V();
function guardRear() {
  _rearV.copy(camera.position).sub(PB); _rearV.y = 0;
  const d = _rearV.length();
  rear.d = d;
  if (dir.free || d < 0.5) { rear.face = -2; return; }
  _ndc.copy(PB); _ndc.y += 0.9; _ndc.project(camera);
  if (_ndc.z > 1 || Math.abs(_ndc.x) > 1.15 || Math.abs(_ndc.y) > 1.15) { rear.face = -2; return; }   // 人不在画面里就不管
  rear.face = _rearV.multiplyScalar(1 / d).x * PF.x + _rearV.z * PF.z;
  if (rear.face > -0.25 && rear.warned !== dir.idx) {
    rear.warned = dir.idx;
    const off = (Math.acos(Math.max(-1, Math.min(1, -rear.face))) * 180 / Math.PI).toFixed(0);
    console.warn(`[月白] 第 ${dir.idx} 镜「${SHOTS[dir.idx].text}」机位进了东坡的正面半球：偏离正后方 ${off}°，距离 ${d.toFixed(1)}m`);
  }
}
let statEl = null;
if (SHOW_STATS) {
  statEl = document.createElement('div');
  statEl.id = 'perf';
  document.body.appendChild(statEl);
  // 手动 reset，才能把一帧里主通道 + 反射 + 阴影 + 泛光的绘制次数累加成真实总数
  renderer.info.autoReset = false;
}
function adaptResolution(dt, cpuMs) {
  prState.acc += dt; prState.n++; prState.ms += cpuMs; prState.cool = Math.max(0, prState.cool - dt);
  if (prState.acc < 0.7) return;
  const fps = prState.n / prState.acc, avgCpu = prState.ms / prState.n;
  prState.acc = 0; prState.n = 0; prState.ms = 0;
  const cap = Math.min(devicePixelRatio, TUNE.pixelRatioCap);
  if (prState.cool > 0) { prState.low = 0; prState.high = 0; }
  else if (fps < 50 && prState.cur > TUNE.pixelRatioMin) {
    prState.low++; prState.high = 0;
    if (prState.low >= 2) { prState.cur = Math.max(TUNE.pixelRatioMin, prState.cur - 0.25); prState.low = 0; applySize(); }
  } else if (fps > 58 && prState.cur < cap) {
    prState.high++; prState.low = 0;
    if (prState.high >= 6) { prState.cur = Math.min(cap, prState.cur + 0.25); prState.high = 0; applySize(); }
  } else { prState.low = 0; prState.high = 0; }
  if (statEl) {
    const r = renderer.info.render;
    statEl.textContent =
      `${fps.toFixed(0)} FPS · ${QUALITY} · ${renderer.backend.isWebGPUBackend ? 'WEBGPU' : 'WEBGL2'} · ${prState.cur.toFixed(2)}x\n` +
      `CPU ${avgCpu.toFixed(1)}ms/帧 · ${r.drawCalls} 绘制 · ${(r.triangles / 1000).toFixed(0)}k 三角\n` +
      `东坡 ${rear.face > -2 ? `距 ${rear.d.toFixed(1)}m · 后侧 ${(Math.acos(Math.max(-1, Math.min(1, -rear.face))) * 180 / Math.PI).toFixed(0)}°` : '未入镜'}`;
  }
}

const fx = { wind: 0, pass: 2 };
const tmpP = V(), tmpT = V();
let T = 0, last = performance.now();
const poseState = { yaw: 0, spin: 0 };
const _q1 = new THREE.Quaternion(), _q2 = new THREE.Quaternion();
const damp = (a, b, k, dt) => a + (b - a) * (1 - Math.exp(-k * dt));

function updateBoatAndPoet(dt) {
  boatAt(T, B); F.copy(boatDir(T)); S.set(-F.z, 0, F.x);
  const yaw = Math.atan2(-F.x, -F.z);
  boat.position.set(B.x, Math.sin(T * 1.1) * 0.035, B.z);
  boat.rotation.set(Math.sin(T * 0.8) * 0.012, yaw, Math.sin(T * 1.05) * 0.025);
  uBoat.value.copy(B);
  const pose = dir.free ? 'look' : SHOTS[dir.idx].pose;
  poseState.spin = damp(poseState.spin, pose === 'dance' ? Math.sin(T * 0.65) * 0.3 : 0, 2.2, dt);
  let rel = Math.atan2(-MOON_DIR.x, -MOON_DIR.z) - yaw; rel = Math.atan2(Math.sin(rel), Math.cos(rel)); rel = Math.max(-1.2, Math.min(1.2, rel));
  poseState.yaw = damp(poseState.yaw, rel, 1.5, dt);
  poet.root.rotation.y = poseState.yaw + poseState.spin;
  const fy = yaw + poseState.yaw; PF.set(-Math.sin(fy), 0, -Math.cos(fy));
  poet.root.getWorldPosition(PB);
  // 静模：呼吸般的前后微摆 + 随舟的左右重心
  poet.model.rotation.set(Math.sin(T * 0.63 + 1) * 0.008, Math.PI, Math.sin(T * 0.9) * 0.012 - Math.sin(T * 1.05) * 0.015);

}

let frameNo = 0;
function frame() {
  const now = performance.now(); const dt = Math.min(0.05, (now - last) / 1000); last = now; T += dt;
  // frameNo 是本帧的节拍：阴影与反射都按它取模，所以必须先自增再触发绘制
  frameNo++;
  syncAspect();
  if (statEl) renderer.info.reset();
  updateBoatAndPoet(dt);
  const shot = SHOTS[dir.idx];

  // 月光方向固定，阴影按帧隔重绘；船与人的位移极缓，肉眼无差别
  // 近看舟人时阴影逐帧更新，否则船上人影会按 3 帧一跳地抖；远景仍隔帧
  if (frameNo % TUNE.shadowEveryFrames === 0 || camera.position.distanceTo(B) < 30 || camMotion.resized) moonLight.shadow.needsUpdate = true;

  if (!dir.free && dir.playing) { dir.t = Math.min(dir.t + dt, shot.dur); if (dir.t >= shot.dur) goto(dir.idx + 1); }
  const u = clamp01(dir.t / shot.dur);

  // 氛围参数
  uCold.value = damp(uCold.value, shot.cold ? 1 : 0, 1.2, dt);
  uSea.value = damp(uSea.value, shot.sea ? 1 : 0, 1.0, dt); sea.visible = uSea.value > 0.01;
  fx.wind = damp(fx.wind, shot.wind ? 1 : 0, 1.5, dt);
  fx.pass = shot.pass ? lerp(-1.25, 1.25, u) : damp(fx.pass, 2.2, 0.4, dt);
  placePassCloud(fx.pass);
  updateSkyLanterns(dt, T, !!shot.sky && !dir.free, u, !!shot.lantern && !dir.free);

  // 镜头
  if (!dir.free) {
    const pose = shot.cam(ease(u));
    let P = pose.p, D = pose.t.clone().sub(pose.p).normalize(), fov = pose.fov;
    const b = dir.blend;
    if (b) {
      if (b.dur === null) b.dur = Math.min(7, Math.max(2.8, 2.2 + b.p.distanceTo(P) / 45));
      b.k += dt / b.dur; const k = ease(clamp01(b.k));
      const arc = Math.min(b.p.distanceTo(P) * 0.12, 28) * Math.sin(Math.PI * k);
      P = b.p.clone().lerp(P, k).add(V(0, arc, 0)); D = b.d.clone().lerp(D, k).normalize(); fov = lerp(b.fov, fov, k);
      if (b.k >= 1) dir.blend = null;
    }
    camera.position.copy(P); dir.look.copy(P).addScaledVector(D, 20); camera.lookAt(dir.look);
    if (Math.abs(camera.fov - fov) > 0.01) { camera.fov = fov; camera.updateProjectionMatrix(); }
  } else {
    controls.update();
    if (camera.position.y < 0.6) camera.position.y = 0.6;
    dir.look.copy(controls.target);
  }

  trackCamera();
  sky.position.copy(camera.position);
  for (const m of mists) m.rotation.y = Math.atan2(camera.position.x - m.position.x, camera.position.z - m.position.z);

  updatePetals(dt, T, fx.wind);
  updateRiver(dt, T);
  updateBirds(T);

  post.render();
  camMotion.resized = false;
  guardRear();
  // 这里量的是「CPU 把一帧交出去」的耗时（含场景更新与提交绘制命令），不含 GPU 实际画完的时间。
  // 配合 FPS 一起看：若 FPS 对应的帧时长远大于这个数，瓶颈就在 GPU 而不是 JS。
  adaptResolution(dt, performance.now() - now);
}

phase('舟 · 花灯 · 后期 · 界面');
loadMsg('候月升起');
renderVerse(SHOTS[0]); markTicks();
await nextFrame();
// WebGL 的并行编译回调在部分内嵌浏览器中不会完成，交给首帧按需编译。
if (renderer.backend.isWebGPUBackend) {
  try { await renderer.compileAsync(scene, camera); } catch (e) { console.warn(e); }
}
renderer.setAnimationLoop(frame);
window.__moon = { goto: (i) => { exitFree(); goto(i); }, dir, setPlaying, exitFree, camera, PB, PF, controls, enterFree };
setTimeout(() => $('#loader').classList.add('gone'), 600);

/* 加载耗时明细：打开控制台即可看到各阶段用时，便于定位瓶颈 */
phase('加载完毕');
// 再等一帧：首帧的耗时里含着色器编译，往往是最长的一段，单独列出来
requestAnimationFrame(() => {
  phase('首帧渲染（含着色器编译）');
  const rows = phases.map((p, i) => ({ 阶段: p.阶段, 耗时毫秒: p.毫秒 - (i ? phases[i - 1].毫秒 : 0), 累计毫秒: p.毫秒 }));
  let lights = 0; scene.traverse((o) => { if (o.isLight) lights++; });
  console.log(`[月白] 画质 ${QUALITY} · 像素比 ${TUNE.pixelRatioMin}~${TUNE.pixelRatioCap} · 多重采样 ${TUNE.samples}x · `
    + `阴影 ${TUNE.shadowMapSize}@每${TUNE.shadowEveryFrames}帧 · 反射 ${TUNE.reflectionScale}@每${TUNE.reflectionEveryFrames}帧 · `
    + `雾片 ${TUNE.mistCount} · 云 ${TUNE.cloudCount} · 光源 ${lights} 盏`);
  console.log('想看帧率/绘制次数/CPU 耗时：加 ?stats；想试光照的开销：加 ?lights=lean');
  console.table(rows);
});
