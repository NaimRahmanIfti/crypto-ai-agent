// 3D backdrop: candle ribbon of the selected market, an "AI core" whose colour
// and pulse follow the model signal, and a drifting particle field.
let THREE;
async function loadThree() {
  // CDN first (cached by the browser); a local copy in web/vendor/ is the offline fallback.
  try { return await import('https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js'); }
  catch { return await import('./vendor/three.module.min.js'); }
}

const COLORS = { up: 0x3df5c0, down: 0xff5c7a, neutral: 0xffc25c, accent: 0x8c7bff };
const ease = t => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);
const back = t => { const c = 1.7; t = Math.min(1, Math.max(0, t)); return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); };

export class Scene3D {
  async init(canvas) {
    THREE = await loadThree();
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x060914, 0.026);
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 200);
    this.camera.position.set(0, 4, 22);
    this.mouse = { x: 0, y: 0 };
    this.clock = new THREE.Clock();
    this.signalColor = new THREE.Color(COLORS.neutral);
    this.targetColor = new THREE.Color(COLORS.neutral);
    this.conf = 0;

    this.scene.add(new THREE.AmbientLight(0x8090ff, 0.35));
    const key = new THREE.PointLight(0x6c8cff, 60, 60); key.position.set(-8, 10, 10); this.scene.add(key);
    this.coreLight = new THREE.PointLight(COLORS.neutral, 80, 40); this.coreLight.position.set(0, 3, 0); this.scene.add(this.coreLight);

    this._buildGrid(); this._buildParticles(); this._buildCore();
    this.ribbon = new THREE.Group(); this.scene.add(this.ribbon);

    addEventListener('resize', () => this._resize());
    addEventListener('pointermove', e => {
      this.mouse.x = e.clientX / innerWidth - 0.5; this.mouse.y = e.clientY / innerHeight - 0.5;
    });
    this._resize();
    this.renderer.setAnimationLoop(() => this._tick());
    return this;
  }

  _resize() {
    const w = innerWidth, h = innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  _buildGrid() {
    const g = new THREE.GridHelper(160, 80, 0x2a3570, 0x141a3a);
    g.position.y = -4; g.material.transparent = true; g.material.opacity = 0.55;
    this.grid = g; this.scene.add(g);
  }

  _buildParticles() {
    const n = 1800, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 120;
      pos[i * 3 + 1] = Math.random() * 50 - 6;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 120;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    // Two-tone particles (ion violet / sky) for depth.
    const col = new Float32Array(n * 3), a = new THREE.Color(0x9f92ff), b = new THREE.Color(0x7fd6ff);
    for (let i = 0; i < n; i++) { const c = Math.random() < 0.6 ? a : b; col.set([c.r, c.g, c.b], i * 3); }
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    this.particleHome = pos.slice();
    this.particles = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 0.1, vertexColors: true, transparent: true, opacity: 0.8, depthWrite: false, blending: THREE.AdditiveBlending }));
    this.scene.add(this.particles);
  }

  _buildCore() {
    this.core = new THREE.Group();
    this.core.position.set(0, 3, -2);
    const mat = new THREE.MeshStandardMaterial({ color: COLORS.neutral, emissive: COLORS.neutral,
      emissiveIntensity: 0.6, metalness: 0.4, roughness: 0.25, wireframe: true });
    this.coreMesh = new THREE.Mesh(new THREE.IcosahedronGeometry(1.6, 1), mat);
    const inner = new THREE.Mesh(new THREE.IcosahedronGeometry(0.9, 0),
      new THREE.MeshStandardMaterial({ color: 0x0b0f25, emissive: COLORS.neutral, emissiveIntensity: 0.9, metalness: 0.9, roughness: 0.1 }));
    this.coreInner = inner;
    this.rings = [2.4, 3.0, 3.6].map((r, i) => {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(r, 0.02 + i * 0.005, 8, 160),
        new THREE.MeshBasicMaterial({ color: COLORS.accent, transparent: true, opacity: 0.55 - i * 0.12 }));
      ring.rotation.x = Math.PI / 2 + (i - 1) * 0.35; ring.rotation.y = i * 0.6;
      this.core.add(ring); return ring;
    });
    this.core.add(this.coreMesh, inner);
    this.scene.add(this.core);
  }

  // bars: [{o,h,l,c}], newest last
  setCandles(bars) {
    this.ribbon.clear();
    const data = bars.slice(-110);
    if (!data.length) return;
    const lo = Math.min(...data.map(b => b.l)), hi = Math.max(...data.map(b => b.h));
    const scale = 9 / (hi - lo || 1);
    const R = 15, span = Math.PI * 1.15, start = Math.PI / 2 + span / 2;
    const upM = new THREE.MeshStandardMaterial({ color: COLORS.up, emissive: COLORS.up, emissiveIntensity: 0.55, metalness: 0.3, roughness: 0.35 });
    const dnM = new THREE.MeshStandardMaterial({ color: COLORS.down, emissive: COLORS.down, emissiveIntensity: 0.55, metalness: 0.3, roughness: 0.35 });
    const wickM = new THREE.MeshBasicMaterial({ color: 0x8b95c9, transparent: true, opacity: 0.6 });
    const pts = [];
    data.forEach((b, i) => {
      const a = start - (i / (data.length - 1)) * span;
      const x = Math.cos(a) * R, z = -Math.sin(a) * R + 6;
      const y0 = (Math.min(b.o, b.c) - lo) * scale - 4, y1 = (Math.max(b.o, b.c) - lo) * scale - 4;
      const body = new THREE.Mesh(new THREE.BoxGeometry(0.42, Math.max(0.05, y1 - y0), 0.42), b.c >= b.o ? upM : dnM);
      body.position.set(x, (y0 + y1) / 2, z); body.rotation.y = a;
      body.userData.baseY = body.position.y; body.userData.i = i;
      body.scale.y = 0.001;
      const wick = new THREE.Mesh(new THREE.BoxGeometry(0.06, (b.h - b.l) * scale, 0.06), wickM);
      wick.position.set(x, ((b.h + b.l) / 2 - lo) * scale - 4, z);
      this.ribbon.add(wick, body);
      pts.push(new THREE.Vector3(x, (b.c - lo) * scale - 4, z));
    });
    const curve = new THREE.CatmullRomCurve3(pts);
    const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 400, 0.05, 8, false),
      new THREE.MeshBasicMaterial({ color: COLORS.accent, transparent: true, opacity: 0.85 }));
    this.ribbon.add(tube);
    this.ribbonBorn = this.clock.getElapsedTime();
  }

  // Boot sequence: camera dollies in from deep space while the particle field
  // bursts out of the AI core and the core itself scales up with an overshoot.
  playIntro(seconds = 2.6) {
    if (!seconds) return;
    this.intro = { start: this.clock.getElapsedTime(), dur: seconds };
  }

  _introTick(t) {
    const I = this.intro;
    const k = (t - I.start) / I.dur;
    const e = ease(k);
    const pos = this.particles.geometry.attributes.position;
    const home = this.particleHome, cx = this.core.position.x, cy = this.core.position.y, cz = this.core.position.z;
    for (let i = 0; i < home.length; i += 3) {
      const d = e * (0.85 + 0.15 * Math.sin(i));   // slight stagger so it reads as a burst, not a scale
      pos.array[i] = cx + (home[i] - cx) * d;
      pos.array[i + 1] = cy + (home[i + 1] - cy) * d;
      pos.array[i + 2] = cz + (home[i + 2] - cz) * d;
    }
    pos.needsUpdate = true;
    this.core.scale.setScalar(Math.max(0.001, back(k * 1.4)));
    this.rings.forEach((r, i) => { r.rotation.z += (1 - e) * 0.15 * (i + 1); });
    this.camera.position.set(Math.sin((1 - e) * 1.2) * 18, 4 + (1 - e) * 26, 22 + (1 - e) * 70);
    this.camera.lookAt(0, 2, 0);
    this.coreLight.intensity = 80 + (1 - e) * 220;
    if (k >= 1) {
      pos.array.set(home); pos.needsUpdate = true;
      this.core.scale.setScalar(1);
      this.intro = null;
    }
  }

  setSignal(tone, exposure) {
    this.targetColor.set(tone === 'up' ? COLORS.up : tone === 'down' ? COLORS.down : COLORS.neutral);
    this.conf = Math.max(0, Math.min(1, exposure));
  }

  _tick() {
    const t = this.clock.getElapsedTime();
    this.signalColor.lerp(this.targetColor, 0.04);
    this.coreMesh.material.color.copy(this.signalColor);
    this.coreMesh.material.emissive.copy(this.signalColor);
    this.coreInner.material.emissive.copy(this.signalColor);
    this.coreLight.color.copy(this.signalColor);
    const pulse = 1 + Math.sin(t * (2 + this.conf * 6)) * (0.04 + this.conf * 0.12);
    this.coreMesh.scale.setScalar(pulse);
    this.coreMesh.rotation.x += 0.004; this.coreMesh.rotation.y += 0.006;
    this.coreInner.rotation.y -= 0.01;
    this.rings.forEach((r, i) => { r.rotation.z += 0.003 * (i + 1) * (i % 2 ? -1 : 1); });
    this.core.position.y = 3 + Math.sin(t * 0.8) * 0.25;

    // candles grow in one after another
    if (this.ribbonBorn !== undefined) {
      const age = t - this.ribbonBorn;
      this.ribbon.children.forEach(m => {
        if (m.userData.i === undefined) return;
        const k = Math.min(1, Math.max(0.001, (age - m.userData.i * 0.012) * 2.5));
        m.scale.y = 1 - Math.pow(1 - k, 3);
      });
    }
    this.ribbon.rotation.y = Math.sin(t * 0.08) * 0.12;
    this.particles.rotation.y = t * 0.01;
    this.grid.position.z = (t * 1.2) % 2;

    if (this.intro) {
      this._introTick(t);
    } else {
      this.camera.position.x += (this.mouse.x * 6 - this.camera.position.x) * 0.03;
      this.camera.position.y += (4 - this.mouse.y * 3 - this.camera.position.y) * 0.03;
      this.camera.position.z += (22 - this.camera.position.z) * 0.05;
      this.camera.lookAt(0, 1.5, 0);
    }
    this.renderer.render(this.scene, this.camera);
  }
}
