// Draws the workflow as a transit map (branches included) and animates a train along it.
const NS = "http://www.w3.org/2000/svg";
const el = (name, attrs = {}, parent) => {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
};
const fmt = (ms) => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(2)} s`);

export class LineMap {
  constructor(container) {
    this.container = container;
    this.layout = null;
    this.history = [];
    this.finished = false;
    window.addEventListener("resize", () => {
      if (this.layout && (container.clientWidth < 440) !== this.vertical) {
        this.draw(this.layout);
        this.replay();
      }
    });
  }

  depths(layout) {
    const depth = { START: 0 };
    for (let changed = true; changed; ) {
      changed = false;
      for (const e of layout.edges) {
        if (depth[e.source] !== undefined && !(depth[e.target] >= depth[e.source] + 1)) {
          depth[e.target] = depth[e.source] + 1;
          changed = true;
        }
      }
    }
    return depth;
  }

  draw(layout) {
    this.layout = layout;
    this.vertical = this.container.clientWidth < 440;
    const depth = this.depths(layout);
    const used = {};
    const slot = {};
    layout.nodes.forEach((n) => { slot[n.id] = used[depth[n.id]] = (used[depth[n.id]] ?? -1) + 1; });
    const cols = Math.max(...Object.values(depth)) + 1;
    const rows = Math.max(...Object.values(used)) + 1;
    const W = this.vertical ? 340 : 820;
    const gap = this.vertical ? 105 : (W - 130) / (cols - 1);
    const H = this.vertical ? 70 + gap * (cols - 1) : 175 + (rows - 1) * 110;
    this.pos = {};
    layout.nodes.forEach((n) => {
      const d = depth[n.id], s = slot[n.id];
      this.pos[n.id] = this.vertical ? { x: 45 + s * 160, y: 45 + d * gap } : { x: 65 + d * gap, y: 70 + s * 110 };
    });

    this.container.replaceChildren();
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "map-svg", "aria-hidden": "true" }, this.container);
    this.segs = {};
    for (const e of layout.edges) {
      const a = this.pos[e.source], b = this.pos[e.target];
      const attrs = { x1: a.x, y1: a.y, x2: b.x, y2: b.y };
      const len = Math.hypot(b.x - a.x, b.y - a.y);
      el("line", { ...attrs, class: e.conditional ? "rail dashed" : "rail" }, svg);
      this.segs[`${e.source}>${e.target}`] = el("line", { ...attrs, class: "rail-fill", "stroke-dasharray": len, "stroke-dashoffset": len }, svg);
    }

    this.st = {};
    for (const n of layout.nodes) {
      const { x, y } = this.pos[n.id];
      const terminal = n.kind === "terminal";
      const g = el("g", { class: "station", "data-state": "idle" }, svg);
      el("circle", { cx: x, cy: y, r: 27, class: "pulse" }, g);
      if (terminal) el("rect", { x: x - 17, y: y - 17, width: 34, height: 34, rx: 9, class: "dot terminal" }, g);
      else el("circle", { cx: x, cy: y, r: 18, class: "dot" }, g);
      el("path", { d: `M${x - 7} ${y} l5 5 l9 -10`, class: "check", fill: "none" }, g);
      const lx = this.vertical ? x + 40 : x, ly = this.vertical ? y + 6 : y + 52;
      const anchor = this.vertical ? "start" : "middle";
      el("text", { x: lx, y: ly, "text-anchor": anchor, class: "st-label" }, g).textContent = n.id;
      const time = el("text", { x: lx, y: ly + 20, "text-anchor": anchor, class: "st-time" }, g);
      time.textContent = terminal ? "" : "waiting";
      this.st[n.id] = { g, time, terminal };
    }

    this.train = el("g", { class: "train" }, svg);
    for (const dx of [0, 1, 2]) el("circle", { cx: -14 - dx * 4, cy: -14, r: 4, class: "puff" }, this.train);
    el("rect", { x: -20, y: -11, width: 28, height: 22, rx: 7, class: "train-body" }, this.train);
    el("path", { d: "M6 -11 h8 a6 6 0 0 1 6 6 v10 a6 6 0 0 1 -6 6 h-8 z", class: "train-nose" }, this.train);
    el("circle", { cx: -11, cy: -1, r: 3.2, class: "train-window" }, this.train);
    el("circle", { cx: -1, cy: -1, r: 3.2, class: "train-window" }, this.train);
    this.cur = "START";
    this.place("START", false);
  }

  place(id, animate = true) {
    const { x, y } = this.pos[id];
    const dx = this.vertical ? -46 : 0, dy = this.vertical ? 0 : -42;
    this.train.style.transition = animate ? "" : "none";
    if (animate) {
      this.train.classList.add("moving");
      setTimeout(() => this.train.classList.remove("moving"), 950);
    }
    this.train.style.transform = `translate(${x + dx}px, ${y + dy}px)`;
    if (!animate) void this.train.getBoundingClientRect();
  }

  set(id, state) { this.st[id].g.dataset.state = state; }

  fill(from, to) {
    const seg = this.segs[`${from}>${to}`];
    if (seg) seg.style.strokeDashoffset = 0;
  }

  reset() {
    this.history = [];
    this.finished = false;
    this.draw(this.layout);
  }

  begin() { this.set("START", "done"); }

  arrive(id) {
    this.fill(this.cur, id);
    this.cur = id;
    this.place(id);
    this.set(id, "active");
    this.st[id].time.textContent = "running…";
  }

  depart(id, ms) {
    this.set(id, "done");
    this.st[id].time.textContent = fmt(ms);
    this.history.push({ id, ms });
  }

  finish() {
    this.finished = true;
    this.fill(this.cur, "END");
    this.cur = "END";
    this.place("END");
    this.set("END", "done");
  }

  fail(id) {
    if (!this.st[id]) return;
    this.set(id, "error");
    this.st[id].time.textContent = "stopped";
  }

  replay() {
    if (!this.history.length) return;
    this.set("START", "done");
    let prev = "START";
    for (const h of this.history) {
      this.fill(prev, h.id);
      this.set(h.id, "done");
      this.st[h.id].time.textContent = fmt(h.ms);
      prev = h.id;
    }
    if (this.finished) { this.fill(prev, "END"); this.set("END", "done"); prev = "END"; }
    this.cur = prev;
    this.place(prev, false);
  }
}
