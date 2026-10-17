import { api, ApiError, loadKey, saveKey } from "./api.js";
import { LineMap } from "./map.js";

const $ = (selector) => document.querySelector(selector);
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const STATE_FIELDS = [
  ["question", "str"],
  ["k", "int"],
  ["sources", "list"],
  ["context", "str"],
  ["answer", "str"],
];
const EXAMPLES = [
  "What does RAG do?",
  "How does LangGraph build workflows?",
  "What is a checkpoint?",
  "What does LangSmith do?",
  "Who won the 2018 World Cup?",
];

let health = null;
let map = null;
let order = [];
let busy = false;
let toastTimer;
let lastQuestion = "";
let audio;
const chime = (freq) => {
  if (!$("#opt-sound").checked) return;
  audio ??= new AudioContext();
  const osc = audio.createOscillator(), gain = audio.createGain();
  osc.type = "triangle";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.08, audio.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.35);
  osc.connect(gain).connect(audio.destination);
  osc.start();
  osc.stop(audio.currentTime + 0.36);
};
const FEEDBACK = ["btn-up", "btn-down"];
const LOG_KEY = "waypoint.log";
const readLog = () => { try { return JSON.parse(localStorage.getItem(LOG_KEY)) || []; } catch { return []; } };
const writeLog = (items) => { try { localStorage.setItem(LOG_KEY, JSON.stringify(items.slice(0, 20))); } catch { /* storage unavailable */ } };

// ---------- small helpers ----------
const toast = (message) => {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("show"), 3200);
};

const shorten = (text, max = 70) => (text.length > max ? text.slice(0, max - 1) + "…" : text);
const seconds = (ms) => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(2)} s`);

// ---------- state panel ----------
function buildState() {
  const list = $("#state");
  list.replaceChildren();
  for (const [name, type] of STATE_FIELDS) {
    const row = document.createElement("div");
    row.className = "state-row";
    row.dataset.field = name;
    row.innerHTML = `<dt><code>${name}</code><span>${type}</span></dt><dd class="pending">not set yet</dd>`;
    list.appendChild(row);
  }
}

function setStateValue(name, value) {
  const row = document.querySelector(`.state-row[data-field="${name}"]`);
  if (!row) return;
  const dd = row.querySelector("dd");
  let text;
  if (name === "sources") text = `${value.length} passage${value.length === 1 ? "" : "s"}`;
  else if (name === "k") text = String(value);
  else text = value ? shorten(String(value).replace(/\s+/g, " "), 90) : "(empty)";
  dd.textContent = text;
  dd.title = typeof value === "string" ? value : "";
  dd.classList.remove("pending");
  row.classList.remove("flash");
  void row.offsetWidth;
  row.classList.add("flash");
}

// ---------- status chips ----------
function renderStatus() {
  $("#chip-llm").textContent = health.demo_mode ? "No model needed" : health.llm;
  $("#chip-docs").textContent = `${health.documents} passages`;
  const mode = $("#chip-mode");
  mode.hidden = !health.demo_mode;
  mode.textContent = "Demo mode";
  mode.classList.add("warn");
  $("#btn-key").hidden = health.demo_mode || health.server_has_key;
}

// ---------- library ----------
async function renderLibrary() {
  const { documents } = await api.documents();
  health.documents = documents.length;
  renderStatus();
  const grid = $("#docs");
  grid.replaceChildren();
  if (!documents.length) {
    grid.innerHTML = '<p class="empty">The library is empty. Add some text above, or reset to the sample.</p>';
    return;
  }
  for (const doc of documents) {
    const card = document.createElement("article");
    card.className = "doc";
    card.dataset.id = doc.id;
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = doc.source;
    const text = document.createElement("p");
    text.textContent = doc.text;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "btn ghost small";
    remove.textContent = "Remove";
    remove.setAttribute("aria-label", `Remove passage: ${shorten(doc.text, 40)}`);
    remove.addEventListener("click", async () => {
      await api.removeDocument(doc.id);
      renderLibrary();
    });
    card.append(tag, text, remove);
    grid.appendChild(card);
  }
}

// ---------- answer + sources ----------
function highlight(text, question) {
  const wanted = new Set((question.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => w.length > 3));
  const frag = document.createDocumentFragment();
  for (const part of text.split(/([A-Za-z0-9]+)/)) {
    if (wanted.has(part.toLowerCase())) {
      const mark = document.createElement("mark");
      mark.textContent = part;
      frag.appendChild(mark);
    } else frag.appendChild(document.createTextNode(part));
  }
  return frag;
}

function markLibrary(sources) {
  document.querySelectorAll(".doc").forEach((card) => {
    card.classList.remove("hit");
    card.querySelector(".badge")?.remove();
  });
  sources.forEach((s, i) => {
    const card = document.querySelector(`.doc[data-id="${s.id}"]`);
    if (!card) return;
    card.classList.add("hit");
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = `Retrieved #${i + 1}`;
    card.appendChild(badge);
  });
}

function linkify(text) {
  const frag = document.createDocumentFragment();
  for (const part of text.split(/(\[\d+\])/)) {
    const m = part.match(/^\[(\d+)\]$/);
    if (!m) { frag.appendChild(document.createTextNode(part)); continue; }
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "cite";
    btn.textContent = m[1];
    btn.setAttribute("aria-label", `Show source ${m[1]}`);
    btn.addEventListener("click", () => {
      const li = document.getElementById(`src-${m[1]}`);
      if (!li) return;
      li.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "center" });
      li.classList.remove("focus");
      void li.offsetWidth;
      li.classList.add("focus");
    });
    frag.appendChild(btn);
  }
  return frag;
}

function renderSources(sources) {
  const list = $("#sources");
  list.replaceChildren();
  sources.forEach((s, i) => {
    const pct = Math.max(0, Math.min(1, s.score));
    const item = document.createElement("li");
    item.id = `src-${i + 1}`;
    item.innerHTML = `
      <span class="rank">${i + 1}</span>
      <div class="src-body">
        <p></p>
        <div class="src-meta"><span class="tag"></span>
          <span class="bar" role="img" aria-label="Match ${Math.round(pct * 100)} percent"><i style="width:${Math.round(pct * 100)}%"></i></span>
          <span class="pct">${Math.round(pct * 100)}% match</span></div>
      </div>`;
    item.querySelector("p").replaceChildren(highlight(s.text, lastQuestion));
    item.querySelector(".tag").textContent = s.source;
    list.appendChild(item);
  });
  $("#sources-panel").hidden = false;
}

let typing;
function showAnswer(text) {
  const target = $("#answer-text");
  clearInterval(typing);
  if (reducedMotion) {
    target.replaceChildren(linkify(text));
    return;
  }
  target.textContent = "";
  let i = 0;
  const step = Math.max(1, Math.ceil(text.length / 70));
  typing = setInterval(() => {
    i += step;
    target.textContent = text.slice(0, i);
    if (i >= text.length) {
      clearInterval(typing);
      target.replaceChildren(linkify(text));
    }
  }, 16);
}

function setStamp(label, kind) {
  const stamp = $("#answer-stamp");
  stamp.textContent = label;
  stamp.dataset.kind = kind;
}

// ---------- the run ----------
async function ask(question) {
  if (busy) return;
  busy = true;
  lastQuestion = question;
  const k = Number($("#opt-k").value);
  $("#btn-copy").hidden = true;
  $("#btn-listen").hidden = true;
  FEEDBACK.forEach((id) => { $("#" + id).hidden = true; $("#" + id).setAttribute("aria-pressed", "false"); });
  speechSynthesis?.cancel();
  markLibrary([]);
  $("#btn-ask").disabled = true;
  $("#ask-error").hidden = true;

  buildState();
  setStateValue("question", question);
  map.reset();
  clearInterval(typing);

  setStamp("Travelling", "busy");
  $("#answer-meta").textContent = "";
  $("#answer-question").textContent = question;
  $("#answer-question").hidden = false;
  $("#answer-text").textContent = "";
  $("#sources-panel").hidden = true;

  let currentNode = null;
  let nodeStartedAt = 0;
  let chain = Promise.resolve();
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  // Events can arrive in a few milliseconds. Play them back at a watchable pace.
  const MIN_DWELL = reducedMotion ? 0 : $("#opt-slow").checked ? 2200 : 800;
  try {
    const handle = async (evt) => {
      switch (evt.event) {
        case "start":
          setStateValue("k", k);
          map.begin();
          break;
        case "node_start":
          currentNode = evt.node;
          nodeStartedAt = performance.now();
          map.arrive(evt.node);
          break;
        case "node_end":
          await sleep(Math.max(0, MIN_DWELL - (performance.now() - nodeStartedAt)));
          map.depart(evt.node, evt.elapsed_ms);
          chime(evt.node === "fallback" ? 330 : evt.node === "retrieve" ? 523 : 659);
          for (const [field, value] of Object.entries(evt.update)) setStateValue(field, value);
          if (evt.update.sources) {
            renderSources(evt.update.sources);
            markLibrary(evt.update.sources);
          }
          break;
        case "done":
          await sleep(MIN_DWELL / 2);
          map.finish();
          if (!evt.path.includes("fallback")) chime(784);
          const rerouted = evt.path.includes("fallback");
          setStamp(rerouted ? "Rerouted" : "Arrived", rerouted ? "warn" : "ok");
          $("#answer-meta").textContent = `Total ${seconds(evt.total_ms)}${rerouted ? " · fallback route" : ""}`;
          showAnswer(evt.answer);
          $("#btn-copy").hidden = false;
          $("#btn-listen").hidden = !("speechSynthesis" in window);
          showFeedback();
          writeLog([{ q: question, a: evt.answer, ms: evt.total_ms, k, sources: evt.sources, t: Date.now() }, ...readLog()]);
          renderLog();
          break;
        case "error":
          throw new ApiError(evt.message, evt.code);
      }
    };
    await api.ask(question, k, (evt) => {
      chain = chain.then(() => handle(evt));
    });
    await chain;
  } catch (err) {
    if (currentNode) map.fail(currentNode);
    setStamp("Signal failure", "error");
    $("#answer-text").textContent = err.message;
    if (err.code === "missing_api_key") openKeyDialog();
  } finally {
    busy = false;
    $("#btn-ask").disabled = false;
  }
}

function showFeedback(value) {
  FEEDBACK.forEach((id) => { $("#" + id).hidden = false; });
  $("#btn-up").setAttribute("aria-pressed", String(value === "up"));
  $("#btn-down").setAttribute("aria-pressed", String(value === "down"));
}

// ---------- journey log ----------
function restore(entry) {
  clearInterval(typing);
  lastQuestion = entry.q;
  $("#question").value = entry.q;
  buildState();
  setStateValue("question", entry.q);
  setStateValue("k", entry.k);
  setStateValue("sources", entry.sources);
  setStateValue("context", entry.sources.map((s) => s.text).join("\n"));
  setStateValue("answer", entry.a);
  setStamp("From the log", "idle");
  $("#answer-meta").textContent = `Total ${seconds(entry.ms)}`;
  $("#answer-question").textContent = entry.q;
  $("#answer-question").hidden = false;
  $("#answer-text").replaceChildren(linkify(entry.a));
  $("#btn-copy").hidden = false;
  $("#btn-listen").hidden = !("speechSynthesis" in window);
  showFeedback(entry.fb);
  renderSources(entry.sources);
  markLibrary(entry.sources);
  scrollTo({ top: $("#ask-form").offsetTop, behavior: reducedMotion ? "auto" : "smooth" });
}

function renderLog() {
  const items = readLog();
  const list = $("#log");
  list.replaceChildren();
  $("#btn-clear-log").hidden = !items.length;
  $("#btn-export").hidden = !items.length;
  if (!items.length) {
    $("#log-stats").textContent = "No journeys yet. Ask something and it will be saved here.";
    return;
  }
  const avg = Math.round(items.reduce((sum, e) => sum + e.ms, 0) / items.length);
  $("#log-stats").textContent = `${items.length} journey${items.length === 1 ? "" : "s"}, ${seconds(avg)} on average. Click one to bring it back.`;
  for (const entry of items) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "entry";
    btn.innerHTML = '<span class="log-q"></span><span class="log-a"></span><span class="log-m"></span>';
    btn.querySelector(".log-q").textContent = entry.q;
    btn.querySelector(".log-a").textContent = shorten(entry.a, 110);
    btn.querySelector(".log-m").textContent = `${entry.fb ? (entry.fb === "up" ? "Helpful · " : "Not helpful · ") : ""}${seconds(entry.ms)} · ${entry.k} passage${entry.k === 1 ? "" : "s"} · ${new Date(entry.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
    btn.addEventListener("click", () => restore(entry));
    li.appendChild(btn);
    list.appendChild(li);
  }
}

// ---------- key dialog ----------
const dialog = $("#key-dialog");
function openKeyDialog() {
  $("#key-input").value = loadKey();
  dialog.showModal();
  $("#key-input").focus();
}

// ---------- boot ----------
async function init() {
  buildState();
  renderLog();
  const root = document.documentElement;
  const paintSwatches = () => document.querySelectorAll(".swatch").forEach((s) => s.setAttribute("aria-checked", String(s.dataset.themePick === root.dataset.theme)));
  paintSwatches();
  document.querySelectorAll(".swatch").forEach((s) => s.addEventListener("click", () => {
    root.dataset.theme = s.dataset.themePick;
    paintSwatches();
    try { localStorage.setItem("waypoint.theme", root.dataset.theme); } catch { /* storage unavailable */ }
  }));
  for (const [id, value] of [["btn-up", "up"], ["btn-down", "down"]]) {
    $("#" + id).addEventListener("click", () => {
      const items = readLog();
      if (!items.length) return;
      items[0].fb = items[0].fb === value ? undefined : value;
      writeLog(items);
      showFeedback(items[0].fb);
      renderLog();
    });
  }
  $("#doc-filter").addEventListener("input", () => {
    const q = $("#doc-filter").value.toLowerCase();
    document.querySelectorAll(".doc").forEach((d) => { d.hidden = !d.textContent.toLowerCase().includes(q); });
  });
  $("#opt-k").addEventListener("input", (e) => ($("#opt-k-out").textContent = e.target.value));
  $("#btn-clear-log").addEventListener("click", () => { writeLog([]); renderLog(); });
  $("#btn-copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("#answer-text").textContent);
      toast("Answer copied.");
    } catch { toast("Couldn't copy. Select the text instead."); }
  });
  $("#btn-listen").addEventListener("click", () => {
    if (speechSynthesis.speaking) return speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance($("#answer-text").textContent));
  });
  $("#btn-export").addEventListener("click", () => {
    const md = "# Waypoint journey log\n\n" + readLog().map((e) =>
      `## ${e.q}\n\n${e.a}\n\n_${seconds(e.ms)} · ${e.k} passages · ${new Date(e.t).toLocaleString()}_\n\nSources:\n${e.sources.map((s) => `- ${s.text} (${Math.round(Math.max(0, s.score) * 100)}% match)`).join("\n")}\n`).join("\n");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([md], { type: "text/markdown" }));
    link.download = "waypoint-journey-log.md";
    link.click();
    URL.revokeObjectURL(link.href);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) {
      e.preventDefault();
      $("#question").focus();
    }
  });
  const examples = $("#examples");
  for (const text of EXAMPLES) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "example";
    chip.textContent = text;
    chip.addEventListener("click", () => {
      $("#question").value = text;
      $("#ask-form").requestSubmit();
    });
    examples.appendChild(chip);
  }

  try {
    health = await api.health();
    const layout = await api.graph();
    map = new LineMap($("#map"));
    map.draw(layout);
    renderStatus();
    await renderLibrary();
  } catch (err) {
    $("#chip-llm").textContent = "Server offline";
    $("#chip-llm").classList.add("bad");
    $("#map").innerHTML = '<p class="empty">Can\'t reach the Waypoint server. Start the backend, then reload this page.</p>';
    $("#btn-ask").disabled = true;
    return;
  }

  $("#ask-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = $("#question").value.trim();
    if (question.length < 2) {
      const error = $("#ask-error");
      error.textContent = "Type a question first.";
      error.hidden = false;
      return;
    }
    ask(question);
  });

  $("#add-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const field = $("#new-text");
    const text = field.value.trim();
    if (text.length < 3) return toast("Add some text first.");
    try {
      const { added } = await api.addDocument(text, "pasted");
      field.value = "";
      toast(`Added ${added.length} passage${added.length === 1 ? "" : "s"}.`);
      renderLibrary();
    } catch (err) {
      toast(err.message);
    }
  });

  $("#file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    if (/\.pdf$/i.test(file.name)) {
      try {
        const { added } = await api.upload(file);
        toast(`Added ${added.length} passage${added.length === 1 ? "" : "s"} from ${file.name}.`);
        renderLibrary();
      } catch (err) { toast(err.message); }
      event.target.value = "";
      return;
    }
    if (file.size > 200_000) return toast("Keep files under 200 KB.");
    $("#new-text").value = await file.text();
    toast(`Loaded ${file.name}. Review it, then add to library.`);
    event.target.value = "";
  });

  $("#btn-reset").addEventListener("click", async () => {
    await api.resetDocuments();
    toast("Library reset to the sample.");
    renderLibrary();
  });

  $("#btn-key").addEventListener("click", openKeyDialog);
  $("#key-cancel").addEventListener("click", () => dialog.close());
  $("#key-clear").addEventListener("click", () => {
    saveKey("");
    dialog.close();
    toast("API key removed from this browser.");
  });
  $("#key-form").addEventListener("submit", () => {
    const value = $("#key-input").value.trim();
    if (value) {
      saveKey(value);
      toast("API key saved in this browser.");
    }
  });
}

init();
