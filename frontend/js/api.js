const base = () => (window.WAYPOINT_CONFIG?.apiBase || "").replace(/\/$/, "");
const KEY_STORE = "waypoint.groqKey";

export const loadKey = () => {
  try { return localStorage.getItem(KEY_STORE) || ""; } catch { return ""; }
};
export const saveKey = (value) => {
  try {
    value ? localStorage.setItem(KEY_STORE, value) : localStorage.removeItem(KEY_STORE);
  } catch { /* storage unavailable */ }
};

export class ApiError extends Error {
  constructor(message, code = "error", status = 0) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

async function request(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (options.body instanceof FormData) delete headers["Content-Type"];
  const key = loadKey();
  if (key) headers["X-Groq-Key"] = key;
  let res;
  try {
    res = await fetch(base() + path, { ...options, headers });
  } catch {
    throw new ApiError("Can't reach the Waypoint server.", "offline");
  }
  if (!res.ok) {
    let detail = {};
    try { detail = (await res.json()).detail ?? {}; } catch { /* not json */ }
    const message = typeof detail === "string" ? detail : detail.message || `Request failed (${res.status}).`;
    throw new ApiError(message, detail.code || "error", res.status);
  }
  return res;
}

const json = async (path, options) => (await request(path, options)).json();

export const api = {
  health: () => json("/api/health"),
  graph: () => json("/api/graph"),
  documents: () => json("/api/documents"),
  addDocument: (text, source) => json("/api/documents", { method: "POST", body: JSON.stringify({ text, source }) }),
  removeDocument: (id) => json(`/api/documents/${id}`, { method: "DELETE" }),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return json("/api/documents/upload", { method: "POST", body: form });
  },
  resetDocuments: () => json("/api/documents/reset", { method: "POST" }),

  // Streams newline-delimited JSON events and calls onEvent for each one.
  async ask(question, k, onEvent) {
    const res = await request("/api/ask/stream", { method: "POST", body: JSON.stringify({ question, k }) });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line) onEvent(JSON.parse(line));
      }
    }
  },
};
