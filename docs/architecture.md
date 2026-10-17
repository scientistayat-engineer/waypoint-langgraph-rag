# Architecture notes

## Request flow

1. The browser sends `POST /api/ask/stream` with `{"question": "..."}` (and an optional `X-Groq-Key` header).
2. FastAPI checks that a key exists (server env var or header) unless demo mode is on.
3. The compiled LangGraph runs with `stream_mode="updates"`. After each node finishes, the backend sends an event.
4. The frontend plays the events back: it moves the train, fills the state rows, and renders sources and the answer.

The Groq key reaches the `generate` node through LangGraph's per-run config (`config["configurable"]["groq_api_key"]`), so it is never stored in the graph or on the server.

## Stream events (NDJSON, one JSON object per line)

| Event | Fields | Sent when |
| --- | --- | --- |
| `start` | `question`, `order` | The run begins. `order` lists the node names in run order |
| `node_start` | `node` | A station begins |
| `node_end` | `node`, `update`, `elapsed_ms` | A node returns. `update` is the state it wrote |
| `done` | `answer`, `sources`, `total_ms` | The run finished |
| `error` | `code`, `message` | Something failed mid-run (for example a rejected key) |

Typical order: `start`, `node_start retrieve`, `node_end retrieve`, `node_start generate`, `node_end generate`, `done`.

The frontend deliberately plays events back with a short minimum pause per station so quick runs are still visible. The `elapsed_ms` values shown on the map are the real server-side timings.

## HTTP errors

| Status | `detail.code` | Meaning |
| --- | --- | --- |
| 401 | `missing_api_key` | Real mode with no server key and no `X-Groq-Key` header |
| 502 | `llm_error` | The model call failed (`/api/ask` only; the stream sends an `error` event) |
| 409 | none | The library is full (60 passages) |
| 404 | none | Passage id not found |
| 422 | none | Validation failed (for example, a question under 2 characters) |

## Library

- Vector store: LangChain's `InMemoryVectorStore`, so it needs no external database and resets on restart.
- Text is split on blank lines, then packed into passages of at most 800 characters. Overlong paragraphs are split at sentence ends.
- Retrieval returns the top-`k` passages with cosine similarity scores.

## Demo mode

`DEMO_MODE=true` swaps in two small stand-ins from `app/demo.py`:

- `KeywordEmbeddings`: hashed bag-of-words vectors, so similar wording scores higher.
- `DemoLLM`: returns the best matching sentence from the retrieved context, or "I don't know" when nothing overlaps with the question.

Everything else (the graph, streaming, library, API) is the same code path as real mode.
