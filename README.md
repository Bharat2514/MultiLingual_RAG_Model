# PolyDoc — Multilingual RAG Assistant

Ask questions in **Hindi, English, or Hinglish** about **Hindi/English PDFs and
DOCX files** — including scanned documents — and get accurate, **source-cited**
answers in the language and register you asked in.

The hard parts, and what this repo actually solves:

| Challenge | Approach |
|---|---|
| **Cross-lingual retrieval** — a Hindi question must find the right English passage | `bge-m3` dense embeddings + learned sparse, fused with RRF, then cross-encoder reranked |
| **Scanned Hindi PDFs** — no text layer means empty context | Per-page text-density routing into Tesseract (`hin`+`eng`) with deskew/denoise preprocessing |
| **Hinglish** — romanized, code-mixed queries are out-of-distribution | Script detection → Devanagari transliteration → dual-query embedding, union before rerank |
| **Trust** — an unsourced answer can't be acted on | Inline citations to filename + page + snippet, and an enforced "not found in your documents" path |

Quality is measured, not asserted: see [Evaluation](#evaluation).

---

## Architecture

```
                ┌──────────────── INGESTION (async, Celery) ────────────────┐
 Upload  →  Validate  →  Text layer?  ──yes──►  Parse (PyMuPDF / python-docx)
 (PDF/DOCX)                   │ no                        │
                              ▼                           ▼
                    OCR (hin+eng, deskew)  ──►  Clean & normalize (NFC, de-hyphenate)
                                                          │
                                          Language detect → sentence-aware chunking
                                                          │
                                        bge-m3 dense + sparse → Qdrant upsert
                └───────────────────────────────────────────────────────────┘

                ┌──────────────────── QUERY (sync, streamed) ───────────────┐
 Query (hi/en/hinglish) → script detect → Hinglish normalize → embed →
   hybrid top-K (dense + sparse, RRF) → cross-encoder rerank → top-N →
   grounded prompt → Claude (streaming) → answer + citations → UI
                └───────────────────────────────────────────────────────────┘
```

| Layer | Choice |
|---|---|
| Backend | FastAPI (Python 3.11+), async SQLAlchemy 2 |
| Ingestion worker | Celery + Redis |
| Parsing / OCR | PyMuPDF, python-docx, Tesseract (`hin`, `eng`), OpenCV |
| Embeddings | `BAAI/bge-m3` — dense + learned sparse from one pass, run directly on `transformers` |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder |
| Vector DB | Qdrant (named vectors: `dense` + `sparse`, payload-filtered) |
| LLM | **Gemini** (`gemini-3-flash-preview`) or Claude — one swappable provider interface (PRD §8.3) |
| Metadata | PostgreSQL 16 |
| Frontend | Next.js 16 (App Router) + Tailwind v4, hand-rolled shadcn-style primitives |
| Eval | RAGAS + custom retrieval metrics |
| Observability | structlog, Prometheus |

Every model, store, and provider sits behind an interface and is config-driven
(NFR-9) — swap `EMBEDDING__MODEL_NAME` or `LLM__PROVIDER` without touching code.

---

## Quick start

**Prerequisites:** Docker, Python 3.11+, Node 20+, and one LLM key —
a [Gemini key](https://aistudio.google.com/apikey) (default) or an Anthropic key.

```bash
git clone <repo> && cd multilingual-rag

make setup-mac      # macOS only: Tesseract + Hindi traineddata + fonts
make setup          # venv, Python deps, .env from template
$EDITOR .env        # set GOOGLE_API_KEY  (or ANTHROPIC_API_KEY + LLM__PROVIDER=anthropic)

make check-llm         # verifies the key and probes Hindi + Hinglish output
make install-ml        # torch + transformers (the retrieval models)
make download-models   # pre-fetch bge-m3 + reranker (~4.6 GB) into ./data/models
make infra             # Postgres + Redis + Qdrant in Docker
make dev               # API on http://localhost:8000  (docs at /docs)
make worker            # ingestion worker (separate terminal)

make web-install       # frontend dependencies
make web               # UI on http://localhost:3000
```

Embeddings and reranking are **local** `bge-m3` models — the LLM key is used
only for answer generation, so the provider choice does not affect retrieval
quality or cost.

Recommended layout on Apple silicon: infrastructure in Docker, API and worker on
the **host**, so the embedding and reranker models get Metal (MPS) acceleration.
Containers fall back to CPU torch.

Fully containerised (reproducible, slower on Mac):

```bash
make up             # everything, including the frontend
```

Verify:

```bash
make health         # readiness probe across Postgres, Redis, Qdrant, LLM
make test           # unit tests
make lint           # ruff + mypy
```

---

## Make targets

| Target | Purpose |
|---|---|
| `make setup` | venv + dependencies + `.env` |
| `make setup-mac` | macOS system deps (Tesseract with Hindi) |
| `make install-ml` | torch + transformers (embedding/reranker stack) |
| `make download-models` | pre-fetch bge-m3 + reranker into `./data/models` |
| `make probe-crosslingual` | verify Hindi↔English retrieval with the real models |
| `make infra` / `make up` / `make down` | infra only / full stack / stop |
| `make dev` / `make worker` | run API / ingestion worker on the host |
| `make test` / `make test-all` / `make cov` | unit / full / coverage |
| `make lint` / `make fmt` | check / auto-fix |
| `make web` / `make web-install` | run / install the frontend |
| `make web-build` / `make web-lint` | production build / lint + typecheck |
| `make check-llm` | verify the LLM key, list reachable models, probe Hindi + Hinglish |
| `make health` | pretty-printed readiness probe |
| `make clean` | stop containers **and delete all volumes** |

---

## Configuration

All settings are environment-driven with typed defaults
([`backend/app/core/config.py`](backend/app/core/config.py)). Nested groups use a
double-underscore delimiter:

```bash
EMBEDDING__DEVICE=mps        # auto | mps | cuda | cpu
RETRIEVAL__TOP_K=20          # hybrid candidates before rerank
RETRIEVAL__TOP_N=5           # chunks passed to the LLM
OCR__MIN_CHARS_PER_PAGE=50   # below this, a page is routed to OCR
LLM__PROVIDER=gemini         # gemini | anthropic | mock
LLM__MODEL=                  # blank = provider default
```

Switching LLM vendor is a two-line change — set `LLM__PROVIDER` and the matching
key. Nothing else in the pipeline moves.

### Gemini quota note

A model appearing in `models.list()` does **not** mean your key can call it. Free
tier grants `limit: 0` on the Pro models, and `gemini-2.5-flash` returns 404
("no longer available to new users") on recently-created projects. `make
check-llm` probes each candidate for real and reports the first with usable
quota, so you get a concrete `LLM__MODEL=` value instead of a runtime 429.

Free tier is fine for development. The **evaluation harness (§12) issues
thousands of calls** across 150+ gold questions × ablation arms — budget for a
billed key before that run, or expect it to take days of throttling.

See [`.env.example`](.env.example) for the annotated full list.

---

## API

Base URL `/api/v1`. Every response carries a `request_id`; every error uses one
envelope:

```json
{ "request_id": "req_789",
  "error": { "code": "UNSUPPORTED_FILE_TYPE", "message": "Only PDF and DOCX are supported." } }
```

| Method | Path | Purpose | Status |
|---|---|---|---|
| `GET` | `/health`, `/health/ready` | Liveness / dependency readiness | ✅ |
| `POST` | `/documents` | Upload one or more files (partial success per file) | ✅ |
| `GET` | `/documents` | List documents in this session | ✅ |
| `GET` | `/documents/{id}` | Ingestion status with per-stage progress | ✅ |
| `DELETE` | `/documents/{id}` | Delete document, file, chunks, and vectors | ✅ |
| `POST` | `/search` | Retrieval only — for eval metrics and ablations | ✅ |
| `POST` | `/chat` | Grounded answer + citations, optionally streamed (SSE) | ✅ |

Every request is scoped by an `X-Session-ID` header. That scope is applied
**inside the database query**, not as a check afterwards, so a forgotten filter
cannot leak another session's documents. A cross-session request returns `404`,
never `403` — existence itself is not disclosed.

Interactive docs at `/docs`; Prometheus metrics at `/metrics`.

---

## Evaluation

v1 is "done" only when the measured targets are met on a trilingual gold set of
≥150 Q/A pairs spanning Hindi, English, Hinglish, cross-lingual, and adversarial
out-of-scope questions.

| Metric | Target |
|---|---|
| Retrieval Recall@5 (cross-lingual) | ≥ 0.80 |
| RAGAS faithfulness | ≥ 0.85 |
| RAGAS answer relevancy | ≥ 0.85 |
| Language fidelity | ≥ 0.95 |
| Anti-hallucination correctness | ≥ 0.95 |
| P95 chat latency | ≤ 9 s |

Ablations reported alongside: dense-only vs hybrid vs hybrid+rerank, with/without
Hinglish normalization, chunk-size sweep, and an embedding-model A/B.

---

## Build status

| Batch | Scope | Status |
|---|---|---|
| 1 | Foundation: config, structured logging, error envelope, data model, Docker Compose, health | ✅ |
| 2 | Ingestion: parsing, per-page OCR routing, cleaning, chunking, Celery pipeline, document API | ✅ |
| 3 | Index & retrieval: bge-m3 dense+sparse, Qdrant hybrid + RRF, cross-encoder rerank, `/search` | ✅ |
| 4 | Query & generation: Hinglish dual-query, grounding contract, citations, SSE streaming | ✅ |
| 5 | Next.js UI: upload + live ingestion progress, streamed chat, clickable citations | ✅ |
| 6 | Eval harness, ablations, observability, docs | ⏳ |

---

## Ingestion pipeline

`upload → validate → parse → OCR (per page) → clean → chunk → index`

Each stage writes an `ingestion_jobs` row, so `GET /documents/{id}` returns real
progress rather than a spinner:

```
parse    succeeded    197.77 ms
ocr      succeeded     11.25 ms   text layer sufficient on every page
clean    succeeded      6.15 ms
chunk    succeeded     36.21 ms
```

Details worth knowing:

- **File type comes from magic bytes**, never the filename or client MIME type.
  A `.zip` renamed to `.docx` is rejected; the Word-specific part must be present.
- **OCR is routed per page, not per document.** Real Hindi circulars are a
  scanned cover sheet plus born-digital annexures — an all-or-nothing choice
  either wastes minutes of OCR or drops the scanned pages entirely. A page is
  OCR'd when its text layer falls below `OCR__MIN_CHARS_PER_PAGE`, and the result
  is kept only if it beats what the text layer already had.
- **NFC normalisation is the highest-impact Hindi fix.** `क` + nukta and `क़` are
  the same word in different byte sequences; without normalising, they produce
  different embeddings and never match.
- **Furniture stripping refuses to eat body text.** If removing repeated
  headers/footers would discard more than half a page, the page is kept intact —
  losing a header is cosmetic, losing the body is silent data loss.
- **Chunks never start mid-sentence**, and carry the page range plus character
  span that makes a citation clickable back to the source.
- If the Celery broker is unreachable, ingestion runs **inline** via background
  tasks, so a Redis outage degrades throughput instead of dropping uploads.

## Retrieval

`query → embed → hybrid search (dense + sparse, RRF) → cross-encoder rerank → top-N`

- **One Qdrant collection, two named vectors.** `dense` and `sparse` both come
  from a single bge-m3 forward pass, so they cannot drift apart. Fusion runs
  server-side via Qdrant's Query API rather than shipping two candidate lists to
  the client to merge.
- **Learned sparse, not BM25.** BM25 needs language-specific tokenisation and
  stemming, and there is no good Hindi stemmer in the usual stacks. bge-m3's
  lexical weights are multilingual by construction, so exact terms — clause IDs,
  names, numbers — stay retrievable in both scripts.
- **RRF is rank-based on purpose.** Dense cosine and sparse lexical scores live
  on incomparable scales; any weighted sum of raw scores would be arbitrary.
- **Reranking is the precision lever.** A bi-encoder never sees query and
  document together; the cross-encoder scores the pair. It only has to run over
  `top_k` candidates, so the cost is bounded.
- **Scope is a required argument, not an option.** `SearchFilter.session_id` has
  no default, so an unscoped search cannot be expressed. Payload indexes keep
  that filter fast instead of turning it into a post-hoc scan.
- **Graceful degradation** (NFR-5): if the reranker fails to load, retrieval
  order is used; if a query yields no lexical weights, hybrid quietly becomes
  dense.

### Measured on `eval/crosslingual_probe.py` (`make probe-crosslingual`)

7 probes over a bilingual corpus where the answer deliberately lives in the
*other* language, using the real bge-m3 and reranker:

| mode | recall@1 | recall@3 |
|---|---|---|
| dense | **100%** | **100%** |
| sparse | 43% | 43% |
| hybrid (0.75 / 0.25) | 57% | 86% |
| **hybrid + rerank** | **100%** | — |

The interesting result is that **hybrid alone scores below dense alone** on
cross-lingual queries. That is not a bug: lexical overlap between a Hindi query
and an English passage is essentially zero, so the sparse arm contributes noise
on exactly the queries this product exists for. Two consequences, both
implemented:

1. Fusion is **weighted** (dense 0.75 / sparse 0.25) and runs client-side —
   Qdrant's built-in RRF is equal-weight only.
2. The **reranker is what makes hybrid safe**. It recovers 100% recall@1 from an
   86% recall@3 candidate set, which is exactly the division of labour intended:
   retrieval optimises recall, reranking optimises precision.

This is a fast smoke test, not the §12 evaluation — the trilingual gold set and
the full ablation grid land in Batch 6, and should set the final weights.

Ablations from §12.4 are driven straight off `POST /search`:

```bash
curl -X POST localhost:8000/api/v1/search -H 'X-Session-ID: s1' \
  -H 'content-type: application/json' \
  -d '{"query":"पेंशन के लिए कितने साल की सेवा चाहिए?","mode":"hybrid","rerank":true}'
```

`mode` is `dense | sparse | hybrid`; `rerank` toggles the cross-encoder.

## Answering

`query → detect register → dual-query → retrieve → ground → generate → cite`

```bash
curl -X POST localhost:8000/api/v1/chat -H 'X-Session-ID: s1' \
  -H 'content-type: application/json' \
  -d '{"query":"late submission par kitna jurmana lagta hai","stream":false}'
```

**Language fidelity (FR-21).** The answer comes back in the register the
question was asked in — verified live against a real English document:

| question | answer |
|---|---|
| `What LLM evaluation work has this person done?` | English, cited |
| `इस व्यक्ति ने AI मूल्यांकन का क्या काम किया है?` | Devanagari Hindi, cited |
| `is candidate ka experience kya hai, mujhe batao` | `"Candidate ke experience ke baare mein…"` — romanized Hinglish, cited |
| `What is the capital of France?` | `"I couldn't find this in your documents."` |

That last row is the point of the whole grounding contract: the model **knows**
the capital of France and still refuses, because it isn't in the corpus (FR-23,
US4 — the §12.3 anti-hallucination target).

**Hinglish is code-mixed, so transliteration must be too.** `"mujhe is document
ka summary do"` becomes `"मुझे इस document का summary दो"` — English words are
left alone, because mangling `document` into `दोचुमेन्त्` produces a token that
matches nothing and pollutes the query vector. Both the original and the
transliteration are searched, and the candidate lists fused (FR-15).

**Grounding is enforced by us, not requested of the model:**

- nothing retrieved → answer "not found" **without calling the LLM at all**;
- the model signals insufficient context with a fixed sentinel, which never
  reaches the client — it is replaced by the localised message;
- every citation is resolved against the chunks actually sent, and any
  reference to a document that wasn't in the context is **discarded** — a
  fabricated citation is worse than none, since it looks verifiable;
- an answer with no resolvable citation is returned with `grounded: false` so
  the UI can mark it rather than present it as sourced.

Streaming (`"stream": true`) emits SSE: `start`, many `token`, one `done`
carrying citations, usage, and latency.

### Measured latency (NFR-2)

| stage | cold (first query) | warm |
|---|---|---|
| embed + retrieve | 7.4s → **1.3s** | **55 ms** |
| rerank (top-20) | 7.1s → **0.4s** | **310 ms** |
| generation (Gemini free tier) | — | **10–12 s** |

Retrieval comfortably meets the budget. **Generation does not, and that is a
quota constraint rather than a code one:** on this free-tier key,
`gemini-3-flash-preview` takes ~12s to return even a *single* token (measured
directly, with thinking confirmed disabled), and every alternative model on the
key is 429/404. NFR-2's ≤9s P95 needs a billed key or a faster model tier —
nothing in the pipeline changes.

Startup warmup is why the "cold" column improves: Metal compiles its kernels on
first use, so the models are exercised during app startup and the first real
query no longer pays a 40x penalty.

## Interface

A two-column workspace: documents on the left, conversation on the right.

- **Upload** (FR-26) — drag-and-drop, and progress reflects the *real* pipeline
  (`Reading document → Running OCR → Splitting into passages → Building the
  index`) rather than an indeterminate spinner. A file rejected for type or size
  is reported per file without failing the rest of the batch.
- **Citations** (FR-28, US3) — the model's `[policy.pdf p.4]` markers become
  numbered chips inline in the answer; clicking one expands the source passage.
  **A citation the backend could not verify is rendered as plain text, never as
  a clickable chip** — an unverifiable reference must not look interactive.
- **Trust signals** — every settled answer carries a `Grounded` or `Unverified`
  badge, the detected language, latency, and token count. OCR-derived documents
  are badged with the share of pages that needed it.
- **Streaming** (FR-25/27) — tokens render as they arrive with a live caret;
  auto-scroll follows the stream but never yanks the view if you have scrolled
  up to read.
- **Devanagari** (NFR-10) — Noto Sans Devanagari is loaded explicitly rather
  than left to OS fallback (which mis-renders matras and conjuncts on many
  Windows installs), with taller leading on any element that can hold Hindi.
- **Theme** — light/dark with an inline pre-paint script so there is no flash,
  and both palettes were picked to meet WCAG AA contrast.
- **Empty and error states** (FR-30) — distinct copy for "no documents yet",
  "upload one to begin asking", API-unreachable, and per-file rejection.

Session identity lives in `localStorage` and is sent as `X-Session-ID` on every
call. "New workspace" mints a fresh key, which is genuinely empty because the
backend filters on it inside the query.

## Project layout

```
backend/
  app/
    core/         config, logging, errors, request context, device selection
    db/           SQLAlchemy models, sessions, bootstrap
    api/v1/       routers
    ingestion/    parsing, OCR, cleaning, chunking      (Batch 2)
    retrieval/    embeddings, vector store, rerank      (Batch 3)
    generation/   prompt contract, LLM providers        (Batch 4)
    workers/      Celery app and tasks                  (Batch 2)
  tests/
frontend/
  src/app/        layout, page shell, theme tokens
  src/components/ document panel, chat panel, answer + citations
  src/lib/        typed API client, SSE reader, session/theme stores
eval/             gold set, harness, ablations          (Batch 6)
docs/PRD.md       product requirements
```

---

## Security & privacy

Documents are scoped per session/user and that filter is enforced at the
retrieval layer, not just the API surface (isolation tests ship in Batch 3).
Secrets come from the environment only. Hard delete removes the file, its rows,
and its vectors. Aligned with India DPDP Act principles; see §16 of the PRD.
