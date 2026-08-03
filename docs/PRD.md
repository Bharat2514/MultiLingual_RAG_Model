# Product Requirements Document (PRD)
## Multilingual RAG Assistant — "PolyDoc"

| Field | Value |
|---|---|
| **Document title** | Multilingual RAG Assistant — Product Requirements Document |
| **Version** | 1.0 |
| **Status** | Draft for review |
| **Author** | Utsav Jain (ML Engineer) |
| **Reviewers** | Engineering Manager, Senior ML Engineer, Product |
| **Last updated** | 2026-07-20 |
| **Related docs** | Architecture Design Doc (TBD), Eval Plan (§12), API Spec (§9) |

---

## 1. Document Control

### 1.1 Revision history
| Version | Date | Author | Change summary |
|---|---|---|---|
| 0.1 | 2026-07-20 | Utsav Jain | Initial skeleton |
| 1.0 | 2026-07-20 | Utsav Jain | First complete draft for review |

### 1.2 Glossary
| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — LLM answers grounded in retrieved document chunks, reducing hallucination. |
| **Chunk** | A contiguous span of document text (target 300–800 tokens) that is embedded and indexed as a retrieval unit. |
| **Embedding** | A dense vector representation of text used for semantic similarity search. |
| **Cross-lingual retrieval** | Retrieving relevant content in language A using a query in language B (e.g., Hindi query → English passage). |
| **Hinglish** | Romanized Hindi written in Latin script, often code-mixed with English (e.g., "mujhe iska summary do"). |
| **Devanagari** | The script used to write Hindi. |
| **BM25** | A sparse lexical (keyword) ranking function; complements dense semantic search. |
| **Reranker** | A cross-encoder model that re-scores retrieved candidates for higher precision. |
| **Hybrid search** | Fusion of dense (semantic) + sparse (lexical) retrieval, typically via Reciprocal Rank Fusion (RRF). |
| **RRF** | Reciprocal Rank Fusion — a rank-based method to merge multiple ranked lists. |
| **Grounding / citation** | Attaching source chunk references to a generated answer so it can be verified. |
| **RAGAS** | An open-source framework for automated RAG evaluation (faithfulness, relevancy, etc.). |
| **P50 / P95** | 50th / 95th percentile latency. |

---

## 2. Executive Summary

PolyDoc is a multilingual, document-grounded question-answering assistant. A user uploads one or more documents (PDF or Word) in **Hindi or English**, then asks questions in **Hindi, English, or Hinglish** and receives accurate, **source-cited** answers in the language/register they asked in.

The core technical challenge — and the reason this is more than a "chat with PDF" clone — is **cross-lingual retrieval** (a Hindi question must find the right English passage and vice versa), **robust ingestion of Hindi PDFs** (including scanned/OCR cases), and **Hinglish understanding** (romanized, code-mixed queries). Success is measured not by "it works on a demo" but by quantified retrieval and answer-quality metrics on a curated trilingual evaluation set.

---

## 3. Problem Statement & Motivation

### 3.1 Problem
Knowledge in Indian organizations, government, education, and legal domains is spread across Hindi and English documents. Users who think in Hindi or type in Hinglish are poorly served by English-only document assistants. Existing "chat with PDF" tools:
- Fail on **scanned Hindi PDFs** (no text layer → empty context).
- Do **poor cross-lingual retrieval** (query and document in different languages retrieve nothing useful).
- Cannot handle **Hinglish** input (romanized Hindi is out-of-distribution for most pipelines).
- Provide **no citations**, so answers can't be trusted for real decisions.

### 3.2 Motivation / Opportunity
A system that ingests messy real-world Hindi/English documents and answers trilingually with citations is directly useful for education, public-sector document navigation, customer support, and internal knowledge bases — and demonstrates end-to-end ML engineering depth (document AI, multilingual NLP, retrieval, evaluation, productionization).

---

## 4. Goals & Non-Goals

### 4.1 Goals (in scope for v1)
- G1. Ingest **text and scanned** PDFs and **DOCX** files in Hindi and English.
- G2. Support queries in **Hindi (Devanagari), English, and Hinglish (romanized)**.
- G3. Perform **cross-lingual** hybrid retrieval with reranking.
- G4. Generate **accurate, grounded answers with inline citations**, in the user's language/register.
- G5. Provide a **measurable evaluation harness** (retrieval + generation metrics) on a trilingual test set.
- G6. Ship a usable **web UI** and a documented **REST API**.
- G7. Handle **multi-document** corpora with per-session or per-user document scoping.

### 4.2 Non-Goals (explicitly out of scope for v1)
- Languages beyond Hindi and English (no Bengali, Tamil, etc. in v1).
- Multi-turn agentic tool use / web browsing.
- Real-time collaborative editing of documents.
- Fine-tuning a custom foundation LLM (we use pretrained + prompting; fine-tuning is a future option).
- Handwriting recognition (printed OCR only).
- Documents with heavy tabular/financial reconciliation semantics (basic table extraction only).
- Mobile native apps (responsive web only).

### 4.3 Assumptions
- Documents are printed (typed), not handwritten.
- A single document is ≤ ~300 pages and ≤ 50 MB.
- Internet access is available for managed LLM/embedding APIs (a fully-local fallback is a stretch goal).
- Users accept a few seconds of latency for higher accuracy.

---

## 5. User Personas & Use Cases

### 5.1 Personas
| Persona | Description | Primary need |
|---|---|---|
| **Student (Riya)** | Reads Hindi & English study material | Ask questions in Hinglish, get simple explanations with sources |
| **Govt/office worker (Suresh)** | Navigates Hindi circulars/policy PDFs (often scanned) | Extract specific clauses; trust via citations |
| **Support agent (Anita)** | English KB, Hindi-speaking customers | Ask in Hindi, answer grounded in English docs |
| **Developer (integration)** | Wants to embed PolyDoc in another app | Stable REST API, predictable latency/cost |

### 5.2 Key user stories
- US1: As a user, I upload a Hindi scanned PDF and ask a question in Hinglish, and get a correct answer in Hinglish with a citation.
- US2: As a user, I ask in Hindi about an English document and get the correct English fact translated into Hindi.
- US3: As a user, I see exactly which page/chunk an answer came from and can click to verify.
- US4: As a user, when the answer isn't in my documents, the system says "I couldn't find this in your documents" instead of hallucinating.
- US5: As a developer, I POST a query to `/chat` and receive a JSON answer with citations and latency metadata.

### 5.3 Acceptance criteria (sample, US4 — anti-hallucination)
- **Given** a corpus with no relevant content, **when** the user asks an out-of-scope question, **then** the system returns an explicit "not found in your documents" message and does **not** fabricate an answer, in ≥ 95% of a 50-question adversarial set.

---

## 6. Functional Requirements

### 6.1 Document ingestion
- FR-1. Accept `.pdf`, `.docx` uploads (single and batch). Reject others with a clear error.
- FR-2. Enforce limits: ≤ 50 MB/file, ≤ 300 pages, configurable max files/session.
- FR-3. Detect whether a PDF has a text layer; if not (or if text density is below a threshold), route to **OCR**.
- FR-4. OCR must support Hindi (`hin`) and English (`eng`); auto-detect per page.
- FR-5. Extract text preserving reading order and page numbers; basic table/heading structure best-effort.
- FR-6. Detect document language(s) per chunk and store as metadata.
- FR-7. Clean text: fix Devanagari Unicode normalization (NFC), strip headers/footers/watermarks heuristically, de-hyphenate line breaks.
- FR-8. Chunk sentence-aware (Devanagari-aware sentence boundaries), target 300–800 tokens with 10–15% overlap; never split mid-sentence where avoidable.
- FR-9. Persist per-chunk metadata: `doc_id`, `page`, `chunk_index`, `language`, `source_filename`, `char_span`.

### 6.2 Indexing
- FR-10. Generate dense embeddings using a multilingual, cross-lingual model.
- FR-11. Generate sparse (BM25/SPLADE-style) representations for hybrid search.
- FR-12. Store vectors + metadata in a vector DB with per-`doc_id`/`session_id` filtering.
- FR-13. Support incremental add and full delete of a document's chunks.

### 6.3 Query handling
- FR-14. Detect query language/script (Devanagari Hindi, English, or romanized Hinglish).
- FR-15. Normalize Hinglish → Devanagari (transliteration) and/or preserve original; embed a query representation that maximizes cross-lingual recall.
- FR-16. Optional query rewriting/expansion for short or ambiguous queries.
- FR-17. Retrieve top-K (config, default K=20) via hybrid search; fuse with RRF.
- FR-18. Rerank fused candidates with a cross-encoder; keep top-N (config, default N=5).
- FR-19. Enforce document/session scoping in retrieval (no cross-user leakage).

### 6.4 Answer generation
- FR-20. Assemble a prompt: system instructions + top-N chunks (with source labels) + user query.
- FR-21. Instruct the LLM to answer **in the user's language/register** (Hindi→Hindi, Hinglish→Hinglish, English→English).
- FR-22. Require **inline citations** mapping claims to source chunks.
- FR-23. Enforce grounding: if context is insufficient, respond with an explicit "not found" message (no fabrication).
- FR-24. Return structured output: `answer`, `citations[]`, `language_detected`, `usage/latency`.
- FR-25. Support streaming responses to the UI.

### 6.5 UI
- FR-26. Upload panel with progress and ingestion status per file.
- FR-27. Chat interface with language auto-handling and streamed answers.
- FR-28. Citations rendered as clickable references showing source filename + page + snippet.
- FR-29. Document management (list, delete) scoped to the session/user.
- FR-30. Error/empty states (upload failed, OCR failed, no answer found).

### 6.6 API
- FR-31. REST endpoints (see §9) for upload, ingest status, chat, document list/delete, health.
- FR-32. Consistent JSON schema and error codes; request IDs for tracing.

---

## 7. Non-Functional Requirements (NFRs)

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | **Accuracy** | Meet the target metrics in §12.3 on the trilingual eval set before "done". |
| NFR-2 | **Latency** | P50 end-to-end chat ≤ 4 s, P95 ≤ 9 s (with managed LLM). Ingestion: text PDF ≤ 5 s/100 pages; OCR ≤ 30 s/100 pages (async). |
| NFR-3 | **Throughput** | Handle ≥ 10 concurrent chat requests in v1 (single node) without errors. |
| NFR-4 | **Cost** | Track and cap per-query LLM+embedding cost; log token usage; target ≤ $X/query (set during design). |
| NFR-5 | **Reliability** | Graceful degradation: if reranker/LLM API fails, fall back (retrieval-only answer or retry with backoff). No unhandled 500s to the user. |
| NFR-6 | **Security/Privacy** | Per-user document isolation; documents encrypted at rest; no training on user data; configurable data-retention/delete. |
| NFR-7 | **Observability** | Structured logs, per-stage latency, retrieval hit logs, eval dashboards. |
| NFR-8 | **Scalability** | Stateless API workers; vector DB and object storage externalized so the app can scale horizontally. |
| NFR-9 | **Maintainability** | Swappable components (embedding model, LLM, vector DB) behind interfaces; config-driven. |
| NFR-10 | **Accessibility/i18n** | UI supports Devanagari rendering, RTL-safe layout, proper fonts; WCAG AA color contrast. |
| NFR-11 | **Compliance** | Respect India DPDP Act principles for personal data; document data flows. |

---

## 8. System Architecture

### 8.1 High-level flow
```
                    ┌──────────────── INGESTION (async) ────────────────┐
 Upload (PDF/DOCX) → Validate → Text-layer? ──yes──► Parse (PyMuPDF / python-docx)
                                    │no                        │
                                    ▼                          ▼
                              OCR (Hindi/Eng) ──────► Clean & normalize (NFC, de-hyphenate)
                                                              │
                                                              ▼
                                        Language detect → Sentence-aware chunking
                                                              │
                                              Dense embed (bge-m3) + Sparse (BM25)
                                                              │
                                                     Vector DB (Qdrant) upsert
                                                              │
                                              Ingestion status → user
                    └────────────────────────────────────────────────────┘

                    ┌──────────────────── QUERY (sync) ─────────────────┐
 User query (Hi/En/Hinglish) → Lang/script detect → Hinglish normalize →
   Embed query → Hybrid retrieve top-K (dense+sparse, RRF) → Rerank → top-N →
   Build grounded prompt → LLM generate (stream) → answer + citations → UI
                    └────────────────────────────────────────────────────┘
```

### 8.2 Components
| Component | Responsibility | Default tech |
|---|---|---|
| **API Gateway / Backend** | REST endpoints, auth, orchestration | FastAPI (Python 3.11+) |
| **Ingestion worker** | Parse, OCR, clean, chunk, embed, index (async) | Celery/RQ + Redis (or FastAPI BackgroundTasks for v1) |
| **Parser** | PDF/DOCX text extraction | PyMuPDF, python-docx |
| **OCR engine** | Scanned Hindi/English → text | Tesseract (`hin`,`eng`) or Surya; vision-LLM fallback |
| **Embedding model** | Cross-lingual dense vectors | `BAAI/bge-m3` (or `intfloat/multilingual-e5-large`) |
| **Sparse index** | Lexical recall | BM25 (Qdrant sparse vectors) |
| **Reranker** | Precision re-scoring | `BAAI/bge-reranker-v2-m3` |
| **Vector DB** | ANN + metadata filtering + hybrid | Qdrant (Docker) |
| **LLM** | Grounded generation | Claude (Opus/Sonnet) primary; Gemini / Indic open model as alternates |
| **Object storage** | Original files | Local FS (dev) → S3-compatible (prod) |
| **Metadata store** | Docs, chunks, sessions, jobs | PostgreSQL (or SQLite for dev) |
| **Cache** | Query/embedding cache, job queue | Redis |
| **Frontend** | Upload + chat UI | Streamlit (v1) → Next.js (v2) |
| **Eval harness** | Automated quality metrics | RAGAS + custom scripts |
| **Observability** | Logs, traces, metrics | structlog + Prometheus/Grafana (or Langfuse for LLM traces) |

### 8.3 Key design decisions (with rationale)
- **Hybrid search + reranking** over dense-only: biggest single accuracy lever, especially for exact terms (names, numbers, clause IDs) that pure semantics miss.
- **bge-m3 embeddings**: strong cross-lingual Hindi↔English performance and supports dense+sparse+multi-vector in one model.
- **Managed LLM for generation**: current frontier models handle Hindi and Hinglish register far better than most open models; LLM kept swappable to allow local/offline deployment later.
- **Hinglish normalization before retrieval**: romanized queries otherwise embed poorly against Devanagari/English content.
- **OCR routing by text-density check**: avoids running expensive OCR on native-text PDFs while still catching scanned Hindi documents.
- **Explicit grounding contract**: the prompt and post-checks force "say I don't know" behavior to protect trust.

---

## 9. API Specification (v1)

Base URL: `/api/v1`. All responses include `request_id`. Auth: bearer token (v1 may use a simple session key).

### 9.1 Endpoints
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents` | Upload one/more files (multipart). Returns `job_id`s. |
| `GET` | `/documents/{doc_id}` | Get ingestion status + metadata. |
| `GET` | `/documents` | List documents in the session/user scope. |
| `DELETE` | `/documents/{doc_id}` | Delete a document and its chunks. |
| `POST` | `/chat` | Ask a question; returns grounded answer + citations. |
| `GET` | `/health` | Liveness/readiness. |

### 9.2 `POST /chat` — request
```json
{
  "session_id": "abc-123",
  "query": "mujhe is document ka summary do",
  "doc_ids": ["doc_1", "doc_2"],
  "top_k": 20,
  "top_n": 5,
  "stream": false
}
```

### 9.3 `POST /chat` — response
```json
{
  "request_id": "req_789",
  "answer": "Is document mein ... (Hinglish answer)",
  "language_detected": "hinglish",
  "citations": [
    {
      "doc_id": "doc_1",
      "filename": "policy.pdf",
      "page": 4,
      "chunk_index": 12,
      "snippet": "…source text…",
      "score": 0.83
    }
  ],
  "grounded": true,
  "usage": { "input_tokens": 1450, "output_tokens": 220 },
  "latency_ms": { "retrieval": 180, "rerank": 90, "generation": 2600, "total": 2980 }
}
```

### 9.4 Error model
```json
{ "request_id": "req_789", "error": { "code": "UNSUPPORTED_FILE_TYPE", "message": "Only PDF and DOCX are supported." } }
```
Error codes: `UNSUPPORTED_FILE_TYPE`, `FILE_TOO_LARGE`, `OCR_FAILED`, `NO_RELEVANT_CONTEXT`, `LLM_UNAVAILABLE`, `RATE_LIMITED`, `INTERNAL`.

---

## 10. Data Model

### 10.1 Entities (PostgreSQL)
- **users**(`user_id`, `created_at`, ...)
- **sessions**(`session_id`, `user_id`, `created_at`)
- **documents**(`doc_id`, `user_id`, `filename`, `mime`, `pages`, `language_primary`, `status` [`queued|parsing|ocr|indexing|ready|failed`], `size_bytes`, `created_at`)
- **ingestion_jobs**(`job_id`, `doc_id`, `stage`, `status`, `error`, `started_at`, `finished_at`)
- **chunks** (metadata mirror; vectors live in Qdrant): (`chunk_id`, `doc_id`, `page`, `chunk_index`, `language`, `char_span`, `text_hash`)

### 10.2 Vector DB (Qdrant) payload
```json
{
  "chunk_id": "…", "doc_id": "…", "user_id": "…", "session_id": "…",
  "page": 4, "chunk_index": 12, "language": "hi", "filename": "policy.pdf",
  "text": "…chunk text…"
}
```
Collection uses named vectors: `dense` (bge-m3) + `sparse` (BM25). Payload indexes on `doc_id`, `user_id`, `session_id` for filtering.

---

## 11. Detailed Processing Specs

### 11.1 OCR routing
- Compute extracted-char-count per page from the text layer. If `chars_per_page < threshold` (e.g., < 50) for a page → OCR that page. Store `ocr_used` flag per chunk.

### 11.2 Chunking
- Split on Devanagari + Latin sentence boundaries (use an Indic-aware sentence splitter; fall back to regex on `।`, `.`, `?`, `!`).
- Target 300–800 tokens; overlap 10–15%; attach page + char span.

### 11.3 Hinglish handling
- Script detect: if query is Latin-script but classified as Hindi/code-mixed → transliterate to Devanagari (e.g., `indic-transliteration`) and build a query that maximizes recall (embed original + transliterated; take union of candidates before rerank).
- Answer language: mirror the input register (detected language) in generation instructions.

### 11.4 Prompt contract (generation)
- System: role, "answer only from provided context", "cite sources as [filename p.N]", "if not in context, say you couldn't find it", "reply in the user's language/register".
- Context: numbered top-N chunks with source labels.
- Guardrail post-check: if answer contains claims with no citation, flag/regenerate (best-effort in v1).

---

## 12. Evaluation Plan (this is what makes it defensible)

### 12.1 Test set
- Build a **trilingual gold set**: ≥ 150 Q/A pairs across Hindi, English, Hinglish, over a fixed document corpus (mix of text + scanned PDFs, Hindi + English docs). Include:
  - Cross-lingual questions (Hindi Q on English doc, and vice versa).
  - Adversarial out-of-scope questions (should return "not found").
  - Numeric/name/clause-lookup questions (tests lexical recall).

### 12.2 Metrics
- **Retrieval**: Recall@K, MRR, nDCG@N (does the gold chunk appear / rank high?).
- **Generation (RAGAS)**: faithfulness, answer relevancy, context precision, context recall.
- **Answer correctness**: exact/semantic match vs. gold (LLM-as-judge + human spot check).
- **Language fidelity**: % answers in the correct language/register.
- **Anti-hallucination**: % correct "not found" on adversarial set.
- **Ops**: P50/P95 latency, cost/query.

### 12.3 Target thresholds (v1 "done" definition)
| Metric | Target |
|---|---|
| Retrieval Recall@5 (cross-lingual) | ≥ 0.80 |
| RAGAS faithfulness | ≥ 0.85 |
| RAGAS answer relevancy | ≥ 0.85 |
| Language fidelity | ≥ 0.95 |
| Anti-hallucination correctness | ≥ 0.95 |
| P95 chat latency | ≤ 9 s |

### 12.4 Ablations to run (for the write-up)
- Dense-only vs. hybrid vs. hybrid+rerank (quantify each lift).
- With vs. without Hinglish normalization.
- Chunk size / overlap sweep.
- Embedding model A/B (bge-m3 vs. e5-large).

---

## 13. Tech Stack Summary

| Layer | Primary | Alternatives |
|---|---|---|
| Language | Python 3.11+ | — |
| RAG framework | LlamaIndex | LangChain |
| Parsing | PyMuPDF, python-docx | pdfplumber, unstructured |
| OCR | Tesseract (`hin`,`eng`) | Surya, vision-LLM |
| Embeddings | bge-m3 | multilingual-e5-large |
| Reranker | bge-reranker-v2-m3 | Cohere Rerank |
| Vector DB | Qdrant | pgvector, Chroma, Weaviate |
| LLM | Claude (Opus/Sonnet) | Gemini, Indic open models |
| Backend | FastAPI | — |
| Queue/cache | Redis + Celery/RQ | BackgroundTasks (v1) |
| DB | PostgreSQL | SQLite (dev) |
| Frontend | Streamlit → Next.js | Gradio |
| Eval | RAGAS + custom | TruLens |
| Observability | structlog, Langfuse, Prometheus/Grafana | — |
| Deploy | Docker Compose (v1) | Kubernetes (later) |

---

## 14. Milestones & Timeline (indicative, ~6 weeks part-time)

| Phase | Deliverable | Exit criteria |
|---|---|---|
| **M0 — Setup** | Repo, Docker Compose (Qdrant, Postgres, Redis), config, CI | `docker compose up` works; health check green |
| **M1 — English text RAG** | Ingest text PDF/DOCX → embed → retrieve → generate → Streamlit | End-to-end English Q&A with citations |
| **M2 — Hindi + OCR** | OCR routing, bge-m3, cross-lingual retrieval | Hindi & scanned docs answerable; cross-lingual recall measured |
| **M3 — Hybrid + rerank** | BM25 + RRF + reranker | Recall@5 ≥ target; ablation numbers recorded |
| **M4 — Hinglish + guardrails** | Hinglish normalization, anti-hallucination, streaming | Language fidelity & anti-hallucination targets met |
| **M5 — Eval + hardening** | RAGAS harness, dashboards, load test, API polish | All §12.3 targets met; P95 within NFR-2 |
| **M6 — Docs + demo** | README, architecture doc, demo video, write-up | Reproducible from scratch; metrics documented |

---

## 15. Risks & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Hindi OCR quality poor on low-res scans | High | High | Preprocess (deskew, denoise, upscale); try Surya/vision-LLM; surface confidence, allow manual re-upload |
| R2 | Cross-lingual retrieval underperforms | Med | High | bge-m3 + hybrid + rerank; Hinglish normalization; ablation-driven tuning |
| R3 | Hinglish out-of-distribution | Med | Med | Transliteration + dual-query embedding; LLM normalization fallback |
| R4 | LLM hallucination despite grounding | Med | High | Strict prompt contract + citation post-check + "not found" path; measured on adversarial set |
| R5 | Latency too high (OCR + rerank + LLM) | Med | Med | Async ingestion, caching, smaller top-K, streaming, model tier selection |
| R6 | Cost blowout on large docs | Med | Med | Token budgeting, caching, cheaper model tier for simple queries |
| R7 | Devanagari rendering/encoding bugs | Med | Med | NFC normalization, proper fonts, encoding tests in CI |
| R8 | Data privacy / leakage across users | Low | High | Strict scoping filters, isolation tests, encryption at rest, retention controls |
| R9 | Scope creep (more languages, agents) | Med | Med | Non-goals fixed in §4.2; defer to backlog |

---

## 16. Security, Privacy & Compliance
- Per-user/session document isolation enforced at the retrieval filter layer; add automated isolation tests.
- Encrypt files at rest; TLS in transit; secrets via env/secret manager, never in code.
- Configurable retention + hard delete (docs, chunks, vectors) on user request.
- No use of user data for model training. Log PII-minimally; redact where feasible.
- Align with **India DPDP Act** principles; document data flow and third-party (LLM API) processing in a data-processing note.

---

## 17. Observability & Ops
- **Logs**: structured, per-request `request_id`, per-stage timings.
- **Metrics**: latency histograms (retrieval/rerank/generation), error rates, token/cost counters, ingestion success rate, OCR usage rate.
- **LLM tracing**: prompt/response + retrieved context captured (Langfuse) for debugging quality.
- **Dashboards**: eval metrics over time; alerting on error-rate and latency SLO breaches.
- **Runbooks**: LLM API outage, vector DB down, OCR failures.

---

## 18. Rollout Plan
- **Env**: local (Docker Compose) → single cloud VM/container → managed services if needed.
- **Stages**: internal dogfooding → limited beta with real Hindi docs → open.
- **Feature flags**: reranker on/off, LLM provider, OCR engine.
- **Rollback**: versioned config + pinned model versions; ability to revert embedding model with reindex job.

---

## 19. Cost Model (to be finalized in design)
- Variable: LLM tokens/query, embedding tokens/doc, OCR compute.
- Fixed: vector DB, Postgres, Redis, hosting.
- Track per-query cost in logs; set NFR-4 cap; provide a cheaper "fast" model tier for simple queries.

---

## 20. Open Questions
- OQ1. Local-only/offline deployment required for any customer? (affects LLM choice)
- OQ2. Max corpus size per user we must support? (affects vector DB sizing)
- OQ3. Do we need multi-turn conversational memory in v1, or single-turn Q&A?
- OQ4. Is table/figure extraction in scope beyond best-effort?
- OQ5. Human-in-the-loop review needed for low-confidence answers?

---

## 21. Future Enhancements (backlog)
- More Indian languages (Bengali, Tamil, Marathi, Telugu) — bge-m3 already multilingual.
- Conversational multi-turn memory + follow-ups.
- Agentic tools (calculator, web, cross-document reasoning).
- Fine-tuned/adapter embedding or reranker on domain data.
- Fully offline/on-prem deployment with local LLM.
- Feedback loop (thumbs up/down) → retrieval/prompt improvement.
- Table and figure understanding (multimodal).

---

## 22. Appendix

### 22.1 Definition of Done (v1)
- All §6 functional requirements implemented.
- All §12.3 metric targets met and documented with ablation numbers.
- API + UI working; Docker Compose reproducible from a clean machine.
- README, architecture doc, eval report, and demo complete.

### 22.2 References
- bge-m3 / bge-reranker (BAAI) model cards.
- RAGAS documentation.
- Qdrant hybrid search documentation.
- AI4Bharat Indic NLP resources.
- India DPDP Act overview.
