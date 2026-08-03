/** Mirrors the backend schemas in `backend/app/schemas/`. */

export type DocumentStatus =
  | "queued"
  | "parsing"
  | "ocr"
  | "chunking"
  | "indexing"
  | "ready"
  | "failed";

export type LanguageCode = "hi" | "en" | "hinglish" | "mixed" | "unknown";

export interface IngestionStage {
  stage: string;
  status: "pending" | "running" | "succeeded" | "failed";
  duration_ms: number | null;
  detail: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface DocumentSummary {
  doc_id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  status: DocumentStatus;
  pages: number | null;
  chunk_count: number;
  language_primary: LanguageCode;
  language_distribution: Record<string, number> | null;
  ocr_used: boolean;
  ocr_page_ratio: number;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentDetail extends DocumentSummary {
  stages: IngestionStage[];
}

export interface UploadAccepted {
  doc_id: string;
  filename: string;
  status: DocumentStatus;
  size_bytes: number;
  duplicate_of: string | null;
}

export interface UploadRejected {
  filename: string;
  code: string;
  message: string;
}

export interface UploadResponse {
  request_id: string;
  session_id: string;
  accepted: UploadAccepted[];
  rejected: UploadRejected[];
  dispatch: string;
}

export interface Citation {
  doc_id: string;
  filename: string;
  page: number;
  chunk_id: string;
  chunk_index: number;
  snippet: string;
  score: number;
  /**
   * 1-based position of this passage in the context the model was given — what
   * a bare `[2]` in the answer refers to. Citations arrive deduplicated and
   * sorted by (filename, page), so array order does not resolve it.
   */
  context_index: number;
}

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
}

export interface LatencyBreakdown {
  retrieval: number;
  rerank: number;
  generation: number;
  total: number;
}

export interface ChatResponse {
  request_id: string;
  session_id: string;
  answer: string;
  language_detected: LanguageCode;
  grounded: boolean;
  citations: Citation[];
  usage: TokenUsage;
  latency_ms: LatencyBreakdown;
  model: string;
  retrieval_mode: string;
  reranked: boolean;
  candidates_considered: number;
  query_variants: string[];
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

/** A turn in the transcript. Assistant turns carry the full response metadata
 *  once streaming completes. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  streaming?: boolean;
  response?: ChatResponse;
  error?: ApiError;
}
