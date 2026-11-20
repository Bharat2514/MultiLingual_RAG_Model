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

export type ClaimVerdict = "supported" | "partial" | "unsupported" | "uncited" | "unchecked";

export interface Claim {
  text: string;
  verdict: ClaimVerdict;
  score: number | null;
  method: string;
  chunk_ids: string[];
  missing_numbers: string[];
  reason: string;
}

export interface Abstention {
  reason: "no_documents" | "no_relevant_passages" | "model_declined";
  searched_documents: number;
  nearest_source: string | null;
  withheld_passages: number;
  hint: string;
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
  answer_id: string | null;
  claims: Claim[];
  claim_summary: { counts: Record<ClaimVerdict, number>; support_ratio: number | null } | null;
  rewritten_query: string | null;
  abstention: Abstention | null;
  withheld_passages: number;
  excluded_documents: { doc_id: string; filename: string; reason: string }[];
  cached: boolean;
}

export type AnswerStyle = "default" | "simple";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

/** A model-extracted value's pointer back into the document. */
export interface SourceRef {
  doc_id: string;
  filename: string;
  page: number;
  chunk_id: string;
  quote: string | null;
  quote_verified: boolean;
}

export interface TimelineItem {
  kind: "deadline" | "effective" | "event" | "issued" | "date" | "penalty" | "obligation";
  text: string;
  date: string | null;
  amount: string | null;
  doc_id: string;
  filename: string;
  page: number;
  chunk_id: string;
}

export interface RuleResult {
  field: string;
  label: string;
  operator: string;
  value: unknown;
  unit: string | null;
  outcome: "pass" | "fail" | "unknown";
  provided: unknown;
  source: SourceRef | null;
}

export interface EligibilityResult {
  doc_id: string;
  filename: string;
  scheme_name: string | null;
  verdict: "eligible" | "not_eligible" | "incomplete" | "no_rules";
  passed: number;
  failed: number;
  unknown: number;
  missing_fields: string[];
  rules: RuleResult[];
  required_documents: { name: string; source: SourceRef | null }[];
  benefits: { text: string; source: SourceRef | null }[];
}

export interface CalculationResult {
  status: "computed" | "needs_inputs" | "not_found" | "invalid_formula";
  target?: string | null;
  expression?: string;
  variables?: { name: string; description: string; unit: string | null; value: number | null; origin: string | null }[];
  missing?: string[];
  result?: number;
  working?: string;
  result_unit?: string | null;
  source?: SourceRef | null;
  error?: string;
}

export interface DiffChange {
  kind: "added" | "removed" | "modified";
  old: string | null;
  new: string | null;
  old_page: number | null;
  new_page: number | null;
  similarity: number | null;
  number_changes: { old: string | null; new: string | null }[];
  word_diff: { op: "equal" | "insert" | "delete"; text: string }[];
}

export interface DiffResult {
  old: { doc_id: string; filename: string };
  new: { doc_id: string; filename: string };
  summary: Record<string, number>;
  changes: DiffChange[];
}

export interface ConflictPair {
  a: { doc_id: string; filename: string };
  b: { doc_id: string; filename: string };
  conflicts: {
    a: string;
    b: string;
    a_page: number;
    b_page: number;
    similarity: number;
    number_changes: { old: string | null; new: string | null }[];
  }[];
}

export interface FaqItem {
  question: string;
  answer: string;
  source: SourceRef | null;
}

export interface CompareRow {
  doc_id: string;
  filename: string;
  cells: Record<string, { value: string | null; source: SourceRef | null }>;
}

export interface Timeline {
  doc_id: string;
  effective_date: string | null;
  superseded_by: string | null;
  note: string | null;
  suggestion?: { effective_date: string | null; text: string; page: number } | null;
}

export interface SecurityReport {
  level: "clean" | "low" | "medium" | "high";
  counts: Record<string, number>;
  findings: { rule: string; severity: string; excerpt: string; page: number | null }[];
}

export interface Highlights {
  method: "text" | "ocr" | "none";
  rects: { x0: number; y0: number; x1: number; y1: number }[];
  pages: number | null;
}

export interface TranslationPage {
  total: number;
  next_offset: number | null;
  items: { chunk_id: string; page: number; original: string; translation: string | null }[];
}

export interface SharedAnswer {
  answer_id: string;
  query: string;
  answer: string;
  language: LanguageCode;
  grounded: boolean;
  citations: Citation[];
  claims: Claim[];
  model: string | null;
  created_at: string | null;
  shared: boolean;
  evidence: {
    status: "intact" | "changed" | "deleted" | "no_citations";
    citations: { chunk_id: string; status: string }[];
    snapshot_untampered: boolean;
  };
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
  /** The user's vote on this answer, once given. */
  rating?: 1 | -1;
}
