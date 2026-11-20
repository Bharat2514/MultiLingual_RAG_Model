import type {
  AnswerStyle,
  ApiError,
  CalculationResult,
  ChatResponse,
  ChatTurn,
  CompareRow,
  ConflictPair,
  DiffResult,
  DocumentDetail,
  DocumentSummary,
  EligibilityResult,
  FaqItem,
  Highlights,
  SecurityReport,
  SharedAnswer,
  Timeline,
  TimelineItem,
  TranslationPage,
  UploadResponse,
} from "./types";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

export class PolyDocError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details?: Record<string, unknown>;

  constructor(error: ApiError, status: number) {
    super(error.message);
    this.name = "PolyDocError";
    this.code = error.code;
    this.status = status;
    this.details = error.details;
  }
}

/** Turn any failure — HTTP, network, or malformed body — into a PolyDocError so
 *  the UI has exactly one error shape to render (FR-30). */
async function toError(response: Response): Promise<PolyDocError> {
  let payload: { error?: ApiError } | null = null;
  try {
    payload = await response.json();
  } catch {
    // Non-JSON body (a proxy error page, an empty 502).
  }
  return new PolyDocError(
    payload?.error ?? {
      code: response.status === 0 ? "NETWORK" : "INTERNAL",
      message: `Request failed with status ${response.status}.`,
    },
    response.status,
  );
}

/** Fetch a binary resource (page image, calendar file) with the session header —
 *  `<img src>` and `<a href>` cannot send one, so these go through a blob URL. */
async function requestBlob(path: string, sessionId: string): Promise<Blob> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, { headers: { "X-Session-ID": sessionId } });
  } catch {
    throw new PolyDocError({ code: "NETWORK", message: "Could not reach the API." }, 0);
  }
  if (!response.ok) throw await toError(response);
  return response.blob();
}

function json(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

async function request<T>(
  path: string,
  sessionId: string,
  init: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: { "X-Session-ID": sessionId, ...(init.headers ?? {}) },
    });
  } catch (cause) {
    throw new PolyDocError(
      {
        code: "NETWORK",
        message:
          "Could not reach the API. Check that the backend is running on " +
          BASE_URL.replace("/api/v1", "") +
          ".",
      },
      0,
    );
  }

  if (!response.ok) throw await toError(response);
  return (await response.json()) as T;
}

export const api = {
  baseUrl: BASE_URL,

  listDocuments(sessionId: string, signal?: AbortSignal) {
    return request<{ documents: DocumentSummary[]; total: number }>(
      "/documents",
      sessionId,
      { signal },
    );
  },

  getDocument(sessionId: string, docId: string, signal?: AbortSignal) {
    return request<DocumentDetail>(`/documents/${docId}`, sessionId, { signal });
  },

  deleteDocument(sessionId: string, docId: string) {
    return request<{ deleted_chunks: number; deleted_vectors: number }>(
      `/documents/${docId}`,
      sessionId,
      { method: "DELETE" },
    );
  },

  uploadDocuments(sessionId: string, files: File[]) {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    return request<UploadResponse>("/documents", sessionId, {
      method: "POST",
      body: form,
    });
  },

  chat(
    sessionId: string,
    query: string,
    options: { docIds?: string[]; signal?: AbortSignal } = {},
  ) {
    return request<ChatResponse>("/chat", sessionId, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, doc_ids: options.docIds ?? null }),
      signal: options.signal,
    });
  },

  // --- trust & sharing -------------------------------------------------------

  sendFeedback(sessionId: string, requestId: string, rating: 1 | -1, comment?: string) {
    return request<{ feedback_id: string }>(
      "/feedback",
      sessionId,
      json({ request_id: requestId, rating, comment: comment ?? null }),
    );
  },

  shareAnswer(sessionId: string, answerId: string) {
    return request<{ shared: boolean }>(`/answers/${answerId}/share`, sessionId, { method: "POST" });
  },

  async getSharedAnswer(answerId: string, sessionId?: string): Promise<SharedAnswer> {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/answers/${answerId}`, {
        headers: sessionId ? { "X-Session-ID": sessionId } : {},
      });
    } catch {
      throw new PolyDocError({ code: "NETWORK", message: "Could not reach the API." }, 0);
    }
    if (!response.ok) throw await toError(response);
    return (await response.json()) as SharedAnswer;
  },

  // --- source viewer ---------------------------------------------------------

  pageImage(sessionId: string, docId: string, page: number) {
    return requestBlob(`/documents/${docId}/pages/${page}/image?dpi=120`, sessionId);
  },

  highlights(sessionId: string, docId: string, page: number, chunkId: string) {
    return request<Highlights>(
      `/documents/${docId}/pages/${page}/highlights?chunk_id=${encodeURIComponent(chunkId)}`,
      sessionId,
    );
  },

  chunk(sessionId: string, chunkId: string) {
    return request<{ text: string; page_start: number; page_end: number; doc_id: string }>(
      `/chunks/${chunkId}`,
      sessionId,
    );
  },

  // --- document intelligence -------------------------------------------------

  deadlines(sessionId: string) {
    return request<{ items: TimelineItem[]; counts: Record<string, number>; documents: number }>(
      "/insights/deadlines",
      sessionId,
    );
  },

  deadlinesIcs(sessionId: string) {
    return requestBlob("/insights/deadlines.ics", sessionId);
  },

  eligibilityFields() {
    return request<{ fields: Record<string, "number" | "text" | "boolean"> }>(
      "/insights/eligibility/fields",
      "",
    );
  },

  checkEligibility(sessionId: string, profile: Record<string, unknown>, docIds?: string[]) {
    return request<{ eligible: number; checked: number; results: EligibilityResult[] }>(
      "/insights/eligibility",
      sessionId,
      json({ profile, doc_ids: docIds ?? null }),
    );
  },

  calculate(sessionId: string, question: string, inputs: Record<string, number>) {
    return request<CalculationResult>("/insights/calculate", sessionId, json({ question, inputs }));
  },

  compare(sessionId: string, docIds: string[], attributes: string[]) {
    return request<{ attributes: string[]; rows: CompareRow[] }>(
      "/insights/compare",
      sessionId,
      json({ doc_ids: docIds, attributes }),
    );
  },

  diff(sessionId: string, oldDocId: string, newDocId: string) {
    return request<DiffResult>(
      "/insights/diff",
      sessionId,
      json({ old_doc_id: oldDocId, new_doc_id: newDocId }),
    );
  },

  conflicts(sessionId: string) {
    return request<{ total: number; pairs: ConflictPair[] }>("/insights/conflicts", sessionId, json({}));
  },

  faq(sessionId: string, docId: string, language?: "hi" | "en") {
    const query = language ? `?language=${language}` : "";
    return request<{ items: FaqItem[]; cached: boolean; language: string }>(
      `/documents/${docId}/faq${query}`,
      sessionId,
    );
  },

  timeline(sessionId: string, docId: string) {
    return request<Timeline>(`/documents/${docId}/timeline`, sessionId);
  },

  setTimeline(
    sessionId: string,
    docId: string,
    body: { effective_date: string | null; superseded_by: string | null },
  ) {
    return request<Timeline>(`/documents/${docId}/timeline`, sessionId, {
      ...json(body),
      method: "PUT",
    });
  },

  security(sessionId: string, docId: string) {
    return request<SecurityReport>(`/documents/${docId}/security`, sessionId);
  },

  translate(sessionId: string, docId: string, target: "hi" | "en", offset: number) {
    return request<TranslationPage>(
      `/documents/${docId}/translate`,
      sessionId,
      json({ target, offset, limit: 4 }),
    );
  },
};

export interface ChatOptions {
  docIds?: string[];
  signal?: AbortSignal;
  history?: ChatTurn[];
  style?: AnswerStyle;
}

// --- streaming ---------------------------------------------------------------

export interface StreamHandlers {
  onToken: (text: string) => void;
  onDone: (response: ChatResponse) => void;
  onError: (error: ApiError) => void;
}

/**
 * Consume the SSE stream from `POST /chat`.
 *
 * Written against the raw body rather than `EventSource` because the endpoint
 * is a POST with a JSON body and a session header, none of which EventSource
 * supports. Events are `start`, many `token`, then one `done` (or `error`).
 */
export async function streamChat(
  sessionId: string,
  query: string,
  handlers: StreamHandlers,
  options: ChatOptions = {},
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Session-ID": sessionId,
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        query,
        doc_ids: options.docIds ?? null,
        stream: true,
        history: options.history ?? [],
        style: options.style ?? "default",
      }),
      signal: options.signal,
    });
  } catch (cause) {
    if ((cause as Error)?.name === "AbortError") return;
    handlers.onError({
      code: "NETWORK",
      message: "Could not reach the API. Is the backend running?",
    });
    return;
  }

  if (!response.ok || !response.body) {
    const error = await toError(response);
    handlers.onError({ code: error.code, message: error.message });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line. A frame can straddle chunk
      // boundaries, so only complete frames are consumed.
      let split: number;
      while ((split = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);

        let event = "message";
        const dataLines: string[] = [];
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        }
        if (dataLines.length === 0) continue;

        let payload: unknown;
        try {
          payload = JSON.parse(dataLines.join("\n"));
        } catch {
          continue;
        }

        if (event === "token") {
          handlers.onToken((payload as { text: string }).text ?? "");
        } else if (event === "done") {
          handlers.onDone(payload as ChatResponse);
        } else if (event === "error") {
          handlers.onError(payload as ApiError);
        }
      }
    }
  } catch (cause) {
    if ((cause as Error)?.name !== "AbortError") {
      handlers.onError({
        code: "STREAM",
        message: "The connection dropped while streaming the answer.",
      });
    }
  } finally {
    reader.releaseLock();
  }
}
