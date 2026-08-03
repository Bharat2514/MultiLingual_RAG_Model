import type {
  ApiError,
  ChatResponse,
  DocumentDetail,
  DocumentSummary,
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
};

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
  options: { docIds?: string[]; signal?: AbortSignal } = {},
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
