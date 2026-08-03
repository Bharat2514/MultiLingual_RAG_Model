"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, PolyDocError } from "./api";
import type { DocumentSummary, UploadRejected } from "./types";

const TERMINAL: ReadonlySet<string> = new Set(["ready", "failed"]);

/** Poll only while something is actually in flight — a corpus at rest should
 *  not generate a request every two seconds. */
const POLL_MS = 1500;

export interface UploadOutcome {
  accepted: number;
  duplicates: number;
  rejected: UploadRejected[];
}

export function useDocuments(sessionId: string) {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(
    async (signal?: AbortSignal) => {
      if (!sessionId) return;
      try {
        const data = await api.listDocuments(sessionId, signal);
        setDocuments(data.documents);
        setError(null);
      } catch (cause) {
        if ((cause as Error)?.name === "AbortError") return;
        setError(
          cause instanceof PolyDocError ? cause.message : "Could not load documents.",
        );
      } finally {
        setLoading(false);
      }
    },
    [sessionId],
  );

  // Initial load plus adaptive polling while any document is still ingesting.
  useEffect(() => {
    if (!sessionId) return;
    const controller = new AbortController();
    let cancelled = false;

    const tick = async () => {
      await refresh(controller.signal);
      if (cancelled) return;
      timer.current = setTimeout(tick, POLL_MS);
    };

    void tick();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer.current) clearTimeout(timer.current);
    };
  }, [sessionId, refresh]);

  // Stop polling once everything has settled; resume when new work arrives.
  const pending = documents.some((d) => !TERMINAL.has(d.status));
  useEffect(() => {
    if (pending || !timer.current) return;
    clearTimeout(timer.current);
    timer.current = null;
  }, [pending]);

  useEffect(() => {
    if (!pending || timer.current || !sessionId) return;
    const controller = new AbortController();
    const tick = async () => {
      await refresh(controller.signal);
      timer.current = setTimeout(tick, POLL_MS);
    };
    timer.current = setTimeout(tick, POLL_MS);
    return () => controller.abort();
  }, [pending, sessionId, refresh]);

  const upload = useCallback(
    async (files: File[]): Promise<UploadOutcome | null> => {
      if (!files.length) return null;
      setUploading(true);
      setError(null);
      try {
        const result = await api.uploadDocuments(sessionId, files);
        await refresh();
        return {
          accepted: result.accepted.filter((a) => !a.duplicate_of).length,
          duplicates: result.accepted.filter((a) => a.duplicate_of).length,
          rejected: result.rejected,
        };
      } catch (cause) {
        setError(cause instanceof PolyDocError ? cause.message : "Upload failed.");
        return null;
      } finally {
        setUploading(false);
      }
    },
    [sessionId, refresh],
  );

  const remove = useCallback(
    async (docId: string) => {
      // Optimistic: the row disappears immediately, and a failed delete is
      // corrected by the next refresh.
      setDocuments((current) => current.filter((d) => d.doc_id !== docId));
      try {
        await api.deleteDocument(sessionId, docId);
      } catch (cause) {
        setError(cause instanceof PolyDocError ? cause.message : "Delete failed.");
      } finally {
        await refresh();
      }
    },
    [sessionId, refresh],
  );

  return {
    documents,
    loading,
    uploading,
    error,
    pending,
    ready: documents.filter((d) => d.status === "ready"),
    upload,
    remove,
    refresh,
    clearError: () => setError(null),
  };
}
