"use client";

import { useCallback, useRef, useState } from "react";
import { streamChat } from "./api";
import type { ChatMessage } from "./types";

let counter = 0;
const nextId = () => `m${++counter}_${Date.now().toString(36)}`;

export function useChat(sessionId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const patchLast = useCallback((patch: Partial<ChatMessage>) => {
    setMessages((current) => {
      if (current.length === 0) return current;
      const copy = [...current];
      const last = copy[copy.length - 1];
      if (!last) return current;
      copy[copy.length - 1] = { ...last, ...patch };
      return copy;
    });
  }, []);

  const send = useCallback(
    async (query: string, docIds?: string[]) => {
      const trimmed = query.trim();
      if (!trimmed || busy || !sessionId) return;

      abortRef.current = new AbortController();
      setBusy(true);

      setMessages((current) => [
        ...current,
        { id: nextId(), role: "user", content: trimmed, createdAt: Date.now() },
        {
          id: nextId(),
          role: "assistant",
          content: "",
          createdAt: Date.now(),
          streaming: true,
        },
      ]);

      // Accumulated locally rather than read back from state, so React's
      // batching cannot drop tokens that arrive in the same tick.
      let buffer = "";

      await streamChat(
        sessionId,
        trimmed,
        {
          onToken: (text) => {
            buffer += text;
            patchLast({ content: buffer });
          },
          onDone: (response) => {
            patchLast({
              // The final payload is authoritative: streamed tokens can be
              // withheld (the not-found path never streams its replacement).
              content: response.answer || buffer,
              streaming: false,
              response,
            });
          },
          onError: (error) => {
            patchLast({ streaming: false, error, content: buffer });
          },
        },
        { docIds, signal: abortRef.current.signal },
      );

      setBusy(false);
      abortRef.current = null;
    },
    [sessionId, busy, patchLast],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    patchLast({ streaming: false });
    setBusy(false);
  }, [patchLast]);

  const clear = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setBusy(false);
  }, []);

  return { messages, busy, send, stop, clear };
}
