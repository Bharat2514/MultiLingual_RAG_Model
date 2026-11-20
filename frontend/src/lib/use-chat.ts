"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, streamChat } from "./api";
import type { AnswerStyle, ChatMessage, ChatTurn } from "./types";

let counter = 0;
const nextId = () => `m${++counter}_${Date.now().toString(36)}`;

/** Turns sent with each question so follow-ups ("aur uska penalty?") can be
 *  resolved. Mirrors the backend's `CONVERSATION__MAX_HISTORY_TURNS`. */
const HISTORY_TURNS = 6;

function historyOf(messages: ChatMessage[]): ChatTurn[] {
  return messages
    .filter((m) => !m.streaming && !m.error && m.content.trim())
    .slice(-HISTORY_TURNS)
    .map((m) => ({ role: m.role, content: m.content }));
}

export function useChat(sessionId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [style, setStyle] = useState<AnswerStyle>("default");
  const abortRef = useRef<AbortController | null>(null);
  // Read inside `send` without making the callback depend on every message.
  const messagesRef = useRef<ChatMessage[]>([]);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

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

  const patchById = useCallback((id: string, patch: Partial<ChatMessage>) => {
    setMessages((current) => current.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }, []);

  const send = useCallback(
    async (query: string, docIds?: string[]) => {
      const trimmed = query.trim();
      if (!trimmed || busy || !sessionId) return;

      const history = historyOf(messagesRef.current);
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
        { docIds, signal: abortRef.current.signal, history, style },
      );

      setBusy(false);
      abortRef.current = null;
    },
    [sessionId, busy, patchLast, style],
  );

  const rate = useCallback(
    async (messageId: string, requestId: string, rating: 1 | -1, comment?: string) => {
      patchById(messageId, { rating });
      try {
        await api.sendFeedback(sessionId, requestId, rating, comment);
      } catch {
        patchById(messageId, { rating: undefined });
      }
    },
    [sessionId, patchById],
  );

  /** Make the answer readable by link and return the URL to share. */
  const share = useCallback(
    async (answerId: string): Promise<string> => {
      await api.shareAnswer(sessionId, answerId);
      return `${window.location.origin}/share/${answerId}`;
    },
    [sessionId],
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

  return { messages, busy, send, stop, clear, rate, share, style, setStyle };
}
