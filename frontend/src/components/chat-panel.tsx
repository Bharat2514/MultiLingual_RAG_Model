"use client";

import * as React from "react";
import {
  AlertTriangle,
  ArrowUp,
  MessagesSquare,
  ShieldCheck,
  ShieldAlert,
  Square,
  Trash2,
} from "lucide-react";
import { Answer, CitationList } from "@/components/answer";
import { Badge, Button, Card, Spinner } from "@/components/ui/primitives";
import type { ChatMessage, Citation } from "@/lib/types";
import { cn, formatDuration, languageLabel } from "@/lib/utils";

const SUGGESTIONS = [
  { label: "English", query: "What are the key points in this document?" },
  { label: "हिन्दी", query: "इस दस्तावेज़ में मुख्य बातें क्या हैं?" },
  { label: "Hinglish", query: "mujhe is document ka summary do" },
];

function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end">
      <div className="script-auto max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-[14.5px] text-white shadow-sm">
        {content}
      </div>
    </div>
  );
}

function AssistantTurn({
  message,
  activeCitation,
  onSelectCitation,
}: {
  message: ChatMessage;
  activeCitation: string | null;
  onSelectCitation: (c: Citation) => void;
}) {
  const response = message.response;
  const citations = response?.citations ?? [];
  // Only meaningful once the answer is complete — mid-stream there is nothing
  // to be grounded against yet.
  const settled = !message.streaming && !!response;

  if (message.error) {
    return (
      <div
        role="alert"
        className="flex items-start gap-2.5 rounded-2xl rounded-bl-md border border-danger/30 bg-danger-soft px-4 py-3"
      >
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-danger" aria-hidden />
        <div className="min-w-0">
          <p className="text-[13px] font-medium text-danger">
            {message.error.code === "NETWORK" ? "Cannot reach the API" : "Something went wrong"}
          </p>
          <p className="mt-0.5 text-[12.5px] leading-relaxed text-danger/90">
            {message.error.message}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-[92%] rounded-2xl rounded-bl-md border border-border bg-surface px-4 py-3 shadow-sm">
      {message.streaming && !message.content ? (
        <div className="flex items-center gap-2 py-1 text-[13px] text-ink-muted">
          <Spinner className="h-3.5 w-3.5" />
          Searching your documents…
        </div>
      ) : (
        <Answer
          answer={message.content}
          citations={citations}
          streaming={message.streaming}
          activeCitation={activeCitation}
          onSelectCitation={onSelectCitation}
        />
      )}

      <CitationList
        citations={citations}
        activeCitation={activeCitation}
        onSelect={onSelectCitation}
      />

      {settled ? (
        <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-border pt-2.5">
          {response.grounded ? (
            <Badge tone="success" title="Every claim was matched to a source passage">
              <ShieldCheck className="h-3 w-3" />
              Grounded
            </Badge>
          ) : (
            <Badge
              tone="warning"
              title="This answer could not be tied back to a passage in your documents"
            >
              <ShieldAlert className="h-3 w-3" />
              Unverified
            </Badge>
          )}
          <Badge tone="neutral">{languageLabel(response.language_detected)}</Badge>
          <span className="ml-auto text-[11px] text-ink-faint">
            {formatDuration(response.latency_ms.total)}
            {response.usage.output_tokens > 0
              ? ` · ${response.usage.output_tokens} tokens`
              : ""}
          </span>
        </div>
      ) : null}
    </div>
  );
}

export function ChatPanel({
  messages,
  busy,
  hasReadyDocuments,
  onSend,
  onStop,
  onClear,
}: {
  messages: ChatMessage[];
  busy: boolean;
  hasReadyDocuments: boolean;
  onSend: (query: string) => void;
  onStop: () => void;
  onClear: () => void;
}) {
  const [input, setInput] = React.useState("");
  const [activeCitation, setActiveCitation] = React.useState<string | null>(null);
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Follow the stream, but never yank the view if the user has scrolled up.
  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 160;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }, [messages]);

  React.useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  const submit = () => {
    const value = input.trim();
    if (!value || busy) return;
    onSend(value);
    setInput("");
  };

  const disabled = !hasReadyDocuments;

  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold text-ink">Ask</h2>
        {messages.length > 0 ? (
          <Button variant="ghost" size="sm" onClick={onClear} title="Clear conversation">
            <Trash2 className="h-3.5 w-3.5" />
            Clear
          </Button>
        ) : null}
      </header>

      <div ref={scrollRef} className="scroll-slim flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-accent-soft">
              <MessagesSquare className="h-5 w-5 text-accent-ink" aria-hidden />
            </div>
            <div>
              <p className="text-[15px] font-medium text-ink">
                {disabled ? "Upload a document to begin" : "Ask about your documents"}
              </p>
              <p className="mt-1 max-w-[38ch] text-[13px] leading-relaxed text-ink-muted">
                {disabled
                  ? "Once a file finishes processing you can ask questions about it."
                  : "Ask in Hindi, English, or Hinglish — the answer comes back in the same language, with sources."}
              </p>
            </div>

            {!disabled ? (
              <div className="mt-1 flex flex-wrap justify-center gap-1.5">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s.label}
                    onClick={() => onSend(s.query)}
                    className="script-auto rounded-full border border-border bg-surface px-3 py-1.5 text-[12px] text-ink-muted transition-colors hover:border-accent hover:text-accent-ink"
                  >
                    <span className="mr-1.5 text-[10px] font-semibold uppercase tracking-wide text-ink-faint">
                      {s.label}
                    </span>
                    {s.query}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        ) : (
          messages.map((message) => (
            <div key={message.id} className="animate-rise">
              {message.role === "user" ? (
                <UserBubble content={message.content} />
              ) : (
                <AssistantTurn
                  message={message}
                  activeCitation={activeCitation}
                  onSelectCitation={(c) =>
                    setActiveCitation((current) =>
                      current === c.chunk_id ? null : c.chunk_id,
                    )
                  }
                />
              )}
            </div>
          ))
        )}
      </div>

      <div className="border-t border-border p-3">
        <div
          className={cn(
            "flex items-end gap-2 rounded-xl border bg-surface-2/60 p-2 transition-colors",
            disabled ? "border-border opacity-60" : "border-border focus-within:border-accent",
          )}
        >
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            disabled={disabled}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={
              disabled ? "Upload a document first…" : "Ask in Hindi, English, or Hinglish…"
            }
            aria-label="Your question"
            className="script-auto max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-[14px] text-ink outline-none placeholder:text-ink-faint disabled:cursor-not-allowed"
          />
          {busy ? (
            <Button size="icon" variant="secondary" onClick={onStop} title="Stop generating">
              <Square className="h-3.5 w-3.5" />
            </Button>
          ) : (
            <Button
              size="icon"
              onClick={submit}
              disabled={disabled || !input.trim()}
              title="Send"
              aria-label="Send question"
            >
              <ArrowUp className="h-4 w-4" />
            </Button>
          )}
        </div>
        <p className="mt-1.5 px-1 text-[11px] text-ink-faint">
          Answers come only from your documents. Enter to send, Shift+Enter for a new line.
        </p>
      </div>
    </Card>
  );
}
