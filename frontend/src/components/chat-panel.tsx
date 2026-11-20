"use client";

import * as React from "react";
import {
  AlertTriangle,
  ArrowUp,
  Check,
  CornerDownRight,
  Info,
  Link2,
  MessagesSquare,
  Mic,
  MicOff,
  ShieldCheck,
  ShieldAlert,
  Square,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Volume2,
  VolumeX,
} from "lucide-react";
import { Answer, CitationList } from "@/components/answer";
import { ClaimCheck } from "@/components/claim-check";
import { SourceViewer, type SourceTarget } from "@/components/source-viewer";
import { Badge, Button, Card, Spinner } from "@/components/ui/primitives";
import { type SpeechLang, useDictation, useSpeaker } from "@/lib/speech";
import type { AnswerStyle, ChatMessage, Citation } from "@/lib/types";
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

interface TurnActions {
  onRate: (messageId: string, requestId: string, rating: 1 | -1) => void;
  onShare: (answerId: string) => Promise<string>;
  onOpenSource: (c: Citation) => void;
  speaker: ReturnType<typeof useSpeaker>;
}

function AnswerActions({ message, actions }: { message: ChatMessage; actions: TurnActions }) {
  const response = message.response;
  const [copied, setCopied] = React.useState(false);
  if (!response) return null;
  const speaking = actions.speaker.speakingId === message.id;

  const share = async () => {
    if (!response.answer_id) return;
    try {
      const url = await actions.onShare(response.answer_id);
      await navigator.clipboard?.writeText(url);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="flex items-center gap-0.5">
      <Button
        variant="ghost"
        size="icon"
        className={cn("h-7 w-7", message.rating === 1 && "text-success")}
        onClick={() => actions.onRate(message.id, response.request_id, 1)}
        aria-pressed={message.rating === 1}
        title="Helpful"
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className={cn("h-7 w-7", message.rating === -1 && "text-danger")}
        onClick={() => actions.onRate(message.id, response.request_id, -1)}
        aria-pressed={message.rating === -1}
        title="Not helpful — flags this question for the evaluation set"
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </Button>
      {response.answer_id ? (
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={share}
          title="Copy a share link that re-verifies the sources when opened"
        >
          {copied ? <Check className="h-3.5 w-3.5 text-success" /> : <Link2 className="h-3.5 w-3.5" />}
        </Button>
      ) : null}
      {actions.speaker.supported ? (
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() =>
            speaking ? actions.speaker.stop() : actions.speaker.speak(message.id, message.content)
          }
          title={speaking ? "Stop reading" : "Read aloud"}
        >
          {speaking ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
        </Button>
      ) : null}
    </div>
  );
}

function AssistantTurn({
  message,
  activeCitation,
  onSelectCitation,
  actions,
}: {
  message: ChatMessage;
  activeCitation: string | null;
  onSelectCitation: (c: Citation) => void;
  actions: TurnActions;
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
      {settled && response.rewritten_query ? (
        <p className="script-auto mb-2 flex items-start gap-1.5 text-[11.5px] text-ink-faint">
          <CornerDownRight className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
          <span>
            Understood as: <span className="text-ink-muted">{response.rewritten_query}</span>
          </span>
        </p>
      ) : null}

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

      {settled && response.abstention ? (
        <p className="script-auto mt-2 flex items-start gap-1.5 rounded-lg bg-surface-2 px-3 py-2 text-[12.5px] leading-relaxed text-ink-muted">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-faint" aria-hidden />
          {response.abstention.hint}
        </p>
      ) : null}

      <CitationList
        citations={citations}
        activeCitation={activeCitation}
        onSelect={onSelectCitation}
        onOpen={actions.onOpenSource}
      />

      {settled ? <ClaimCheck response={response} /> : null}

      {settled && (response.withheld_passages > 0 || response.excluded_documents.length > 0) ? (
        <div className="mt-2 space-y-1 text-[11.5px] text-ink-faint">
          {response.withheld_passages > 0 ? (
            <p className="flex items-center gap-1.5 text-warning">
              <ShieldAlert className="h-3 w-3" aria-hidden />
              {response.withheld_passages} passage(s) withheld: they contained instructions aimed at the AI.
            </p>
          ) : null}
          {response.excluded_documents.length > 0 ? (
            <p className="script-auto">
              Not used:{" "}
              {response.excluded_documents
                .map((d) => `${d.filename} (${d.reason === "superseded" ? "superseded" : "not yet in force"})`)
                .join(", ")}
            </p>
          ) : null}
        </div>
      ) : null}

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
          {response.cached ? (
            <Badge tone="accent" title="Served from cache — same question, same documents">
              Cached
            </Badge>
          ) : null}
          <span className="ml-auto flex items-center gap-2">
            <AnswerActions message={message} actions={actions} />
            <span className="text-[11px] text-ink-faint">
              {formatDuration(response.latency_ms.total)}
              {response.usage.output_tokens > 0 ? ` · ${response.usage.output_tokens} tokens` : ""}
            </span>
          </span>
        </div>
      ) : null}
    </div>
  );
}

export function ChatPanel({
  sessionId,
  messages,
  busy,
  hasReadyDocuments,
  style,
  onStyleChange,
  onSend,
  onStop,
  onClear,
  onRate,
  onShare,
}: {
  sessionId: string;
  messages: ChatMessage[];
  busy: boolean;
  hasReadyDocuments: boolean;
  style: AnswerStyle;
  onStyleChange: (style: AnswerStyle) => void;
  onSend: (query: string) => void;
  onStop: () => void;
  onClear: () => void;
  onRate: (messageId: string, requestId: string, rating: 1 | -1) => void;
  onShare: (answerId: string) => Promise<string>;
}) {
  const [input, setInput] = React.useState("");
  const [activeCitation, setActiveCitation] = React.useState<string | null>(null);
  const [source, setSource] = React.useState<SourceTarget | null>(null);
  const [voiceLang, setVoiceLang] = React.useState<SpeechLang>("hi-IN");
  const speaker = useSpeaker();
  const dictation = useDictation(
    React.useCallback((text: string) => setInput((current) => (current ? `${current} ${text}` : text)), []),
  );

  const actions: TurnActions = React.useMemo(
    () => ({
      onRate,
      onShare,
      speaker,
      onOpenSource: (c: Citation) =>
        setSource({
          docId: c.doc_id,
          chunkId: c.chunk_id,
          filename: c.filename,
          page: c.page,
          snippet: c.snippet,
        }),
    }),
    [onRate, onShare, speaker],
  );
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
                  actions={actions}
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
          {dictation.supported ? (
            <Button
              size="icon"
              variant={dictation.listening ? "danger" : "ghost"}
              disabled={disabled}
              onClick={() => (dictation.listening ? dictation.stop() : dictation.start(voiceLang))}
              title={
                dictation.listening
                  ? "Stop listening"
                  : `Speak your question (${voiceLang === "hi-IN" ? "Hindi" : "English"})`
              }
              aria-label={dictation.listening ? "Stop voice input" : "Start voice input"}
            >
              {dictation.listening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
            </Button>
          ) : null}
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
        <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 px-1 text-[11px] text-ink-faint">
          <label className="inline-flex cursor-pointer items-center gap-1.5">
            <input
              type="checkbox"
              checked={style === "simple"}
              onChange={(e) => onStyleChange(e.target.checked ? "simple" : "default")}
              className="h-3 w-3 accent-[var(--color-accent)]"
            />
            Simple language
          </label>
          {dictation.supported ? (
            <label className="inline-flex items-center gap-1">
              Voice
              <select
                value={voiceLang}
                onChange={(e) => setVoiceLang(e.target.value as SpeechLang)}
                className="rounded border border-border bg-surface px-1 py-px text-[11px] text-ink-muted"
              >
                <option value="hi-IN">हिन्दी</option>
                <option value="en-IN">English</option>
              </select>
            </label>
          ) : null}
          {dictation.error ? <span className="text-danger">{dictation.error}</span> : null}
          <span className="ml-auto hidden sm:inline">
            Answers come only from your documents. Follow-ups keep context.
          </span>
        </div>
      </div>

      {source ? (
        <SourceViewer sessionId={sessionId} target={source} onClose={() => setSource(null)} />
      ) : null}
    </Card>
  );
}
