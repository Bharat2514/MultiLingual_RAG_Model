"use client";

import * as React from "react";
import { CalendarClock, HelpCircle, Languages, ShieldAlert, ShieldCheck } from "lucide-react";
import { Badge, Button, Spinner } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { DocumentSummary, SecurityReport, Timeline, TranslationPage } from "@/lib/types";
import { DocSelect, ErrorNote, SourceChip, inputClass, useAction } from "./shared";

/** Per-document tools: FAQ, validity dates, security scan, bilingual reader. */
export function DocumentView({ sessionId, documents }: { sessionId: string; documents: DocumentSummary[] }) {
  const [docId, setDocId] = React.useState(documents[0]?.doc_id ?? "");
  const document = documents.find((d) => d.doc_id === docId);

  return (
    <div className="space-y-4">
      <DocSelect documents={documents} value={docId} onChange={setDocId} label="Document" />
      {document ? (
        <>
          <SecurityCard key={`s-${docId}`} sessionId={sessionId} docId={docId} />
          <TimelineCard key={`t-${docId}`} sessionId={sessionId} document={document} documents={documents} />
          <FaqCard key={`f-${docId}`} sessionId={sessionId} document={document} />
          <ReaderCard key={`r-${docId}`} sessionId={sessionId} document={document} />
        </>
      ) : null}
    </div>
  );
}

function Section({
  icon: Icon,
  title,
  children,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-2 border-t border-border pt-3">
      <h3 className="flex items-center gap-1.5 text-[12px] font-semibold text-ink">
        <Icon className="h-3.5 w-3.5 text-ink-faint" aria-hidden />
        {title}
      </h3>
      {children}
    </section>
  );
}

function SecurityCard({ sessionId, docId }: { sessionId: string; docId: string }) {
  const [report, setReport] = React.useState<SecurityReport | null>(null);
  React.useEffect(() => {
    let live = true;
    api.security(sessionId, docId).then((r) => live && setReport(r)).catch(() => live && setReport(null));
    return () => {
      live = false;
    };
  }, [sessionId, docId]);

  if (!report) return null;
  const clean = report.level === "clean" || report.level === "low";
  return (
    <Section icon={clean ? ShieldCheck : ShieldAlert} title="Security scan">
      <p className="text-[12px] text-ink-muted">
        <Badge tone={clean ? "success" : report.level === "high" ? "danger" : "warning"}>
          {report.level === "clean" ? "No planted instructions" : `${report.level} risk`}
        </Badge>{" "}
        Checked for hidden text and instructions aimed at the AI.
      </p>
      {report.findings.slice(0, 5).map((f, i) => (
        <p key={i} className="script-auto rounded bg-surface-2 px-2 py-1 text-[11.5px] text-ink-muted">
          <span className="font-medium">{f.rule.replace(/_/g, " ")}</span>
          {f.page ? ` · p.${f.page}` : ""} — {f.excerpt}
        </p>
      ))}
    </Section>
  );
}

function TimelineCard({
  sessionId,
  document,
  documents,
}: {
  sessionId: string;
  document: DocumentSummary;
  documents: DocumentSummary[];
}) {
  const [timeline, setTimeline] = React.useState<Timeline | null>(null);
  const [effective, setEffective] = React.useState("");
  const [supersededBy, setSupersededBy] = React.useState("");
  const save = React.useCallback(
    () =>
      api.setTimeline(sessionId, document.doc_id, {
        effective_date: effective || null,
        superseded_by: supersededBy || null,
      }),
    [sessionId, document.doc_id, effective, supersededBy],
  );
  const { loading, error, run, data: saved } = useAction(save);

  React.useEffect(() => {
    let live = true;
    api.timeline(sessionId, document.doc_id).then((t) => {
      if (!live) return;
      setTimeline(t);
      setEffective(t.effective_date ?? "");
      setSupersededBy(t.superseded_by ?? "");
    });
    return () => {
      live = false;
    };
  }, [sessionId, document.doc_id]);

  return (
    <Section icon={CalendarClock} title="In force">
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        <label className="flex flex-col gap-1 text-[11px] font-medium text-ink-muted">
          Effective from
          <input className={inputClass} type="date" value={effective} onChange={(e) => setEffective(e.target.value)} />
        </label>
        <DocSelect
          documents={documents}
          value={supersededBy}
          onChange={setSupersededBy}
          label="Replaced by"
          exclude={document.doc_id}
        />
      </div>
      {timeline?.suggestion?.effective_date && !effective ? (
        <button
          onClick={() => setEffective(timeline.suggestion?.effective_date ?? "")}
          className="script-auto block text-left text-[11.5px] text-accent-ink hover:underline"
        >
          Suggested from p.{timeline.suggestion.page}: {timeline.suggestion.effective_date} — “
          {timeline.suggestion.text.slice(0, 90)}…”
        </button>
      ) : null}
      <div className="flex items-center gap-2">
        <Button size="sm" variant="secondary" onClick={() => run()} disabled={loading}>
          Save
        </Button>
        {saved ? <span className="text-[11.5px] text-success">Saved — chat now answers from documents in force.</span> : null}
      </div>
      <ErrorNote error={error} />
    </Section>
  );
}

function FaqCard({ sessionId, document }: { sessionId: string; document: DocumentSummary }) {
  const load = React.useCallback(
    (lang: "hi" | "en") => api.faq(sessionId, document.doc_id, lang),
    [sessionId, document.doc_id],
  );
  const { data, loading, error, run } = useAction(load);

  return (
    <Section icon={HelpCircle} title="Questions this document answers">
      <div className="flex gap-1.5">
        <Button size="sm" variant="secondary" onClick={() => run("en")} disabled={loading}>
          {loading ? <Spinner className="h-3.5 w-3.5" /> : null}
          English FAQ
        </Button>
        <Button size="sm" variant="secondary" onClick={() => run("hi")} disabled={loading}>
          हिन्दी FAQ
        </Button>
      </div>
      <ErrorNote error={error} />
      <ul className="space-y-2">
        {data?.items.map((item, i) => (
          <li key={i} className="rounded-lg border border-border p-2.5">
            <p className="script-auto text-[12.5px] font-medium text-ink">{item.question}</p>
            <p className="script-auto mt-1 text-[12.5px] text-ink-muted">{item.answer}</p>
            <div className="mt-1.5">
              <SourceChip source={item.source} />
            </div>
          </li>
        ))}
      </ul>
    </Section>
  );
}

function ReaderCard({ sessionId, document }: { sessionId: string; document: DocumentSummary }) {
  const target: "hi" | "en" = document.language_primary === "hi" ? "en" : "hi";
  const [pages, setPages] = React.useState<TranslationPage["items"]>([]);
  const [next, setNext] = React.useState<number | null>(0);
  const load = React.useCallback(
    (offset: number) => api.translate(sessionId, document.doc_id, target, offset),
    [sessionId, document.doc_id, target],
  );
  const { loading, error, run } = useAction(load);

  const more = async () => {
    if (next === null) return;
    const page = await run(next);
    if (page) {
      setPages((p) => [...p, ...page.items]);
      setNext(page.next_offset);
    }
  };

  return (
    <Section icon={Languages} title={`Side-by-side ${target === "hi" ? "Hindi" : "English"} translation`}>
      {pages.length ? (
        <div className="space-y-2">
          {pages.map((p) => (
            <div key={p.chunk_id} className="grid grid-cols-1 gap-2 rounded-lg border border-border p-2.5 sm:grid-cols-2">
              <p className="script-auto whitespace-pre-wrap text-[12px] leading-relaxed text-ink-muted">{p.original}</p>
              <p className="script-auto whitespace-pre-wrap border-t border-border pt-2 text-[12px] leading-relaxed text-ink sm:border-l sm:border-t-0 sm:pl-2 sm:pt-0">
                {p.translation ?? "—"}
              </p>
            </div>
          ))}
        </div>
      ) : null}
      <ErrorNote error={error} />
      {next !== null ? (
        <Button size="sm" variant="secondary" onClick={more} disabled={loading}>
          {loading ? <Spinner className="h-3.5 w-3.5" /> : null}
          {pages.length ? "Translate more" : "Start reading"}
        </Button>
      ) : (
        <p className="text-[11.5px] text-ink-faint">End of document.</p>
      )}
    </Section>
  );
}
