"use client";

import * as React from "react";
import { use } from "react";
import { ShieldAlert, ShieldCheck, ShieldQuestion } from "lucide-react";
import { Answer, CitationList } from "@/components/answer";
import { ClaimCheck } from "@/components/claim-check";
import { Badge, Card, Spinner } from "@/components/ui/primitives";
import { api, PolyDocError } from "@/lib/api";
import { useSessionId } from "@/lib/session";
import type { ChatResponse, SharedAnswer } from "@/lib/types";
import { languageLabel } from "@/lib/utils";

const EVIDENCE = {
  intact: {
    tone: "success" as const,
    icon: ShieldCheck,
    text: "Every cited passage still says exactly what it said when this answer was given.",
  },
  changed: {
    tone: "warning" as const,
    icon: ShieldAlert,
    text: "A cited passage has changed since this answer was given — re-check before relying on it.",
  },
  deleted: {
    tone: "danger" as const,
    icon: ShieldAlert,
    text: "A cited source document has been deleted; this answer can no longer be verified.",
  },
  no_citations: {
    tone: "neutral" as const,
    icon: ShieldQuestion,
    text: "This answer cites no passages.",
  },
};

/**
 * A shared answer, with a live re-check of its evidence: the server re-hashes
 * every cited passage on each view, so the badge reflects the documents as
 * they are now, not as they were when the link was made.
 */
export default function SharedAnswerPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const sessionId = useSessionId();
  const [answer, setAnswer] = React.useState<SharedAnswer | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [active, setActive] = React.useState<string | null>(null);

  React.useEffect(() => {
    let live = true;
    // The owner's session can read an unshared snapshot; anyone can read a shared one.
    api
      .getSharedAnswer(id, sessionId || undefined)
      .then((a) => live && setAnswer(a))
      .catch((cause) => live && setError(cause instanceof PolyDocError ? cause.message : "Could not load this answer."));
    return () => {
      live = false;
    };
  }, [id, sessionId]);

  const evidence = answer ? EVIDENCE[answer.evidence.status] : null;
  const Icon = evidence?.icon ?? ShieldQuestion;

  return (
    <main className="mx-auto min-h-dvh max-w-2xl bg-canvas px-4 py-8">
      <p className="mb-4 text-[12px] font-medium uppercase tracking-wide text-ink-faint">PolyDoc · shared answer</p>
      {error ? (
        <Card className="p-6 text-center text-[13px] text-danger">{error}</Card>
      ) : !answer ? (
        <div className="flex justify-center py-16">
          <Spinner className="h-5 w-5 text-ink-faint" />
        </div>
      ) : (
        <Card className="space-y-4 p-5">
          <p className="script-auto text-[15px] font-medium text-ink">{answer.query}</p>
          <Answer
            answer={answer.answer}
            citations={answer.citations}
            activeCitation={active}
            onSelectCitation={(c) => setActive((a) => (a === c.chunk_id ? null : c.chunk_id))}
          />
          <CitationList
            citations={answer.citations}
            activeCitation={active}
            onSelect={(c) => setActive((a) => (a === c.chunk_id ? null : c.chunk_id))}
          />
          <ClaimCheck response={{ claims: answer.claims, claim_summary: null } as unknown as ChatResponse} />
          {evidence ? (
            <div className="flex items-start gap-2 rounded-lg border border-border bg-surface-2 p-3">
              <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              <div className="text-[12.5px] leading-relaxed">
                <Badge tone={evidence.tone}>Evidence {answer.evidence.status.replace("_", " ")}</Badge>
                <p className="mt-1 text-ink-muted">{evidence.text}</p>
                {!answer.evidence.snapshot_untampered ? (
                  <p className="mt-1 text-danger">The stored answer does not match its fingerprint.</p>
                ) : null}
              </div>
            </div>
          ) : null}
          <p className="text-[11px] text-ink-faint">
            {languageLabel(answer.language)}
            {answer.created_at ? ` · ${new Date(answer.created_at).toLocaleString("en-IN")}` : ""}
            {answer.model ? ` · ${answer.model}` : ""}
          </p>
        </Card>
      )}
    </main>
  );
}
