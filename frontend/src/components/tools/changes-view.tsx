"use client";

import * as React from "react";
import { ArrowRight, GitCompareArrows, Scale } from "lucide-react";
import { Badge, Button, Spinner } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { DiffChange, DocumentSummary } from "@/lib/types";
import { DocSelect, Empty, ErrorNote, useAction } from "./shared";

function WordDiff({ change }: { change: DiffChange }) {
  return (
    <p className="script-auto text-[12.5px] leading-relaxed text-ink-muted">
      {change.word_diff.map((part, i) =>
        part.op === "equal" ? (
          <span key={i}>{part.text} </span>
        ) : part.op === "delete" ? (
          <del key={i} className="rounded bg-danger-soft px-0.5 text-danger">
            {part.text}
          </del>
        ) : (
          <ins key={i} className="rounded bg-success-soft px-0.5 text-success no-underline">
            {part.text}
          </ins>
        ),
      )}
    </p>
  );
}

/** What changed between two versions, and which documents disagree. */
export function ChangesView({ sessionId, documents }: { sessionId: string; documents: DocumentSummary[] }) {
  const [oldId, setOldId] = React.useState("");
  const [newId, setNewId] = React.useState("");
  const diff = React.useCallback((a: string, b: string) => api.diff(sessionId, a, b), [sessionId]);
  const conflicts = React.useCallback(() => api.conflicts(sessionId), [sessionId]);
  const d = useAction(diff);
  const c = useAction(conflicts);

  return (
    <div className="space-y-4">
      <section className="space-y-2">
        <h3 className="text-[12px] font-semibold text-ink">Compare versions</h3>
        <div className="flex items-end gap-2">
          <DocSelect documents={documents} value={oldId} onChange={setOldId} label="Older" exclude={newId} />
          <ArrowRight className="mb-2 h-4 w-4 shrink-0 text-ink-faint" />
          <DocSelect documents={documents} value={newId} onChange={setNewId} label="Newer" exclude={oldId} />
        </div>
        <Button size="sm" onClick={() => d.run(oldId, newId)} disabled={!oldId || !newId || d.loading}>
          {d.loading ? <Spinner className="h-3.5 w-3.5" /> : <GitCompareArrows className="h-3.5 w-3.5" />}
          Show changes
        </Button>
        <ErrorNote error={d.error} />
        {d.data ? (
          <div className="space-y-2">
            <div className="flex flex-wrap gap-1.5">
              <Badge tone="warning">{d.data.summary.number_changes} number changes</Badge>
              <Badge tone="neutral">{d.data.summary.modified} modified</Badge>
              <Badge tone="success">{d.data.summary.added} added</Badge>
              <Badge tone="danger">{d.data.summary.removed} removed</Badge>
            </div>
            {d.data.changes.length === 0 ? <Empty>No differences in meaning.</Empty> : null}
            <ul className="space-y-2">
              {d.data.changes.slice(0, 60).map((change, i) => (
                <li key={i} className="rounded-lg border border-border p-2.5">
                  <div className="mb-1 flex flex-wrap items-center gap-1.5">
                    <Badge tone={change.kind === "added" ? "success" : change.kind === "removed" ? "danger" : "neutral"}>
                      {change.kind}
                    </Badge>
                    {change.number_changes.map((n, j) => (
                      <Badge key={j} tone="warning">
                        {n.old ?? "—"} → {n.new ?? "—"}
                      </Badge>
                    ))}
                    <span className="ml-auto text-[11px] text-ink-faint">
                      {change.old_page ? `old p.${change.old_page}` : ""}
                      {change.old_page && change.new_page ? " · " : ""}
                      {change.new_page ? `new p.${change.new_page}` : ""}
                    </span>
                  </div>
                  {change.kind === "modified" ? (
                    <WordDiff change={change} />
                  ) : (
                    <p className="script-auto text-[12.5px] leading-relaxed text-ink-muted">{change.old ?? change.new}</p>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </section>

      <section className="space-y-2 border-t border-border pt-3">
        <h3 className="text-[12px] font-semibold text-ink">Contradictions across documents</h3>
        <p className="text-[11.5px] text-ink-faint">
          Near-identical clauses in different documents that state different numbers.
        </p>
        <Button size="sm" variant="secondary" onClick={() => c.run()} disabled={c.loading || documents.length < 2}>
          {c.loading ? <Spinner className="h-3.5 w-3.5" /> : <Scale className="h-3.5 w-3.5" />}
          Check for conflicts
        </Button>
        <ErrorNote error={c.error} />
        {c.data && c.data.total === 0 ? <Empty>No conflicting numbers found.</Empty> : null}
        {c.data?.pairs.map((pair, i) => (
          <div key={i} className="space-y-2 rounded-lg border border-warning/40 p-2.5">
            {pair.conflicts.map((conflict, j) => (
              <div key={j} className="space-y-1 text-[12px]">
                <div className="flex flex-wrap gap-1.5">
                  {conflict.number_changes.map((n, k) => (
                    <Badge key={k} tone="warning">
                      {n.old ?? "—"} vs {n.new ?? "—"}
                    </Badge>
                  ))}
                </div>
                <p className="script-auto text-ink-muted">
                  <span className="font-medium text-ink">{pair.a.filename} p.{conflict.a_page}:</span> {conflict.a}
                </p>
                <p className="script-auto text-ink-muted">
                  <span className="font-medium text-ink">{pair.b.filename} p.{conflict.b_page}:</span> {conflict.b}
                </p>
              </div>
            ))}
          </div>
        ))}
      </section>
    </div>
  );
}
