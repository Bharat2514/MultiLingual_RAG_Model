"use client";

import * as React from "react";
import { Table } from "lucide-react";
import { Button, Spinner } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { DocumentSummary } from "@/lib/types";
import { ErrorNote, SourceChip, inputClass, useAction } from "./shared";

const PRESETS = [
  "EMD amount, Bid deadline, Eligibility, Contract period",
  "Benefit amount, Eligibility, Last date, Documents required",
];

/** A comparison table across documents, with a cited source in every cell. */
export function CompareView({ sessionId, documents }: { sessionId: string; documents: DocumentSummary[] }) {
  const [selected, setSelected] = React.useState<string[]>([]);
  const [attributes, setAttributes] = React.useState<string>(PRESETS[0] ?? "");
  const compare = React.useCallback(
    (ids: string[], attrs: string[]) => api.compare(sessionId, ids, attrs),
    [sessionId],
  );
  const { data, loading, error, run } = useAction(compare);
  const attrs = attributes.split(",").map((a) => a.trim()).filter(Boolean);

  return (
    <div className="space-y-3">
      <div className="space-y-1">
        <p className="text-[11px] font-medium text-ink-muted">Documents (2–5)</p>
        {documents.map((d) => (
          <label key={d.doc_id} className="script-auto flex items-center gap-2 text-[12.5px] text-ink">
            <input
              type="checkbox"
              checked={selected.includes(d.doc_id)}
              disabled={!selected.includes(d.doc_id) && selected.length >= 5}
              onChange={(e) =>
                setSelected((s) => (e.target.checked ? [...s, d.doc_id] : s.filter((x) => x !== d.doc_id)))
              }
            />
            {d.filename}
          </label>
        ))}
      </div>
      <label className="flex flex-col gap-1 text-[11px] font-medium text-ink-muted">
        What to compare (comma-separated)
        <input className={inputClass} value={attributes} onChange={(e) => setAttributes(e.target.value)} />
      </label>
      <div className="flex flex-wrap gap-1.5">
        {PRESETS.map((p) => (
          <button
            key={p}
            onClick={() => setAttributes(p)}
            className="rounded-full border border-border px-2 py-0.5 text-[11px] text-ink-muted hover:border-accent"
          >
            {p.split(",")[0]}…
          </button>
        ))}
      </div>
      <Button
        size="sm"
        onClick={() => run(selected, attrs)}
        disabled={loading || selected.length < 2 || attrs.length === 0}
      >
        {loading ? <Spinner className="h-3.5 w-3.5" /> : <Table className="h-3.5 w-3.5" />}
        Build table
      </Button>

      <ErrorNote error={error} />

      {data ? (
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full border-collapse text-[12px]">
            <thead>
              <tr className="bg-surface-2">
                <th className="border-b border-border px-2 py-1.5 text-left font-semibold text-ink">Attribute</th>
                {data.rows.map((r) => (
                  <th key={r.doc_id} className="script-auto border-b border-border px-2 py-1.5 text-left font-semibold text-ink">
                    {r.filename}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.attributes.map((a) => (
                <tr key={a} className="align-top">
                  <td className="border-b border-border px-2 py-1.5 font-medium text-ink-muted">{a}</td>
                  {data.rows.map((r) => {
                    const cell = r.cells[a];
                    return (
                      <td key={r.doc_id} className="border-b border-border px-2 py-1.5">
                        {cell?.value ? (
                          <div className="space-y-1">
                            <p className="script-auto text-ink">{cell.value}</p>
                            <SourceChip source={cell.source} />
                          </div>
                        ) : (
                          <span className="text-ink-faint">not stated</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
