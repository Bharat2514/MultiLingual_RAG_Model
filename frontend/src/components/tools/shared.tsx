"use client";

import * as React from "react";
import { BadgeCheck, TriangleAlert } from "lucide-react";
import { PolyDocError } from "@/lib/api";
import type { DocumentSummary, SourceRef } from "@/lib/types";
import { cn } from "@/lib/utils";

/** One async action with loading / error / result state. */
export function useAction<T, A extends unknown[]>(fn: (...args: A) => Promise<T>) {
  const [data, setData] = React.useState<T | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const run = React.useCallback(
    async (...args: A) => {
      setLoading(true);
      setError(null);
      try {
        const result = await fn(...args);
        setData(result);
        return result;
      } catch (cause) {
        setError(cause instanceof PolyDocError ? cause.message : "Something went wrong.");
        return null;
      } finally {
        setLoading(false);
      }
    },
    [fn],
  );

  return { data, loading, error, run, setData };
}

/** Where a model-extracted value came from, and whether its quote checked out. */
export function SourceChip({ source }: { source: SourceRef | null | undefined }) {
  if (!source) return <span className="text-[11px] text-ink-faint">no source</span>;
  const verified = source.quote_verified;
  return (
    <span
      title={
        verified
          ? `Quote found verbatim on this page: “${source.quote ?? ""}”`
          : "The model's quote could not be found verbatim in the passage — check it"
      }
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-px text-[10.5px] font-medium",
        verified
          ? "border-transparent bg-success-soft text-success"
          : "border-transparent bg-warning-soft text-warning",
      )}
    >
      {verified ? <BadgeCheck className="h-3 w-3" /> : <TriangleAlert className="h-3 w-3" />}
      <span className="script-auto max-w-[14ch] truncate">{source.filename}</span> p.{source.page}
    </span>
  );
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <p role="alert" className="rounded-lg bg-danger-soft px-3 py-2 text-[12.5px] text-danger">
      {error}
    </p>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <p className="py-6 text-center text-[12.5px] text-ink-faint">{children}</p>;
}

export function DocSelect({
  documents,
  value,
  onChange,
  label,
  exclude,
}: {
  documents: DocumentSummary[];
  value: string;
  onChange: (docId: string) => void;
  label: string;
  exclude?: string;
}) {
  return (
    <label className="flex min-w-0 flex-1 flex-col gap-1 text-[11px] font-medium text-ink-muted">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="script-auto h-8 rounded-lg border border-border bg-surface px-2 text-[12.5px] text-ink"
      >
        <option value="">Choose a document…</option>
        {documents
          .filter((d) => d.doc_id !== exclude)
          .map((d) => (
            <option key={d.doc_id} value={d.doc_id}>
              {d.filename}
            </option>
          ))}
      </select>
    </label>
  );
}

export const inputClass =
  "h-8 w-full rounded-lg border border-border bg-surface px-2 text-[12.5px] text-ink outline-none focus:border-accent";

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
