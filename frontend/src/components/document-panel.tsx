"use client";

import * as React from "react";
import {
  AlertCircle,
  CheckCircle2,
  FileText,
  ScanLine,
  Trash2,
  Upload,
} from "lucide-react";
import { Badge, Button, Card, Progress, Spinner } from "@/components/ui/primitives";
import type { DocumentStatus, DocumentSummary, UploadRejected } from "@/lib/types";
import type { UploadOutcome } from "@/lib/use-documents";
import { cn, formatBytes, languageLabel } from "@/lib/utils";

const ACCEPT = ".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document";

/** The real pipeline stages, so progress reflects work rather than a guess. */
const STAGE_ORDER: DocumentStatus[] = [
  "queued",
  "parsing",
  "ocr",
  "chunking",
  "indexing",
  "ready",
];

const STAGE_LABEL: Record<DocumentStatus, string> = {
  queued: "Queued",
  parsing: "Reading document",
  ocr: "Running OCR",
  chunking: "Splitting into passages",
  indexing: "Building the index",
  ready: "Ready",
  failed: "Failed",
};

function progressFor(status: DocumentStatus): number {
  const index = STAGE_ORDER.indexOf(status);
  return index < 0 ? 0 : index / (STAGE_ORDER.length - 1);
}

function DocumentRow({
  doc,
  onDelete,
}: {
  doc: DocumentSummary;
  onDelete: (id: string) => void;
}) {
  const failed = doc.status === "failed";
  const ready = doc.status === "ready";

  return (
    <li className="group animate-rise rounded-lg border border-border bg-surface p-3 transition-colors hover:border-border-strong">
      <div className="flex items-start gap-3">
        <div
          className={cn(
            "mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md",
            failed ? "bg-danger-soft text-danger" : ready ? "bg-success-soft text-success" : "bg-accent-soft text-accent-ink",
          )}
        >
          {failed ? (
            <AlertCircle className="h-4 w-4" />
          ) : ready ? (
            <CheckCircle2 className="h-4 w-4" />
          ) : (
            <Spinner className="h-4 w-4" />
          )}
        </div>

        <div className="min-w-0 flex-1">
          <p
            className="script-auto truncate text-[13px] font-medium leading-snug text-ink"
            title={doc.filename}
          >
            {doc.filename}
          </p>

          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-ink-faint">
            <span>{formatBytes(doc.size_bytes)}</span>
            {doc.pages ? (
              <>
                <span aria-hidden>·</span>
                <span>
                  {doc.pages} page{doc.pages === 1 ? "" : "s"}
                </span>
              </>
            ) : null}
            {ready && doc.chunk_count > 0 ? (
              <>
                <span aria-hidden>·</span>
                <span>{doc.chunk_count} passages</span>
              </>
            ) : null}
          </div>

          {ready ? (
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <Badge tone="neutral">{languageLabel(doc.language_primary)}</Badge>
              {doc.ocr_used ? (
                <Badge tone="warning" title="Text was recovered from a scan using OCR">
                  <ScanLine className="h-3 w-3" />
                  OCR {Math.round(doc.ocr_page_ratio * 100)}%
                </Badge>
              ) : null}
            </div>
          ) : failed ? (
            <p className="mt-1.5 text-[11px] leading-relaxed text-danger">
              {doc.error_message ?? "Ingestion failed."}
            </p>
          ) : (
            <div className="mt-2 space-y-1">
              <Progress value={progressFor(doc.status)} />
              <p className="text-[11px] text-ink-faint">{STAGE_LABEL[doc.status]}…</p>
            </div>
          )}
        </div>

        <Button
          variant="ghost"
          size="icon"
          aria-label={`Delete ${doc.filename}`}
          title="Delete document"
          onClick={() => onDelete(doc.doc_id)}
          className="shrink-0 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100 hover:text-danger"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </li>
  );
}

export function DocumentPanel({
  documents,
  loading,
  uploading,
  error,
  onUpload,
  onDelete,
  onDismissError,
}: {
  documents: DocumentSummary[];
  loading: boolean;
  uploading: boolean;
  error: string | null;
  onUpload: (files: File[]) => Promise<UploadOutcome | null>;
  onDelete: (id: string) => void;
  onDismissError: () => void;
}) {
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = React.useState(false);
  const [rejected, setRejected] = React.useState<UploadRejected[]>([]);

  const handleFiles = React.useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return;
      setRejected([]);
      const outcome = await onUpload(Array.from(files));
      if (outcome?.rejected.length) setRejected(outcome.rejected);
    },
    [onUpload],
  );

  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold text-ink">Documents</h2>
        {documents.length > 0 ? (
          <span className="text-[11px] text-ink-faint">{documents.length}</span>
        ) : null}
      </header>

      <div className="p-3">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void handleFiles(e.dataTransfer.files);
          }}
          className={cn(
            "rounded-lg border border-dashed p-4 text-center transition-colors",
            dragging
              ? "border-accent bg-accent-soft"
              : "border-border-strong bg-surface-2/50 hover:border-accent",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            multiple
            className="sr-only"
            onChange={(e) => {
              void handleFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <Upload className="mx-auto h-5 w-5 text-ink-faint" aria-hidden />
          <p className="mt-2 text-[13px] font-medium text-ink">
            Drop PDFs or Word files
          </p>
          <p className="mt-0.5 text-[11px] text-ink-faint">
            Hindi or English · scans welcome
          </p>
          <Button
            size="sm"
            variant="secondary"
            className="mt-3"
            disabled={uploading}
            onClick={() => inputRef.current?.click()}
          >
            {uploading ? (
              <>
                <Spinner className="h-3.5 w-3.5" />
                Uploading…
              </>
            ) : (
              "Choose files"
            )}
          </Button>
        </div>

        {error ? (
          <div
            role="alert"
            className="mt-3 flex items-start gap-2 rounded-lg bg-danger-soft px-3 py-2 text-[12px] text-danger"
          >
            <AlertCircle className="mt-px h-3.5 w-3.5 shrink-0" aria-hidden />
            <span className="flex-1">{error}</span>
            <button
              onClick={onDismissError}
              className="shrink-0 font-medium underline underline-offset-2"
            >
              Dismiss
            </button>
          </div>
        ) : null}

        {rejected.length > 0 ? (
          <ul className="mt-3 space-y-1.5" role="alert">
            {rejected.map((r) => (
              <li
                key={r.filename}
                className="rounded-lg bg-warning-soft px-3 py-2 text-[12px] text-warning"
              >
                <span className="script-auto font-medium">{r.filename}</span> — {r.message}
              </li>
            ))}
          </ul>
        ) : null}
      </div>

      <div className="scroll-slim flex-1 overflow-y-auto px-3 pb-3">
        {loading && documents.length === 0 ? (
          <ul className="space-y-2" aria-busy="true">
            {[0, 1].map((i) => (
              <li key={i} className="h-16 animate-pulse-soft rounded-lg bg-surface-2" />
            ))}
          </ul>
        ) : documents.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 py-10 text-center">
            <FileText className="h-7 w-7 text-ink-faint/60" aria-hidden />
            <p className="text-[13px] text-ink-muted">No documents yet</p>
            <p className="max-w-[22ch] text-[11px] text-ink-faint">
              Upload one to start asking questions about it.
            </p>
          </div>
        ) : (
          <ul className="space-y-2">
            {documents.map((doc) => (
              <DocumentRow key={doc.doc_id} doc={doc} onDelete={onDelete} />
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}
