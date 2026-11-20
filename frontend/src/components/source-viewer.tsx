"use client";

import * as React from "react";
import { ChevronLeft, ChevronRight, FileText, X } from "lucide-react";
import { Badge, Button, Spinner } from "@/components/ui/primitives";
import { api, PolyDocError } from "@/lib/api";
import type { Highlights } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface SourceTarget {
  docId: string;
  chunkId: string;
  filename: string;
  page: number;
  snippet: string;
}

/**
 * Opens a citation *in the document*: the rendered page with the cited passage
 * highlighted, so a claim can be checked against the original at a glance.
 *
 * Highlight rectangles come back normalised (0–1) and are overlaid as
 * percentages, so they line up at any rendered size. On scanned pages the
 * backend locates the passage from OCR word boxes; DOCX files have no pages to
 * render, so the full passage text is shown instead.
 */
export function SourceViewer({
  sessionId,
  target,
  onClose,
}: {
  sessionId: string;
  target: SourceTarget;
  onClose: () => void;
}) {
  const [page, setPage] = React.useState(target.page);
  const [image, setImage] = React.useState<string | null>(null);
  const [marks, setMarks] = React.useState<Highlights | null>(null);
  const [passage, setPassage] = React.useState<string | null>(null);
  const [state, setState] = React.useState<"loading" | "ready" | "text" | "error">("loading");
  const [error, setError] = React.useState<string | null>(null);
  const dialogRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    dialogRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  React.useEffect(() => {
    let url: string | null = null;
    let cancelled = false;

    (async () => {
      setState("loading");
      try {
        const [blob, located] = await Promise.all([
          api.pageImage(sessionId, target.docId, page),
          page === target.page
            ? api.highlights(sessionId, target.docId, page, target.chunkId)
            : Promise.resolve(null),
        ]);
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setImage(url);
        setMarks(located);
        setState("ready");
      } catch (cause) {
        if (cancelled) return;
        if (cause instanceof PolyDocError && cause.code === "UNSUPPORTED_OPERATION") {
          // DOCX: no page to render — show the passage itself.
          try {
            const chunk = await api.chunk(sessionId, target.chunkId);
            if (!cancelled) {
              setPassage(chunk.text);
              setState("text");
            }
          } catch {
            if (!cancelled) {
              setPassage(target.snippet);
              setState("text");
            }
          }
          return;
        }
        setError(cause instanceof PolyDocError ? cause.message : "Could not load the page.");
        setState("error");
      }
    })();

    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [sessionId, target, page]);

  const totalPages = marks?.pages ?? null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-3 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={`Source: ${target.filename}, page ${page}`}
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
        className="flex max-h-full w-full max-w-3xl flex-col overflow-hidden rounded-[var(--radius-card)] border border-border bg-surface shadow-xl outline-none"
      >
        <header className="flex items-center gap-2 border-b border-border px-4 py-2.5">
          <FileText className="h-4 w-4 shrink-0 text-ink-faint" aria-hidden />
          <p className="script-auto min-w-0 flex-1 truncate text-[13px] font-medium text-ink">
            {target.filename}
          </p>
          {state === "ready" ? (
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
                aria-label="Previous page"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-[12px] tabular-nums text-ink-muted">
                p.{page}
                {totalPages ? ` / ${totalPages}` : ""}
              </span>
              <Button
                variant="ghost"
                size="icon"
                disabled={totalPages !== null && page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
                aria-label="Next page"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          ) : null}
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close source viewer">
            <X className="h-4 w-4" />
          </Button>
        </header>

        <div className="scroll-slim min-h-[240px] overflow-auto bg-surface-2 p-3">
          {state === "loading" ? (
            <div className="flex h-60 items-center justify-center gap-2 text-[13px] text-ink-muted">
              <Spinner className="h-4 w-4" /> Rendering page…
            </div>
          ) : state === "error" ? (
            <p className="p-6 text-center text-[13px] text-danger">{error}</p>
          ) : state === "text" ? (
            <div className="rounded-lg border border-border bg-surface p-4">
              <p className="mb-2 text-[11px] uppercase tracking-wide text-ink-faint">
                Passage · p.{target.page}
              </p>
              <p className="script-auto whitespace-pre-wrap text-[13.5px] leading-relaxed text-ink">
                <mark className="rounded bg-warning-soft px-0.5 text-ink">{passage}</mark>
              </p>
            </div>
          ) : (
            <div className="relative mx-auto w-fit">
              {/* eslint-disable-next-line @next/next/no-img-element -- a session-scoped blob URL, not an optimisable asset */}
              <img
                src={image ?? ""}
                alt={`Page ${page} of ${target.filename}`}
                className="block max-w-full rounded border border-border bg-white"
              />
              {page === target.page
                ? marks?.rects.map((r, i) => (
                    <span
                      key={i}
                      aria-hidden
                      className="pointer-events-none absolute rounded-[2px] bg-yellow-300/45 mix-blend-multiply ring-1 ring-yellow-500/60"
                      style={{
                        left: `${r.x0 * 100}%`,
                        top: `${r.y0 * 100}%`,
                        width: `${(r.x1 - r.x0) * 100}%`,
                        height: `${(r.y1 - r.y0) * 100}%`,
                      }}
                    />
                  ))
                : null}
            </div>
          )}
        </div>

        {state === "ready" && page === target.page ? (
          <footer className="flex items-center gap-2 border-t border-border px-4 py-2 text-[11.5px] text-ink-muted">
            <Badge tone={marks?.method === "none" ? "warning" : "success"}>
              {marks?.method === "ocr"
                ? "Located via OCR"
                : marks?.method === "text"
                  ? "Highlighted"
                  : "Not located on page"}
            </Badge>
            <span className={cn("script-auto truncate", marks?.method !== "none" && "hidden sm:inline")}>
              {marks?.method === "none" ? target.snippet : "The cited passage is highlighted."}
            </span>
          </footer>
        ) : null}
      </div>
    </div>
  );
}
