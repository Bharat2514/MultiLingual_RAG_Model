"use client";

import * as React from "react";
import { CalendarPlus, RefreshCw } from "lucide-react";
import { Badge, Button, Spinner } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { TimelineItem } from "@/lib/types";
import { Empty, ErrorNote, downloadBlob, useAction } from "./shared";

const KIND: Record<TimelineItem["kind"], { label: string; tone: "danger" | "accent" | "warning" | "neutral" | "success" }> = {
  deadline: { label: "Deadline", tone: "danger" },
  effective: { label: "Takes effect", tone: "accent" },
  event: { label: "Event", tone: "accent" },
  penalty: { label: "Penalty", tone: "warning" },
  obligation: { label: "Must do", tone: "neutral" },
  issued: { label: "Issued", tone: "neutral" },
  date: { label: "Date", tone: "neutral" },
};

function formatDate(iso: string): string {
  return new Date(`${iso}T00:00:00`).toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

/** Every date, duty and penalty across the workspace, extracted without a model. */
export function DeadlinesView({ sessionId }: { sessionId: string }) {
  const load = React.useCallback(() => api.deadlines(sessionId), [sessionId]);
  const { data, loading, error, run } = useAction(load);
  const [filter, setFilter] = React.useState<"actionable" | "all">("actionable");

  React.useEffect(() => {
    void run();
  }, [run]);

  const items = (data?.items ?? []).filter(
    (i) => filter === "all" || ["deadline", "effective", "event", "penalty", "obligation"].includes(i.kind),
  );
  const today = new Date().toISOString().slice(0, 10);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="flex rounded-lg border border-border p-0.5 text-[11.5px]">
          {(["actionable", "all"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`rounded-md px-2 py-0.5 ${filter === f ? "bg-surface-2 font-medium text-ink" : "text-ink-faint"}`}
            >
              {f === "actionable" ? "Actionable" : "All dates"}
            </button>
          ))}
        </div>
        <Button variant="ghost" size="icon" onClick={() => run()} title="Refresh" className="ml-auto">
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="secondary"
          size="sm"
          disabled={!data?.items.some((i) => i.date)}
          onClick={async () => downloadBlob(await api.deadlinesIcs(sessionId), "polydoc-deadlines.ics")}
        >
          <CalendarPlus className="h-3.5 w-3.5" />
          Add to calendar
        </Button>
      </div>

      <ErrorNote error={error} />
      {loading && !data ? (
        <div className="flex justify-center py-6">
          <Spinner className="h-4 w-4 text-ink-faint" />
        </div>
      ) : items.length === 0 ? (
        <Empty>No deadlines or obligations found in your ready documents.</Empty>
      ) : (
        <ol className="space-y-2">
          {items.map((item, i) => {
            const meta = KIND[item.kind];
            const past = item.date !== null && item.date < today;
            return (
              <li key={`${item.chunk_id}-${i}`} className="rounded-lg border border-border p-2.5">
                <div className="flex flex-wrap items-center gap-1.5">
                  <Badge tone={meta.tone}>{meta.label}</Badge>
                  {item.date ? (
                    <span className={`text-[12px] font-semibold ${past ? "text-ink-faint line-through" : "text-ink"}`}>
                      {formatDate(item.date)}
                    </span>
                  ) : null}
                  {item.amount ? <Badge tone="warning">{item.amount}</Badge> : null}
                  <span className="script-auto ml-auto truncate text-[11px] text-ink-faint">
                    {item.filename} p.{item.page}
                  </span>
                </div>
                <p className="script-auto mt-1.5 text-[12.5px] leading-relaxed text-ink-muted">{item.text}</p>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
