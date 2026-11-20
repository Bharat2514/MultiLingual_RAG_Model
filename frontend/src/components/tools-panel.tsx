"use client";

import * as React from "react";
import { Card } from "@/components/ui/primitives";
import { CalculatorView } from "@/components/tools/calculator-view";
import { ChangesView } from "@/components/tools/changes-view";
import { CompareView } from "@/components/tools/compare-view";
import { DeadlinesView } from "@/components/tools/deadlines-view";
import { DocumentView } from "@/components/tools/document-view";
import { EligibilityView } from "@/components/tools/eligibility-view";
import type { DocumentSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

const TABS = [
  { id: "deadlines", label: "Deadlines" },
  { id: "eligibility", label: "Eligibility" },
  { id: "calculator", label: "Calculator" },
  { id: "compare", label: "Compare" },
  { id: "changes", label: "Changes" },
  { id: "document", label: "Document" },
] as const;

type TabId = (typeof TABS)[number]["id"];

/**
 * Document intelligence: what you can do *with* the documents beyond asking.
 * Every view keeps the chat's contract — each value shows the page it came
 * from, and model-extracted quotes are marked when they could not be verified.
 */
export function ToolsPanel({ sessionId, documents }: { sessionId: string; documents: DocumentSummary[] }) {
  const [tab, setTab] = React.useState<TabId>("deadlines");

  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <nav
        className="scroll-slim flex shrink-0 gap-1 overflow-x-auto border-b border-border px-3 py-2"
        aria-label="Document tools"
      >
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            aria-current={tab === t.id ? "page" : undefined}
            className={cn(
              "shrink-0 rounded-md px-2.5 py-1 text-[12.5px] transition-colors",
              tab === t.id ? "bg-accent-soft font-medium text-accent-ink" : "text-ink-muted hover:bg-surface-2",
            )}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <div className="scroll-slim flex-1 overflow-y-auto p-4">
        {documents.length === 0 ? (
          <p className="py-10 text-center text-[13px] text-ink-faint">
            Upload a document and wait for it to finish processing to use these tools.
          </p>
        ) : tab === "deadlines" ? (
          <DeadlinesView sessionId={sessionId} />
        ) : tab === "eligibility" ? (
          <EligibilityView sessionId={sessionId} documents={documents} />
        ) : tab === "calculator" ? (
          <CalculatorView sessionId={sessionId} />
        ) : tab === "compare" ? (
          <CompareView sessionId={sessionId} documents={documents} />
        ) : tab === "changes" ? (
          <ChangesView sessionId={sessionId} documents={documents} />
        ) : (
          <DocumentView sessionId={sessionId} documents={documents} />
        )}
      </div>
    </Card>
  );
}
