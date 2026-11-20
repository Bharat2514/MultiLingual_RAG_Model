"use client";

import * as React from "react";
import { ChevronDown, CircleCheck, CircleDashed, CircleHelp, CircleX, CircleAlert } from "lucide-react";
import type { Claim, ChatResponse, ClaimVerdict } from "@/lib/types";
import { cn } from "@/lib/utils";

const VERDICT: Record<
  ClaimVerdict,
  { label: string; icon: React.ComponentType<{ className?: string }>; tone: string }
> = {
  supported: { label: "Supported", icon: CircleCheck, tone: "text-success" },
  partial: { label: "Partly supported", icon: CircleAlert, tone: "text-warning" },
  unsupported: { label: "Not supported", icon: CircleX, tone: "text-danger" },
  uncited: { label: "No citation", icon: CircleHelp, tone: "text-ink-faint" },
  unchecked: { label: "Not checkable", icon: CircleDashed, tone: "text-ink-faint" },
};

/**
 * Sentence-by-sentence verification of an answer against the passages each
 * sentence cites. Collapsed to a one-line summary; problems open it by default,
 * because a wrong number in a cited sentence is exactly what a reader skims past.
 */
export function ClaimCheck({ response }: { response: ChatResponse }) {
  const claims = response.claims ?? [];
  const problems = claims.filter((c) => c.verdict === "unsupported" || c.verdict === "partial");
  const [open, setOpen] = React.useState(problems.length > 0);

  if (!claims.length) return null;
  const counts = response.claim_summary?.counts;
  const ratio = response.claim_summary?.support_ratio;

  return (
    <div className="mt-3 rounded-lg border border-border bg-surface-2/50">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-[12px]"
      >
        <span className="font-medium text-ink">Claim check</span>
        <span className="text-ink-faint">
          {counts?.supported ?? 0}/{claims.length} supported
          {problems.length ? ` · ${problems.length} to review` : ""}
          {ratio !== null && ratio !== undefined ? ` · ${Math.round(ratio * 100)}%` : ""}
        </span>
        <ChevronDown
          className={cn("ml-auto h-3.5 w-3.5 text-ink-faint transition-transform", open && "rotate-180")}
          aria-hidden
        />
      </button>
      {open ? (
        <ul className="space-y-1.5 border-t border-border px-3 py-2">
          {claims.map((claim, i) => (
            <ClaimRow key={i} claim={claim} />
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function ClaimRow({ claim }: { claim: Claim }) {
  const meta = VERDICT[claim.verdict];
  const Icon = meta.icon;
  return (
    <li className="flex gap-2 text-[12.5px] leading-relaxed">
      <Icon className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", meta.tone)} aria-hidden />
      <div className="min-w-0">
        <p className="script-auto text-ink">{claim.text}</p>
        <p className={cn("text-[11px]", meta.tone)}>
          {meta.label}
          {claim.score !== null ? ` · ${Math.round(claim.score * 100)}%` : ""}
          {claim.reason ? <span className="text-ink-faint"> — {claim.reason}</span> : null}
        </p>
      </div>
    </li>
  );
}
