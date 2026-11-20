"use client";

import * as React from "react";
import { CircleCheck, CircleHelp, CircleX } from "lucide-react";
import { Badge, Button, Spinner } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { DocumentSummary, EligibilityResult } from "@/lib/types";
import { Empty, ErrorNote, SourceChip, inputClass, useAction } from "./shared";

const FIELD_LABELS: Record<string, string> = {
  age: "Age",
  gender: "Gender",
  annual_income: "Annual income (₹)",
  category: "Category (SC/ST/OBC/General/EWS)",
  state: "State",
  occupation: "Occupation",
  land_holding_acres: "Land holding (acres)",
  years_of_service: "Years of service",
  is_farmer: "Farmer",
  is_government_employee: "Government employee",
  is_bpl: "BPL card holder",
  has_bank_account: "Has bank account",
  is_student: "Student",
  education_level: "Education level",
  disability_percent: "Disability (%)",
};

const VERDICT: Record<EligibilityResult["verdict"], { label: string; tone: "success" | "danger" | "warning" | "neutral" }> = {
  eligible: { label: "Eligible", tone: "success" },
  not_eligible: { label: "Not eligible", tone: "danger" },
  incomplete: { label: "Need more details", tone: "warning" },
  no_rules: { label: "No conditions found", tone: "neutral" },
};

/**
 * "Which schemes do I qualify for?" — rules are read from each document once by
 * the model, then evaluated here as code, rule by rule, with the clause behind
 * every pass or fail.
 */
export function EligibilityView({ sessionId, documents }: { sessionId: string; documents: DocumentSummary[] }) {
  const [fields, setFields] = React.useState<Record<string, "number" | "text" | "boolean">>({});
  const [profile, setProfile] = React.useState<Record<string, string>>({});
  const [selected, setSelected] = React.useState<string[]>([]);
  const check = React.useCallback(
    (p: Record<string, unknown>, ids: string[]) => api.checkEligibility(sessionId, p, ids.length ? ids : undefined),
    [sessionId],
  );
  const { data, loading, error, run } = useAction(check);

  React.useEffect(() => {
    api.eligibilityFields().then((r) => setFields(r.fields)).catch(() => setFields({}));
  }, []);

  // Fields the rules asked about that the form does not show yet.
  const extraFields = React.useMemo(() => {
    const known = new Set(Object.keys(fields));
    const extra = new Set<string>();
    for (const r of data?.results ?? []) for (const f of r.missing_fields) if (!known.has(f)) extra.add(f);
    return [...extra];
  }, [data, fields]);

  const submit = () => {
    const typed: Record<string, unknown> = {};
    for (const [key, raw] of Object.entries(profile)) {
      if (raw === "") continue;
      typed[key] = fields[key] === "number" ? Number(raw) : fields[key] === "boolean" ? raw === "true" : raw;
    }
    void run(typed, selected);
  };

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {[...Object.entries(fields), ...extraFields.map((f) => [f, "text"] as const)].map(([name, type]) => (
          <label key={name} className="flex flex-col gap-1 text-[11px] font-medium text-ink-muted">
            {FIELD_LABELS[name] ?? name.replace(/_/g, " ")}
            {type === "boolean" ? (
              <select
                className={inputClass}
                value={profile[name] ?? ""}
                onChange={(e) => setProfile((p) => ({ ...p, [name]: e.target.value }))}
              >
                <option value="">—</option>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
            ) : (
              <input
                className={inputClass}
                type={type === "number" ? "number" : "text"}
                value={profile[name] ?? ""}
                onChange={(e) => setProfile((p) => ({ ...p, [name]: e.target.value }))}
              />
            )}
          </label>
        ))}
      </div>

      <details className="text-[12px] text-ink-muted">
        <summary className="cursor-pointer">Schemes to check ({selected.length || "all"})</summary>
        <div className="mt-1.5 space-y-1">
          {documents.map((d) => (
            <label key={d.doc_id} className="script-auto flex items-center gap-2">
              <input
                type="checkbox"
                checked={selected.includes(d.doc_id)}
                onChange={(e) =>
                  setSelected((s) => (e.target.checked ? [...s, d.doc_id] : s.filter((x) => x !== d.doc_id)))
                }
              />
              {d.filename}
            </label>
          ))}
        </div>
      </details>

      <Button size="sm" onClick={submit} disabled={loading || documents.length === 0}>
        {loading ? <Spinner className="h-3.5 w-3.5" /> : null}
        Check eligibility
      </Button>
      <p className="text-[11px] text-ink-faint">
        First check reads each scheme&apos;s rules with the AI (cached afterwards); the verdict itself is computed, not generated.
      </p>

      <ErrorNote error={error} />
      {data && data.results.length === 0 ? <Empty>No ready documents to check.</Empty> : null}

      <div className="space-y-2.5">
        {data?.results.map((r) => (
          <div key={r.doc_id} className="rounded-lg border border-border p-3">
            <div className="flex items-center gap-2">
              <Badge tone={VERDICT[r.verdict].tone}>{VERDICT[r.verdict].label}</Badge>
              <p className="script-auto truncate text-[13px] font-medium text-ink">{r.scheme_name ?? r.filename}</p>
            </div>
            <ul className="mt-2 space-y-1.5">
              {r.rules.map((rule, i) => (
                <li key={i} className="flex items-start gap-2 text-[12px]">
                  {rule.outcome === "pass" ? (
                    <CircleCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-success" />
                  ) : rule.outcome === "fail" ? (
                    <CircleX className="mt-0.5 h-3.5 w-3.5 shrink-0 text-danger" />
                  ) : (
                    <CircleHelp className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-faint" />
                  )}
                  <span className="script-auto min-w-0 flex-1 text-ink-muted">
                    {rule.label}
                    {rule.outcome === "unknown" ? (
                      <span className="text-ink-faint"> — enter {FIELD_LABELS[rule.field] ?? rule.field}</span>
                    ) : null}
                  </span>
                  <SourceChip source={rule.source} />
                </li>
              ))}
            </ul>
            {r.required_documents.length ? (
              <p className="script-auto mt-2 text-[11.5px] text-ink-faint">
                Documents needed: {r.required_documents.map((d) => d.name).join(", ")}
              </p>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}
