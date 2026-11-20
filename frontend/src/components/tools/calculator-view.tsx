"use client";

import * as React from "react";
import { Calculator } from "lucide-react";
import { Button, Spinner } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { Empty, ErrorNote, SourceChip, inputClass, useAction } from "./shared";

/**
 * Amounts computed from a formula in the documents. The model only transcribes
 * the formula; the arithmetic runs as code, and the working is shown.
 */
export function CalculatorView({ sessionId }: { sessionId: string }) {
  const [question, setQuestion] = React.useState("");
  const [inputs, setInputs] = React.useState<Record<string, string>>({});
  const calc = React.useCallback(
    (q: string, values: Record<string, number>) => api.calculate(sessionId, q, values),
    [sessionId],
  );
  const { data, loading, error, run } = useAction(calc);

  const submit = () => {
    const values: Record<string, number> = {};
    for (const [k, v] of Object.entries(inputs)) if (v !== "" && !Number.isNaN(Number(v))) values[k] = Number(v);
    void run(question, values);
  };

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <input
          className={inputClass}
          placeholder="e.g. How much pension will I get? / मेरी पेंशन कितनी होगी?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && question.trim() && submit()}
        />
        <Button size="sm" onClick={submit} disabled={loading || question.trim().length < 3}>
          {loading ? <Spinner className="h-3.5 w-3.5" /> : <Calculator className="h-3.5 w-3.5" />}
          Calculate
        </Button>
      </div>

      <ErrorNote error={error} />

      {data?.status === "not_found" ? <Empty>No formula for this was found in your documents.</Empty> : null}

      {data && data.status !== "not_found" ? (
        <div className="space-y-3 rounded-lg border border-border p-3">
          <div className="flex items-start gap-2">
            <p className="text-[12px] text-ink-muted">
              Formula{data.target ? ` for ${data.target}` : ""}:{" "}
              <code className="rounded bg-surface-2 px-1 font-mono text-[12px] text-ink">{data.expression}</code>
            </p>
            <span className="ml-auto">
              <SourceChip source={data.source} />
            </span>
          </div>
          {data.source?.quote ? (
            <blockquote className="script-auto border-l-2 border-border-strong pl-3 text-[12px] text-ink-muted">
              “{data.source.quote}”
            </blockquote>
          ) : null}

          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {data.variables?.map((v) => (
              <label key={v.name} className="flex flex-col gap-1 text-[11px] font-medium text-ink-muted">
                {v.description}
                {v.unit ? ` (${v.unit})` : ""}
                <input
                  className={inputClass}
                  type="number"
                  placeholder={v.origin === "document" ? `${v.value} (from document)` : "enter value"}
                  value={inputs[v.name] ?? ""}
                  onChange={(e) => setInputs((s) => ({ ...s, [v.name]: e.target.value }))}
                />
              </label>
            ))}
          </div>

          {data.status === "needs_inputs" ? (
            <Button size="sm" variant="secondary" onClick={submit} disabled={loading}>
              Compute with these values
            </Button>
          ) : null}

          {data.status === "computed" ? (
            <div className="rounded-lg bg-success-soft px-3 py-2">
              <p className="text-[18px] font-semibold text-success">
                {data.result?.toLocaleString("en-IN")} {data.result_unit ?? ""}
              </p>
              <p className="font-mono text-[11.5px] text-ink-muted">= {data.working}</p>
            </div>
          ) : null}

          {data.status === "invalid_formula" ? <ErrorNote error={`The formula could not be used: ${data.error}`} /> : null}
        </div>
      ) : null}
    </div>
  );
}
