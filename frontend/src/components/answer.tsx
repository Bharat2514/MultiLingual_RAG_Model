"use client";

import * as React from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText, Quote } from "lucide-react";
import { Badge } from "@/components/ui/primitives";
import type { Citation } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Renders an answer as Markdown, with its inline citations turned into
 * clickable chips (FR-28, US3).
 *
 * Answers are Markdown: the model emits **bold** and bulleted lists, and a
 * summary is almost entirely a bulleted list. Rendered as plain text the
 * asterisks show up raw, so the answer has to go through a Markdown pass.
 *
 * Citations complicate that, because they sit *inside* the Markdown — a bullet
 * can hold bold text and a citation at once — so they cannot be resolved by
 * splitting the string beforehand. Instead a rehype pass walks the parsed tree
 * and swaps citation markers out of text nodes for chip elements, which leaves
 * the Markdown structure intact around them.
 *
 * The model writes `[policy.pdf p.4]`, a bare `[2]`, or a group like `[1, 4]` —
 * the same forms the backend resolves, and the two must stay in step or a
 * citation the backend verified renders as literal text. Anything that does not
 * match a verified citation is left as plain text rather than made clickable:
 * an unverifiable reference must not look interactive.
 */

// One entry of an index group: `2`, or `2 p.5` / `2 pp.5-7` when the model
// half-complies with the [file p.N] instruction.
const INDEX_ENTRY = String.raw`\d{1,2}(?:\s*pp?\.\s*\d+(?:\s*[-–]\s*\d+)?)?`;

const CITATION_RE = new RegExp(
  // [policy.pdf p.4] / [policy.pdf pp.4-6]
  String.raw`\[\s*(?<file>[^[\]]+?)\s+pp?\.\s*(?<page>\d+)(?:\s*[-–]\s*\d+)?\s*\]` +
    // [1] / [1, 4] / [2 p.5, 3 p.19]
    String.raw`|\[\s*(?<indexes>${INDEX_ENTRY}(?:\s*,\s*${INDEX_ENTRY})*)\s*\]`,
  "g",
);

const INDEX_ENTRY_RE = new RegExp(String.raw`^(\d{1,2})`);

/** The citations a single marker resolves to, in the order they were written. */
function resolveMarker(
  groups: Record<string, string | undefined>,
  citations: Citation[],
): Citation[] {
  const { file, page, indexes } = groups;

  if (file && page) {
    const wanted = Number(page);
    const group = citations.filter((c) => c.filename.toLowerCase() === file.trim().toLowerCase());
    const match = group.find((c) => c.page === wanted) ?? group[0];
    return match ? [match] : [];
  }

  if (!indexes) return [];

  const resolved: Citation[] = [];
  for (const raw of indexes.split(",")) {
    const ordinal = Number(INDEX_ENTRY_RE.exec(raw.trim())?.[1]);
    // Match on the passage position the model actually cited, never on array
    // position: citations are deduplicated and re-sorted, so the Nth citation
    // is not the Nth passage and indexing by it links to the wrong source.
    const match = citations.find((c) => c.context_index === ordinal);
    if (match && !resolved.includes(match)) resolved.push(match);
  }
  return resolved;
}

type HastNode = {
  type: string;
  tagName?: string;
  value?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
};

/**
 * Rehype pass replacing citation markers inside text nodes with `<cite>`
 * elements, which `components.cite` below renders as chips.
 */
function citationPlugin(citations: Citation[]) {
  return () => (tree: HastNode) => {
    if (!citations.length) return;

    const split = (value: string): HastNode[] => {
      const out: HastNode[] = [];
      let cursor = 0;

      CITATION_RE.lastIndex = 0;
      for (const match of value.matchAll(CITATION_RE)) {
        const resolved = resolveMarker(match.groups ?? {}, citations);
        if (!resolved.length) continue; // unverified → leave the marker as text

        const start = match.index ?? 0;
        if (start > cursor) out.push({ type: "text", value: value.slice(cursor, start) });
        for (const citation of resolved) {
          out.push({
            type: "element",
            tagName: "cite",
            properties: { dataChunkId: citation.chunk_id },
            children: [],
          });
        }
        cursor = start + match[0].length;
      }

      if (!out.length) return [{ type: "text", value }];
      if (cursor < value.length) out.push({ type: "text", value: value.slice(cursor) });
      return out;
    };

    const walk = (node: HastNode) => {
      if (!node.children) return;
      const next: HastNode[] = [];
      for (const child of node.children) {
        if (child.type === "text" && child.value) {
          next.push(...split(child.value));
        } else {
          walk(child);
          next.push(child);
        }
      }
      node.children = next;
    };

    walk(tree);
  };
}

function CitationChip({
  citation,
  index,
  onSelect,
  active,
}: {
  citation: Citation;
  index: number;
  onSelect: (c: Citation) => void;
  active: boolean;
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(citation)}
      title={`${citation.filename} p.${citation.page} — click to see the source passage`}
      className={cn(
        "mx-0.5 inline-flex translate-y-[-1px] items-center gap-1 rounded border px-1.5 py-px align-middle",
        "text-[11px] font-medium leading-none transition-colors",
        active
          ? "border-accent bg-accent text-white"
          : "border-accent-soft bg-accent-soft text-accent-ink hover:border-accent",
      )}
    >
      <Quote className="h-2.5 w-2.5" aria-hidden />
      {index}
    </button>
  );
}

/**
 * Element styles for the rendered Markdown.
 *
 * Spelled out per element rather than pulled from a typography plugin: the set
 * a grounded answer uses is small, and the spacing has to sit inside a chat
 * bubble rather than a document page.
 */
function markdownComponents(
  citations: Citation[],
  activeCitation: string | null | undefined,
  onSelectCitation: (c: Citation) => void,
): Components {
  return {
    p: ({ children }) => <p className="mb-2.5 last:mb-0">{children}</p>,
    ul: ({ children }) => (
      <ul className="mb-2.5 list-disc space-y-1 pl-5 last:mb-0 marker:text-ink-faint">
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol className="mb-2.5 list-decimal space-y-1 pl-5 last:mb-0 marker:text-ink-faint">
        {children}
      </ol>
    ),
    li: ({ children }) => <li className="pl-0.5 leading-relaxed">{children}</li>,
    strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
    em: ({ children }) => <em className="italic">{children}</em>,
    // Headings in a chat bubble should read as emphasis, not as page structure.
    h1: ({ children }) => <p className="mb-1.5 font-semibold text-ink">{children}</p>,
    h2: ({ children }) => <p className="mb-1.5 font-semibold text-ink">{children}</p>,
    h3: ({ children }) => <p className="mb-1.5 font-semibold text-ink">{children}</p>,
    h4: ({ children }) => <p className="mb-1.5 font-semibold text-ink">{children}</p>,
    h5: ({ children }) => <p className="mb-1.5 font-semibold text-ink">{children}</p>,
    h6: ({ children }) => <p className="mb-1.5 font-semibold text-ink">{children}</p>,
    code: ({ children }) => (
      <code className="rounded bg-surface-2 px-1 py-px font-mono text-[12.5px] text-ink">
        {children}
      </code>
    ),
    pre: ({ children }) => (
      <pre className="mb-2.5 overflow-x-auto rounded-lg border border-border bg-surface-2 p-3 text-[12.5px] last:mb-0">
        {children}
      </pre>
    ),
    blockquote: ({ children }) => (
      <blockquote className="mb-2.5 border-l-2 border-border-strong pl-3 text-ink-muted last:mb-0">
        {children}
      </blockquote>
    ),
    hr: () => <hr className="my-3 border-border" />,
    a: ({ href, children }) => (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-accent-ink underline underline-offset-2 hover:text-accent"
      >
        {children}
      </a>
    ),
    // Documents produce tables often enough that they must not blow out the
    // bubble width; the wrapper scrolls instead.
    table: ({ children }) => (
      <div className="mb-2.5 overflow-x-auto last:mb-0">
        <table className="w-full border-collapse text-[13px]">{children}</table>
      </div>
    ),
    th: ({ children }) => (
      <th className="border border-border bg-surface-2 px-2 py-1 text-left font-semibold">
        {children}
      </th>
    ),
    td: ({ children }) => <td className="border border-border px-2 py-1 align-top">{children}</td>,
    cite: ({ node }) => {
      const chunkId = node?.properties?.dataChunkId as string | undefined;
      const citation = citations.find((c) => c.chunk_id === chunkId);
      if (!citation) return null;
      return (
        <CitationChip
          citation={citation}
          index={citations.indexOf(citation) + 1}
          onSelect={onSelectCitation}
          active={activeCitation === citation.chunk_id}
        />
      );
    },
  };
}

export function Answer({
  answer,
  citations,
  streaming,
  activeCitation,
  onSelectCitation,
}: {
  answer: string;
  citations: Citation[];
  streaming?: boolean;
  activeCitation?: string | null;
  onSelectCitation: (c: Citation) => void;
}) {
  const rehypePlugins = React.useMemo(() => [citationPlugin(citations)], [citations]);
  const components = React.useMemo(
    () => markdownComponents(citations, activeCitation, onSelectCitation),
    [citations, activeCitation, onSelectCitation],
  );

  return (
    <div
      className={cn(
        "script-auto text-[14.5px] leading-relaxed text-ink",
        streaming && "caret-blocks",
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={rehypePlugins}
        components={components}
      >
        {answer}
      </ReactMarkdown>
    </div>
  );
}

/** The expandable source list under an answer. */
export function CitationList({
  citations,
  activeCitation,
  onSelect,
}: {
  citations: Citation[];
  activeCitation: string | null;
  onSelect: (c: Citation) => void;
}) {
  if (!citations.length) return null;

  return (
    <div className="mt-3 space-y-1.5">
      <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-ink-faint">
        <FileText className="h-3 w-3" aria-hidden />
        Sources
      </p>
      <ul className="space-y-1.5">
        {citations.map((citation, i) => {
          const active = activeCitation === citation.chunk_id;
          return (
            <li key={citation.chunk_id}>
              <button
                type="button"
                onClick={() => onSelect(citation)}
                aria-expanded={active}
                className={cn(
                  "w-full rounded-lg border px-3 py-2 text-left transition-colors",
                  active
                    ? "border-accent bg-accent-soft"
                    : "border-border bg-surface-2/60 hover:border-border-strong",
                )}
              >
                <div className="flex items-center gap-2">
                  <Badge tone={active ? "accent" : "neutral"}>{i + 1}</Badge>
                  <span className="script-auto min-w-0 flex-1 truncate text-[12px] font-medium text-ink">
                    {citation.filename}
                  </span>
                  <span className="shrink-0 text-[11px] text-ink-faint">
                    p.{citation.page}
                  </span>
                </div>
                {active ? (
                  <p className="script-auto mt-2 border-l-2 border-accent pl-3 text-[12.5px] leading-relaxed text-ink-muted">
                    {citation.snippet}
                  </p>
                ) : null}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
