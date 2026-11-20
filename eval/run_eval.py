"""PolyDoc evaluation harness (PRD §12).

    make eval                 # full ablation grid, retrieval only
    make eval-full            # ablations + end-to-end generation + LLM judge
    make eval-ci              # 30-question subset, gated against the baseline
    make eval-sweep           # chunk size / overlap sweep

Run `python eval/run_eval.py --help` for the full flag list.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import RESULTS_DIR
from harness.arms import SHIPPED_ARM, resolve
from harness.corpus import load_corpus
from harness.generation_eval import GenerationReport, run_generation
from harness.goldset import QueryKind, QueryLang, load_goldset, validate
from harness.index import Backends, build_embedder, build_reranker, ensure_index
from harness.metrics import ArmSummary
from harness.report import delta, pct, table, write_json, write_markdown
from harness.retrieval_eval import ArmResult, retrieval_pool, run_arm

# The harness prints its own progress; the pipeline's structured logs would bury
# it under one `retrieval_complete` line per question per arm.
logging.disable(logging.INFO)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="run_eval", description=__doc__)
    parser.add_argument("--arms", nargs="*", default=None, help="Ablation arm ids (default: all).")
    parser.add_argument(
        "--embedder",
        choices=("bge", "mock"),
        default="bge",
        help="`mock` runs without downloading models — wiring check only, not a result.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stratified subset of N questions.")
    parser.add_argument("--top-k", type=int, default=0, help="Candidates before rerank.")
    parser.add_argument("--generation", action="store_true", help="Also run the answer pipeline.")
    parser.add_argument("--judge", action="store_true", help="Score answers with an LLM judge.")
    parser.add_argument(
        "--sweep", action="store_true", help="Chunk size/overlap sweep (re-indexes per arm)."
    )
    parser.add_argument("--force-index", action="store_true", help="Rebuild the vector index.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the gold set resolves against the corpus and exit. Needs no models.",
    )
    parser.add_argument(
        "--out", type=Path, default=RESULTS_DIR / "latest.json", help="JSON results path."
    )
    parser.add_argument("--markdown", type=Path, default=None, help="Write a Markdown report here.")
    parser.add_argument(
        "--baseline", type=Path, default=None, help="Compare against this results JSON."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help="Max allowed Recall@5 drop vs the baseline before exiting non-zero.",
    )
    return parser.parse_args(argv)


# --- rendering ---------------------------------------------------------------


def render_ablations(results: list[ArmResult]) -> str:
    # Deltas are against dense-only, which is the arm every other one adds a
    # stage to. Falling back to the first arm keeps `--arms hybrid_rerank` from
    # crashing, though the column then compares a row with itself.
    reference = next((r for r in results if r.arm.is_baseline), results[0] if results else None)
    baseline = reference.summary.hit_at_5 if reference else None
    rows = []
    for result in results:
        s = result.summary
        low, high = s.recall_at_5_ci
        rows.append(
            [
                result.arm.label,
                pct(s.hit_at_1),
                pct(s.hit_at_5),
                delta(s.hit_at_5, baseline),
                f"{low:.2f}–{high:.2f}",
                pct(s.mrr),
                pct(s.ndcg_at_5),
                f"{s.latency_p50_ms:.0f}",
                f"{s.latency_p95_ms:.0f}",
            ]
        )
    return table(
        [
            "arm",
            "Hit@1",
            "Hit@5",
            "Δ vs dense",
            "Hit@5 95% CI",
            "MRR",
            "nDCG@5",
            "P50 ms",
            "P95 ms",
        ],
        rows,
    )


def render_slices(result: ArmResult, key: str, values: list[str]) -> str:
    rows = []
    for value in values:
        summary = result.slice_summary(value, lambda s, v=value: getattr(s, key) == v)
        if summary is None:
            continue
        rows.append(
            [
                value,
                str(summary.n),
                pct(summary.hit_at_1),
                pct(summary.hit_at_5),
                pct(summary.mrr),
                pct(summary.ndcg_at_5),
            ]
        )
    return table([key, "n", "Hit@1", "Hit@5", "MRR", "nDCG@5"], rows)


def render_hinglish_ablation(results: list[ArmResult]) -> str | None:
    """The Hinglish normalisation ablation, restricted to Hinglish questions.

    Reported on the slice rather than the pool because dual-query only fires on a
    romanized query: averaged over 150 questions the effect is diluted by the 118
    it cannot possibly change, and a real +18 points on Hinglish would print as
    +4 overall.
    """
    with_norm = next((r for r in results if r.arm.id == SHIPPED_ARM), None)
    without = next((r for r in results if r.arm.id == "hybrid_rerank_no_hinglish"), None)
    if not with_norm or not without:
        return None

    def slice_of(result: ArmResult) -> ArmSummary | None:
        return result.slice_summary("hinglish", lambda s: s.lang == "hinglish")

    a, b = slice_of(with_norm), slice_of(without)
    if not a or not b:
        return None

    return table(
        ["configuration", "n", "Hit@1", "Hit@5", "MRR"],
        [
            ["with Hinglish normalisation", str(a.n), pct(a.hit_at_1), pct(a.hit_at_5), pct(a.mrr)],
            [
                "without (romanized query only)",
                str(b.n),
                pct(b.hit_at_1),
                pct(b.hit_at_5),
                pct(b.mrr),
            ],
            [
                "delta (points)",
                "",
                delta(a.hit_at_1, b.hit_at_1),
                delta(a.hit_at_5, b.hit_at_5),
                delta(a.mrr, b.mrr),
            ],
        ],
    )


def render_generation(report: GenerationReport, *, judged: bool) -> str:
    inp, out = report.mean_tokens
    rows = [
        ["Language fidelity (FR-21)", pct(report.language_fidelity), "≥ 0.95"],
        ["Anti-hallucination (abstention accuracy)", pct(report.abstention_accuracy), "≥ 0.95"],
        ["  └ abstention recall (adversarial refused)", pct(report.abstention_recall), "—"],
        ["  └ abstention precision (refusals correct)", pct(report.abstention_precision), "—"],
        ["  └ false abstention on answerable", pct(report.false_abstention), "—"],
        ["Grounded answers (citation resolved)", pct(report.grounded_rate), "—"],
        ["Citation points at a passage with the answer", pct(report.citation_accuracy), "—"],
    ]
    if judged:
        rows.extend(
            [
                ["RAGAS-equivalent faithfulness", pct(report.judged("faithfulness")), "≥ 0.85"],
                [
                    "RAGAS-equivalent answer relevancy",
                    pct(report.judged("answer_relevancy")),
                    "≥ 0.85",
                ],
                ["Answer correctness vs reference", pct(report.judged("correctness")), "—"],
                ["Context precision", pct(report.judged("context_precision")), "—"],
                ["  └ answers successfully judged", str(report.judge_coverage), "—"],
            ]
        )
    rows.extend(
        [
            ["P50 chat latency (ms)", f"{report.latency(50):.0f}", "—"],
            ["P95 chat latency (ms)", f"{report.latency(95):.0f}", "≤ 9000"],
            ["Mean input tokens", f"{inp:.0f}", "—"],
            ["Mean output tokens", f"{out:.0f}", "—"],
            ["Mean cost per query (USD)", f"{report.mean_cost_usd:.6f}", "—"],
            ["Total run cost (USD)", f"{report.total_cost_usd:.4f}", "—"],
        ]
    )
    if report.errors:
        rows.append(["Provider errors (excluded from means)", str(len(report.errors)), "—"])
    return table(["metric", "value", "target"], rows, align_right_from=1)


# --- the run -----------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    started = time.time()
    corpus = load_corpus()
    gold = load_goldset()
    validate(gold, corpus)
    if args.limit:
        gold = gold.sample(args.limit)
        validate(gold, corpus)

    pool = len(retrieval_pool(gold))
    print(
        f"corpus  : {len(corpus.documents)} documents, {len(corpus.chunks)} chunks "
        f"(target {corpus.chunking.target_tokens} tok, overlap {corpus.chunking.overlap_ratio:.0%})"
    )
    print(
        f"gold set: {len(gold)} questions — {pool} scored for retrieval, "
        f"{len(gold.adversarial)} adversarial, {len(gold.of_kind(QueryKind.SUMMARY))} summary"
    )

    arms = resolve(args.arms)
    # The stub embedder pairs with the stub reranker. Loading a 2.2 GB
    # cross-encoder to rerank hash-vector candidates costs minutes and cannot
    # make the mock numbers mean anything — `--embedder mock` is a wiring check.
    needs_rerank = (any(a.rerank for a in arms) or args.generation) and args.embedder != "mock"

    print(f"loading embedder ({args.embedder}) ...", flush=True)
    embedder = build_embedder(args.embedder)
    reranker = build_reranker(needs_rerank)
    store = await ensure_index(corpus, embedder, force=args.force_index)
    backends = Backends(embedder, reranker, store, store.collection)
    print(f"index   : {store.collection}\n")

    top_k = args.top_k or None
    from app.core.config import settings

    top_k = top_k or settings.retrieval.top_k

    results: list[ArmResult] = []
    for arm in arms:
        arm_started = time.time()
        print(f"  {arm.label:38s} ", end="", flush=True)
        result = await run_arm(arm, gold, corpus, backends, top_k=top_k)
        results.append(result)
        s = result.summary
        print(f"Hit@5 {s.hit_at_5:.3f}   MRR {s.mrr:.3f}   ({time.time() - arm_started:.0f}s)")

    print("\n" + render_ablations(results) + "\n")

    shipped = next((r for r in results if r.arm.id == SHIPPED_ARM), results[-1])
    sections: list[tuple[str, str]] = [
        (f"Ablation grid — retrieval only, n={pool}", render_ablations(results)),
        (
            f"Shipped configuration ({shipped.arm.label}) by question type",
            render_slices(shipped, "kind", [str(k) for k in QueryKind]),
        ),
        (
            f"Shipped configuration ({shipped.arm.label}) by query language",
            render_slices(shipped, "lang", [str(v) for v in QueryLang]),
        ),
    ]
    hinglish = render_hinglish_ablation(results)
    if hinglish:
        sections.append(("Hinglish normalisation ablation (Hinglish questions only)", hinglish))

    print(sections[1][0] + "\n" + sections[1][1] + "\n")
    print(sections[2][0] + "\n" + sections[2][1] + "\n")
    if hinglish:
        print(sections[3][0] + "\n" + hinglish + "\n")

    misses = shipped.misses
    if misses:
        print(f"{len(misses)} question(s) missed by the shipped arm at k=5:")
        for score in misses[:15]:
            print(f"    {score.question_id}  ({score.kind}/{score.lang})")
        if len(misses) > 15:
            print(f"    ... and {len(misses) - 15} more")
        print()

    generation: GenerationReport | None = None
    if args.generation:
        judge_provider = None
        if args.judge:
            from app.generation.service import get_provider

            judge_provider = get_provider()
            print(f"judge   : {judge_provider.name}/{judge_provider.model}")

        print(f"running end-to-end generation over {len(gold)} questions ...", flush=True)
        done = 0

        def progress(_question, _report) -> None:
            nonlocal done
            done += 1
            if done % 10 == 0 or done == len(gold):
                print(f"    {done}/{len(gold)}", flush=True)

        generation = await run_generation(
            gold, corpus, backends, judge_provider=judge_provider, progress=progress
        )
        block = render_generation(generation, judged=args.judge)
        print("\n" + block + "\n")
        sections.append((f"Generation and ops, n={len(gold)}", block))

    payload = {
        "corpus": {
            "documents": len(corpus.documents),
            "chunks": len(corpus.chunks),
            "fingerprint": corpus.fingerprint,
            "target_tokens": corpus.chunking.target_tokens,
            "overlap_ratio": corpus.chunking.overlap_ratio,
        },
        "goldset": {
            "total": len(gold),
            "retrieval_pool": pool,
            "adversarial": len(gold.adversarial),
        },
        "embedder": embedder.name,
        "reranker": reranker.name,
        "top_k": top_k,
        "arms": {r.arm.id: r.summary for r in results},
        "per_question": {r.arm.id: r.scores for r in results},
    }
    if generation is not None:
        payload["generation"] = {
            "language_fidelity": generation.language_fidelity,
            "abstention_accuracy": generation.abstention_accuracy,
            "abstention_recall": generation.abstention_recall,
            "abstention_precision": generation.abstention_precision,
            "false_abstention": generation.false_abstention,
            "grounded_rate": generation.grounded_rate,
            "citation_accuracy": generation.citation_accuracy,
            "faithfulness": generation.judged("faithfulness"),
            "answer_relevancy": generation.judged("answer_relevancy"),
            "correctness": generation.judged("correctness"),
            "context_precision": generation.judged("context_precision"),
            "latency_p50_ms": generation.latency(50),
            "latency_p95_ms": generation.latency(95),
            "mean_cost_usd": generation.mean_cost_usd,
            "errors": len(generation.errors),
            "per_question": generation.scores,
        }

    write_json(args.out, payload)
    print(f"results → {args.out}")

    if args.markdown:
        preamble = (
            f"<!-- Generated by `python eval/run_eval.py`. Do not edit by hand. -->\n\n"
            f"Corpus: {len(corpus.documents)} documents / {len(corpus.chunks)} chunks. "
            f"Gold set: {len(gold)} questions ({pool} scored for retrieval). "
            f"Embedder: `{embedder.name}`, reranker: `{reranker.name}`, top_k={top_k}."
        )
        write_markdown(args.markdown, sections, preamble=preamble)
        print(f"markdown → {args.markdown}")

    await store.close()
    print(f"\ntotal {time.time() - started:.0f}s")

    if args.baseline:
        return gate(payload, args.baseline, args.tolerance)
    return 0


def gate(payload: dict, baseline_path: Path, tolerance: float) -> int:
    """Fail the build when the shipped arm's Recall@5 regresses (Tier-1 item 2)."""
    import json

    if not baseline_path.exists():
        print(f"\nno baseline at {baseline_path} — nothing to gate against")
        return 0

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = payload["arms"].get(SHIPPED_ARM)
    previous = baseline.get("arms", {}).get(SHIPPED_ARM)
    if current is None or previous is None:
        print(f"\nbaseline has no `{SHIPPED_ARM}` arm — nothing to gate against")
        return 0

    now = current.hit_at_5 if hasattr(current, "hit_at_5") else current["hit_at_5"]
    before = previous["hit_at_5"]
    drop = before - now

    print(
        f"\ngate: Hit@5 {before:.3f} (baseline) → {now:.3f} (current), "
        f"delta {(-drop) * 100:+.1f} pts, tolerance {tolerance * 100:.0f} pts"
    )
    if drop > tolerance:
        print("FAIL: retrieval quality regressed beyond tolerance")
        return 1
    print("PASS")
    return 0


async def run_sweep(args: argparse.Namespace) -> int:
    """§12.4 chunk size / overlap sweep.

    Each configuration is a separate index — chunk boundaries move, so the
    vectors are not reusable — which is why this is a separate target rather
    than another row in the ablation grid.
    """
    from app.core.config import settings

    # `model_copy`, not `dataclasses.replace`: ChunkingSettings is a pydantic
    # model, and `replace` raises on one rather than returning a modified copy.
    base = settings.chunking
    grid = [
        base.model_copy(update={"target_tokens": 250, "min_tokens": 100, "max_tokens": 400}),
        base.model_copy(update={"target_tokens": 400, "min_tokens": 150, "max_tokens": 600}),
        base.model_copy(update={"target_tokens": 550, "min_tokens": 200, "max_tokens": 800}),
        base.model_copy(update={"target_tokens": 750, "min_tokens": 300, "max_tokens": 1000}),
        base.model_copy(
            update={
                "target_tokens": 550,
                "min_tokens": 200,
                "max_tokens": 800,
                "overlap_ratio": 0.0,
            }
        ),
        base.model_copy(
            update={
                "target_tokens": 550,
                "min_tokens": 200,
                "max_tokens": 800,
                "overlap_ratio": 0.25,
            }
        ),
    ]

    print(f"loading embedder ({args.embedder}) ...", flush=True)
    embedder = build_embedder(args.embedder)
    reranker = build_reranker(True)
    gold = load_goldset()

    from harness.arms import ARMS_BY_ID

    arm = ARMS_BY_ID[SHIPPED_ARM]
    rows = []
    payload_arms = {}

    for cfg in grid:
        corpus = load_corpus(chunking=cfg)
        validate(gold, corpus)
        store = await ensure_index(corpus, embedder, prefix="sweep")
        backends = Backends(embedder, reranker, store, store.collection)
        label = f"{cfg.target_tokens} tok / {cfg.overlap_ratio:.0%} overlap"
        print(f"  {label:28s} {len(corpus.chunks):4d} chunks ", end="", flush=True)

        result = await run_arm(arm, gold, corpus, backends, top_k=args.top_k or 20)
        s = result.summary
        print(f"Hit@5 {s.hit_at_5:.3f}")
        rows.append(
            [
                label,
                str(len(corpus.chunks)),
                pct(s.hit_at_1),
                pct(s.hit_at_5),
                pct(s.mrr),
                pct(s.ndcg_at_5),
                f"{s.latency_p50_ms:.0f}",
            ]
        )
        payload_arms[label] = s
        await store.close()

    block = table(["chunking", "chunks", "Hit@1", "Hit@5", "MRR", "nDCG@5", "P50 ms"], rows)
    print("\n" + block + "\n")
    write_json(args.out, {"sweep": payload_arms})
    print(f"results → {args.out}")
    if args.markdown:
        write_markdown(args.markdown, [("Chunk size and overlap sweep", block)])
        print(f"markdown → {args.markdown}")
    return 0


def validate_only() -> int:
    """Resolve the gold set against the corpus without loading a model.

    This is the fast CI check (Tier-1 item 2): a mistyped anchor, a renamed
    document or a chunker change that splits an answer sentence in half all show
    up here in two seconds, instead of as an unexplained recall drop after a
    fifteen-minute model run.
    """
    corpus = load_corpus()
    gold = load_goldset()
    report = validate(gold, corpus, strict=False)

    print(f"corpus  : {len(corpus.documents)} documents, {len(corpus.chunks)} chunks")
    print(
        f"gold set: {report.total} questions "
        f"({report.answerable} answerable, {report.adversarial} adversarial)"
    )
    resolved = sum(len(q.relevant_chunk_ids) for q in gold.answerable)
    print(f"anchors : {resolved} chunk labels resolved")

    if report.problems:
        print(f"\nFAIL: {len(report.problems)} unresolved label(s)")
        for problem in report.problems:
            print(f"  - {problem}")
        return 1
    print("\nPASS: every gold label resolves to a discriminating chunk")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_only:
        return validate_only()
    coroutine = run_sweep(args) if args.sweep else run(args)
    return asyncio.run(coroutine)


if __name__ == "__main__":
    raise SystemExit(main())
