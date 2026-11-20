"""Cross-lingual retrieval probe — the smoke test for goal G3.

A Hindi question must find the right English passage and vice versa. Runs the
real bge-m3 model against real Qdrant; no mocks, no stubs.

This is a fast sanity check (7 probes, ~30s), not the §12 evaluation. The full
trilingual gold set and the complete §12.4 ablation grid land in Batch 6 — this
exists so a regression in the cross-lingual path is caught immediately rather
than at eval time.

    make probe-crosslingual
"""

import asyncio
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.retrieval.embeddings.bgem3 import BGEM3EmbeddingProvider
from app.retrieval.rerank import BGEReranker, NoOpReranker
from app.retrieval.service import RetrievalService
from app.retrieval.types import SearchMode
from app.retrieval.vectorstore.base import PendingChunk, SearchFilter
from app.retrieval.vectorstore.qdrant_store import QdrantVectorStore

SESSION = "xl_demo"

# Deliberately mixed: the answer to each query lives in the *other* language.
CORPUS = [
    (
        "en",
        "Applications for a district licence must be filed within thirty days of the notice.",
        "rules_en.pdf",
        1,
    ),
    (
        "en",
        "The committee reviews all appeals at its quarterly meeting in April.",
        "rules_en.pdf",
        2,
    ),
    (
        "en",
        "A penalty of five thousand rupees applies to late submissions.",
        "rules_en.pdf",
        3,
    ),
    (
        "hi",
        "राष्ट्रीय शिक्षा नीति के अंतर्गत प्राथमिक विद्यालयों में मातृभाषा में शिक्षण अनिवार्य किया गया है।",
        "policy_hi.pdf",
        1,
    ),
    (
        "hi",
        "सभी सरकारी कर्मचारियों को प्रतिवर्ष संपत्ति का विवरण प्रस्तुत करना होगा।",
        "policy_hi.pdf",
        2,
    ),
    ("hi", "पेंशन के लिए न्यूनतम बीस वर्ष की सेवा आवश्यक है।", "policy_hi.pdf", 3),
]

# (query, language of query, substring that MUST appear in the top hit)
PROBES = [
    ("How long do I have to submit a licence application?", "en->en", "thirty days"),
    ("लाइसेंस के आवेदन के लिए कितने दिन का समय है?", "hi->EN doc", "thirty days"),
    ("What is the penalty for submitting late?", "en->en", "five thousand"),
    ("देर से जमा करने पर कितना जुर्माना लगता है?", "hi->EN doc", "five thousand"),
    ("What is the minimum service needed for a pension?", "en->HI doc", "बीस वर्ष"),
    ("पेंशन के लिए कितने साल की सेवा चाहिए?", "hi->hi", "बीस वर्ष"),
    ("Which language must primary schools teach in?", "en->HI doc", "मातृभाषा"),
]


async def main() -> int:
    print("Loading bge-m3 (first run downloads ~2.2 GB)...")
    started = time.time()
    embedder = BGEM3EmbeddingProvider()
    embedder.warmup()
    print(f"  loaded on {embedder.device} in {time.time() - started:.1f}s\n")

    collection = f"xl_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(collection=collection, dense_dim=embedder.dense_dim)
    await store.ensure_collection()

    pending = [
        PendingChunk(
            chunk_id=f"c{i}",
            doc_id=f"doc_{fn}",
            session_id=SESSION,
            text=text,
            filename=fn,
            page_start=page,
            page_end=page,
            chunk_index=i,
            language=lang,
            ocr_used=False,
        )
        for i, (lang, text, fn, page) in enumerate(CORPUS)
    ]

    svc = RetrievalService(embedder=embedder, store=store, reranker=NoOpReranker())
    started = time.time()
    await svc.index_chunks(pending)
    print(
        f"Indexed {len(pending)} chunks in {time.time() - started:.1f}s "
        f"({len(pending) / (time.time() - started):.1f} chunks/s)\n"
    )

    scope = SearchFilter(session_id=SESSION)
    # recall@1 is the precision of the final answer; recall@3 is what actually
    # matters for the retrieval stage, because the reranker re-orders whatever
    # candidate set it is handed.
    at1: dict[str, list[bool]] = {"dense": [], "sparse": [], "hybrid": []}
    at3: dict[str, list[bool]] = {"dense": [], "sparse": [], "hybrid": []}

    print(f"{'query':50s} {'direction':12s}  dense    sparse   hybrid")
    print(f"{'':50s} {'':12s}  @1 @3    @1 @3    @1 @3")
    print("-" * 96)
    for query, direction, expected in PROBES:
        row = []
        for mode in (SearchMode.DENSE, SearchMode.SPARSE, SearchMode.HYBRID):
            res = await svc.retrieve(query, scope, mode=mode, top_n=3, rerank=False)
            texts = [c.text for c in res.chunks]
            hit1 = bool(texts) and expected in texts[0]
            hit3 = any(expected in t for t in texts[:3])
            at1[str(mode)].append(hit1)
            at3[str(mode)].append(hit3)
            row.append(f" {'Y' if hit1 else '.'}  {'Y' if hit3 else '.'}   ")
        print(f"{query[:48]:50s} {direction:12s} " + "".join(row))

    print()
    for mode, hits1 in at1.items():
        r1 = sum(hits1) / len(hits1)
        r3 = sum(at3[mode]) / len(at3[mode])
        print(f"  {mode:7s} recall@1 = {r1:.0%}   recall@3 = {r3:.0%}")

    # --- reranking ----------------------------------------------------------
    print("\nLoading reranker...")
    started = time.time()
    reranker = BGEReranker()
    reranker.warmup()
    print(f"  loaded on {reranker.device} in {time.time() - started:.1f}s")

    svc_rr = RetrievalService(embedder=embedder, store=store, reranker=reranker)
    reranked_hits = 0
    latencies = []
    for query, _direction, expected in PROBES:
        started = time.time()
        res = await svc_rr.retrieve(query, scope, top_k=6, top_n=1, rerank=True)
        latencies.append((time.time() - started) * 1000)
        if res.chunks and expected in res.chunks[0].text:
            reranked_hits += 1
    print(
        f"  hybrid+rerank recall@1 = {reranked_hits}/{len(PROBES)} = {reranked_hits / len(PROBES):.0%}"
    )
    print(f"  mean retrieval latency = {sum(latencies) / len(latencies):.0f} ms")

    await store._get_client().delete_collection(collection)
    await store.close()

    ok = sum(at3["hybrid"]) >= len(PROBES) - 1 and reranked_hits >= len(PROBES) - 1
    print(
        "\n"
        + (
            "PASS: cross-lingual retrieval works"
            if ok
            else "FAIL: cross-lingual recall too low"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
