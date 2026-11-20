# Evaluation harness

Everything behind the numbers in [`docs/EVALUATION.md`](../docs/EVALUATION.md).

```
corpus/     16 documents — 6 carry gold labels, 10 are distractors
gold/       182 hand-written questions across Hindi, English and Hinglish
harness/    the machinery: corpus loading, metrics, arms, judge, report
run_eval.py the CLI
results/    generated output, including the baseline the CI gate compares against
```

## Running it

```bash
make eval-validate   # 2s, no models: does every gold label still resolve?
make eval            # the retrieval ablation grid
make eval-full       # + end-to-end generation and the LLM judge
make eval-sweep      # chunk size / overlap sweep
make eval-ci         # the 30-question gate CI runs
```

`make eval` needs Qdrant up (`make infra`) and the models present
(`make install-ml && make download-models`). `--embedder mock` runs the whole
harness without either, which checks the wiring but produces numbers that mean
nothing — the stub is a character n-gram hash, so it scores 0.00 on every
cross-lingual question by construction.

## How relevance is defined

A gold question does **not** name a chunk id. It carries **anchors**: short
substrings that must all appear in a passage for that passage to count as
relevant.

```json
{"id":"hi-003","query":"पूर्ण पेंशन के लिए कितने वर्ष की सेवा आवश्यक है?",
 "lang":"hi","kind":"numeric","doc":"hi_pension_circular",
 "anchors":["न्यूनतम बीस वर्ष की अर्हकारी सेवा"],
 "answer":"न्यूनतम बीस वर्ष की अर्हकारी सेवा।","expect_language":"hi"}
```

Chunk ids would have tied every label to one chunking configuration, and §12.4
sweeps chunk size — under a 250-token target the same answer sits in a different
chunk. Anchors survive that, and they are reviewable: you can check a label by
reading the corpus, not a vector dump.

`make eval-validate` fails if an anchor matches **no** chunk of its document, or
**every** chunk of it. Both are broken labels, and the second is the dangerous
one: it silently inflates recall.

## The corpus

Hand-written rather than scraped, for two reasons that matter more than
authenticity: a scraped corpus cannot be redistributed with the repository, and
real circulars are not annotated, so the gold set would have to be produced by a
model — which makes the evaluation a measurement of that model's agreement with
itself.

Six documents carry questions: an English HR handbook, an English tender notice,
an English information security policy, a Hindi pension circular, a Hindi
education-policy directive and a Hindi agriculture subsidy scheme.

Ten further documents carry no questions at all. They exist to make retrieval
hard: each is *topically adjacent* to a labelled document without duplicating its
facts — a procurement manual beside the tender notice, an RTI circular beside the
pension circular, a seed subsidy scheme beside the irrigation subsidy scheme.
Adjacency is deliberate. A distractor about an unrelated subject is not retrieved
and therefore measures nothing; a distractor that shares vocabulary, register and
document structure is the one that actually competes for a slot in the top 5.

They are deliberately **not** near-duplicates with contradictory numbers. A
second HR handbook saying probation is three months would not make the task
harder, it would make the label wrong: both documents would genuinely answer the
question, and the gold set has no way to say "either is acceptable".

## Question kinds

| kind | n | what it probes |
|---|---|---|
| `factoid` | 34 | the answer is a phrase in one passage |
| `numeric` | 70 | amounts, durations, rates — lexical recall |
| `clause` | 8 | definitions and clause references |
| `crosslingual` | 38 | the question's language is not the document's |
| `summary` | 8 | document-level; no single passage answers it |
| `adversarial` | 24 | answerable from no document — must return "not found" |

`summary` and `adversarial` questions are **excluded** from the retrieval
headline. Every chunk of a document is relevant to a summary of it, so Recall@5
over those is arithmetic rather than measurement; adversarial questions have no
relevant chunk at all. They are scored by the coverage and abstention metrics
instead.

The adversarial set is topically adjacent on purpose. "What is the capital of
France?" is in there as a control, but the questions that earn their place are
the ones like "What is the notice period for resignation?" — which sounds exactly
like the handbook it is not in. That is the failure mode that reaches users.

## Adding a question

1. Find the sentence in `corpus/` that answers it.
2. Pick an anchor: the shortest substring of that sentence that appears nowhere
   else in the same document.
3. Append a line to the right file in `gold/`. Ids are `<prefix>-<nnn>` and must
   be unique across all files.
4. Run `make eval-validate`.
