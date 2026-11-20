"""Document intelligence: what PolyDoc does *with* documents beyond answering.

Retrieval-augmented generation answers the question asked. The modules here
act on the documents themselves — extracting deadlines, executable eligibility
rules and formulas, diffing versions, scanning for planted instructions — and
every one of them keeps the same contract as the chat path: each output is tied
back to a quoted, page-referenced passage that a person can check.

Deterministic where it can be (dates, diffs, arithmetic, PII, injection scans);
LLM-assisted only for reading comprehension, and then the model's output is
validated against the source text before anything is returned.
"""
