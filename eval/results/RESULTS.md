# TrialSage — evaluation results

> **50 of 50 gold answers are unreviewed drafts.** Every number below is provisional until they are checked by hand. The gold set was drafted by the same system it grades, which makes it circular until a human verifies it.

Corpus: 39,343 interventional trials, 966,641 eligibility criteria. Question set: 50 labelled questions (17 structured, 17 semantic, 16 hybrid). All models local: llama3.1:8b via Ollama, bge-small-en-v1.5 embeddings.

## Router vs the two baselines

Same 50 questions through all three configurations. They share the synthesizer and the citation guardrail, so the comparison isolates retrieval strategy.

| metric | router | vector-only | SQL-only |
|---|---|---|---|
| **routing accuracy** | 92% | n/a | n/a |
| structured answered correctly | 94% | 0% | 100% |
| semantic answered with citations | 100% | 100% | 12% |
| hybrid answered with citations | 81% | 0% | 0% |
| retrieval term recall | 92% | 92% | 0% |
| hybrid filter precision | 87% | 6% | n/a |
| polarity correct | 100% | 100% | n/a |
| citation validity | 100% | 100% | 100% |
| refusal rate | 2% | 0% | 60% |
| mean latency | 14.9s | 19.1s | 3.8s |
| mean tokens | 1627 | 1288 | 1354 |

## Where each baseline fails

| route | n | router | vector-only | SQL-only |
|---|---|---|---|---|
| structured | 17 | 94% | 0% | 100% |
| semantic | 17 | 100% | 100% | 12% |
| hybrid | 16 | 81% | 0% | 0% |

**Vector-only cannot count.** It scores 0% on structured questions: asked "how many phase 3 diabetes trials started in 2024", it retrieves criteria from diabetes trials and describes them. Fluent, cited, and not an answer to the question.

**Vector-only ignores filters.** On hybrid questions its filter precision is 6% — 94% of the trials it discusses do not satisfy the phase / status / location the question asked for. Because it still cites real NCT IDs, every citation-based metric says it did fine. Only checking the citations against the filter reveals the answer is about the wrong trials.

**SQL-only cannot read prose.** It refuses 60% of the time and manages 12% of semantic questions, because "prior immunotherapy failure" is not a column. It is the strongest configuration on structured questions (100%) and useless beyond them.


## Routing: rules beat the local LLM

Measured three ways on the same 50 questions:

| router strategy | accuracy | latency | tokens |
|---|---|---|---|
| **deterministic rules** (shipped) | **92%** | ~0 ms | 0 |
| rules, LLM consulted when unsure | 88% | varies | ~250 |
| LLM classifier (llama3.1:8b) | 76% | ~2–9 s | ~250 |

The local model's characteristic error is over-routing to `hybrid`: it reads "exclude patients with X" as a structured filter and sends a purely semantic question down the hybrid path, where an empty SQL filter produces a false "no matching trials found". Letting it arbitrate only the low-confidence cases still lost 4 questions the rules had right.

The LLM is retained for what it is genuinely better at: splitting a hybrid question into its structured and semantic halves.


## Reranker: before / after

`bge-reranker-base` cross-encoder over the top-50 shortlist.

| precision@k | without reranker | with reranker | delta |
|---|---|---|---|
| @1 | 100.0% | 96.9% | -3.1% |
| @3 | 94.8% | 94.8% | +0.0% |
| @5 | 94.4% | 95.6% | +1.3% |
| @10 | 95.0% | 94.1% | -0.9% |
| mean latency | 14.9s | 17.5s | +2.6s |

**The reranker does not help here.** Every delta is within ±3%, which is noise at n=33, and it costs 17% more latency. The reason is headroom: the bi-encoder already retrieves ~95% on-topic criteria, so there is almost nothing for a reranker to recover. It is wired in and off by default (`--rerank` to enable); on a corpus with noisier retrieval it would likely earn its place.


## Grounding: faithfulness

| metric | value |
|---|---|
| claims judged | 101 |
| supported by retrieved context | 99 |
| **faithfulness (as judged)** | **98.0%** |
| faithfulness (after manual review of flagged claims) | **99.0%** |

**RAGAS could not be run with the local judge.** llama3.1:8b deterministically wraps its JSON in prose and markdown headers, which RAGAS's output parser rejects — 3/3 failures in isolation, 12/12 in the harness, across three separate configurations (concurrency, context size, metric selection). This is an incompatibility with the judge model, not a transient error. `eval/faithfulness.py` measures the same property with one YES/NO question per claim, which an 8B model answers reliably.

Both claims the judge flagged were checked by hand. One was a judge error (the context did support it). The other was a real defect worth recording: the synthesizer wrote *"none of the retrieved trials allow prior stem cell transplant"* when two of the ten retrieved criteria were INCLUSIONS reading *"Prior stem cell transplant allowed"*. Notably the polarity metric scored 100% on this answer, because it only checks the exclusion-described-as-allowed direction. Over-generalising to "none" is the mirror failure and needs its own check.


---

*Regenerate with `python -m eval.report`. Raw per-question output is in `eval/results/*.json`.*
