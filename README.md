# TrialSage

A hybrid agentic RAG assistant over public ClinicalTrials.gov data. Ask a
question in plain English, get a grounded answer with an NCT ID citation for
every trial-specific claim.

**Status: Phase 3 of 5 complete and verified** — router, hybrid path and cited synthesis
work end to end. All three example questions answer correctly; `eval/phase3_verify.py`
passes 23/23. See [Roadmap](#roadmap).

## The idea

Each trial holds two kinds of information that want two different retrievers:

| | Example | Best tool |
|---|---|---|
| **Structured** | phase, status, enrollment, dates, location | SQL |
| **Free text** | eligibility criteria — who can and cannot join | vector search |

Naive RAG cannot count, and text-to-SQL cannot reason about "prior
immunotherapy failure". So a **router** classifies each question as
`structured`, `semantic`, or `hybrid` and dispatches accordingly. For hybrid
questions the SQL filter runs *first* to produce a candidate set, and the vector
search is restricted to those trials — more accurate and cheaper than searching
everything.

Target questions:

- *How many phase 3 diabetes trials started in 2024?* → structured
- *Find trials whose eligibility mentions prior immunotherapy failure* → semantic
- *Which recruiting phase 2 oncology trials in Massachusetts allow patients with a history of autoimmune disease?* → hybrid

## Setup

Requires Docker and Python 3.11+.

```bash
git clone <this repo> && cd TrialSage
make install      # creates .venv, installs deps, copies .env.example -> .env
make db-up        # starts Postgres+pgvector on port 5433, applies schema
make ingest       # fetches and loads diabetes (~2,700 trials, under a minute)
make verify       # prints a data-quality report
make test         # full test suite
```

Then the retrievers (needs `ollama pull llama3.1:8b` first):

```bash
make ingest AREA=cardiovascular    # + AREA=oncology for the full corpus
make embed                         # embed criteria, build the HNSW index
make sql-smoke                     # text-to-SQL accuracy against gold SQL
make sql      Q="how many phase 3 trials started in 2024"
make semantic Q="prior immunotherapy failure"
```

`make help` lists every target. To start over: `make db-reset`.

### Configuration

- **`.env`** — credentials and the LLM provider. Gitignored; `.env.example` is the template.
- **`config.yaml`** — everything else: therapeutic areas, corpus filter, chunking, model names, retrieval settings.

The LLM provider is swappable via `LLM_PROVIDER` (`ollama` | `anthropic` |
`openai`). Default is local Ollama, so nothing costs money. `SQL_LLM_PROVIDER`
can override the provider for the SQL agent alone.

## What Phase 1 built

```
ClinicalTrials.gov API v2
        │  fetch.py    paginated, cached to data/raw/<area>.jsonl
        ▼
     parse.py          verified field paths -> dataclasses
        │
        ├── ages.py         "6 Months" -> 0.5 years
        └── eligibility.py  raw text -> per-criterion chunks, tagged
        │                   inclusion / exclusion / unspecified
        ▼
     load.py           upsert into Postgres, refresh v_trials
```

### Loaded corpus

| | |
|---|---|
| Trials | 39,343 |
| Eligibility criteria | 965,998 |
| — inclusion / exclusion / unspecified | 410,544 / 532,528 / 22,926 |
| Sites | 550,934 |
| Areas | oncology 30,589 · cardiovascular 7,937 · diabetes 2,696 |
| Trials in more than one area | 1,869 |
| Database size | 4.0 GB (1.5 GB of that is the HNSW index) |

Exclusion criteria outnumber inclusion roughly 1.3:1 — verified against an
independent bullet count of the raw text, not just assumed.

### The two normalisation jobs

**Age parsing** (`ingest/ages.py`) turns `"18 Years"`, `"6 Months"`, `"N/A"`
into a numeric years column. The important distinction is `None` (no age
stated) versus `0.0` (birth — real, and used by neonatal trials). Conflating
them would make "trials open to newborns" match every trial that simply never
specified a lower bound.

**Eligibility splitting** (`ingest/eligibility.py`) splits raw text into
inclusion and exclusion sections and chunks **per criterion**, not by token
window. This matters because a fixed 512-token window routinely straddles the
Inclusion/Exclusion boundary, producing a chunk that is half one polarity and
half the other — and "includes patients with autoimmune disease" versus
"excludes patients with autoimmune disease" are opposite facts.

Real-corpus behaviours it handles, all found by profiling live records:

- Escaped markdown (`\> 365 days`, `\[COG\]`) — unescaped before chunking.
- Three bullet styles: `*` (82%), numbered (15%), none (2%).
- Missing headers (5% of trials) — tagged `unspecified` rather than guessed at.
- Short but real criteria — "Pregnancy", "Prisoners", "Hypertension" are
  genuine exclusions under 15 characters, so junk is filtered by *shape*
  (no letters, or ends in a colon) rather than by length.

## What Phase 2 built

Two retrievers, deliberately kept independently runnable. Being able to
interrogate each one alone is what will make the Phase 3 router debuggable —
when a hybrid answer looks wrong, you need to know which half is at fault.

### Text-to-SQL

`question → LLM → guard → execute`, with the Postgres error fed back for a
repair attempt. Measured against hand-written gold SQL on 12 questions
(`make sql-smoke`):

| Metric | llama3.1:8b (local) |
|---|---|
| Execution accuracy (strict) | 11/12 (92%) |
| Execution accuracy (lenient) | 12/12 (100%) |
| Correct on first attempt | 11/12 (92%) |
| Guard rejections | 0 |
| Mean latency | 2.3 s |

Execution accuracy compares query *results* against gold SQL rather than
comparing SQL strings — there are many correct ways to write the same query.
"Lenient" means the right value was present but in an unexpected column
(the model returned an extra label alongside the number).

**A local 8B model is good enough at SQL here, so no API fallback is needed.**
That was the main open risk going into this phase. Two things made it work:
the denormalized single-table view (no joins to hallucinate) and putting the
real enum values in the prompt.

### Semantic search

pgvector HNSW over **965,998** eligibility criteria, bge-small-en-v1.5
(384-dim), embedded locally on the Apple GPU. The index is 1.5 GB; the whole
database is 4 GB.

Measured warm latency (median of 5, after a one-time 6.8 s model load):

| Query | Median |
|---|---|
| unfiltered, top 10 criteria | 25 ms |
| unfiltered, top 10 trials (deduplicated) | 25 ms |
| polarity-filtered (exclusion only) | 28 ms |
| scoped to a 434-trial candidate set | 43 ms |

Every hit carries its **polarity** (inclusion vs exclusion), and search can be
restricted to a candidate set of NCT IDs — that restriction is the mechanism
the Phase 3 hybrid route is built on.

Filtered search needs `hnsw.iterative_scan`. By default pgvector walks the
graph for `ef_search` candidates and applies the `WHERE` clause *afterwards*,
so an inclusion-only search can return zero rows even though 410k inclusion
criteria are indexed — and it returns them as an empty result, not an error.
`eval/phase2_verify.py` asserts that filtered searches come back full,
specifically to catch that regression.

## What Phase 3 built

```
question
   |
   v
ROUTER  ── LLM classify, rule-based fallback ──> {route, confidence, reasoning}
   |
   +── structured ──> text-to-SQL ─────────────────┐
   |                                               |
   +── semantic ────> vector search ───────────────┤
   |                                               |
   +── hybrid ──────> SQL filter FIRST             |
                      -> candidate NCT IDs         |
                      -> vector search scoped ─────┤
                                                   v
                                            SYNTHESIZER
                                                   |
                                          CITATION GUARDRAIL
                                                   |
                                            answer + trace
```

### The router is inspectable

Every decision returns the route, a confidence, a plain-language reason, and
which mechanism decided (`llm`, `rules`, or `llm+rules`):

```bash
make ask Q="which recruiting phase 2 oncology trials in Massachusetts allow autoimmune history?"
```
```
[route: hybrid | confidence 0.90 | via llm]
  why: clear structured filter and eligibility concept
```

`--explain` additionally shows the generated SQL, the candidate count, the
retrieved criteria with their polarity, and the stage-by-stage timing.

There is always a **deterministic rule-based fallback**. It triggers when the
local model is unreachable, returns unparseable JSON, invents a route name, or
reports low confidence — and `source` records what actually decided, so a
degraded decision is visible rather than disguised as a confident one. The
fallback routes all three example questions correctly with no LLM call at all.

### The hybrid route splits the question

The router separates a hybrid question into its two halves:

| | |
|---|---|
| structured half | "recruiting phase 2 oncology trials in Massachusetts" |
| semantic half | "history of autoimmune disease" |

This split is load-bearing. Sending the whole question to the SQL agent made it
try to express the medical concept as a column filter —
`AND 'autoimmune disease' = ANY(conditions)` — which matches nothing and turned
a good question into a false "no trials found".

### Citation guardrail

Two failure modes, handled differently:

- **Fabricated citations** — an NCT ID never present in the retrieved context.
  Always neutralised, replaced with `[unverified: NCTxxxxxxxx]` by default.
  Silent deletion would leave the sentence intact and still looking sourced.
- **Uncited claims** — a trial-specific assertion naming no trial. Flagged, not
  removed, because the detection is heuristic.

Aggregate summaries ("None of the trials allow X") are exempt *when the answer
cites specifics elsewhere* — a guardrail that cries wolf on correct answers is
one people learn to ignore.

### Tracing

Every query appends a JSON line to `logs/traces.jsonl` with the route,
confidence, per-stage latency, token counts and guardrail outcome. Latency is
split by stage because "14 seconds" is not actionable but "route 3.0s,
retrieve 3.9s, synth 9.3s" is.

Measured over the verification run (llama3.1:8b, all local):

| Route | Total | Route | Retrieve | Synth |
|---|---|---|---|---|
| structured | 3.6–18.3 s | 0.0–9.0 s | 2.1–7.8 s | 1.5 s |
| semantic | 18.8 s | 2.3 s | 7.6 s | 8.9 s |
| hybrid | 16.1–37.3 s | 3.0–8.8 s | 3.9–14.4 s | 9.3–14.1 s |

Mean 21.1 s and 2,045 tokens per query. Zero fabricated citations across every
traced query.


## Safety

The text-to-SQL agent (Phase 2) is constrained by three independent layers,
because prompt instructions are not a security control:

1. **Database grants** — `trialsage_ro` has `SELECT` on two views and no write
   grant anywhere. Base tables are invisible to it.
2. **Statement validation** — generated SQL is parsed and rejected unless it is
   a single `SELECT` over whitelisted relations, with a row cap injected.
3. **Session defaults** — read-only transactions and a statement timeout,
   applied at the role level so every connection inherits them.

Layers 1 and 2 are live and tested independently. `tests/test_readonly_role.py`
proves the database refuses writes, DDL and stacked statements, and that
`SELECT * FROM trials` gets `permission denied`.
`tests/test_sql_guard.py` attacks the parser with stacked statements,
`SELECT ... INTO`, comment-hidden payloads, CTEs laundering forbidden tables,
`pg_catalog` reads, and filesystem functions.

The guard is a parse-and-inspect allowlist, not a keyword blocklist — keyword
filtering is trivially defeated by casing, comments, or whitespace, whereas
rejecting anything whose syntax tree is not a plain SELECT over approved
relations is not.

The synthesizer will answer only from retrieved context, cite an NCT ID for
every trial-specific claim, and say "no matching trials found" rather than
inventing anything.

## Layout

```
sql/            schema, views, indexes, read-only role  (idempotent migrations)
src/trialsage/
  config.py     config.yaml + .env
  db.py         connect() read/write, connect_readonly() for the SQL agent
  ingest/       fetch, parse, ages, eligibility, load, run, verify
  llm/          swappable providers (ollama | anthropic | openai) + roles
  embed/        bge-small embeddings, resumable bulk embed, HNSW build
  retrieval/    guard, schema_card, sql_agent, semantic, cli
eval/           sql_smoke.py — text-to-SQL accuracy vs gold SQL
docs/           api_fields.md — verified API paths and profiling notes
tests/          parsers, SQL guard, LLM config, read-only role, semantic search
```

`v_trials` is a materialized view that flattens phases, conditions, sites and
countries into array columns, so most structured questions become single-table
queries with no joins. That is a deliberate accommodation for a small local
model, and it doubles as the security boundary.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Ingest, normalise, load structured tables | ✅ done |
| 2 | Semantic search and text-to-SQL, separately; ingest remaining areas | ✅ done |
| 3 | Router + hybrid path + synthesizer with citation guardrails | ✅ done |
| 4 | Eval harness, vector-only and SQL-only baselines, reranker before/after | next |
| 5 | Streamlit UI, tracing, architecture diagram and results | |

## Data source

ClinicalTrials.gov API v2, public and unauthenticated. Corpus scope:
interventional, started 2018 or later, phase 1–4, in oncology, diabetes or
cardiovascular — about 41,000 trials. See [docs/api_fields.md](docs/api_fields.md).
