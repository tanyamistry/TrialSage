# TrialSage

A hybrid agentic RAG assistant over public ClinicalTrials.gov data. Ask a
question in plain English, get a grounded answer with an NCT ID citation for
every trial-specific claim.

**Status: Phase 2 of 5 complete** (both retrievers working independently). See [Roadmap](#roadmap).

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

### Loaded (diabetes)

| | |
|---|---|
| Trials | 2,696 |
| Eligibility criteria | 51,646 |
| Sites | 36,922 |
| Conditions | 4,968 |
| Interventions | 6,069 |
| Ingest time | ~7s cold, ~3s cached |

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
| Mean latency | 1.5 s |

Execution accuracy compares query *results* against gold SQL rather than
comparing SQL strings — there are many correct ways to write the same query.
"Lenient" means the right value was present but in an unexpected column
(the model returned an extra label alongside the number).

**A local 8B model is good enough at SQL here, so no API fallback is needed.**
That was the main open risk going into this phase. Two things made it work:
the denormalized single-table view (no joins to hallucinate) and putting the
real enum values in the prompt.

### Semantic search

pgvector HNSW over ~1M eligibility criteria, bge-small-en-v1.5 (384-dim),
embedded locally on the Apple GPU at ~180 chunks/sec. Typical query latency is
~200 ms warm, with top hits scoring 0.9+ cosine similarity.

Every hit carries its **polarity** (inclusion vs exclusion), and search can be
restricted to a candidate set of NCT IDs — that restriction is the mechanism
the Phase 3 hybrid route is built on.

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
| 3 | Router + hybrid path + synthesizer with citation guardrails | next |
| 4 | Eval harness, vector-only and SQL-only baselines, reranker before/after | |
| 5 | Streamlit UI, tracing, architecture diagram and results | |

## Data source

ClinicalTrials.gov API v2, public and unauthenticated. Corpus scope:
interventional, started 2018 or later, phase 1–4, in oncology, diabetes or
cardiovascular — about 41,000 trials. See [docs/api_fields.md](docs/api_fields.md).
