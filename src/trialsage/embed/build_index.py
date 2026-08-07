"""Populate eligibility_chunks.embedding, then build the pgvector HNSW index.

Resumable: only rows where embedding IS NULL are processed, so an interrupted
run picks up where it stopped and re-running after a new ingest embeds only the
new criteria.

The HNSW index is created *after* the vectors exist. Building it on an empty or
partially filled table is wasted work -- pgvector builds far faster over a
populated table, and an index maintained during bulk insert slows every write.

    python -m trialsage.embed.build_index
    python -m trialsage.embed.build_index --index-only
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Tuple

from pgvector.psycopg import register_vector

from ..config import settings
from ..db import connect
from .model import describe, embed_documents

FETCH_SIZE = 2000


def count_pending(conn) -> Tuple[int, int]:
    (pending,) = conn.execute(
        "SELECT count(*) FROM eligibility_chunks WHERE embedding IS NULL").fetchone()
    (total,) = conn.execute("SELECT count(*) FROM eligibility_chunks").fetchone()
    return pending, total


def embed_pending(batch_size: Optional[int] = None, limit: Optional[int] = None) -> int:
    """Embed every chunk that does not yet have a vector. Returns the count."""
    cfg = settings()["embedding"]
    batch_size = batch_size or cfg["batch_size"]

    with connect() as conn:
        register_vector(conn)
        pending, total = count_pending(conn)
        if not pending:
            print(f"  nothing to embed ({total:,} chunks already have vectors)")
            return 0

        target = min(pending, limit) if limit else pending
        print(f"  {pending:,} of {total:,} chunks need embedding"
              f"{f' (limiting to {target:,})' if limit else ''}")
        print(f"  model: {describe()}")

        done = 0
        started = time.perf_counter()

        # Cursor-based paging, not a plain `WHERE embedding IS NULL LIMIT n`.
        #
        # The naive form re-scans from the start of the primary key on every
        # batch and filters out everything already embedded, so batch N pays
        # for all N-1 batches before it -- quadratic overall. Measured on this
        # corpus it had already degraded to "Rows Removed by Filter: 61708"
        # after only 6% of the work, and would have ended up scanning ~960k
        # rows per batch.
        #
        # Because we walk chunk_id in ascending order and only ever fill in
        # embeddings, everything at or below the cursor is already done, so
        # advancing past it is safe. A fresh run starts at 0 and still finds
        # the first unembedded row, which keeps this resumable.
        cursor = 0
        while done < target:
            rows: List[Tuple[int, str]] = conn.execute(
                "SELECT chunk_id, criterion_text FROM eligibility_chunks"
                " WHERE embedding IS NULL AND chunk_id > %s"
                " ORDER BY chunk_id LIMIT %s",
                (cursor, min(FETCH_SIZE, target - done)),
            ).fetchall()
            if not rows:
                break
            cursor = rows[-1][0]

            ids = [r[0] for r in rows]
            vectors = embed_documents([r[1] for r in rows], batch_size=batch_size)

            # One bulk UPDATE per batch rather than executemany.
            #
            # executemany issues a separate UPDATE per row. Measured on a
            # 2000-row batch of real criteria: the round trips cost ~8.2s
            # against ~10.7s of actual embedding, so the database was roughly
            # 43% of wall time and the pipeline ran at ~106 chunks/s. The bulk
            # form does the same work in 0.33s.
            #
            # Embedding is now the bottleneck at ~188 chunks/s (97% of the
            # time), which is where it should be -- there is no further win
            # available on the database side.
            #
            # Vectors go over as text and are cast server-side: psycopg adapts
            # a single numpy array to `vector`, but not an array *of* vectors,
            # which is what unnest needs here.
            literals = ["[" + ",".join(f"{x:.6g}" for x in vec) + "]" for vec in vectors]
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE eligibility_chunks AS e"
                    " SET embedding = v.emb::vector"
                    " FROM (SELECT unnest(%s::bigint[]) AS id,"
                    "              unnest(%s::text[])  AS emb) AS v"
                    " WHERE e.chunk_id = v.id",
                    (ids, literals),
                )
            conn.commit()

            done += len(rows)
            rate = done / max(time.perf_counter() - started, 1e-6)
            remaining = (target - done) / rate if rate else 0
            print(f"  embedded {done:,}/{target:,}  ({rate:.0f}/s, "
                  f"~{remaining / 60:.1f} min left)", end="\r", flush=True)

        elapsed = time.perf_counter() - started
        print(f"  embedded {done:,} chunks in {elapsed / 60:.1f} min "
              f"({done / max(elapsed, 1e-6):.0f}/s)                    ")
        return done


def build_hnsw(conn) -> None:
    """Create the approximate-nearest-neighbour index.

    HNSW rather than IVFFlat: better recall at the same speed, and it does not
    need a training step or a row-count-dependent `lists` parameter that goes
    stale as the corpus grows.

    Cosine distance (`vector_cosine_ops`) matches how the vectors were
    normalised at write time. Using a different operator class here than the
    one the search query uses would silently disable the index.
    """
    (pending,) = conn.execute(
        "SELECT count(*) FROM eligibility_chunks WHERE embedding IS NULL").fetchone()
    if pending:
        print(f"  WARNING: {pending:,} chunks still have no embedding; "
              "they will be invisible to search")

    print("  building HNSW index (this takes a few minutes on a large corpus)...")
    started = time.perf_counter()
    conn.execute("SET maintenance_work_mem = '512MB'")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS elig_chunks_embedding_hnsw"
        " ON eligibility_chunks USING hnsw (embedding vector_cosine_ops)"
        " WITH (m = 16, ef_construction = 64)"
    )
    conn.commit()
    print(f"  index built in {time.perf_counter() - started:.1f}s")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Embed eligibility criteria")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="embed at most N chunks (for a quick smoke test)")
    parser.add_argument("--index-only", action="store_true",
                        help="skip embedding, just build the HNSW index")
    parser.add_argument("--no-index", action="store_true",
                        help="embed but do not build the index yet")
    args = parser.parse_args(argv)

    if not args.index_only:
        print("1/2 embedding")
        embed_pending(batch_size=args.batch_size, limit=args.limit)

    if not args.no_index:
        print("2/2 index")
        with connect(autocommit=True) as conn:
            register_vector(conn)
            build_hnsw(conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
