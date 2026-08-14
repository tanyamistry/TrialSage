"""Ask a question end to end from the command line.

    python -m trialsage.ask "how many phase 3 diabetes trials started in 2024?"
    python -m trialsage.ask "..." --explain     # full routing + retrieval trace
    python -m trialsage.ask "..." --no-llm-router
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .pipeline import ask
from .synth.citations import append_warnings


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--explain", action="store_true",
                        help="show the routing decision and retrieval internals")
    parser.add_argument("-k", type=int, default=None, help="number of results to retrieve")
    parser.add_argument("--no-llm-router", action="store_true",
                        help="use the deterministic rule-based router only")
    parser.add_argument("--strip", action="store_true",
                        help="remove fabricated citations instead of flagging them")
    args = parser.parse_args(argv)

    result = ask(
        args.question,
        k=args.k,
        use_llm_router=not args.no_llm_router,
        citation_mode="strip" if args.strip else "flag",
    )

    if args.explain:
        print(result.explain())
    else:
        d = result.decision
        print(f"[route: {d.route} | confidence {d.confidence:.2f} | via {d.source}]")
        print(f"  why: {d.reasoning}\n")
        print(append_warnings(result.audit) if result.audit else result.text)

    return 0 if result.grounded else 1


if __name__ == "__main__":
    sys.exit(main())
