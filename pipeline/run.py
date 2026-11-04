"""Stage runner (implementation.md §1.8).

    python -m pipeline.run --stage load
    python -m pipeline.run --stage all
    python -m pipeline.run --stage all --rebuild

`load` hits the network; every other stage reads from disk only.
"""

import argparse
import sys
import time

from pipeline.lib.config import load_config
from pipeline.lib.io import write_json
from pipeline.stages import chunker, embedder, indexer, loader

STAGES = {"load": loader, "chunk": chunker, "embed": embedder, "index": indexer}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.run")
    ap.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    ap.add_argument("--force", action="store_true",
                    help="re-fetch / re-process even when unchanged")
    ap.add_argument("--rebuild", action="store_true",
                    help="drop and recreate the vector collection")
    args = ap.parse_args(argv)

    cfg = load_config()
    todo = list(STAGES) if args.stage == "all" else [args.stage]

    results = {}
    for name in todo:
        started = time.perf_counter()
        try:
            results[name] = STAGES[name].run(
                cfg, force=args.force, rebuild=args.rebuild)
        except NotImplementedError as exc:
            print(f"[{name}] not implemented yet: {exc}")
            results[name] = {"failed": 1, "message": str(exc)}
        results[name]["ms"] = int((time.perf_counter() - started) * 1000)
        print(f"[{name}] {results[name]}")

    write_json("artifacts/runs/last_run.json", results)
    return 0 if all(not r.get("failed") for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
