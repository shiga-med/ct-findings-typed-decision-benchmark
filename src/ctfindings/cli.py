"""Command line: prepare, train, predict, thresholds, evaluate, tables.

Exit codes: 0 success, 1 input or check mismatch, 2 missing credentials or endpoint.
"""

import argparse
import json
import sys
from pathlib import Path

from .data import DataError

ROOT = Path(__file__).resolve().parents[2]


def load_env(path=Path(".env")):
    """Read KEY=VALUE lines from .env into the environment (existing values win)."""
    import os
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#") and value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ctfindings", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=Path("work"), help="output folder")
    parser.add_argument("--config", type=Path, default=ROOT / "configs")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="download (if missing), check and split the data")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--no-download", action="store_true")
    p = sub.add_parser("train", help="fine-tune an encoder (GPU)")
    p.add_argument("--condition", required=True, choices=["E1", "E2"])
    p.add_argument("--language", required=True, choices=["en", "ja"])
    p.add_argument("--fraction", type=int, choices=[2, 10, 100])
    p = sub.add_parser("predict", help="run one model condition on one case set")
    p.add_argument("--condition", required=True)
    p.add_argument("--language", required=True, choices=["en", "ja"])
    p.add_argument("--text", required=True,
                   choices=["en", "ja_MT", "ja_Revised", "ja_Checked", "external"])
    p.add_argument("--set", required=True, choices=["dev450", "test150", "external"])
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--endpoint", help="vLLM server URL for Qwen/MedGemma, e.g. http://localhost:8000")
    p.add_argument("--seed-salt", help="optional per-request sampling seed for local generation")
    p.add_argument("--dry-run", action="store_true", help="show the request count; send nothing")
    p = sub.add_parser("thresholds", help="development thresholds from dev450 probabilities")
    p.add_argument("--condition", required=True)
    p.add_argument("--language", required=True, choices=["en", "ja"])
    sub.add_parser("evaluate", help="metrics for every prediction file")
    sub.add_parser("tables", help="paper tables from the metrics")
    args = parser.parse_args(argv)
    load_env()
    try:
        out = run(args)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except PermissionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"error: {exc.name} is not installed; run `uv sync --group api` "
              "(API conditions) or `uv sync --group train` (encoders, T1)", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def run(args):
    if args.command == "prepare":
        from .data import download, load_yaml, prepare
        if not args.no_download:
            download(args.data, load_yaml(args.config / "data.yaml")["sources"])
        return prepare(args.data, args.work, args.config)
    if args.command == "train":
        from .encoder import train
        return train(args.work, args.config, args.condition, args.language, args.fraction)
    if args.command == "predict":
        from .runner import predict
        return predict(args.work, args.config, args.condition, args.language, args.text, args.set,
                       args.run, endpoint=args.endpoint, seed_salt=args.seed_salt,
                       dry_run=args.dry_run)
    from . import tables
    if args.command == "thresholds":
        return tables.thresholds(args.work, args.condition, args.language)
    if args.command == "evaluate":
        return tables.evaluate(args.work, args.config)
    return tables.tables(args.work, args.config)


if __name__ == "__main__":
    sys.exit(main())
