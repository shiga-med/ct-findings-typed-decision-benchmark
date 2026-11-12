"""Prediction files: one JSON line per case, resumable, with a dry run."""

import json
import time
from pathlib import Path

from . import predictors as P
from .data import DataError, cohort, few_shot, load_labels, load_yaml, read_jsonl


def prediction_path(work, condition, language, text, set_name, run):
    """work/predictions/<condition>.<language>.<text>.<set>.<run>.jsonl (run: run1 or seed42)."""
    return Path(work) / "predictions" / f"{condition}.{language}.{text}.{set_name}.{run}.jsonl"


def parse_name(path):
    condition, language, text, set_name, run = Path(path).name.removesuffix(".jsonl").split(".")
    return {"condition": condition, "language": language, "text": text, "set": set_name,
            "run": run}


def make_predictor(name, condition, config, endpoint=None, seed_salt=None):
    family = condition["family"]
    if family == "openai":
        return P.OpenAIGeneration(condition)
    if family == "jev":
        return P.Jev(condition)
    if family == "decisions":
        return P.Decisions(condition)
    runtime = load_yaml(Path(config) / "conditions.yaml")["local_runtime"]
    if not endpoint:
        raise PermissionError(f"{name} needs --endpoint (URL of the vLLM server)")
    if family == "local":
        return P.LocalGeneration(condition, endpoint, runtime, seed_salt)
    if family == "typed":
        return P.TypedQwen(condition, endpoint, runtime)
    raise DataError(f"{name} is an encoder; use `ctfindings train`")


def predict(work, config, name, language, text, set_name, run, *, endpoint=None,
            seed_salt=None, dry_run=False, predictor=None):
    conditions = load_yaml(Path(config) / "conditions.yaml")["conditions"]
    if name not in conditions or language not in conditions[name]["languages"]:
        raise DataError(f"Unknown condition/language: {name} {language}")
    condition = conditions[name]
    if (text == "external") != (set_name == "external") or (
            text != "external" and (text == "en") != (language == "en")):
        raise DataError("Text does not match the language or set")
    labels, ja_names = load_labels(config)
    prompt = json.loads((Path(config).parent / "prompts/definition.json").read_text("utf-8"))
    shots = few_shot(work) if condition.get("few_shot") and set_name != "external" else []
    cases = cohort(work, set_name)
    path = prediction_path(work, name, language, text, set_name, f"run{run}")
    done = {r["case_id"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [c for c in cases if c["case_id"] not in done]
    if predictor is None:
        if not dry_run and condition["family"] in ("openai", "jev", "decisions"):
            P.require_env({"jev": "TYPESAFE_API_KEY"}.get(condition["family"], "OPENAI_API_KEY"))
        predictor = make_predictor(name, condition, config, endpoint, seed_salt)
    if dry_run:
        first = P.build_requests(predictor, todo[0], text, language, labels, ja_names, prompt,
                                 shots) if todo else None
        per_case = len(labels) if condition["family"] == "typed" else 1
        return {"output": str(path), "cases": len(cases), "already_done": len(done),
                "requests_to_send": len(todo) * per_case, "first_request": first}
    path.parent.mkdir(parents=True, exist_ok=True)
    for case in todo:
        body = P.build_requests(predictor, case, text, language, labels, ja_names, prompt, shots)
        row, started, attempts = None, time.monotonic(), 0
        while row is None and attempts < 2:  # one retry for transient errors
            attempts += 1
            try:
                row = predictor(body, labels)
            except (PermissionError, ImportError):
                raise
            except Exception as exc:  # noqa: BLE001 - network/server errors
                error = type(exc).__name__
        if row is None:
            row = {"status": "failed", "predictions": [None] * len(labels), "probabilities": None,
                   "requests": 2, "input_tokens": None, "output_tokens": None, "model": None,
                   "error": error}
        row = {"case_id": case["case_id"], **row, "attempts": attempts,
               "seconds": time.monotonic() - started}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = read_jsonl(path)
    return {"output": str(path), "cases": len(rows),
            "status": {s: sum(r["status"] == s for r in rows) for s in
                       ("success", "invalid", "failed")}}
