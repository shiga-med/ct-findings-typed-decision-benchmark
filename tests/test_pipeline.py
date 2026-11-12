"""Requests, resume, dry run and the evaluation pipeline with fake model clients."""

import json

import numpy as np
import pytest
import yaml

from ctfindings import cli, data, predictors, runner, tables
from ctfindings.thresholds import optimize


class FakeJev(predictors.Jev):
    """Jev request builder with a fake client call (no SDK needed)."""
    def __init__(self):
        super().__init__({"model": "jev-1.14.0"})
        self.calls = []

    def __call__(self, body, labels):
        self.calls.append(body)
        return predictors.from_probabilities([0.7] * len(labels), {"input_tokens": 10},
                                             body["model"])


def test_requests_share_the_question_text(project):
    config, work = project
    labels, ja = data.load_labels(config)
    prompt = json.loads((config.parent / "prompts/definition.json").read_text())
    case = data.cohort(work, "test150")[0]
    jev = predictors.Jev({"model": "jev-1.14.0"})
    dec = predictors.Decisions({"model": "gpt-6-luna"})
    a = predictors.build_requests(jev, case, "ja_Checked", "ja", labels, ja, prompt, [])
    b = predictors.build_requests(dec, case, "ja_Checked", "ja", labels, ja, prompt, [])
    assert list(a["questions"].values()) == [q["instructions"] for q in b["questions"]]
    assert len(b["questions"]) == 18 and a["state"].startswith("所見:\n")
    assert "肺結節（Lung nodule）" in a["questions"]["Lung nodule"]


def test_generation_requests(project):
    config, work = project
    labels, ja = data.load_labels(config)
    prompt = json.loads((config.parent / "prompts/definition.json").read_text())
    conditions = yaml.safe_load((config / "conditions.yaml").read_text())
    case, shots = data.cohort(work, "test150")[0], data.few_shot(work)
    g = predictors.OpenAIGeneration(conditions["conditions"]["G1"])
    body = predictors.build_requests(g, case, "en", "en", labels, ja, prompt, [])
    assert body["temperature"] == 0 and body["reasoning"] == {"effort": "none"}
    local = predictors.LocalGeneration(conditions["conditions"]["L8"], "http://x",
                                       conditions["local_runtime"], seed_salt="r1")
    body = predictors.build_requests(local, case, "en", "en", labels, ja, prompt, shots)
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "medium"}
    assert len(body["messages"]) == 2 + 2 * len(shots) and isinstance(body["seed"], int)
    assert predictors.parse_binary(json.dumps(dict.fromkeys(labels, 1)), labels) == [1] * 18
    assert predictors.parse_binary('{"x": 1}', labels) is None


def test_typed_probability():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            return {"Yes": [1], " Yes": [2], "yes": [3], " yes": [4], "No": [5], " No": [6],
                    "no": [7], " no": [8, 9]}[text]
    t = predictors.TypedQwen({"model": "q"}, "http://x", {}, tokenizer=Tokenizer())
    assert t.candidates == {"yes": [1, 2, 3, 4], "no": [5, 6, 7]}
    lp = {f"token_id:{i}": np.log(0.1) for i in range(1, 8)}
    assert t.probability(lp) == pytest.approx(4 / 7)


def test_dry_run_sends_nothing_and_resume(project):
    config, work = project
    fake = jev = FakeJev()
    plan = runner.predict(work, config, "J1", "en", "en", "test150", 1, dry_run=True,
                          predictor=jev)
    assert plan["requests_to_send"] == 6 and not fake.calls
    path = runner.prediction_path(work, "J1", "en", "en", "test150", "run1")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"case_id": data.cohort(work, "test150")[0]["case_id"],
                                "status": "success", "predictions": [0] * 18}) + "\n")
    out = runner.predict(work, config, "J1", "en", "en", "test150", 1, predictor=jev)
    assert len(fake.calls) == 5 and out["cases"] == 6


def test_missing_credentials_exit_code(project, monkeypatch):
    config, work = project
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.chdir(work)
    code = cli.main(["--work", str(work), "--config", str(config), "predict", "--condition",
                     "J1", "--language", "en", "--text", "en", "--set", "test150"])
    assert code == 2


def test_failed_request_is_retried_once(project):
    config, work = project

    class Broken:
        calls = 0

        def __call__(self, body, labels):
            Broken.calls += 1
            raise ConnectionError

        def request(self, state, questions):
            return {}
    broken = Broken()
    runner.predict(work, config, "J1", "en", "en", "test150", 1, predictor=broken)
    rows = data.read_jsonl(runner.prediction_path(work, "J1", "en", "en", "test150", "run1"))
    assert Broken.calls == 12 and {r["status"] for r in rows} == {"failed"}
    assert all(r["attempts"] == 2 for r in rows)


def write_population(work, condition, language, text, set_name, run, rng, probabilities=True):
    rows = []
    for case in data.cohort(work, set_name):
        p = rng.random(18).tolist()
        rows.append({"case_id": case["case_id"], "status": "success",
                     "probabilities": p if probabilities else None,
                     "predictions": [int(v >= 0.5) for v in p], "requests": 1,
                     "input_tokens": 100, "output_tokens": 5, "seconds": 0.1})
    data.write_jsonl(runner.prediction_path(work, condition, language, text, set_name, run), rows)


def test_evaluate_and_tables(project):
    config, work = project
    rng = np.random.default_rng(1)
    analysis = yaml.safe_load((config / "analysis.yaml").read_text())
    analysis["comparisons"] = [["en", "T1", "G1"]]
    analysis["bootstrap"]["repetitions"] = 50
    (config / "analysis.yaml").write_text(yaml.safe_dump(analysis))
    write_population(work, "T1", "en", "en", "dev450", "run1", rng)
    tables.thresholds(work, "T1", "en")
    for run in ("run1", "run2"):
        write_population(work, "T1", "en", "en", "test150", run, rng)
        write_population(work, "G1", "en", "en", "test150", run, rng, probabilities=False)
    write_population(work, "J1", "ja", "external", "external", "run1", rng)
    tables.evaluate(work, config)
    tables.tables(work, config)
    res = work / "results"
    first = {p.name: p.read_bytes() for p in res.glob("*.csv")}
    tables.tables(work, config)
    assert first == {p.name: p.read_bytes() for p in res.glob("*.csv")}  # byte-identical
    metrics = tables.read_csv(res / "metrics.csv")
    assert {r["threshold"] for r in metrics if r["condition"] == "G1"} == {"development"}
    t1 = json.loads((work / "thresholds/T1.en.json").read_text())["thresholds"]
    dev = data.cohort(work, "dev450")
    rows = data.read_jsonl(runner.prediction_path(work, "T1", "en", "en", "dev450", "run1"))
    assert t1 == optimize([c["labels"] for c in dev], [r["probabilities"] for r in rows])[0]
    assert len(tables.read_csv(res / "table12_comparisons.csv")) == 1
    assert len(tables.read_csv(res / "table14_threshold_refit.csv")) == 1  # J1 uses frozen values
