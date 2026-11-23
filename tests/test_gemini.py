"""Gemini protocol and v2 aggregation, without API requests."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ctfindings import data, predictors, runner, tables


class RawResponse:
    def __init__(self, text):
        self.content = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def text(self):
        return self.content


def adapter(raw):
    def create(**body):
        assert body["model"] == "gemini-decision-1"
        return RawResponse(raw if isinstance(raw, str) else json.dumps(raw))
    client = SimpleNamespace(decisions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create)))
    return predictors.GeminiDecisions({"model": "gemini-decision-1"}, client=client)


def response():
    return {"answers": {f"finding_{i}": {"type": "noul", "noul": .7} for i in range(18)},
            "usage": {"total_input_tokens": 10, "total_output_tokens": 0}}


def test_named_questions_and_shared_clinical_wording(project):
    config, work = project
    labels, ja = data.load_labels(config)
    prompt = json.loads((config.parent / "prompts/definition.json").read_text())
    case = data.cohort(work, "test150")[0]
    gemini = adapter(response())
    body = predictors.build_requests(gemini, case, "ja_Checked", "ja", labels, ja, prompt, [])
    jev = predictors.build_requests(predictors.Jev({"model": "j"}), case, "ja_Checked", "ja",
                                    labels, ja, prompt, [])
    assert list(body["questions"]) == [f"finding_{i}" for i in range(18)]
    assert [q["instructions"] for q in body["questions"].values()] == list(
        jev["questions"].values())
    out = gemini(body, labels)
    assert out["status"] == "success" and out["probabilities"] == [.7] * 18
    assert out["model"] is None and out["response_model_version_status"] == "not_provided"
    assert out["input_tokens"] == 10 and out["output_tokens"] == 0


@pytest.mark.parametrize("value", [True, -1, 1.1, float("nan"), "0.7", None])
def test_strict_probabilities(value):
    raw = response()
    raw["answers"]["finding_0"]["noul"] = value
    out = adapter(raw)({"model": "gemini-decision-1"}, list(range(18)))
    assert out["status"] == "invalid" and out["probabilities"][0] is None
    assert out["probabilities"][1:] == [.7] * 17


@pytest.mark.parametrize("raw", [
    '{"answers":{},"answers":{}}', '[]', '{invalid',
    {**response(), "prompt_feedback": {"block_reason": "SAFETY"}},
    {"answers": {**response()["answers"], "unknown": {"type": "noul", "noul": .5}}},
])
def test_invalid_envelopes(raw):
    out = adapter(raw)({"model": "gemini-decision-1"}, list(range(18)))
    assert out["status"] == "invalid" and out["probabilities"] == [None] * 18


def test_model_version_and_usage_types():
    raw = {**response(), "model_version": "actual-response-version",
           "usage": {"total_input_tokens": True, "total_output_tokens": -1}}
    out = adapter(raw)({"model": "gemini-decision-1"}, list(range(18)))
    assert out["model"] == "actual-response-version"
    assert out["input_tokens"] is None and out["output_tokens"] is None


def test_gemini_dry_run_credentials_and_resume(project, monkeypatch):
    config, work = project
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    plan = runner.predict(work, config, "GD1", "en", "en", "test150", 1, dry_run=True)
    assert plan["requests_to_send"] == 6
    assert len(plan["first_request"]["questions"]) == 18
    with pytest.raises(PermissionError, match="GEMINI_API_KEY"):
        runner.predict(work, config, "GD1", "en", "en", "test150", 1)
    fake = adapter(response())
    runner.predict(work, config, "GD1", "en", "en", "test150", 1, predictor=fake)
    assert runner.predict(work, config, "GD1", "en", "en", "test150", 1,
                          dry_run=True)["requests_to_send"] == 0


def population(run, p=.7, text="en"):
    return {"condition": "GD1", "language": "en", "set": "test150", "text": text,
            "run": run, "p": np.full((2, 18), p), "t": np.full(18, .5)}


def test_no_external_refit_for_gemini():
    assert tables.refit([population("run1") | {"set": "external"}]) == ([], [])


def test_gemini_single_run_pipeline(project):
    config, work = project
    fake = adapter(response())
    runner.predict(work, config, "GD1", "en", "en", "dev450", 1, predictor=fake)
    tables.thresholds(work, "GD1", "en")
    runner.predict(work, config, "GD1", "en", "en", "test150", 1, predictor=fake)
    tables.evaluate(work, config)
    tables.tables(work, config)
    metrics = tables.read_csv(work / "results" / "metrics.csv")
    assert {r["run"] for r in metrics} == {"run1"}
    assert len(data.load_yaml(config / "analysis.yaml")["comparisons"]) == 15


def test_retry_preserves_unknown_usage(project):
    config, work = project
    fake = adapter(response())

    class Retry:
        calls = 0
        request = fake.request

        def __call__(self, body, labels):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError
            return fake(body, labels)

    runner.predict(work, config, "GD1", "en", "en", "test150", 1, predictor=Retry())
    rows = data.read_jsonl(runner.prediction_path(work, "GD1", "en", "en", "test150", "run1"))
    assert rows[0]["requests"] == 2 and rows[0]["usage_incomplete"]
    pop = population("run1") | {"usage": rows}
    efficiency = tables.efficiency([pop])[0]
    assert efficiency["requests"] == 7 and efficiency["input_tokens_per_report"] == ""
