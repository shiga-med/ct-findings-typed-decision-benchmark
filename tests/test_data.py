import json

import pytest
import yaml

from ctfindings import data, encoder


def test_prepare_builds_sets(project):
    _, work = project
    cases = data.load_cases(work)
    assert {c["split"] for c in cases} == {"train", "development", "test"}
    assert len(data.cohort(work, "dev450")) == 4
    assert len(data.cohort(work, "test150")) == 6
    assert len(data.cohort(work, "external")) == 5
    manifest = json.loads((work / "data/manifest.json").read_text())
    assert len(manifest["few_shot"]) == 2
    assert all(c["split"] == "train" for c in data.few_shot(work))


def test_same_patient_stays_in_one_split(project):
    _, work = project
    split = {}
    for c in data.load_cases(work):
        if c["split"] != "test":
            split.setdefault(c["patient_id"], set()).add(c["split"])
    assert all(len(s) == 1 for s in split.values())


def test_prepare_is_deterministic(project, tmp_path):
    config, work = project
    first = (work / "data/cases.jsonl").read_bytes()
    data.prepare(tmp_path / "data", tmp_path / "again", config)
    assert (tmp_path / "again/data/cases.jsonl").read_bytes() == first


def test_changed_file_stops(project, tmp_path):
    config, _ = project
    with (tmp_path / "data/rr-findings/RR-Findings.csv").open("a") as fh:
        fh.write("\n")
    with pytest.raises(data.DataError, match="SHA-256"):
        data.prepare(tmp_path / "data", tmp_path / "w2", config)


def test_audit_mismatch_stops(project, tmp_path):
    config, _ = project
    settings = yaml.safe_load((config / "data.yaml").read_text())
    settings["audit"]["excluded_leak"] = 2
    (config / "data.yaml").write_text(yaml.safe_dump(settings))
    with pytest.raises(data.DataError, match="excluded_leak"):
        data.prepare(tmp_path / "data", tmp_path / "w2", config)


def test_fraction_subsets_are_nested_and_exact():
    labels = {f"c{i:03d}": [(i * k) % 3 == 0 for k in range(1, 19)] for i in range(200)}
    subsets = encoder.fraction_subsets(labels, [2, 10, 100], [42])
    half, tenth, hundredth = (set(subsets[42][d]) for d in (2, 10, 100))
    assert (len(half), len(tenth), len(hundredth)) == (100, 20, 2)
    assert hundredth <= tenth <= half
    assert encoder.fraction_subsets(labels, [2, 10, 100], [42]) == subsets
