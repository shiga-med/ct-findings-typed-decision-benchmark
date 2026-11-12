"""Synthetic CT-RATE / CT-RATE-JPN / RR-Findings files with the real layout (no real data)."""

import csv
import random
import shutil
from pathlib import Path

import pytest
import yaml

from ctfindings import data

ROOT = Path(__file__).resolve().parents[1]
LABELS = [en for en, _ in yaml.safe_load((ROOT / "configs/labels.yaml").read_text())["labels"]]


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def make_data(folder):
    rng = random.Random(0)
    label = lambda: [rng.randint(0, 1) for _ in LABELS]  # noqa: E731
    train, test = [], []
    for patient in range(1, 21):
        for scan in "ab":
            train.append((f"train_{patient}_{scan}_1.nii.gz", f"finding {patient}{scan}",
                          f"impression {patient}{scan}", label()))
    for patient in range(1, 7):
        test.append((f"valid_{patient}_a_1.nii.gz", f"test finding {patient}", "imp", label()))
    train[0] = (train[0][0], "test finding 1", "other", train[0][3])  # leak: excluded
    root = Path(folder)
    for split, rows, report, labels_file, ja in (
            ("train", train, "train_reports.csv", "train_predicted_labels.csv",
             "train_reports_ja.csv"),
            ("valid", test, "validation_reports.csv", "valid_predicted_labels.csv",
             "valid_reports_ja.csv")):
        write_csv(root / "ct-rate/dataset/radiology_text_reports" / report,
                  ["VolumeName", "Findings_EN", "Impressions_EN"], [r[:3] for r in rows])
        write_csv(root / "ct-rate/dataset/multi_abnormality_labels" / labels_file,
                  ["VolumeName", *LABELS], [[r[0], *r[3]] for r in rows])
        suffixes = [""] if split == "train" else ["_MT", "_Revised", "_Checked"]
        header = ["VolumeName", "Findings_EN", "Impressions_EN", *LABELS]
        for s in suffixes:
            header += ["Findings_JPN" + s, "Impressions_JPN" + s]
        ja_rows = []
        for i, r in enumerate(rows):
            text = [] if (split == "train" and i == 1) else [f"所見{r[0]}", "結論"]  # 1 missing
            ja_rows.append([*r[:3], *r[3], *(text or ["", ""]) * len(suffixes)])
        write_csv(root / "ct-rate-jpn" / ja, header, ja_rows)
    write_csv(root / "rr-findings/RR-Findings.csv", ["file_name", "report", *LABELS],
              [[f"{i}.txt", f"レポート{i}", *label()] for i in range(5)])


@pytest.fixture
def project(tmp_path):
    """Prepared synthetic project: (config folder, work folder)."""
    make_data(tmp_path / "data")
    shutil.copytree(ROOT / "configs", tmp_path / "repo/configs")
    shutil.copytree(ROOT / "prompts", tmp_path / "repo/prompts")
    config = tmp_path / "repo/configs"
    settings = yaml.safe_load((config / "data.yaml").read_text())
    for name, source in settings["sources"].items():
        source["files"] = {f: data.sha256(tmp_path / "data" / name / f) for f in source["files"]}
    settings["audit"] = {"train_pool_cases": 38, "test_cases": 6, "paired_test_cases": 6,
                         "excluded_missing": 1, "excluded_leak": 1, "external_reports": 5}
    settings["split"].update({"development_fraction": 0.25, "dev450_size": 4,
                              "few_shot_count": 2})
    del settings["split"]["few_shot_ids"]
    (config / "data.yaml").write_text(yaml.safe_dump(settings))
    data.prepare(tmp_path / "data", tmp_path / "work", config)
    return config, tmp_path / "work"
