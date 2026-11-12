"""Data download, checks, preparation and splits (paper sections 2.1-2.2)."""

import csv
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

VOLUME = re.compile(r"^(train|valid)_(\d+)_([a-z]+)_(\d+)\.nii\.gz$")
TEXTS = ("en", "ja_MT", "ja_Revised", "ja_Checked")
SETS = ("dev450", "test150", "external")


class DataError(Exception):
    """Input or check mismatch (exit code 1)."""


# ---------------------------------------------------------------- small helpers
def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_labels(config):
    rows = load_yaml(Path(config) / "labels.yaml")["labels"]
    return [en for en, _ in rows], {en: ja for en, ja in rows}


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def ids_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


# ---------------------------------------------------------------- download and checks
def download(data_dir, sources):
    """Fetch missing files of the pinned revisions from Hugging Face."""
    import httpx

    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for name, source in sources.items():
        for file in source["files"]:
            path = Path(data_dir) / name / file
            if path.exists():
                continue
            url = (f"https://huggingface.co/datasets/{source['repo']}/resolve/"
                   f"{source['revision']}/{file}")
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(path.suffix + ".part")
            with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                              timeout=120) as response:
                if response.status_code in (401, 403):
                    raise DataError(f"{source['repo']}: accept the dataset terms on Hugging Face "
                                    "and set HF_TOKEN")
                response.raise_for_status()
                with part.open("wb") as fh:
                    for chunk in response.iter_bytes():
                        fh.write(chunk)
            part.rename(path)


def verify(data_dir, sources):
    for name, source in sources.items():
        for file, expected in source["files"].items():
            path = Path(data_dir) / name / file
            if not path.exists():
                raise DataError(f"Missing file: {path}")
            if sha256(path) != expected:
                raise DataError(f"SHA-256 differs from revision {source['revision']}: {path}")


# ---------------------------------------------------------------- CT-RATE / CT-RATE-JPN
def volume_ids(name):
    match = VOLUME.fullmatch(name)
    if not match:
        raise DataError(f"Invalid VolumeName: {name}")
    split, patient, scan, _ = match.groups()
    return f"{split}_{patient}", f"{split}_{patient}_{scan}"


def read_csv(path, required):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not set(required) <= set(reader.fieldnames or []):
            raise DataError(f"Missing columns: {path}")
        rows, seen = [], set()
        for row in reader:
            volume_ids(row["VolumeName"])
            if row["VolumeName"] in seen:
                raise DataError(f"Duplicate VolumeName: {row['VolumeName']}")
            seen.add(row["VolumeName"])
            rows.append(row)
    return rows


def binary(row, labels):
    values = [row[name] for name in labels]
    if any(v not in ("0", "1") for v in values):
        raise DataError("Non-binary label")
    return [int(v) for v in values]


def load_ct(data_dir, labels):
    """Read and cross-check the English reports, labels and Japanese translations."""
    root = Path(data_dir)
    files = {"train": ("train_reports.csv", "train_predicted_labels.csv", "train_reports_ja.csv"),
             "test": ("validation_reports.csv", "valid_predicted_labels.csv",
                      "valid_reports_ja.csv")}
    en, lab, ja = {}, {}, {}
    for split, (report, label, jp) in files.items():
        en[split] = read_csv(root / "ct-rate/dataset/radiology_text_reports" / report,
                             ["VolumeName", "Findings_EN", "Impressions_EN"])
        lab[split] = {r["VolumeName"]: binary(r, labels) for r in read_csv(
            root / "ct-rate/dataset/multi_abnormality_labels" / label, ["VolumeName", *labels])}
        ja[split] = read_csv(root / "ct-rate-jpn" / jp,
                             ["VolumeName", "Findings_EN", "Impressions_EN", *labels])
        if set(lab[split]) != {r["VolumeName"] for r in en[split]}:
            raise DataError("Report and label IDs differ")
        lookup = {r["VolumeName"]: r for r in en[split]}
        for row in ja[split]:
            source = lookup[row["VolumeName"]]
            if any(source[k] != row[k] for k in ("Findings_EN", "Impressions_EN")):
                raise DataError("English and Japanese source texts differ")
            if binary(row, labels) != lab[split][row["VolumeName"]]:
                raise DataError("English and Japanese labels differ")
    return en, lab, ja


def build_cases(en, lab, ja):
    """Training pool (leak-free) and paired test cases; exact text, no normalisation."""
    groups = {}
    for split in ("train", "test"):
        grouped = defaultdict(list)
        for row in en[split]:
            grouped[row["Findings_EN"], row["Impressions_EN"]].append(row)
        for members in grouped.values():
            if len({tuple(lab[split][r["VolumeName"]]) for r in members}) != 1:
                raise DataError("Identical report with conflicting labels")
        groups[split] = grouped
    test_findings = {r["Findings_EN"] for r in en["test"]}

    def case(row, jp, split):
        members = groups[split][row["Findings_EN"], row["Impressions_EN"]]
        texts = {"en": {"findings": row["Findings_EN"], "impressions": row["Impressions_EN"]}}
        for key, suffix in ((("ja_MT", ""),) if split == "train" else
                            (("ja_MT", "_MT"), ("ja_Revised", "_Revised"),
                             ("ja_Checked", "_Checked"))):
            texts[key] = {"findings": jp["Findings_JPN" + suffix],
                          "impressions": jp["Impressions_JPN" + suffix]}
        return {"case_id": row["VolumeName"], "patient_id": volume_ids(row["VolumeName"])[0],
                "source_patient_ids": sorted({volume_ids(r["VolumeName"])[0] for r in members}),
                "texts": texts, "labels": lab[split][row["VolumeName"]]}

    counts = defaultdict(int)
    train = []
    en_train = {r["VolumeName"]: r for r in en["train"]}
    for jp in ja["train"]:
        row = en_train[jp["VolumeName"]]
        if any(not row[k].strip() for k in ("Findings_EN", "Impressions_EN")) or any(
                not jp[k].strip() for k in ("Findings_JPN", "Impressions_JPN")):
            counts["excluded_missing"] += 1
        elif row["Findings_EN"] in test_findings:
            counts["excluded_leak"] += 1
        else:
            train.append(case(row, jp, "train"))
    jp_test = {r["VolumeName"]: r for r in ja["test"]}
    test, test_cases = [], 0
    for members in groups["test"].values():
        test_cases += 1
        # Keep the ID that has a Japanese translation when an exact report occurs twice.
        candidates = [r for r in members if r["VolumeName"] in jp_test]
        if len(candidates) > 1:
            raise DataError("Several Japanese rows for one English report")
        if candidates:
            test.append(case(candidates[0], jp_test[candidates[0]["VolumeName"]], "test"))
    train.sort(key=lambda r: r["case_id"])
    test.sort(key=lambda r: r["case_id"])
    counts.update({
        "train_pool_cases": len(train),
        "train_representative_patients": len({r["patient_id"] for r in train}),
        "test_cases": test_cases,
        "test_source_patients": len({volume_ids(r["VolumeName"])[0] for r in en["test"]}),
        "paired_test_cases": len(test)})
    return train, test, dict(counts)


def split_development(cases, seed, fraction):
    """Connected components of patient IDs and exact texts; whole components go to development."""
    ordered = sorted(cases, key=lambda r: r["case_id"])
    parent = {r["case_id"]: r["case_id"] for r in ordered}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    first = {}
    for row in ordered:
        keys = [("patient", p) for p in row["source_patient_ids"]]
        for lang in ("en", "ja_MT"):
            text = row["texts"][lang]
            keys += [(lang + "_findings", text["findings"]),
                     (lang + "_text", (text["findings"], text["impressions"]))]
        for key in keys:
            if key in first:
                a, b = find(row["case_id"]), find(first[key])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first[key] = row["case_id"]
    groups = defaultdict(list)
    for row in ordered:
        groups[find(row["case_id"])].append(row["case_id"])
    components = sorted(groups)
    random.Random(seed).shuffle(components)
    target, size, development = round(len(ordered) * fraction), 0, set()
    for component in components:
        if abs(size + len(groups[component]) - target) < abs(size - target):
            development.update(groups[component])
            size += len(groups[component])
    return development


def prepare(data_dir, work, config):
    """Build work/data/{cases.jsonl, external.jsonl, manifest.json}."""
    settings = load_yaml(Path(config) / "data.yaml")
    labels, _ = load_labels(config)
    verify(data_dir, settings["sources"])
    train, test, counts = build_cases(*load_ct(data_dir, labels))
    rule = settings["split"]
    development = split_development(train, rule["seed"], rule["development_fraction"])
    dev_ids = sorted(c["case_id"] for c in train if c["case_id"] in development)
    picked = random.Random(rule["dev450_seed"]).sample(range(len(dev_ids)), rule["dev450_size"])
    dev450 = {dev_ids[i] for i in picked}
    train_ids = sorted(c["case_id"] for c in train if c["case_id"] not in development)
    few_shot = random.Random(rule["few_shot_seed"]).sample(train_ids, rule["few_shot_count"])
    if "few_shot_ids" in rule and few_shot != rule["few_shot_ids"]:
        raise DataError("Few-shot examples differ from the expected case IDs")
    external = read_external(Path(data_dir) / "rr-findings/RR-Findings.csv", labels)
    counts.update({"train_cases": len(train_ids), "development_cases": len(dev_ids),
                   "external_reports": len(external)})
    for key, expected in settings["audit"].items():
        if counts.get(key) != expected:
            raise DataError(f"Audited count differs: {key}={counts.get(key)}, expected {expected}")
    cases = []
    for c in train:
        split = "development" if c["case_id"] in development else "train"
        cases.append({**c, "split": split, "sets": ["dev450"] if c["case_id"] in dev450 else []})
    cases += [{**c, "split": "test", "sets": ["test150"]} for c in test]
    data = Path(work) / "data"
    write_jsonl(data / "cases.jsonl", cases)
    write_jsonl(data / "external.jsonl", external)
    manifest = {
        "revisions": {k: v["revision"] for k, v in settings["sources"].items()},
        "counts": counts, "few_shot": few_shot,
        "id_hashes": {"train": ids_hash(train_ids), "development": ids_hash(dev_ids),
                      "dev450": ids_hash(dev450), "test150": ids_hash(c["case_id"] for c in test),
                      "external": ids_hash(c["case_id"] for c in external)}}
    write_json(data / "manifest.json", manifest)
    return manifest


def read_external(path, labels):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        if next(reader) != ["file_name", "report", *labels]:
            raise DataError("Unexpected RR-Findings columns")
        rows = []
        for file_name, report, *values in reader:
            if any(v not in ("0", "1") for v in values) or not report.strip():
                raise DataError(f"Invalid RR-Findings row: {file_name}")
            rows.append({"case_id": file_name.removesuffix(".txt"), "report": report,
                         "labels": [int(v) for v in values]})
    return rows


# ---------------------------------------------------------------- access for later steps
def load_cases(work):
    return read_jsonl(Path(work) / "data/cases.jsonl")


def cohort(work, name):
    """Cases of one evaluation set, sorted by case ID."""
    if name == "external":
        return sorted(read_jsonl(Path(work) / "data/external.jsonl"), key=lambda r: r["case_id"])
    if name not in SETS:
        raise DataError(f"Unknown set: {name}")
    return [c for c in load_cases(work) if name in c["sets"]]


def few_shot(work):
    ids = json.loads((Path(work) / "data/manifest.json").read_text())["few_shot"]
    lookup = {c["case_id"]: c for c in load_cases(work)}
    return [lookup[i] for i in ids]


def report_text(case, text, headings):
    """Model input: Findings + Impressions with language headings, or the external report."""
    if text == "external":
        return case["report"]
    findings, impressions = headings["en" if text == "en" else "ja"]
    t = case["texts"][text]
    return f"{findings}:\n{t['findings']}\n\n{impressions}:\n{t['impressions']}"
