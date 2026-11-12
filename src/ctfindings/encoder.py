"""Encoder fine-tuning (paper section 2.3) and training-fraction subsets (section 2.7).

For each learning rate and seed: train up to 5 epochs; after each epoch, score the 450
development cases at 0.5 and keep the earliest best epoch. The learning rate with the
highest mean over the 3 seeds is chosen (ties: lower rate). Per seed, thresholds come from
the development probabilities of the chosen epoch. Requires the `train` group and a GPU.
"""

import contextlib
import json
import random
import shutil
from pathlib import Path

import numpy as np

from .data import cohort, load_cases, load_yaml, report_text, write_json, write_jsonl
from .thresholds import micro_f1, optimize

HEADINGS = {"en": ["Findings", "Impressions"], "ja": ["所見", "結論"]}


# ---------------------------------------------------------------- training fractions
def stratified_subset(ids, labels, keep, rng):
    """Iterative multi-label stratification (Sechidis et al., 2011), exact size.

    An all-negative indicator is added as a 19th label. Deterministic for a given rng.
    """
    ids = sorted(ids)
    vectors = {i: [*labels[i], int(not any(labels[i]))] for i in ids}
    width = len(vectors[ids[0]])
    totals = [sum(vectors[i][k] for i in ids) for k in range(width)]
    desired = [[t * r for t in totals] for r in (keep / len(ids), 1 - keep / len(ids))]
    size, remaining, kept = [keep, len(ids) - keep], set(ids), []
    while remaining:
        counts = [sum(vectors[i][k] for i in remaining) for k in range(width)]
        label = min((c, k) for k, c in enumerate(counts) if c)[1]
        members = sorted(i for i in remaining if vectors[i][label])
        rng.shuffle(members)
        for sample in members:
            best = max(desired[f][label] for f in (0, 1))
            folds = [f for f in (0, 1) if desired[f][label] == best]
            if len(folds) > 1:
                most = max(size[f] for f in folds)
                folds = [f for f in folds if size[f] == most]
            fold = folds[0] if len(folds) == 1 else rng.choice(folds)
            if fold == 0:
                kept.append(sample)
            size[fold] -= 1
            for k in range(width):
                desired[fold][k] -= vectors[sample][k]
            remaining.remove(sample)
    # Stratification leaves the size approximate: move single reports, each time the one
    # that keeps the kept label counts closest to their proportional target.
    kept, target = set(kept), np.asarray(totals, dtype=float) * keep / len(ids)
    order = list(ids)
    rng.shuffle(order)
    matrix = {i: np.asarray(vectors[i], dtype=float) for i in order}
    counts = sum((matrix[i] for i in kept), np.zeros(len(target)))
    while len(kept) != keep:
        deviation = counts - target
        if len(kept) > keep:
            pick = max((i for i in order if i in kept),
                       key=lambda i: 2 * deviation @ matrix[i] - matrix[i] @ matrix[i])
            kept.remove(pick)
            counts -= matrix[pick]
        else:
            pick = min((i for i in order if i not in kept),
                       key=lambda i: 2 * deviation @ matrix[i] + matrix[i] @ matrix[i])
            kept.add(pick)
            counts += matrix[pick]
    return sorted(kept)


def fraction_subsets(train_labels, denominators, seeds):
    """{seed: {denominator: ids}}, nested per seed: 1/100 within 1/10 within 1/2."""
    total, subsets = len(train_labels), {}
    for seed in seeds:
        parent, chain = sorted(train_labels), {}
        for d in denominators:
            rng = random.Random(f"training-fraction-{seed}-{d}")
            parent = stratified_subset(parent, train_labels, round(total / d), rng)
            chain[d] = parent
        subsets[seed] = chain
    return subsets


# ---------------------------------------------------------------- training
def _steps(model, batches, optimizer, device, bf16):
    """One optimiser step over accumulated batches, weighted by batch size."""
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = sum(len(b["labels"]) for b in batches)
    for batch in batches:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "case_id"}
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
        with ctx:
            loss = model(**inputs).loss * len(batch["labels"]) / total
        loss.backward()
    optimizer.step()


def _infer(model, batches, device, bf16):
    import torch
    model.eval()
    out = []
    with torch.inference_mode():
        for batch in batches:
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "case_id")}
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else \
                contextlib.nullcontext()
            with ctx:
                out += model(**inputs).logits.float().sigmoid().cpu().numpy().tolist()
    return np.asarray(out)


def _batches(tokenizer, rows, text, size, max_length):
    import torch
    for start in range(0, len(rows), size):
        chunk = rows[start:start + size]
        encoded = [tokenizer(report_text(r, text, HEADINGS), truncation=False) for r in chunk]
        if any(len(e["input_ids"]) > max_length for e in encoded):
            raise ValueError("Report longer than the encoder context; reports are not truncated")
        batch = tokenizer.pad(encoded, return_tensors="pt")
        batch["labels"] = torch.tensor([r["labels"] for r in chunk], dtype=torch.float32)
        batch["case_id"] = [r["case_id"] for r in chunk]
        yield batch


def train_one(condition, language, lr, seed, training, dev450, folder, settings):
    """Train one learning rate x seed; keep the earliest best epoch on dev450 at 0.5."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    for s in (random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all):
        s(seed)
    device, bf16 = "cuda", settings["bf16"]
    model_id, revision = condition["model"][language], condition["revision"][language]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, num_labels=18,
        problem_type="multi_label_classification").to(device)
    max_length = min(settings["max_length"], model.config.max_position_embeddings)
    text = "en" if language == "en" else "ja_MT"
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    y = np.asarray([r["labels"] for r in dev450])
    training, best, curve = list(training), {"micro_f1": -1.0}, []
    for epoch in range(1, settings["max_epochs"] + 1):
        random.Random(seed + epoch).shuffle(training)
        group = []
        for batch in _batches(tokenizer, training, text, settings["batch_size"], max_length):
            group.append(batch)
            if len(group) == settings["gradient_accumulation"]:
                _steps(model, group, optimizer, device, bf16)
                group = []
        if group:
            _steps(model, group, optimizer, device, bf16)
        p = _infer(model, _batches(tokenizer, dev450, text, 1, max_length), device, bf16)
        score = micro_f1(y, (p >= 0.5).astype(int))
        curve.append({"epoch": epoch, "dev450_micro_f1": score})
        if score > best["micro_f1"]:
            best = {"epoch": epoch, "micro_f1": score, "probabilities": p.tolist()}
            model.save_pretrained(folder / "checkpoint")
            tokenizer.save_pretrained(folder / "checkpoint")
    write_json(folder / "selection.json", {"learning_rate": lr, "seed": seed, "curve": curve,
                                           "model_revision": revision, **best})
    return best


def train(work, config, name, language, fraction=None):
    """Train all learning rates x seeds, choose, set thresholds and predict test sets."""
    conditions = load_yaml(Path(config) / "conditions.yaml")
    settings, condition = conditions["encoder_training"], conditions["conditions"][name]
    tag = name if fraction is None else f"{name}-frac{fraction}"
    cases = load_cases(work)
    dev450 = cohort(work, "dev450")
    pool = [c for c in cases if c["split"] == "train"]
    subsets = None
    if fraction is not None:
        analysis = load_yaml(Path(config) / "analysis.yaml")["training_fractions"]
        subsets = fraction_subsets({c["case_id"]: c["labels"] for c in pool},
                                   analysis["denominators"], analysis["seeds"])
    runs = {}
    for lr in settings["learning_rates"]:
        for seed in settings["seeds"]:
            folder = Path(work) / "models" / f"{tag}.{language}.lr{lr:g}.seed{seed}"
            if (folder / "selection.json").exists():
                runs[lr, seed] = json.loads((folder / "selection.json").read_text())
                continue
            keep = set(subsets[seed][fraction]) if subsets else None
            training = [c for c in pool if keep is None or c["case_id"] in keep]
            runs[lr, seed] = train_one(condition, language, lr, seed, training, dev450, folder,
                                       settings)
    seeds = settings["seeds"]
    chosen = max(settings["learning_rates"],
                 key=lambda lr: (np.mean([runs[lr, s]["micro_f1"] for s in seeds]), -lr))
    y = np.asarray([c["labels"] for c in dev450])
    texts = ["en"] if language == "en" else ["ja_MT", "ja_Revised", "ja_Checked"]
    for seed in seeds:
        folder = Path(work) / "models" / f"{tag}.{language}.lr{chosen:g}.seed{seed}"
        thresholds, dev_f1 = optimize(y, runs[chosen, seed]["probabilities"])
        write_json(Path(work) / "thresholds" / f"{tag}.{language}.seed{seed}.json",
                   {"thresholds": thresholds, "dev450_micro_f1": dev_f1, "learning_rate": chosen,
                    "epoch": runs[chosen, seed]["epoch"]})
        sets = [("test150", t) for t in texts]
        if language == "ja" and name in conditions["external"]:
            sets.append(("external", "external"))
        for set_name, text in sets:
            predict_set(work, folder / "checkpoint", tag, language, text, set_name, seed, settings)
    # Weights of learning rates that were not chosen are no longer needed.
    for lr in settings["learning_rates"]:
        for seed in seeds:
            folder = Path(work) / "models" / f"{tag}.{language}.lr{lr:g}.seed{seed}"
            if lr != chosen and (folder / "checkpoint").exists():
                shutil.rmtree(folder / "checkpoint")
    return {"condition": tag, "language": language, "learning_rate": chosen}


def predict_set(work, checkpoint, tag, language, text, set_name, seed, settings):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .runner import prediction_path
    rows = cohort(work, set_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to("cuda")
    max_length = min(settings["max_length"], model.config.max_position_embeddings)
    p = _infer(model, _batches(tokenizer, rows, text, 1, max_length), "cuda", settings["bf16"])
    write_jsonl(prediction_path(work, tag, language, text, set_name, f"seed{seed}"), [
        {"case_id": r["case_id"], "status": "success", "probabilities": q,
         "predictions": [int(v >= 0.5) for v in q], "requests": 0}
        for r, q in zip(rows, p.tolist(), strict=True)])
