# ct-findings-typed-decision-benchmark

Code for the study comparing three ways to classify 18 findings in chest CT reports:
fine-tuned encoders, generative LLMs, and typed decisions (per-finding yes/no questions that
return a probability). Internal test: 150 paired English/Japanese CT-RATE reports.
External test: 243 natural Japanese reports (RR-Findings).

[日本語版 README](README.ja.md)

This repository contains code and settings only. It contains no report text, labels,
predictions or model weights. You download the data yourself from the original sources.

## Data and terms of use

| Dataset | Source (pinned revision) | License |
|---|---|---|
| CT-RATE | [ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) `deeca4d8` | CC BY-NC-SA 4.0 (gated: accept the terms first) |
| CT-RATE-JPN | [YYama0/CT-RATE-JPN](https://huggingface.co/datasets/YYama0/CT-RATE-JPN) `41d80fdb` | CC BY-NC-SA 4.0 |
| RR-Findings | [YYama0/RR-Findings](https://huggingface.co/datasets/YYama0/RR-Findings) `0911b306` | CC BY 4.0 |

Follow each dataset's license and cite the datasets. The MIT license of this code does not
cover the data. Internal labels are the automatic labels distributed with CT-RATE; external
labels are human annotations.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --locked                 # analysis only (tables from predictions)
uv sync --locked --group api     # + Jev, Decisions API, Gemini Decisions, OpenAI generation
uv sync --locked --group train   # + encoders and the Qwen typed decision (T1); Linux + NVIDIA GPU
cp .env.example .env             # then fill in the keys you need
```

Keys are read only from the environment or `.env`: `OPENAI_API_KEY`, `TYPESAFE_API_KEY`,
`GEMINI_API_KEY` (Gemini Decisions), `HF_TOKEN` (CT-RATE download).

## Steps

```bash
# 1. Download, check (SHA-256 and audited counts) and split
uv run ctfindings prepare

# 2a. Encoders (GPU): 3 learning rates x seed 42, thresholds and test predictions
uv run ctfindings train --condition E1 --language ja
uv run ctfindings train --condition E1 --language ja --fraction 10     # training-data size (1/2, 1/10, 1/100)
#     The paper used three seeds; add them to reproduce it: --seed 42 123 456

# 2b. Other conditions, once each (--run 1): always check with --dry-run first (nothing is sent).
#     The paper repeated several conditions three times; add --run 2 and --run 3 to do so.
uv run ctfindings predict --condition J1 --language ja --text ja_Checked --set test150 --run 1 --dry-run
uv run ctfindings predict --condition J1 --language ja --text ja_Checked --set test150 --run 1

# 3. Thresholds from the 450 development cases (T1; Jev and Decisions API use the frozen
#    values in configs/frozen_thresholds.yaml)
uv run ctfindings predict --condition T1 --language en --text en --set dev450 --endpoint http://localhost:8000
uv run ctfindings thresholds --condition T1 --language en

# 4. Metrics and tables
uv run ctfindings evaluate
uv run ctfindings tables
```

Outputs go to `work/`. A stopped `predict` resumes and skips finished cases. A failed
request is retried once; refusals and invalid outputs are kept and counted as negative.

Texts: `en`, `ja_MT` (machine translation), `ja_Revised` (physician revised),
`ja_Checked` (specialist checked), `external`. Sets: `dev450`, `test150`, `external`.
Conditions and settings: [configs/conditions.yaml](configs/conditions.yaml).

### Local models (Qwen, MedGemma)

Start a vLLM 0.31.0 OpenAI-compatible server with the settings used in the paper, then pass
`--endpoint http://127.0.0.1:8000`. Keep the default served model name (the model ID).

```bash
MODEL=Qwen/Qwen3.8-27B; REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
# MedGemma: MODEL=google/medgemma-27b-text-it; REV=5b667cf2ddcf064085bc90952edb35a0edbfb79c
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve $MODEL --revision $REV --tokenizer-revision $REV \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --tensor-parallel-size 2 \
  --max-model-len 16384 --gpu-memory-utilization 0.9 --max-num-seqs 1 \
  --language-model-only --enforce-eager --generation-config vllm --reasoning-parser qwen3 \
  --logprobs-mode raw_logprobs --max-logprobs 20
```

`--seed-salt` adds a per-request sampling seed (used for the English Qwen runs in the paper).

## Outputs and paper tables

| File in `work/results/` | Paper |
|---|---|
| `table02_03_internal.csv` | Tables 2, 3, 5 (internal) |
| `table04_external.csv` | Tables 4, 5 (external) |
| `table06_internal_to_external.csv` | Table 6 |
| `table07_findings_internal.csv`, `table07_findings_external.csv` | Table 7 |
| `table08_fixed05_difference.csv` | Table 8 |
| `table09_calibration.csv` | Table 9 |
| `table10_efficiency.csv` | Table 10 |
| `table11_variability.csv`, `table11_run_agreement.csv` | Table 11 |
| `table12_comparisons.csv` | Table 12 (paired bootstrap, 10,000 resamples, seed 20261110) |
| `table13_training_fraction.csv` | Table 13 |
| `table14_threshold_refit.csv`, `table14_threshold_refit_runs.csv` | Table 14 (post hoc, upper bound: thresholds refit on the test data) |

`metrics.csv` and `findings.csv` hold every population before averaging.

## Reproducibility notes

- Version 1.1.0 adds Gemini Decisions (`GD1`, `gemini-decision-1`) and the second
  manuscript's 15 paired comparisons. AP/mAP groups tied probabilities at each distinct
  score threshold, so changing the order of tied observations cannot change the result.
- API and sampled local-model runs may differ from the paper; service versions may be
  retired. This release is checked offline against stored predictions, not by new paid
  API calls or GPU training. The authors' predictions are not distributed here.
- Output filenames retain the original table numbering for compatibility. For manuscript version 2:
  `table02_03_internal.csv` and `table04_external.csv` supply Tables 2–3 and S2–S4;
  `table12_comparisons.csv` supplies Table 4 (15 comparisons, 95% and 99.6667% intervals);
  calibration, efficiency, variability, findings and refit files supply the corresponding
  supplementary tables. No external-threshold refit is produced for Gemini.

### Gemini Decisions

Use `uv sync --locked --group api` and set `GEMINI_API_KEY`. The SDK is pinned to
`google-genai==3.6.1`, API `v1beta`. Each request sends the same 18 clinical questions
as Jev, keyed as `finding_0` through `finding_17`. No text-generation fallback is used.
Only an actual `model_version` response field is recorded as the response version;
its absence is recorded as unknown.

```bash
# Dry run first. Then remove --dry-run to send the development requests.
uv run ctfindings predict --condition GD1 --language en --text en --set dev450 --run 1 --dry-run
uv run ctfindings predict --condition GD1 --language ja --text ja_MT --set dev450 --run 1 --dry-run
# After both development runs finish:
uv run ctfindings thresholds --condition GD1 --language en
uv run ctfindings thresholds --condition GD1 --language ja
# Evaluate with those first-run thresholds (example; also run each of the other texts below):
uv run ctfindings predict --condition GD1 --language ja --text ja_Checked --set test150 --run 1 --dry-run
```

Run `test150` for `en`, `ja_MT`, `ja_Revised` and `ja_Checked`, and `external` for Japanese,
once each (`--run 1`) with the development thresholds, as for the Decisions API (`D1`).

Request time includes failed attempts. Token usage is left blank when an attempt's usage
is unknown, rather than counted as zero. The model's response version and base-model
variant may be unavailable; do not infer them from the requested model name.

## Resources

Analysis: a laptop, minutes. Encoders: one NVIDIA GPU with bfloat16. Qwen3.8-27B and
MedGemma-27B: two GPUs (tensor parallel 2). One run of every API and local generation
condition on the test sets is about 7,500 requests (shown by `--dry-run`); T1 sends 18
requests per report.

## Tests

```bash
uv run pytest
uv run ruff check
```

## Citation

See [CITATION.cff](CITATION.cff). License: [MIT](LICENSE).
