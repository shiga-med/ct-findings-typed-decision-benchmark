# ct-findings-typed-decision-benchmark（日本語版）

胸部CTレポートの18所見分類で、追加学習したエンコーダー、生成型LLM、型付き判断（所見ごとの
Yes/No質問に確率を返す方式）を比べた研究のコードです。内部評価はCT-RATEの英日対応150レポート、
外部評価は自然な日本語243レポート（RR-Findings）です。

[English README](README.md)

コードと設定だけを含みます。レポート本文、ラベル、予測、モデルの重みは含みません。
データは各配布元から取得してください。

## データと利用条件

| データ | 配布元（固定した版） | ライセンス |
|---|---|---|
| CT-RATE | [ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) `deeca4d8` | CC BY-NC-SA 4.0（先に利用条件への同意が必要） |
| CT-RATE-JPN | [YYama0/CT-RATE-JPN](https://huggingface.co/datasets/YYama0/CT-RATE-JPN) `41d80fdb` | CC BY-NC-SA 4.0 |
| RR-Findings | [YYama0/RR-Findings](https://huggingface.co/datasets/YYama0/RR-Findings) `0911b306` | CC BY 4.0 |

各データのライセンスに従い、データも引用してください。コードのMITライセンスはデータに及びません。
内部評価のラベルはCT-RATE付属の自動ラベル、外部評価のラベルは人手注釈です。

## 準備

[uv](https://docs.astral.sh/uv/)とPython 3.12を使います。

```bash
uv sync --locked                 # 解析のみ（予測から表を作る）
uv sync --locked --group api     # Jev、Decisions API、OpenAI生成も使う
uv sync --locked --group train   # エンコーダーとQwen型付き判断（T1）も使う（Linux＋NVIDIA GPU）
cp .env.example .env             # 必要なキーを記入
```

キーは環境変数か`.env`からだけ読みます：`OPENAI_API_KEY`、`TYPESAFE_API_KEY`、`HF_TOKEN`（CT-RATEの取得）。

## 手順

```bash
# 1. 取得・照合（SHA-256と監査件数）・分割
uv run ctfindings prepare

# 2a. エンコーダー（GPU）：学習率3×seed 3、閾値、評価データの予測
uv run ctfindings train --condition E1 --language ja
uv run ctfindings train --condition E1 --language ja --fraction 10     # 学習データ量（1/2・1/10・1/100）

# 2b. その他の条件：まず--dry-runで確認（何も送信しない）
uv run ctfindings predict --condition J1 --language ja --text ja_Checked --set test150 --run 1 --dry-run
uv run ctfindings predict --condition J1 --language ja --text ja_Checked --set test150 --run 1

# 3. 開発450件から閾値（T1。JevとDecisions APIはconfigs/frozen_thresholds.yamlの凍結値）
uv run ctfindings predict --condition T1 --language en --text en --set dev450 --endpoint http://localhost:8000
uv run ctfindings thresholds --condition T1 --language en

# 4. 指標と表
uv run ctfindings evaluate
uv run ctfindings tables
```

出力は`work/`です。止まった`predict`は再実行で完了済みの症例を飛ばします。失敗した要求は1回だけ
再試行します。回答拒否と無効な出力は除外せず、陰性として数えます。

本文：`en`、`ja_MT`（機械翻訳）、`ja_Revised`（医師修正）、`ja_Checked`（専門医確認）、`external`。
集合：`dev450`、`test150`、`external`。条件と設定は[configs/conditions.yaml](configs/conditions.yaml)。

ローカルモデル（Qwen・MedGemma）は、[README.md](README.md#local-models-qwen-medgemma)の
コマンドでvLLM 0.31.0を起動し、`--endpoint`を指定します。

## 出力と論文の表

`work/results/`のファイルと論文の表の対応は[README.md](README.md#outputs-and-paper-tables)の表のとおりです。
表14は評価データで閾値を取り直した事後解析（上限値）で、妥当な推定値ではありません。

## 再現性

- 著者の保存予測を入力すると、`evaluate`と`tables`は表2〜9・11〜14の値を完全に再現します
  （表11のDecisions APIの2行は、予測を保存していない初回との比較のため除く）。
  `prepare`は論文と同じ分割を作ります。
- APIモデルとサンプリングを伴うローカル生成は、新しく実行すると論文の値と一致しません。
  APIのモデル版は提供終了する場合があります。公開にあたり有料API・GPUでの再実行は行っていません。

## 試験

```bash
uv run pytest
uv run ruff check
```

引用は[CITATION.cff](CITATION.cff)、ライセンスは[MIT](LICENSE)です。
