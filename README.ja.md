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
uv sync --locked --group api     # Jev、Decisions API、Gemini Decisions、OpenAI生成も使う
uv sync --locked --group train   # エンコーダーとQwen型付き判断（T1）も使う（Linux＋NVIDIA GPU）
cp .env.example .env             # 必要なキーを記入
```

キーは環境変数か`.env`からだけ読みます：`OPENAI_API_KEY`、`TYPESAFE_API_KEY`、`GEMINI_API_KEY`（Gemini Decisions）、`HF_TOKEN`（CT-RATEの取得）。

## 手順

```bash
# 1. 取得・照合（SHA-256と監査件数）・分割
uv run ctfindings prepare

# 2a. エンコーダー（GPU）：学習率3×seed 42、閾値、評価データの予測
uv run ctfindings train --condition E1 --language ja
uv run ctfindings train --condition E1 --language ja --fraction 10     # 学習データ量（1/2・1/10・1/100）
#     論文は3つのseedを用いた。再現するときは --seed 42 123 456 を加える

# 2b. その他の条件：各1回（--run 1）。まず--dry-runで確認（何も送信しない）
#     論文では一部の条件を3回実行した。同様にするときは --run 2・--run 3 も実行する
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

- 1.1.0ではGemini Decisions（`GD1`、`gemini-decision-1`）と第2版の15比較を追加しています。
  AP・mAPは同率確率をまとめて計算し、同率の症例順に依存しません。
- 保存済み予測を使ってオフラインで照合します。今回、有料APIの再送・GPU学習は行っていません。
  API・サンプリングを伴うローカル生成の再実行は論文と一致するとは限らず、モデル版が終了する場合もあります。
- 出力名は互換性のため旧表番号を維持します。第2版では内部・外部CSVが本文表2〜3・補足S2〜S4、
  `table12_comparisons.csv`が本文表4（15比較、95%・99.6667%区間）に対応します。
  校正・効率・実行変動・所見別・閾値取り直しは対応する補足表へ使います。Geminiの外部閾値取り直しは出力しません。

## Gemini Decisions

`uv sync --locked --group api`で準備し、`GEMINI_API_KEY`を設定します。
SDKは`google-genai==3.6.1`、APIは`v1beta`です。共通18質問を`finding_0`〜`finding_17`に対応させ、
1要求で送信します。応答に実際の`model_version`がなければ版は不明として記録します。

```bash
# まず送信なしで確認し、実送信するときだけ--dry-runを外す
uv run ctfindings predict --condition GD1 --language en --text en --set dev450 --run 1 --dry-run
uv run ctfindings predict --condition GD1 --language ja --text ja_MT --set dev450 --run 1 --dry-run
# 開発の実送信が終わってから初回閾値を固定する
uv run ctfindings thresholds --condition GD1 --language en
uv run ctfindings thresholds --condition GD1 --language ja
uv run ctfindings predict --condition GD1 --language ja --text ja_Checked --set test150 --run 1 --dry-run
```

内部の`en`・`ja_MT`・`ja_Revised`・`ja_Checked`（`test150`）と日本語外部（`external`）を、
開発集合で定めた閾値で各1回（`--run 1`）実行します。Decisions API（`D1`）と同じく1回の実行です。

要求時間は失敗試行を含みます。失敗試行などのトークン使用量が不明なら、ゼロにせず空欄にします。
応答版・基盤モデルの派生が不明な場合、要求モデル名から推測しません。

## 試験

```bash
uv run pytest
uv run ruff check
```

引用は[CITATION.cff](CITATION.cff)、ライセンスは[MIT](LICENSE)です。
