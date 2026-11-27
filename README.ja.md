# kagiana

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)

[English](README.md) | **日本語**

kagiana は SMILES のライブラリをファーマコフォアでスクリーニングします。モードは 2 つで、参照分子
1 つに対する 2D フィンガープリント類似度と、拘束付き埋め込みによる 3D マッチングです。CLI は人間
よりもプログラムが使うことを前提にしており、非対話・自己記述的で、終了コードが安定しています。

SMILES の列を持つ CSV を渡すとスコア列を追記します。それ以外の列は、受け取ったままの姿で返します。
スクリーニングは CPU のコアに分散し、1 分子が run 全体を止めることはなく、重い部分には CUDA 経路が
あります。

> **状態: バージョン 0.1、開発中。** CPU 経路はテスト済みで、実データでも動かしています。GPU 経路は
> 実装済みですが一度も実行していません。[検証状況](#検証状況)を参照してください。

- [特長](#特長)
- [仕組み](#仕組み)
- [必要なもの](#必要なもの)
- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [入力ファイル](#入力ファイル)
- [出力ファイル](#出力ファイル)
- [CLI リファレンス](#cli-リファレンス)
- [並列化とタイムアウト](#並列化とタイムアウト)
- [再開](#再開)
- [GPU 高速化](#gpu-高速化)
- [トラブルシューティング](#トラブルシューティング)
- [開発](#開発)
- [検証状況](#検証状況)
- [ライセンス](#ライセンス)

## 特長

- **2 つのスクリーニングモード。** 2D ファーマコフォアフィンガープリントと、拘束付き埋め込みに
  よる 3D マッチング。
- **分子ごとの期限。** 埋め込みが収束しない分子は打ち切り、run は続行します。
- **マルチコア。** 分子をワーカープロセスに分散し、結果は入力順で、並列数を変えても同一です。
- **再開可能。** 中断した run は、どのファイルが残っているかではなく、自身が完了として記録した
  ところから再開します。
- **プログラム向け。** データは CSV、進捗は stderr、全行に status、失敗には理由、機械可読な実行
  サマリ、意味のある終了コード。
- **あなたの列はあなたのもの。** スコアしない列は書かれたまま引き継ぎます。

## 仕組み

**2D。** 参照分子とライブラリの各分子を 2D ファーマコフォアシグネチャにします。3 つの類似度は
1 つの共通ビット数から導くので常に整合します。3 つのカットオフをすべて満たした行が残ります。

**3D。** 各分子について、まず必要なフィーチャ型をそもそも持つかを調べ、距離境界行列を導き、距離が
クエリに収まるフィーチャと原子の割り当てを列挙し、それぞれで配座の埋め込みを試みます。スコアは
埋め込めたポーズ数です。

すべての分子は期限付きでワーカープロセスで処理されます。3D の重い部分は RDKit の C++ 層で動き、
Python レベルの割り込みが届かないため、期限が過ぎたら親がワーカーを強制終了し、代替を起動します。

## 必要なもの

- Python 3.11 以降
- RDKit, pandas, numpy, tqdm（パッケージと一緒に入ります）
- CUDA 経路の場合: Linux x86_64、compute capability 7.0 以降の NVIDIA GPU、ドライバ 560.28 以降、
  CUDA ビルドの PyTorch

## インストール

```bash
uv venv --python 3.11
uv pip install -e .
```

テストを動かす場合は dev extra を入れます。

```bash
uv pip install -e ".[dev]"
```

CUDA 経路は、GPU のあるマシンで次のとおりです。

```bash
uv pip install --torch-backend=cu128 -e ".[gpu]"
```

そのマシンで何ができるかを確認します。

```bash
kagiana backends --json
```

### Docker

```bash
docker build -f docker/Dockerfile.cpu -t kagiana:cpu .
docker run --rm \
    -v "$PWD/data:/work/data" -v "$PWD/results:/work/results" \
    kagiana:cpu pharm3d \
        --library /work/data/library.csv \
        --pharmacophore /work/data/pharmacophore.csv \
        --fdef /work/data/features.fdef \
        --out /work/results/pharm3d.csv
```

GPU イメージは x86_64 のホストと NVIDIA Container Toolkit が必要です。

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.gpu -t kagiana:gpu .
docker run --rm --gpus all kagiana:gpu backends
```

Compose なら `./data` と `./results` をマウントします。

```bash
docker compose -f docker/docker-compose.yml --profile cpu run --rm cpu backends
docker compose -f docker/docker-compose.yml --profile gpu run --rm gpu backends
```

## クイックスタート

```bash
# まず入力を検査する。スクリーニングは行わない
kagiana validate --library data/library.csv --fdef settings/features.fdef \
    --pharmacophore data/pharmacophore.csv --json

# 2D: 参照分子 1 つに対してライブラリをスコアリングする
kagiana pharm2d \
    --library data/library.csv \
    --query data/reference.csv \
    --fdef settings/features.fdef \
    --params data/params.json \
    --out results/pharm2d.csv

# 3D: クエリを満たすポーズを各分子が何通り取れるかを数える
kagiana pharm3d \
    --library data/library.csv \
    --pharmacophore data/pharmacophore.csv \
    --fdef settings/features.fdef \
    --out results/pharm3d.csv \
    --jobs 8
```

## 入力ファイル

### ライブラリ CSV

`SMILES` 列が必須です。それ以外の列は手を加えず出力へ引き継ぎます。セルは文字列として読むので、
ID の `00123` が `123` になることはなく、注記の `NA` が空欄になることもありません。

```csv
ID,SMILES,vendor_note
LIB-001,NCCc1ccc(O)cc1,dopamine-like
LIB-002,NCCc1ccccc1,phenethylamine
```

### 参照分子 CSV

`pharm2d --query` 用です。`SMILES` 列を持ち、**先頭行**を参照分子として使います。2 行目以降は
無視します。

### 特徴点 CSV

`pharm3d --pharmacophore` 用です。1 行 1 特徴点で、列は `feature_type`, `X`, `Y`, `Z`。
`feature_type` は fdef が定義する family でなければならず、誤記はヒット 0 件の run ではなく入力
エラーになります。

```csv
feature_type,X,Y,Z
Donor,-3.606,0.408,0.856
Acceptor,4.101,0.237,-0.398
Aromatic,1.412,-0.215,-0.031
```

点間の距離境界はこの座標から導きます。参照距離 `d` の各ペアについて、下限は
`max(0, d - limit_lower)`、上限は `d + limit_upper` です。**すべての**特徴点を満たす必要が
あります。

### params JSON

`pharm2d --params` 用です。7 つのキーすべてが必須で、隠れた既定値はありません。

```json
{
  "n_maxpoint": 3,
  "bins": [[0, 2], [2, 5], [5, 8]],
  "skip_feature": ["ZnBinder"],
  "chunksize": 100000,
  "tanimoto_cutoff": 0.6,
  "simpson_cutoff": 0.6,
  "dice_cutoff": 0.6
}
```

`n_maxpoint` はシグネチャの最大特徴点数で、最小は常に 2 です。`bins` はトポロジカル距離のビン、
`skip_feature` は除外する family、`chunksize` は一度に読むライブラリの行数です。行が残るのは
**3 つすべて**のカットオフを満たしたときだけです。

3 つのカットオフと `chunksize` はコマンドラインで上書きできますが、シグネチャのパラメータは
できません。run のフィンガープリントは常に params ファイルと一致します。

## 出力ファイル

### 列

入力の列のうしろにスコア列、その後ろに診断列が付きます。

| モード | スコア列 | 診断列 |
| --- | --- | --- |
| `pharm2d` | `tanimoto`, `dice`, `simpson` | `status`, `elapsed_s` |
| `pharm3d` | `match`, `numMatch` | `status`, `elapsed_s`, `stage`, `best_rmsd` |

`match` は必要なフィーチャ型をそもそも持つか、`numMatch` は条件を満たすポーズを何個埋め込めたか
です。`stage` はどの経路がその行を作ったか、`best_rmsd` は GPU プリフィルターの最良の剛体フィット
です。

### 行の status

| 値 | 意味 |
| --- | --- |
| `ok` | 正常にスコアリングした |
| `no_feature` | スクリーニングは正常。必要なフィーチャを持たない |
| `parse_error` | セルが分子を表していない（解釈不能、または空） |
| `timeout` | `--timeout` を超えてワーカーが強制終了された |
| `error` | スコアリングが例外を投げた。理由は診断ログにある |

正常にスコアリングできなかった行は `<out>_errors.jsonl` にも書かれます。1 行 1 JSON オブジェクト
で、チャンク、行、SMILES、理由が入ります。結果 CSV はスキーマを保ち、説明はその隣に置きます。

### ファイル名

`pharm2d` は常に `chunksize` 行ずつライブラリを読みます。

| ファイル | 内容 |
| --- | --- |
| `<out>.csv` | 3 つのカットオフを満たした行 |
| `<out>_all.csv` | 全行とそのスコア（`--no-all-rows` で抑止） |

`pharm3d` の既定である全件モードは、ライブラリを一括で読みます。

| ファイル | 内容 |
| --- | --- |
| `<out>.csv` | 全行とそのスコア |
| `<out>_screened.csv` | `numMatch > 0` の行 |

`pharm3d --large` は `--chunksize` 行ずつ読んで逐次書き出すので、中断しても使える部分結果が
残ります。

| ファイル | 内容 |
| --- | --- |
| `<out>_<i>.csv` | チャンク `i` の全行 |
| `<out>_screened_<i>.csv` | チャンク `i` のヒット |
| `<out>_screened_all.csv` | run 全体のヒット |

## CLI リファレンス

```text
kagiana pharm2d   --library L.csv --query Q.csv --fdef F.fdef --out R.csv --params params.json
                  [--tanimoto-cutoff F] [--dice-cutoff F] [--simpson-cutoff F] [--chunksize N]
                  [--jobs N] [--backend auto|cpu|gpu] [--no-all-rows]
                  [--resume] [--json-summary S.json] [--quiet]

kagiana pharm3d   --library L.csv --pharmacophore P.csv --fdef F.fdef --out R.csv
                  [--large] [--chunksize N]
                  [--conformers N] [--max-poses N] [--limit-upper F] [--limit-lower F]
                  [--timeout SEC] [--jobs N]
                  [--backend auto|cpu|gpu] [--prefilter-rmsd F] [--prefilter-conformers N]
                  [--resume] [--json-summary S.json] [--quiet]

kagiana backends  [--json]
kagiana validate  --library L.csv [--fdef F.fdef] [--pharmacophore P.csv]
                  [--params P.json] [--query Q.csv] [--json]
```

データは出力 CSV へ、進捗は stderr へ行くので、呼び出す側はノイズを除かずに結果を解析できます。
`--json-summary` は件数、使ったバックエンド、生成したファイルを書き出します。

### 既定値

`pharm3d` は実績のある設定値を既定にしています。

| オプション | 既定値 | 意味 |
| --- | --- | --- |
| `--conformers` | 10 | 割り当てごとに要求する配座数 |
| `--max-poses` | 20 | `--conformers` を掛けた値が分子ごとの上限 |
| `--limit-upper` | 1.5 | 各参照距離に足すオングストローム |
| `--limit-lower` | 0.5 | 各参照距離から引くオングストローム |
| `--timeout` | 10 | 拘束付き埋め込み段で 1 分子に許す秒数 |
| `--chunksize` | 1000000 | `--large` 時のチャンクあたりの行数 |

`--jobs` の既定は使用可能な全コア、`--backend` の既定は `auto` です。入出力パスに既定値はなく、
未指定は入力エラーになります。意図しない場所へ書くことはありません。

### 終了コード

| コード | 意味 |
| --- | --- |
| 0 | run が完了した |
| 1 | 実行時の失敗 |
| 2 | 入力の誤り。ファイルがない、CSV や params が壊れている、`validate` が失敗した、明示的に指定したバックエンドが使えない |

## 並列化とタイムアウト

分子はワーカープロセスに分散して処理されます。3D の重い部分は RDKit の C++ 層で動き、Python
レベルの割り込みが届かないため、距離幾何の埋め込みが収束しない分子をワーカーの内側から止めること
はできません。代わりに親が各分子の投入時刻を追跡し、`--timeout` を過ぎたら**ワーカーを強制終了**
して代替を起動します。run は続き、その行は `timeout` となり、すでに埋め込めたポーズ数に中断され
た割り当ての 1 を足した値を保ちます。

期限を過ぎてから完了した分子も timeout として報告します。遅れの原因が処理そのもの、ワーカーの
起動、親の読み取りのいずれであっても同じです。

**期限が及ぶ範囲。** `--timeout` は拘束付き埋め込み段に適用されます。GPU プリフィルターは対象外
です。バッチ全体の配座を 1 回のデバイス呼び出しで生成するため、その中に分子ごとの中断点を設ける
とバッチ処理の利点そのものを失います。設計と、これまでに間違えた経緯は
[docs/reliability.md](docs/reliability.md) にあります。

### 実測のスケーリング

50,000 分子、12 コアのマシン、単位は秒。並列数を変えても結果は同一です。

| モード | `--jobs 1` | `--jobs 4` | `--jobs 12` |
| --- | --- | --- | --- |
| `pharm3d` | 24.8 | 8.0 | 4.0 |
| `pharm2d` | 4.3 | 1.9 | 1.3 |

ワーカーはチャンクごとに起動し、fork ではなく spawn のプラットフォームでは各ワーカーが RDKit を
再 import します。数千分子のライブラリではこの起動コストがスクリーニング自体を上回ることがあり、
その場合は `--jobs 1` のほうが速いこともあります。効果が出るのは 1 チャンクが数万分子になってから
です。

## 再開

`--resume` がスキップするのは、**この**入力と設定の組み合わせが完了として記録した分だけです。出力
ファイル自体はどの run が作ったかを語れないので、run は結果の隣の隠しディレクトリ
`.<out>_state/` に小さな記録を残します。実行した入力、完了したチャンク、そして各チャンクの診断
です。

したがって別のライブラリを同じ出力先にスクリーニングすると、以前の結果の半分を引き継ぐのではなく
やり直します。チャンクが完了扱いになるのは、その診断と CSV のすべてがディスクに載ってからです。
結果ファイルはすべて一時名に書いてから置換するので、最終名のファイルは完成しています。

## GPU 高速化

CUDA 経路は [nvMolKit](https://github.com/NVIDIA-BioNeMo/nvMolKit) を使います。nvMolKit はビルド
時の RDKit バージョンを固定します。

`--backend auto` は GPU が使えればそれを使い、使えなければ stderr に通知して CPU に落ちます。
`--backend gpu` は落ちずに終了コード 2 で失敗します。nvMolKit 自体に CPU フォールバックがなく、
デバイスを要求した run が黙って CPU の run に変わるべきではないからです。

各モードがオフロードするもの:

- **2D** — ファーマコフォアフィンガープリントの構築は RDKit のグラフ処理なので CPU ワーカーに
  残り、GPU は全対 1 のビット比較を担います。したがって高速化はフィンガープリント生成に律速され、
  効くのは大規模ライブラリです。
- **3D** — 拘束付き埋め込みは GPU に移せません。GPU の埋め込みは境界行列も座標マップも受け取らず、
  それこそがファーマコフォア拘束の表現方法だからです。そこで GPU 経路は 2 段階で篩います。まず
  デバイス上で自由配座アンサンブルを作り、各分子のフィーチャをクエリ点へ重ねた最良の剛体フィット
  を測り、近づけない分子を棄却します。通過した分子は CPU バックエンドと同じ拘束付き埋め込み段へ
  進むので、`match` と `numMatch` の意味は変わりません。

プリフィルターは仕事を減らすだけです。判定を下すのは、クエリの全フィーチャについてフィットを測れ、
かつそれが明らかに遠い場合のみで、その行は `match=True`、`numMatch=0`、
`stage=gpu_prefilter_reject` を報告します。分子はフィーチャを持っていて、合わないのは幾何だから
です。測れなかったもの（使える配座がない、フィーチャ family が欠けている、割り当て探索が予算に
達した）は判定せず拘束付き段へ回します。第 1 段の厳しさは `--prefilter-rmsd` で決めます。

## トラブルシューティング

**`pharm2d` で "distance bin not found" の `error` 行が多く出る。** これはライブラリ側の問題では
なく、RDKit の 2D signature factory の制約です。フィーチャ間の距離はフィーチャの*原子集合*の間で
測られるため、三角不等式が許さない距離の組が生じることがあり、factory は構築時にまさにその bin の
組み合わせを枝刈りします。bin を広げる、`n_maxpoint` を 2 に下げる、`skip_feature` で複数原子の
family を除く、のいずれでも減ります。どちらにしても run 自体は完走し、該当行は印が付いて
`<out>_errors.jsonl` に理由が残ります。

**`--backend gpu` が終了コード 2 になる。** GPU 経路が使えません。`kagiana backends --json` で
理由を確認してください。CPU で処理するなら `--backend auto` を使います。

**状態ディレクトリの衝突が報告される。** `--resume` が依存する隠しの記録は、出力パスから名前を
作ります。すでに自分のディレクトリがそこにある場合、kagiana は削除せず報告します。移動するか、
別の `--out` を選んでください。

## 開発

```bash
uv pip install -e ".[dev]"
uv run pytest
uv run ruff check .
python scripts/check_readme_pair.py
```

[PROGRESS.md](PROGRESS.md) は、何が終わり、何を測り、次に何をするかのセッション横断の記録です。
[CLAUDE.md](CLAUDE.md) にはこのプロジェクトが従う規則があります。

2 つの README は同時に変更します。`scripts/check_readme_pair.py` が見出しの構成、コードブロック
中のコマンド、リンク、表の数値を比較し、CI がこれを実行します。

## 検証状況

CPU 経路はテストで覆われており、実データでも、CPU のコンテナイメージの中でも動かしています。

**GPU 経路は一度も実行していません。** 開発環境が macOS arm64 で CUDA を載せられず、nvMolKit は
x86_64 Linux の wheel しか公開していないためです。`linux/amd64` での依存解決と CUDA ベースイメージ
のタグの存在は確認しましたが、GPU イメージのビルドも `--backend gpu` 自体も実行していません。
適切なハードウェアでの検証手順は次のとおりです。

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.gpu -t kagiana:gpu .
docker run --rm --gpus all kagiana:gpu backends --json
uv run pytest -m gpu
```

## ライセンス

Apache-2.0。[LICENSE](LICENSE) を参照してください。
