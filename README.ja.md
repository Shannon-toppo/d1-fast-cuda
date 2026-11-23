# d1-fast-cuda

[LiquidAI d1-3B](https://huggingface.co/LiquidAI/d1-3B)と[d1-2B](https://huggingface.co/LiquidAI/d1-2B)をCUDAで速く動かすパッケージです。WindowsでもLinuxでも動きます。d1-fast-mlxのCUDA版です。[English README](README.md)

d1は、状態についての型付きの質問（`noul`、`choice`、`score`、d1-2Bでは`text`、`number`、`date`のspan読み取り）に1回のforward passで答えます。文章は生成しません。このパッケージは、チェックポイント自身の`trust_remote_code`のモデルをそのまま読み込み、言語モデルのforwardと選択肢のreadoutだけを差し替えます。API、プロンプト、画像の前処理、答えの形式は元のままです。

```python
import d1_fast_cuda

model = d1_fast_cuda.load("LiquidAI/d1-3B")       # または "LiquidAI/d1-2B"。fp8=True で高速モード
model.compile()                                   # 任意: カーネルの融合（初回に約1分）
model.system_one(
    "I was charged twice and the app crashes when I open settings.",
    {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Payments.", "engineering": "Bugs and crashes."}},
    },
)
```

## 何が違うか

- **Windowsで動きます。** 元のコードはPyTorchのflash attentionの演算を直接呼びますが、Windows版のPyTorchにはそれがありません（`USE_FLASH_ATTENTION was not enabled for build`で止まります）。このパッケージはマスク付きのSDPAだけでattentionを計算します。
- **速くなります。** 1件ずつの呼び出しはCUDA graphで再生し、最後の層ではreadoutが読む行だけを計算し、logitsは選択肢のembeddingの行だけをfloat32で読み、複数質問の木はtrunkとbranchに分けてattentionを計算します。オプションのfp8モード（Ada、RTX 40シリーズ以降）では、1回のpassで読む重みのバイト数が半分になり、cuBLASより速い行数ではTritonで書いたfp8の行列積を使います。

## 速さと精度

RTX 4070 Ti、ウォーム呼び出しの中央値、2回の通し計測の速いほう。「元の実装」はチェックポイント自身のコードで、行ごとに速いほうの設定（モデルカードの方法の`compile`あり、またはなし）です。Windowsでは動かないのでWSL2で測りました。本パッケージはWindowsで測りました（WSL2でも数%以内の差です。全体の表を参照）。

| ms（小さいほど速い） | d1-2B 元の実装 | d1-2B bf16 | d1-2B fp8 | d1-3B 元の実装 | d1-3B bf16 | d1-3B fp8 |
|---|---:|---:|---:|---:|---:|---:|
| 1問 | 15.6 | 9.4 | 6.7 | 16.5 | 13.8 | 8.8 |
| 3問同時 | 22.8 | 11.2 | 7.6 | 36.2 | 18.4 | 11.5 |
| 長文 約4200トークン | 199 | 187 | 135 | 321 | 283 | 188 |
| 画像384px 1問 | 27.5 | 19.6 | 15.8 | 36.8 | 32.6 | 25.0 |
| バッチ64件（件/秒） | 196 | 206 | 283 | 109 | 155 | 215 |
| GPUメモリ 常駐（GB） | | 3.8 | 2.5 | | 6.3 | 4.0 |

bf16とfp8は`model.compile()`のあとの数字です。compileしなくても1件ずつの呼び出しはCUDA graphで動きます（1問で2Bが12.4ms、3Bが15.8ms）。全体の表と方法は[results/benchmark.md](results/benchmark.md)と[docs/design.ja.md](docs/design.ja.md)にあります。

compare.pyの118問の正解率は、どの設定でも元の実装と同じです。元の実装をCPUのfloat32で動かした答えとの確率の差は、bfloat16（既定）で最大0.0069（d1-2B）と0.0118（d1-3B）でした。元の実装をCUDAのbfloat16で動かした場合は0.0094と0.0351で、本パッケージのほうが参照に近い値です。float32では5e-5以内で一致します。fp8は最大0.035と0.044で、ラベルと正解率は変わりません。設計と計測の詳細は[docs/design.ja.md](docs/design.ja.md)にあります。

## インストール

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe -e ".[dev,compile]"
```

`compile`はWindowsで`triton-windows`を入れます（`model.compile()`とfp8モードに必要です）。fp8にはAda（RTX 40シリーズ）以降のGPUが必要です。

## テスト

テストは、元の実装をCPUのfloat32で動かした答え（`results/runs/*-torch-cpu.json`、同梱）と比べます。作り直す場合は次のとおりです。

```powershell
.venv/Scripts/python.exe bench/reference_torch.py LiquidAI/d1-2B --device cpu
.venv/Scripts/python.exe -m pytest tests -q
```

## ライセンス

このパッケージはd1チェックポイントのコードを動かし、その一部に沿って書いているため、LFM Open License v1.0です。[NOTICE](NOTICE)と[LICENSES/](LICENSES)を見てください。モデルの重みとチェックポイントのコードは、読み込み時にHugging Face HubからLFM Open License v1.0のもとでダウンロードされます。
