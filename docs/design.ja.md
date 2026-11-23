# d1-fast-cuda 設計メモ

d1-3Bとd1-2BをCUDAで速く動かすために決めたこと、測ったこと、採用しなかったことをまとめます。数字はすべてRTX 4070 Ti（12GB、Ada、SM 60基）、PyTorch 2.14.1+cu130で測りました。ベンチマークの表は[../results/benchmark.md](../results/benchmark.md)にあります。

## 前提

d1は文章を生成しません。状態と質問を1回のforward passに通し、回答位置のlogitsから選択肢の確率を読みます。推論はprefillだけで、1問のプロンプトは約100トークンです（d1-fast-mlxの設計メモと同じ前提です）。

比較の相手は、各モデルカードの方法で動かした元のPyTorch実装です。d1-2Bは`model.compile()`、d1-3Bは`model.compile(mode="reduce-overhead")`を呼び、ウォーム呼び出しの中央値を取ります（`bench/benchmark.py`）。

始めて分かったことが1つあります。**元のコードはWindows版のPyTorchでは動きません。** d1-2Bのsliding attentionとset attention、d1-3Bの木構造（trunk/branch）のattentionが`torch.ops.aten._flash_attention_forward`を直接呼びます。Windows版のPyTorchはFlashAttentionなしでビルドされていて、`USE_FLASH_ATTENTION was not enabled for build`で止まります。元の実装の計測は、同じGPUを使うWSL2（Ubuntu、Linux版PyTorch）で行いました（`bench/wsl.sh`）。

## 構成

MLX版と違い、モデルのコードを持ち込みません。`load()`は元のチェックポイントを`AutoModel.from_pretrained(..., trust_remote_code=True)`でそのまま読み込み、そのengine（`model.engine`、元の`runner.SystemOne`）の下で次の2か所だけを差し替えます（`d1_fast_cuda/engine.py`）。

- d1-2B: `SystemOne._slot_logits`。1つのpadded batchから回答位置のlogitsとpointerの読み取りを返す関数です。画像の埋め込みと配置は元のmultimodal wrapperが行い、その中の言語モデルのforwardだけをこのパッケージのものにしました。
- d1-3B: `SystemOne.run`。元のrunnerと同じ手順（1問は1本のchain、1つの状態への複数質問はtrunkとbranchの木、1問ずつのリクエスト群は共通のsystem promptをtrunkにした木）で、このパッケージの言語モデルを呼びます。

プロンプトの組み立て、tokenizer、画像の前処理、vision tower、readout、calibration、APIの形はすべて元のコードのままです。答えの形式が元と違うことはありません。

言語モデル本体（`d1_fast_cuda/lm.py`の`HybridLM`）は、両モデル共通のハイブリッド構造（短いゲート付き畳み込み層とGQA attention層、SwiGLU MLP）を、行列積、`F.conv1d`、マスク付きの`F.scaled_dot_product_attention`だけで書いたものです。flash kernelを使わないので、WindowsでもLinuxでも動きます。重みは元のモデルのテンソルをそのまま参照します。融合した射影（q/k/v/gate、MLPのgate/up）だけは連結した1枚を作り、元のLinearの重みをその一部のviewに置き換えるので、メモリを二重に持ちません。fp8モードの行列積は`d1_fast_cuda/fp8.py`（Ada以降、TritonとCUTLASS）にあります。

## 検証の方法

参照には、元のPyTorch実装をCPUのfloat32で動かした答えを使います（`bench/reference_torch.py`）。問題はd1-fast-mlxと同じ`bench/cases.py`で、compare.pyの118問に、1つの状態への複数質問、JSONの状態、約4000トークンの長文、2Bのspan読み取り、画像などを足した81問（3B）と88問（2B）です。CPUで作り直した参照は、d1-fast-mlxのときの参照と差0.0000で一致しました。

| 実行 | d1-2B 最大差 | d1-3B 最大差 |
|---|---:|---:|
| 本パッケージ float32 | 0.0000 | 0.0000 |
| 本パッケージ bfloat16（既定） | 0.0069 | 0.0118 |
| 元の実装 bfloat16（CUDA） | 0.0094 | 0.0351 |
| 本パッケージ fp8（オプション） | 0.0349 | 0.0437 |

最大差は、確率値ごとの参照との差の最大値です。float32では両モデルとも5e-5未満で、最後の層の削減、木の詰め方、CUDA graph、選択肢の行だけのreadoutのどれも答えを変えていないことを示します。bfloat16では元の実装より参照に近くなりました。理由はd1-fast-mlxと同じで、選択肢のlogitsをfloat32で計算し、RoPEの角度もfloat32に保つためです（元のd1-2Bはcos/sinをbfloat16に丸めます）。118問の正解率は、どの設定でも参照と同じでした（2B 115/118、3B 113/118）。テストは`tests/test_equivalence.py`です。

## 速度について分かったこと

1問（約100トークン、バケットで128行）の時間を分けて測ると、GPUの時間がほぼすべてでした（`bench/profile_one.py`）。d1-2Bは全体12.6msのうちCUDA graphの再生が10.7ms、残りの約2msが元のコードのホスト処理（プロンプト、tokenizer、readout）です。d1-3Bは全体13.8msのうちGPUが13.0msでした。

GPU時間の大半は行列積です。cuBLASのbf16行列積を、依存関係のある連鎖として測りました（`bench/gemm_shapes.py`、M=128）。

| 形状 (K, N) | 用途 | 時間 | 帯域の何% |
|---|---|---:|---:|
| (2048, 5136) | 2B q/k/v/gate | 61 µs | 68% |
| (2048, 6144) | conv in | 73 µs | 68% |
| (2048, 15360) | 2B MLP up | 165 µs | 76% |
| (7680, 2048) | 2B MLP down | 85 µs | 73% |
| (2048, 21504) | 3B MLP up | 221 µs | 79% |
| (10752, 2048) | 3B MLP down | 117 µs | 75% |
| (2048, 2048) | 出力の射影 | 30 µs | 56% |

帯域は504GB/sとして、重みを1回読む時間との比です。M=128では行列積は重みの読み出しで決まり、cuBLASはその7〜8割を出しています。全層の行列積を足すと2Bで約8.4ms、3Bで約12.9msです。3Bの1問のGPU時間（13.0ms）は、行列積の和とほぼ同じでした。重みを1回読む下限は2Bで約6ms、3Bで約10msです。

ここから、bf16のままでは1問を大きく縮める余地は小さいと判断しました。削れるのは行列積以外の時間（2Bで約2ms）と、読むバイト数そのものです。

## 採用した高速化

| 変更 | 効果 | 答えへの影響 |
|---|---|---|
| 1件ずつの呼び出しをCUDA graphで再生（`graphs.py`） | カーネル起動の間隔を消す。Windows（WDDM）では起動が特に重い | なし（float32で差0.0000） |
| 最後の層で必要な行だけ計算（`HybridLM._tail`） | 3Bで30層中約2.7層分、2Bで24層中約1.7層分を省く | なし |
| 選択肢の行だけをfloat32で読む | 語彙12.8万行×2048（0.5GB）の読み出しを省く | 精度が上がる |
| `model.compile()`でnormやRoPEを融合 | 2Bの1問のgraph再生が10.7msから8.1ms | なし |
| 木を部分ごとにattention（`HybridLM._tree_attend`） | 3Bのバッチ64件が137件/秒から155件/秒 | なし |
| 2Bの長文でmaskを外し、sliding層をブロックに分ける | 2Bの長文（4180トークン）が207msから187ms | なし |
| fp8（オプション、`load(..., fp8=True)`） | 行列積の読み出しが半分。2Bの1問が9.4msから6.4ms | 最大差0.035〜0.044、正解率は同じ |

### CUDA graph

約100トークンのforwardでは、GPUが1つのカーネルを終えるより、ホストが次のカーネルを起動するほうが遅くなります。元の実装もd1-3Bで`reduce-overhead`（CUDA graph）を使っています。本パッケージは、プロンプト長を32トークン単位、必要な行数を4単位に切り上げたバケットごとにgraphを1つ作ります。パディングの行は本物の行の後ろに置きます。因果的な演算子（畳み込み、causal attention）では後ろの行は前の行に届かず、full attentionのマスクはパディングのkeyを隠します（パディングのqueryは自分だけを見ます）。graphはメモリプールを共有し、出力は次の再生の前に複製します。1024トークンを超えるプロンプトとバッチはeagerで動きます。

### 最後の層の削減

d1-fast-mlxと同じ考え方を、行列の形が固定になるように書き直しました。最後のattention層より後ろには短い畳み込みしかありません（2Bは1層でtaps 7、3Bは2層でtaps 3）。必要な行ごとに、畳み込みの文脈の分だけ前の行を含む「窓」（2Bで7行、3Bで5行）を作ります。最後のattention層は、keyとvalueを全行から作り、queryとMLPは窓の行だけで計算します。後ろの畳み込みは窓をvalidで畳み込み、窓を1行まで縮めます。窓の数と幅が固定なので、CUDA graphに乗ります。木のbranchの先頭の窓は、畳み込みの順序に従ってtrunkの末尾から前の行を取ります。

### 選択肢の行だけのreadout

答えは選択肢のtoken idのlog確率のsoftmaxです。語彙全体のlog-sum-expは全選択肢に共通の定数なので打ち消し合います。calibrationも温度で割るだけなので、この性質を壊しません。そこで、tied embeddingのうち必要な行だけを取り出してfloat32で掛けます。元の実装は2Bでも語彙全体の行列をbf16で掛けていました。

### 木を部分ごとにattention

d1-3Bは、1問ずつのリクエスト64件を、共通のsystem promptをtrunkにした1本の木（約6000トークン）として読みます。最初は木全体にT×Tのマスクを掛けていました。マスク付きのattentionはマスクの全要素を計算するので、ここが重くなっていました。元の`hybrid.py`はflashのvarlen呼び出しでbranchごとに分けています。本パッケージでは、trunkをcausalに1回、branchをpaddedのバッチにして「trunkのkey＋自分のkey」を見るように1回、の2回のSDPAにしました。スコアはT×(P＋L)で済みます。graphで動く小さい木（3問同時など）は、形を固定するためにT×Tのマスクのままです。

### 長文のd1-2B

2Bの1問はset attentionで、1問だけのときはプロンプト全体が全体を見ます。パディングのない1本の行なら、full attention層はマスクなしの非causalなSDPAで済みます。sliding attention層（窓256）は、窓の大きさのブロックに分け、各ブロックが自分と1つ前のブロックだけを見るようにしました（元の`attention._windowed`と同じ方法）。

### fp8（オプション）

RTX 4070 Ti（Ada）はfp8（e4m3）のtensor coreを持ちます。`torch._scaled_mm`は、M=128で幅の広い形状をbf16の約半分の時間で計算しました（`bench/fp8_shapes.py`）。

| 形状 (K, N) | bf16 | fp8 |
|---|---:|---:|
| (2048, 15360) 2B MLP up | 165 µs | 77 µs |
| (7680, 2048) 2B MLP down | 82 µs | 55 µs |
| (2048, 21504) 3B MLP up | 222 µs | 115 µs |
| (10752, 2048) 3B MLP down | 120 µs | 74 µs |
| (2048, 2048) 出力の射影 | 31 µs | 49 µs |

Adaでは行ごとのscale（rowwise scaling）が使えず、`_scaled_mm`はテンソル全体のscaleしか受け取りません。そこで重みは出力の行ごとにscaleしてe4m3に量子化し、活性化はトークンの行ごとにscaleして量子化し、両方のscaleは`_scaled_mm`のfloat32の出力に掛けています。2048×2048の射影はfp8のほうが遅かったので、bf16のまま残しました。変換した層の元のbf16の重みは解放します。

答えへの影響は、参照との最大差が0.035（2B）と0.044（3B）でした。元の実装のbf16（3Bで0.035）と同じくらいの幅で、ラベルの不一致はなく、118問の正解率も同じでした。既定はbf16にし、fp8は明示的に選ぶオプションにしています。

### Ada向けのfp8行列積カーネル（`fp8.py`）

fp8の1回のpassは、9割以上がcuBLASのfp8行列積です（3Bの1問で8.5ms中8.1ms）。読む重みのバイト数とメモリ帯域（504GB/s）から見ると、cuBLASは帯域の約7割しか使えていません。また、Adaでは`_scaled_mm`が行ごとのscaleを受け取れないので、float32の出力にscaleを掛ける処理が別に要ります。そこで、e4m3×e4m3をfp8のtensor coreで掛け、活性化の行ごとのscaleと重みのチャネルごとのscaleをepilogueで掛けてbf16で書き出すTritonのカーネルを書きました。

形状と行数ごとに、最も速いタイルの設定を総当たりで探して表（`fp8_table.json`、`bench/fp8_tune.py`で作成）にしています。計測では、L2（48MB）に載らない数の重みを順に読ませ、呼び出しはCUDA graphから再生します。Pythonから起動すると、Windowsでは1回あたり約20µsかかり、小さい形状ではそれがカーネルの時間を隠してしまうためです。

分かったこと：

- 1つの行列積の単位では、cuBLASより大きく速くはなりません。差がはっきり出たのは3BのMLPのup（M=128で128µs→110µs）とdown（68µs→63µs）だけで、多くの形状は±5%、M=192では幅2048の出力でcuBLASのほうが速くなります。cuBLASの欠点は、行数によって遅いアルゴリズムを選ぶことがある点でした。
- fp8のtensor core（mma e4m3）は、fp32の累積器への足し込みをfp32より粗い精度で行います（Hopperと同じ性質です）。そのまま長いKを累積すると、K=10752で出力の誤差がbf16の丸めの約1.8倍になりました。BKのタイル2枚ごとに別の累積器で受け、それをfp32で足すようにして、cuBLASと同じ誤差（bf16の丸め）に戻しました。`acc += tl.dot(...)`と書いても、Tritonが`dot(x, w, acc)`にまとめてしまうので効きません。`max_num_imprecise_acc`もsm_89では効きません。
- custom opの中からcuBLASを呼ぶ形にすると、float32の出力にscaleを掛ける処理をinductorが次の演算と融合できず、かえって遅くなりました（2Bの3問同時で7.5ms→8.5ms）。カーネルを使わない行数では、従来どおり`_scaled_mm`を直接呼び、scaleの処理はtorch.compileに融合させます。
- 2048×2048の出力の射影もfp8にすると、2Bの1問が約5%速くなりますが、参照との最大差が2倍（0.073と0.080）になったので、bf16のまま残しました。

### CUTLASSのfp8行列積（`csrc/fp8_cutlass.cu`、`cutlass_fp8.py`）

「帯域の9割を出す行列積」を狙って、CUTLASS 4.7.1の2.x API（`GemmUniversal`、Sm89、e4m3×e4m3、fp32の累積を定期的にfp32へ移すstaged accumulation）で、タイル9種類×（データ並列／Stream-K）の18通りを作り、同じ条件（L2に載らない重み、CUDA graphから再生）でcuBLASと比べました（`bench/cutlass_fp8_bench.py`）。

| 行数 M | 帯域に対する割合（cuBLAS → CUTLASSの最良） |
|---|---|
| 16〜64 | 幅の広い射影で80〜91% → 80〜93%。どちらもすでに9割前後 |
| 128 | 3BのMLP down（10752→2048）64% → 82%、2Bのdown（7680→2048）64% → 79%。幅の広い射影は69〜83%で同じ |
| 192〜256 | 幅の広い射影はcuBLASのほうが速い。downは13〜20%速い |
| 512〜4096 | ほとんどの形状でcuBLASのほうが速い。downだけ同等〜2割速い |

分かったこと：

- 帯域の9割が出るのはM≦64だけで、そこではcuBLASもすでに出ています。M=128〜256で割合が下がるのは、帯域ではなくfp8 tensor coreの演算量が効き始めるためです（M=192、2048→15360では、重みの読み出しが62µs、演算が約75µs）。1問の行数（128前後）では、帯域の9割は原理的に届きません。
- CUTLASSがはっきり勝つのは、出力が2048列でKが長いMLP downでした。cuBLASのデータ並列のカーネルでは出力のタイルが少なく、60基のSMが余ります。Stream-KはKの方向の仕事をSM全体に均して配るので、その余りを埋めます。
- WindowsのMSVCでは、CUTLASSのCuTeのヘッダーがC++20で通りません（`stride.hpp`でC3545）。epilogueでscaleを掛けるSm80のEVT（epilogue visitor tree）は、C++17でもsm90の部分がnvccで通りませんでした。そこで、CuTeを使わない2.xのAPIだけを使い、float32で出力する（scaleはTritonを使わない経路と同じく外で掛け、torch.compileが融合します）形にしました。PyTorchのヘッダーはC++20を要求するので、CUTLASSの部分はC++17の独立したDLLにして、ctypesで呼びます（`fp8.gemm32`というcustom opの中）。
- 精度はcuBLASと同じでした（float64に対する平均誤差1.3e-4）。

使い分けは表（`fp8_table.json`の`gemm`）で決めます。形状と行数ごとに、CUTLASSがcuBLASより3%以上速ければCUTLASS、そうでなければcuBLASです。選ばれるのはほぼMLP downだけです。

### 使い分け

Tritonのカーネル（bf16でscale済みの出力）と`gemm32`（CUTLASSかcuBLAS、float32の出力、scaleは外）を、モデルごとに次のように使い分けます（`Table.rows`）。

- Tritonは幅の広い射影（出力3072列以上）にだけ使います。比べる相手は`gemm32`の時間に、float32で書いて読み戻すぶん（1要素あたり4バイト）を足したものです。
- 行数の範囲は、表の上で幅の広い射影の合計が相手より遅くならず、どこかで5%以上速い連続した範囲です。32行以下は表の分解能が粗く、2Bの最後の層の窓（1〜21行）で遅くなったので、`gemm32`のままです。結果は両モデルとも33〜512行です。
- 出力の幅が2048の射影（MLP down）は、常に`gemm32`です。

効果（GPUのgraph再生時間、同じ日に設定だけを変えて計測）：

| | cuBLASだけ | +CUTLASS | +CUTLASS+Triton |
|---|---:|---:|---:|
| 2Bの1問 | 5.32ms | 5.15ms | 5.01ms（−6%） |
| 2Bの3問同時 | 5.94ms | 5.93ms | 5.85ms（−1.5%） |
| 3Bの1問 | 8.41ms | 8.05ms | 7.70ms（−8.5%） |
| 3Bの3問同時 | 9.46ms | 9.42ms | 9.46ms（同じ） |

答えは従来のfp8と同じ誤差の範囲で、118問の正解率も変わりません。`D1_FP8_KERNEL=0`でTritonを、`D1_CUTLASS_GEMM=0`でCUTLASSを外せます。CUTLASSのDLLは、初めてfp8を使うときにnvccでビルドされ、`~/.cache/d1_fast_cuda`に置かれます。nvcc、ホストのコンパイラ（WindowsではVS 2022のMSVC）、CUTLASSのヘッダー（`third_party/cutlass`か`D1_CUTLASS`）のどれかがなければ、警告を出してcuBLASだけで動きます。

## 試して採用しなかったこと、つまずいたこと

- inductorの`max_autotune`の行列積は、「SMが68基未満」の判定で4070 Ti（60基）では使われません。強制して試すほどの見込みはないと判断しました。cuBLASはすでに帯域の7〜8割を出しているためです。
- Windowsでtorch.profilerが止まったように見えた件は、profilerのせいではありませんでした。計測用に`bench/kernels.py`という名前のファイルを置いたところ、transformersがHub kernels用のパッケージ`kernels`の有無を調べる`import kernels`がそのファイルを読み込み、モデルの読み込みの途中で同じスクリプトを再帰的に実行して止まっていました。名前を変えて解決しています。

## 結果

全体の表は[../results/benchmark.md](../results/benchmark.md)です（`bench/run_all.sh`を2回通し、設定ごとに速いほうを採りました）。元の実装の行ごとに速いほうの設定と比べた、本パッケージの`compile`後の数字です。

| | d1-2B bf16 | d1-2B fp8 | d1-3B bf16 | d1-3B fp8 |
|---|---:|---:|---:|---:|
| 1問 | 15.6→9.4ms（1.65倍） | 6.4ms（2.44倍） | 16.5→13.8ms（1.19倍） | 8.4ms（1.95倍） |
| 3問同時 | 22.8→11.2ms（2.03倍） | 7.5ms（3.02倍） | 36.2→18.4ms（1.97倍） | 10.7ms（3.38倍） |
| 長文 約4200トークン | 199→187ms（1.07倍） | 135ms（1.48倍） | 321→283ms（1.13倍） | 188ms（1.71倍） |
| 画像384px 1問 | 27.5→19.6ms（1.41倍） | 16.1ms（1.71倍） | 36.8→32.6ms（1.13倍） | 25.2ms（1.46倍） |
| バッチ64件 | 196→206件/秒（1.05倍） | 285件/秒（1.45倍） | 109→155件/秒（1.42倍） | 215件/秒（1.97倍） |

fp8の列は、TritonとCUTLASSの行列積を入れたあとに測り直した数字です。日によってGPU全体の速さが数%揺れるので、行列積の効果は、同じ日に設定だけを変えて測ったGPUの再生時間（上の節）で判断しています。

bf16で大きく速くなったのは、2Bの短いプロンプトと、両モデルの複数質問です。3Bの1問は、元の実装もCUDA graphで動いていて、時間がcuBLASの行列積の和で決まるため、差は19%にとどまりました。長文とバッチも行列積が大半で、bf16では差が小さく、fp8で1.5〜2倍になります。

メモリは、常駐する量（読み込み後）で2Bが3.84GB、3Bが6.26GBです。fp8では2.48GBと3.95GBになります。表のピークがfp8で大きいのは、変換の途中でbf16とfp8の重みが一時的に両方あるためです。

## 計測上の注意

元の実装のWSL2での数字は、通し計測ごとに揺れました。2Bの1問はある回で13.8ms、別の回で16.2ms、3Bの3問同時（compileあり）は34.2msから56.0msまで動きました。このGPUは画面の表示も受け持っているので、その影響もありえます。元の実装に不利にならないよう、2回の通し計測の速いほうを採り、さらにcompileあり・なしの速いほうと比べています。3Bの3問同時は、元の実装ではcompileなしのほうが速く出ました（`reduce-overhead`がCUDA graphにするのは1問の経路だけで、木の経路はcompileされません）。

元の実装はWSL2でしか動かないので、本パッケージはWindowsとWSL2の両方で測りました。同じ設定のWSL2の数字は、Windowsとほぼ同じか少し遅い程度でした（ベンチマークの表の最後の列）。速さの差はOSの違いによるものではありません。

compileは初回に時間がかかります（2Bで約1分、3Bで約2分、Windowsのtriton）。プロンプト長のバケットごとにCUDA graphの捕捉が走るので、提供するプロンプト長は事前にウォームアップしてください。

## 今後の候補

- fp8の1問は、行列積の和でほぼ決まっています。TritonとCUTLASSのfp8行列積を書いてみましたが、1問の行数（128前後）では演算量が効き始めていて、帯域の9割は届きません（上の節）。重みだけをfp8にしたW8A16も試しましたが、Tritonではe4m3からbf16への変換が重く、W8A8より遅くなりました。これ以上縮めるには、読む重みのバイト数をさらに減らす（int4など。ただしAdaに固有の機能ではありません）か、層をまたいだ融合などの別の手が要ります。
- 3Bの画像1問は、SigLIP2のvision towerが大きな割合を占めます。d1-fast-mlxで効果のあった「vision towerの残差だけをfloat32に保つ」変更は、精度が基準内なので入れていません。
