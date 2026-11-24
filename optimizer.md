# DGX Spark 最適化チーム（`bench-verifier` + `/swarm`）立ち上げガイド

TensorFold `v1.3.4`（NVIDIA DGX Spark / GB10, 121 GiB 統合メモリ）上で、高コストな Zig / CUDA カーネル解析・実装・検証を **`.agents/hooks.json` の決定論的ガードレール下** で安全に回すための `/swarm` チーム立ち上げテンプレートです。

---

## 1. クイックスタート（レバー選定後の 3 ステップ）

1. **セッション再読込／新規開始**（`.agents/hooks.json` と `.agents/skills/spark-kernel-swarm/SKILL.md` を有効化）。
2. **下の「コピー用 `/swarm` 起動プロンプト」の `【ここを埋める】` 箇所だけ書き換える**（対象レバー・ファイル・期待効果）。
3. **チャットで `/swarm` を実行して貼り付ける**。

### コピー用 `/swarm` 起動プロンプト

```text
/swarm
スキル `.agents/skills/spark-kernel-swarm/SKILL.md` と `optimizer.md` に従って、以下の最適化レバーの解析・実装・GPU 検証チームを立ち上げてください。

- 最適化レバー: 【例: Prefill NVFP4 MoE の小バッチ CTA タイル最適化】
- 主対象ファイル: 【例: zig/kernels/cuda/nvfp4_experts.cu, zig/src/families/flashnext/cuda_batch.zig】
- 狙う指標と期待効果: 【例: 4K/32K/128K prefill +5〜10%（c1/c5/TTFT/メモリ悪化なし、greedy ビット完全一致）】
- 採用基準:
  1. `python3 tools/check_invariants.py` 全 PASS
  2. `tools/exact.py` および `tools/exact.py --pad 3000` が serial / nodraft / conc ビット完全一致
  3. 交互 A/B（`ctl-a / cand / ctl-b`, `HOST=127.0.0.1`）で両側の対照群を上回り、他指標に退行なし
  4. ピーク統合メモリ <= 112.0 GiB、`tools/prompt_reuse.py 30000` 最低 90% 以上
  5. `tools/receipt.py record` でフィンガープリント一致の GPU 検証レシートを発行済み

チーム構成（役割と workspace を厳守）:
1. `profiler` (`workspace: "inherit"`): 対象カーネルと `work/opt/THEORY.md` を突き合わせ、ボトルネック・レジスタ/帯域見積り・実装仕様を作成してボードへ報告。
2. `author-1` (`workspace: "share"`): 独立 worktree で `patches/000N-*.patch`（必要なら `scripts/config.sh`）を実装。GPU は起動せず `tools/check_invariants.py` と静的/コンパイル検査（`zig build check-native-cuda-host`, `nvcc --ptxas-options=-v`）まで完了して差分を報告。
3. `critic` (`workspace: "inherit"`): `author-1` の差分を敵対的レビュー（数値丸め誤差、CUDA Graph キャプチャ阻害、`environment.zig` 登録漏れ、KV/プロンプトキャッシュ圧迫、slop コード有無）。問題があれば `author-1` に差し戻す。
4. `bench-verifier` (`workspace: "inherit"`, 単一 GPU 専有): `critic` の LGTM 後のみ起動。`tools/gpu_lock.py acquire --holder bench-verifier` で単一 GPU ロックを取得し、イメージビルド・交互 A/B・`exact.py`・`prompt_reuse.py` を実行、PASS 時のみ `tools/receipt.py record` を発行してロックを解放する。

coordinator は全ワーカーを統括し、未検証コードやレシート未発行のままマージ・コミットしないこと。
```

---

## 2. チーム構成と役割境界（フック強制一覧）

すべてのワーカーは自動的に [`.agents/hooks.json`](.agents/hooks.json) を継承します。

| メンバー | `workspace` | 主な責務 | フックによる自動制約 |
| --- | --- | --- | --- |
| `coordinator` | `inherit` | 仮説管理、ボード進行、検証済みパッチのマージ、`git commit` | Stage-1（`Stop` 静的検査）＋ Stage-2（`git commit` 時のハッシュ一致 GPU レシート必須） |
| `profiler` | `inherit` | TensorFold ソース読解、ルーフライン／バイト数・レジスタ見積り | 保護ファイル（`tools/exact.py`, `.agents/hooks*` 等）の編集は `force_ask` |
| `author-<k>` | `share` | 独立 git worktree での `patches/000N-*.patch` 作成、静的・コンパイル検査 | **副 worktree での GPU 実行・`./start.sh`・`nsys`/`ncu` を自動 `deny`**。`PostToolUse`→`PreInvocation` で不変条件違反を即時通知 |
| `critic` | `inherit` | 敵対的差分レビュー（数値精度・CUDA Graph・メモリ・`environment.zig`） | 保護ファイルの編集は `force_ask` |
| `bench-verifier` *(1 体のみ)* | `inherit` | 単一 GPU 専有でのビルド、交互 A/B、適格性ゲート実行、GPU レシート発行 | `work/hooks/gpu.lock` による排他制御（他ホルダー保持中は `deny`）、`MEM_CHECK=0` や reserve < 9 GiB は `deny` |

> **複数仮説を並列実装する場合**: `author-1`, `author-2` のように `workspace: "share"` で複数立てて構いません（両方 GPU 実行がブロックされるため安全です）。ただし **`bench-verifier` は必ず 1 体のみ** にし、1 仮説ずつ直列で GPU 検証します。

---

## 3. 各メンバー向けブリーフ（そのまま流用可能）

### `profiler`（解析担当 / `workspace: "inherit"`）
```text
あなたは `profiler` です。`work/opt/THEORY.md`（GB10 実測: D2D 225.2 GB/s, BF16 GEMM 97.5 TFLOPS）と対象 TensorFold v1.3.4 ソース（必要なら一時ディレクトリへ `TF_REF=12ebd414de443485d281768036ac4835e5abe8fe` を shallow clone）を読み、以下をボードへ報告してください:
1. 現在のボトルネック（計算律速 / 帯域律速 / 起動オーバーヘッド / ローンチ数）
2. 変更対象ファイル・関数・カーネルの正確な位置とデータ構造
3. 予測ゲイン（% または ms/tok）と統合メモリ増減（GiB）
4. CUDA Graph キャプチャや fp8 KV・`exact.py` ビット一致への影響リスク
完了条件: `author-1` が迷わずパッチを書ける粒度の実装仕様をボードに投稿すること。
```

### `author-1`（実装担当 / `workspace: "share"`）
```text
あなたは `author-1` です。独立 worktree で `profiler` の仕様に基づき `patches/000N-*.patch`（連番厳守）を実装してください。
厳守事項:
- 副 worktree では GPU 実行・`./start.sh` はフックで禁止されています（静的検査・コンパイル確認のみ実行）。
- 新しい `TF_FLASHNEXT_*` 変数を追加する場合は必ず `zig/src/platform/environment.zig` にも登録する。
- 非テストコードに `std.debug.print`、CUDA カーネルに `printf(`、未ガードの `cuda*Synchronize()`、`TODO`/`FIXME`/`HACK` を残さない。
- `scripts/config.sh` のデフォルト制約（`CONTEXT=262144`, `PARALLEL=5`, `VISION=1`, `KV_DTYPE=fp8`, `TENSORFOLD_MEMORY_RESERVE_GIB>=9`, `TF_FLASHNEXT_CHUNK_SOLO<=8192`）を破らない。
完了条件: `python3 tools/check_invariants.py` が全 PASS かつ `git apply --check` / 静的コンパイル確認が通った状態で、`@critic` にレビュー依頼すること。
```

### `critic`（敵対的レビュー担当 / `workspace: "inherit"`）
```text
あなたは `critic` です。`@author-1` が作成したパッチを壊すつもりで厳密レビューしてください。
確認項目:
1. 数値一貫性: greedy 出力が `draft: true` / `draft: false` / 同時実行でビット一致し続けるか（チャンク境界やバッチ幅依存の丸め差がないか）
2. CUDA Graph 安全性: キャプチャ経路内に同期やホスト分岐・使い捨てグラフ爆発が紛れ込んでいないか
3. メモリ安全性: 5×262K fp8 KV とプロンプトキャッシュ（>= 2.5 GiB）を圧迫しないか
4. 登録・保守性: `environment.zig` 登録漏れ、パッチ番号連番、余計なデバッグコードがないか
完了条件: 問題があれば具体的行番号付きで `@author-1` に差し戻し、全項目クリア時のみ `LGTM` を出して `@bench-verifier` に引き継ぐこと。
```

### `bench-verifier`（単一 GPU 検証・レシート発行担当 / `workspace: "inherit"`）
```text
あなたは単一 GPU 専有の `bench-verifier` です。`@critic` の LGTM 後、`@author-1` のパッチを検証用に取り込み、以下の手順を厳守して実行してください（Docker は必要に応じ `sg docker -c 'bash ...'` を使用）:

1. 単一 GPU ロック取得:
   python3 tools/gpu_lock.py acquire --holder bench-verifier --reason "<patch-name>"
2. クリーンビルド＆メモリ確認（`MemAvailable` と `docker ps` が空であることを確認）:
   `scripts/prepare.sh --rebuild`（またはテスト用タグ）でビルドし、警告ゼロ・グラフキャプチャ失敗ゼロを確認。
3. 交互 A/B 計測（必ず `HOST=127.0.0.1` で外部ポーリングを遮断）:
   - `ctl-a` → `cand` → `ctl-b` の順で計測し、`cand` が両側の対照群を上回ることを確認。
   - `c1`（code/prose）、`c5` 同時、4K/32K/128K prefill、定常 TTFT、起動時間、ピーク統合メモリ（<= 112.0 GiB）を記録。
4. 必須品質ゲート:
   - `python3 tools/exact.py` PASS
   - `python3 tools/exact.py --pad 3000` PASS
   - `python3 tools/prompt_reuse.py 30000` PASS（最低 90% 以上）
   - `python3 tools/check_invariants.py` PASS
5. 全ゲート PASS 時のみハッシュ拘束 GPU レシートを発行:
   python3 tools/receipt.py record \
     --holder bench-verifier \
     --image "<image-tag>" \
     --exact-pass --exact-pad3000-pass --ab-pass \
     --mem-peak-gib <PEAK_GIB> \
     --prompt-reuse-pct <WORST_PCT> \
     --summary "<具体的な A/B 差分サマリ>"
   python3 tools/receipt.py verify
6. 後片付け（成否にかかわらず必ず実行）:
   - 検証コンテナを停止・削除
   - `python3 tools/gpu_lock.py release --holder bench-verifier`
7. 結果（A/B 数値・SHA 一致・レシートパス）をボードへ報告し、`@coordinator` に引き継ぐ。
```

---

## 4. 現在のベースラインと次の高コスト最適化レバー候補

### 現行ベースライン（コミット `089f211` / レシート `978876ff8744d051`）
- **`bench_tf.py`（greedy, thinking OFF, 別プロンプト並列）**:
  - `c1`: **55.36 tok/s**（prose 48.98 / code 81.79）— 帯域理論上限比 86–97%
  - `c5`: **110.72 tok/s** — 帯域理論上限比 93.5%
  - Prefill 4K / 32K / 128K: **2,677 / 2,666 / 2,532 tok/s** — BF16 50% 実用上限比 **66.6–68.5%**（最大の伸び代）
  - 定常 TTFT: **103.8 ms** / 起動時間: **42.86 s** / ピークメモリ: **85.57 GiB**
- **`tools/bench_streams.py`（同一プロンプト並列, thinking ON）**:
  - prose 1/3/5: **57.2 / 155.9 / 231.4 tok/s**、code 1/3/5: **49.8 / 134.5 / 202.6 tok/s**
  - Prefill 4K / 31.5K / 126K: **2,518 / 2,682 / 2,485 tok/s**、200K needle: **84.87 s**

### 次に狙うべき高コスト最適化レバー候補（優先順）

| 優先度 | レバー候補 | 主対象 | 狙う指標 | 理論根拠・見込み |
| :---: | --- | --- | --- | --- |
| **1** | **Prefill NVFP4 MoE 小バッチタイル最適化** | `nvfp4_experts.cu`, `cuda_moe.zig` | 4K / 32K / 128K prefill | 8192 行でも 512 experts × top-10 により平均 ~160 行/expert。`nvcc --ptxas-options=-v` でレジスタスピル・SM 12.1 占有率を解析し、小 M 帯向けタイル／ディスパッチを最適化（残余 ~30% ギャップの最大要因）。 |
| **2** | **GDN（36 層線形アテンション）Prefill チャンク並列化／融合** | `flashnext_gdn*.cu`, `cuda_gdn.zig` | 32K / 128K / 200K prefill | 全 48 層中 36 層が GDN。チャンク内状態更新・`in_proj`/`out_proj` 周辺のメモリ往復とローンチ回数を削減。 |
| **3** | **QSA スパースインデクサ & Top-2048 選択カーネル最適化** | `flashnext_qsa*.cu`, `flashnext_attn.cu` | 128K / 200K prefill・長文 TTFT | 32K → 128K で prefill が 2,666 → 2,532 tok/s（−5%）へ落ちる要因をプロファイルし、圧縮インデクサ計算と選択のオーバーヘッドを削減。 |
| **4** | **ドラフトヘッド語彙剪定（79,591 → 32K/48K 検証）** | `cuda_mtp_load.zig`, `cuda_mtp_ops.zig` | `c1` / `c5` decode・TTFT | ドラフト 1 ステップあたり ~0.127 GB → ~0.05–0.08 GB へ削減（理論上 +3–5% prose / +8–12% code、`exact.py` ビット一致は維持されるため品質劣化なし、採択率とのトレードオフを A/B で厳密判定）。 |

---

## 5. マージ・終了前チェックリスト（`coordinator` 用）

- [ ] `author-*` の成果をメイン作業ツリーへマージ済み（`swarm kill` は必ずマージ後に実行。`kill` すると `share` worktree が削除される）
- [ ] `python3 tools/check_invariants.py` および `python3 tools/test_hooks_harness.py` が全 PASS
- [ ] `python3 tools/receipt.py verify --staged` が PASS（`patches/*.patch` または `scripts/config.sh` 変更時）
- [ ] `python3 tools/gpu_lock.py status` で `locked: false` を確認、`docker ps` に残留検証コンテナがないことを確認
- [ ] `git commit`（Stage-1 / Stage-2 フック自動通過）および必要に応じ `README.md` / `work/opt/PLAN.md` を更新
