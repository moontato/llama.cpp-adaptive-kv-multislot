# Unified multi-slot adaptive KV: test instructions

This experimental patch is for `RaymondHuang210129/llama.cpp-adaptive-kv-streaming`, branch `feature/kv-stream-phase-arena`, base commit `d7680c67b89f7cd4e3069b4bc2f26747aa588e3d`. It does not apply to `feature/adaptive-kv-stream-v2`. All executable validation belongs on the target machine under `/mnt/ramdisk`. The authoring environment has neither that mount nor CUDA, so revision 8 has been source reviewed but has not been compiled or executed there.

Revision 7 built in 32.7 seconds after restoring nine unchanged files' timestamps. All 64 required short-model comparisons were bit-identical. Pressure still failed 38 of 64 comparisons against the single-span FP32 reference, with maximum logit error 10.3782, although all 64 greedy token IDs agreed. Server checks and the CPU oracle validation passed. CTest failed only two assertions that still compared the revised FP32 resident path bit-for-bit against stock FP16 attention. Revision 8 corrects those assertions to use resident FP32 attention and keeps exact equality.

The pressure diagnostics found exact resident-prefix and safely inspected streamed-page K/V bytes in all checked layers. On selected actual-input prefill rows, streamed attention and the resident FP32 reference were similarly close to the double-precision CPU oracle, while stock FP16 attention was substantially further away. One- and two-query attention differences against ordinary CUDA were around 1e-6. These observations do not establish that every cache operation is correct. They suggest that changing the attention span partition could introduce small rounding differences that amplify through this quantized hybrid model. That explanation is still a hypothesis.

Revision 8 tests that hypothesis with a new GPU-resident reference. Each streamed context records only its attention arithmetic schedule: operation shapes/types/parameters, MMA or vector selection, query tile size, partial count, and ordered KV token spans. A separate context, with the arena disabled, maintains its own entirely GPU-resident KV and recurrent state, processes the same teacher-forced token histories and slot schedule, and replays those boundaries. It reads no recorded K/V values, queries, activations or logits. It bypasses the host KV cache, transfer ring, residency tracking, prefetch and dirty-row updates. The reference shares the CUDA partial-attention arithmetic and merge with streaming, so it tests storage and cache scheduling rather than independently proving that arithmetic.

All 64 whole-model cache comparisons remain required, with the unchanged max logit error <= 0.15 and RMSE <= 0.02. They now use `partition_resident_*` CSV names. Revision 7's single-span FP32 reference remains as 39 `fp32_resident_vs_streamed_*` diagnostics, alongside 39 stock CUDA diagnostics; those have `required=0`. They cannot be used to claim a pressure-model pass against single-span arithmetic. If the partition reference fails, the run still fails. If it passes while single-span results disagree, the evidence supports an arithmetic-partition explanation for that disagreement.

The runner requires four complete plans, checks recorded/replayed call counts and order, and rejects missing operations or inconsistent geometry. Backend replay validates a shape/type/parameter hash and contiguous, nonoverlapping spans covering the entire KV extent. Pressure must include split MMA prefill and split vector attention in the two-slot context. CUDA graphs are disabled in the model integration executable so every operation is recorded and replayed; these model results do not validate graph replay or serving performance. Normal server and CUDA unit checks remain separate. Record/replay flags are internal test controls, unset for normal serving, and the runner removes inherited test flags.

The attention diagnostic also compares the entire output tensor, for each checked operation, with an independently uploaded GPU-resident cache under the same span partition. This keeps Q, K, V and the mask identical at the comparison boundary and catches cache errors without downstream amplification. Its max absolute error and RMSE must both be <= 0.0005, the existing CUDA-attention tolerance. Exact resident-prefix and safe streamed-page byte checks must pass. `kv_stream_fp32_pair` additionally records the full-tensor difference against the single-span FP32 reference, to measure partition sensitivity directly.

Independent mathematics checks are retained. The CPU oracle uses double-precision dot products, global softmax and weighted value sums for representative actual-input rows. The model has 64 hybrid blocks but only 16 KV-attention layers; the checked first, middle and last cache-layer indices are 0, 8 and 15. The oracle samples first/middle/last queries across representative GQA heads plus the largest stock-versus-streamed difference, at most 13 query/head pairs per operation. `f16_inputs` rounds K/V and query inputs as MMA does, including rounding Q before multiplication by the rounded scale. `quantized_inputs` uses CPU-dequantized F32 K/V and original F32 queries. Neither mode emulates CUDA probability rounding or accumulation order. Streamed, stock and single-span resident FP32 outputs are compared with the oracle.

The existing constant-value Q5_1 tests cover 3K and 32K spans with forced spill and an independent analytic answer. The runner always validates the CPU oracle on the 3K known-answer case and requires streamed and single-span FP32 errors <= 0.0005 there and on sampled first-cache-layer FP16-input model rows. Middle/final-layer oracle errors are recorded and must be finite; no new uncalibrated threshold is imposed on their absolute scale. The oracle supports Q5_1, F16/F32 KV, one sequence dimension and attention without sinks, ALiBi or softcap. Diagnostics are bounded to KV extents <= 16,384 and run once per query-count category in each selected layer/runtime. Sampled oracle checks are not an exhaustive numerical proof.

Keep the 128 MiB pressure arena and `--skip-long`. The next result must come from the target machine. This revision changes the validation reference and diagnostics; it makes no further change to production attention precision beyond revision 7. A passing short/pressure report still cannot establish 250K-per-slot support.

## Run the current validation

Unpack the supplied bundle. It contains this guide, `kv-stream-multislot.patch`, and a standalone Python runner. Install Git, Python 3.9+, CMake 3.21+, a compatible C++ compiler, CUDA toolkit, and the NVIDIA driver. `cmake`, `ctest`, `git`, `nvcc`, and `nvidia-smi` must be on PATH. The runner downloads the pinned source from GitHub, applies the patch, builds the server and selected tests, and runs them. No Python packages are required.

Use your existing mounted 24 GiB ramdisk at `/mnt/ramdisk`. If you need to create it first on Linux:

```bash
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=24G,mode=1777 tmpfs /mnt/ramdisk
```

From the unpacked bundle, run:

```bash
python3 run_kv_multislot_tests.py \
  --model /absolute/path/to/your-model.gguf \
  --arena-mib 4096 \
  --pressure-arena-mib 128 \
  --skip-long
```

Set `--arena-mib` to the size that works for your current single-slot setup. Defaults are Q5_1 K/V, RTX 4080 CUDA architecture 89, 512-token logical batches, 128-token micro-batches, and two build jobs. The optional long diagnostic uses two 250,000-token histories and independent reference histories, and can take hours. `--model-timeout-hours 24` is the timeout per model test, not an estimated runtime.

Keep `--skip-long` for this investigation. Its successful report says `PASS_SHORT_ONLY`, so it cannot establish 250K-per-slot support. The runner performs the short model checks with your arena, plus approximately 3K tokens per slot with a 128 MiB arena. The smaller arena is a pressure test, not a recommended serving setting. `--pressure-arena-mib` can override that size if the compute workspace requires more room; the runner requires a trace showing two-query generation beyond the uniform resident capacity. `--ubatch 256` can test your usual micro-batch size; rerun with a fresh invocation for each value. `--gpu-layers` and `--cuda-arch` can be overridden. GPU tests use CUDA device 0; use `CUDA_VISIBLE_DEVICES` if necessary. Do not run other GPU-heavy work at the same time.

## Resume the existing build

Extract the revision 8 bundle into `/mnt/ramdisk`. From its directory, run the new runner against the existing workspace:

```bash
python3 run_kv_multislot_tests.py \
  --resume /mnt/ramdisk/kv-multislot-20261004-090100-122194 \
  --model /absolute/path/to/your-model.gguf \
  --arena-mib 4096 \
  --pressure-arena-mib 128 \
  --skip-long
```

The runner checks the base commit, safely replaces its previously applied patch with the included revision, and resumes the existing CMake build. It retains unchanged compiled objects and restores timestamps only when patched file content and mode remain identical. `resume-timestamps.json` records the decision. Revision 8 changes the main CUDA attention source, the CUDA attention unit source and the model integration source. Attention template headers are unchanged from revision 7, avoiding the broad template rebuild. Keep the original CUDA architecture and build settings. If the workspace was removed, omit `--resume` for a fresh build.

The long diagnostic remains available without `--skip-long`, and is skipped after any prerequisite runtime failure unless `--long-on-failure` is supplied. Its existing serial-versus-parallel reference also includes ordinary batching differences, which the returned short tests demonstrated can be substantial; it needs a revised reference before its numerical result can establish long-context streaming correctness. Do not spend another long run on this revision while the pressure failure is unresolved. While a command is running, the terminal prints `WAIT <stage>` about once a minute, including the latest prompt progress where available. Model logs report processed tokens, elapsed time and average tokens/second about every 30 seconds between decode calls. A single decode call can take longer than this interval; the terminal heartbeat continues and does not by itself prove forward progress.

Each resumed run writes a separate `results-<timestamp>-<PID>` directory and matching archive inside that workspace. The original logs and archive remain available. Source edits that prevent safe patch replacement cause the runner to stop and report the conflict.

## Send results

At completion the runner prints one path like:

```text
/mnt/ramdisk/kv-multislot-YYYYMMDD-HHMMSS-PID/results.tar.gz
```

For a resumed run, the filename is `results-<timestamp>-<PID>.tar.gz`. Attach the archive printed by the latest run to the conversation. It contains the exact patch, settings, source commit, command logs and exit codes, CTest XML, model comparison CSVs, HTTP responses, slot overlap observations, server metrics, periodic GPU/host memory samples, and `summary.json`. The diagnostic always runs and also includes `model_attention_diagnostic.log`, `model-attention-diagnostic.csv`, `model-attention-diagnostic-proof.json`, `model-attention-diagnostic-attention-checks.json`, and `model-attention-diagnostic-attention-oracles.json`. The known-answer reference check is captured in `attention_oracle_validation.log` and `attention-oracle-validation.json`. The four `model-*.csv.*.plan` files per model run contain arithmetic schedules. `model-*-reference-proof.json` records complete replay coverage, split MMA/vector use and required comparison coverage. The diagnostic also includes `model-attention-diagnostic-partition-checks.json` and `model-attention-diagnostic-fp32-pairs.json`; `resume-timestamps.json` records the timestamp repair when a previous patch was replaced. It contains no model weights. If prerequisites, compilation, or any test fail, the same archive captures the failure. Independent runtime checks continue after a check fails; fatal build or resource errors stop dependent work.

## What is tested

| Check | Evidence |
| --- | --- |
| Configuration and phase planning | All seven existing KV-stream CTest targets, plus partial-slot generation and unified-cache validation cases |
| Q5_1 sequence isolation | Two queries with disjoint interleaved histories, unequal lengths, holes, changing dirty rows, and a forced streaming ring; compared to resident CUDA attention |
| Long-span prefill precision | Q5_1 with a 2,816- or 32,512-token resident prefix, forced spill, and constant values; compared to the independent analytic answer |
| Short model correctness | Streamed serial and two-slot logits against independent GPU-resident caches replaying identical attention spans; single-span FP32, stock and serial/parallel differences recorded separately |
| Model streaming pressure | Approximately 3K tokens per slot with a 128 MiB arena, complete GPU-resident plan replay, split MMA/vector calls and two-query generation beyond resident capacity |
| Attention diagnostic on every run | Full-tensor same-partition resident attention, single-span FP32 differences, double-precision samples in cache layers 0/8/15, and exact resident-prefix/safe last-page byte checks |
| Reference validation on every run | Existing constant-value Q5_1 test with the oracle enabled; streamed and GPU-resident FP32 errors checked against its unchanged analytic threshold |
| Slot scheduling | Both active, only slot 1 active, a mixed prompt/generation batch, slot 0 replacement, and return to both active |
| State restore | Short per-sequence snapshot after generation, removal and restore, resumed logits and other-slot preservation; HTTP save/erase/restore |
| Server concurrency | Serial requests, concurrent unequal requests, observed overlap, and a replacement request while the longer slot remains active |
| Invalid combinations | Non-unified parallel KV and parallel speculative decoding must fail with the intended error |
| Optional long diagnostic | Two distinct 250,000-token histories in one shared context, compared against independent streamed histories; batching differences confound numerical attribution |

The long test avoids a resident GPU reference and large state snapshots; short tests cover those paths. Prompts are synthetic repeated text, so this suite validates cache mechanics and numerical consistency, not long-context answer quality. CUDA attention comparisons require maximum absolute error <= 0.0005. Whole-model comparisons use max absolute logit error <= 0.15 and RMSE <= 0.02; these initial tolerances have not been calibrated on this GPU/model. Both metrics and greedy token IDs are recorded. In short-run CSVs, `required=1` marks same-partition GPU-resident versus streamed comparisons that affect the exit status. `required=0` marks single-span FP32 and stock-precision comparisons and serial-versus-parallel diagnostics; each includes an arithmetic or batching difference beyond cache storage. The long run retains serial-versus-parallel comparisons as required checks because a resident GPU reference is too large. HTTP token equality is diagnostic because close logits can change greedy choices and then subsequent histories. All logit comparisons use identical token histories.

## Resource behavior

All execution, build output, temporary files, compilation caches, slot snapshots, reports, and result archives live in a runner workspace under `/mnt/ramdisk`. The runner requires a mount at that exact path and never falls back to disk. The existing model is linked into the workspace; all GGUF shards are linked when applicable. The original files are read as input and are not copied or modified.

A periodic guard stops owned processes at 23 GiB of workspace data or below 1 GiB free space, preserving room for reports within the 24 GiB budget. This is a sampled guard, not a filesystem quota: the 24 GiB tmpfs mount supplies the hard limit. Build jobs default to two to reduce peaks. Source and binaries remain in the ramdisk after the run, for inspection and manual use. Only the results directory is archived. Move the archive somewhere durable before unmounting or rebooting.

The ramdisk consumes system RAM in addition to the model and pinned host KV cache. Two 250K histories require substantially more host KV capacity than one. The monitor records host available memory and GPU memory, but does not impose a host-memory quota. A low-memory failure is reported, not treated as a passing test.

## Use the patched server after validation

The build is at the printed workspace's `build/bin/llama-server`. For example:

```bash
/mnt/ramdisk/kv-multislot-YYYYMMDD-HHMMSS-PID/build/bin/llama-server \
  --model /absolute/path/to/your-model.gguf \
  --parallel 2 --kv-unified \
  --ctx-size 524288 --kv-unified-per-slot 262144 \
  -fa on -ctk q5_1 -ctv q5_1 -ngl all -b 512 -ub 128 \
  --kv-stream-arena-mib 2048
```

The arena is shared dynamically by occupied cache rows, without a fixed half per slot. Aggregate context capacity is separate from the per-slot limit. Unified KV is a logical shared cache, not CUDA managed memory. Parallel speculative decoding is rejected. The multi-query attention path already exists, but its throughput and latency may differ from TG1's cross-layer lookahead path. Model validation targets two slots; larger counts are experimental.
