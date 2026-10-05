# Staged long-context validation for the passing V8 build

The returned V8 run passed all seven CTest targets, all 192 required model comparisons across short/pressure/diagnostic runs, and all ten full attention-output checks. Every required model comparison and every full-output check was bit-identical. The pressure replay covered 2,560 attention operations, including split MMA prefill and two-slot split vector attention, with complete recorded/replayed coverage. Server concurrency, slot replacement and short state restore also passed. The older single-span comparisons still disagree under pressure; this result validates cache mechanics under matching arithmetic, not equivalence with different attention partitions.

This bundle adds a standalone long server test. It reuses the existing V8 binary without fetching source, replacing the patch or rebuilding CUDA. Production source is unchanged. The script was source reviewed but has not been executed in the authoring environment; all execution and syntax validation belongs on your target machine under `/mnt/ramdisk`.

## Run

Extract this bundle into `/mnt/ramdisk`, then:

```bash
cd /mnt/ramdisk/kv-stream-multislot-long-v1
python3 run_kv_multislot_long_tests.py \
  --resume /mnt/ramdisk/kv-multislot-20261004-090100-122194 \
  --model /mnt/ssd/llamacpp_models/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf \
  --arena-mib 4096
```

The default largest prompt is 125,000 tokens in slot 0 and 124,743 in slot 1: approximately 250K aggregate. For approximately 250K in each slot, add:

```text
--tokens-per-slot 250000
```

That option uses about 500K aggregate KV capacity. It requires more host RAM and substantially more processing time. The arena remains shared at 4 GiB. Context capacity is increased to fit both requested histories plus generation space; an explicit per-slot limit keeps either request within the model's training context.

Defaults are Q5_1 K/V, 512-token batches, 128-token micro-batches, all GPU layers and two slots with unified KV. The script requires a passing V8 report for the same model, arena, GPU layers and micro-batch size. To change those settings, run the V8 short validation with them first. A context below 8,192 or above 250,000 tokens per slot is rejected.

Git, Python 3.9+ and the NVIDIA driver are required. No Python packages or CUDA rebuild are needed. The imported `run_kv_multislot_tests.py` is the unchanged V8 helper; run the new `run_kv_multislot_long_tests.py` entry point for this test. Do not run the old long-model mode: its serial/parallel numerical comparison remains confounded by batching differences.

## Stages and evidence

| Stage | Slot 0 prompt | Slot 1 prompt | Approximate aggregate |
| --- | ---: | ---: | ---: |
| First | 8,192 | 7,935 | 16K |
| Second | 32,768 | 32,511 | 65K |
| Final, default | 125,000 | 124,743 | 250K |
| Final with override | 250,000 | 249,743 | 500K |

Each stage starts a fresh server so its context allocation, GPU/host memory use and timings are recorded separately. It first fills the two distinct histories with one generated token each. It then submits two cached continuations together, generating 16 and 64 tokens. Filling both histories first avoids a long prefill hiding simultaneous generation.

After the shorter continuation completes, slot 0 is replaced with a new 779-token history while the long slot 1 is still processing. A final cached continuation from slot 1 checks that its long history remains usable after the other slot's replacement. Prompt sizes, response routing, generation counts, lack of truncation and reuse of the expected prefix are required. Every reported generated-token log probability must be present and finite. Both initial concurrency and concurrency during replacement must be observed through `/slots`.

The script captures the server log interval from concurrent request submission to the shorter continuation's completion. It requires a two-query decode trace at the full requested aggregate length, with both arena phases recorded. The final stage additionally requires a two-query working set exceeding uniform resident capacity, proving that the arena actually streams KV at the target length. Earlier stages may fit in VRAM.

`PASS_LONG_SERVER` means these functional and resource checks passed. It is separate from V8's numerical cache validation. This long test does not compare full logits against another implementation, establish long-context answer quality, or validate every attention output at the final length. Reported finite token probabilities are a health check, not a full-vocabulary numerical proof. Synthetic repeated prompts make exact lengths reproducible and are not a quality benchmark.

## Progress, interruption and results

The stages stop after the first failure. A minute heartbeat prints the stage, operation, elapsed time and latest slot token counters. Each completed request prints prompt length, cached tokens, generated tokens and elapsed seconds. Long prefill can take considerable time; increasing context can increase attention work faster than token count. `--stage-timeout-hours` defaults to eight hours per stage across all its requests, rather than eight hours per request. Server startup has a separate 15-minute limit. A heartbeat can continue while a decode is slow; repeated counters alone do not prove forward progress.

Ctrl-C stops owned server processes and writes an `INTERRUPTED` archive containing progress so far. Failures, timeout or resource-guard stops also produce an archive. At completion, attach the exact `results-<timestamp>-<PID>.tar.gz` printed by this invocation. It includes stage commands and logs, original prompt token arrays, HTTP responses and timings, slot observations, arena trace proof, server metrics, GPU/host samples, source/binary identity, scripts and summaries. It includes no weights or source checkout.

All generated files, temporary data, compilation caches and execution stay in the existing workspace under `/mnt/ramdisk`. The script requires a tmpfs mount with a hard limit of at most 24 GiB. It retains V8's sampled workspace guard at 23 GiB and 1 GiB free to preserve report space. The original model and its existing shard links are read without copying or modification. It verifies the pinned base, V8 patch and patched-file hashes before starting long work. GPU/host memory is recorded separately because pinned KV and model memory are not ramdisk file usage.

The existing V8 build and reports remain available. No long state snapshot is created, so large serialization files do not consume the ramdisk. Source checks are read-only. The default server execution uses its usual CUDA graph settings; the script clears the model-test graph-disable flag inherited from the environment.
