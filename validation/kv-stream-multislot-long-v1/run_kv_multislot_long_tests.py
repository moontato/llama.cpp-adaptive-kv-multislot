#!/usr/bin/env python3
"""Exercise the passing V8 server at increasing context lengths under /mnt/ramdisk."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tarfile
import threading
import time
import traceback
import urllib.error

sys.dont_write_bytecode = True
from run_kv_multislot_tests import BASE, GIB, ROOT, Runner, http, launch_server, server_command, stop, unused_port, write_json

PATCH_SHA256 = "9b36afe5b0083dbcb87751ebd75ba2c39a877063758f17a1ca2ed3746f0ecfe3"
SEEDS = (
    "Amber notebook. The observatory records stars and quiet weather. ",
    "Blue notebook. The workshop records engines and loud machines. ",
    "Green notebook. A new visitor asks about rivers, maps and bridges. ",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validated_build(runner, model, patch):
    workspace = runner.workspace
    if sha256(patch) != PATCH_SHA256 or sha256(workspace / "applied-kv-stream.patch") != PATCH_SHA256:
        raise RuntimeError("require the unchanged V8 patch and existing V8 build")
    source = workspace / "source"
    runner.run("source_base", ["git", "-C", source, "rev-parse", "HEAD"], timeout=30)
    if (runner.results / "source_base.log").read_text().strip() != BASE:
        raise RuntimeError("source is not at the pinned V8 base")
    runner.run("source_patch", ["git", "-C", source, "apply", "--reverse", "--check", patch], timeout=30)
    expected = json.loads(Path(__file__).with_name("manifest.json").read_text())["v8_patched_file_sha256"]
    if any(sha256(source / name) != digest for name, digest in expected.items()):
        raise RuntimeError("a patched source file differs from V8; rerun short validation for the current source")
    runner.run("source_changed_files", ["git", "-C", source, "diff", "--name-only", "HEAD"], timeout=30)
    if not set((runner.results / "source_changed_files.log").read_text().splitlines()).issubset(expected):
        raise RuntimeError("additional tracked source changes are present beyond V8")
    candidates = []
    for summary_path in sorted(workspace.glob("results*/summary.json"), reverse=True):
        manifest_path = summary_path.with_name("manifest.json")
        if not manifest_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if (summary.get("status") == "PASS_SHORT_ONLY" and not summary.get("failed_checks") and
                manifest.get("patch_sha256") == PATCH_SHA256 and manifest.get("model_name") == model.name and
                manifest.get("model_bytes") == model.stat().st_size and
                all(str(manifest.get("settings", {}).get(key)) == str(getattr(runner.args, key))
                    for key in ("arena_mib", "gpu_layers", "ubatch"))):
            candidates.append(summary_path)
    if not candidates:
        raise RuntimeError("no passing V8 short/pressure report for this model, arena, GPU layers and ubatch")
    model_link = workspace / "model" / model.name
    if not model_link.is_symlink() or model_link.resolve(strict=True) != model:
        raise RuntimeError("existing model link does not match --model")
    binary = (workspace / "build" / "bin" / "llama-server").resolve(strict=True)
    if not binary.is_relative_to(workspace) or not os.access(binary, os.X_OK):
        raise RuntimeError("server binary must be executable inside the existing ramdisk workspace")
    write_json(runner.results / "validated-build.json", {
        "passing_report": str(candidates[0]), "base": BASE, "patch_sha256": PATCH_SHA256,
        "binary": str(binary), "binary_sha256": sha256(binary), "source_modified": False,
    })
    return binary, model_link


def make_prompt(base, count, seed):
    ordinary = http(base, "/tokenize", {"content": seed, "add_special": False})["tokens"]
    initial = http(base, "/tokenize", {"content": seed, "add_special": True})["tokens"]
    if not ordinary or not initial or any(type(token) is not int for token in ordinary + initial):
        raise RuntimeError("invalid seed tokenization")
    tail = max(0, count - len(initial))
    return (initial + ordinary*((tail + len(ordinary) - 1)//len(ordinary)))[:count]


class LongStage:
    def __init__(self, runner, binary, model, count):
        self.runner = runner
        self.name = "server_long_" + str(count)
        self.count = count
        self.deadline = time.monotonic() + runner.args.stage_timeout_hours*3600
        self.next_heartbeat = time.monotonic() + 60
        self.started = time.monotonic()
        self.last_slots = []
        self.overlap = False
        self.data = {"tokens_per_slot": [count, count - 257], "numerical_equivalence_test": False}
        cap = ((count + 512 + 255)//256)*256
        command = server_command(runner, binary, model, 2, unused_port(), extra=("--kv-unified-per-slot", str(cap)))
        command[command.index("--ctx-size") + 1] = str(2*cap)
        self.data["aggregate_context_capacity"] = 2*cap
        self.process, self.stage, self.base = launch_server(runner, self.name, command)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.samples = (runner.results / (self.name + "-slots.jsonl")).open("w")

    def check_alive(self):
        if self.runner.fault:
            raise RuntimeError(self.runner.fault)
        if self.process.poll() is not None:
            raise RuntimeError(self.name + " exited; see its log")
        if time.monotonic() >= self.deadline:
            raise TimeoutError(self.name + " exceeded --stage-timeout-hours")

    def sample(self, label):
        self.check_alive()
        try:
            slots = http(self.base, "/slots", timeout=3)
            if not isinstance(slots, list) or {slot.get("id") for slot in slots} != {0, 1}:
                raise RuntimeError("invalid two-slot inventory")
            self.last_slots = slots
            overlap = sum(bool(slot.get("is_processing")) for slot in slots) == 2
            self.overlap |= overlap
            if label == "slot_reuse_with_long_slot1":
                self.data["reuse_overlap_observed"] = self.data.get("reuse_overlap_observed", False) or overlap
            self.samples.write(json.dumps({"time": time.time(), "step": label, "slots": slots}) + "\n")
            self.samples.flush()
        except (urllib.error.URLError, TimeoutError) as error:
            self.samples.write(json.dumps({"time": time.time(), "step": label, "poll_error": str(error)}) + "\n")
        if time.monotonic() >= self.next_heartbeat:
            progress = [{"slot": slot["id"], "processing": slot.get("is_processing"),
                         "prompt_tokens": slot.get("n_prompt_tokens"),
                         "cached": slot.get("n_prompt_tokens_cache"),
                         "processed": slot.get("n_prompt_tokens_processed"),
                         "generation": slot.get("next_token")} for slot in self.last_slots]
            print("WAIT", self.name, label, "elapsed=%.1fmin" % ((time.monotonic() - self.started)/60),
                  json.dumps(progress), flush=True)
            self.next_heartbeat = time.monotonic() + 60

    def wait(self, future, label):
        while not future.done():
            self.sample(label)
            concurrent.futures.wait([future], timeout=0.05 if not self.overlap and label == "concurrent_cached_generation" else 1)
        self.check_alive()
        return future.result()

    def completion(self, label, prompt, slot, count, barrier=None, require_cached=False):
        if barrier is not None:
            barrier.wait(timeout=30)
        self.check_alive()
        started = time.monotonic()
        result = http(self.base, "/completion", {
            "prompt": prompt, "id_slot": slot, "n_predict": count, "temperature": 0, "seed": 123,
            "ignore_eos": True, "cache_prompt": True, "return_tokens": True, "n_probs": 5,
            "post_sampling_probs": False,
            "response_fields": ["id_slot", "content", "tokens", "tokens_predicted", "tokens_evaluated",
                                "tokens_cached", "truncated", "stop_type", "timings", "completion_probabilities"],
        }, timeout=max(1, self.deadline - time.monotonic()))
        write_json(self.runner.results / (self.name + "-" + label + ".json"), result)
        if (result.get("error") or result.get("id_slot") != slot or result.get("truncated") is not False or
                result.get("tokens_evaluated") != len(prompt) or result.get("tokens_predicted") != count or
                len(result.get("tokens", [])) != count or any(type(token) is not int for token in result.get("tokens", []))):
            raise RuntimeError(label + " returned truncated, misrouted or incomplete output; see its response JSON")
        probabilities = result.get("completion_probabilities", [])
        if len(probabilities) != count:
            raise RuntimeError(label + " did not return probabilities for every generated token")
        for token, row in zip(result["tokens"], probabilities):
            top = row.get("top_logprobs", [])
            values = [row.get("logprob"), *(item.get("logprob") for item in top)]
            if (row.get("id") != token or not top or any(type(value) not in (int, float) or
                    not math.isfinite(value) or value > 0 for value in values)):
                raise RuntimeError(label + " has missing or non-finite reported token probabilities")
        cached = result.get("timings", {}).get("cache_n", 0)
        if require_cached and cached < len(prompt) - 2:
            raise RuntimeError(label + " did not reuse the expected long history; see cache_n in its response")
        print("COMPLETED", self.name, label, "prompt=%d cached=%d generated=%d elapsed=%.1fs" %
              (len(prompt), cached, count, time.monotonic() - started), flush=True)
        return result

    def run(self):
        prompts = [make_prompt(self.base, self.count - 257*slot, SEEDS[slot]) for slot in (0, 1)]
        replacement = make_prompt(self.base, 779, SEEDS[2])
        write_json(self.runner.results / (self.name + "-prompts.json"), prompts)
        # Fill both histories first, so long prefill cannot hide simultaneous generation.
        histories = []
        for slot in (0, 1):
            result = self.wait(self.executor.submit(self.completion, "warm_" + str(slot), prompts[slot], slot, 1),
                               "warm_" + str(slot))
            histories.append(prompts[slot] + result["tokens"])
        log = self.runner.results / (self.name + ".log")
        parallel_offset = log.stat().st_size
        barrier = threading.Barrier(2)
        first = self.executor.submit(self.completion, "parallel_0", histories[0], 0, 16, barrier, True)
        second = self.executor.submit(self.completion, "parallel_1", histories[1], 1, 64, barrier, True)
        first_result = self.wait(first, "concurrent_cached_generation")
        parallel_end = log.stat().st_size
        if second.done():
            raise RuntimeError("slot 1 completed before slot 0 could be reused; see timing evidence")
        self.data["slot1_active_at_slot0_reuse"] = True
        reused = self.executor.submit(self.completion, "replacement_0", replacement, 0, 12)
        second_result = self.wait(second, "slot_reuse_with_long_slot1")
        reused_result = self.wait(reused, "replacement_0")
        preserved_history = histories[1] + second_result["tokens"]
        followup = self.wait(self.executor.submit(self.completion, "single_active_1", preserved_history, 1, 8,
                                                 require_cached=True), "single_active_1")
        self.data.update(overlap_observed=self.overlap, generated_counts=[len(first_result["tokens"]),
                         len(second_result["tokens"]), len(reused_result["tokens"]), len(followup["tokens"])])
        with log.open("rb") as source:
            prefix = source.read(parallel_offset).decode(errors="replace")
            source.seek(parallel_offset)
            tail = source.read(parallel_end - parallel_offset).decode(errors="replace")
        phases = re.findall(r"activated (decode|prefill) phase", prefix)
        phase = phases[-1] if phases else None
        traces = []
        for line in tail.splitlines():
            changed = re.search(r"activated (decode|prefill) phase", line)
            if changed:
                phase = changed[1]
            sample = re.search(r"kv_stream_adapt: queries (\d+), active (\d+), resident (\d+), ring (\d+)", line)
            if sample and phase == "decode":
                traces.append(tuple(map(int, sample.groups())))
        total = sum(map(len, prompts))
        at_length = [sample for sample in traces if sample[0] == 2 and sample[1] >= total]
        streamed = [sample for sample in at_length if (sample[1] + 255)//256 > sample[2]]
        self.data.update(two_query_at_requested_length=bool(at_length), two_query_nonresident_working_set=bool(streamed),
                         two_query_samples=at_length[:32], prompt_tokens_total=total,
                         concurrent_decode_log_bytes=[parallel_offset, parallel_end])
        self.data["arena_phases_logged"] = {phase: "activated " + phase + " phase" in log.read_text(errors="replace")
                                            for phase in ("decode", "prefill")}
        if (not self.overlap or not self.data.get("reuse_overlap_observed") or not at_length or
                not all(self.data["arena_phases_logged"].values())):
            raise RuntimeError("stage lacks observed two-slot overlap, full-length two-query attention or both arena phases")
        if self.count == self.runner.args.tokens_per_slot and not streamed:
            raise RuntimeError("final stage did not force two-query streaming beyond resident capacity; reduce --arena-mib after validating that arena with V8")
        (self.runner.results / (self.name + "-metrics.txt")).write_text(http(self.base, "/metrics", timeout=30))

    def close(self):
        stop(self.process)
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.samples.close()
        self.runner.finish(self.process, self.stage, 15, accepted=(0, -signal.SIGTERM))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True, type=Path, help="existing passing V8 runner workspace")
    parser.add_argument("--model", required=True, type=Path, help="same original model used by the passing V8 run")
    parser.add_argument("--arena-mib", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--tokens-per-slot", type=int, default=125000, help="125000 tests about 250K total; 250000 tests about 250K per slot")
    parser.add_argument("--stage-timeout-hours", type=float, default=8, help="timeout for each stage, including all its requests")
    args = parser.parse_args()
    workspace = args.resume.expanduser().resolve(strict=True)
    model = args.model.expanduser().resolve(strict=True)
    if not ROOT.is_dir() or not os.path.ismount(ROOT):
        parser.error("require an existing /mnt/ramdisk mount; no disk fallback")
    mounts = [line.split(" - ") for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    if not any(a.split()[4] == str(ROOT) and b.split()[0] == "tmpfs" for a, b in mounts):
        parser.error("/mnt/ramdisk must be tmpfs")
    if shutil.disk_usage(ROOT).total > 24*GIB:
        parser.error("require a tmpfs hard limit of at most 24 GiB")
    if workspace == ROOT.resolve() or not workspace.is_relative_to(ROOT.resolve()):
        parser.error("--resume must be a workspace inside /mnt/ramdisk")
    if any(not (workspace / name).resolve().is_relative_to(workspace) for name in ("source", "build")):
        parser.error("source and build must remain inside the ramdisk workspace")
    if not model.is_file() or not 8192 <= args.tokens_per_slot <= 250000 or args.arena_mib <= 0 or not 2 <= args.ubatch <= 512:
        parser.error("invalid model, tokens per slot (8192..250000), arena or ubatch")
    if not math.isfinite(args.stage_timeout_hours) or args.stage_timeout_hours <= 0:
        parser.error("stage timeout must be finite and positive")
    runner = Runner(workspace, args)
    runner.env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    status, error = "FAIL", None
    stage_results = []
    try:
        local_patch = Path(__file__).with_name("kv-stream-multislot.patch")
        for name in ("run_kv_multislot_long_tests.py", "run_kv_multislot_tests.py", "kv-stream-multislot.patch"):
            shutil.copyfile(Path(__file__).with_name(name), runner.results / name)
        shutil.copyfile(Path(__file__).with_name("manifest.json"), runner.results / "bundle-manifest.json")
        write_json(runner.results / "manifest.json", {"kind": "long-server-functional", "base": BASE,
                   "patch_sha256": PATCH_SHA256, "model_name": model.name, "model_bytes": model.stat().st_size,
                   "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                   "numerical_equivalence_test": False})
        runner.run("environment", ["nvidia-smi", "-q"], timeout=30)
        runner.run("python_syntax", [sys.executable, "-m", "py_compile", runner.results / "run_kv_multislot_long_tests.py"], timeout=30)
        binary, model_link = validated_build(runner, model, local_patch)
        for count in sorted({min(8192, args.tokens_per_slot), min(32768, args.tokens_per_slot), args.tokens_per_slot}):
            stage = LongStage(runner, binary, model_link, count)
            try:
                stage.run()
                stage.data["passed"] = True
                print("PASS", stage.name, flush=True)
            except BaseException as failure:
                stage.data.update(passed=False, error=str(failure) or type(failure).__name__)
                raise
            finally:
                stage_results.append(stage.data)
                write_json(runner.results / (stage.name + "-summary.json"), stage.data)
                stage.close()
        status = "PASS_LONG_SERVER"
    except BaseException as failure:
        error = str(failure) or type(failure).__name__
        if isinstance(failure, KeyboardInterrupt):
            status = "INTERRUPTED"
        (runner.results / "failure.txt").write_text(traceback.format_exc())
        print(status + ":", error, flush=True)
    finally:
        runner.close()
        write_json(runner.results / "summary.json", {"status": status, "error": error, "base": BASE,
                   "stages": stage_results, "numerical_equivalence_test": False,
                   "tokens_per_slot_requested": args.tokens_per_slot, "peak_workspace_bytes": runner.peak_workspace_bytes,
                   "workspace_limit_bytes": 24*GIB, "resource_guard_error": runner.fault})
        archive = workspace / (runner.results.name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(runner.results, arcname="results")
        print("\n" + status + "\nAttach this file to the conversation:\n" + str(archive), flush=True)
    return 0 if status == "PASS_LONG_SERVER" else 1


if __name__ == "__main__":
    sys.exit(main())
