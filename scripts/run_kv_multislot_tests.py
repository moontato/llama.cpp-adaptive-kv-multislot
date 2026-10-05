#!/usr/bin/env python3
"""Build and test the pinned adaptive-KV patch entirely under /mnt/ramdisk."""

import argparse
import csv
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.error
import urllib.request

REPOSITORY = "https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming.git"
BASE = "d7680c67b89f7cd4e3069b4bc2f26747aa588e3d"
ROOT = Path("/mnt/ramdisk")
GIB = 1024**3
TESTS = ["test-kv-stream-" + suffix for suffix in (
    "plan", "config", "softmax", "bench-config", "cuda-buffer", "cuda-attn", "cuda-set-rows")]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


class Runner:
    def __init__(self, workspace, args):
        self.workspace = workspace
        self.args = args
        self.results = workspace / (
            "results-" + time.strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid()) if args.resume else "results")
        self.results.mkdir()
        self.env = os.environ.copy()
        # Avoid inherited llama flags and speculative settings changing a test.
        for key in list(self.env):
            if key.startswith("LLAMA_ARG_"):
                del self.env[key]
        self.env.pop("GGML_CUDA_KV_STREAM_CHECK_ATTENTION", None)
        self.env.pop("GGML_CUDA_KV_STREAM_FP32_REFERENCE", None)
        self.env.pop("GGML_CUDA_KV_STREAM_RECORD_PLAN", None)
        self.env.pop("GGML_CUDA_KV_STREAM_REPLAY_PLAN", None)
        for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "CCACHE_DIR"):
            path = workspace / key.lower()
            path.mkdir(exist_ok=bool(args.resume))
            self.env[key] = str(path)
        self.env.update(PYTHONDONTWRITEBYTECODE="1", GIT_CONFIG_NOSYSTEM="1",
                        GIT_CONFIG_GLOBAL="/dev/null", PYTHONPYCACHEPREFIX=str(workspace / "pycache"),
                        LLAMA_KV_STREAM_TRACE="1")
        self.processes = []
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.fault = None
        self.stages = []
        self.peak_workspace_bytes = 0
        self.monitor = threading.Thread(target=self.sample, daemon=True)
        self.monitor.start()

    def sample(self):
        fields = "index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw"
        with (self.results / "resources.jsonl").open("w") as output:
            count = 0
            while not self.done.is_set():
                sample = {"time": time.time(), "free_workspace_bytes": shutil.disk_usage(self.workspace).free}
                try:
                    gpu = subprocess.run(["nvidia-smi", "--query-gpu=" + fields,
                                          "--format=csv,noheader,nounits"], env=self.env,
                                         cwd=self.workspace, capture_output=True, text=True, timeout=4)
                    sample["gpu_csv"] = gpu.stdout.strip()
                    sample["gpu_query_exit"] = gpu.returncode
                    sample["meminfo"] = {line.split(":")[0]: line.split(":")[1].strip()
                                         for line in Path("/proc/meminfo").read_text().splitlines()
                                         if line.startswith(("MemAvailable:", "SwapFree:"))}
                    if count % 6 == 0:
                        used = 0
                        for directory, _, names in os.walk(self.workspace, followlinks=False):
                            for name in names:
                                path = Path(directory) / name
                                try:
                                    if not path.is_symlink():
                                        used += path.stat().st_blocks * 512
                                except FileNotFoundError:
                                    pass
                        self.peak_workspace_bytes = max(self.peak_workspace_bytes, used)
                        sample["workspace_bytes"] = used
                        if used > 23 * GIB:
                            self.fault = "workspace reached 23 GiB; stopped to preserve result space"
                    if sample["free_workspace_bytes"] < GIB:
                        self.fault = "less than 1 GiB free on ramdisk; stopped to preserve result space"
                    if self.fault:
                        with self.lock:
                            running = list(self.processes)
                        for process in running:
                            stop(process)
                except Exception as error:
                    sample["monitor_error"] = str(error)
                output.write(json.dumps(sample) + "\n")
                output.flush()
                count += 1
                self.done.wait(5)

    def start(self, name, command):
        if self.fault:
            raise RuntimeError(self.fault)
        log = (self.results / (name + ".log")).open("wb")
        started = time.time()
        process = subprocess.Popen([str(item) for item in command], cwd=self.workspace, env=self.env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        with self.lock:
            self.processes.append(process)
        stage = {"name": name, "command": [str(item) for item in command], "started": started}
        self.stages.append(stage)
        write_json(self.results / "stages.json", self.stages)
        print("RUN", name, flush=True)
        return process, stage

    def finish(self, process, stage, timeout, accepted=(0,)):
        deadline = time.monotonic() + timeout
        next_report = time.monotonic() + 60
        while process.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                stop(process)
                stage["timeout"] = True
                break
            if now >= next_report:
                with (self.results / (stage["name"] + ".log")).open("rb") as log:
                    log.seek(0, os.SEEK_END)
                    log.seek(max(0, log.tell() - 8192))
                    lines = log.read().decode(errors="replace").splitlines()
                progress = [line for line in lines if "kv-multislot progress:" in line]
                latest = (progress or lines or ["no log output yet"])[-1]
                print("WAIT", stage["name"], "elapsed=%.1fmin" % ((time.time() - stage["started"])/60), latest[-500:], flush=True)
                next_report = now + 60
            try:
                process.wait(timeout=min(5, max(0.01, deadline - now)))
            except subprocess.TimeoutExpired:
                pass
        code = process.returncode
        stage.update(exit_code=code, elapsed_seconds=round(time.time() - stage["started"], 3))
        write_json(self.results / "stages.json", self.stages)
        print("RESULT", stage["name"], code, flush=True)
        if self.fault:
            raise RuntimeError(self.fault)
        if code not in accepted or stage.get("timeout"):
            raise RuntimeError(stage["name"] + " failed; see its log")
        return code

    def run(self, name, command, timeout=1800, accepted=(0,)):
        process, stage = self.start(name, command)
        return self.finish(process, stage, timeout, accepted)

    def close(self):
        with self.lock:
            running = list(self.processes)
        for process in running:
            stop(process)
        self.done.set()
        self.monitor.join(timeout=6)


def http(base, endpoint, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + endpoint, data=data,
                                     headers={"Content-Type": "application/json"})
    # Ignore proxy environment settings for the loopback-only test server.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        content = response.read().decode()
        return json.loads(content) if content.startswith(("{", "[")) else content


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def server_command(runner, binary, model, slots, port, unified=True, extra=()):
    saves = runner.workspace / "slot-saves"
    saves.mkdir(exist_ok=True)
    return [binary, "--model", model, "--host", "127.0.0.1", "--port", str(port),
            "--parallel", str(slots), "--ctx-size", "8192", "--batch-size", "512",
            "--ubatch-size", str(runner.args.ubatch), "--n-gpu-layers", str(runner.args.gpu_layers),
            "--flash-attn", "on", "--cache-type-k", "q5_1", "--cache-type-v", "q5_1",
            "--kv-stream-arena-mib", str(runner.args.arena_mib),
            "--kv-unified" if unified else "--no-kv-unified", "--fit", "off",
            "--cache-ram", "0", "--ctx-checkpoints", "0", "--no-context-shift",
            "--no-warmup", "--slots", "--metrics", "--verbosity", "4",
            "--slot-save-path", str(saves) + "/", *extra]


def launch_server(runner, name, command):
    process, stage = runner.start(name, command)
    port = command[command.index("--port") + 1]
    base = "http://127.0.0.1:" + port
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None or runner.fault:
            raise RuntimeError(name + " exited before becoming ready; see its log")
        try:
            if http(base, "/health", timeout=3).get("status") == "ok":
                return process, stage, base
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError(name + " startup timed out")


def complete(base, prompt, slot, count, cache=False):
    result = http(base, "/completion", {"prompt": prompt, "id_slot": slot,
                  "n_predict": count, "temperature": 0, "seed": 123,
                  "ignore_eos": True, "cache_prompt": cache, "return_tokens": True})
    if result.get("error") or result.get("id_slot") != slot or len(result.get("tokens", [])) != count:
        raise RuntimeError("incomplete or incorrectly routed completion: " + json.dumps(result))
    return result


def server_tests(runner, binary, model):
    for name, unified, extra, expected in (
        ("reject_non_unified", False, [], "parallel sequences requires --kv-unified"),
        ("reject_speculation", True, ["--spec-type", "ngram-simple"], "does not support speculative decoding"),
    ):
        command = server_command(runner, binary, model, 2, unused_port(), unified, extra)
        process, stage = runner.start(name, command)
        runner.finish(process, stage, 900, accepted=tuple(range(1, 256)))
        if expected not in (runner.results / (name + ".log")).read_text(errors="replace"):
            raise RuntimeError(name + " failed for an unexpected reason")

    prompts = ["Amber notebook: describe stars and quiet weather. " * 70,
               "Blue notebook: describe engines and loud machines. " * 53,
               "Green notebook: describe rivers, maps and bridges. " * 61]
    references = []
    process, stage, base = launch_server(runner, "server_serial",
        server_command(runner, binary, model, 1, unused_port()))
    try:
        for prompt, count in zip(prompts, (8, 48, 12)):
            references.append(complete(base, prompt, 0, count))
    finally:
        stop(process)
        runner.finish(process, stage, 15, accepted=(0, -signal.SIGTERM))

    process, stage, base = launch_server(runner, "server_parallel",
        server_command(runner, binary, model, 2, unused_port()))
    data = {"references": references, "slot_samples": [], "comparisons_are_diagnostic": True}
    try:
        if len(http(base, "/slots")) != 2:
            raise RuntimeError("server did not create two slots")
        barrier = threading.Barrier(2)
        def concurrent_completion(index):
            barrier.wait(timeout=30)
            return complete(base, prompts[index], index, (8, 48)[index])
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(concurrent_completion, 0)
            second = executor.submit(concurrent_completion, 1)
            overlap = False
            while not first.done():
                slots = http(base, "/slots", timeout=30)
                data["slot_samples"].append({"time": time.time(), "slots": slots})
                overlap |= sum(bool(slot.get("is_processing")) for slot in slots) == 2
                time.sleep(0.1)
            first_result = first.result()
            # The long request must still be alive when the short slot is reused.
            if second.done():
                raise RuntimeError("arrival test missed overlap; increase the second request length")
            replacement = executor.submit(complete, base, prompts[2], 0, 12)
            results = [first_result, second.result(), replacement.result()]
        if not overlap:
            raise RuntimeError("never observed both slots processing")
        data["results"] = results
        data["exact_token_matches"] = [a["tokens"] == b["tokens"] for a, b in zip(references, results)]
        # Numerical tolerance is checked by the C++ teacher-forced logit test.
        # Greedy ties may change later HTTP tokens; record them for inspection.
        data["save"] = http(base, "/slots/0?action=save", {"filename": "slot0.bin"})
        data["erase"] = http(base, "/slots/0?action=erase", {})
        data["restore"] = http(base, "/slots/0?action=restore", {"filename": "slot0.bin"})
        if data["save"].get("n_written", 0) <= 0 or data["restore"].get("n_read", 0) <= 0:
            raise RuntimeError("slot save/restore did not transfer state")
        data["after_restore"] = complete(base, prompts[2], 0, 12, cache=True)
        (runner.results / "server-metrics.txt").write_text(http(base, "/metrics"))
    finally:
        write_json(runner.results / "server-tests.json", data)
        stop(process)
        runner.finish(process, stage, 15, accepted=(0, -signal.SIGTERM))
    log = (runner.results / "server_parallel.log").read_text(errors="replace")
    data["arena_phases_logged"] = {phase: "activated " + phase + " phase" in log for phase in ("decode", "prefill")}
    write_json(runner.results / "server-tests.json", data)
    if not all(data["arena_phases_logged"].values()):
        raise RuntimeError("server did not log both arena phases at verbosity 4")


def prepare_source(runner, patch):
    source = runner.workspace / "source"
    if not runner.args.resume:
        runner.run("git_init", ["git", "init", source])
        runner.run("git_remote", ["git", "-C", source, "remote", "add", "origin", REPOSITORY])
        runner.run("git_fetch", ["git", "-c", "credential.helper=", "-C", source,
                                "fetch", "--depth", "1", "origin", BASE])
        runner.run("git_checkout", ["git", "-C", source, "checkout", "--detach", BASE])
        runner.run("patch_check", ["git", "-C", source, "apply", "--check", patch])
        runner.run("patch_apply", ["git", "-C", source, "apply", patch])
    else:
        runner.run("resume_base", ["git", "-C", source, "rev-parse", "HEAD"])
        if (runner.results / "resume_base.log").read_text().strip() != BASE:
            raise RuntimeError("resume checkout is not at the required base commit")
        current = runner.run("resume_current_patch", ["git", "-C", source,
                             "apply", "--reverse", "--check", patch], accepted=(0, 1)) == 0
        if not current:
            previous = runner.workspace / "applied-kv-stream.patch"
            if not previous.is_file():
                previous = runner.workspace / "results" / "kv-stream-multislot.patch"
            if not previous.is_file():
                raise RuntimeError("resume workspace is missing its previously applied patch")
            saved_previous = runner.results / "previous-kv-stream.patch"
            shutil.copyfile(previous, saved_previous)
            runner.run("resume_previous_patch", ["git", "-C", source,
                       "apply", "--reverse", "--check", saved_previous])
            timestamps = {}
            for patch_file in (saved_previous, patch):
                for line in patch_file.read_text().splitlines():
                    if not line.startswith(("--- a/", "+++ b/")):
                        continue
                    relative = Path(line[6:])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise RuntimeError("unsafe patch path")
                    path = source / relative
                    if path.is_file() and not path.is_symlink() and path not in timestamps:
                        stat = path.stat()
                        timestamps[path] = (hashlib.sha256(path.read_bytes()).digest(), stat)
            try:
                runner.run("resume_remove_previous", ["git", "-C", source,
                           "apply", "--reverse", saved_previous])
                try:
                    runner.run("patch_check", ["git", "-C", source, "apply", "--check", patch])
                    runner.run("patch_apply", ["git", "-C", source, "apply", patch])
                except BaseException:
                    runner.run("resume_restore_previous", ["git", "-C", source, "apply", saved_previous])
                    raise
            finally:
                restored = []
                changed = []
                for path, (digest, stat) in timestamps.items():
                    if (path.is_file() and not path.is_symlink() and path.stat().st_mode == stat.st_mode and
                            hashlib.sha256(path.read_bytes()).digest() == digest):
                        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                        if path.stat().st_mtime_ns != stat.st_mtime_ns:
                            raise RuntimeError("could not restore unchanged source timestamp")
                        restored.append(str(path.relative_to(source)))
                    else:
                        changed.append(str(path.relative_to(source)))
                write_json(runner.results / "resume-timestamps.json", {
                    "content_identical_timestamps_restored": restored, "changed_files": changed})
                print("RESUME unchanged timestamps restored=" + str(len(restored)), flush=True)
    installed_patch = runner.workspace / "applied-kv-stream.patch"
    if patch.resolve() != installed_patch.resolve():
        shutil.copyfile(patch, installed_patch)
    runner.run("diff_check", ["git", "-C", source, "diff", "--check"])
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="existing GGUF; linked read-only, never copied")
    parser.add_argument("--patch", type=Path, default=Path(__file__).resolve().with_name("kv-stream-multislot.patch"))
    parser.add_argument("--resume", type=Path, help="reuse a previous runner workspace under /mnt/ramdisk and its compiled objects")
    parser.add_argument("--arena-mib", type=int, default=2048, help="use your working single-slot arena size")
    parser.add_argument("--pressure-arena-mib", type=int, default=128, help="smaller arena for the 3K-per-slot streaming-pressure check")
    parser.add_argument("--cuda-arch", default="89", help="89 for RTX 4080")
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--long-prompt", type=int, default=250000)
    parser.add_argument("--skip-long", action="store_true", help="quick run; report marks long validation skipped")
    parser.add_argument("--long-on-failure", action="store_true", help="run the expensive long-context diagnostic even if prerequisite runtime checks fail")
    parser.add_argument("--model-timeout-hours", type=float, default=24)
    parser.add_argument("--max-abs", type=float, default=0.15)
    parser.add_argument("--max-rmse", type=float, default=0.02)
    args = parser.parse_args()
    model = args.model.expanduser().resolve(strict=True)
    patch = args.patch.expanduser().resolve(strict=True)
    if not ROOT.is_dir() or not os.path.ismount(ROOT):
        parser.error("/mnt/ramdisk must be an existing mounted ramdisk; no fallback path is used")
    if not model.is_file() or not patch.is_file():
        parser.error("model and patch must be files")
    if args.arena_mib <= 0 or args.pressure_arena_mib <= 0 or not 1 <= args.jobs <= 4 or not 2 <= args.ubatch <= 512 or not 512 <= args.long_prompt <= 1000000:
        parser.error("invalid arena, jobs (1..4), ubatch (2..512), or long prompt length")
    if args.model_timeout_hours <= 0 or not args.max_abs > 0 or not args.max_rmse > 0:
        parser.error("timeouts and comparison tolerances must be positive")
    if args.resume:
        workspace = args.resume.expanduser().resolve(strict=True)
        if workspace == ROOT.resolve() or not workspace.is_relative_to(ROOT.resolve()):
            parser.error("--resume must point to a runner workspace inside /mnt/ramdisk")
        if not (workspace / "source" / ".git").is_dir() or not (workspace / "build" / "CMakeCache.txt").is_file():
            parser.error("--resume requires an existing source checkout and CMake build directory")
        if any(not (workspace / name).resolve().is_relative_to(workspace) for name in ("source", "build")):
            parser.error("resume source and build directories must remain inside the ramdisk workspace")
    else:
        workspace = ROOT / ("kv-multislot-" + time.strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid()))
        workspace.mkdir()
    runner = Runner(workspace, args)
    status = "FAIL"
    error = None
    failed_checks = []
    long_attempted = False
    try:
        shutil.copyfile(patch, runner.results / "kv-stream-multislot.patch")
        shutil.copyfile(Path(__file__).resolve(), runner.results / "run_kv_multislot_tests.py")
        write_json(runner.results / "manifest.json", {
            "repository": REPOSITORY, "base": BASE, "model_name": model.name,
            "model_bytes": model.stat().st_size, "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "long_validation_skipped": args.skip_long,
        })
        for tool in ("git", "cmake", "ctest", "nvcc", "nvidia-smi"):
            if shutil.which(tool) is None:
                raise RuntimeError("required tool not found: " + tool)
        runner.run("environment", ["nvidia-smi", "-q"], timeout=30)
        runner.run("compiler", ["nvcc", "--version"], timeout=30)
        runner.run("python_syntax", [sys.executable, "-m", "py_compile", runner.results / "run_kv_multislot_tests.py"])
        source = prepare_source(runner, patch)
        model_dir = workspace / "model"
        model_dir.mkdir(exist_ok=bool(args.resume))
        # Link all shards when the selected model is a split GGUF.
        split = re.fullmatch(r"(.*)-\d{5}-of-(\d{5})\.gguf", model.name)
        shards = sorted(model.parent.glob(split[1] + "-?????-of-" + split[2] + ".gguf")) if split else [model]
        if split and len(shards) != int(split[2]):
            raise RuntimeError("not all GGUF shards are present beside the selected model")
        for shard in shards:
            link = model_dir / shard.name
            if link.is_symlink() and link.resolve() == shard.resolve():
                continue
            if link.exists() or link.is_symlink():
                raise RuntimeError("existing model link points to a different input: " + str(link))
            link.symlink_to(shard)
        linked_model = model_dir / model.name
        build = workspace / "build"
        runner.run("configure", ["cmake", "-S", source, "-B", build,
                   "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON", "-DGGML_CUDA_FA_ALL_QUANTS=ON",
                   "-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_arch, "-DLLAMA_BUILD_TESTS=ON",
                   "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_EXAMPLES=OFF",
                   "-DLLAMA_OPENSSL=OFF", "-DGGML_CCACHE=OFF"], timeout=1800)
        runner.run("build", ["cmake", "--build", build, "--parallel", str(args.jobs),
                   "--target", "llama-server", "test-kv-stream-multislot", *TESTS], timeout=14400)
        expression = "^(" + "|".join(TESTS) + ")$"
        runner.run("ctest_inventory", ["ctest", "--test-dir", build, "--show-only=json-v1", "-R", expression])
        inventory = json.loads((runner.results / "ctest_inventory.log").read_text())
        if set(item["name"] for item in inventory["tests"]) != set(TESTS):
            raise RuntimeError("required CTest targets are missing")
        def independent_check(name, function):
            try:
                function()
            except Exception as failure:
                failed_checks.append({"name": name, "error": str(failure)})
                (runner.results / (name + "-failure.txt")).write_text(traceback.format_exc())
                print("CHECK FAILED:", name, str(failure), flush=True)
                if runner.fault:
                    raise
        independent_check("ctest", lambda: runner.run("ctest", ["ctest", "--test-dir", build,
                   "--output-on-failure", "--verbose", "-j", "1", "--timeout", "1800", "--output-junit",
                   runner.results / "ctest.xml", "-R", expression], timeout=14400))
        binary = build / "bin" / "test-kv-stream-multislot"
        model_command = [binary, "--model", linked_model, "--arena-mib", str(args.arena_mib),
                         "--gpu-layers", str(args.gpu_layers), "--ubatch", str(args.ubatch),
                         "--max-abs", str(args.max_abs), "--max-rmse", str(args.max_rmse)]
        def model_reference_proof(name, report, minimum_kv=0):
            log = (runner.results / (name + ".log")).read_text(errors="replace")
            calls = [tuple(map(int, pair)) for pair in re.findall(
                r"kv_stream_fp32_reference: queries=(\d+) kv=(\d+)", log)]
            recorded = {}
            replayed = {}
            for line in log.splitlines():
                for label, collection in (("kv_stream_plan_record:", recorded), ("kv_stream_plan_replay:", replayed)):
                    if label in line:
                        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
                        fields["file"] = line.split(" file=", 1)[1]
                        collection.setdefault(fields["file"], []).append(fields)
            plans = []
            for suffix in (".serial-0.plan", ".serial-1.plan", ".serial-2.plan", ".parallel.plan"):
                path = Path(str(report) + suffix)
                records = recorded.get(str(path), [])
                replays = replayed.get(str(path), [])
                lines = path.read_text().splitlines() if path.is_file() else []
                complete = bool(lines) and len(lines) == len(records) == len(replays)
                split_mma = split_vec = 0
                for index, line in enumerate(lines):
                    values = list(map(int, line.split()))
                    if len(values) < 6:
                        complete = False
                        continue
                    version, shape, mode, parts, queries, spans = values[:6]
                    complete &= (version == 1 and 0 <= shape < 2**64 and mode in range(4) and
                                 parts > 0 and queries > 0 and spans > 0 and len(values) == 6 + 2*spans)
                    split_mma += mode == 2 and spans > 1
                    split_vec += mode == 3 and spans > 1
                    if index >= len(records) or index >= len(replays):
                        complete = False
                        continue
                    a, b = records[index], replays[index]
                    complete &= (all(a[key] == b[key] for key in ("queries", "kv", "mode", "spans")) and
                                 int(b["call"]) == index + 1 and int(a["mode"]) == mode and int(a["spans"]) == spans)
                    next_token = 0
                    for begin, count in zip(values[6::2], values[7::2]):
                        complete &= begin == next_token and count > 0 and count % 256 == 0
                        next_token += count
                    complete &= next_token == int(a["kv"])
                plans.append({"file": path.name, "records": len(records), "replays": len(replays),
                              "complete": complete, "split_mma_calls": split_mma, "split_vector_calls": split_vec})
            with report.open() as source_report:
                rows = list(csv.DictReader(source_report))
            required = [row for row in rows if row["required"] == "1"]
            proof = {"gpu_resident_fp32_attention_calls": len(calls),
                     "maximum_kv_tokens": max((int(row["kv"]) for entries in replayed.values() for row in entries), default=0),
                     "required_comparisons": len(required),
                     "required_comparisons_use_partition_baseline": len(required) == 64 and all(
                         row["case"].startswith("partition_resident_") for row in required),
                     "plans": plans,
                     "single_span_fp32_diagnostic_comparisons": sum(row["case"].startswith("fp32_resident_vs_") for row in rows),
                     "stock_diagnostic_comparisons": sum(row["case"].startswith("stock_resident_") for row in rows)}
            write_json(runner.results / (name.replace("_", "-") + "-reference-proof.json"), proof)
            if (not calls or proof["maximum_kv_tokens"] < minimum_kv or
                    not proof["required_comparisons_use_partition_baseline"] or not all(plan["complete"] for plan in plans) or
                    proof["stock_diagnostic_comparisons"] != 39 or proof["single_span_fp32_diagnostic_comparisons"] != 39 or
                    set(recorded) != set(replayed) or len(recorded) != 4 or
                    (minimum_kv and (not any(plan["split_mma_calls"] for plan in plans) or not plans[-1]["split_vector_calls"]))):
                raise RuntimeError("model check lacks complete resident partition replay or retained diagnostics; see reference proof")
        def model_short():
            report = runner.results / "model-short.csv"
            try:
                runner.run("model_short", [*model_command, "--prompt-tokens", "1023", "--report", report],
                           timeout=args.model_timeout_hours*3600)
            finally:
                model_reference_proof("model_short", report)
        independent_check("model_short", model_short)
        def model_pressure(name="model_pressure", diagnostic=False):
            stem = name.replace("_", "-")
            previous_check = runner.env.get("GGML_CUDA_KV_STREAM_CHECK_ATTENTION")
            if diagnostic:
                runner.env["GGML_CUDA_KV_STREAM_CHECK_ATTENTION"] = "1"
            else:
                runner.env.pop("GGML_CUDA_KV_STREAM_CHECK_ATTENTION", None)
            try:
                runner.run(name, [*model_command, "--arena-mib", str(args.pressure_arena_mib),
                           "--prompt-tokens", "3071", "--second-prompt-tokens", "2814",
                           "--report", runner.results / (stem + ".csv")],
                           timeout=args.model_timeout_hours*3600, accepted=(0, 1) if diagnostic and any(
                               check["name"] == "model_pressure" for check in failed_checks) else (0,))
            finally:
                if previous_check is None:
                    runner.env.pop("GGML_CUDA_KV_STREAM_CHECK_ATTENTION", None)
                else:
                    runner.env["GGML_CUDA_KV_STREAM_CHECK_ATTENTION"] = previous_check
                log = (runner.results / (name + ".log")).read_text(errors="replace")
                samples = [tuple(map(int, match)) for match in re.findall(
                    r"kv_stream_adapt: queries (\d+), active (\d+), resident (\d+), ring (\d+)", log)]
                pressure = any(queries == 2 and (active + 255)//256 > resident for queries, active, resident, _ in samples)
                write_json(runner.results / (stem + "-proof.json"), {
                    "arena_mib": args.pressure_arena_mib, "two_query_nonresident_working_set": pressure,
                    "trace_samples": len(samples)})
                checks = []
                pairs = []
                partitions = []
                oracles = []
                oracle_skips = []
                for line in log.splitlines():
                    if "kv_stream_attention_check:" in line:
                        checks.append(dict(re.findall(r"(\w+)=([^\s]+)", line)))
                    elif "kv_stream_fp32_pair:" in line:
                        pair = dict(re.findall(r"(\w+)=([^\s]+)", line))
                        pair["attention_check_index"] = len(checks) - 1
                        pairs.append(pair)
                    elif "kv_stream_partition_check:" in line:
                        partition = dict(re.findall(r"(\w+)=([^\s]+)", line))
                        partition["attention_check_index"] = len(checks) - 1
                        partitions.append(partition)
                    elif "kv_stream_attention_oracle:" in line:
                        oracle = dict(re.findall(r"(\w+)=([^\s]+)", line))
                        oracle["attention_check_index"] = len(checks) - 1
                        oracles.append(oracle)
                    elif "kv_stream_attention_oracle_skip:" in line:
                        oracle_skips.append(dict(re.findall(r"(\w+)=([^\s]+)", line)))
                if diagnostic:
                    write_json(runner.results / (stem + "-attention-checks.json"), checks)
                    write_json(runner.results / (stem + "-attention-oracles.json"), {
                        "samples": oracles, "skipped": oracle_skips})
                    write_json(runner.results / (stem + "-fp32-pairs.json"), pairs)
                    write_json(runner.results / (stem + "-partition-checks.json"), partitions)
                    for check in checks:
                        print("ATTENTION CHECK", json.dumps(check), flush=True)
                    for oracle in oracles:
                        if oracle.get("worst_row") == "1":
                            print("ATTENTION ORACLE", json.dumps(oracle), flush=True)
                    for pair in pairs:
                        print("FP32 PAIR", json.dumps(pair), flush=True)
                    for partition in partitions:
                        print("PARTITION CHECK", json.dumps(partition), flush=True)
                model_reference_proof(name, runner.results / (stem + ".csv"), 3072)
            if not pressure:
                raise RuntimeError("pressure check did not exercise two-query generation beyond resident capacity; lower --pressure-arena-mib")
            if diagnostic and (not checks or "multislot result:" not in log):
                raise RuntimeError("attention diagnostic did not finish or capture streamed attention")
            if diagnostic:
                byte_fields = ("k_bad", "v_bad", "k_bad_visible", "v_bad_visible", "stream_k_bad", "stream_v_bad",
                               "stream_k_bad_visible", "stream_v_bad_visible")
                if (any(check.get("finite") != "1" or any(check.get(key) != "0" for key in byte_fields) for check in checks) or
                        not any(check.get("stream_checked") == "1" for check in checks)):
                    raise RuntimeError("attention diagnostic found non-finite output, unequal cache bytes or no safe streamed-page check")
                if ({row["attention_check_index"] for row in partitions} != set(range(len(checks))) or
                        len(partitions) != len(checks) or any(row.get("finite") != "1" or
                            not 0 <= float(row["max_abs"]) <= 0.0005 or not 0 <= float(row["rmse"]) <= 0.0005 for row in partitions)):
                    raise RuntimeError("full-tensor resident partition reference exceeds the CUDA-attention tolerance or lacks coverage")
                prefill_checks = [check for check in checks if check.get("kernel") == "400" and int(check["queries"]) > 2]
                layers = int(prefill_checks[0]["layers"]) if prefill_checks else 0
                expected_layers = {0, layers//2, layers - 1} if layers else set()
                captured_layers = {int(check["layer"]) for check in prefill_checks}
                if not expected_layers or not expected_layers.issubset(captured_layers):
                    raise RuntimeError("attention diagnostic did not capture first, middle and final prefill layers")
                for index, check in enumerate(checks):
                    if check.get("kernel") != "400" or int(check["queries"]) <= 2:
                        continue
                    captured = [sample for sample in oracles if sample["attention_check_index"] == index]
                    pair_rows = [pair for pair in pairs if pair["attention_check_index"] == index]
                    if (len(pair_rows) != 1 or any(not 0 <= float(pair_rows[0][key]) < float("inf") for key in ("max_abs", "rmse"))):
                        raise RuntimeError("attention diagnostic lacks a finite full-tensor FP32 partition comparison")
                    if (not captured or any(sample.get("finite") != "1" for sample in captured) or
                            {sample.get("mode") for sample in captured if sample.get("worst_row") == "1"} !=
                            {"f16_inputs", "quantized_inputs"}):
                        raise RuntimeError("attention diagnostic lacks finite double-precision samples for an MMA check; see its log")
                layer0 = [sample for sample in oracles if sample.get("layer") == "0" and sample.get("mode") == "f16_inputs"]
                if not layer0 or any(not float(sample[key]) <= 0.0005 for sample in layer0
                                       for key in ("streamed_max_abs", "resident_fp32_max_abs")):
                    raise RuntimeError("first-layer streamed or resident FP32 attention exceeds the independent CUDA-attention threshold")
        independent_check("model_pressure", model_pressure)
        def attention_oracle_validation():
            name = "attention_oracle_validation"
            previous_check = runner.env.get("GGML_CUDA_KV_STREAM_CHECK_ATTENTION")
            runner.env["GGML_CUDA_KV_STREAM_CHECK_ATTENTION"] = "1"
            try:
                runner.run(name, [build / "bin" / "test-kv-stream-cuda-attn",
                           "Q5_1 prefill preserves constant values across long resident spans"], timeout=1800)
            finally:
                if previous_check is None:
                    runner.env.pop("GGML_CUDA_KV_STREAM_CHECK_ATTENTION", None)
                else:
                    runner.env["GGML_CUDA_KV_STREAM_CHECK_ATTENTION"] = previous_check
                log = (runner.results / (name + ".log")).read_text(errors="replace")
                samples = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in log.splitlines()
                           if "kv_stream_attention_oracle:" in line]
                write_json(runner.results / "attention-oracle-validation.json", samples)
            f16_samples = [sample for sample in samples if sample.get("mode") == "f16_inputs" and
                           sample.get("queries") == "127" and sample.get("kv") == "3072"]
            if (not f16_samples or any(sample.get("finite") != "1" for sample in samples) or
                    any(not float(sample[key]) <= 0.0005 for sample in f16_samples
                        for key in ("streamed_max_abs", "resident_fp32_max_abs"))):
                raise RuntimeError("double-precision oracle did not reproduce the known-answer Q5_1 test; see its log")
            print("ORACLE VALIDATION sampled_max_abs=" + str(max(
                float(sample["streamed_max_abs"]) for sample in f16_samples)), flush=True)
        independent_check("attention_oracle_validation", attention_oracle_validation)
        independent_check("model_attention_diagnostic", lambda: model_pressure("model_attention_diagnostic", True))
        independent_check("server", lambda: server_tests(runner, build / "bin" / "llama-server", linked_model))
        if not args.skip_long and (not failed_checks or args.long_on_failure):
            long_attempted = True
            independent_check("model_long", lambda: runner.run("model_long", [*model_command, "--prompt-tokens", str(args.long_prompt),
                       "--second-prompt-tokens", str(args.long_prompt), "--report", runner.results / "model-long.csv"],
                       timeout=args.model_timeout_hours*3600))
        elif not args.skip_long:
            print("SKIP model_long: prerequisite runtime checks failed; inspect their logs first", flush=True)
        if failed_checks:
            raise RuntimeError("failed checks: " + ", ".join(item["name"] for item in failed_checks))
        status = "PASS_SHORT_ONLY" if args.skip_long else "PASS"
    except BaseException as failure:
        error = str(failure) or type(failure).__name__
        (runner.results / "failure.txt").write_text(traceback.format_exc())
        print("FAILED:", error, flush=True)
    finally:
        runner.close()
        write_json(runner.results / "summary.json", {"status": status, "error": error,
                   "base": BASE, "long_prompt_tokens_per_slot": args.long_prompt,
                   "long_validation_skipped": not long_attempted, "peak_workspace_bytes": runner.peak_workspace_bytes,
                   "workspace_limit_bytes": 24*GIB, "resource_guard_error": runner.fault,
                   "failed_checks": failed_checks,
                   "comparison_thresholds": {"max_abs": args.max_abs, "rmse": args.max_rmse}})
        archive = workspace / (runner.results.name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(runner.results, arcname="results")
        print("\n" + status + "\nAttach this file to the conversation:\n" + str(archive), flush=True)
    return 0 if status.startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
