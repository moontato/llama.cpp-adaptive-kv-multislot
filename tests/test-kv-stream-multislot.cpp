#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Model-dependent integration test; run through scripts/run_kv_multislot_tests.py.
namespace {
using clock_type = std::chrono::steady_clock;
using tokens = std::vector<llama_token>;
using logits = std::vector<float>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

struct options {
    std::string model;
    std::string report;
    int arena_mib = 2048;
    int prompt_tokens = 1023;
    int second_prompt_tokens = 0;
    int steps = 12;
    int gpu_layers = 999;
    int ubatch = 128;
    double max_abs = 0.15;
    double max_rmse = 0.02;
};

struct reference {
    tokens prompt;
    tokens continuation;
    std::vector<logits> output;
};

struct observation {
    std::string name;
    int step;
    logits values;
};

struct environment_scope {
    std::string name;
    std::string previous;
    bool had_previous;
    static int set(const char * name, const char * value) {
#ifdef _WIN32
        return _putenv_s(name, value ? value : "");
#else
        return value ? setenv(name, value, 1) : unsetenv(name);
#endif
    }
    environment_scope(const char * key, const char * selected) : name(key) {
        const char * value = std::getenv(key);
        had_previous = value != nullptr;
        if (value) { previous = value; }
        if (set(key, selected) != 0) {
            throw std::runtime_error("cannot set test environment: " + name);
        }
    }
    ~environment_scope() { set(name.c_str(), had_previous ? previous.c_str() : nullptr); }
};

struct plan_scope {
    environment_scope record;
    environment_scope replay;
    plan_scope(bool stream, const std::string & path) :
        record("GGML_CUDA_KV_STREAM_RECORD_PLAN", stream && !path.empty() ? path.c_str() : nullptr),
        replay("GGML_CUDA_KV_STREAM_REPLAY_PLAN", !stream && !path.empty() ? path.c_str() : nullptr) {
        if (stream && !path.empty()) {
            std::ofstream output(path, std::ios::trunc);
            if (!output) { throw std::runtime_error("cannot create attention plan: " + path); }
        }
    }
};

struct batch_owner {
    llama_batch batch = llama_batch_init(512, 0, 1);
    ~batch_owner() { llama_batch_free(batch); }
    void clear() { batch.n_tokens = 0; }
    void add(llama_token token, llama_pos pos, llama_seq_id seq, bool output) {
        const int i = batch.n_tokens++;
        if (batch.n_tokens > 512) { throw std::runtime_error("test batch overflow"); }
        batch.token[i] = token;
        batch.pos[i] = pos;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = output;
    }
};

void decode(llama_context * ctx, batch_owner & batch, llama_decode_phase phase) {
    llama_set_decode_phase(ctx, phase);
    const int rc = llama_decode(ctx, batch.batch);
    if (rc != 0) { throw std::runtime_error("llama_decode returned " + std::to_string(rc)); }
}

logits read_logits(llama_context * ctx, int n_vocab, int index = -1) {
    const float * data = llama_get_logits_ith(ctx, index);
    if (!data) { throw std::runtime_error("missing logits"); }
    return logits(data, data + n_vocab);
}

llama_token greedy(const logits & values) {
    return llama_token(std::max_element(values.begin(), values.end()) - values.begin());
}

tokens prompt(const llama_vocab * vocab, int count, const std::string & text) {
    tokens seed(text.size()*4 + 32);
    const int n = llama_tokenize(vocab, text.data(), int(text.size()), seed.data(), int(seed.size()), false, false);
    if (n <= 0) { throw std::runtime_error("seed tokenization failed"); }
    seed.resize(n);
    tokens result;
    result.reserve(count);
    if (llama_vocab_get_add_bos(vocab)) { result.push_back(llama_vocab_bos(vocab)); }
    while (int(result.size()) < count) {
        result.push_back(seed[result.size() % seed.size()]);
    }
    return result;
}

context_ptr context(llama_model * model, const options & opts, int slots, bool stream) {
    auto params = llama_context_default_params();
    params.n_ctx = ((2*(std::max(opts.prompt_tokens, opts.second_prompt_tokens) + opts.steps + 1024) + 511)/512)*512;
    params.n_batch = 512;
    params.n_ubatch = opts.ubatch;
    params.n_seq_max = slots;
    params.n_outputs_max = slots;
    params.n_outputs_max_per_seq = 1;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.type_k = params.type_v = GGML_TYPE_Q5_1;
    params.kv_unified = true;
    params.offload_kqv = true;
    params.kv_stream_arena_mib = stream ? opts.arena_mib : 0;
    context_ptr ctx(llama_init_from_model(model, params), llama_free);
    if (!ctx) { throw std::runtime_error("context creation failed"); }
    return ctx;
}

logits prefill(llama_context * ctx, batch_owner & batch, const tokens & input,
        llama_seq_id seq, int n_vocab, int start = 0) {
    const auto started = clock_type::now();
    auto last_report = started;
    std::fprintf(stderr, "kv-multislot progress: prompt seq=%d tokens=%zu start=%d\n", seq, input.size(), start);
    for (int pos = start; pos < int(input.size());) {
        batch.clear();
        // An odd chunk size also exercises the one-token final prompt tail.
        const int end = std::min(pos + 511, int(input.size()));
        for (; pos < end; ++pos) {
            batch.add(input[pos], pos, seq, pos + 1 == int(input.size()));
        }
        decode(ctx, batch, LLAMA_DECODE_PHASE_PROMPT);
        const auto now = clock_type::now();
        if (now - last_report >= std::chrono::seconds(30) || pos == int(input.size())) {
            const double elapsed = std::chrono::duration<double>(now - started).count();
            std::fprintf(stderr, "kv-multislot progress: prompt seq=%d %d/%zu tokens elapsed=%.1fs rate=%.1f tok/s\n",
                seq, pos, input.size(), elapsed, (pos - start)/std::max(elapsed, 1e-6));
            last_report = now;
        }
    }
    return read_logits(ctx, n_vocab);
}

reference serial(llama_model * model, const options & opts, tokens input,
        bool stream, const reference * forced = nullptr, bool fp32_reference = true, const std::string & plan = {}) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    environment_scope reference_mode("GGML_CUDA_KV_STREAM_FP32_REFERENCE", !stream && fp32_reference && plan.empty() ? "1" : nullptr);
    plan_scope arithmetic_plan(stream, plan);
    auto ctx = context(model, opts, 1, stream);
    batch_owner batch;
    reference result;
    result.prompt = std::move(input);
    result.output.push_back(prefill(ctx.get(), batch, result.prompt, 0, n_vocab));
    for (int step = 0; step < opts.steps; ++step) {
        const llama_token token = forced ? forced->continuation[step] : greedy(result.output.back());
        result.continuation.push_back(token);
        batch.clear();
        batch.add(token, int(result.prompt.size()) + step, 0, true);
        decode(ctx.get(), batch, LLAMA_DECODE_PHASE_GENERATION);
        result.output.push_back(read_logits(ctx.get(), n_vocab));
    }
    return result;
}

struct comparisons {
    const options & opts;
    std::ofstream report;
    int failures = 0;
    explicit comparisons(const options & opts) : opts(opts), report(opts.report) {
        if (!report) { throw std::runtime_error("cannot open comparison report"); }
        report << "case,step,max_abs,rmse,reference_top,actual_top,pass,required\n";
    }
    void check(const std::string & name, int step, const logits & expected, const logits & actual, bool required = true) {
        if (expected.size() != actual.size()) { throw std::runtime_error("logit size mismatch"); }
        double max_abs = 0;
        double squared = 0;
        bool finite = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
            const double difference = double(actual[i]) - expected[i];
            max_abs = std::max(max_abs, std::abs(difference));
            squared += difference*difference;
        }
        const double rmse = std::sqrt(squared/expected.size());
        const bool pass = finite && max_abs <= opts.max_abs && rmse <= opts.max_rmse;
        failures += required && !pass;
        report << name << ',' << step << ',' << max_abs << ',' << rmse << ','
               << greedy(expected) << ',' << greedy(actual) << ',' << pass << ',' << required << '\n';
        report.flush();
        std::fprintf(stderr, "%s step=%d max_abs=%.6g rmse=%.6g %s%s\n",
            name.c_str(), step, max_abs, rmse, pass ? "PASS" : "FAIL", required ? "" : " (diagnostic)");
    }
};

std::vector<observation> parallel(llama_model * model, const options & opts,
        const std::vector<reference> & refs, comparisons & checks, bool stream,
        const std::vector<observation> * streamed = nullptr, const std::string & plan = {}) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    environment_scope reference_mode("GGML_CUDA_KV_STREAM_FP32_REFERENCE", !stream && plan.empty() ? "1" : nullptr);
    plan_scope arithmetic_plan(stream, plan);
    auto ctx = context(model, opts, 2, stream);
    std::vector<observation> outputs;
    const std::string prefix = stream ? "streamed_" : "partition_resident_";
    auto check = [&](const std::string & name, int step, const logits & serial_output, const logits & actual) {
        // Same batching separates arena regressions from serial/parallel kernel differences.
        checks.check(prefix + "serial_vs_" + name, step, serial_output, actual, stream && plan.empty());
        if (streamed) {
            const size_t i = outputs.size();
            if (stream || i >= streamed->size() || (*streamed)[i].name != name || (*streamed)[i].step != step) {
                throw std::runtime_error("resident parallel observation mismatch");
            }
            checks.check("partition_resident_parallel_vs_streamed_" + name, step, actual, (*streamed)[i].values);
        }
        outputs.push_back({ name, step, actual });
    };
    batch_owner batch;
    std::vector<logits> initial(2);
    int positions[2] = { 0, 0 };
    const auto started = clock_type::now();
    auto last_report = started;
    std::fprintf(stderr, "kv-multislot progress: %s parallel prompts total=%zu tokens\n", prefix.c_str(), refs[0].prompt.size() + refs[1].prompt.size());
    // Interleave prompt chunks so each history occupies scattered shared cells.
    while (positions[0] < int(refs[0].prompt.size()) || positions[1] < int(refs[1].prompt.size())) {
        for (int seq = 0; seq < 2; ++seq) {
            if (positions[seq] == int(refs[seq].prompt.size())) { continue; }
            batch.clear();
            const int end = std::min(positions[seq] + 511, int(refs[seq].prompt.size()));
            while (positions[seq] < end) {
                const int pos = positions[seq]++;
                batch.add(refs[seq].prompt[pos], pos, seq, positions[seq] == int(refs[seq].prompt.size()));
            }
            decode(ctx.get(), batch, LLAMA_DECODE_PHASE_PROMPT);
            if (positions[seq] == int(refs[seq].prompt.size())) { initial[seq] = read_logits(ctx.get(), n_vocab); }
        }
        const auto now = clock_type::now();
        if (now - last_report >= std::chrono::seconds(30) ||
                positions[0] + positions[1] == int(refs[0].prompt.size() + refs[1].prompt.size())) {
            const double elapsed = std::chrono::duration<double>(now - started).count();
            std::fprintf(stderr, "kv-multislot progress: parallel seq0=%d/%zu seq1=%d/%zu elapsed=%.1fs rate=%.1f tok/s\n",
                positions[0], refs[0].prompt.size(), positions[1], refs[1].prompt.size(), elapsed,
                (positions[0] + positions[1])/std::max(elapsed, 1e-6));
            last_report = now;
        }
    }
    check("parallel_prefill_0", 0, refs[0].output[0], initial[0]);
    check("parallel_prefill_1", 0, refs[1].output[0], initial[1]);
    std::vector<uint8_t> checkpoint;
    int reused_step = 0;
    for (int step = 1; step <= opts.steps; ++step) {
        if (step == 5) {
            if (std::max(opts.prompt_tokens, opts.second_prompt_tokens) <= 4096) {
                checkpoint.resize(llama_state_seq_get_size(ctx.get(), 1));
                if (checkpoint.empty() || llama_state_seq_get_data(
                        ctx.get(), checkpoint.data(), checkpoint.size(), 1) != checkpoint.size()) {
                    throw std::runtime_error("sequence checkpoint failed");
                }
            }
            if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
                throw std::runtime_error("slot removal failed");
            }
        }
        batch.clear();
        if (step == 8) {
            // Refill slot 0 while slot 1 contributes its next generation token.
            const int end = std::min(127, int(refs[2].prompt.size()));
            for (int pos = 0; pos < end; ++pos) { batch.add(refs[2].prompt[pos], pos, 0, false); }
            const int output_index = batch.batch.n_tokens;
            batch.add(refs[1].continuation[step - 1], int(refs[1].prompt.size()) + step - 1, 1, true);
            decode(ctx.get(), batch, LLAMA_DECODE_PHASE_AUTOMATIC);
            check("mixed_generation_1", step, refs[1].output[step], read_logits(ctx.get(), n_vocab, output_index));
            const auto output = prefill(ctx.get(), batch, refs[2].prompt, 0, n_vocab, end);
            check("reused_slot_prefill_0", 0, refs[2].output[0], output);
            continue;
        }
        int output0 = -1;
        if (step <= 4 || step > 8) {
            const reference & ref = refs[step <= 4 ? 0 : 2];
            const int local_step = step <= 4 ? step : ++reused_step;
            output0 = batch.batch.n_tokens;
            batch.add(ref.continuation[local_step - 1], int(ref.prompt.size()) + local_step - 1, 0, true);
        }
        const int output1 = batch.batch.n_tokens;
        batch.add(refs[1].continuation[step - 1], int(refs[1].prompt.size()) + step - 1, 1, true);
        decode(ctx.get(), batch, LLAMA_DECODE_PHASE_GENERATION);
        if (output0 >= 0) {
            check(step <= 4 ? "parallel_generation_0" : "reused_slot_generation_0",
                step <= 4 ? step : reused_step, refs[step <= 4 ? 0 : 2].output[step <= 4 ? step : reused_step],
                read_logits(ctx.get(), n_vocab, output0));
        }
        check(step <= 4 || step > 8 ? "parallel_generation_1" : "single_active_slot_1",
            step, refs[1].output[step], read_logits(ctx.get(), n_vocab, output1));
    }
    if (!checkpoint.empty()) {
        if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 1, -1, -1) ||
                llama_state_seq_set_data(ctx.get(), checkpoint.data(), checkpoint.size(), 1) != checkpoint.size()) {
            throw std::runtime_error("sequence restore failed");
        }
        batch.clear();
        batch.add(refs[1].continuation[4], int(refs[1].prompt.size()) + 4, 1, true);
        decode(ctx.get(), batch, LLAMA_DECODE_PHASE_GENERATION);
        check("restored_slot_1", 5, refs[1].output[5], read_logits(ctx.get(), n_vocab));
        batch.clear();
        batch.add(refs[2].continuation[reused_step], int(refs[2].prompt.size()) + reused_step, 0, true);
        decode(ctx.get(), batch, LLAMA_DECODE_PHASE_GENERATION);
        check("restore_preserves_other_slot_0", reused_step + 1,
            refs[2].output[reused_step + 1], read_logits(ctx.get(), n_vocab));
    }
    if (streamed && outputs.size() != streamed->size()) {
        throw std::runtime_error("missing resident parallel observations");
    }
    return outputs;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        options opts;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) { throw std::runtime_error("missing option value"); }
            const std::string key = argv[i];
            const std::string value = argv[i + 1];
            if      (key == "--model")         { opts.model = value; }
            else if (key == "--report")        { opts.report = value; }
            else if (key == "--arena-mib")     { opts.arena_mib = std::stoi(value); }
            else if (key == "--prompt-tokens") { opts.prompt_tokens = std::stoi(value); }
            else if (key == "--second-prompt-tokens") { opts.second_prompt_tokens = std::stoi(value); }
            else if (key == "--steps")         { opts.steps = std::stoi(value); }
            else if (key == "--gpu-layers")    { opts.gpu_layers = std::stoi(value); }
            else if (key == "--ubatch")        { opts.ubatch = std::stoi(value); }
            else if (key == "--max-abs")       { opts.max_abs = std::stod(value); }
            else if (key == "--max-rmse")      { opts.max_rmse = std::stod(value); }
            else { throw std::runtime_error("unknown option: " + key); }
        }
        if (opts.second_prompt_tokens == 0) { opts.second_prompt_tokens = opts.prompt_tokens - 257; }
        if (opts.model.empty() || opts.report.empty() || opts.arena_mib <= 0 ||
                opts.prompt_tokens < 512 || opts.prompt_tokens > 1000000 || opts.steps < 10 ||
                opts.second_prompt_tokens < 128 || opts.second_prompt_tokens > 1000000 ||
                opts.steps > 128 || opts.ubatch < 2 || opts.ubatch > 512 ||
                !(opts.max_abs > 0) || !(opts.max_rmse > 0)) {
            throw std::runtime_error("invalid options: require --model GGUF --report CSV, positive arena, prompt 512..1000000, steps 10..128, ubatch 2..512");
        }
        // Capture every attention operation, including repeated generation graphs.
        environment_scope disable_graphs("GGML_CUDA_DISABLE_GRAPHS", "1");
        ggml_backend_load_all();
        llama_backend_init();
        auto params = llama_model_default_params();
        params.n_gpu_layers = opts.gpu_layers;
        params.load_mtp = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(opts.model.c_str(), params), llama_model_free);
        if (!model) { throw std::runtime_error("model load failed"); }
        const auto * vocab = llama_model_get_vocab(model.get());
        const tokens inputs[] = {
            prompt(vocab, opts.prompt_tokens, "Amber notebook. The observatory records stars and quiet weather. "),
            prompt(vocab, opts.second_prompt_tokens, "Blue notebook. The workshop records engines and loud machines. "),
            prompt(vocab, 779, "Green notebook. A new visitor asks about rivers, maps and bridges. "),
        };
        comparisons checks(opts);
        std::vector<reference> refs;
        const bool short_run = std::max(opts.prompt_tokens, opts.second_prompt_tokens) <= 4096;
        for (int seq = 0; seq < 3; ++seq) {
            std::fprintf(stderr, "kv-multislot progress: serial reference %d/3 prompt=%zu tokens\n", seq + 1, inputs[seq].size());
            const std::string plan = short_run ? opts.report + ".serial-" + std::to_string(seq) + ".plan" : "";
            auto ref = serial(model.get(), opts, inputs[seq], true, nullptr, false, plan);
            // A resident reference is affordable only for the short run.
            if (short_run) {
                const auto partition_resident = serial(model.get(), opts, inputs[seq], false, &ref, false, plan);
                for (int step = 0; step <= opts.steps; ++step) {
                    checks.check("partition_resident_vs_streamed_" + std::to_string(seq), step,
                        partition_resident.output[step], ref.output[step]);
                }
                const auto resident = serial(model.get(), opts, inputs[seq], false, &ref);
                for (int step = 0; step <= opts.steps; ++step) {
                    checks.check("fp32_resident_vs_streamed_" + std::to_string(seq), step, resident.output[step], ref.output[step], false);
                }
                const auto stock = serial(model.get(), opts, inputs[seq], false, &ref, false);
                for (int step = 0; step <= opts.steps; ++step) {
                    checks.check("stock_resident_vs_streamed_" + std::to_string(seq), step, stock.output[step], ref.output[step], false);
                }
            }
            refs.push_back(std::move(ref));
        }
        if (short_run) {
            const std::string plan = opts.report + ".parallel.plan";
            const auto streamed = parallel(model.get(), opts, refs, checks, true, nullptr, plan);
            parallel(model.get(), opts, refs, checks, false, &streamed, plan);
        } else {
            parallel(model.get(), opts, refs, checks, true);
        }
        std::fprintf(stderr, "multislot result: %d comparison failures; long runs omit large state snapshots\n", checks.failures);
        model.reset();
        llama_backend_free();
        return checks.failures ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "multislot test failed: %s\n", error.what());
        return 1;
    }
}
