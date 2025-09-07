// fp64_peak_bench.cu
// CPU vs GPU FP64 peak microbenchmark (Windows MSVC + CUDA 12.1)
// - CPU: OpenMP multi-thread, SSE2 allowed, no AVX; high compute-intensity per load/store
// - GPU: register-only FP64 FMA, auto-calibrated iterations to avoid WDDM TDR
// Build notes:
//   * Visual Studio CUDA project
//   * Enable OpenMP for host compiler
//   * Do NOT use /arch:AVX/AVX2/AVX512 (keep default on x64 so no AVX is used)
//   * Optimize /O2; optionally /fp:fast
//
// Optional CLI args:
//   --runs N
//   --warmup N
//   --cpu-threads N
//   --cpu-size N
//   --cpu-stages S
//   --gpu-blocks B
//   --gpu-tpb T
//   --gpu-inner U

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>  // For _MM_SET_FLUSH_ZERO_MODE / DENORMALS_ZERO_MODE
#include <xmmintrin.h>
#endif

// -------------------- CUDA error check --------------------
static inline void cudaCheck(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %d (%s) at %s:%d\n",
            (int)e, cudaGetErrorString(e), file, line);
        std::fflush(stderr);
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(cmd) cudaCheck((cmd), __FILE__, __LINE__)

// -------------------- Utility: flush denormals on CPU --------------------
static inline void enable_flush_denorms_cpu() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#if defined(_MM_DENORMALS_ZERO_ON)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
#endif
}

// -------------------- Timer --------------------
struct WallTimer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    void start() { t0 = clock::now(); }
    double stop_s() const {
        auto t1 = clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        return dt.count();
    }
};

// -------------------- Aligned allocation (Windows) --------------------
static inline void* aligned_malloc64(size_t bytes) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, 64);
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
    return p;
#endif
}
static inline void aligned_free64(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

// -------------------- Padded struct to avoid false sharing --------------------
struct alignas(64) PaddedDouble {
    double v;
    char pad[64 - sizeof(double)];
};

// -------------------- CPU benchmark --------------------
struct CPUSettings {
    int threads = 0;          // 0 => omp_get_max_threads()
    int perThreadN = 8192;    // elements per thread
    int stages = 128;         // per-element multiply-add stages (each stage = 2 flops)
    int warmup = 3;
    int runs = 10;
};
struct CPUResult {
    double best_gflops = 0.0;
    double avg_gflops = 0.0;
    double best_s = 0.0;
    int repeats_used = 0;
};

static CPUResult run_cpu_bench(CPUSettings s) {
    enable_flush_denorms_cpu();

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    int T = (s.threads > 0 ? s.threads : max_threads);
    if (T < 1) T = 1;

#ifdef _OPENMP
    omp_set_num_threads(T);
#endif

    const int N = s.perThreadN;
    const int STAGES = s.stages;

    // Allocate per-thread arrays in one big block to improve locality
    size_t totalN = size_t(T) * size_t(N);
    double* data = static_cast<double*>(aligned_malloc64(totalN * sizeof(double)));
    if (!data) {
        std::fprintf(stderr, "CPU: allocation failed\n");
        std::exit(EXIT_FAILURE);
    }

    // Initialize (small values to avoid overflow, but non-trivial)
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        double* a = data + size_t(t) * N;
        for (int i = 0; i < N; ++i) {
            a[i] = 1e-3 + 1e-6 * double(i + 1 + 997 * t);
        }
    }

    // Constants used in the compute loop (avoid trivial optimization to 0/1)
    const double alpha = 1.00000011920928955078125;  // ~1 + 2^-23
    const double beta = 1.0000002384185791015625;   // ~1 + 2^-22 (offset)
    // Each stage: x = x * alpha + beta => 2 floating ops per stage

    // Calibrate "repeats" to target ~0.3–1.0 s per run
    auto estimate_time_with_repeats = [&](int repeats)->double {
        // One short pass timing
        WallTimer timer;
        timer.start();
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            double* a = data + size_t(tid) * N;
            double sum = 0.0;

            for (int rep = 0; rep < repeats; ++rep) {
                // Vectorize across i; inner k loop increases compute intensity
#pragma omp for schedule(static) nowait
                for (int i = 0; i < N; ++i) {
                    double x = a[i];
                    for (int k = 0; k < STAGES; ++k) {
                        x = x * alpha + beta;
                    }
                    a[i] = x;
                    sum += x; // keep it alive
                }
            }
            (void)sum;
        }
        return timer.stop_s();
        };

    int repeats = 64; // starting point
    // Quick calibration loop
    {
        double t = estimate_time_with_repeats(std::max(1, repeats / 8));
        if (t < 0.05) repeats *= 16;
        else if (t < 0.10) repeats *= 8;
        else if (t < 0.20) repeats *= 4;
        else if (t < 0.30) repeats *= 2;

        // refine
        double tgt_lo = 0.30, tgt_hi = 1.00;
        for (int it = 0; it < 6; ++it) {
            double tt = estimate_time_with_repeats(repeats);
            if (tt < tgt_lo) repeats = (int)std::ceil(repeats * (tgt_lo / std::max(1e-9, tt)));
            else if (tt > tgt_hi) repeats = std::max(1, (int)std::floor(repeats * (tgt_hi / tt)));
            else break;
            if (repeats <= 0) { repeats = 1; break; }
        }
        repeats = std::max(1, repeats);
    }

    // Warmup runs
    for (int w = 0; w < s.warmup; ++w) {
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            double* a = data + size_t(tid) * N;
            double sum = 0.0;
            for (int rep = 0; rep < std::max(1, repeats / 2); ++rep) {
#pragma omp for schedule(static) nowait
                for (int i = 0; i < N; ++i) {
                    double x = a[i];
                    for (int k = 0; k < STAGES; ++k) {
                        x = x * alpha + beta;
                    }
                    a[i] = x;
                    sum += x;
                }
            }
            (void)sum;
        }
    }

    // Measured runs
    std::vector<double> gflops_list;
    gflops_list.reserve(s.runs);
    double best_gflops = 0.0, best_s = 0.0;

    for (int r = 0; r < s.runs; ++r) {
        WallTimer timer;
        timer.start();
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            double* a = data + size_t(tid) * N;
            double sum = 0.0;
            for (int rep = 0; rep < repeats; ++rep) {
#pragma omp for schedule(static) nowait
                for (int i = 0; i < N; ++i) {
                    double x = a[i];
#pragma loop(ivdep)  // assert independence across i
                    for (int k = 0; k < STAGES; ++k) {
                        x = x * alpha + beta;
                    }
                    a[i] = x;
                    sum += x;
                }
            }
            // Write per-thread sum to avoid dead-code elimination
            static PaddedDouble sinks[256]; // enough for typical thread counts
            sinks[tid % 256].v = sum;
        }
        double secs = timer.stop_s();

        // Total flops: T * N * repeats * (2 * STAGES)
        double flops = double(T) * double(N) * double(repeats) * double(2 * STAGES);
        double gflops = flops / secs / 1e9;

        gflops_list.push_back(gflops);
        if (gflops > best_gflops) {
            best_gflops = gflops;
            best_s = secs;
        }
        std::cout << "CPU run " << (r + 1) << "/" << s.runs
            << ": " << std::fixed << std::setprecision(2)
            << gflops << " GFLOP/s, time " << std::setprecision(3)
            << secs << " s\n";
    }

    double avg = 0.0;
    for (double v : gflops_list) avg += v;
    avg /= std::max(1, (int)gflops_list.size());

    aligned_free64(data);
    CPUResult res;
    res.best_gflops = best_gflops;
    res.avg_gflops = avg;
    res.best_s = best_s;
    res.repeats_used = repeats;
    return res;
}

// -------------------- GPU benchmark --------------------
struct GPUSettings {
    int device = 0;
    int blocks_per_sm = 16;   // blocks per SM
    int threads_per_block = 256;
    int inner_unroll = 8;     // per-iteration unroll groups (8 FMAs per group)
    int warmup = 2;
    int runs = 5;
};
struct GPUResult {
    double best_gflops = 0.0;
    double avg_gflops = 0.0;
    double best_ms = 0.0;
    int iters_used = 0;
    int blocks = 0;
    int tpb = 0;
    int total_threads = 0;
};

// Kernel: each iteration does inner_unroll * 8 FMAs per thread
template<int INNER_UNROLL>
__global__ void fp64_fma_kernel(double* __restrict__ out, int iters, double seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 8 independent accumulators to increase ILP
    double r0 = 1.0 + seed * (idx + 1);
    double r1 = 1.1 + seed * (idx + 2);
    double r2 = 1.2 + seed * (idx + 3);
    double r3 = 1.3 + seed * (idx + 4);
    double r4 = 1.4 + seed * (idx + 5);
    double r5 = 1.5 + seed * (idx + 6);
    double r6 = 1.6 + seed * (idx + 7);
    double r7 = 1.7 + seed * (idx + 8);

    const double a = 1.00000011920928955078125;
    const double b = 1.0000002384185791015625;
    const double c = 0.99999988079071044921875;
    const double d = 1.00000035762786865234375;

#pragma unroll 1
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int k = 0; k < INNER_UNROLL; ++k) {
            r0 = __fma_rn(r0, a, b);
            r1 = __fma_rn(r1, c, d);
            r2 = __fma_rn(r2, a, d);
            r3 = __fma_rn(r3, c, b);
            r4 = __fma_rn(r4, a, b);
            r5 = __fma_rn(r5, c, d);
            r6 = __fma_rn(r6, a, d);
            r7 = __fma_rn(r7, c, b);
        }
    }

    double res = (r0 + r1) + (r2 + r3) + (r4 + r5) + (r6 + r7);
    out[idx] = res; // keep result
}

template<int INNER_UNROLL>
static float launch_and_time_gpu(int blocks, int tpb, int iters, double seed, double* d_out) {
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));
    fp64_fma_kernel<INNER_UNROLL> << <blocks, tpb >> > (d_out, iters, seed);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return ms;
}

static GPUResult run_gpu_bench(GPUSettings s) {
    CUDA_CHECK(cudaSetDevice(s.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, s.device));
    const int SMs = prop.multiProcessorCount;

    const int blocks = std::max(1, SMs * s.blocks_per_sm);
    const int tpb = s.threads_per_block;
    const int total_threads = blocks * tpb;

    // Output buffer to avoid compiler dropping work
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, size_t(total_threads) * sizeof(double)));

    // Auto-calibrate iters to target ~0.3–1.0 s, but cap to avoid TDR (~1.2 s)
    const float tgt_lo_ms = 300.0f, tgt_hi_ms = 1000.0f, hard_cap_ms = 1200.0f;
    int iters = 2048; // starting point
    {
        // quick grow if too fast
        float ms = 0.0f;
        for (int step = 0; step < 10; ++step) {
            ms = (s.inner_unroll == 8) ? launch_and_time_gpu<8>(blocks, tpb, iters, 1e-9, d_out)
                : (s.inner_unroll == 16) ? launch_and_time_gpu<16>(blocks, tpb, iters, 1e-9, d_out)
                : (s.inner_unroll == 4) ? launch_and_time_gpu<4>(blocks, tpb, iters, 1e-9, d_out)
                : launch_and_time_gpu<8>(blocks, tpb, iters, 1e-9, d_out);
            if (ms < 50.0f) iters *= 8;
            else if (ms < 100.0f) iters *= 4;
            else if (ms < 200.0f) iters *= 2;
            else break;
            if (iters > (1 << 30)) break;
        }
        // refine
        for (int step = 0; step < 10; ++step) {
            ms = (s.inner_unroll == 8) ? launch_and_time_gpu<8>(blocks, tpb, iters, 2e-9, d_out)
                : (s.inner_unroll == 16) ? launch_and_time_gpu<16>(blocks, tpb, iters, 2e-9, d_out)
                : (s.inner_unroll == 4) ? launch_and_time_gpu<4>(blocks, tpb, iters, 2e-9, d_out)
                : launch_and_time_gpu<8>(blocks, tpb, iters, 2e-9, d_out);
            if (ms < tgt_lo_ms) {
                float scale = std::max(1.1f, tgt_lo_ms / std::max(1.0f, ms));
                iters = std::min<int>(int(iters * scale), iters * 8);
            }
            else if (ms > tgt_hi_ms) {
                float scale = std::max(1.1f, ms / tgt_hi_ms);
                iters = std::max(1, int(iters / scale));
            }
            else break;
            if (ms > hard_cap_ms) { // emergency shrink
                iters = std::max(1, iters / 2);
                break;
            }
        }
        iters = std::max(1, iters);
    }

    // Warmup
    for (int w = 0; w < s.warmup; ++w) {
        (s.inner_unroll == 8) ? (void)launch_and_time_gpu<8>(blocks, tpb, std::max(1, iters / 2), 3e-9, d_out)
            : (s.inner_unroll == 16) ? (void)launch_and_time_gpu<16>(blocks, tpb, std::max(1, iters / 2), 3e-9, d_out)
            : (s.inner_unroll == 4) ? (void)launch_and_time_gpu<4>(blocks, tpb, std::max(1, iters / 2), 3e-9, d_out)
            : (void)launch_and_time_gpu<8>(blocks, tpb, std::max(1, iters / 2), 3e-9, d_out);
    }

    // Measured runs
    std::vector<double> gflops_list;
    gflops_list.reserve(s.runs);
    double best_gflops = 0.0, best_ms = 0.0;

    for (int r = 0; r < s.runs; ++r) {
        float ms = (s.inner_unroll == 8) ? launch_and_time_gpu<8>(blocks, tpb, iters, 4e-9 + r * 1e-10, d_out)
            : (s.inner_unroll == 16) ? launch_and_time_gpu<16>(blocks, tpb, iters, 4e-9 + r * 1e-10, d_out)
            : (s.inner_unroll == 4) ? launch_and_time_gpu<4>(blocks, tpb, iters, 4e-9 + r * 1e-10, d_out)
            : launch_and_time_gpu<8>(blocks, tpb, iters, 4e-9 + r * 1e-10, d_out);

        // Per iter ops: INNER_UNROLL * 8 FMAs => flops = *2
        const int inner = (s.inner_unroll == 8 || s.inner_unroll == 16 || s.inner_unroll == 4) ? s.inner_unroll : 8;
        double flops = double(total_threads) * double(iters) * double(inner) * 8.0 /*FMAs*/ * 2.0 /*flops/FMA*/;
        double gflops = flops / (double(ms) / 1000.0) / 1e9;

        gflops_list.push_back(gflops);
        if (gflops > best_gflops) {
            best_gflops = gflops;
            best_ms = ms;
        }
        std::cout << "GPU run " << (r + 1) << "/" << s.runs
            << ": " << std::fixed << std::setprecision(2)
            << gflops << " GFLOP/s, time " << std::setprecision(3)
            << (ms / 1000.0) << " s\n";
    }

    double avg = 0.0;
    for (double v : gflops_list) avg += v;
    avg /= std::max(1, (int)gflops_list.size());

    CUDA_CHECK(cudaFree(d_out));

    GPUResult res;
    res.best_gflops = best_gflops;
    res.avg_gflops = avg;
    res.best_ms = best_ms;
    res.iters_used = iters;
    res.blocks = blocks;
    res.tpb = tpb;
    res.total_threads = total_threads;
    return res;
}

// -------------------- CLI parse helpers --------------------
static bool arg_eq(const char* a, const char* b) {
#ifdef _WIN32
    return _stricmp(a, b) == 0;
#else
    return std::strcmp(a, b) == 0;
#endif
}
static int arg_to_int(const char* s, int defv) {
    if (!s) return defv;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return defv;
    return (int)v;
}

// -------------------- Main --------------------
int main(int argc, char** argv) {
    // Defaults
    CPUSettings cpu;
    GPUSettings gpu;
    // runs and warmups consistent across CPU/GPU by default
    int runs = 5, warmup = 2;
    cpu.runs = runs; cpu.warmup = warmup;
    gpu.runs = runs; gpu.warmup = warmup;

    // Parse args (very lightweight)
    for (int i = 1; i < argc; ++i) {
        if (arg_eq(argv[i], "--runs") && i + 1 < argc) { runs = arg_to_int(argv[++i], runs); cpu.runs = runs; gpu.runs = runs; }
        else if (arg_eq(argv[i], "--warmup") && i + 1 < argc) { warmup = arg_to_int(argv[++i], warmup); cpu.warmup = warmup; gpu.warmup = warmup; }
        else if (arg_eq(argv[i], "--cpu-threads") && i + 1 < argc) { cpu.threads = arg_to_int(argv[++i], cpu.threads); }
        else if (arg_eq(argv[i], "--cpu-size") && i + 1 < argc) { cpu.perThreadN = arg_to_int(argv[++i], cpu.perThreadN); }
        else if (arg_eq(argv[i], "--cpu-stages") && i + 1 < argc) { cpu.stages = arg_to_int(argv[++i], cpu.stages); }
        else if (arg_eq(argv[i], "--gpu-blocks") && i + 1 < argc) { gpu.blocks_per_sm = arg_to_int(argv[++i], gpu.blocks_per_sm); }
        else if (arg_eq(argv[i], "--gpu-tpb") && i + 1 < argc) { gpu.threads_per_block = arg_to_int(argv[++i], gpu.threads_per_block); }
        else if (arg_eq(argv[i], "--gpu-inner") && i + 1 < argc) { gpu.inner_unroll = arg_to_int(argv[++i], gpu.inner_unroll); }
        else if (arg_eq(argv[i], "--help")) {
            std::cout <<
                "Usage: fp64_peak_bench [--runs N] [--warmup N]\n"
                "                      [--cpu-threads N] [--cpu-size N] [--cpu-stages S]\n"
                "                      [--gpu-blocks B] [--gpu-tpb T] [--gpu-inner U]\n";
            return 0;
        }
    }

    // Print device info
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "CUDA device: " << prop.name
        << " (SMs=" << prop.multiProcessorCount
        << ", cc " << prop.major << "." << prop.minor
        << ", mem " << std::fixed << std::setprecision(1) << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0))
        << " GiB)\n";
    std::cout << "Note: Many consumer GPUs have limited FP64 throughput (e.g., 1/32 or 1/64 of FP32).\n";

#ifdef _OPENMP
    std::cout << "OpenMP enabled. ";
#else
    std::cout << "OpenMP NOT enabled (single-thread CPU path). ";
#endif
#if defined(_M_X64)
    std::cout << "Target: x64";
#else
    std::cout << "Target: x86";
#endif
    std::cout << " (ensure no /arch:AVX* to avoid AVX on CPU)\n\n";

    // Run CPU
    std::cout << "CPU settings: threads=" << (cpu.threads > 0 ? cpu.threads :
#ifdef _OPENMP
        omp_get_max_threads()
#else
        1
#endif
        )
        << ", perThreadN=" << cpu.perThreadN
        << ", stages=" << cpu.stages
        << ", warmup=" << cpu.warmup
        << ", runs=" << cpu.runs
        << "\n";
    CPUResult cpu_res = run_cpu_bench(cpu);
    std::cout << std::fixed << std::setprecision(2)
        << "CPU best: " << cpu_res.best_gflops << " GFLOP/s"
        << " (time " << std::setprecision(3) << cpu_res.best_s << " s"
        << ", repeats=" << cpu_res.repeats_used << ")\n";
    std::cout << std::fixed << std::setprecision(2)
        << "CPU avg:  " << cpu_res.avg_gflops << " GFLOP/s\n\n";

    // Run GPU
    std::cout << "GPU settings: blocks/SM=" << gpu.blocks_per_sm
        << ", TPB=" << gpu.threads_per_block
        << ", inner=" << gpu.inner_unroll
        << ", warmup=" << gpu.warmup
        << ", runs=" << gpu.runs
        << "\n";
    GPUResult gpu_res = run_gpu_bench(gpu);
    std::cout << std::fixed << std::setprecision(2)
        << "GPU best: " << gpu_res.best_gflops << " GFLOP/s"
        << " (time " << std::setprecision(3) << (gpu_res.best_ms / 1000.0) << " s"
        << ", iters=" << gpu_res.iters_used
        << ", blocks=" << gpu_res.blocks
        << ", TPB=" << gpu_res.tpb
        << ")\n";
    std::cout << std::fixed << std::setprecision(2)
        << "GPU avg:  " << gpu_res.avg_gflops << " GFLOP/s\n\n";

    // Summary
    std::cout << std::fixed << std::setprecision(2)
        << "Summary: CPU best " << cpu_res.best_gflops << " GFLOP/s, "
        << "GPU best " << gpu_res.best_gflops << " GFLOP/s, "
        << "GPU/CPU = " << (gpu_res.best_gflops / std::max(1e-9, cpu_res.best_gflops)) << "x\n";

    return 0;
}