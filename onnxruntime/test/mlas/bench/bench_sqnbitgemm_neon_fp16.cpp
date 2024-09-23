#include "bench_util.h"
#include "mlasi.h"

void BM_ConvertF16ToF32(benchmark::State& state) {
    bool aligned = static_cast<bool>(state.range(0));
    const size_t count = 1 << 18;
    const auto src = RandomVectorUniform<unsigned short>(count, 0, 60000);
    const auto dst = std::vector<float>(count + 16);
    auto aligned_dst = (reinterpret_cast<intptr_t>(dst.data()) + 15) & (~15);
    const float* dst_start = aligned ? reinterpret_cast<float*>(aligned_dst)
                                     : reinterpret_cast<float*>(aligned_dst + 1);

    // Warm up
    MlasCastF16ToF32KernelNeon(src.data(), dst_start, count);

    for (auto _ : state) {
        MlasCastF16ToF32KernelNeon(src.data(), dst_start, count);
    }
}

void BM_ConvertF32ToF16(benchmark::State& state) {
    bool aligned = static_cast<bool>(state.range(0));
    const size_t count = 1 << 18;
    const auto src = RandomVectorUniform(count, -30000.0f, 30000.0f);
    const auto dst = std::vector<unsigned short>(count + 16);
    auto aligned_dst = (reinterpret_cast<intptr_t>(dst.data()) + 15) & (~15);
    const unsigned short* dst_start = aligned ? reinterpret_cast<unsigned short*>(aligned_dst)
                                              : reinterpret_cast<unsigned short*>(aligned_dst + 1);

    // Warm up
    MlasCastF32ToF16KernelNeon(src.data(), dst_start, count);

    for (auto _ : state) {
        MlasCastF32ToF16KernelNeon(src.data(), dst_start, count);
    }
}

BENCHMARK(BM_ConvertF16ToF32)
    ->UseRealTime()
    ->Apply([](benchmark::internal::Benchmark* b) {
      b->ArgNames({"aligned"});
      b->Args({0, 1});
    });

BENCHMARK(BM_ConvertF32ToF16)
    ->UseRealTime()
    ->Apply([](benchmark::internal::Benchmark* b) {
      b->ArgNames({"aligned"});
      b->Args({0, 1});
    });
