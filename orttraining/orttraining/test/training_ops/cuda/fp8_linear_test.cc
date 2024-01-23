// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_FP8_TRAINING

#include "test/providers/compare_provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

void PrepareFp8LinearData(int b, int m, int n, int k, std::vector<MLFloat16>& input, std::vector<MLFloat16>& weight,
                          std::vector<MLFloat16>& bias, std::vector<MLFloat16>& output,
                          std::vector<Float8E4M3FN>& input_t, std::vector<Float8E4M3FN>& weight_t,
                          std::vector<float>& scale_inv) {
  input.resize(b * m * k);
  weight.resize(n * k);
  bias.resize(n);
  output.resize(b * m * n);
  input_t.resize(k * b * m);
  weight_t.resize(k * n);
  scale_inv.resize(3, 1.0f);

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.8f, 1.0f);

  for (int i = 0; i < b * m * k; ++i) {
    input[i] = MLFloat16(distribution(generator));
  }

  for (int i = 0; i < n * k; ++i) {
    weight[i] = MLFloat16(distribution(generator));
  }

  for (int i = 0; i < n; ++i) {
    bias[i] = MLFloat16(distribution(generator));
  }

  for (int i = 0; i < b * m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < k; ++l) {
        sum += (float)input[i * k + l] * (float)weight[j * k + l];
      }
      output[i * n + j] = MLFloat16(sum + (float)bias[j]);
    }
  }

  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < b * m; ++j) {
      input_t[i * b * m + j] = Float8E4M3FN((float)input[j * k + i]);
    }
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; ++j) {
      weight_t[i * n + j] = Float8E4M3FN((float)weight[j * k + i]);
    }
  }
}

TEST(Fp8LinearOpTest, Gemm) {
  OpTester test("FP8Linear", 1, "com.microsoft");
  int m = 32, n = 16, k = 16;
  std::vector<MLFloat16> input;
  std::vector<MLFloat16> weight;
  std::vector<MLFloat16> bias;
  std::vector<MLFloat16> output;
  std::vector<Float8E4M3FN> input_t;
  std::vector<Float8E4M3FN> weight_t;
  std::vector<float> scale_inv;
  PrepareFp8LinearData(1, m, n, k, input, weight, bias, output, input_t, weight_t, scale_inv);

  test.AddInput<MLFloat16>("input", {m, k}, input);
  test.AddInput<MLFloat16>("weight", {n, k}, weight);
  test.AddInput<MLFloat16>("bias", {n}, bias);
  test.AddOutput<MLFloat16>("output", {m, n}, output, false, 0.05f);
  test.AddOutput<Float8E4M3FN>("input_t", {k, m}, input_t);
  test.AddOutput<Float8E4M3FN>("weight_t", {k, n}, weight_t);
  test.AddOutput<float>("scale_inv", {3}, scale_inv);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(Fp8LinearOpTest, BatchGemm) {
  OpTester test("FP8Linear", 1, "com.microsoft");
  int b = 16, m = 32, n = 16, k = 16;
  std::vector<MLFloat16> input;
  std::vector<MLFloat16> weight;
  std::vector<MLFloat16> bias;
  std::vector<MLFloat16> output;
  std::vector<Float8E4M3FN> input_t;
  std::vector<Float8E4M3FN> weight_t;
  std::vector<float> scale_inv;
  PrepareFp8LinearData(b, m, n, k, input, weight, bias, output, input_t, weight_t, scale_inv);

  test.AddInput<MLFloat16>("input", {b, m, k}, input);
  test.AddInput<MLFloat16>("weight", {n, k}, weight);
  test.AddInput<MLFloat16>("bias", {n}, bias);
  test.AddOutput<MLFloat16>("output", {b, m, n}, output, false, 0.05f);
  test.AddOutput<Float8E4M3FN>("input_t", {k, b * m}, input_t);
  test.AddOutput<Float8E4M3FN>("weight_t", {k, n}, weight_t);
  test.AddOutput<float>("scale_inv", {3}, scale_inv);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

void PrepareFp8LinearGradData(int b, int m, int n, int k, std::vector<MLFloat16>& grad_output,
                              std::vector<Float8E4M3FN>& input_t, std::vector<Float8E4M3FN>& weight_t,
                              std::vector<float>& scale_inv, std::vector<MLFloat16>& grad_input,
                              std::vector<MLFloat16>& grad_weight, std::vector<MLFloat16>& grad_bias) {
  grad_output.resize(b * m * n);
  input_t.resize(k * b * m);
  weight_t.resize(k * n);
  scale_inv.resize(3, 1.0f);
  grad_input.resize(b * m * k);
  grad_weight.resize(n * k);
  grad_bias.resize(n);

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.8f, 1.0f);

  for (int i = 0; i < b * m * n; ++i) {
    grad_output[i] = MLFloat16(distribution(generator));
  }

  for (int i = 0; i < k * b * m; ++i) {
    input_t[i] = Float8E4M3FN(distribution(generator));
  }

  for (int i = 0; i < k * n; ++i) {
    weight_t[i] = Float8E4M3FN(distribution(generator));
  }

  for (int i = 0; i < b * m; ++i) {
    for (int j = 0; j < k; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < n; ++l) {
        sum += (float)grad_output[i * n + l] * (float)weight_t[j * n + l];
      }
      grad_input[i * k + j] = MLFloat16(sum);
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < b * m; ++l) {
        sum += (float)grad_output[l * n + i] * (float)input_t[j * b * m + l];
      }
      grad_weight[i * k + j] = MLFloat16(sum);
    }
  }

  for (int i = 0; i < n; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < b * m; ++j) {
      sum += (float)grad_output[j * n + i];
    }
    grad_bias[i] = MLFloat16(sum);
  }
}

TEST(Fp8LinearGradOpTest, Gemm) {
  OpTester test("FP8LinearGrad", 1, "com.microsoft");
  int m = 32, n = 16, k = 16;
  std::vector<MLFloat16> grad_output;
  std::vector<Float8E4M3FN> input_t;
  std::vector<Float8E4M3FN> weight_t;
  std::vector<float> scale_inv;
  std::vector<MLFloat16> grad_input;
  std::vector<MLFloat16> grad_weight;
  std::vector<MLFloat16> grad_bias;
  PrepareFp8LinearGradData(1, m, n, k, grad_output, input_t, weight_t, scale_inv, grad_input, grad_weight, grad_bias);

  test.AddInput<MLFloat16>("grad_output", {m, n}, grad_output);
  test.AddInput<Float8E4M3FN>("input_t", {k, m}, input_t);
  test.AddInput<Float8E4M3FN>("weight_t", {k, n}, weight_t);
  test.AddInput<float>("scale_inv", {3}, scale_inv);
  test.AddOutput<MLFloat16>("grad_input", {m, k}, grad_input, false, 0.05f);
  test.AddOutput<MLFloat16>("grad_weight", {n, k}, grad_weight, false, 0.05f);
  test.AddOutput<MLFloat16>("grad_bias", {n}, grad_bias);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(Fp8LinearGradOpTest, BatchGemm) {
  OpTester test("FP8LinearGrad", 1, "com.microsoft");
  int b = 16, m = 32, n = 16, k = 16;
  std::vector<MLFloat16> grad_output;
  std::vector<Float8E4M3FN> input_t;
  std::vector<Float8E4M3FN> weight_t;
  std::vector<float> scale_inv;
  std::vector<MLFloat16> grad_input;
  std::vector<MLFloat16> grad_weight;
  std::vector<MLFloat16> grad_bias;
  PrepareFp8LinearGradData(b, m, n, k, grad_output, input_t, weight_t, scale_inv, grad_input, grad_weight, grad_bias);

  test.AddInput<MLFloat16>("grad_output", {b, m, n}, grad_output);
  test.AddInput<Float8E4M3FN>("input_t", {k, b * m}, input_t);
  test.AddInput<Float8E4M3FN>("weight_t", {k, n}, weight_t);
  test.AddInput<float>("scale_inv", {3}, scale_inv);
  test.AddOutput<MLFloat16>("grad_input", {b, m, k}, grad_input, false, 0.05f);
  test.AddOutput<MLFloat16>("grad_weight", {n, k}, grad_weight, false, 0.05f);
  test.AddOutput<MLFloat16>("grad_bias", {n}, grad_bias);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // ENABLE_FP8_TRAINING
