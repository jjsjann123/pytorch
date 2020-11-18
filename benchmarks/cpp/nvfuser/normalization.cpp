
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

using namespace torch::jit::fuser::cuda;

static std::vector<c10::IValue> setupInputs(
    c10::IntArrayRef input_shape,
    c10::ScalarType kDtype) {
  at::manual_seed(0);

  auto options = at::TensorOptions().dtype(kDtype).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);

  return {at_x, at_y};
}

static void analyzeFusion(
    Fusion* fusion,
    std::vector<TensorView*>& reduction_tv,
    std::vector<TensorView*>& other_tv) {
  auto all_values = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, fusion->outputs());

  for (auto tv : ir_utils::filterByType<TensorView>(all_values)) {
    if (tv->hasReduction()) {
      reduction_tv.push_back(tv);
    } else if (!fusion->hasInput(tv)) {
      other_tv.push_back(tv);
    }
  }
}

static TensorView* mySoftmax(Fusion* fusion,
                        TensorView* input,
                        const int kNumberOfDims,
                        const int kReductionAxis) {
  FusionGuard fg(fusion);

  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  broadcast_mask[kReductionAxis] = true;

  auto max_val = max(input, {kReductionAxis});
  auto bcast_max = broadcast(max_val, broadcast_mask);
  auto x_max_sub = sub(input, bcast_max);
  auto exp = unaryOp(UnaryOpType::Exp, x_max_sub);
  auto sum_exp = sum(exp, {kReductionAxis});
  auto bcast_sum = broadcast(sum_exp, broadcast_mask);
  auto output = div(exp, bcast_sum);
  return output;
}

static TensorView* myBatchNorm(Fusion* fusion,
                        TensorView* input,
                        TensorView* weight,
                        TensorView* bias,
                        const int kNumberOfDims) {
  FusionGuard fg(fusion);

  const float kEps = 1e-5;
  std::vector<int> reduction_axes;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  torch::jit::fuser::cuda::Val* num_features = nullptr;
  for (size_t axis = 0; axis < kNumberOfDims; ++axis) {
    if (axis != 1) {
      reduction_axes.push_back(axis);
      broadcast_mask[axis] = true;
      num_features = (axis == 0)
          ? input->domain()->domain()[0]->extent()
          : mul(num_features, input->domain()->domain()[axis]->extent());
    }
  }

  auto x_sum = sum(input, reduction_axes);
  auto x_sum_bcast = broadcast(x_sum, broadcast_mask);
  auto x_mean = div(x_sum_bcast, num_features);

  auto x_mean_sub = sub(input, x_mean);
  auto x_mean_sub_pow = mul(x_mean_sub, x_mean_sub);
  auto var_sum = sum(x_mean_sub_pow, reduction_axes);
  auto var_sum_bcast = broadcast(var_sum, broadcast_mask);
  auto var = div(var_sum_bcast, num_features);

  auto var_eps = add(var, new Float(kEps));
  auto rvar = unaryOp(UnaryOpType::Rsqrt, var_eps);
  auto norm = mul(x_mean_sub, rvar);

  auto weight_bcast = broadcast(weight, broadcast_mask);
  auto bias_bcast = broadcast(bias, broadcast_mask);
  auto norm_gamma = mul(norm, weight_bcast);
  auto norm_gamma_bias = add(norm_gamma, bias_bcast);
  return norm_gamma_bias;
}

static TensorView* myLayerNorm(Fusion* fusion,
                        TensorView* input,
                        const int kNumberOfDims,
                        std::vector<int64_t>& norm_shape) {
  FusionGuard fg(fusion);

  const float kEps = 1e-5;
  std::vector<int> reduction_axes(norm_shape.size());
  std::vector<bool> broadcast_mask(input->nDims(), false);
  torch::jit::fuser::cuda::Val* num_features = nullptr;
  for (int idx = 0; idx < norm_shape.size(); ++idx) {
    const int axis = input->nDims() - 1 - idx;
    reduction_axes[idx] = axis;
    broadcast_mask[axis] = true;
    num_features = (num_features == nullptr)
        ? input->domain()->domain()[axis]->extent()
        : mul(num_features, input->domain()->domain()[axis]->extent());
  }

  // Reduction
  auto x_sum = sum(input, reduction_axes);
  // Broadcast
  auto x_sum_bcast = broadcast(x_sum, broadcast_mask);
  // Point-wise
  auto x_mean = div(x_sum_bcast, num_features);
  auto x_mean_sub = sub(input, x_mean);

  auto x_mean_sub_pow = mul(x_mean_sub, x_mean_sub);
  // Reduction
  auto var_sum = sum(x_mean_sub_pow, reduction_axes);
  // Broadcast
  auto var_sum_bcast = broadcast(var_sum, broadcast_mask);
  // Point-wise
  auto var = div(var_sum_bcast, num_features);
  auto var_eps = add(var, new Float(kEps));
  auto rvar = unaryOp(UnaryOpType::Rsqrt, var_eps);
  auto output = mul(x_mean_sub, rvar);
  return output;
}


//------------------------------------------------------------------------------

static void MagicScheduler_Softmax(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{benchmark_state.range(1), benchmark_state.range(0)};
  const int kReductionAxis = benchmark_state.range(2);

  // setup fusion
  auto input = TensorViewBuilder().ndims(input_shape.size()).dtype(DataType::Float).build();
  fusion.addInput(input);
  auto output = mySoftmax(&fusion, input, input_shape.size(), kReductionAxis);
  fusion.addOutput(output);

  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  std::vector<c10::IValue> inputs({at_x});

  // outputs
  std::vector<at::Tensor> outputs;

  auto reduction_params =
      getMultipleReductionHeuristics(&fusion, inputs, reduction_tensors);
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  scheduleMultipleReduction(
      &fusion, reduction_params.value(), reduction_tensors, other_tensors);

  FusionExecutor executor;
  executor.setMeasureKernelTimeFlag(true);
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(
        c10::ArrayRef<c10::IValue>(inputs), reduction_params.value().lparams);
    benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
    cudaDeviceSynchronize();
  }
}

static void MagicScheduler_Softmax_Baseline(benchmark::State& benchmark_state) {
  std::vector<int64_t> input_shape{benchmark_state.range(1), benchmark_state.range(0)};
  const int kReductionAxis = benchmark_state.range(2);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    // Create
    float kernel_time_ms_ = 0;
    cudaEvent_t start_event = {};
    cudaEvent_t finish_event = {};

    // Setup
    cudaEventCreate(&start_event);
    cudaEventCreate(&finish_event);
    cudaEventRecord(start_event);

    // Run
    auto output = at::_softmax(at_x, kReductionAxis, false);

    // Record
    cudaEventRecord(finish_event);
    cudaEventSynchronize(start_event);
    cudaEventSynchronize(finish_event);
    cudaEventElapsedTime(&kernel_time_ms_, start_event, finish_event);

    benchmark_state.SetIterationTime(kernel_time_ms_ / 1000.0);
    cudaDeviceSynchronize();
  }
}

static void MagicScheduler_BatchNorm(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{
      32,
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1)};

  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  auto weight = TensorViewBuilder().ndims(1).dtype(DataType::Float).build();
  auto bias = TensorViewBuilder().ndims(1).dtype(DataType::Float).build();
  fusion.addInput(input);
  fusion.addInput(weight);
  fusion.addInput(bias);

  auto output = myBatchNorm(&fusion, input, weight, bias, input_shape.size());
  fusion.addOutput(output);

  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  std::vector<c10::IValue> inputs({at_x, at_weight, at_bias});

  // outputs
  std::vector<at::Tensor> outputs;

  auto reduction_params =
      getMultipleReductionHeuristics(&fusion, inputs, reduction_tensors);
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  scheduleMultipleReduction(
      &fusion, reduction_params.value(), reduction_tensors, other_tensors);

  FusionExecutor executor;
  executor.setMeasureKernelTimeFlag(true);
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(
        c10::ArrayRef<c10::IValue>(inputs), reduction_params.value().lparams);
    benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
    cudaDeviceSynchronize();
  }
}

static void MagicScheduler_BatchNorm_Baseline(benchmark::State& benchmark_state) {
  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  std::vector<int64_t> input_shape{
      32,
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1)};

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  at::Tensor at_mean = at::zeros({input_shape[1]}, options);
  at::Tensor at_var = at::ones({input_shape[1]}, options);

  auto ato_weight = c10::optional<at::Tensor>(at_weight);
  auto ato_bias = c10::optional<at::Tensor>(at_bias);
  auto ato_running_mean = c10::optional<at::Tensor>(at_mean);
  auto ato_running_var = c10::optional<at::Tensor>(at_var);

  cudaDeviceSynchronize();

  for (auto _ : benchmark_state) {
    // Create
    float kernel_time_ms_ = 0;
    cudaEvent_t start_event = {};
    cudaEvent_t finish_event = {};

    // Setup
    cudaEventCreate(&start_event);
    cudaEventCreate(&finish_event);
    cudaEventRecord(start_event);

    // Run
    auto output = at::batch_norm(
        at_x,
        ato_weight,
        ato_bias,
        ato_running_mean,
        ato_running_var,
        true,
        kMomentum,
        kEps,
        false);

    // Record
    cudaEventRecord(finish_event);
    cudaEventSynchronize(start_event);
    cudaEventSynchronize(finish_event);
    cudaEventElapsedTime(&kernel_time_ms_, start_event, finish_event);

    benchmark_state.SetIterationTime(kernel_time_ms_ / 1000.0);
    cudaDeviceSynchronize();
  }
}

static void MagicScheduler_LayerNorm(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{656, benchmark_state.range(0)};
  const int kReductionAxis = 1;
  std::vector<int64_t> norm_shape;
  for(int idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }

  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  fusion.addInput(input);
  auto output = myLayerNorm(&fusion, input, input_shape.size(), norm_shape);
  fusion.addOutput(output);

  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  std::vector<c10::IValue> inputs({at_x});

  // outputs
  std::vector<at::Tensor> outputs;

  auto reduction_params =
      getMultipleReductionHeuristics(&fusion, inputs, reduction_tensors);
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  scheduleMultipleReduction(
      &fusion, reduction_params.value(), reduction_tensors, other_tensors);

  FusionExecutor executor;
  executor.setMeasureKernelTimeFlag(true);
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(
        c10::ArrayRef<c10::IValue>(inputs), reduction_params.value().lparams);
    benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
    cudaDeviceSynchronize();
  }
}

static void MagicScheduler_LayerNorm_Baseline(benchmark::State& benchmark_state) {
  std::vector<int64_t> input_shape{656, benchmark_state.range(0)};
  const int kReductionAxis = 1;
  std::vector<int64_t> norm_shape;
  for(int idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    // Create
    float kernel_time_ms_ = 0;
    cudaEvent_t start_event = {};
    cudaEvent_t finish_event = {};

    // Setup
    cudaEventCreate(&start_event);
    cudaEventCreate(&finish_event);
    cudaEventRecord(start_event);

    // Run
    auto output = at::layer_norm(at_x, norm_shape);

    // Record
    cudaEventRecord(finish_event);
    cudaEventSynchronize(start_event);
    cudaEventSynchronize(finish_event);
    cudaEventElapsedTime(&kernel_time_ms_, start_event, finish_event);

    benchmark_state.SetIterationTime(kernel_time_ms_ / 1000.0);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(MagicScheduler_BatchNorm)
          -> RangeMultiplier(2)
          -> Ranges({{64, 1024}, {8, 256}})
          -> Unit(benchmark::kMicrosecond)
          -> UseManualTime();

BENCHMARK(MagicScheduler_BatchNorm_Baseline)
          -> RangeMultiplier(2)
          -> Ranges({{64, 1024}, {8, 256}})
          -> Unit(benchmark::kMicrosecond)
          -> UseManualTime();

BENCHMARK(MagicScheduler_LayerNorm)
          -> RangeMultiplier(2)
          -> Ranges({{8, 8 << 13}})
          -> Unit(benchmark::kMicrosecond)
          -> UseManualTime();

BENCHMARK(MagicScheduler_LayerNorm_Baseline)
          -> RangeMultiplier(2)
          -> Ranges({{8, 8 << 13}})
          -> Unit(benchmark::kMicrosecond)
          -> UseManualTime();

BENCHMARK(MagicScheduler_Softmax)
          -> RangeMultiplier(2)
          -> Ranges({{656, 656}, {8, 8 << 13}, {0, 1}})
          -> Unit(benchmark::kMicrosecond)
          -> UseManualTime();

BENCHMARK(MagicScheduler_Softmax_Baseline)
          -> RangeMultiplier(2)
          -> Ranges({{656, 656}, {8, 8 << 13}, {0, 1}})
          -> Unit(benchmark::kMicrosecond)
          -> UseManualTime();

//------------------------------------------------------------------------------

static void MagicScheduler_Softmax_Dropout(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{256,12,100, benchmark_state.range(0)};
  const int kReductionAxis = 3;

  constexpr int kHiddenSize = 768;
  constexpr int kNumAttentionHeads = 12;
  constexpr int kAttentionHeadSize = kHiddenSize / kNumAttentionHeads;
  constexpr float kDropoutProbability = 0.9;

  // setup fusion
  auto attention_scores = TensorViewBuilder()
                              .ndims(input_shape.size())
                              .dtype(DataType::Float)
                              .build();
  auto attention_mask = TensorViewBuilder()
                            .ndims(input_shape.size())
                            .dtype(DataType::Float)
                            .build();
  Float* divisor = new Float();
  fusion.addInput(attention_scores);
  fusion.addInput(attention_mask);
  fusion.addInput(divisor);

  attention_scores = div(attention_scores, divisor);
  attention_scores = add(attention_scores, attention_mask);
  auto attention_probs =
      mySoftmax(&fusion, attention_scores, input_shape.size(), kReductionAxis);
  auto random = unaryOp(UnaryOpType::RandLike, attention_probs);
  auto mask = binaryOp(
      BinaryOpType::LT, random, new Float(kDropoutProbability));
  auto float_mask = castOp(DataType::Float, mask);
  auto dropout = mul(attention_probs, float_mask);
  auto output = mul(dropout, new Float(1.0f / kDropoutProbability));

  fusion.addOutput(attention_scores);
  fusion.addOutput(attention_probs);
  fusion.addOutput(mask);
  fusion.addOutput(output);

  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_scores = at::randn(input_shape, options);
  at::Tensor at_mask = at::randn(input_shape, options);
  std::vector<c10::IValue> inputs({at_scores, at_mask, sqrt(kAttentionHeadSize)});

  // outputs
  std::vector<at::Tensor> outputs;

  auto reduction_params =
      getMultipleReductionHeuristics(&fusion, inputs, reduction_tensors);
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  scheduleMultipleReduction(
      &fusion, reduction_params.value(), reduction_tensors, other_tensors);

  FusionExecutor executor;
  executor.setMeasureKernelTimeFlag(true);
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(
        c10::ArrayRef<c10::IValue>(inputs), reduction_params.value().lparams);
    benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
    cudaDeviceSynchronize();
  }
}

static void MagicScheduler_Softmax_Dropout_Baseline(benchmark::State& benchmark_state) {
  std::vector<int64_t> input_shape{256,12,100, benchmark_state.range(0)};
  const int kReductionAxis = 3;

  constexpr int kHiddenSize = 768;
  constexpr int kNumAttentionHeads = 12;
  constexpr float kDropoutProbability = 0.1;
  constexpr int kAttentionHeadSize = kHiddenSize / kNumAttentionHeads;

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor attention_scores = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);

  cudaDeviceSynchronize();

  for (auto _ : benchmark_state) {
    // Create
    float kernel_time_ms_ = 0;
    cudaEvent_t start_event = {};
    cudaEvent_t finish_event = {};

    // Setup
    cudaEventCreate(&start_event);
    cudaEventCreate(&finish_event);
    cudaEventRecord(start_event);

    // Run
    attention_scores = attention_scores / sqrt(kAttentionHeadSize);
    attention_scores = attention_scores + at_y;
    auto attention_probs = at::_softmax(attention_scores, kReductionAxis, false);
    attention_probs = at::dropout(attention_probs, kDropoutProbability, true);

    // Record
    cudaEventRecord(finish_event);
    cudaEventSynchronize(start_event);
    cudaEventSynchronize(finish_event);
    cudaEventElapsedTime(&kernel_time_ms_, start_event, finish_event);

    benchmark_state.SetIterationTime(kernel_time_ms_ / 1000.0);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(MagicScheduler_Softmax_Dropout)
          ->Arg(8)
          ->Arg(16)
          ->Arg(24)
          ->Arg(32)
          ->Arg(40)
          ->Arg(48)
          ->Arg(56)
          ->Arg(64)
          ->Arg(72)
          ->Arg(80)
          ->Arg(88)
          ->Arg(96)
          ->Arg(104)
          ->Arg(112)
          ->Arg(120)
          ->Arg(128)
          -> Unit(benchmark::kMicrosecond)
          -> UseManualTime();

BENCHMARK(MagicScheduler_Softmax_Dropout_Baseline)
          ->Arg(8)
          ->Arg(16)
          ->Arg(24)
          ->Arg(32)
          ->Arg(40)
          ->Arg(48)
          ->Arg(56)
          ->Arg(64)
          ->Arg(72)
          ->Arg(80)
          ->Arg(88)
          ->Arg(96)
          ->Arg(104)
          ->Arg(112)
          ->Arg(120)
          ->Arg(128)
          -> Unit(benchmark::kMicrosecond)
          -> UseManualTime();

//------------------------------------------------------------------------------