
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

static void mySoftmax(Fusion* fusion,
                        const DataType kDtype,
                        const int kNumberOfDims,
                        const int kReductionAxis) {
  FusionGuard fg(fusion);

  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  broadcast_mask[kReductionAxis] = true;

  auto x = TensorViewBuilder().ndims(kNumberOfDims).dtype(kDtype).build();
  auto y = TensorViewBuilder().ndims(kNumberOfDims).dtype(kDtype).build();

  // Pointwise Fusion
  auto input = add(x,y);

  auto max_val = max(input, {kReductionAxis});
  auto bcast_max = broadcast(max_val, broadcast_mask);
  auto x_max_sub = sub(input, bcast_max);
  auto exp = unaryOp(UnaryOpType::Exp, x_max_sub);
  auto sum_exp = sum(exp, {kReductionAxis});
  auto bcast_sum = broadcast(sum_exp, broadcast_mask);
  auto output = div(exp, bcast_sum);

  fusion->addInput(x);
  fusion->addInput(y);
  fusion->addOutput(output);
}

static void myBatchNorm(Fusion* fusion,
                        const DataType kDtype,
                        const int kNumberOfDims) {
  FusionGuard fg(fusion);

  const float kEps = 1e-5;
  auto x = TensorViewBuilder().ndims(kNumberOfDims).dtype(kDtype).build();
  auto y = TensorViewBuilder().ndims(kNumberOfDims).dtype(kDtype).build();
  auto weight = TensorViewBuilder().ndims(1).dtype(kDtype).build();
  auto bias = TensorViewBuilder().ndims(1).dtype(kDtype).build();

  // Pointwise Fusion
  auto input = add(x,y);

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

  fusion->addInput(x);
  fusion->addInput(y);
  fusion->addInput(weight);
  fusion->addInput(bias);
  fusion->addOutput(norm_gamma_bias);
}

static void myLayerNorm(Fusion* fusion,
                        const DataType kDtype,
                        const int kNumberOfDims,
                        std::vector<int64_t>& norm_shape) {
  FusionGuard fg(fusion);

  const float kEps = 1e-5;
  auto x = TensorViewBuilder().ndims(kNumberOfDims).dtype(kDtype).build();
  auto y = TensorViewBuilder().ndims(kNumberOfDims).dtype(kDtype).build();

  // Pointwise Fusion
  auto input = add(x,y);

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

  fusion->addInput(x);
  fusion->addInput(y);
  fusion->addOutput(output);
}


//------------------------------------------------------------------------------

static void MagicScheduler_Softmax(benchmark::State& benchmark_state) {
  Fusion fusion;

  std::vector<int64_t> input_shape{benchmark_state.range(0), benchmark_state.range(1)};
  const int kReductionAxis = benchmark_state.range(2);

  // setup fusion
  mySoftmax(
      &fusion,
      DataType::Float,
      input_shape.size(),
      kReductionAxis);

  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);
  std::vector<c10::IValue> inputs({at_x, at_y});

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
  std::vector<int64_t> input_shape{benchmark_state.range(0), benchmark_state.range(1)};
  const int kReductionAxis = benchmark_state.range(2);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
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
    auto input = at::add(at_x, at_y);
    auto output = at::_softmax(input, kReductionAxis, false);

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

  std::vector<int64_t> input_shape{64, benchmark_state.range(0), 35, 45};

  // setup fusion
  myBatchNorm(
      &fusion,
      DataType::Float,
      input_shape.size());

  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  std::vector<c10::IValue> inputs({at_x, at_y, at_weight, at_bias});

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
  std::vector<int64_t> input_shape{64, benchmark_state.range(0), 35, 45};

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);
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
    auto input = at::add(at_x, at_y);
    auto output = at::batch_norm(
        input,
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

  std::vector<int64_t> input_shape{656, benchmark_state.range(0)};
  const int kReductionAxis = 1;
  std::vector<int64_t> norm_shape;
  for(int idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }

  // setup fusion
  myLayerNorm(
      &fusion,
      DataType::Float,
      input_shape.size(),
      norm_shape);

  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);
  std::vector<c10::IValue> inputs({at_x, at_y});

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
    auto input = at::add(at_x, at_y);
    auto output = at::layer_norm(input, norm_shape);

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
          -> Ranges({{8, 8 << 10}})
          -> Unit(benchmark::kMicrosecond)
          -> UseManualTime();

BENCHMARK(MagicScheduler_BatchNorm_Baseline)
          -> RangeMultiplier(2)
          -> Ranges({{8, 8 << 10}})
          -> Unit(benchmark::kMicrosecond)
          -> UseManualTime();

BENCHMARK(MagicScheduler_Softmax)
          -> RangeMultiplier(2)
          -> Ranges({{656, 656}, {8, 8 << 13}, {1, 1}})
          -> Unit(benchmark::kMicrosecond)
          -> UseManualTime();

BENCHMARK(MagicScheduler_Softmax_Baseline)
          -> RangeMultiplier(2)
          -> Ranges({{656, 656}, {8, 8 << 13}, {1, 1}})
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

//------------------------------------------------------------------------------