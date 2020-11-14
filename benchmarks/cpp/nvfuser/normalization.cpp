
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

using namespace torch::jit::fuser::cuda;

static void mySoftmax(Fusion* fusion,
                        const DataType kDtype,
                        const int kNumberOfDims,
                        const int kReductionAxis) {
  FusionGuard fg(fusion);

  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  broadcast_mask[kReductionAxis] = true;

  auto x = TensorViewBuilder().ndims(kNumberOfDims).dtype(kDtype).build();
  auto y = TensorViewBuilder().ndims(kNumberOfDims).dtype(kDtype).build();

  TensorView* input = add(x,y);
  TensorView* max_val = max(input, {kReductionAxis});
  TensorView* bcast_max = broadcast(max_val, broadcast_mask);
  TensorView* x_max_sub = sub(input, bcast_max);
  TensorView* exp = unaryOp(UnaryOpType::Exp, x_max_sub);
  TensorView* sum_exp = sum(exp, {kReductionAxis});
  TensorView* bcast_sum = broadcast(sum_exp, broadcast_mask);
  TensorView* output = div(exp, bcast_sum);

  fusion->addInput(x);
  fusion->addInput(y);
  fusion->addOutput(output);
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

static std::vector<c10::IValue> setupInputs(
    c10::IntArrayRef input_shape,
    c10::ScalarType kDtype) {
  at::manual_seed(0);

  auto options = at::TensorOptions().dtype(kDtype).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);

  return {at_x, at_y};
}

//------------------------------------------------------------------------------

static void MagicScheduler_Softmax(benchmark::State& benchmark_state) {
  Fusion fusion;

  std::vector<int64_t> input_shape{benchmark_state.range(0), benchmark_state.range(1)};
  const int kReductionAxis = benchmark_state.range(2);

  // setup fusion
  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  mySoftmax(
      &fusion,
      DataType::Float,
      input_shape.size(),
      kReductionAxis);

  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs(input_shape, at::kFloat);

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
    outputs = executor.runFusion(c10::ArrayRef<c10::IValue>(inputs));
    benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
    cudaDeviceSynchronize();
  }
}

static void MagicScheduler_Softmax_Baseline(benchmark::State& benchmark_state) {
  Fusion fusion;

  std::vector<int64_t> input_shape{benchmark_state.range(0), benchmark_state.range(1)};
  const int kReductionAxis = benchmark_state.range(2);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs(input_shape, at::kFloat);

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
    auto input = at::add(inputs[0].toTensor(), inputs[1].toTensor());
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

//------------------------------------------------------------------------------