#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

// NB: Be sure to run DCE before fusion, because dead instructions
// can prevent fusion opportunities from being exploited.
// On Windows will noop, NYI
TORCH_API void CudaFuseGraph(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
