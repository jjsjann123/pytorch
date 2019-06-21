/** \brief This file defines passes used for mixed precision conversion.
 *
 * The passes have python-bindings and can be invoked directly for debug.
 * Eventually it should be integrated into the default optimization pipeline
 * enabling through explicit flags.
 */
#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

TORCH_API void insertCastingNode(std::shared_ptr<Graph>& graph);

TORCH_API void removeConsecutiveCastingNode(std::shared_ptr<Graph>& graph);

TORCH_API void insertCastingWithScalingNode(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
