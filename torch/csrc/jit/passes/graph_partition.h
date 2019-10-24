#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

// \brief GraphPartition follows the implementation of GraphFuser 
// It mutates graph in-place, using a node-level callback to determine the
// inclusion of nodes in a subgraph.
//
// The motivatoin of separate implementation is to relaxing the requirements
// imposed by GraphFusion, which is not necessary for simple partitioning.
//
// \arg graph The graph to be modified in-place
// \arg kind The label given to the resultant fused subgraph
// \arg fn A callback run on each fusable node in the graph.
TORCH_API void GraphPartition(
    std::shared_ptr<Graph>& graph,
		Symbol kind,
		std::function<bool(Node*)> fn);

} // namespace jit
} // namespace torch
