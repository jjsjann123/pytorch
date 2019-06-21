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

enum class TraversalDirection {
  FollowInputs,
  FollowOutputs,
  FollowInputsAndOutputs,
};

/**
 * Graph Traversal utility
 *
 * Traces input/output across nested blocks
 */
class GraphTraversalUtil {
public:
  typedef std::unordered_set<const Node*> NodeSet;
  typedef void (*ProcessNodeFn)(const Node*);
  typedef bool (*CheckNodeFn)(const Node*);
  typedef bool (*CheckValueFn)(const Value*);

  TORCH_API explicit GraphTraversalUtil(
      std::shared_ptr<Graph> graph, CheckValueFn fn = nullptr);
  TORCH_API ~GraphTraversalUtil();

  TORCH_API NodeSet followEdges(const Node* n, TraversalDirection direction);

  template<typename CheckNodeFn, typename CheckProcessNodeFn, typename ProcessNodeFn>
  TORCH_API NodeSet traverseFromNode(const Node* n,
      CheckNodeFn enter_node_fn,
      CheckProcessNodeFn should_mark_node_fn,
      ProcessNodeFn mark_node_fn,
      TraversalDirection direction) {
    NodeSet visited_nodes;
    NodeSet marked_nodes;
    traverse(n,
             enter_node_fn,
						 should_mark_node_fn,
						 mark_node_fn,
						 direction,
						 visited_nodes,
						 marked_nodes);
    return marked_nodes;
  }

private:
  template<typename CheckNodeFn, typename CheckProcessNodeFn, typename ProcessNodeFn>
  void traverse(const Node* n,
      CheckNodeFn enter_node_fn,
      CheckProcessNodeFn should_mark_node_fn,
      ProcessNodeFn mark_node_fn,
      TraversalDirection direction,
      NodeSet &visited_nodes,
      NodeSet &marked_nodes) {
    if (visited_nodes.count(n) != 0) {
      return;
    }
    visited_nodes.emplace(n);
    if (enter_node_fn(n)) {
      // marking nodes and adding it to marked_nodes;
      if (should_mark_node_fn(n) && marked_nodes.count(n) == 0) {
        mark_node_fn(n);
        marked_nodes.emplace(n);
      }

      auto next_nodes = followEdges(n, direction);
      for (auto node : next_nodes) {
        traverse(
            node,
            enter_node_fn,
						should_mark_node_fn,
            mark_node_fn,
            direction,
            visited_nodes,
            marked_nodes);
      }
    }
  }

  void followInput(const Value* v, NodeSet &result, ValueSet &value_set);
  void followOutput(const Value* v, NodeSet &result, ValueSet &value_set);

  std::shared_ptr<Graph> graph_;
  std::unordered_map<const Node*, NodeSet> input_maps_;
  std::unordered_map<const Node*, NodeSet> output_maps_;
  CheckValueFn enter_edge_fn_;
};

TORCH_API void graphPartitioning(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
