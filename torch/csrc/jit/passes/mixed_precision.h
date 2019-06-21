/** \brief This file defines auto cast optimization pass that allows mixed
 * precision training
 *
 * The passes have python-bindings and can be invoked directly for debug.
 */
#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/passes/alias_analysis.h>

namespace torch {
namespace jit {

enum class TraversalDirection {
  FollowInputs,
  FollowOutputs,
  FollowInputsAndOutputs,
};

/**
 * Graph Traversal/Marking utility
 *
 * Traces input/output through restricted edges (Values) across nested blocks.
 * This utility allows us to navigate across blocks introduced by branch/loop in
 * JIT IR.
 *
 * Note:
 * For efficiency reason, we cache queries in our traversal and assumes the
 * graph is not altered along the life time of the instance of
 * `GraphTraversalUtil`. This serves the requirement of our graph painting,
 * which annotates the partition through a separate lists, instead of mutating
 * the graph in-place.
 */
class GraphTraversalUtil {
public:
  typedef std::unordered_set<const Node*> NodeSet;
  typedef void (*ProcessNodeFn)(const Node*);
  typedef bool (*CheckNodeFn)(const Node*);
  typedef bool (*CheckValueFn)(const Value*);

  // Instantiate GraphTraversalUtil with specified `CheckValueFn fn`
  // fn restrict the traversal to edges (Value) that returns true on fn;
  TORCH_API explicit GraphTraversalUtil(
      std::shared_ptr<Graph> graph,
      CheckValueFn fn = nullptr);

  ~GraphTraversalUtil() = default;

  // Traverse the graph through specified edges and apply mark_node_fn on
  // visited nodes, return a set of nodes that has been marked;
  // providing further control knobs:
  //   1. enter_node_fn:
  //     returns true if traversal include current node;
  //   2. should_mark_node_fn
  //     returns true if traversal `mark_node_fn` should be applied on current
  //     node;
  //   3. mark_node_fn
  //     function that processes nodes;
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
  // Follow edge in the specified direction to return all connected nodes;
  // Forward calls to corresponding followInput/followOutput.
  NodeSet followEdges(const Node* n, TraversalDirection direction);

  // Follow input across nested blocks to connected nodes;
  void followInput(const Value* v, NodeSet &result, ValueSet &value_set);

  // Follow output across nested blocks to connected nodes;
  void followOutput(const Value* v, NodeSet &result, ValueSet &value_set);

  // Traverse graph from node n:
  //   1. only enter node that returns true on `enter_node_fn`;
  //   2. apply `mark_node_fn` to each visited node that returns true on
  //      `should_mark_node_fn`;
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

  std::shared_ptr<Graph> graph_;
  std::unordered_map<const Node*, NodeSet> input_maps_;
  std::unordered_map<const Node*, NodeSet> output_maps_;
  CheckValueFn enter_edge_fn_;
};

// Auto casting optimization pass used for automatic mixed precision:
// This optimization pass mutates the graph in-place. The pass inserts cast ops
// that allows certain operations to execute computation with reduced precision
TORCH_API void AmpAutoCasting(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
