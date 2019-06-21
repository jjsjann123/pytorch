#include <stack>

#include <torch/csrc/jit/passes/mixed_precision.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <torch/csrc/jit/pass_manager.h>

namespace torch {
namespace jit {

using ::c10::Symbol;

#define DEBUG false

namespace {

#define ASYNC_FLAG true

std::unordered_set<Symbol> WhiteList = {
  aten::relu,
  aten::addmm,
  aten::matmul,
  aten::mm,
  aten::mul
};

std::unordered_set<Symbol> BlackList = {
  aten::tanh,
  aten::batch_norm,
  // temporary hack
  prim::TupleConstruct,
  aten::softmax,
  // apparently plain `aten::tanh_` doesn't work. no idea why
  Symbol::fromQualString("aten::tanh_"),
  Symbol::fromQualString("aten::pow_"),
  Symbol::fromQualString("aten::pow")
};

template<bool preorder, typename func_t>
void graphTopoOrderTraversal(Block* block, const func_t& f) {
  for (auto node : block->nodes()) {
    // apply transform per node;
    if (preorder)
      f(node);
    for (auto sub_block : node->blocks()) {
      graphTopoOrderTraversal<preorder>(sub_block, f);
    }
    if (!preorder)
      f(node);
  }
}

} // namespace

GraphTraversalUtil::~GraphTraversalUtil() = default;

GraphTraversalUtil::GraphTraversalUtil(std::shared_ptr<Graph> graph, CheckValueFn fn)
    : graph_(std::move(graph)), enter_edge_fn_(fn) {
  input_maps_.clear();
  output_maps_.clear();
}

GraphTraversalUtil::NodeSet GraphTraversalUtil::followEdges(
    const Node* n, TraversalDirection direction) {
  NodeSet res;
  if (DEBUG) {
    std::cout << "check node: " << *n << std::endl;
  }
  if (direction == TraversalDirection::FollowInputs ||
      direction == TraversalDirection::FollowInputsAndOutputs) {
    if (input_maps_.count(n) == 0) {
      if (DEBUG) {
        std::cout << "no caching, search input:" << std::endl;
      }
      NodeSet inputs;
      ValueSet visited_values;
      for (auto v : n->inputs()) {
        followInput(v, inputs, visited_values);
      }
      input_maps_[n] = inputs;
      if (DEBUG) {
        std::cout << "                  found :" << inputs.size() << std::endl;
      }
    }
    res.insert(input_maps_[n].begin(), input_maps_[n].end());
  }
  if (direction == TraversalDirection::FollowOutputs ||
      direction == TraversalDirection::FollowInputsAndOutputs) {
    if (output_maps_.count(n) == 0) {
      if (DEBUG) {
        std::cout << "no caching, search output:" << std::endl;
      }
      NodeSet outputs;
      ValueSet visited_values;
      for (auto v : n->outputs()) {
        followOutput(v, outputs, visited_values);
      }
      output_maps_[n] = outputs;
      if (DEBUG) {
        std::cout << "                  found :" << outputs.size() << std::endl;
      }
    }
    res.insert(output_maps_[n].begin(), output_maps_[n].end());
  }
  return res;
}

void GraphTraversalUtil::followInput(
    const Value* v, NodeSet &result, ValueSet &visited_values) {
  // breaking from re-entry and check for entry condition;
  if (visited_values.count(v) != 0 ||
      (enter_edge_fn_ != nullptr && !enter_edge_fn_(v))) {
    return;
  }
  visited_values.insert(v);

  if (v->node()->kind() == prim::If) {
    // prim::If redirect edge to corresponding outputs in both nested blocks;
    auto index = v->offset();

    // propagate through subblocks, jumping from parent node to
    // output Value inside each subblocks
    for (auto block : v->node()->blocks()) {
      // tracing corresponding value through inputs to param::Return
      auto v_next = block->outputs()[index];
      // recursively call followInput, in case we have nested blocks;
      followInput(v_next, result, visited_values);
    }
  } else if (v->node()->kind() == prim::Loop) {
    // prim::Loop redirects edge to a couple tensors:
    // 1) output of nested block
    // 2) input of the prim::Loop node;

    // redirecting input to 1) output of nested block
    auto block = v->node()->blocks()[0];
    // tracing corresponding value through inputs to param::Return
    auto v_next = block->outputs()[v->offset() + 1];
    // recursively call followInput, in case we have nested blocks;
    followInput(v_next, result, visited_values);

    // redirecting input to 2) input of the prim::Loop node;
    v_next = v->node()->inputs()[v->offset() + 2];
    // recursively call followInput, in case we have nested blocks;
    followInput(v_next, result, visited_values);
  } else if (v->node()->kind() == prim::Param) {
    // We would encounter a prim::Param in 2 cases:
    //   1) We are at the root block of a given graph;
    //   2) We are inside a block attached to a prim::Loop node;
    //
    // We only need propagation for block attached to prim::Loop
    auto block_nested = v->node()->owningBlock();
    if (block_nested->owningNode() &&
        block_nested->owningNode()->kind() == prim::Loop) {
      // cut off propagation when it's tracing indexing
      if (v->offset() != 0) {
        // block inside prim::Loop redirects edge to a couple tensors:
        // 1) output of current block
        // 2) input of parent prim::Loop node;

        // propagation along 1) output of current block
        // recursively call followInput, in case we have nested blocks;
        followInput(block_nested->outputs()[v->offset()], result, visited_values);

        // propagation along 2) input of parent prim::Loop node
        // recursively call followInput, in case we have nested blocks;
        followInput(
            block_nested->owningNode()->inputs()[v->offset()+1],
            result,
            visited_values);
      }
    } else {
      // assert that we are at the root block, we don't want silent errors;
      AT_ASSERT(block_nested == graph_->block());
    }
  } else {
    result.emplace(v->node());
  }
}

void GraphTraversalUtil::followOutput(
    const Value* v, NodeSet &result, ValueSet &visited_values) {
  // breaking from re-entry and check for entry condition;
  if (visited_values.count(v) != 0 ||
      (enter_edge_fn_ != nullptr && !enter_edge_fn_(v))) {
    return;
  }
  visited_values.insert(v);

  for (auto use : v->uses()) {
    
    if (use.user->kind() == prim::If) {
      // skip propagation along prim::If input (not a data flow)
    } else if (use.user->kind() == prim::Loop) {
      // skip propagation along first two inputs of prim::Loop (not a data flow)
      if (use.offset > 1) {
        // prim::Loop redirects edge to a couple tensors:
        // 1) input of nested block
        // 2) output of the prim::Loop node;

        // redirecting propagation along 1) input of nested block
        auto block = use.user->blocks()[0];
        // tracing corresponding value through inputs to param::Return
        auto v_next = block->inputs()[use.offset - 1];
        // recursively call followOutput, in case we have nested blocks;
        followOutput(v_next, result, visited_values);

        // redirecting propagation along 2) output of the prim::Loop node;
        v_next = use.user->outputs()[use.offset - 2];
        // recursively call followOutput, in case we have nested blocks;
        followOutput(v_next, result, visited_values);
      }
    } else if (use.user->kind() == prim::Return) {
      // We would encounter a prim::Param in 3 cases:
      //   1) We are at the root block of a given graph;
      //   2) We are inside a block attached to a prim::If node;
      //   3) We are inside a block attached to a prim::Loop node;
      //
      // We only continue propagation under 2) and 3)
      auto block_nested = use.user->owningBlock();
      if (block_nested->owningNode()) {
        if (block_nested->owningNode()->kind() == prim::If) {
          // condition: 2) We are inside a block attached to a prim::If node
          // redirect propagation along the output of prim::If node
 
          // recursively call followOutput, in case we have nested blocks;
          followOutput(
              block_nested->owningNode()->outputs()[use.offset],
              result,
              visited_values);
        } else if (block_nested->owningNode()->kind() == prim::Loop) {
          // condition: 3) We are inside a block attached to a prim::Loop node;

          // skip propagation along first element (not a data flow)
          if (use.offset != 0) {
            // block inside prim::Loop redirects edge to a couple tensors:
            // 1) input of current block
            // 2) output of parent prim::Loop node;

            // propagation along 1) input of current block
            // recursively call followOutput, in case we have nested blocks;
            followOutput(block_nested->inputs()[use.offset], result, visited_values);

            // propagation along 2) output of parent prim::Loop node
            // recursively call followOutput, in case we have nested blocks;
            followOutput(
                block_nested->owningNode()->outputs()[use.offset-1],
                result,
                visited_values);
          }
        } else {
          // Exception: we have another node that has enclosed block; raise
          // error to avoid silent errors
        }
      } else {
        // condition: 1) We are at the root block of a given graph;
        // assert that we are at the root block, we don't want silent errors;
        AT_ASSERT(block_nested == graph_->block());
      }
    } else {
      result.emplace(use.user);
    }
  }
}

void graphPartitioning(std::shared_ptr<Graph>& graph) {
  /*partitionMixedPrecision(graph);*/
  //throw std::runtime_error("Pass not implemented yet!");
  GraphTraversalUtil gtu(
      graph,
      [](const Value* v) -> bool {
          return v->type()->isSubtypeOf(TensorType::get());
      }); 

  GraphTraversalUtil::NodeSet white_nodes;
  GraphTraversalUtil::NodeSet black_nodes;
  
  graphTopoOrderTraversal<true>(
      graph->block(),
      [&](const Node* n) {
        if (WhiteList.count(n->kind()) != 0) {
          white_nodes.insert(n);
        }
        if (BlackList.count(n->kind()) != 0) {
          black_nodes.insert(n);
        }
      });

  std::cout << "=========================initial paint according to list=========================" << std::endl;
  for (auto node : black_nodes) {
    std::cout << "black node: " << *node << std::endl;
  }
  for (auto node : white_nodes) {
    std::cout << "white node: " << *node << std::endl;
  }

  GraphTraversalUtil::NodeSet black_upstream_nodes;

  // Mark black_upstream_nodes
  graphTopoOrderTraversal<true>(
      graph->block(),
      [&](const Node* n) {
        if (black_nodes.count(n) != 0) {
          gtu.traverseFromNode(
              n,
              [&] (const Node* node) -> bool {
                if (node == n  ||
                    (black_nodes.count(node) == 0 &&
                     white_nodes.count(node) == 0 &&
                     black_upstream_nodes.count(node) == 0)) {
                  return true;
                } else {
                  return false;
                }
              },
              [&] (const Node* node) -> bool {
                if (black_nodes.count(node) == 0 &&
                    black_upstream_nodes.count(node) == 0) {
                  return true;
                } else {
                  return false;
                }
              },
              [&] (const Node* n) {
                black_upstream_nodes.insert(n);
              },
              TraversalDirection::FollowInputs);
        }
      });

  /*
  for (auto node : black_upstream_nodes) {
    std::cout << "black_upstream node: " << *node << std::endl;
  }
  */

  // Paint black nodes
  graphTopoOrderTraversal<true>(
      graph->block(),
      [&](const Node* n) {
        if (black_nodes.count(n) != 0) {
          gtu.traverseFromNode(
              n,
              [&] (const Node* node) -> bool {
                if (node == n  ||
                    (black_nodes.count(node) == 0 &&
                     white_nodes.count(node) == 0 &&
                     black_upstream_nodes.count(node) != 0)) {
                  return true;
                } else {
                  return false;
                }
              },
              [&] (const Node* node) -> bool {
                if (black_nodes.count(node) == 0 &&
                    black_upstream_nodes.count(node) != 0) {
                  return true;
                } else {
                  return false;
                }
              },
              [&] (const Node* n) {
                black_nodes.insert(n);
              },
              TraversalDirection::FollowOutputs);
        }
      });

  std::cout << "=========================final paint=========================" << std::endl;
  for (auto node : black_nodes) {
    std::cout << "black node: " << *node << std::endl;
  }

  // Paint white nodes
  graphTopoOrderTraversal<true>(
      graph->block(),
      [&](const Node* n) {
        if (white_nodes.count(n) != 0) {
          gtu.traverseFromNode(
              n,
              [&] (const Node* node) -> bool {
                if (node == n  ||
                    (black_nodes.count(node) == 0 &&
                     white_nodes.count(node) == 0)) {
                  return true;
                } else {
                  return false;
                }
              },
              [&] (const Node* node) -> bool {
                if (black_nodes.count(node) == 0 &&
                    white_nodes.count(node) == 0) {
                  return true;
                } else {
                  return false;
                }
              },
              [&] (const Node* n) {
                white_nodes.insert(n);
              },
              TraversalDirection::FollowInputsAndOutputs);
        }
      });

  for (auto node : white_nodes) {
    std::cout << "white node: " << *node << std::endl;
  }

  // Try fusing white nodes
  // TODO: we prolly want our own Symbol other than reusing the
  // prim::FusionGroup just to save us from confusion, although this temporary
  // node is not supposed to see the light of the day EVER.
  const auto amp_fp16_symbol = Symbol::fromQualString("prim::FusionGroup");
  CustomFuseGraph(
      graph,
      [&](Node* n) -> bool {
          if (white_nodes.count(n) != 0) {
              return true;
          }
          return false;
      },
      amp_fp16_symbol);

  // FuseGraph inserts bunch of shape & Broadcast ops inside, which is not
  // needed since we'll inline the fusion afterwards;
  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);
  std::cout << "=========================graph partitioning=========================" << std::endl;
  std::cout << *graph << std::endl;

  graphTopoOrderTraversal<true>(
      graph->block(),
      [&](const Node* n) {
        if (n->kind() == amp_fp16_symbol) {
          auto subgraph = n->g(attr::Subgraph);
          auto sync = subgraph->insertConstant(false);
          std::cout << (*subgraph);
          //vector<Node*> cast(subgraph->inputs().size());
          for (auto input : subgraph->inputs()) {
            auto n = subgraph->create(Symbol::fromQualString("aten::_cast_Half"));
            n->insertAfter(input->node());
            input->replaceAllUsesWith(n->outputs()[0]);
            n->addInput(input);
            n->addInput(sync);
          }

          for (auto output : subgraph->outputs()) {
            auto n = subgraph->create(Symbol::fromQualString("aten::_cast_Float"));
            n->insertAfter(output->node());
            output->replaceAllUsesWith(n->outputs()[0]);
            n->addInput(output);
            n->addInput(sync);
          }
          sync->node()->moveAfter(subgraph->inputs()[0]->node());
        }
      });

  std::cout << "=========================cast inserted=========================" << std::endl;
  std::cout << *graph << std::endl;

  std::cout << "=========================final graph=========================" << std::endl;
  bool any_changed = true;
  while (any_changed) {
    any_changed = false;
    for (auto it = graph->nodes().rbegin(); it != graph->nodes().rend();) {
      if (it->kind() == amp_fp16_symbol) {
        any_changed = true;
        // inline Subgraph and continue traversal on the last output node;
        it = ++inlineCallTo(*it, (*it->g(attr::Subgraph))).back()->node()->reverseIterator();
      } else {
        it++;
      }
    }
  }
  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);

  std::cout << *graph << std::endl;
}

}

} // namespace jit
} // namespace torch
