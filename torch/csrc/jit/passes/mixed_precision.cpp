#include <stack>

#include <torch/csrc/jit/passes/mixed_precision.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/graph_partition.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/utils/memory.h>

#include <torch/csrc/jit/pass_manager.h>

namespace torch {
namespace jit {

using ::c10::Symbol;

namespace {

// white list ops that is desired/safe to run with reduced precision;
std::unordered_set<Symbol> WhiteList = {
  aten::conv1d,
  aten::conv2d,
  aten::conv3d,
  Symbol::fromQualString("aten::_convolution"),
  Symbol::fromQualString("aten::conv1d"),
  Symbol::fromQualString("aten::conv2d"),
  Symbol::fromQualString("aten::conv3d"),
  Symbol::fromQualString("aten::conv_transpose1d"),
  Symbol::fromQualString("aten::conv_transpose2d"),
  Symbol::fromQualString("aten::conv_transpose3d"),
  aten::matmul,
  aten::addmm,
  aten::addmv,
  aten::addr,
  aten::mm,
  aten::mv
};

// black list ops that is unsafe to run with reduced precision;
std::unordered_set<Symbol> BlackList = {
  Symbol::fromQualString("aten::tanh_"),
  prim::TupleConstruct,
  aten::softmax,
  Symbol::fromQualString("prim::GetAttr"),
  Symbol::fromQualString("aten::__getitem__"),
  Symbol::fromQualString("aten::pow_"),
  Symbol::fromQualString("aten::pow")
};

// Traversal of a graph through Topo ordre
template<typename func_t>
void graphTopoOrderTraversal(Block* block, const func_t& f) {
  for (auto node : block->nodes()) {
    // apply transform per node;
    f(node);
    for (auto sub_block : node->blocks()) {
      graphTopoOrderTraversal(sub_block, f);
    }
  }
}

// Traversal of a graph through Topo ordre
template<typename enter_t, typename modify_t>
void graphTopoOrderModify(
    Block* block,
    const enter_t& t,
    const modify_t& f) {
  bool any_changed = true;
  while (any_changed) {
    any_changed = false;
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
      if (t(*it)) {
        any_changed = true;
        it = f(*it);
      } else {
        it++;
      }
    }
  }
  for (auto node : block->nodes()) {
    for (auto sub_block : node->blocks()) {
      graphTopoOrderModify(sub_block, t, f);
    }
  }
}

// apply cast_n op for each individual Tensor in a `List[Tensor]`
Value* castListOfTensors(Value* val, Graph* subgraph, Node* cast_n) {
  auto len_n  = subgraph->create(Symbol::fromQualString("aten::len"));
  len_n->output()->setType(IntType::get());
  auto loop_n = subgraph->create(Symbol::fromQualString("prim::Loop"));
  loop_n->output()->setType(ListType::ofTensors());
  auto get_n = subgraph->create(Symbol::fromQualString("aten::__getitem__"));
  auto list_outer = subgraph->create(Symbol::fromQualString("prim::ListConstruct"));
  auto list_inner = subgraph->create(Symbol::fromQualString("prim::ListConstruct"));
  list_outer->output()->setType(ListType::ofTensors());
  list_inner->output()->setType(ListType::ofTensors());
  auto append_n = subgraph->create(Symbol::fromQualString("aten::add_"));
  append_n->output()->setType(ListType::ofTensors());

  val->replaceAllUsesWith(loop_n->output());
  len_n->addInput(val);
  subgraph->setInsertPoint(val->node()->next());
  auto enter = subgraph->insertConstant(true);
  auto sync = subgraph->insertConstant(false);
  subgraph->insertNode(len_n);
  subgraph->insertNode(list_outer);
  subgraph->insertNode(loop_n);

  loop_n->addInput(len_n->output());
  loop_n->addInput(enter);
  loop_n->addInput(list_outer->output());
  auto b = loop_n->addBlock();
  b->addInput()->setType(IntType::get());
  b->addInput()->copyMetadata(list_outer->output());
  b->appendNode(get_n);
  get_n->addInput(val);
  get_n->addInput(b->inputs()[0]);
  b->appendNode(cast_n);
  cast_n->addInput(get_n->output());
  cast_n->addInput(sync);
  b->appendNode(list_inner);
  list_inner->addInput(cast_n->output());
  b->appendNode(append_n);
  append_n->addInput(b->inputs()[1]);
  append_n->addInput(list_inner->output());
  b->registerOutput(enter);
  b->registerOutput(append_n->output());
  return loop_n->output();
}

} // namespace

GraphTraversalUtil::GraphTraversalUtil(std::shared_ptr<Graph> graph, CheckValueFn fn)
    : graph_(std::move(graph)), enter_edge_fn_(fn) {
  input_maps_.clear();
  output_maps_.clear();
}

GraphTraversalUtil::NodeSet GraphTraversalUtil::followEdges(
    const Node* n, TraversalDirection direction) {
  NodeSet res;
  if (direction == TraversalDirection::FollowInputs ||
      direction == TraversalDirection::FollowInputsAndOutputs) {
    if (input_maps_.count(n) == 0) {
      NodeSet inputs;
      ValueSet visited_values;
      for (auto v : n->inputs()) {
        followInput(v, inputs, visited_values);
      }
      input_maps_[n] = inputs;
    }
    res.insert(input_maps_[n].begin(), input_maps_[n].end());
  }
  if (direction == TraversalDirection::FollowOutputs ||
      direction == TraversalDirection::FollowInputsAndOutputs) {
    if (output_maps_.count(n) == 0) {
      NodeSet outputs;
      ValueSet visited_values;
      for (auto v : n->outputs()) {
        followOutput(v, outputs, visited_values);
      }
      output_maps_[n] = outputs;
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

void AmpAutoCasting(std::shared_ptr<Graph>& graph) {
  /*partitionMixedPrecision(graph);*/
  //throw std::runtime_error("Pass not implemented yet!");
  GraphTraversalUtil gtu(
      graph,
      [](const Value* v) -> bool {
          return v->type()->isSubtypeOf(TensorType::get());
      }); 

  GraphTraversalUtil::NodeSet white_nodes;
  GraphTraversalUtil::NodeSet black_nodes;
  
  // Initial painting of Black/White nodes according to the lists
  graphTopoOrderTraversal(
      graph->block(),
      [&](const Node* n) {
        if (WhiteList.count(n->kind()) != 0) {
          white_nodes.insert(n);
        }
        if (BlackList.count(n->kind()) != 0) {
          black_nodes.insert(n);
        }
      });

  GraphTraversalUtil::NodeSet black_upstream_nodes;

  // Mark black_upstream_nodes
  graphTopoOrderTraversal(
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

  // Paint black nodes
  graphTopoOrderTraversal(
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

  // Paint white nodes
  graphTopoOrderTraversal(
      graph->block(),
      [&](const Node* n) {
        if (white_nodes.count(n) != 0) {
          gtu.traverseFromNode(
              n,
              [&] (const Node* node) -> bool {
                if (node->kind() != prim::GetAttr &&
                    (node == n  ||
                    (black_nodes.count(node) == 0 &&
                     white_nodes.count(node) == 0))) {
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

  // Try fusing white nodes
  // TODO: we prolly want our own Symbol other than reusing the
  // prim::FusionGroup just to save us from confusion, although this temporary
  // node is not supposed to see the light of the day EVER.
  // TODO: we need to update that white_nodes list as we remove/destroy nodes!
  //const auto amp_fp16_symbol = Symbol::fromQualString("prim::FusionGroup");
  const auto amp_fp16_symbol = Symbol::fromQualString("prim::AmpGroup");
  GraphPartition(
      graph,
      amp_fp16_symbol,
      [&](Node* n) -> bool {
				return white_nodes.count(n) != 0 || n->kind() == amp_fp16_symbol;
			});

  // FuseGraph inserts bunch of shape & Broadcast ops inside, which is not
  // needed since we'll inline the fusion afterwards;
  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);

  // Inserting cast ops in the subgraph of `prim::AmpGroup` node;
  graphTopoOrderTraversal(
      graph->block(),
      [&](const Node* n) {
        if (n->kind() == amp_fp16_symbol) {
          auto subgraph = n->g(attr::Subgraph);
          auto sync = subgraph->insertConstant(false);
          for (auto input : subgraph->inputs()) {
            auto n = subgraph->create(Symbol::fromQualString("aten::_cast_Half"));
            if (input->type()->isSubtypeOf(TensorType::get())) {
              input->replaceAllUsesWith(n->outputs()[0]);
              n->addInput(input);
              n->addInput(sync);
              n->insertAfter(input->node());
            } else if (input->type()->isSubtypeOf(ListType::ofTensors())) {
              castListOfTensors(input, subgraph.get(), n);
            }
          }

          // not safe to do so!
          //for (auto output : subgraph->outputs()) {
          for (int i = subgraph->outputs().size()-1; i >= 0; i--) {
            auto n = subgraph->create(Symbol::fromQualString("aten::_cast_Float"));
            auto output = subgraph->outputs()[i];
            if (output->type()->isSubtypeOf(TensorType::get())) {
              // TODO: mistakes I've made in the past. Keep them here and put a
              // comment.
              // `n->insertAfter(output->node());`
              // `output->replaceAllUsesWith(n->outputs()[0]);`
              n->insertBefore(subgraph->return_node());
              subgraph->return_node()->replaceInputWith(output, n->outputs()[0]);
              n->addInput(output);
              n->addInput(sync);
            } else if (output->type()->isSubtypeOf(ListType::ofTensors())) {
              auto casted_output = castListOfTensors(output, subgraph.get(), n);
              casted_output->replaceAllUsesWith(output);
              subgraph->return_node()->replaceInputWith(output, casted_output);
            }
          }
          sync->node()->moveAfter(subgraph->inputs()[0]->node());
        }
      });

  graphTopoOrderModify(graph->block(),
      [&](const Node* n) -> bool {
        return n->kind() == amp_fp16_symbol;
      },
      [&](Node* n) -> graph_node_list::iterator {
        return ++inlineCallTo(n, *(n->g(attr::Subgraph))).back()->node()->reverseIterator();
      });

  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);
}

} // namespace jit
} // namespace torch
