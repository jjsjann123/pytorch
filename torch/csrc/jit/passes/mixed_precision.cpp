#include <stack>

#include <torch/csrc/jit/passes/mixed_precision.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/utils/memory.h>

#include <torch/csrc/jit/pass_manager.h>

namespace torch {
namespace jit {

using ::c10::Symbol;

#define DEBUG false

namespace {

#define ASYNC_FLAG true

std::unordered_set<Symbol> WhiteList = {
  aten::mul,
  aten::conv1d,
  aten::conv2d,
  aten::conv3d,
  Symbol::fromQualString("aten::_convolution"),
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

std::unordered_set<Symbol> BlackList = {
  Symbol::fromQualString("aten::tanh_"),
  //aten::tanh,
  //aten::batch_norm,
  // temporary hack
  prim::TupleConstruct,
  aten::softmax,
  Symbol::fromQualString("prim::GetAttr"),
  // apparently plain `aten::tanh_` doesn't work. no idea why
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

} // namespace

GraphPartition::~GraphPartition() = default;

GraphPartition::GraphPartition(
    std::shared_ptr<Graph> graph,
    Symbol kind,
    std::function<bool(Node*)> fn,
    bool debug)
      : graph_(std::move(graph)),
        kind_(kind),
        fn_(fn),
        debug_(debug){
  aliasDb_ = torch::make_unique<AliasDb>(graph_);
}

void GraphPartition::refreshAliasDb() {
  aliasDb_ = torch::make_unique<AliasDb>(graph_);
}

void GraphPartition::partition() {
  mergeNodesInBlock(graph_->block());
}

void GraphPartition::mergeNodesInBlock(Block* block) {
  bool any_changed = true;
  while (any_changed) {
    any_changed = false;
    refreshAliasDb();
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
      bool merge_node = false;
      if (debug_) {
        std::cout << "====check if we should try: ";
        it->dump();
      }
      if (fn_(*it)) {
        if (debug_) {
          std::cout << "====merging node with producer: ";
          it->dump();
        }
        value_list reverse_topological_inputs;
        for (auto i : it->inputs()) {
          if (i->node()->owningBlock() == block) {
            reverse_topological_inputs.push_back(i);
          }
        }
        // Sort in reverse topological order
        std::sort(
            reverse_topological_inputs.begin(),
            reverse_topological_inputs.end(),
            [&](Value* a, Value* b) {
              return a->node()->isAfter(b->node());
            });
        for (auto producer : reverse_topological_inputs) {
          if (debug_) {
            std::cout << "try merging node: " << std::endl;
            //std::cout << "                  " << (*it) << std::endl;
            std::cout << "                  "; 
            it->dump();
            std::cout << "                  " << *producer->node() << std::endl;
          }
          auto partition = tryMerge(*it, producer);
          if (partition) {
            if (debug_) {
              std::cout << "     merged" << std::endl;
            }
            it = partition.value()->reverseIterator();
            merge_node = true;
						break;
          } else {
            if (debug_) {
              std::cout << "     not merged" << std::endl;
            }
          }
        }
      }
if (!merge_node) {
        it++;
      } else {
        any_changed |= merge_node;
      }
    }
  }

  for (Node* node : block->nodes()) {
    for (Block* sub_block : node->blocks()) {
      mergeNodesInBlock(sub_block);
    }
  }
}

Graph& GraphPartition::getSubgraph(Node* n) {
  AT_ASSERT(n->kind() == kind_);
  return *n->g(attr::Subgraph);
}

// insert a producer node into a consuming partition.
// DOES NOT WORK if n is a consumer of an output of the partition
// returns the node _inside_ the partition that represents the node
Node* GraphPartition::mergeNodeIntoPartition(Node* partition, Node* n) {
  AT_ASSERT(n->kind() != kind_);
  auto subgraph = &getSubgraph(partition);
  // map from nodes in the surrounding graph to parameters in the partition 
  // that is correspond to them
  std::unordered_map<Value*, Value*> inputs_map;
  size_t i = 0;
  AT_ASSERT(partition->inputs().size() == subgraph->inputs().size());
  for (auto input : partition->inputs()) {
    inputs_map[input] = subgraph->inputs()[i++];
  }
  // add n's inputs to the fusion group's input list if we don't already have
  // them
  WithInsertPoint guard(*subgraph->nodes().begin());
  for (auto input : n->inputs()) {
    if (inputs_map.count(input) == 0) {
      auto in_partition = subgraph->addInput();
      in_partition->setType(input->type());
      inputs_map[input] = in_partition;
      partition->addInput(input);
    }
  }

  // copy n into the graph, remapping its inputs to internal nodes
  Node* in_graph = subgraph->createClone(
      n, [&](Value* k) -> Value* { return inputs_map[k]; });

  // if n's outputs are already inputs to the fusion partition,
  // we need to remove them because n is now inside the fusion partition.
  //
  // i.e.,
  // x = f(w); partition(x, y, z) becomes partition(w, y, z).
  // x, y, z = f(w); partition(x, y, z) becomes partition(w).
  //
  // remapping nodes that used the input to the newly-merged node
  // n is not an input when the fusion partition is empty
  auto inputs = partition->inputs();
  for (size_t i = 0; i < n->outputs().size(); ++i) {
    auto it = std::find(inputs.begin(), inputs.end(), n->outputs()[i]);
    if (it != inputs.end()) {
      size_t p = it - inputs.begin();
      partition->removeInput(p);
      subgraph->inputs()[p]->replaceAllUsesWith(in_graph->outputs()[i]);
      subgraph->eraseInput(p);
    }
  }
  auto ret_n = subgraph->insertNode(in_graph);

  // If any of the outputs are still used then we need to add them
  auto outputs = n->outputs();
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto output = outputs[i];
    if (output->uses().size() == 0)
      continue;
    subgraph->registerOutput(ret_n->outputs()[i]);
    auto new_output = partition->addOutput();
    output->replaceAllUsesWith(new_output);
    new_output->setType(output->type());
  }
}

Node* GraphPartition::createSingleNodePartition(Node* n) {
  auto partition = graph_->createWithSubgraph(kind_);
  // propogate position information for the new node so we can always
  // have a valid mapping
  partition->insertBefore(n);
  Node* mergedNode = mergeNodeIntoPartition(partition, n);
  n->destroy();
  return partition;
}

void GraphPartition::mergePartitions(Node* consumer_partition, Node* producer_partition) {
  // Now we have two partitions!
  // Inline the first graph - place all inner nodes of producer back in the outer
  // graph.
  std::vector<Node*> temporary_nodes;
  auto producer_subgraph = &getSubgraph(producer_partition);

  if (debug_) {
    std::cout << "------- subgraph nodes" << std::endl;
    for (auto node : producer_subgraph->nodes()) {
      node->dump();
    }
  }
  // inlineCallTo is not really safe to use here, because there's not protocol
  // on where the insertion is.
  {
    auto anchor_node = producer_partition->next();
    WithInsertPoint guard(producer_partition);

    std::unordered_map<Value*, Value*> value_map;
    auto value_map_func = [&](Value* v) { return value_map.at(v); };
    AT_ASSERT(producer_subgraph->inputs().size() == producer_partition->inputs().size());
    for (size_t i = 0; i < producer_partition->inputs().size(); ++i) {
      value_map[producer_subgraph->inputs()[i]] = producer_partition->inputs()[i];
    }
    for (auto* node : producer_subgraph->nodes()) {
      auto* new_node = graph_->insertNode(graph_->createClone(node, value_map_func));
      temporary_nodes.emplace_back(new_node);
      if (debug_) {
        std::cout << "push inlining node: ";
        new_node->dump();
      }
      for (size_t i = 0; i < node->outputs().size(); ++i) {
        value_map[node->outputs()[i]] = new_node->outputs()[i];
      }
    }
 
    std::vector<Value*> new_outputs;
    for (auto* output : producer_subgraph->outputs()) {
      new_outputs.push_back(value_map_func(output));
    }
//    auto new_outputs =
//        insertGraph(*graph_, *producer_subgraph, producer_partition->inputs());
    const auto& old_outputs = producer_partition->outputs();

    /*
    for (auto iter = ++(producer_partition->iterator());
				 iter != anchor_node->iterator();
				 iter++) {
      if (debug_) {
        std::cout << "push inlining node: ";
        iter->dump();
      }
      temporary_nodes.emplace_back(*iter);
    }
    */

    AT_ASSERT(new_outputs.size() == old_outputs.size());
    for (size_t i = 0; i < old_outputs.size(); ++i) {
      if (old_outputs[i]->hasDebugName()) {
        new_outputs[i]->setDebugName(old_outputs[i]->debugName());
      }
      old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
    }
    producer_partition->destroy();
    // Just to get a clear error in case someone uses it
		producer_partition = nullptr;
	}

  // Inline the temporary nodes into the first group
  auto consumer_subgraph = &getSubgraph(consumer_partition);
  for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
       ++it) {
    Node* node = *it;
    Node* merged = mergeNodeIntoPartition(consumer_partition, node);
    node->destroy();
  }
}

at::optional<Node*> GraphPartition::tryMerge(Node* consumer, Value* producer) {
  // this handles cases where producer can be moved _into_ the fusion group of
  // consumer.
  // TODO: extend to fusion of consumer into _producer's_ fusion blob
  // if the consumer allInputsAreThisProducer(consumer,producer)
  // we can move the consumer up into the producer.
  // but this requires better handling of merging fusion groups so it is not
  // done now
  bool shouldMerge = fn_(producer->node()) &&
      // Rearrange nodes such that all uses of producer are after the
      // consumer. Fusion will rewrite those later uses to use the version of
      // producer generated by the fused blob. In this case, producer becomes
      // an output of the fusion group.
      aliasDb_->moveBeforeTopologicallyValid(producer->node(), consumer);

  if (debug_) {
      std::cout << "try merging two nodes: " << std::endl << "---" << (*consumer) << "---" << (*producer->node());
  }

  if (!shouldMerge) {
    if (debug_) {
      std::cout << "==== should not merge" << std::endl;
    }
    return at::nullopt;
  }

  if (debug_) {
    std::cout << "==== should merge" << std::endl;
  }

  auto partition = consumer;
  if (consumer->kind() != kind_) {
    if (debug_) {
      std::cout << "==== create single node partition" << std::endl;
    }
    partition = createSingleNodePartition(consumer);
  }

  if (producer->node()->kind() == kind_) {
    if (debug_) {
      std::cout << "==== merge two partitions" << std::endl;
    }
    mergePartitions(partition, producer->node());
    return partition;
  }
  if (debug_) {
    std::cout << "==== merge single node into partition" << std::endl;
  }
  Node* merged = mergeNodeIntoPartition(partition, producer->node());
  producer->node()->destroy();
  return partition;
}


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

  if (DEBUG) {
    std::cout << "=========================initial paint according to list=========================" << std::endl;
    for (auto node : black_nodes) {
      std::cout << "black node: " << *node << std::endl;
    }
    for (auto node : white_nodes) {
      std::cout << "white node: " << *node << std::endl;
    }
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

  if (DEBUG) {
    std::cout << "=========================final paint=========================" << std::endl;
    for (auto node : black_nodes) {
      std::cout << "black node: " << *node << std::endl;
    }
  }

  // Paint white nodes
  graphTopoOrderTraversal<true>(
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

  if (DEBUG) {
    std::cout << "=========================final paint part 2=========================" << std::endl;
    for (auto node : white_nodes) {
      std::cout << "white node: " << *node << std::endl;
    }
  }

  // Try fusing white nodes
  // TODO: we prolly want our own Symbol other than reusing the
  // prim::FusionGroup just to save us from confusion, although this temporary
  // node is not supposed to see the light of the day EVER.
  // TODO: we need to update that white_nodes list as we remove/destroy nodes!
  //const auto amp_fp16_symbol = Symbol::fromQualString("prim::FusionGroup");
  const auto amp_fp16_symbol = Symbol::fromQualString("prim::AmpGroup");
  GraphPartition gp(
      graph,
      amp_fp16_symbol,
      [&](Node* n) -> bool {
				return white_nodes.count(n) != 0 || n->kind() == amp_fp16_symbol;
			},
      DEBUG);
  gp.partition();

  // FuseGraph inserts bunch of shape & Broadcast ops inside, which is not
  // needed since we'll inline the fusion afterwards;
  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);
  if (DEBUG) {
    std::cout << "=========================graph partitioning=========================" << std::endl;
    std::cout << *graph << std::endl;
  }

  // GraphPartition gp_second(
  //     graph,
	// 		amp_fp16_symbol,
  //     [&](Node* n) -> bool {
	// 			return white_nodes.count(n) != 0;
	// 		},
  //     true);
  // gp_second.partition();

  // std::cout << "=========================try fuse the two more nodes=========================" << std::endl;
  // std::cout << *graph << std::endl;

  graphTopoOrderTraversal<true>(
      graph->block(),
      [&](const Node* n) {
        if (n->kind() == amp_fp16_symbol) {
          auto subgraph = n->g(attr::Subgraph);
          auto sync = subgraph->insertConstant(false);
          std::cout << (*subgraph);
          for (auto input : subgraph->inputs()) {
            auto n = subgraph->create(Symbol::fromQualString("aten::_cast_Half"));
            if (input->type()->isSubtypeOf(TensorType::get())) {
              input->replaceAllUsesWith(n->outputs()[0]);
              n->addInput(input);
              n->addInput(sync);
              n->insertAfter(input->node());
            } else if (input->type()->isSubtypeOf(ListType::ofTensors())) {
              throw std::runtime_error("List Tensor not implemented yet\n");
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
              throw std::runtime_error("List Tensor not implemented yet\n");
            }
          }
          sync->node()->moveAfter(subgraph->inputs()[0]->node());
        }
      });

  if (DEBUG) {
    std::cout << "=========================cast inserted=========================" << std::endl;
    std::cout << *graph << std::endl;

    std::cout << "=========================final graph=========================" << std::endl;
  }

  graphTopoOrderModify(graph->block(),
      [&](const Node* n) -> bool {
        return n->kind() == amp_fp16_symbol;
      },
      [&](Node* n) -> graph_node_list::iterator {
        if (DEBUG) {
          std::cout << std::endl << "inlining " << std::endl;
          std::cout << (*n);
          std::cout << "done inlining " << std::endl;
	}
        return ++inlineCallTo(n, *(n->g(attr::Subgraph))).back()->node()->reverseIterator();
      });

  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);

  if (DEBUG) {
    std::cout << *graph << std::endl;
  }
}

} // namespace jit
} // namespace torch
