#include <stack>

#include <torch/csrc/jit/passes/graph_partition.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

using ::c10::Symbol;

namespace {

class GraphPartitioner {
public:
  GraphPartitioner(
      std::shared_ptr<Graph> graph,
      Symbol kind,
      std::function<bool(Node*)> fn,
      bool debug=false)
        : graph_(std::move(graph)),
          kind_(kind),
          fn_(fn),
          debug_(debug){
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
  }

  void partition() {
    mergeNodesInBlock(graph_->block());
  }

private:

  Symbol kind_;
  std::unique_ptr<AliasDb> aliasDb_;
  std::shared_ptr<Graph> graph_;
  std::function<bool(Node*)> fn_;

  bool debug_;

  void refreshAliasDb() {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
  }

  void mergeNodesInBlock(Block* block) {
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
  
  Graph& getSubgraph(Node* n) {
    AT_ASSERT(n->kind() == kind_);
    return *n->g(attr::Subgraph);
  }
  
  // insert a producer node into a consuming partition.
  // DOES NOT WORK if n is a consumer of an output of the partition
  // returns the node _inside_ the partition that represents the node
  Node* mergeNodeIntoPartition(Node* partition, Node* n) {
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
  
  Node* createSingleNodePartition(Node* n) {
    auto partition = graph_->createWithSubgraph(kind_);
    // propogate position information for the new node so we can always
    // have a valid mapping
    partition->insertBefore(n);
    Node* mergedNode = mergeNodeIntoPartition(partition, n);
    n->destroy();
    return partition;
  }
  
  void mergePartitions(Node* consumer_partition, Node* producer_partition) {
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
  
  at::optional<Node*> tryMerge(Node* consumer, Value* producer) {
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
};

} // anonymous namespace

void GraphPartition(
    std::shared_ptr<Graph>& graph,
    Symbol kind,
    std::function<bool(Node*)> fn) {
  auto gp = GraphPartitioner(graph, kind, fn);
  gp.partition();
}

} // namespace jit
} // namespace torch
