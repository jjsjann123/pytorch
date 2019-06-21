#include <stack>

#include <torch/csrc/jit/passes/mixed_precision.h>
#include <torch/csrc/jit/passes/alias_analysis.h>

#include <torch/csrc/jit/pass_manager.h>

namespace torch {
namespace jit {

using ::c10::Symbol;

namespace {

#define ASYNC_FLAG true
typedef void (*TransFn)(Node*);
typedef bool (*CheckFn)(Node*);

std::unordered_set<Symbol> WhiteList = {
  aten::relu
};

bool shouldCastToHalf(Node* node) {
  if (WhiteList.find(node->kind()) != WhiteList.end()) {
    printf("found node!\n");
    node->dump();
    return true;
  }
  return false;
}

void castToHalf(Node* node) {
  auto inputs = node->inputs();
  auto graph = node->owningGraph();

  // loops through the number of inputs, as we needed the index when we replace
  // the inputs.
  // Injects input cast;
  for (int i = 0; i < inputs.size(); i++) {
    // TODO: what's the requirement from ASYNC_FLAG here?
    auto v = graph->insert(aten::_cast_Half, {inputs[i], ASYNC_FLAG});
    node->replaceInput(i, v);
    v->node()->moveBefore(node);
    v->node()->inputs()[1]->node()->moveBefore(v->node());
  }

  // rewire output with fp32 cast;
  for (auto output : node->outputs()) {
    auto v = graph->insert(aten::_cast_Float, {output, ASYNC_FLAG});
    // replaceAllUsesWith also hijack the input to _cast_Float;
    output->replaceAllUsesWith(v);
    v->node()->replaceInput(0, output);
    v->node()->moveAfter(node);
    v->node()->inputs()[1]->node()->moveBefore(v->node());
  }
}

bool detectConsecutiveCast(Node* node) {
  // TODO: should I also check synchronous flag here?
  return node->kind() == aten::_cast_Half &&
         node->inputs()[0]->node()->kind() == aten::_cast_Float;
}

void removeConsecutiveCast(Node* node) {
  // detectConsecutiveCast matches a graph pattern:
  //    tensor_0->cast->cast->tensor_1
  // So we simply re-wire tensor_1 to use tensor_0. Hoping that later
  // optimization passes should identify and prune dangling cast ops.
  auto fp16_input = node->inputs()[0]->node()->inputs()[0];

  // TODO: should I worry about dangling nodes/values?
  node->outputs()[0]->replaceAllUsesWith(fp16_input);
  // printf("removing node!");
  // node->dump();
}

void graphTransform(std::shared_ptr<Graph>& graph,
                   CheckFn check_fn,
                   TransFn trans_fn) {
  std::stack<Node*> node_list;
  std::stack<Block*> block_stack;
  block_stack.push(graph->block());
  // BFS traversal of the graph;
  while(!block_stack.empty()) {
    Block* block = block_stack.top();
    block_stack.pop();
    for (auto node : block->nodes()) {
      // apply transform per node;
      if (check_fn(node)) {
        node_list.push(node);
      }
      for (auto sub_block : node->blocks()) {
        block_stack.push(sub_block);
      }
    }
  }
  while(!node_list.empty()) {
    trans_fn(node_list.top());
    node_list.pop();
  }
}

RegisterPass p(&insertCastingNode);
RegisterPass q(&removeConsecutiveCastingNode);

} // namespace

// removing/collapsing cast ops (semantic-preserving transform)
void removeConsecutiveCastingNode(std::shared_ptr<Graph>& graph) {
  // TODO: currently assuming all cast op to be fp16/fp32. While we shoulda
  // collapse/fuse cast ops and check for tensor type (is this available
  // statically?) to safely replace it with no-op.
  graphTransform(graph, &detectConsecutiveCast, &removeConsecutiveCast);
}

// inserting cast ops around white list ops, modifying graph semantics
void insertCastingNode(std::shared_ptr<Graph>& graph) {
  graphTransform(graph, &shouldCastToHalf, &castToHalf);
}

void insertCastingWithScalingNode(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

} // namespace jit
} // namespace torch
