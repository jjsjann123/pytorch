
#include <torch/csrc/jit/passes/autocast.h>

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

#include <stack>
#include <unordered_set>

namespace torch {
namespace jit {

namespace {

struct AutocastScope {
  Value* instance = nullptr;
  bool enabled = false;
};

// If we have an autocast instance, return it
//
// This is the pattern we're looking for (this is done after
//  autocast.__init__() has been inlined)
//
// %4 : bool = prim::Constant[value=1]()
// %5 : __torch__.torch.cuda.amp.autocast_mode.autocast = prim::CreateObject()
//  = prim::SetAttr[name="_enabled"](%5, %4)
//
// Notes:
//  1. There's no guarantee that the autocast instance is in the same block
//    as the prim::Enter() node
//  2. `prim::SetAttr` must follow `prim::CreateObject()` in the same block,
//    but there might be other nodes in between
//
c10::optional<AutocastScope> parseAutocast(Value* value) {
  const auto class_name = getModuleName(value);
  if (class_name &&
      *class_name == "__torch__.torch.cuda.amp.autocast_mode.autocast") {
    if (value->node()->kind() == prim::CreateObject) {
      // Search for `prim::SetAttr[name="_enabled"]`
      for (Use use : value->uses()) {
        if (use.user->kind() == prim::SetAttr &&
            use.user->s(attr::name) == "_enabled") {
          const auto enabled = constant_as<bool>(use.user->input(1));
          if (enabled.has_value()) {
            // We have an autocast instance
            AutocastScope scope;
            scope.instance = value;
            scope.enabled = *enabled;
            return scope;
          } else {
            // TODO: better error message
            AT_ERROR("Autocast argument must be a constant");
          }
        }
      }
    } else {
      // We only support simple and static autocast expressions. For example,
      // the following should report an error (since the autocast would not
      // work as expected)
      //
      //    autocast_on = autocast(enabled=True)
      //    autocast_off = autocast(enabled=False)
      //    with autocast_on if condition else autocast_off:
      //        ...
      //
      // TODO: better error message
      //
      AT_ERROR("Unsupported autocast syntax");
    }
  }

  // Not an autocast...
  return c10::nullopt;
}

void castTensorInputs(Node* node, at::ScalarType dtype) {
  const auto graph = node->owningGraph();

  WithInsertPoint insert_point(node);

  const auto dtype_value = graph->insertConstant(dtype);
  const auto false_value = graph->insertConstant(false);
  const auto none_value = graph->insertConstant(IValue());

  std::unordered_set<Value*> casted_inputs;

  for (auto input : node->inputs()) {
    if (input->type()->kind() == TensorType::Kind) {
      casted_inputs.insert(input);
    }
  }

  for (auto input : casted_inputs) {
    const auto new_input = graph->insert(
        aten::to, {input, dtype_value, false_value, false_value, none_value});
    node->replaceInputWith(input, new_input);
  }
}

void castInputsToWidestType(Node* node) {
  // Figure out the widest type
  // (really, just looking for any float32 inputs)
  //
  // TODO: revisit this (do we need to consider float64 types?)
  //
  for (auto input : node->inputs()) {
    if (auto tensor_type = input->type()->cast<TensorType>()) {
      const auto dtype = tensor_type->scalarType();
      if (!dtype.has_value() || *dtype != at::ScalarType::Half) {
        castTensorInputs(node, at::ScalarType::Float);
        return;
      }
    }
  }
}

void handleBlock(Block* block, bool initial_state) {
  std::stack<AutocastScope> autocast_stack;

  // The current autocast enabled/disabled state
  auto current_state = [&] {
    return autocast_stack.empty() ? initial_state
                                  : autocast_stack.top().enabled;
  };

  for (Node* node : block->nodes()) {
    switch (node->kind()) {
      case prim::CallFunction:
        TORCH_INTERNAL_ASSERT(false, "Calls are not expected with AMP & JIT");
        break;

      case prim::CallMethod:
        if (auto class_type = node->input(0)->type()->cast<ClassType>()) {
          const auto& name = node->s(attr::name);
          const auto& function = class_type->getMethod(name);
          TORCH_INTERNAL_ASSERT(
              !function.isGraphFunction(),
              "Calls are not expected with AMP & JIT");
        } else {
          TORCH_INTERNAL_ASSERT(false, "Unexpected prim::CallMethod form");
        }
        break;

      case prim::Enter:
        if (auto autocast_scope = parseAutocast(node->input())) {
          autocast_stack.push(*autocast_scope);
        }
        break;

      case prim::Exit:
        // TODO: technically we can avoid parseAutocast() here
        if (auto autocast_scope = parseAutocast(node->input())) {
          TORCH_INTERNAL_ASSERT(!autocast_stack.empty());
          TORCH_INTERNAL_ASSERT(
              autocast_stack.top().instance == autocast_scope->instance);
          autocast_stack.pop();
        }
        break;

      // CastPolicy::fp16 (cast all inputs to float16)
      case aten::_convolution:
      case aten::_convolution_nogroup:
      case aten::conv1d:
      case aten::conv2d:
      case aten::conv3d:
      case aten::conv_tbc:
      case aten::conv_transpose1d:
      case aten::convolution:
      case aten::cudnn_convolution:
      case aten::cudnn_convolution_transpose:
      case aten::prelu:
      case aten::addmm:
      case aten::addmv:
      case aten::addr:
      case aten::matmul:
      case aten::mm:
      case aten::mv:
      case aten::linear:
      case aten::addbmm:
      case aten::baddbmm:
      case aten::bmm:
      case aten::chain_matmul:
      case aten::_thnn_fused_lstm_cell:
      case aten::_thnn_fused_gru_cell:
      case aten::lstm_cell:
      case aten::gru_cell:
      case aten::rnn_tanh_cell:
      case aten::rnn_relu_cell:
        if (current_state()) {
          castTensorInputs(node, at::ScalarType::Half);
        }
        break;

      // CastPolicy::fp32 (cast all inputs to float32)
      case aten::native_layer_norm:
      case aten::acos:
      case aten::asin:
      case aten::cosh:
      case aten::erfinv:
      case aten::exp:
      case aten::expm1:
      case aten::log:
      case aten::log10:
      case aten::log2:
      case aten::log1p:
      case aten::reciprocal:
      case aten::rsqrt:
      case aten::sinh:
      case aten::tan:
      case aten::pow:
      case aten::softplus:
      case aten::gelu:
      case aten::layer_norm:
      case aten::group_norm:
      case aten::frobenius_norm:
      case aten::nuclear_norm:
      case aten::cosine_similarity:
      case aten::cosine_embedding_loss:
      case aten::nll_loss:
      case aten::nll_loss2d:
      case aten::hinge_embedding_loss:
      case aten::kl_div:
      case aten::l1_loss:
      case aten::smooth_l1_loss:
      case aten::mse_loss:
      case aten::margin_ranking_loss:
      case aten::multilabel_margin_loss:
      case aten::soft_margin_loss:
      case aten::triplet_margin_loss:
      case aten::multi_margin_loss:
      case aten::binary_cross_entropy_with_logits:
      case aten::dist:
      case aten::pdist:
      case aten::cdist:
      case aten::renorm:
        if (current_state()) {
          castTensorInputs(node, at::ScalarType::Float);
        }
        break;

      // CastPolicy::promote (promote inputs to the widest type)
      case aten::addcdiv:
      case aten::addcmul:
      case aten::atan2:
      case aten::bilinear:
      case aten::cat:
      case aten::_cat:
      case aten::cross:
      case aten::dot:
      case aten::equal:
      case aten::index_put:
      case aten::stack:
      case aten::tensordot:
        if (current_state()) {
          castInputsToWidestType(node);
        }
        break;

      // Banned in autocast, see binary_cross_entropy_banned()
      case aten::binary_cross_entropy:
        AT_ERROR("Unsafe to autocast");
    }

    // process sub-blocks, if any
    for (Block* sub_block : node->blocks()) {
      handleBlock(sub_block, current_state());
    }
  }

  // Sanity check: make sure there's no unbalanced transition
  TORCH_INTERNAL_ASSERT(autocast_stack.empty());
}

} // namespace

void Autocast(const std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("\nBefore Autocast: ", graph);
  handleBlock(graph->block(), false);
  GRAPH_DUMP("\nAfter Autocast: ", graph);
}

} // namespace jit
} // namespace torch