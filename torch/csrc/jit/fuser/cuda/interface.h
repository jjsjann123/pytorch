#pragma once
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/fuser/interface.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class CUDAFusionBackend : public FusionBackend {
public:
  CUDAFusionBackend(isFusibleFunc is_fusible,
      fuseFunc fuse,
      compileFusionFunc compile_fusion,
      callFusionFunc call_fusion) :
    FusionBackend(is_fusible, fuse, compile_fusion, call_fusion) {}
};

TORCH_API bool isFusibleOnCUDA(const Node* const node);

// Copy cat, we may want a context here (const Node* const fusion) to aid the
// decision making.
TORCH_API int fuseOnCUDA(const Node* const node);

TORCH_API void compileFusionOnCUDA(Node* fusion);

TORCH_API void callFusionOnCUDA(
    const Node* const fusion,
    std::vector<at::Tensor>& outputs,
    at::ArrayRef<IValue> inputs);
}}}} // namespace torch::jit::fuser::cuda
