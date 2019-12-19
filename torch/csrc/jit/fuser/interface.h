#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>
#include <ATen/core/stack.h>
#include <c10/core/DeviceType.h>

#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>

namespace torch {
namespace jit {

/*
 * NEW INTERFACE
*/

#define FUSER_DEBUG 1

class FusionBackend {
  typedef bool (*isFusibleFunc)(const Node* const);
  typedef int (*fuseFunc)(const Node* const);
  typedef void (*compileFusionFunc)(Node*);
  typedef void (*callFusionFunc)(const Node* const, Stack&);

public:
  FusionBackend(isFusibleFunc is_fusible,
      fuseFunc fuse,
      compileFusionFunc compile_fusion,
      callFusionFunc call_fusion) :
  is_fusible_(is_fusible),
  fuse_(fuse),
  compile_fusion_(compile_fusion),
  call_fusion_(call_fusion) {}

  bool isFusible(const Node* const node);
  int fuse(const Node* const node);
  void compileFusion(Node* fusion);
  void callFusion(const Node* const node, Stack& stack);

protected:
  isFusibleFunc is_fusible_;
  fuseFunc fuse_;
  compileFusionFunc compile_fusion_;
  callFusionFunc call_fusion_;
};

TORCH_API void registerFusionBackendEx(
    at::Device::Type backend_type,
    FusionBackend* backend);
TORCH_API bool hasFusionBackendEx(at::Device::Type backend_type);

struct TORCH_API RegisterFusionBackendEx {
  RegisterFusionBackendEx(
      at::Device::Type backend_type,
      FusionBackend* backend);
};

// Returns true iff the node is fusible
TORCH_API bool isFusible(const Node* const node);

// Creates a fusion consisting of just the given node and returns its
// corresponding key
TORCH_API int fuse(const Node* const node);

// Compiles the given fusion node
TORCH_API void compileFusion(Node* fusion);

// TODO: remove key, it can be acquired from the node
TORCH_API void callFusion(const Node* const node, Stack& stack);

/*
 * OLD INTERFACE BELOW
*/


constexpr int kCPUDevice = -1;

// Assigns a "key" to the given fusion_group that it can use to run its
// fusion later (via runFusion() below).
TORCH_API int64_t registerFusion(const Node* fusion_group);

// Runs the fusion corresponding to the given key on the inputs
// found on the stack. Outputs are placed on the same stack.
// In some cases a fusion cannot be run and a fallback path where
// PyTorch's interpreter runs the graph instead is attempted.
TORCH_API void runFusion(const int64_t key, Stack& stack);

// True if the respective devices can fuse, false otherwise
TORCH_API bool canFuseOnCPU();
TORCH_API bool canFuseOnGPU();

// Sets whether fusion on the CPU is allowed (disabled by default due to
// flakiness)
TORCH_API void overrideCanFuseOnCPU(bool value);

// Treats the given graph as a fusion group and launches it on the
// specified device with the given inputs.
// Returns the outputs.
TORCH_API std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs);

// Treats the given graph as a fusion group and returns the generated code.
TORCH_API std::string debugGetFusedKernelCode(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs);

TORCH_API size_t nCompiledKernels();

} // namespace jit
} // namespace torch
