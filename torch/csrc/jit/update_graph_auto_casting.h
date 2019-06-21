#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
namespace torch {
namespace jit{
TORCH_API void setGraphAutoCasting(bool o);
TORCH_API bool getGraphAutoCasting();
}
}
