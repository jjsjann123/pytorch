#include "update_graph_auto_casting.h"

namespace torch {
namespace jit {

thread_local bool kAutoCasting = false;
void setGraphAutoCasting(bool o) {
  kAutoCasting = o;
}
bool getGraphAutoCasting() {
  return kAutoCasting;
}
}
}
