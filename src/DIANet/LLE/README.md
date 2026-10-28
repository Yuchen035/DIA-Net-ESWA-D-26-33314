# LLE research components

This directory preserves standalone shared/specific router implementations,
brightness-category experiments, loss components, data helpers and development
tests. It is not the selected FLUX Kontext training path or a complete
implementation of the manuscript.

`train_dual_route_simple.py` uses a mock backbone and has an unresolved Trainer
constructor mismatch. `infer_dual_route.py` requires explicit `--use-mock`;
requesting the real-model branch raises an error. The prototype's step argument
does not implement an ODE loop, and its expert-only checkpoint omits the shared
router. These limitations are documented in
[the method map](../../../docs/PAPER_CODE_MAP.md).

Tests and legacy entry points may choose CUDA, load resources or run training.
They were not executed for this release. Use the package-root inspection tools
for model-free review. The original development README and audit history are
kept with the separate author records; their historical completion claims do
not override the implementation status described here.
