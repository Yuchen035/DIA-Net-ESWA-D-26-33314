# Inference entry points

`inference_moe_lora.py` is selected by the package launcher. Version 0.2 uses
the paired raw-key adapter checkpoint protocol described in
[ENGINEERING.md](../../../docs/ENGINEERING.md), requires an adapter or explicit
`--base-model-only`, and rejects unsupported legacy checkpoints. No real model
inference was run to validate this change.

Its default remains 28 scheduler steps. A configurable step count is not evidence
of the manuscript's complete four-step semantic dual-routing procedure.

Other Python files are archived demonstrations with different loading assumptions
and example paths. They are retained for inspection, are not invoked by the
launcher, and are not covered by the new checkpoint protocol. See
[PAPER_CODE_MAP.md](../../../docs/PAPER_CODE_MAP.md) before interpreting any
entry point as a paper implementation.
