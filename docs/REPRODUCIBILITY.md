# Source availability and reproducibility scope

Version 0.2 provides the available DIA-Net project source, explicit method
coverage, and documented engineering changes for inspection. It is not a
complete executable reconstruction of the submitted manuscript. No additional
manuscript implementation was supplied for this edition, and no missing
scientific algorithm was written to fill the gaps.

## What the source supports

The FLUX/PEFT engineering path contains real-model loading, ordinary LoRA and
token-wise feature MoE injection, velocity-MSE training, scheduler-based
inference infrastructure, and adapter serialization. The local PEFT source and
specific Diffusers snapshot are required dependencies, not evidence that the
historical environment has been recovered.

The separate LLE tree provides shared/specific routing, brightness
classification, standalone loss components, data transforms and metric helpers.
These are meaningful implemented components. Their existence does not establish
a connected four-category, real-backbone DIA-Net system.

Reviewers can inspect the parameterization, gating, training objective,
data-pairing logic and entry points. The PSNR and simplified SSIM helpers can
calculate values for suitably prepared inputs; their input conventions and
benchmark limitations are set out in [PAPER_CODE_MAP.md](PAPER_CODE_MAP.md).

## Implementation coverage

- **Semantic perception and conditioning:** Gemini-to-Qwen annotation/distillation
  and four-class inference were not found in the supplied project source. The
  prototype implements five brightness bins. Neither training path implements
  the manuscript's spatial semantic-label channel concatenation.
- **Routing and injection:** real-model MoE is token-wise feature routing; the
  shared/specific prototype uses a pooled MLP router and different defaults.
  Some archived single-stream injection targets do not exist in the supplied
  FLUX class; `proj_mlp` is not targeted. These method/configuration differences
  remain documented rather than rewritten into a new implementation.
- **Objective:** main loops optimize flow-matching MSE. Reconstruction and color
  files are placeholders. Existing adversarial components are unconnected and
  differ from the manuscript's FiLM conditioning and relative-score formula.
  The complete four-term training objective is not supplied.
- **Prototype execution:** the dual-route training entry has a constructor
  mismatch. Its inference entry requires explicit Mock mode and performs one
  fixed-time forward pass, despite exposing a step-count argument. The
  unsupported real-model branch is rejected. The real FLUX pipeline exposes
  scheduler steps but not a complete manuscript-equivalent semantic workflow.
- **Evaluation:** original split/membership manifests, full metric settings,
  per-image predictions, semantic classification evaluation and hardware
  measurement records are unavailable. The referenced project LPIPS module is
  absent, and simplified global SSIM is not a verified benchmark protocol.

Flow-matching time direction is a separate convention issue: reversing time can
relate `noise-clean` to `clean-noise`. A sign difference alone does not establish
a wrong algorithm. Consistency of time input and sampling must be assessed as a
whole; this edition did not run a numerical trajectory.

## Engineering changes and checkpoints

The main save/load path uses the new `dia-net-adapter-v2` protocol in
`src/DIANet/adapter_checkpoint.py`. A Safetensors file is accompanied by a JSON
sidecar appended to its full filename. Configuration, raw adapter key metadata,
file integrity and model-structure metadata are checked before strict
restoration. Only one adapter with `bias="none"` and no `modules_to_save` is
supported; the protocol serializes `lora_*` state only. The base-model signature
checks public architecture configuration, not the identity of all pretrained
weight bytes. Unsupported legacy files are rejected; missing expert
configuration is not guessed from tensor shapes.

This protocol is an engineering change made for this edition. It replaces the
documented earlier save/load mismatch, but it has not been validated through a
real-model save/reload run. No existing manuscript checkpoint was authenticated
or converted. The separate LLE prototype still uses an expert-only format that
omits its shared router. See [ENGINEERING.md](ENGINEERING.md) for the supported
format and exact failure checks.

The two partial LoRA exports found in the original debugging output directories
are not distributed here and were not authenticated as manuscript checkpoints.
Their existence is not evidence of a complete trained semantic model; their
exclusion also must not be described as proof that the original archive had no
weights at all.

## Execution and evidence boundary

No training, model inference, model/data download, GPU workload or benchmark was
performed in preparing this edition. [verification/REPORT.md](../verification/REPORT.md)
lists the actual static and lightweight engineering checks. A syntax check,
dry-run plan, integrity check or isolated serializer check does not establish
real-model execution or reproduction of an experiment.

Pretrained FLUX/text/VAE assets, benchmark data and compatible model-specific
weights must be obtained separately under their distribution terms. Runtime
dependency inventories are not a tested environment lock. Missing complete
weights or logs limits reproducibility evidence but is not, by itself, a reason
to require a new experiment when disclosing existing source. Conversely, those
asset limits do not explain away a missing implementation.

No PSNR, SSIM, LPIPS, semantic accuracy, ablation, parameter-count, memory or
timing result was regenerated. Reported numbers remain manuscript results.
This collection supports transparent source inspection; it does not certify
that all manuscript results can be reproduced or that the reviewer's request
has been fully resolved.
