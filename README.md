# DIA-Net source materials for review

**Manuscript:** ESWA-D-26-33314  
**Title:** DIA-Net: A Dual-Illumination Adaptive Network with Semantic Mixture-of-Experts Low-Rank Adaptation for Image Enhancement  
**Source review edition:** 28 October 2026

This archive makes the available project source accessible for inspection. It
contains a FLUX Kontext/PEFT engineering path and separate LLE research
components. **It does not contain an integrated, complete implementation of the
manuscript's four-category semantic DIA-Net pipeline.** No additional manuscript
implementation was supplied for this edition. Coverage and limitations are
documented in [PAPER_CODE_MAP.md](docs/PAPER_CODE_MAP.md).

No training, model inference, model/data download, or numerical reproduction was
performed in preparing this edition. The historical [verification/REPORT.md](verification/REPORT.md) describes the 12 October package; current checks are summarized in [RELEASE_NOTE.md](RELEASE_NOTE.md). Passing package checks does not establish
that the manuscript results can be regenerated.

## Inspect the package

Run from the extracted package root with Python 3.10 or newer. This is a syntax
baseline for inspection, not a validated model runtime environment.

```sh
python tools/review.py check
python tools/verify_package.py
python tools/review.py train --config configs/train_review.example.json
```

These commands check files or print a plan; they do not train a model. The
example deliberately contains unresolved local-resource paths. Read
[ENGINEERING.md](docs/ENGINEERING.md) and [DEPENDENCIES.md](docs/DEPENDENCIES.md)
before considering explicit execution. Runtime dependencies, assets and actual
model behavior remain unverified.

## Source coverage

| Location | Contents |
| --- | --- |
| `src/DIANet/train/`, `inference/`, `core/` | Real-model loading and adaptation code; the active training objective is flow-matching velocity MSE. |
| `src/DIANet/peft/` | Preserved local PEFT source with token-wise MoE extensions. |
| `src/DIANet/diffusers/` | Preserved upstream Diffusers snapshot used by the engineering path. |
| `src/DIANet/LLE/` | Shared/specific routers, independent loss/data/metric components and Mock demonstrations with documented integration gaps. |
| `src/DIANet/adapter_checkpoint.py` | Version 0.2 engineering addition for explicit adapter metadata and strict restoration. |
| `configs/` | Portable engineering example and separately labelled, non-executable manuscript settings. |
| `docs/`, `provenance/`, `verification/` | Method correspondence, attribution, source/change records and check results. |

The main limitations are:

- The Gemini-to-Qwen semantic perceiver, four-category inference, and spatial
  semantic-label concatenation described in the manuscript are not supplied.
- Real-model PEFT routing is token-wise feature MoE. Shared/specific dual routing
  exists in the separate prototype with different defaults; its demonstration
  uses Mock models. Some archived injection rules do not match the supplied FLUX
  class structure. These scientific configurations were not rewritten here.
- Main training loops do not implement the four-term manuscript objective.
  Reconstruction/color files are placeholders; standalone adversarial components
  use different loss and conditioning definitions from the manuscript.
- Complete manuscript checkpoints, data/split manifests and the original
  evaluation workflow are unavailable. Existing PSNR and simplified SSIM helpers
  are not a verified evaluator for the reported tables.

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the practical consequences.

## Version 0.2 engineering changes

The main adapter save/load path now uses the explicit `dia-net-adapter-v2` format:
a Safetensors file plus a JSON sidecar, such as `lora_final.safetensors.json`.
It records configuration and raw adapter keys, validates the saved assets, and
rejects unsupported legacy files. It supports one adapter. This repairs the
documented interface design; a real-model checkpoint round trip has not been
run, and historical weights have not been converted or authenticated. See
[ENGINEERING.md](docs/ENGINEERING.md) and [CHANGELOG.md](docs/CHANGELOG.md).

The recovered scientific components and engineering additions have distinct
provenance. These changes do not supply missing manuscript algorithms or change
reported results. Manuscript settings remain separate from executable examples.

## Sources and external assets

LoRA, MoE, flow matching, FLUX and the underlying libraries are prior work.
[METHOD_AND_ATTRIBUTION.md](docs/METHOD_AND_ATTRIBUTION.md) also documents the
observed correspondence with ICEdit code; a local modification is not by itself
evidence of an original algorithm.

Models and datasets must be obtained separately under their applicable terms;
see [DATA_AND_MODELS.md](docs/DATA_AND_MODELS.md). The archive excludes original
logs, generated outputs, selected development tests, caches, Git objects and two
partial LoRA exports from debugging paths, which were not authenticated as
manuscript checkpoints. Historical development claims are not validation results.

Read [DISTRIBUTION.md](DISTRIBUTION.md) and
[THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md) for distribution and licensing
scope. This package does not assert public repository publication or successful
resolution of the reviewer's reproducibility request.
