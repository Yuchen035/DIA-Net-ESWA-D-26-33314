# Third-party provenance and notices

This document records the provenance of the source retained under `src/DIANet/`.
Repository identifiers and comparisons come from the supplied archive, not a
reconstructed experiment environment. They do not establish successful model
execution or reproduction of the manuscript results.

## Retained runtime source

| Component | Recorded upstream repository | Recorded HEAD | Local package version | License |
|---|---|---|---|---|
| PEFT | https://github.com/huggingface/peft.git | `b10527e82c2171568f538f5b822817e8a753672a` | `0.18.1.dev0` | Apache License 2.0 |
| Diffusers | https://github.com/huggingface/diffusers.git | `1b91856d0eee7b6fb58340e9b54ea2c3d5424311` | `0.36.0.dev0` | Apache License 2.0 |

Full licenses remain at `src/DIANet/peft/LICENSE` and
`src/DIANet/diffusers/LICENSE`, with byte-identical copies under `licenses/`.
Both are 11,357 bytes, SHA-256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
No separate NOTICE file was found in either supplied checkout. Original
source-level copyright and license headers are retained.

PEFT is a locally modified snapshot. An independent comparison of the original
archive against all 635 tracked files at its recorded HEAD identified four
changed files:

- `peft/src/peft/tuners/lora/config.py`: MoE-LoRA fields and module-mode configuration.
- `peft/src/peft/tuners/lora/layer.py`: expert construction, routing, and MoE forward behavior.
- `peft/src/peft/tuners/lora/model.py`: propagation of layer context and MoE configuration.
- `peft/src/peft/utils/save_and_load.py`: inclusion of expert weights in adapter state export.

`provenance/peft_local_changes.patch` records those supplied changes; its content
was independently regenerated from the archived Git objects and source bytes.
An unmodified PEFT checkout or stock PyPI installation is not a substitute.
The four files retain the modification notices added during v0.1.0 packaging.
Original and packaged hashes are recorded in `provenance/source_manifest.json`;
the notice-only changes are in `provenance/packaging_notices.patch`.
Local differences from Hugging Face HEAD are not, by themselves, evidence of
original scientific contributions. See the ICEdit correspondence below.

All 2,289 original Diffusers tracked files matched the recorded HEAD in the original archive. In this package, one class docstring was removed from `diffusers/utils/source_code_parsing_utils.py`; the retained executable statements are unchanged. The original snapshot and historical comparison are documented in `provenance/`.
Both libraries retain their complete non-cache `src/` trees and installation
metadata: 189 files for PEFT and 774 for Diffusers. Non-Python resources,
including PEFT's BOFT C++/CUDA files and Diffusers' `model_card_template.md`, are
included. Upstream examples, tests and documentation are omitted. The retained
snapshot versions should not be replaced arbitrarily with current PyPI releases.

## Referenced projects omitted from the runtime source bundle

| Project | Recorded upstream and HEAD | Supplied license | Relationship to the selected DIA-Net code |
|---|---|---|---|
| PhotoDoodle | https://github.com/showlab/PhotoDoodle.git — `0a437fbdc44413a99526dbe490f1e127eae04893` | MIT; Copyright (c) 2025 Show Lab | No source-subtree import was found. Optional PhotoDoodle LoRA weights are loaded through Diffusers; omitting this checkout does not supply or remove that external weight requirement. |
| ICEdit | https://github.com/River-Zhang/ICEdit.git — `b111091c92dd6f9facaf527497744a83778c0cf4` | FLUX.1 [dev] Non-Commercial License | No runtime import from this checkout was found; implementation correspondence with the retained MoE code requires separate attribution clarification below. The checkout is not bundled. |

Full original texts are retained as `licenses/PHOTODOODLE-LICENSE.txt` and
`licenses/ICEDIT-LICENSE.txt`; their hashes are in
`SHA256SUMS.txt`. The supplied ICEdit license is not
MIT or Apache. A reference license does not assign a license to unrelated files,
and PhotoDoodle's source license does not establish external model-weight terms.

## ICEdit implementation correspondence

The original archive's `ICEdit/icedit/peft/tuners/lora/layer.py`,
`Linear.moe_forward`, and the retained
`src/DIANet/peft/src/peft/tuners/lora/layer.py`, `Linear.moe_forward`, share a
concrete token-routing structure: `lora_route`, `expert_i`, top-k selection,
softmax, scatter and per-expert weighted accumulation. Their `LoraConfig`
definitions also share six fields and defaults: `num_experts=6`,
`expert_rank=32`, `expert_alpha=32`, `top_k=1`, `blc_alpha=0`, `blc_weight=0`.

The authors need to confirm the implementation lineage and identify any reused
or adapted portions separately from their own changes. Absence of a runtime
import does not rule out adaptation. These static correspondences do not
establish copying direction, authorship, originality or a license violation;
this package does not claim that the whole local PEFT patch is original work.

The provenance records do not exhaust every line-level attribution. No single
license for the author-owned DIA-Net files was supplied at the project root;
this package assigns no blanket open-source license to the collection.

The omitted educational `Learning/MoE_with_CNN` checkout records
https://github.com/frezazadeh/Mixture-of-Experts-MoE.git at
`d2ba6e5e7fe2f2731fe6ad389f8d8c22a8ef01b0`. No license was found in its supplied
tree. `provenance/EXCLUSIONS.json` records other deliberate omissions, including
author tests and partial LoRA exports from debugging paths.
