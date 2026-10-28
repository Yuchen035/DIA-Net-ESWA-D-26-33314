> Historical v0.2 record: statements and checks below refer to the 12 October 2026 package, not the current package. See RELEASE_NOTE.md for current scope.

# Source package changes

## 0.2.0 — submission candidate

- Clarified the difference between available implementations, known interface
  defects, missing manuscript components and untested runtime behavior. This
  release remains a source collection, not a complete reproduction package.
- Added `adapter_checkpoint.py` with the `dia-net-adapter-v2` contract: raw LoRA
  safetensors and a companion named `<weights-file>.json`, such as
  `lora_final.safetensors.json`. It records a single adapter's `LoraConfig`,
  key/shape/dtype inventory, weight SHA-256, base-model reference and structural
  configuration signature. Loading checks these fields and rejects absent
  metadata, incompatible/empty states, multiple adapters and incomplete MoE
  configuration. Historical checkpoints without this metadata are rejected;
  their missing configuration is not guessed. The main saver, selected inference
  entry and internal validation use this protocol. This engineering correction
  does not implement the paper's missing semantic routing.
- Kept the review commands in dry-run mode by default. Execution remains an
  explicit action. The selected inference entry requires an explicit base-only
  choice when no adapter is used, and rejects invalid checkpoint inputs and
  offload failures. These safeguards do not validate or repair the separate
  research prototypes as manuscript implementations.
- Corrected direct training-entry import precedence while keeping `train/core`
  ahead of the unrelated top-level `core` namespace. Missing AdamW8bit support
  now raises an error instead of switching optimizers. The LLE dual-route
  inference prototype now requires explicit Mock selection and rejects its
  unsupported real-model path.
- Replaced stale v0.1 verification claims with v0.2 checks. Historical debugging
  prose and old check records are retained separately for the authors.
- Added explicit ICEdit/PEFT implementation correspondence and the unresolved
  author attribution question. Identified Diffusers as an unmodified local
  upstream snapshot. Preserved the four supplied license texts.
- Added `provenance/EXCLUSIONS.json` for the three omitted author tests, two
  partial LoRA exports from debugging paths, and other excluded material.
  Historical logs and planning reports remain author-facing evidence.

Validation is limited to the lightweight checks recorded with this candidate.
No real-model/tensor save-load round trip, model loading, training, inference,
benchmark, or GPU experiment was performed. No missing scientific algorithm,
loss term, dataset split, paper parameter claim or result value was backfilled.

## 0.1.0 — initial source review collection

### Selected existing materials

- Retained the main FLUX Kontext training modules, inference sources, shared
  utilities, LLE research source and configuration files.
- Preserved local PEFT and Diffusers runtime source and installation metadata.
- Preserved upstream license texts and recorded source provenance.
- Excluded tutorial material, third-party examples/test
  suites, three author tests, images, old logs, two partial debugging-path LoRA
  exports and repository object data. The exports were previously described too
  broadly as "toy weights"; v0.2.0 records their actual scope and limitations.

### Non-algorithmic changes

- Added prominent notices to the four supplied modified PEFT files, preserving
  original copyright headers and recording both original and packaged hashes.

- Replaced the original machine-specific project root in selected source/config
  files with `__LOCAL_PROJECT_ROOT__`; the exact diff is retained.
- Added an English entry guide, Chinese author guide, paper–code map,
  dependency/resource instructions and distribution notices.
- Added a portable configuration derived from the retained engineering
  configuration, together with a separately labeled, non-executable manuscript
  settings record.
- Added inspection and explicit-execution wrappers. These control entry-point
  selection, path resolution and local-library import precedence without
  implementing missing paper components.
- Added source/checksum manifests and lightweight packaging verification.

No scientific implementation was backfilled. No loss term, expert architecture,
dataset split or benchmark value was changed to manufacture agreement with the
manuscript. No experiment was run as part of this edition.
