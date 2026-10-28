# Engineering entry points

The selected engineering path is `src/DIANet/train/train_lora.py` and
`src/DIANet/inference/inference_moe_lora.py`. Version 0.2 adds a matching adapter
checkpoint protocol and explicit failure handling. These changes do not supply
the missing manuscript algorithms identified in [PAPER_CODE_MAP.md](PAPER_CODE_MAP.md).
No real model execution or numerical reproduction has been validated.

## Inspection commands

Run from the extracted package root with Python 3.10 or newer:

```sh
python tools/verify_package.py
python tools/review.py check
python tools/review.py train --config configs/train_review.example.json
python tools/review.py infer --model-path assets/LOCAL_MODEL --lora-path runs/LOCAL_RUN/lora_final.safetensors --image data/INPUT.png --output runs/NEW_OUTPUT.png --prompt "make it bright"
```

These commands use the standard library and print inspection results or plans.
The inference preflight also reads the new checkpoint metadata helper, which has
no model-library imports at module scope. Training and inference entry points
are not imported by a dry run. Missing inputs are reported in `issues`; a dry
run can exit with code 0 despite missing prerequisites. `check` is a selected-file
AST and presence check, not the full-package syntax check or an import test.

The JSON example preserves archived engineering settings and changes only eight
resource/output paths. Relative training paths resolve against the config file's
directory. Inference paths resolve against the caller's working directory.
`paper_reported_settings.json` is a manuscript reference and is not executable.

## Adapter files in version 0.2

The selected saver writes a pair:

```text
lora_final.safetensors
lora_final.safetensors.json
```

Keep the two files together. The JSON sidecar records the `dia-net-adapter-v2`
format, adapter configuration and name, raw adapter tensor keys and shapes,
checkpoint SHA-256, and base-model configuration information. The training saver
and its validation sampling path use this same protocol.

The selected inference entry point validates the sidecar and checkpoint before
loading the base model, reconstructs the adapter configuration on the transformer,
and restores the raw keys with explicit key/shape coverage checks. It does not
send this custom MoE checkpoint through the upstream pipeline's ordinary
`load_lora_weights` conversion. This avoids the v0.1 prefix filter and repeated
adapter-name insertion identified by the source audit. The local custom PEFT
snapshot is required. The protocol supports one adapter per model, with
`bias="none"` and no `modules_to_save`; it serializes `lora_*` state only.
The supplied PEFT expert dictionaries do not establish independent multi-adapter
support. Configurations requiring other trainable state are rejected.

Legacy `.safetensors` files without this sidecar are rejected. There is no
automatic conversion that guesses missing ranks, expert counts, routing modes,
or model identity. The two partial development exports in the original archive
are not supplied as manuscript checkpoints. A matching config signature checks
architecture information, not the identity of all pretrained weight bytes.

The serializer and rejection paths have model-free checks; the actual
PyTorch/PEFT/FLUX save-and-reload operation remains untested. See
[verification/REPORT.md](../verification/REPORT.md) for the exact boundary.

## Explicit execution interface

`--execute` delegates to a model workload; it is not needed to inspect this
package. Execution requires compatible dependencies, authorized local pretrained
assets, data, and enough memory. The known method differences and remaining
engineering constraints below still apply. A guarded `--execute` request using
the unresolved example was tested and refused during preflight; no model
workload was launched to prepare this release.

The launcher prepends the local PEFT/Diffusers trees and selects the working
directory used by the relevant `core` package. It sets offline flags, refuses
missing resources, nonempty training output directories and existing inference
output files. These flags are dependency conventions, not a network sandbox.
The caller controls other cache/log/output locations in a custom configuration.

Omitting the adapter requires explicit `--base-model-only`, both in the wrapper
and the selected inference CLI. Such output would be unadapted FLUX, not DIA-Net.
Requested offloading errors terminate the selected inference path rather than
continuing in an unknown device state. If AdamW8bit is configured but
bitsandbytes is unavailable, training raises an error; it no longer
selects ordinary AdamW automatically. Changing optimizer choice must be explicit.

## Remaining source constraints

- The engineering path has token-wise feature routing and a velocity-MSE objective.
  It does not implement the paper's complete semantic dual-routing method.
- Some archived injection patterns do not match the local FLUX class; single-stream
  `ff.net.*` and `attn.to_out.0` are absent, while `proj_mlp` is not targeted.
  The generic `proj_out` suffix also includes the model's final output projection.
  Injection rules and algorithm parameters have not been silently rewritten.
- The main dataset loader pairs lowercase PNG/JPG filenames and resizes images.
  Per-image prompt-file, center-crop and random-flip config fields are not consumed
  there. The archived save-limit and TensorBoard fields are not a complete
  implemented retention/logging pipeline. Independent LLE utilities do not
  establish that the main entry point uses them.
- Per-batch timestep broadcasting needs validation before using batches larger
  than the archived default of one. The supplied runtime dependency inventory
  is not a tested environment lock. Inference defaults remain 28 steps and are
  not relabelled as the paper's four-step semantic protocol.
- `LLE/` is an archived research path with a Trainer API mismatch, incomplete
  losses, expert-only prototype checkpoints and a one-call inference demonstration.
  Its dual-routing inference now requires explicit `--use-mock`; without that
  flag it refuses execution instead of substituting a mock model. This guard does
  not turn the prototype into a working paper implementation.
- Historical tests can load models, select CUDA, download resources or train.
  They are not the lightweight verification entry points listed above.

## Preparing a future runtime

Resolve [DEPENDENCIES.md](DEPENDENCIES.md) in a separate environment. The
following local installs do not resolve or certify the rest of that environment:

```sh
python -m pip install -e ./src/DIANet/peft --no-deps
python -m pip install -e ./src/DIANet/diffusers --no-deps
```

No installation was performed for this release. Record the resolved environment,
model identities and any future execution results separately from packaging checks.
