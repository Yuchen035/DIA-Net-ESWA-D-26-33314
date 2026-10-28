> This map was prepared against the original submitted 24-page manuscript. Section labels and page numbers may differ from the revised manuscript; the source-coverage limitations remain subject to review.

# Manuscript-to-source correspondence

This map compares the available source with the separately supplied, submitted
24-page manuscript for ESWA-D-26-33314. Page numbers refer to that manuscript,
not the Elsevier template examples in the reference archive. No additional
manuscript implementation was supplied for version 0.2. Coverage statements are
limited to these materials; they do not establish what exists elsewhere.

Code paths below are relative to `src/DIANet/`. Function names identify the
implementation independently of line-number changes. The two development paths
must be read separately:

- **FLUX/PEFT engineering path:** `train/`, `inference/`, `core/`, and the supplied
  PEFT/Diffusers trees. It includes real pretrained-model loading and token-wise
  feature MoE; it is not the manuscript's integrated semantic dual-routing model.
- **LLE prototype:** `LLE/` contains independent shared/specific routing, losses,
  data and metric components. The dual-route demonstration uses Mock models and
  has documented interface gaps.

## Semantic perception and routing

| Manuscript component | Source evidence | Coverage and difference |
| --- | --- | --- |
| Four indoor/outdoor × low-light/backlit categories; Gemini 3 Pro teacher and Qwen3-VL-4B-Instruct student (§3.2–3.3, pp7–8; training p13) | `LLE/core/utils/exposure.py`, `classify_exposure`; `LLE/core/data/dataset.py`, `ImagePairDataset.__getitem__` | Five brightness bins from thresholds `[0.1, 0.2, 0.3, 0.5]` are implemented. They do not identify the four semantic categories. Manuscript-specific annotation, distillation and four-class inference code was not found in the supplied project sources. |
| Shared and specific expert outputs (§3.4, pp9–11) | `LLE/core/lora/moe/router.py`, `SpecificRouter.forward`, `SharedRouter.forward`; `moe_lora_layer.py`, `MoELoRALayer._forward_dual_route` | Real standalone components select one category-specific expert with weight 1 and add a weighted Top-k shared output. The shared router pools tokens and uses Linear–ReLU–Dropout–Linear, whereas the manuscript specifies a single linear scorer. No complete real-FLUX integration of this combination is supplied. |
| Four specific + four shared experts, rank 16, shared Top-2 (p13) | `LLE/train_dual_route_simple.py`, `inject_dual_route_moe_lora`, `main`; `LLE/core/inference/pipeline.py`, constructor | Demonstration defaults are five specific + four shared experts, rank 8 and alpha 8. Changing the numbers alone would not provide semantic perception or the missing integrated objective. |
| Feature-based MoE in the main engineering path | `train/core/lora.py`, `LoRAConfigurator._apply_moe_lora`; `peft/src/peft/tuners/lora/layer.py`, `Linear.update_moe_layer`, `Linear.moe_forward` | Actual expert/router implementation exists. `moe_forward` fixes `moe_type="token_wise"`, routes from each token feature, and has no semantic-specific branch. It evaluates every expert before weighting outputs; sparse weights alone do not establish Top-k sparse execution cost. |
| Spatial semantic conditioning (§3.5.1, p11, after Eq.18) | `train/core/trainer.py`, `Trainer.train`; `LLE/core/inference/pipeline.py`, `enhance` | The manuscript expands `C_sem` spatially and concatenates it along latent channels. The main trainer concatenates input-image control and target tokens, not semantic channels. The prototype passes brightness categories to routers and concatenates `[latent, latent]` for its Mock input. Neither implements the stated semantic-channel construction. |

## Backbone and injection

`train/core/model.py`, `ModelLoader.load_all`, loads a real FLUX transformer, VAE,
CLIP/T5 encoders and flow-matching scheduler. `train/core/lora.py` injects adapters
through `transformer.add_adapter`; the retained PEFT changes propagate custom
MoE fields and module modes. Real loading source therefore exists, although its
runtime has not been validated in this edition.

The archived `train/config/train_lora_config.yaml` assigns ordinary LoRA to
dual-stream blocks and MoE to single-stream blocks: ordinary rank/alpha 16/16,
MoE rank/alpha 32/32, four experts and Top-2. The manuscript instead describes
Semantic MoE-LoRA in both block types, with the 4+4/rank-16 setting (pp10–13).

The following differences are visible in the supplied
`diffusers/src/diffusers/models/transformers/transformer_flux.py`; they do not
require a model run to identify:

| Archived rule or description | Supplied FLUX structure |
| --- | --- |
| `single_transformer_blocks.*.attn.to_out.0` | `FluxSingleTransformerBlock` sets attention `pre_only=True`; `FluxAttention` then omits `to_out`. |
| `single_transformer_blocks.*.ff.net.0.proj` / `ff.net.2` | The single-stream block uses `proj_mlp` and `proj_out`, not `ff.net.*`. Its actual `proj_mlp` is absent from the adapter target list. |
| A general `proj_out` suffix | It can also match the final model-level `proj_out`, beyond block outputs. The PEFT `Linear` fallback selects MoE for unmatched `proj_out` modes. |
| Separate gated FFN up/down/gate projections in §3.4.4 | The supplied dual-stream block uses GELU feed-forward modules. These cannot be equated with three separate gated FFN projections without an explicit mapping. |

These scientific injection rules were not rewritten for version 0.2. Exact
instantiated target counts, parameter counts and numerical behavior remain
unverified. The general `LLE/core/lora/configurator.py` supports `lora`, `moe`
and `freeze`, but the call in `LLE/train_lora.py` follows its ordinary-LoRA
compatibility path; configuration examples alone do not activate dual routing.

## Objectives and inference

| Manuscript component | Source evidence | Coverage and difference |
| --- | --- | --- |
| Conditional flow matching (§3.5.1, p11) | `train/core/trainer.py`, `Trainer.train`; `LLE/core/trainer.py`, `Trainer.train`; `LLE/core/loss/flow_matching.py`, `FlowMatchingLoss.forward` | Both main loops backpropagate velocity MSE. They use `(1-t)*clean+t*noise` and `noise-clean`; the separate component and manuscript use the reverse direction and `clean-noise`. A time change `s=1-t` can relate the conventions. Sign difference alone is not proof of an erroneous objective; model time inputs and sampling direction must be checked together. |
| Semantic-conditioned adversarial term with time FiLM and `f(D(fake)-D(real))` (§3.5.2, p12) | `LLE/core/loss/adversarial.py`, `AdversarialLoss`; `LLE/core/loss/discriminator.py`, `DualRouteDiscriminator` | Standalone adversarial code exists. It uses hinge/BCE/WGAN losses; its generator loss uses only the fake score. The discriminator concatenates image, time and category features before an MLP, rather than applying the stated time scale/shift FiLM. It is not the manuscript-equivalent adversarial term and is not connected to the main training objective. |
| Image-space L1 reconstruction and per-pixel cosine color consistency (§3.5.3, p12) | `LLE/core/loss/reconstruction.py`, `color.py` | Both files are placeholders. A corresponding project implementation was not found elsewhere in the supplied author code. These losses and the complete four-term objective are not supplied as an integrated training workflow. |
| Four-step semantic-conditioned ODE inference (p13) | `inference/inference_moe_lora.py`, `main`; `LLE/core/inference/pipeline.py`, `enhance` | The real FLUX pipeline accepts scheduler steps; the archived default is 28. The dual-route prototype makes one call at `t=0.5`, despite accepting a step argument. Its demonstration is not an ODE solver. Setting the real CLI to four steps does not supply the missing semantic model. |

## Entry points and checkpoint state

`LLE/train_dual_route_simple.py` injects into `transformer.blocks[i][0]`, the Mock
layout. Its `Trainer` call supplies `loss_fn`, `load_balance_loss_fn` and
`verbose`, whereas `LLE/core/trainer.py` accepts `config` and `logger`. This
archived interface mismatch remains unresolved. Version 0.2 requires an explicit
`--use-mock` selection in `LLE/infer_dual_route.py` and rejects its unsupported
real-model branch instead of substituting a Mock model. A real loader exists
elsewhere, but this does not make that entry point a working real-backbone
dual-route workflow.

The main engineering path now uses the version 0.2
`adapter_checkpoint.py` functions `save_adapter_checkpoint` and
`load_adapter_checkpoint`. The `dia-net-adapter-v2` format pairs a Safetensors
file with a `.safetensors.json` sidecar containing configuration, raw-key
inventory and integrity/model-structure metadata. It supports one adapter,
restores the raw adapter state strictly, and rejects legacy files without the
required metadata. This is a documented engineering replacement for the prior
incompatible save/load interface, not a recovered experiment checkpoint.
The main trainer's internal validation path uses the same protocol. No
real-model save/reload round trip or historical checkpoint conversion has been
performed. See [ENGINEERING.md](ENGINEERING.md) for protocol details.

The separate prototype format is unchanged:
`LLE/train_dual_route_simple.py` and `LLE/core/inference/pipeline.py` save/load
expert A/B lists only, omitting the shared-router classifier. Its loader also
marks state as loaded even when no layer keys match. That format cannot restore
the complete trained dual-route function and is not interchangeable with v2.

## Data, evaluation and reported quantities

| Item | Source and limit |
| --- | --- |
| Paired data | `train/core/data.py` pairs exact filenames and resizes images. `LLE/core/data/` contains additional components. Main configuration fields for prompts/cropping/flipping do not prove those operations are connected. The manuscript MixBL membership/split manifest, prescribed augmentation and model-selection workflow have not been recovered. |
| PSNR / SSIM | `LLE/core/utils/metrics.py` implements MSE-based PSNR and simplified global-statistics SSIM. PSNR does not force integer arrays to float before subtraction/squaring; callers must supply appropriate floating inputs and range. The SSIM helper is not a verified windowed benchmark evaluator. |
| LPIPS | `LLE/core/metrics/__init__.py` imports `.image_quality`, which is absent. This declaration does not supply a working LPIPS evaluator. |
| Parameter counts, memory and speed | Existing parameter/memory helpers are inspectable; they do not constitute measurement records for the manuscript tables. No parameter, memory, timing or quality result was regenerated for this edition. |

Manuscript settings in `configs/paper_reported_settings.json` are a labelled
transcription, not a recovered executable experiment configuration. Complete
paper checkpoints, per-image benchmark outputs and the original evaluation
environment are not supplied. These asset limits do not by themselves require
retraining; they also do not resolve the implementation differences above.

For prior methods and implementation provenance, including observed ICEdit
correspondence, see [METHOD_AND_ATTRIBUTION.md](METHOD_AND_ATTRIBUTION.md).
