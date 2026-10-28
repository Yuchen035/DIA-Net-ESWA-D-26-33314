# Method orientation and implementation attribution

The manuscript proposes combining semantic perception with shared and
category-specific low-rank adaptation for low-light and backlit enhancement.
Indoor/outdoor context yields four semantic categories. This is the manuscript's
claimed research combination; [PAPER_CODE_MAP.md](PAPER_CODE_MAP.md) states which
parts are actually present in the supplied source. A proposal, directory name
or local patch is not independent proof of implementation completeness or
algorithmic originality.

## Prior methods and software

| Component | Source | Role in this collection |
| --- | --- | --- |
| Low-rank adaptation | Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | Existing low-rank parameterization used through PEFT and local adaptations. LoRA itself is not a contribution of this manuscript. |
| Sparse mixture of experts | Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | Prior expert-routing principle. The retained implementations must be examined for their particular routing behavior. |
| Flow matching and conditional paths | Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | Existing generative training framework. The velocity objective is not an original theoretical contribution of this manuscript. |
| FLUX Kontext | Black Forest Labs et al., [FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space](https://arxiv.org/abs/2506.15742) | Pretrained backbone family selected by the recovered engineering configuration. Backbone algorithms and weights remain prior work. |
| Qwen3-VL-4B-Instruct | [Original model distribution](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Pretrained model named as the manuscript's student. Manuscript-specific annotation/distillation/inference code is not supplied here. |
| PEFT and Diffusers | Exact supplied commits and notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Upstream software. Four supplied PEFT files contain local modifications; the archived Diffusers tracked files matched their recorded upstream snapshot; this package removes one non-functional class docstring from one Diffusers file. |

These references identify prior components; they do not certify that the
manuscript has already been revised to include every necessary citation. The
source package does not amend the article's bibliography.

## Local PEFT changes and ICEdit correspondence

The preserved `provenance/peft_local_changes.patch` records differences from the
supplied upstream PEFT snapshot in `config.py`, `layer.py`, `model.py` and
`utils/save_and_load.py`. These include MoE fields, module-mode selection,
expert/router parameters and state handling. The patch makes local changes
inspectable; it does not establish that every changed idea or line originated
with DIA-Net.

The original archive also contains an ICEdit checkout. Its
`ICEdit/icedit/peft/tuners/lora/layer.py` MoE forward implementation and the
retained `peft/src/peft/tuners/lora/layer.py`, `Linear.moe_forward`, show concrete
correspondence: token-wise routing, `lora_route`, `expert_i` keys,
Top-k → softmax → scatter, and evaluation/weighting of each expert. The
associated configuration files also share six MoE fields and defaults.

This correspondence is relevant implementation provenance even though the
retained runtime does not import the ICEdit checkout. The available archive does
not establish the direction or full history of reuse, or which overlapping
elements are original. The author-confirmed file/function-level adaptation
history remains unresolved. Neither a no-import observation nor a local patch
should be used to assert independent originality. The retained ICEdit and other
third-party license texts must be considered under their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[DISTRIBUTION.md](../DISTRIBUTION.md).

## Distinguishing source roles

| Source group | Interpretation |
| --- | --- |
| `train/`, `inference/`, modified PEFT | Real-model engineering source with ordinary LoRA/token-wise MoE and flow-matching MSE; not the integrated semantic architecture described in the manuscript. |
| `LLE/core/lora/moe/` | Available shared/specific routing prototypes. The shared router differs from the manuscript's single-linear scorer and the demonstrated integration uses Mock models. |
| `LLE/core/loss/` | Available standalone flow-matching/adversarial components plus reconstruction/color placeholders. Existing adversarial code is not equivalent to the manuscript FiLM/relative-score definition. |
| `adapter_checkpoint.py`, launchers and packaging documents | Traceable engineering/packaging additions. The version 0.2 checkpoint protocol does not constitute a new scientific method or a recovered historical experiment record. |
| Manuscript-only stages identified in the source map | Missing from the supplied implementation materials; their proposed novelty cannot be validated from absent code. |

Paths in this table are relative to `src/DIANet/`, except package launchers and
documents. Start at `train/train_lora.py`, follow adapter injection in
`train/core/lora.py`, then inspect the objective receiving the backward pass in
`train/core/trainer.py`. Inspect `LLE/core/lora/moe/` separately. This separates
prior methods, local engineering, standalone components and missing integration
without presenting them as one validated experiment pipeline.
