# Data and model inputs

This source snapshot does not redistribute pretrained models, benchmark images,
or authenticated manuscript checkpoints. Obtain external assets from their
original distributors under the applicable terms. Links below identify resources;
their files were not downloaded or executed for this release.

| Resource | Entry point | Relationship to this snapshot |
| --- | --- | --- |
| FLUX.1 Kontext dev | [Model page](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | Base family named by the archived engineering configuration. Supply a complete local snapshot. |
| FLUX.1 dev | [Model page](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Referenced by a separate LLE path; not an automatic substitute for Kontext. |
| Qwen3 VL 4B Instruct | [Model page](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Semantic student named in the manuscript; the manuscript-specific annotation, fine-tuning and classification pipeline is absent. |
| LOLv2 | [Dataset project](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) | Dataset source; exact manuscript membership lists were not supplied. |
| BAID | [Dataset project](https://github.com/lvxiaoqian/BacklitNet) | Dataset source; exact manuscript membership lists were not supplied. |
| PhotoDoodle | [Project](https://github.com/showlab/PhotoDoodle) | Optional external adapter referenced by legacy loaders; not the final DIA-Net checkpoint. |

The selected data loader pairs filenames between degraded-input and target
directories. A directory pair is not a benchmark split manifest. Manuscript
counts are transcribed separately in `configs/paper_reported_settings.json`;
they do not identify individual images or establish the original evaluation protocol.

Version 0.2 adapter inference requires the weight file and its
`.safetensors.json` sidecar from the new saver. Legacy weight files without
configuration metadata are refused instead of being interpreted as ordinary LoRA.
See [ENGINEERING.md](ENGINEERING.md). This replaces the known source-interface
mismatch through explicit code changes; real checkpoint compatibility remains untested.

Missing experiment materials include semantic teacher prompts and annotations,
student training/checkpoint records, authenticated enhancement and ablation
checkpoints, MixBL image membership, per-image outputs, and the exact PSNR/SSIM/LPIPS
evaluation protocol. Their absence is a limit of this snapshot; this release
does not fabricate them or recompute the manuscript's results.
