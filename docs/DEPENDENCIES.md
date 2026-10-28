# Dependency inventory for source review

This is a static inventory of the supplied source and package metadata. It is not the original experiment environment, a lock file, a tested installation recipe, or evidence that the paper results can be reproduced. No model workload or installation was executed for this release. The standard-library packaging and checkpoint-contract checks are recorded separately.

## Minimal source layout

Retain `src/DIANet/{train,inference,core,LLE}` with their original relative layout. Retain complete `src/DIANet/peft/src/` and `src/DIANet/diffusers/src/`, plus each package's `setup.py`, `pyproject.toml`, `README.md`, and `LICENSE`. Also retain `diffusers/MANIFEST.in` and any other packaging metadata present and required. Keeping the full source trees preserves lazy imports, utility modules, CUDA source resources, `py.typed`, and the Diffusers model-card template.

Third-party examples, tests, documentation, repository history, the educational `Learning` tree, ICEdit and PhotoDoodle source checkouts, images, logs, and debug weights are outside this minimal source-only bundle. The author LLE prototype remains available for review but must not be described as the verified manuscript evaluation pipeline.

## Local packages are required

- **PEFT `0.18.1.dev0`, modified snapshot:** preserve the four modified files recorded in `THIRD_PARTY_NOTICES.md`. Install or resolve this local package rather than PyPI PEFT. `peft/setup.py:59–70` declares Python >= 3.10.0 and the dependencies below.
- **Diffusers `0.36.0.dev0`, supplied snapshot:** preserve the local source version. Its recorded source equals the upstream HEAD for currently extracted tracked files. `diffusers/setup.py:261–291` defines installation metadata; optional package minimums appear at lines 97–149.
- `train/train_lora.py` and `train/core/model.py` prepend local PEFT and Diffusers paths. The selected v0.2 inference entry point also selects local PEFT and Diffusers explicitly; older archived inference scripts do not consistently do so. LLE's entrypoint selects its own `core` package through `sys.path`. Runtime setup must make both vendored packages visible and preserve the expected working directory/import precedence. Copying files alone does not validate imports.

## Runtime packages and source-derived constraints

| Package | Source-derived declaration / role |
|---|---|
| Python | >= 3.10 from local PEFT metadata; this is a metadata minimum, not a validated environment version. |
| torch | >= 1.13.0 from PEFT metadata; tensor/model training and inference. This lower bound alone does not establish compatibility with every FLUX API used here. |
| torchvision | Direct image transformation imports; no author version pin established. Select a version compatible with the selected PyTorch build. |
| transformers | CLIP/T5 models, tokenizers, scheduler helpers; Diffusers optional metadata declares >= 4.41.2. This is not a tested combined-environment guarantee. |
| accelerate | >= 0.31.0 is the stronger of the local Diffusers/PEFT metadata minima; offloading and model loading. |
| optimum-quanto | Direct top-level `from optimum.quanto import QTensor` in both `train/core/trainer.py` and `LLE/core/trainer.py`; required when importing those trainers even if quantization is disabled in a config. Diffusers optional metadata declares >= 0.2.6. |
| numpy | >= 1.17 from PEFT metadata; image arrays and metrics. |
| Pillow | Direct `PIL` imports; no author version pin established. |
| PyYAML | Direct config parsing; can be the only external package in a separate lightweight static-review requirements file if the review scripts only need YAML parsing. |
| einops | Direct trainer imports. |
| tqdm | Direct training, caching, and inference progress imports. |
| loguru | Direct loader/memory-module imports. |
| safetensors | >= 0.3.1 from Diffusers metadata; adapter and cache serialization. |
| packaging | >= 20.0 from PEFT metadata. |
| psutil | PEFT metadata dependency. |
| huggingface-hub | >= 0.34.0, < 2.0 from local Diffusers metadata, stronger than the local PEFT minimum. |
| httpx | < 1.0.0 from Diffusers metadata. |
| importlib_metadata, filelock, requests | Diffusers base dependencies; no tighter author pins established. |
| regex | != 2019.12.17 from Diffusers metadata. |
| setuptools | Local packages use setuptools packaging; this is a build/install dependency, not a reconstructed environment pin. |

The listed version bounds are transcribed from supplied source metadata. They must not be represented as a resolved or tested set. In particular, no exact PyTorch/CUDA, GPU, or Python version is verified by this inventory. Prefer local editable/package installs of the two bundled libraries over replacing them with stock `peft` or `diffusers` requirements.

## Conditional dependencies and resources

- **bitsandbytes:** used by 8-bit optimizer/memory paths. `train/core/optimizer.py`, `create_optimizer`, imports it when AdamW8bit is requested and now fails explicitly if unavailable. Selecting ordinary AdamW requires an explicit configuration change. Local Diffusers optional metadata declares >= 0.43.3. Other selected memory paths should be reviewed against the chosen configuration before treating it as universally optional.
- **sentencepiece / protobuf:** tokenizer/model-loading support may require them depending on which supplied tokenizer assets are used. Local Diffusers optional metadata lists `sentencepiece>=0.1.91,!=0.1.92` and `protobuf>=3.20.3,<4`; these are not demonstrated author-environment pins.
- **pytest:** imported by a subset of retained author tests, not required merely to read the source. Tests have not been executed.
- **External FLUX assets:** base-model transformer, VAE, text encoders/tokenizers and other files referenced by the selected loader must be supplied at configured paths, with the applicable original model terms. They are not supplied by copying the Diffusers library.
- **PhotoDoodle weights:** `core/model_loader.py:106–134` and `LLE/core/model_loader.py:308–325` call `pipeline.load_lora_weights(lora_path)`. These use the Diffusers API and an external checkpoint, not the `PhotoDoodle/` source directory. Some supplied configs enable this step. Omitting the PhotoDoodle source checkout is therefore compatible with source review, but does not eliminate the required weight file when enabled.
- **ICEdit:** no static import from the ICEdit checkout was found in selected `core`, `train`, `LLE`, or `inference` Python files. It is not an established import dependency of these entrypoints. Related token-wise MoE source in its supplied tree requires implementation attribution; see THIRD_PARTY_NOTICES.md.
- **Formal DIA-Net reproduction:** the base assets, actual trained manuscript checkpoints, exact dataset splits and evaluation inputs, and a verified environment remain separate requirements. This source-only bundle does not establish their completeness.

Do not merge `PhotoDoodle/requirements.txt` or `ICEdit/requirements.txt` into the core environment blindly: those are requirements for independent upstream projects and contain versions different from the two bundled libraries. The broader PEFT `requirements.txt` also contains test/development dependencies and an unqualified stock `peft` entry, so it is not a suitable DIA-Net runtime lock file.
