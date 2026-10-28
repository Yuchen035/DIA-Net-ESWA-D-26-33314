"""Versioned, single-adapter checkpoints for the recovered engineering path.

This is serialization glue, not a DIA-Net algorithm or a legacy-weight converter.
The JSON sidecar is mandatory: expert configuration cannot be inferred safely
from tensor shapes. Imports here are standard-library only; model libraries are
imported only by the explicit save/load functions. Real-model round trips have
not been validated by the source-package checks.
"""
import hashlib
import json
import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
import tempfile

FORMAT = "dia-net-adapter-v2"
VERSION = 2
MOE_FIELDS = {"num_experts", "expert_rank", "expert_alpha", "top_k",
              "blc_alpha", "blc_weight", "module_modes"}


def _jsonable(value):
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Checkpoint configuration keys must be strings.")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=_canonical)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("Non-JSON configuration value: " + type(value).__name__)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_digest(config):
    return hashlib.sha256(_canonical(config).encode("utf-8")).hexdigest()


def _base_config(transformer):
    config = transformer.config
    values = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    # Private metadata includes machine paths/library version, not architecture.
    return _jsonable({key: value for key, value in values.items()
                      if not key.startswith("_")})


def _validate_lora_config(config):
    if not isinstance(config, dict) or config.get("peft_type") != "LORA":
        raise ValueError("A complete LORA configuration is required.")
    required = MOE_FIELDS | {"r", "lora_alpha", "lora_dropout", "target_modules"}
    if not required.issubset(config):
        raise ValueError("Missing adapter configuration fields: " +
                         ", ".join(sorted(required - config.keys())))
    if config.get("bias", "none") != "none" or config.get("modules_to_save"):
        raise ValueError("This protocol supports lora_* state only; bias/modules_to_save are unsupported.")
    _canonical(config)  # Also reject non-finite or non-JSON values.


def read_checkpoint_metadata(weights_path):
    """Validate the v02 sidecar and weight hash without importing model code.

    A matching base configuration is an architecture check, not proof that two
    base-model weight distributions are identical. Keep the recorded reference.
    """
    path = Path(weights_path)
    if path.suffix.lower() != ".safetensors" or not path.is_file():
        raise ValueError("An existing .safetensors checkpoint file is required.")
    sidecar = Path(str(path) + ".json")
    if not sidecar.is_file():
        raise ValueError("Missing v02 sidecar: " + str(sidecar) +
                         ". Legacy checkpoints are not inferred or silently loaded.")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("format") != FORMAT or metadata.get("version") != VERSION:
        raise ValueError("Unsupported adapter checkpoint format/version.")
    if metadata.get("adapter_count") != 1:
        raise ValueError("Only one adapter is supported by the local MoE expert naming.")
    name = metadata.get("adapter_name")
    if not isinstance(name, str) or not name or "." in name:
        raise ValueError("Invalid single-adapter name.")
    if metadata.get("weights_file") != path.name:
        raise ValueError("Checkpoint filename does not match its sidecar.")
    if metadata.get("weights_sha256") != _sha256(path):
        raise ValueError("Checkpoint SHA-256 does not match its sidecar.")
    _validate_lora_config(metadata.get("lora_config"))
    base_config = metadata.get("base_model_config")
    if not isinstance(base_config, dict) or not base_config:
        raise ValueError("Missing base-model architecture configuration.")
    if metadata.get("base_model_config_sha256") != _config_digest(base_config):
        raise ValueError("Base-model configuration signature mismatch.")
    if "base_model_reference" not in metadata or not isinstance(metadata["base_model_reference"], (str, type(None))):
        raise ValueError("Missing or invalid base-model reference field.")
    tensors = metadata.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("Checkpoint contains no declared adapter tensors.")
    for key, item in tensors.items():
        if not isinstance(key, str) or "lora_" not in key or not isinstance(item, dict):
            raise ValueError("Invalid adapter tensor declaration.")
        shape = item.get("shape")
        if not isinstance(shape, list) or not all(type(n) is int and n > 0 for n in shape):
            raise ValueError("Invalid tensor shape: " + key)
        if not isinstance(item.get("dtype"), str):
            raise ValueError("Missing tensor dtype: " + key)
    return metadata


def save_adapter_checkpoint(transformer, weights_path, *, adapter_name="default",
                            save_dtype=None, base_model_reference=None):
    """Save raw adapter keys plus full configuration; refuse multi-adapter state."""
    configs = getattr(transformer, "peft_config", {})
    if set(configs) != {adapter_name}:
        raise ValueError("Saving requires exactly the requested single adapter.")
    if not isinstance(adapter_name, str) or not adapter_name or "." in adapter_name:
        raise ValueError("Invalid adapter name.")
    lora_config = _jsonable(configs[adapter_name].to_dict())
    _validate_lora_config(lora_config)
    state = {}
    for key, value in transformer.state_dict().items():
        if "lora_" in key:
            tensor = value.detach().to(device="cpu", dtype=save_dtype or value.dtype)
            state[key] = tensor.contiguous()
    if not state:
        raise ValueError("Refusing to save an empty adapter checkpoint.")
    base_config = _base_config(transformer)
    if not base_config:
        raise ValueError("Missing base-model architecture configuration.")
    path = Path(weights_path)
    sidecar = Path(str(path) + ".json")
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Checkpoint filename must end in .safetensors.")
    if path.exists() or sidecar.exists():
        raise ValueError("Refusing to overwrite an existing checkpoint or sidecar.")
    path.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory(prefix=".adapter-save-", dir=path.parent) as directory:
        temporary_weights = Path(directory) / path.name
        save_file(state, str(temporary_weights))
        metadata = {
            "format": FORMAT, "version": VERSION, "adapter_count": 1,
            "adapter_name": adapter_name, "lora_config": lora_config,
            "base_model_reference": str(base_model_reference) if base_model_reference else None,
            "base_model_config": base_config,
            "base_model_config_sha256": _config_digest(base_config),
            "weights_file": path.name, "weights_sha256": _sha256(temporary_weights),
            "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                        for key, value in state.items()},
        }
        temporary_json = Path(directory) / sidecar.name
        temporary_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False,
                                             allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary_weights, path)
        os.replace(temporary_json, sidecar)
    return metadata


def load_adapter_checkpoint(transformer, weights_path):
    """Restore raw keys with the recorded config; any adapter coverage error fails.

    After an error the caller must discard the model, never fall back to an
    unadapted pipeline. This routine does not convert legacy checkpoints.
    """
    metadata = read_checkpoint_metadata(weights_path)
    if getattr(transformer, "peft_config", {}):
        raise ValueError("Load into a clean transformer; multiple adapters are unsupported.")
    if _config_digest(_base_config(transformer)) != metadata["base_model_config_sha256"]:
        raise ValueError("Checkpoint and base-model architecture configurations differ.")
    from safetensors.torch import load_file
    state = load_file(str(weights_path), device="cpu")
    declared = metadata["tensors"]
    if set(state) != set(declared) or not state:
        raise ValueError("Checkpoint tensor keys do not match the sidecar.")
    for key, tensor in state.items():
        if list(tensor.shape) != declared[key]["shape"] or str(tensor.dtype) != declared[key]["dtype"]:
            raise ValueError("Checkpoint tensor shape/dtype differs from declaration: " + key)
    from peft import LoraConfig
    config = LoraConfig(**metadata["lora_config"])
    name = metadata["adapter_name"]
    transformer.add_adapter(config, adapter_name=name)
    expected = {key: value for key, value in transformer.state_dict().items() if "lora_" in key}
    if set(expected) != set(state):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(f"Adapter keys do not match reconstructed configuration: missing={missing}; unexpected={unexpected}")
    for key, tensor in state.items():
        if tuple(expected[key].shape) != tuple(tensor.shape):
            raise ValueError("Adapter shape differs from reconstructed model: " + key)
    result = transformer.load_state_dict(state, strict=False)
    missing_adapter = [key for key in result.missing_keys if "lora_" in key]
    if missing_adapter or result.unexpected_keys:
        raise ValueError(f"Adapter restore incomplete: missing={missing_adapter}; unexpected={result.unexpected_keys}")
    transformer.set_adapter(name)
    transformer.eval()
    transformer.requires_grad_(False)
    return {"adapter_name": name, "loaded_tensor_count": len(state),
            "weights_sha256": metadata["weights_sha256"],
            "base_model_reference": metadata["base_model_reference"]}
