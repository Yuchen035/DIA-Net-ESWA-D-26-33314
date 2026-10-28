#!/usr/bin/env python3
"""Static review launcher for recovered source. No model imports in dry-run mode.

This launcher does not implement DIA-Net, reproduce manuscript results, install
dependencies, or download assets. Explicit --execute calls a retained entry point.
"""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "src" / "DIANet"
ENTRY = {"train": "train/train_lora.py", "infer": "inference/inference_moe_lora.py"}
REQUIRED = [
    *ENTRY.values(), "adapter_checkpoint.py", "train/core/config.py", "train/core/trainer.py",
    "peft/src/peft/tuners/lora/config.py", "peft/src/peft/tuners/lora/layer.py",
    "peft/src/peft/tuners/lora/model.py", "peft/src/peft/utils/save_and_load.py",
    "diffusers/src/diffusers/pipelines/flux/pipeline_flux_kontext.py",
    "peft/LICENSE", "diffusers/LICENSE",
]


def validate_adapter(path, root):
    """Read the v0.2 checkpoint contract without importing model libraries."""
    module_path = root / "adapter_checkpoint.py"
    spec = importlib.util.spec_from_file_location("dia_checkpoint_inspection", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("Checkpoint metadata inspector is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_checkpoint_metadata(path)
PATH_FIELDS = [
    ("model", "name_or_path"), ("dataset", "input_image_dir"),
    ("dataset", "target_image_dir"), ("save", "output_dir"),
    ("memory_optimization", "cache_dir"), ("logging", "tensorboard_dir"),
    ("validation", "save_dir"), ("validation", "validation_image"),
]
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
               "PYTHONDONTWRITEBYTECODE": "1"}


def local_path(value, base):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def placeholders(value):
    return any(token in str(value) for token in ("REPLACE_ME", "<", ">", "${"))


def inspect_source(root):
    errors = []
    for rel in REQUIRED:
        if not (root / rel).is_file():
            errors.append("Missing retained file: " + rel)
    files = set((root / "train").rglob("*.py"))
    files.update(root / rel for rel in REQUIRED if rel.endswith(".py"))
    count = 0
    for path in sorted(files):
        if not path.is_file():
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            count += 1
        except (SyntaxError, UnicodeError) as exc:
            errors.append(str(exc))
    return {"scope": "AST parse and file presence only", "parsed_python_files": count,
            "errors": errors, "model_imports": False, "runtime_validated": False,
            "paper_results_reproduced": False}


def load_train_config(path):
    """JSON is a YAML subset accepted by retained TrainingConfig.from_yaml."""
    if path.suffix.lower() != ".json":
        raise ValueError("Use a .json configuration: dry-run intentionally uses only the standard library. "
                         "The preserved YAML is a reference, not parsed by this wrapper.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Training configuration must be a JSON object.")
    normalized = json.loads(json.dumps(data))
    issues = []
    for section, key in PATH_FIELDS:
        value = normalized.get(section, {}).get(key)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise ValueError(section + "." + key + " must be a path string.")
        if placeholders(value):
            issues.append("Set local path: " + section + "." + key)
        normalized[section][key] = str(local_path(value, path.parent))
    return normalized, issues


def validate_train(config, issues):
    required = [("model", "name_or_path"), ("dataset", "input_image_dir"),
                ("dataset", "target_image_dir")]
    for section, key in required:
        value = config.get(section, {}).get(key)
        if not value or not Path(value).is_dir():
            issues.append("Required local directory missing: " + section + "." + key)
    output = config.get("save", {}).get("output_dir")
    if not output:
        issues.append("Set save.output_dir explicitly.")
    elif Path(output).exists() and (not Path(output).is_dir() or any(Path(output).iterdir())):
        issues.append("Choose a new or empty output directory to avoid overwriting retained evidence.")
    val = config.get("validation", {})
    if val.get("enabled") and not Path(val.get("validation_image", "")).is_file():
        issues.append("Enabled validation requires a local validation.validation_image file.")
    inputs = Path(config.get("dataset", {}).get("input_image_dir", "__missing__"))
    targets = Path(config.get("dataset", {}).get("target_image_dir", "__missing__"))
    if inputs.is_dir() and targets.is_dir():
        images = sorted(list(inputs.glob("*.png")) + list(inputs.glob("*.jpg")))
        if not images:
            issues.append("Input directory has no .png/.jpg files supported by the retained loader.")
        missing = [p.name for p in images if not (targets / p.name).is_file()]
        if missing:
            issues.append("Target files missing for " + str(len(missing)) + " input filename(s).")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="mode")
    sub.add_parser("check", help="File and AST inspection; does not import retained code")
    train = sub.add_parser("train", help="Print a plan; --execute calls preserved training")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--execute", action="store_true")
    infer = sub.add_parser("infer", help="Print a plan; --execute calls preserved inference")
    for flag in ("model-path", "image", "output", "prompt"):
        infer.add_argument("--" + flag, required=True)
    infer.add_argument("--lora-path")
    infer.add_argument("--base-model-only", action="store_true",
                       help="Explicitly request unadapted backbone; not DIA-Net inference")
    infer.add_argument("--height", type=int, default=1024)
    infer.add_argument("--width", type=int, default=1024)
    infer.add_argument("--num-steps", type=int, default=28)
    infer.add_argument("--guidance", type=float, default=2.5)
    infer.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float16")
    infer.add_argument("--offload", choices=["sequential", "model", "none"], default="sequential")
    infer.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.source_root.expanduser().resolve()
    status = inspect_source(root)
    if args.mode in (None, "check"):
        print(json.dumps({"source_root": str(root), **status}, indent=2))
        return 1 if status["errors"] else 0
    issues = list(status["errors"])
    normalized = None
    if args.mode == "train":
        config_path = args.config.expanduser().resolve()
        normalized, config_issues = load_train_config(config_path)
        issues.extend(config_issues)
        validate_train(normalized, issues)
        arguments = ["--config", "<temporary-normalized-config.json>"]
        details = {"config_path": str(config_path), "normalized_config": normalized}
    else:
        arguments = []
        for key in ("model_path", "image", "output", "prompt", "height", "width",
                    "num_steps", "guidance", "dtype", "offload"):
            value = getattr(args, key)
            if key in ("model_path", "image", "output"):
                value = str(local_path(value, Path.cwd()))
            arguments.extend(["--" + key.replace("_", "-"), str(value)])
        if not Path(args.model_path).expanduser().is_dir():
            issues.append("--model-path must name an existing local model directory.")
        if not Path(args.image).expanduser().is_file():
            issues.append("--image must name an existing local input image.")
        if Path(args.output).expanduser().exists():
            issues.append("--output already exists; choose a fresh filename.")
        if args.num_steps < 1 or args.height < 16 or args.width < 16:
            issues.append("Steps must be positive and height/width at least 16.")
        if args.height % 16 or args.width % 16:
            issues.append("Choose height and width divisible by 16 for retained FLUX latent packing.")
        if args.lora_path and args.base_model_only:
            issues.append("Choose either --lora-path or --base-model-only.")
        if args.lora_path:
            adapter = local_path(args.lora_path, Path.cwd())
            try:
                validate_adapter(adapter, root)
            except (ValueError, OSError, RuntimeError) as exc:
                issues.append("Adapter preflight failed: " + str(exc))
            arguments.extend(["--lora-path", str(adapter)])
        elif not args.base_model_only:
            issues.append("Supply --lora-path, or explicitly select --base-model-only.")
        if args.base_model_only:
            arguments.append("--base-model-only")
        details = {"inference_kind": "base model only" if args.base_model_only else "retained adapter path"}
    plan = {"mode": args.mode, "execute_requested": args.execute,
            "entrypoint": str(root / ENTRY[args.mode]), "arguments": arguments,
            "working_directory": str(root / "train" if args.mode == "train" else root),
            "offline_environment": OFFLINE_ENV, "issues": issues,
            "runtime_validated": False, "paper_results_reproduced": False, **details}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("DRY RUN ONLY: retained code was not imported or executed.")
        return 0
    if issues:
        print("Execution not started: resolve the listed local prerequisites.", file=sys.stderr)
        return 2
    helper = Path(__file__).with_name("_execute_entry.py")
    if not helper.is_file():
        raise ValueError("Missing companion _execute_entry.py")
    env = os.environ.copy()
    env.update(OFFLINE_ENV)
    command = [sys.executable, str(helper), str(root), args.mode]
    if normalized is not None:
        with tempfile.TemporaryDirectory(prefix="dia_review_config_") as directory:
            cfg_path = Path(directory) / "config.json"
            cfg_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
            return subprocess.call(command + ["--config", str(cfg_path)], env=env)
    return subprocess.call(command + arguments, env=env)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("Preflight error: " + str(exc), file=sys.stderr)
        sys.exit(2)
