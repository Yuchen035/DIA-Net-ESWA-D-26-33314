#!/usr/bin/env python3
"""Companion to review.py; explicit execution only, with local library priority.

Do not invoke for static review. This imports/executes recovered model code.
"""
import os
from pathlib import Path
import runpy
import sys


def main():
    if len(sys.argv) < 3 or sys.argv[2] not in ("train", "infer"):
        raise SystemExit("Use review.py to build and check the execution plan.")
    root = Path(sys.argv[1]).resolve()
    mode = sys.argv[2]
    relative = "train/train_lora.py" if mode == "train" else "inference/inference_moe_lora.py"
    entry = root / relative
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root / "peft" / "src"), str(root / "diffusers" / "src"),
                   str(entry.parent), str(root)]
    os.chdir(root / "train" if mode == "train" else root)
    sys.argv = [str(entry)] + sys.argv[3:]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
