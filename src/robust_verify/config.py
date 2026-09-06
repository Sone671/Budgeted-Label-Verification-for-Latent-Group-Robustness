from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")

    required = ["project", "data", "features", "training", "noise", "experiment"]
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    output_dir = Path(config["project"]["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent.parent / output_dir).resolve()
    config["project"]["output_dir"] = str(output_dir)

    if "input_dir" in config["project"]:
        input_dir = Path(config["project"]["input_dir"]).expanduser()
        if not input_dir.is_absolute():
            input_dir = (config_path.parent.parent / input_dir).resolve()
        config["project"]["input_dir"] = str(input_dir)

    data_root = Path(config["data"]["root"]).expanduser()
    if not data_root.is_absolute():
        data_root = (config_path.parent.parent / data_root).resolve()
    config["data"]["root"] = str(data_root)
    config["_config_path"] = str(config_path)

    return config


def output_layout(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "root": root,
        "canonical": root / "canonical",
        "features": root / "features",
        "manifests": root / "manifests",
        "stage1": root / "stage1",
        "probes": root / "stage1" / "probes",
        "dynamics": root / "stage1" / "dynamics",
        "queries": root / "stage1" / "queries",
        "checkpoints": root / "stage1" / "checkpoints",
        "plots": root / "stage1" / "plots",
    }


def ensure_output_layout(output_dir: str | Path) -> dict[str, Path]:
    layout = output_layout(output_dir)
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout
