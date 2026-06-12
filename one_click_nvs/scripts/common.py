from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = "config.json") -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def relpath(cfg: dict[str, Any], *keys: str) -> Path:
    value: Any = cfg
    for key in keys:
        value = value[key]
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_command(cmd: list[str], dry_run: bool = False, cwd: str | Path | None = None) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True)


def require_command(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Required command not found on PATH: {name}")
    return found


def newest_file(root: Path, pattern: str, min_mtime: float = 0.0) -> Path | None:
    candidates = [p for p in root.rglob(pattern) if p.stat().st_mtime >= min_mtime]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def hamming_bits(a: str, b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def print_section(title: str) -> None:
    print("")
    print(f"==== {title} ====")


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def flatten_numeric(prefix: str, value: Any) -> Iterable[tuple[str, float]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_numeric(child_prefix, child)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from flatten_numeric(f"{prefix}[{i}]", child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)
