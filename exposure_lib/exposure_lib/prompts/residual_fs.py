from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib # type: ignore
except ModuleNotFoundError:  # Python <3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "residual_fs requires Python 3.11+ or the `tomli` package."
        ) from exc


_TOML_DIR = Path(__file__).with_name("residual_fs_toml")
_PROMPT_KEYS = ("reference_label", "occupation", "task", "examples_strs")
_FORMAT_VERSION = 1


def _read_required_str(
    sample: dict[str, Any],
    key: str,
    source_path: Path,
    sample_idx: int,
) -> str:
    value = sample.get(key)
    if not isinstance(value, str):
        raise ValueError(
            f"{source_path.name}: sample[{sample_idx}] missing string key {key!r}"
        )
    return value


def _load_samples(source_path: Path) -> list[tuple[dict[str, str], str]]:
    with source_path.open("rb") as f:
        parsed = tomllib.load(f)

    if parsed.get("format_version") != _FORMAT_VERSION:
        raise ValueError(f"{source_path.name}: unsupported or missing format_version")

    raw_samples = parsed.get("sample")
    if not isinstance(raw_samples, list):
        raise ValueError(f"{source_path.name}: expected [[sample]] entries")

    samples: list[tuple[dict[str, str], str]] = []
    allowed_keys = set(_PROMPT_KEYS) | {"response"}

    for idx, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, dict):
            raise ValueError(f"{source_path.name}: sample[{idx}] must be a table")

        unexpected_keys = sorted(set(raw_sample) - allowed_keys)
        if unexpected_keys:
            raise ValueError(
                f"{source_path.name}: sample[{idx}] has unexpected keys {unexpected_keys}"
            )

        prompt_meta = {
            key: _read_required_str(raw_sample, key, source_path, idx)
            for key in _PROMPT_KEYS
        }
        response = _read_required_str(raw_sample, "response", source_path, idx)
        samples.append((prompt_meta, response))

    return samples


def _discover_toml_files() -> list[Path]:
    toml_paths = sorted(p for p in _TOML_DIR.glob("*.toml") if p.is_file())
    if len(toml_paths) == 0:
        raise ValueError(f"No few-shot TOML files found in {_TOML_DIR}")
    return toml_paths


FS_SAMPLES_BY_TOML_FN = {
    source_path.name: _load_samples(source_path)
    for source_path in _discover_toml_files()
}
