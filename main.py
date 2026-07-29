import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from registry import pipeline_builder_register
import registry.process_builder  # noqa: F401


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Blastocyst-String pipeline.")
    parser.add_argument("--pipeline_config", required=True, help="Committed pipeline YAML.")
    parser.add_argument("--runtime_config", help="Local runtime YAML with paths and overrides.")
    parser.add_argument("--runtime_key", help="Top-level key to select from runtime YAML.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Runtime override. May be repeated, for example: --set max_cases=5.",
    )
    return parser.parse_args()


def build_runtime_data(args: argparse.Namespace) -> dict[str, Any]:
    runtime_data: dict[str, Any] = {}
    if args.runtime_config:
        runtime_config = load_yaml(args.runtime_config)
        if args.runtime_key:
            if args.runtime_key not in runtime_config:
                raise KeyError(
                    f"runtime_key {args.runtime_key!r} not found in {args.runtime_config}"
                )
            selected = runtime_config[args.runtime_key]
            if not isinstance(selected, dict):
                raise TypeError(f"runtime_key {args.runtime_key!r} must point to a mapping.")
            if isinstance(runtime_config.get("paths"), dict):
                runtime_data["paths"] = runtime_config["paths"]
            runtime_data.update(selected)
        else:
            runtime_data.update(runtime_config)

    runtime_data.update(parse_set_overrides(args.set))
    return runtime_data


def resolve_path_refs(config: dict[str, Any], paths: dict[str, Any]) -> dict[str, Any]:
    return _resolve_path_refs_value(config, paths)


def _resolve_path_refs_value(value: Any, paths: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        resolved = {
            key: _resolve_path_refs_value(item, paths)
            for key, item in value.items()
            if not key.endswith("_ref")
        }
        for key, item in value.items():
            if key.endswith("_ref") and item:
                resolved[_path_ref_target_key(key)] = _resolve_path_ref(paths, str(item))
        return resolved
    if isinstance(value, list):
        return [_resolve_path_refs_value(item, paths) for item in value]
    return value


def _path_ref_target_key(ref_key: str) -> str:
    aliases = {
        "eval_ref": "output_dir",
        "feature_matrix_ref": "feature_matrix_dir",
        "feature_output_ref": "feature_output_dir",
        "metadata_ref": "metadata_csv",
    }
    return aliases.get(ref_key, ref_key.removesuffix("_ref"))


def _resolve_path_ref(paths: dict[str, Any], ref: str) -> Any:
    if not ref:
        raise ValueError("Path ref must be a non-empty string.")
    if ref in paths and not isinstance(paths[ref], Mapping):
        return paths[ref]

    matches = list(_find_ref_matches(paths, ref))
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        locations = ", ".join(location for location, _ in matches)
        raise KeyError(f"Path ref {ref!r} is ambiguous; found {locations}.")
    raise KeyError(f"Missing path ref {ref!r} in runtime paths.")


def _find_ref_matches(payload: Mapping[str, Any], ref: str, prefix: str = "paths") -> list[tuple[str, Any]]:
    matches: list[tuple[str, Any]] = []
    for key, value in payload.items():
        location = f"{prefix}.{key}"
        if key == ref and not isinstance(value, Mapping):
            matches.append((location, value))
        elif isinstance(value, Mapping):
            matches.extend(_find_ref_matches(value, ref, location))
    return matches


def parse_set_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set override must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set override has an empty key: {item}")
        overrides[key] = parse_scalar(value.strip())
    return overrides


def apply_pipeline_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if "." not in key:
            continue
        parts = key.split(".")
        applied = False
        for handler_cfg in cfg.get("pipeline", {}).values():
            if not isinstance(handler_cfg, dict):
                continue
            handler_config = handler_cfg.get("config", handler_cfg)
            if isinstance(handler_config, dict) and parts[0] in handler_config:
                set_nested_value(handler_config, parts, value)
                applied = True
        if not applied:
            set_nested_value(cfg, parts, value)
    return cfg


def set_nested_value(payload: dict[str, Any], parts: list[str], value: Any) -> None:
    current = payload
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise TypeError(f"Cannot apply override through non-mapping key: {'.'.join(parts)}")
        current = next_value
    current[parts[-1]] = value


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def main() -> None:
    args = parse_args()
    pipeline_config = load_yaml(args.pipeline_config)
    runtime_data = build_runtime_data(args)

    builder_name = pipeline_config["builder"]
    builder_cls = pipeline_builder_register.get(builder_name)
    runtime_paths = runtime_data.get("paths") or runtime_data.get("pipeline") or {}
    cfg = resolve_path_refs(pipeline_config.get("cfg", {}), runtime_paths)
    cfg = apply_pipeline_overrides(cfg, parse_set_overrides(args.set))
    pipeline = builder_cls().build(cfg)

    output_data = pipeline.process(runtime_data)
    print_summary(output_data)


def print_summary(output_data: dict[str, Any]) -> None:
    if output_data.get("pipeline_error"):
        print(f"Pipeline error: {output_data['pipeline_error']}")

    summary = output_data.get("summary", {})
    if not summary:
        print("Pipeline finished with no summary.")
        return

    print("Pipeline summary:")
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value)
        print(f"  {key}: {rendered}")


if __name__ == "__main__":
    main()
