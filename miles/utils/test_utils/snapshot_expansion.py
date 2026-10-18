import json
from pathlib import Path

import yaml
from pydantic import JsonValue, TypeAdapter

from miles.utils.audit_utils.config_snapshot.compact_codec import (
    CompactCode,
    CompactConfigSnapshotCase,
    ConfigSnapshotBases,
    DeltaCodec,
)
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotCase
from miles.utils.test_utils.snapshot_text import read_snapshot
from miles.utils.test_utils.snapshot_yaml import expand_snapshot_delta


def expand_snapshot_directory(*, directory: Path) -> dict[str, JsonValue]:
    assert directory.is_dir(), f"Snapshot directory does not exist: {directory}"
    cases = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    cases -= {"README.md", "runtime_config/base.yaml"}
    assert cases, f"No snapshot cases: {directory}"
    result: dict[str, JsonValue] = {}
    for name in sorted(cases):
        path = directory / name
        key = name.removesuffix(".delta.yaml") + ".yaml" if name.endswith(".delta.yaml") else name
        assert key not in result, f"Duplicate snapshot case: {key}"
        text = _read_expanded_snapshot(path=path, ancestors=frozenset())
        if name.startswith("runtime_config/"):
            result[key] = _expand_runtime(text=text, directory=directory)
        elif name.startswith("charts/") and name.endswith(".yaml"):
            result[key] = {"documents": list(yaml.safe_load_all(text))}
        elif name.endswith(".yaml"):
            result[key] = yaml.safe_load(text)
        else:
            assert path.suffix == ".txt", f"Unsupported snapshot format: {name}"
            result[key] = text
    validated = TypeAdapter(dict[str, JsonValue]).validate_python(result, strict=True)
    return json.loads(json.dumps(validated))


def _read_expanded_snapshot(*, path: Path, ancestors: frozenset[Path]) -> str:
    assert path not in ancestors, f"Cyclic snapshot base: {path}"
    text = read_snapshot(path)
    if not path.name.endswith(".delta.yaml"):
        return text

    base_name = yaml.safe_load(text)["base"]
    assert isinstance(base_name, str) and Path(base_name).name == base_name, "Expected a sibling base name"
    candidates = [path.with_name(base_name + suffix) for suffix in (".yaml", ".delta.yaml")]
    bases = [candidate for candidate in candidates if candidate.is_file()]
    assert len(bases) == 1, f"Expected exactly one snapshot base for {path}: {bases}"
    base = _read_expanded_snapshot(path=bases[0], ancestors=ancestors | {path})
    return expand_snapshot_delta(base=base, delta=text)


def _expand_runtime(*, text: str, directory: Path) -> JsonValue:
    value = yaml.safe_load(text)
    compact = "shared_overrides" in value or any("template" in process for process in value["processes"].values())
    if compact:
        bases = ConfigSnapshotBases.model_validate(
            yaml.safe_load((directory / "runtime_config/base.yaml").read_text()), strict=True
        )
        case = CompactCode.decode(case=CompactConfigSnapshotCase.model_validate(value, strict=True), bases=bases)
    else:
        case = ConfigSnapshotCase.model_validate(value, strict=True)
    assert case.processes, "Runtime snapshots must contain processes"
    result: dict[str, JsonValue] = {}
    for name, process in case.processes.items():
        assert "process_config-0000" not in process.diffs, "Base stage occurs twice"
        result[name] = {
            "ranks": process.ranks,
            "stages": {
                "process_config-0000": process.base,
                **{
                    stage: DeltaCodec.decode(base=process.base, overrides=delta)
                    for stage, delta in process.diffs.items()
                },
            },
        }
    return result
