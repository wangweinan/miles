import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.snapshot_comparison_models import (
    SnapshotComparisonReport,
    SnapshotComparisonRules,
    SnapshotDifference,
    SnapshotListSort,
    SnapshotValue,
)
from miles.utils.test_utils.snapshot_expansion import expand_snapshot_directory

app = typer.Typer(add_completion=False)


@app.command()
def compare(
    before: Annotated[Path, typer.Option()],
    after: Annotated[Path, typer.Option()],
    rules: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    assert not output.exists(), f"Report already exists: {output}"
    specification = SnapshotComparisonRules.model_validate_json(rules.read_text(), strict=True)
    expected = expand_snapshot_directory(directory=before)
    actual = expand_snapshot_directory(directory=after)
    for move in specification.moves:
        assert move.op and move.reason, "Each move needs its source op and reason"
        assert move.source != move.target and move.target[: len(move.source)] != move.source
        source, source_key = _parent(value=expected, path=move.source)
        assert source_key in source, f"Missing move source: {move.source}"
        target, target_key = _parent(value=expected, path=move.target, create=True)
        assert target_key not in target, f"Move destination exists: {move.target}"
        target[target_key] = source.pop(source_key)

    sorting_differences = _sort_lora_targets(value=expected, rules=specification.list_sorts)
    differences = {_difference_key(item): item for item in _differences(before=expected, after=actual, path=[])}
    approvals = specification.approved_differences
    assert len({tuple(item.path) for item in approvals}) == len(approvals), "Duplicate approvals"
    assert all(item.human_decision and item.op for item in approvals), "Each approval needs a human decision and op"
    approved = {_difference_key(item): item for item in approvals}
    report = SnapshotComparisonReport(
        before_directory=str(before.resolve()),
        after_directory=str(after.resolve()),
        moves=specification.moves,
        approved_differences=[item for key, item in approved.items() if key in differences],
        unexplained=[item for key, item in differences.items() if key not in approved],
        unused_approvals=[item for key, item in approved.items() if key not in differences],
        list_sorts=specification.list_sorts,
        sorting_differences=sorting_differences,
    )
    atomic_write_text(path=output, text=report.model_dump_json(indent=2))
    if report.unexplained or report.unused_approvals:
        raise typer.Exit(code=1)


def _sort_lora_targets(*, value: dict[str, JsonValue], rules: list[SnapshotListSort]) -> list[SnapshotDifference]:
    assert len({tuple(rule.path) for rule in rules}) == len(rules), "Duplicate list sorts"
    differences: list[SnapshotDifference] = []
    for rule in rules:
        assert rule.op == "op48-15" and rule.human_decision, "List sorting needs the approved op and human decision"
        assert rule.path[:-1] == ["parsed_config/megatron_lora_targets.yaml", "config"]
        assert rule.path[-1] in {"hf_lora_targets", "lora_adapter_targets"}
        parent, key = _parent(value=value, path=rule.path)
        original = parent[key]
        assert isinstance(original, list) and all(isinstance(item, str) for item in original)
        ordered = sorted(original)
        differences.extend(_differences(before=original, after=ordered, path=rule.path))
        parent[key] = ordered
    return differences


def _differences(*, before: JsonValue, after: JsonValue, path: list[str]) -> list[SnapshotDifference]:
    if isinstance(before, dict) and isinstance(after, dict):
        result: list[SnapshotDifference] = []
        for key in sorted(before.keys() | after.keys()):
            child = [*path, key]
            if key in before and key in after:
                result.extend(_differences(before=before[key], after=after[key], path=child))
            else:
                result.append(
                    SnapshotDifference(
                        path=child,
                        before=SnapshotValue(present=key in before, value=before.get(key)),
                        after=SnapshotValue(present=key in after, value=after.get(key)),
                    )
                )
        return result
    if json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True):
        return []
    return [
        SnapshotDifference(
            path=path,
            before=SnapshotValue(present=True, value=before),
            after=SnapshotValue(present=True, value=after),
        )
    ]


def _difference_key(value: SnapshotDifference) -> str:
    return json.dumps(value.model_dump(include={"path", "before", "after"}), sort_keys=True)


def _parent(*, value: dict[str, JsonValue], path: list[str], create: bool = False) -> tuple[dict[str, JsonValue], str]:
    assert path, "A field move must have a nonempty path"
    for key in path[:-1]:
        if create and key not in value:
            value[key] = {}
        child = value[key]
        assert isinstance(child, dict), f"Field move traverses a non-mapping: {path}"
        value = child
    return value, path[-1]


if __name__ == "__main__":
    app()
