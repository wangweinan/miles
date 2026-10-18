import json
from pathlib import Path

import pytest
from pydantic import JsonValue
from typer.testing import CliRunner

from miles.utils.test_utils.snapshot_comparison import _differences, app
from miles.utils.test_utils.snapshot_refactor_rules import app as rules_app


class TestSnapshotDifferences:
    @pytest.mark.parametrize(
        ("before", "after"),
        [(False, 0), (1, 1.0), ({}, {"value": None}), ([1, 2], [2, 1])],
    )
    def test_type_presence_and_order_changes_are_reported(self, before: JsonValue, after: JsonValue) -> None:
        """Comparison preserves scalar types, missing fields and list order."""
        differences = _differences(before=before, after=after, path=[])

        assert len(differences) == 1
        if before == {}:
            assert differences[0].before.present is False
            assert differences[0].after.present is True

    def test_mapping_order_does_not_create_a_difference(self) -> None:
        """Only mapping contents affect comparison."""
        assert _differences(before={"a": 1, "b": 2}, after={"b": 2, "a": 1}, path=[]) == []


class TestSnapshotComparison:
    @pytest.mark.parametrize(
        ("after_targets", "other", "passes"),
        [
            (["a", "b", "b"], [2, 1], True),
            (["a", "b"], [2, 1], False),
            (["b", "a", "b"], [2, 1], False),
            (["a", "b", "b"], [1, 2], False),
        ],
    )
    def test_explicit_lora_sort_preserves_duplicates_and_other_list_order(
        self, tmp_path: Path, after_targets: list[str], other: list[int], passes: bool
    ) -> None:
        """Only the approved before field is sorted and the report retains its original order."""
        before, after = tmp_path / "before", tmp_path / "after"
        for directory, targets, unrelated in ((before, ["b", "a", "b"], [2, 1]), (after, after_targets, other)):
            (directory / "parsed_config").mkdir(parents=True)
            (directory / "parsed_config/megatron_lora_targets.yaml").write_text(
                json.dumps({"config": {"hf_lora_targets": targets, "other": unrelated}})
            )
        rule = {
            "path": ["parsed_config/megatron_lora_targets.yaml", "config", "hf_lora_targets"],
            "op": "op48-15",
            "human_decision": "Explicitly sort this list",
        }
        rules, output = tmp_path / "rules.json", tmp_path / "report.json"
        rules.write_text(json.dumps({"moves": [], "approved_differences": [], "list_sorts": [rule]}))

        result = CliRunner().invoke(
            app, ["--before", str(before), "--after", str(after), "--rules", str(rules), "--output", str(output)]
        )

        assert result.exit_code == (0 if passes else 1), result.exception
        report = json.loads(output.read_text())
        assert report["list_sorts"] == [rule]
        assert report["sorting_differences"][0]["before"]["value"] == ["b", "a", "b"]
        assert report["sorting_differences"][0]["after"]["value"] == ["a", "b", "b"]

    @pytest.mark.parametrize("approval", ["none", "exact", "wrong_type", "unused"])
    def test_only_exact_used_approvals_allow_changed_values(self, tmp_path: Path, approval: str) -> None:
        """Approvals must match both sides and cannot conceal extra changes."""
        before = tmp_path / "before"
        after = tmp_path / "after"
        before.mkdir()
        after.mkdir()
        (before / "case.yaml").write_text("enabled: false\n")
        (after / "case.yaml").write_text("enabled: true\n")
        approvals = (
            []
            if approval == "none"
            else [
                {
                    "path": ["case.yaml", "other" if approval == "unused" else "enabled"],
                    "before": {"present": True, "value": 0 if approval == "wrong_type" else False},
                    "after": {"present": True, "value": True},
                    "human_decision": "Explicit test decision",
                    "op": "op-test",
                }
            ]
        )
        rules = tmp_path / "rules.json"
        rules.write_text(json.dumps({"moves": [], "approved_differences": approvals}))
        output = tmp_path / "report.json"

        result = CliRunner().invoke(
            app, ["--before", str(before), "--after", str(after), "--rules", str(rules), "--output", str(output)]
        )

        assert result.exit_code == (0 if approval == "exact" else 1), result.exception
        report = json.loads(output.read_text())
        assert len(report["approved_differences"]) == (1 if approval == "exact" else 0)
        assert len(report["unexplained"]) == (0 if approval == "exact" else 1)
        assert len(report["unused_approvals"]) == (1 if approval in {"wrong_type", "unused"} else 0)

    @pytest.mark.parametrize("invalid", ["none", "missing_source", "occupied_target", "descendant"])
    def test_moves_preserve_values_and_reject_invalid_destinations(self, tmp_path: Path, invalid: str) -> None:
        """A move cannot invent a source or overwrite an existing field."""
        before = tmp_path / "before"
        after = tmp_path / "after"
        before.mkdir()
        after.mkdir()
        (before / "case.yaml").write_text("old: 7\n" + ("backend: {new: 8}\n" if invalid == "occupied_target" else ""))
        (after / "case.yaml").write_text("backend: {new: 7}\n")
        move = {
            "source": ["case.yaml", "missing" if invalid == "missing_source" else "old"],
            "target": ["case.yaml", "old" if invalid == "descendant" else "backend", "new"],
            "op": "op-test",
            "reason": "Move a field without changing its value",
        }
        rules = tmp_path / "rules.json"
        rules.write_text(json.dumps({"moves": [move], "approved_differences": []}))
        output = tmp_path / "report.json"

        result = CliRunner().invoke(
            app, ["--before", str(before), "--after", str(after), "--rules", str(rules), "--output", str(output)]
        )

        if invalid == "none":
            assert result.exit_code == 0, result.exception
            assert json.loads(output.read_text())["unexplained"] == []
        else:
            assert isinstance(result.exception, AssertionError)
            assert not output.exists()


class TestSnapshotRefactorRules:
    @pytest.mark.parametrize("backend", ["megatron", "fsdp"])
    @pytest.mark.parametrize("changed_type", [False, True])
    def test_generated_moves_preserve_unlisted_fields_and_scalar_types(
        self, tmp_path: Path, backend: str, changed_type: bool
    ) -> None:
        """Audited parsed moves leave schema, copied fields and unlisted prefixes untouched."""
        before, after = tmp_path / "before", tmp_path / "after"
        original = {
            "train_backend": backend,
            "lr": 1e-6,
            "bf16": False,
            "seed": 7,
            "sglang_tp_size": 2,
            "sglang_unlisted": 42,
            "router_policy": "cache_aware",
        }
        backend_values = {"lr": 1e-6, "bf16": 0 if changed_type else False}
        actual = {
            "train_backend": backend,
            "seed": 7,
            "sglang_unlisted": 42,
            "sglang": {"base_args": {"tp_size": 2}},
            "router_args": {"policy": "cache_aware"},
            **(
                {"raw_megatron": {"base_args": backend_values}}
                if backend == "megatron"
                else {"raw_fsdp": backend_values}
            ),
        }
        for directory, config in ((before, original), (after, actual)):
            (directory / "parsed_config").mkdir(parents=True)
            (directory / "argument_schema").mkdir()
            (directory / "parsed_config/case.yaml").write_text(json.dumps({"config": config}))
            (directory / "parsed_config/error.yaml").write_text(json.dumps({"error": "invalid combination"}))
            (directory / "argument_schema/case.yaml").write_text(json.dumps({"config": original}))
        rules, output = tmp_path / "rules.json", tmp_path / "report.json"
        runner = CliRunner()
        generated = runner.invoke(rules_app, ["--before", str(before), "--output", str(rules)])
        assert generated.exit_code == 0, generated.exception

        compared = runner.invoke(
            app, ["--before", str(before), "--after", str(after), "--rules", str(rules), "--output", str(output)]
        )
        assert compared.exit_code == int(changed_type), compared.exception
        report = json.loads(output.read_text())
        assert len(report["unexplained"]) == int(changed_type)
        if changed_type:
            assert report["unexplained"][0]["path"][-1] == "bf16"
            assert report["unexplained"][0]["before"]["value"] is False
            assert type(report["unexplained"][0]["after"]["value"]) is int

    @pytest.mark.parametrize("backend", ["megatron", "fsdp"])
    @pytest.mark.parametrize("changed_type", [False, True])
    def test_runtime_moves_apply_only_to_megatron_trainers_and_preserve_values(
        self, tmp_path: Path, backend: str, changed_type: bool
    ) -> None:
        """Trainer ownership moves cannot hide changed values or move controller inputs."""
        before, after = tmp_path / "before", tmp_path / "after"
        original = {"train_backend": backend, "lr": 1e-6, "mtp_detach_heads": False, "unlisted": 9}
        values = {"lr": 1e-6, "mtp_detach_heads": 0 if changed_type else False}
        actual = {"train_backend": backend, "unlisted": 9, "backend": values}
        for directory, config in ((before, original), (after, actual)):
            (directory / "runtime_config").mkdir(parents=True)
            (directory / "runtime_config/case.yaml").write_text(
                json.dumps(
                    {
                        "processes": {
                            "trainer": {"ranks": [0], "base": {"args": config}, "diffs": {}},
                            "controller": {"ranks": [], "base": {"args": original}, "diffs": {}},
                        }
                    }
                )
            )
        rules, output = tmp_path / "rules.json", tmp_path / "report.json"
        runner = CliRunner()
        generated = runner.invoke(rules_app, ["--before", str(before), "--output", str(rules)])
        assert generated.exit_code == 0, generated.exception

        compared = runner.invoke(
            app, ["--before", str(before), "--after", str(after), "--rules", str(rules), "--output", str(output)]
        )

        assert compared.exit_code == int(backend != "megatron" or changed_type), compared.exception
        report = json.loads(output.read_text())
        assert all(item["path"][1] == "trainer" for item in report["unexplained"])
        if backend == "megatron" and changed_type:
            assert len(report["unexplained"]) == 1
            assert report["unexplained"][0]["path"][-2:] == ["backend", "mtp_detach_heads"]
            assert report["unexplained"][0]["before"]["value"] is False
            assert type(report["unexplained"][0]["after"]["value"]) is int
