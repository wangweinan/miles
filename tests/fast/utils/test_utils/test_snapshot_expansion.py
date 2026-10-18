import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from miles.utils.test_utils.snapshot_expansion import expand_snapshot_directory
from miles.utils.test_utils.snapshot_yaml import dump_snapshot_delta


class TestSnapshotExpansion:
    def test_compact_runtime_expands_shared_overrides_and_independent_stages(self, tmp_path: Path) -> None:
        """Every stage starts from the process base and preserves deletions and empty containers."""
        directory = tmp_path / "runtime_config"
        directory.mkdir()
        (directory / "base.yaml").write_text(
            json.dumps({"templates": {"trainer": {"args": {"value": 0, "removed": 9, "empty": {}}}}})
        )
        (directory / "case.yaml").write_text(
            json.dumps(
                {
                    "shared_overrides": {"trainer": {"set": {"/args/value": 1}}},
                    "processes": {
                        "actor": {
                            "ranks": [0, 1],
                            "template": "trainer",
                            "overrides": {"set": {"/args/empty": []}},
                            "diffs": {
                                "checkpoint_load-0000": {"set": {"/args/value": 2}, "remove": ["/args/removed"]},
                                "checkpoint_load-0001": {"set": {"/args/value": 3}},
                            },
                        }
                    },
                }
            )
        )

        assert expand_snapshot_directory(directory=tmp_path) == {
            "runtime_config/case.yaml": {
                "actor": {
                    "ranks": [0, 1],
                    "stages": {
                        "process_config-0000": {"args": {"value": 1, "removed": 9, "empty": []}},
                        "checkpoint_load-0000": {"args": {"value": 2, "empty": []}},
                        "checkpoint_load-0001": {"args": {"value": 3, "removed": 9, "empty": []}},
                    },
                }
            }
        }

    def test_chart_documents_preserve_boundaries_order_and_empty_documents(self, tmp_path: Path) -> None:
        """A chart YAML stream remains distinct from one document containing a list."""
        charts = tmp_path / "charts"
        charts.mkdir()
        (charts / "stream.yaml").write_text("kind: Pod\n---\nkind: Service\n---\n")
        (charts / "list.yaml").write_text("- {kind: Pod}\n- {kind: Service}\n- null\n")

        assert expand_snapshot_directory(directory=tmp_path) == {
            "charts/stream.yaml": {"documents": [{"kind": "Pod"}, {"kind": "Service"}, None]},
            "charts/list.yaml": {"documents": [[{"kind": "Pod"}, {"kind": "Service"}, None]]},
        }

    def test_non_string_keys_cannot_collapse_during_alias_expansion(self, tmp_path: Path) -> None:
        """Invalid YAML keys cannot overwrite distinct string keys during JSON conversion."""
        (tmp_path / "case.yaml").write_text("1: first\n'1': second\n")

        with pytest.raises(ValidationError):
            expand_snapshot_directory(directory=tmp_path)

    @pytest.mark.parametrize("rank", [False, 0.0, "0"])
    @pytest.mark.parametrize("compact", [False, True])
    def test_runtime_rank_types_are_not_coerced(self, tmp_path: Path, rank: bool | float | str, compact: bool) -> None:
        """Malformed rank types cannot disappear through schema coercion."""
        directory = tmp_path / "runtime_config"
        directory.mkdir()
        (directory / "base.yaml").write_text('{"templates": {"trainer": {"args": {}}}}')
        process = (
            {"ranks": [rank], "template": "trainer"}
            if compact
            else {"ranks": [rank], "base": {"args": {}}, "diffs": {}}
        )
        (directory / "case.yaml").write_text(json.dumps({"processes": {"trainer": process}}))

        with pytest.raises(ValidationError, match="valid integer"):
            expand_snapshot_directory(directory=tmp_path)

    def test_delta_uses_its_own_directory_base_and_canonical_case_name(self, tmp_path: Path) -> None:
        """A compressed case expands using the adjacent verified base."""
        base = "value: 1\n"
        (tmp_path / "base.yaml").write_text(base)
        (tmp_path / "case.delta.yaml").write_text(
            dump_snapshot_delta(base=base, actual="value: 2\n", base_name="base")
        )

        assert expand_snapshot_directory(directory=tmp_path) == {
            "base.yaml": {"value": 1},
            "case.yaml": {"value": 2},
        }

        (tmp_path / "base.yaml").write_text("value: 3\n")
        with pytest.raises(AssertionError, match="base has changed"):
            expand_snapshot_directory(directory=tmp_path)

    def test_nested_delta_expands_and_validates_each_base(self, tmp_path: Path) -> None:
        """A delta can use another delta while retaining both base checksum checks."""
        (tmp_path / "base.yaml").write_text("value: 1\n")
        middle = tmp_path / "middle.delta.yaml"
        middle.write_text(dump_snapshot_delta(base="value: 1\n", actual="value: 2\n", base_name="base"))
        (tmp_path / "case.delta.yaml").write_text(
            dump_snapshot_delta(base="value: 2\n", actual="value: 3\n", base_name="middle")
        )

        assert expand_snapshot_directory(directory=tmp_path) == {
            "base.yaml": {"value": 1},
            "middle.yaml": {"value": 2},
            "case.yaml": {"value": 3},
        }

        middle.write_text(dump_snapshot_delta(base="value: 1\n", actual="value: 4\n", base_name="base"))
        with pytest.raises(AssertionError, match="base has changed"):
            expand_snapshot_directory(directory=tmp_path)

    def test_cyclic_delta_bases_are_rejected(self, tmp_path: Path) -> None:
        """A cyclic base reference fails explicitly instead of exhausting recursion."""
        for name, base in (("first", "second"), ("second", "first")):
            (tmp_path / f"{name}.delta.yaml").write_text(
                dump_snapshot_delta(base="value: 1\n", actual="value: 1\n", base_name=base)
            )

        with pytest.raises(AssertionError, match="Cyclic snapshot base"):
            expand_snapshot_directory(directory=tmp_path)

    @pytest.mark.parametrize("ambiguous", [False, True])
    def test_missing_or_ambiguous_delta_base_is_rejected(self, tmp_path: Path, ambiguous: bool) -> None:
        """Base resolution cannot silently choose between representations or omit a missing base."""
        (tmp_path / "a_case.delta.yaml").write_text(
            dump_snapshot_delta(base="value: 1\n", actual="value: 2\n", base_name="base")
        )
        if ambiguous:
            (tmp_path / "base.yaml").write_text("value: 1\n")
            (tmp_path / "base.delta.yaml").write_text("unused")

        with pytest.raises(AssertionError, match="Expected exactly one snapshot base"):
            expand_snapshot_directory(directory=tmp_path)

    def test_yaml_aliases_are_independent_after_expansion(self, tmp_path: Path) -> None:
        """Moving one field cannot mutate another case through a YAML alias."""
        (tmp_path / "case.yaml").write_text("first: &shared {value: 1}\nsecond: *shared\n")

        result = expand_snapshot_directory(directory=tmp_path)
        case = result["case.yaml"]
        assert isinstance(case, dict) and isinstance(case["first"], dict)
        case["first"].pop("value")

        assert case["second"] == {"value": 1}

    def test_duplicate_full_and_delta_cases_are_rejected(self, tmp_path: Path) -> None:
        """Two representations cannot silently overwrite the same comparison case."""
        (tmp_path / "base.yaml").write_text("a: 1\n")
        (tmp_path / "case.yaml").write_text("a: 2\n")
        (tmp_path / "case.delta.yaml").write_text(
            dump_snapshot_delta(base="a: 1\n", actual="a: 2\n", base_name="base")
        )

        with pytest.raises(AssertionError, match="Duplicate snapshot case"):
            expand_snapshot_directory(directory=tmp_path)

    def test_empty_directory_is_rejected(self, tmp_path: Path) -> None:
        """An absent capture cannot pass as an empty successful comparison."""
        with pytest.raises(AssertionError, match="No snapshot cases"):
            expand_snapshot_directory(directory=tmp_path)

    def test_unknown_files_are_not_silently_skipped(self, tmp_path: Path) -> None:
        """Unsupported artifacts force an explicit decision about their comparison."""
        (tmp_path / "case.json").write_text("{}")

        with pytest.raises(AssertionError, match="Unsupported snapshot format"):
            expand_snapshot_directory(directory=tmp_path)
