import shlex
from pathlib import Path

import pytest

from miles.utils.external_utils.command_utils import base_backend
from miles.utils.external_utils.command_utils.base_backend import _add_config_snapshot_name


class TestAutomaticSnapshotNames:
    @pytest.mark.parametrize(
        "arguments",
        [
            "--regular value",
            "--ci-test --ci-disable-config-snapshot",
            "--ci-test --config-snapshot-name explicit",
            "--ci-test --config-snapshot-name=explicit",
        ],
    )
    def test_non_ci_disabled_and_explicit_names_are_preserved(
        self, arguments: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Automatic naming cannot override an explicit identity or enable disabled capture."""
        monkeypatch.setattr(base_backend, "_CONFIG_SNAPSHOT_LAUNCH_COUNTS", {})

        assert _add_config_snapshot_name(arguments) == arguments
        assert base_backend._CONFIG_SNAPSHOT_LAUNCH_COUNTS == {}

    def test_launch_indices_are_scoped_to_the_invoking_script(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each script's launches keep stable identities independent of other scripts' launches."""
        monkeypatch.setattr(base_backend, "_CONFIG_SNAPSHOT_LAUNCH_COUNTS", {})
        root = Path(base_backend.repo_base_dir)
        monkeypatch.setattr("sys.argv", [str(root / "tests/e2e/first.py")])
        first = shlex.split(_add_config_snapshot_name("--ci-test"))
        second = shlex.split(_add_config_snapshot_name("--ci-test"))
        monkeypatch.setattr("sys.argv", [str(root / "tests/e2e/second.py")])
        other = shlex.split(_add_config_snapshot_name("--ci-test"))

        assert first[-2:] == ["--config-snapshot-name", "tests/e2e/first/run-0000"]
        assert second[-2:] == ["--config-snapshot-name", "tests/e2e/first/run-0001"]
        assert other[-2:] == ["--config-snapshot-name", "tests/e2e/second/run-0000"]
