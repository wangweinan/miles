from pathlib import Path
import pytest
from miles.utils.audit_utils.config_snapshot.runner import ConfigSnapshotTestRunner
from miles.utils.test_utils.snapshot import SNAPSHOT_DIRECTORY, SNAPSHOT_RECORD_DIR_ENV_VAR, SNAPSHOT_UPDATE_ENV_VAR


class TestSnapshotAttemptLifecycle:

    def test_capture_requires_an_explicit_record_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Capture cannot silently write raw records into an implicit working directory."""
        monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR, raising=False)
        with pytest.raises(ValueError, match='MILES_SNAPSHOT_RECORD_DIR'):
            ConfigSnapshotTestRunner.create(test='tests/e2e/test_example.py', repo_root=tmp_path)

    def test_repeated_attempts_get_distinct_directories_and_start_incomplete(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retries of one test cannot overwrite each other's raw evidence."""
        import json
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / 'records'))
        first = ConfigSnapshotTestRunner.create(test='tests/e2e/test_example.py', repo_root=tmp_path)
        second = ConfigSnapshotTestRunner.create(test='tests/e2e/test_example.py', repo_root=tmp_path)
        assert first.directory != second.directory
        for runner in (first, second):
            assert runner.record_directory.is_dir()
            assert json.loads((runner.directory / 'attempt.json').read_text())['completed'] is False

    def test_failed_execution_preserves_records_without_comparing_or_updating_a_golden(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed child retains its raw evidence without interpreting partial snapshots."""
        import json
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / 'records'))
        monkeypatch.setenv(SNAPSHOT_UPDATE_ENV_VAR, '1')
        runner = ConfigSnapshotTestRunner.create(test='tests/e2e/test_example.py', repo_root=tmp_path)
        invalid = runner.record_directory / 'partial.json'
        invalid.write_text('incomplete json')
        runner.finish(returncode=7)
        assert invalid.read_text() == 'incomplete json'
        assert not runner.golden.exists()
        assert json.loads((runner.directory / 'attempt.json').read_text())['completed'] is False

    def test_success_without_records_completes_without_creating_an_empty_golden(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tests that never launch training remain valid without synthetic empty snapshots."""
        import json
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / 'records'))
        runner = ConfigSnapshotTestRunner.create(test='tests/e2e/test_example.py', repo_root=tmp_path)
        runner.finish(returncode=0)
        assert not runner.golden.exists()
        assert json.loads((runner.directory / 'attempt.json').read_text())['completed'] is True

    @pytest.mark.parametrize('test', ['../outside.py', 'tests/../outside.py'])
    def test_parent_traversal_is_rejected(self, tmp_path: Path, test: str) -> None:
        """A test identifier cannot escape the repository's snapshot tree."""
        with pytest.raises(ValueError, match='within the repository'):
            ConfigSnapshotTestRunner.golden_path(test=test, repo_root=tmp_path)

    def test_symlink_escape_is_rejected(self, tmp_path: Path) -> None:
        """A relative test path cannot redirect its golden through a symlink outside the snapshot tree."""
        root = tmp_path / 'repo'
        directory = root / SNAPSHOT_DIRECTORY / 'runtime_config'
        directory.mkdir(parents=True)
        outside = tmp_path / 'outside'
        outside.mkdir()
        (directory / 'escaped').symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match='within the snapshot directory'):
            ConfigSnapshotTestRunner.golden_path(test='escaped/test.py', repo_root=root)
