from collections.abc import Callable
from pathlib import Path
import pytest
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotTestAttempt
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.config_snapshot.sync import _collect_snapshots


def _write_attempt(*, directory: Path, test: str) -> None:
    directory.mkdir(parents=True)
    (directory / 'attempt.json').write_text(ConfigSnapshotTestAttempt(test=test, completed=True).model_dump_json())


class TestCompletedAttemptSelection:

    def test_incomplete_records_without_a_successful_attempt_block_sync(self, tmp_path: Path, make_record: Callable) -> None:
        """Partial captures cannot be promoted to golden snapshots after a failed run."""
        attempt = tmp_path / 'raw' / 'attempt'
        _write_attempt(directory=attempt, test='tests/e2e/test_training.py')
        (attempt / 'attempt.json').write_text(ConfigSnapshotTestAttempt(test='tests/e2e/test_training.py', completed=False).model_dump_json())
        ConfigSnapshotStorage(directory=attempt / 'records').write(make_record())
        with pytest.raises(ValueError, match='No completed attempt'):
            _collect_snapshots(directory=tmp_path / 'raw', repo_root=tmp_path)

    def test_failed_attempts_do_not_override_successful_retry_contents(self, tmp_path: Path, make_record: Callable) -> None:
        """A successful retry replaces failed evidence without requiring its partial values to agree."""
        root = tmp_path / 'raw'
        failed, succeeded = (root / 'failed', root / 'succeeded')
        for directory in (failed, succeeded):
            _write_attempt(directory=directory, test='tests/e2e/test_training.py')
        (failed / 'attempt.json').write_text(ConfigSnapshotTestAttempt(test='tests/e2e/test_training.py', completed=False).model_dump_json())
        ConfigSnapshotStorage(directory=failed / 'records').write(make_record(config={'value': 'partial'}))
        ConfigSnapshotStorage(directory=succeeded / 'records').write(make_record(config={'value': 'complete'}))
        snapshots = _collect_snapshots(directory=root, repo_root=tmp_path)
        assert len(snapshots) == 1
        assert 'complete' in next(iter(snapshots.values()))
        assert 'partial' not in next(iter(snapshots.values()))

    def test_an_archive_without_completed_records_is_rejected(self, tmp_path: Path) -> None:
        """An empty artifact download cannot report a successful golden refresh."""
        with pytest.raises(ValueError, match='No completed snapshot records'):
            _collect_snapshots(directory=tmp_path, repo_root=tmp_path)
