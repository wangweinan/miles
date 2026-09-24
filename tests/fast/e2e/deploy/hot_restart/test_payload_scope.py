from tests.fast.e2e.deploy.hot_restart.cluster_facts import RELEASE
from tests.fast.e2e.deploy.hot_restart.cluster_facts import ROLLOUT_EXECUTOR, TRAINER, cluster_snapshot, pod_fact, workload_fact
from tests.utils.deploy.hot_restart.cluster_observer import SnapshotRecorder
from tests.utils.deploy.hot_restart.evidence import HotRestartEvidence
from miles.utils.workers.connection_config import StaticConnConfig
from miles.utils.workers.serving.worker_config import ServeWorkerConfig


def _evidence() -> HotRestartEvidence:
    snapshots = tuple((cluster_snapshot(pods=[pod_fact(f'{ROLLOUT_EXECUTOR}-0', uid=uid), pod_fact(f'{TRAINER}-0', uid='t')], workloads=[workload_fact(ROLLOUT_EXECUTOR), workload_fact(TRAINER)]) for uid in ('r0', 'r0', 'r1')))
    commands = {'t': ('python', 'trainer')}
    for index in range(2):
        payload = ServeWorkerConfig(worker_type='rollout_executor', args={'save_debug_rollout_data': f'/gen{index}'}, static_connections=StaticConnConfig())
        commands[f'r{index}'] = ('python', '--config', payload.model_dump_json())
    return HotRestartEvidence(records=(), snapshots=snapshots, release=RELEASE, commands_of_pod_uid=commands)


class TestObservedCommands:

    def test_replacement_keeps_the_old_command_without_duplicating_it_in_each_snapshot(self) -> None:
        """Command evidence survives replacement and is stored once for every observed pod UID."""
        evidence = _evidence()
        recorder = SnapshotRecorder(release=RELEASE)
        for snapshot in evidence.snapshots:
            recorder.record(snapshot.model_copy(update={'commands_of_pod_uid': {p.uid: evidence.commands_of_pod_uid[p.uid] for p in snapshot.pods}}))
        assert recorder.commands_of_pod_uid == evidence.commands_of_pod_uid
        assert recorder.snapshots
        assert all((snapshot.commands_of_pod_uid == {} for snapshot in recorder.snapshots))
