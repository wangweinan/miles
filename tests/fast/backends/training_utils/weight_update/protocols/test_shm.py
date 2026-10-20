import json
import subprocess
import sys
from argparse import Namespace
from concurrent.futures import Future

import pytest
import torch
import torch.multiprocessing

from miles.backends.training_utils.weight_update.protocols.shm import UpdateWeightFromSharedMemory

_BUCKETS = [
    [
        ("layers.0.w", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
        ("norm.w", torch.ones(3, dtype=torch.bfloat16)),
    ],
    [("layers.1.w", torch.full((4,), 7.0))],
    [("lm_head.w", torch.full((2, 2), -1.0, dtype=torch.bfloat16))],
]

# what the engine does with each payload, in a process that shares nothing with the trainer
_ENGINE_SIDE = """
import json, sys
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
rebuilt = []
for serialized in json.load(sys.stdin):
    payload = MultiprocessingSerializer.deserialize(serialized)
    bucket = FlattenedTensorBucket(flattened_tensor=payload["flattened_tensor"], metadata=payload["metadata"])
    rebuilt.append([[n, str(t.dtype), t.float().tolist()] for n, t in bucket.reconstruct_tensors()])
json.dump(rebuilt, sys.stdout)
"""


class _RecordingCellUpdater:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def submit_client_call(self, name: str, **kwargs) -> Future:
        self.calls.append((name, kwargs))
        done: Future = Future()
        done.set_result({"success": True})
        return done


@pytest.fixture(autouse=True)
def _restore_sharing_strategy():
    strategy = torch.multiprocessing.get_sharing_strategy()
    yield
    torch.multiprocessing.set_sharing_strategy(strategy)


def _protocol(*, replica_rank: int, replica_size: int, tp_sizes: list[int]) -> UpdateWeightFromSharedMemory:
    protocol = UpdateWeightFromSharedMemory(Namespace())
    protocol.cell_updaters_of_cell_id = {f"cell-{i}": _RecordingCellUpdater() for i in range(len(tp_sizes))}
    protocol._engine_tp_sizes = tp_sizes
    protocol._selector = "all"
    protocol._replica_rank, protocol._replica_size = replica_rank, replica_size
    return protocol


def _publish(protocol: UpdateWeightFromSharedMemory) -> None:
    protocol.begin_sync(weight_version=1, iter_buckets=None)
    for bucket in _BUCKETS:
        protocol.send_bucket(bucket)
    protocol.after_base_weights()


def _rebuild_in_another_process(payloads: list[str]) -> list:
    result = subprocess.run(
        [sys.executable, "-c", _ENGINE_SIDE], input=json.dumps(payloads), capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def test_another_process_rebuilds_every_tp_rank_bucket():
    """The engine maps the published host copies by name, so they must outlive the call that hands them over."""
    protocol = _protocol(replica_rank=0, replica_size=1, tp_sizes=[1, 2])
    _publish(protocol)

    for updater, tp_size in zip(protocol.cell_updaters_of_cell_id.values(), [1, 2], strict=True):
        assert [name for name, _ in updater.calls] == ["update_weights_from_tensor"] * len(_BUCKETS)
        assert all(kwargs["load_format"] == "flattened_bucket" for _, kwargs in updater.calls)
        assert all(len(kwargs["serialized_named_tensors"]) == tp_size for _, kwargs in updater.calls)
        payloads = [kwargs["serialized_named_tensors"][-1] for _, kwargs in updater.calls]
        expected = [[[n, str(t.dtype), t.float().tolist()] for n, t in bucket] for bucket in _BUCKETS]
        assert _rebuild_in_another_process(payloads) == expected
    protocol.finalize(weight_version=1)


def test_a_replica_rank_publishes_only_its_own_buckets():
    protocol = _protocol(replica_rank=1, replica_size=2, tp_sizes=[1])
    _publish(protocol)

    (updater,) = protocol.cell_updaters_of_cell_id.values()
    (payload,) = updater.calls[0][1]["serialized_named_tensors"]
    assert len(updater.calls) == 1
    assert [name for name, _, _ in _rebuild_in_another_process([payload])[0]] == ["layers.1.w"]
