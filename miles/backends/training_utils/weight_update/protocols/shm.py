from argparse import Namespace
from collections.abc import Sequence
from concurrent.futures import Future

import torch
import torch.multiprocessing
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.utils.data_replica import get_data_replica_rank_and_size
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.utils import async_utils


class UpdateWeightFromSharedMemory(WeightTransferProtocol):
    """
    Host weight sync through shared memory, for engines on the trainer's hosts. The ranks holding the
    gathered buckets split them round-robin, copy each flattened bucket into a shared-memory tensor
    once the collective gathers are done, and hand the engines its handle through
    /update_weights_from_tensor; each engine maps it in place, with no file written or parsed.
    A bucket is published as soon as it is on the host, so the copies overlap the engine loads.
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        # shared-memory tensors travel by /dev/shm file name, which an unrelated process can open
        torch.multiprocessing.set_sharing_strategy("file_system")
        self._bucket_index = 0
        self._owned_buckets: list[list[tuple[str, torch.Tensor]]] = []
        # keeps each host copy alive until the engines have mapped it
        self._published: list[torch.Tensor] = []
        self._futures: list[Future] = []

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, self.rollout_engines, engine_cell_ids)
        self.group_name = "miles-shm"
        self.is_sender = True
        self._selector = selector
        self._engine_tp_sizes = engine_gpu_counts
        self._replica_rank, self._replica_size = get_data_replica_rank_and_size(parallel_state, placement)

    def begin_sync(self, weight_version: int, iter_buckets) -> bool:
        self._bucket_index = 0
        self._owned_buckets = []
        return True

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        if self._bucket_index % self._replica_size == self._replica_rank:
            self._owned_buckets.append(bucket)
        self._bucket_index += 1

    def after_base_weights(self) -> None:
        # copied out only after the collective gathers finish, so the ranks' host copies run in parallel
        cell_updaters = list(self.cell_updaters_of_cell_id.values())
        for bucket in self._owned_buckets:
            flattened = FlattenedTensorBucket(named_tensors=bucket)
            device_tensor = flattened.get_flattened_tensor()
            host_tensor = torch.empty_like(device_tensor, device="cpu").share_memory_()
            host_tensor.copy_(device_tensor)
            self._published.append(host_tensor)
            payload = MultiprocessingSerializer.serialize(
                {"flattened_tensor": host_tensor, "metadata": flattened.get_metadata()}, output_str=True
            )
            self._futures += [
                updater.submit_client_call(
                    "update_weights_from_tensor",
                    serialized_named_tensors=[payload] * tp_size,
                    load_format="flattened_bucket",
                    selector=self._selector,
                )
                for updater, tp_size in zip(cell_updaters, self._engine_tp_sizes, strict=True)
            ]
        self._owned_buckets = []

    def finalize(self, weight_version: int) -> None:
        check_weight_sync_results(async_utils.wait_futures(self._futures), is_lora=False)
        self._futures = []
        self._published = []
