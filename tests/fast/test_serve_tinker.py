import json
from types import SimpleNamespace

import pytest
import serve_tinker
from tests.fast.fixtures.serve_tinker_fakes import tinker_startup

from miles.tinker.runtime import MilesBackend
from miles.utils import http_utils
from miles.utils.async_utils import with_disposer

_ = tinker_startup


class TestServe:
    async def test_observed_engines_initialize_a_client_that_can_sample(self, tinker_startup: SimpleNamespace) -> None:
        """A gateway starting with unknown topology can sample after its engines initialize."""
        args = tinker_startup.args
        assert args.inference_runtime_mut_state.engine_count == 0

        await with_disposer(serve_tinker.serve, args)

        assert args.inference_runtime_mut_state.engine_count == 2
        assert http_utils._http_client is not None
        assert http_utils._client_concurrency == 24
        result = await MilesBackend(trainer=tinker_startup.trainer, router_url="http://router:30000").sample(
            payload={
                "prompt_tokens": [1, 2],
                "num_samples": 1,
                "sampling_params": {"max_tokens": 1},
                "prompt_logprobs": False,
                "topk_prompt_logprobs": 0,
            },
            lora_name=None,
        )

        assert result == {"sequences": [{"tokens": [42], "logprobs": [-0.25], "stop_reason": "length"}]}
        [request] = tinker_startup.requests
        assert request.method == "POST"
        assert str(request.url) == "http://router:30000/generate"
        assert json.loads(request.content)["input_ids"] == [1, 2]

    async def test_topology_query_failure_disposes_the_initialized_controller(
        self, tinker_startup: SimpleNamespace
    ) -> None:
        """A failed topology query still cleans up the controller that started its engines."""
        tinker_startup.controller.state_error = RuntimeError("topology unavailable")

        with pytest.raises(RuntimeError, match="topology unavailable"):
            await with_disposer(serve_tinker.serve, tinker_startup.args)

        assert tinker_startup.controller.disposed
        assert http_utils._http_client is None


class TestTrainerTokenBudget:
    @pytest.mark.parametrize(
        "context_parallel_size,local_budget,rejected", [(1, 1, False), (2, 1, True), (2, 3, False)]
    )
    async def test_zigzag_rounding_uses_raw_backend_context_parallelism(
        self, tinker_startup: SimpleNamespace, context_parallel_size: int, local_budget: int, rejected: bool
    ) -> None:
        """Only context-parallel training requires a budget large enough for two local zigzag chunks."""
        args = tinker_startup.args
        args.max_tokens_per_gpu = local_budget
        args.data_pad_size_multiplier = 1
        args.raw_megatron.base_args["context_parallel_size"] = context_parallel_size
        assert "context_parallel_size" not in vars(args)

        if rejected:
            with pytest.raises(AssertionError, match="at least one padding block"):
                await with_disposer(serve_tinker.serve, args)
            assert not tinker_startup.controller.initialized
            assert tinker_startup.trainer.request is None
        else:
            await with_disposer(serve_tinker.serve, args)
            assert tinker_startup.trainer.request is not None
