import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.utils.args.runtime import OrchestratorConfig
from miles.utils.args.trainer_utils import compute_trainer_ckpt_load_config


class TestOrchestratorCheckpointRequests:
    @pytest.mark.parametrize(
        "extra",
        [(), ("--advantage-estimator", "ppo"), ("--megatron-config", encode_megatron_config("a", "b"))],
    )
    def test_loading_uses_only_fields_owned_by_the_orchestrator(self, extra: tuple[str, ...]) -> None:
        """Checkpoint selection must work after deployment-only scaling fields have been removed."""
        args = parse_megatron_test_config(*extra, "--load", "/run", "--ref-load", "/reference")
        orchestrator = OrchestratorConfig.slice_from(args)

        assert "actor_num_nodes" not in type(orchestrator).model_fields
        for trainer in args.raw_megatron.trainers:
            request = compute_trainer_ckpt_load_config(orchestrator, trainer)
            assert request == compute_trainer_ckpt_load_config(args, trainer)
            assert request.load == "/reference"
