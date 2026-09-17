from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.component_trainer import TrainerOnlyConfig
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import compute_custom_function_config


class TestCustomFunctionView:

    def test_trainer_backend_fields_reach_legacy_hooks(self) -> None:
        """Legacy trainer hooks see backend-owned values in their flat configuration."""
        args = TrainerOnlyConfig(backend=MegatronArgsNamespace(tensor_model_parallel_size=4), trainer_role='actor', trainer_actor_index=2)
        view = compute_custom_function_config(args, CustomFunctionConfig(path='legacy'))
        assert view.tensor_model_parallel_size == 4
        assert view.trainer_role == 'actor'
        assert view.trainer_actor_index == 2
