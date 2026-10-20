from argparse import Namespace
from collections.abc import Mapping


def should_return_sampling_mask(
    args: Namespace,
    sampling_params: Mapping[str, object] | None = None,
    *,
    evaluation: bool = False,
) -> bool:
    """Validate whether a training request must return its realized sampling support."""
    if evaluation:
        return False

    params = sampling_params or {}
    return validate_sampling_support_request(
        params,
        replay_enabled=args.use_sampling_support_replay,
        expected_temperature=float(getattr(args, "rollout_temperature", 1.0)),
    )


def validate_sampling_support_request(
    sampling_params: Mapping[str, object],
    *,
    replay_enabled: bool,
    expected_temperature: float | None,
) -> bool:
    """Validate one resolved training request against its sampling replay contract."""
    request_top_p = float(1.0 if sampling_params.get("top_p") is None else sampling_params["top_p"])
    if not 0.0 < request_top_p <= 1.0:
        raise ValueError(f"training request top_p must be in (0, 1], got {request_top_p}")

    if not replay_enabled:
        raw_top_k = sampling_params.get("top_k")
        request_top_k = -1 if raw_top_k is None else int(raw_top_k)
        if request_top_p < 1.0 or request_top_k > 0:
            raise ValueError("bounded training-request sampling requires bounded rollout sampling")
        return False

    missing_params = [name for name in ("top_p", "top_k", "temperature") if sampling_params.get(name) is None]
    if missing_params:
        raise ValueError(f"sampling-support replay requires explicit request parameters: {', '.join(missing_params)}")

    request_top_k = int(sampling_params["top_k"])
    if request_top_k <= 0:
        raise ValueError(f"training request top_k must be positive, got {request_top_k}")

    request_temperature = float(sampling_params["temperature"])
    if expected_temperature is None:
        raise ValueError("sampling-support replay requires a registered training temperature")
    if request_temperature != expected_temperature:
        raise ValueError(
            f"request temperature {request_temperature} does not match the training temperature "
            f"{expected_temperature}"
        )

    unsupported = {
        "frequency_penalty": (0, 0.0, None),
        "presence_penalty": (0, 0.0, None),
        "repetition_penalty": (1, 1.0, None),
        "logit_bias": ({}, None),
        "custom_logit_processor": (None,),
    }
    for name, allowed_values in unsupported.items():
        if sampling_params.get(name) not in allowed_values:
            raise ValueError(
                f"{name} is not supported with sampling-support replay because "
                "the trainer cannot reproduce its logit transformation"
            )
    return True
