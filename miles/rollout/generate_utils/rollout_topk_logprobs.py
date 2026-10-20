"""Record the sampler's top-k candidate log-probs at generation time."""

from argparse import Namespace
from collections.abc import Mapping
from typing import Any

import numpy as np

from miles.utils.rollout_topk_logprobs import validate_rollout_topk_logprobs, validate_rollout_topk_logprobs_sampling
from miles.utils.types import Sample


def configure_rollout_topk_logprobs_request(args: Namespace, request: dict[str, Any], *, openai: bool = False) -> None:
    """Request candidate probabilities, validating the actual per-call settings.

    The rollout args own ``top_logprobs_num`` / ``top_logprobs`` and ``sampling_logprobs_mode``
    on training requests and silently override client-supplied values.
    """
    k = args.rollout_top_logprobs_num
    if not k:
        return
    sampling = request if openai else request["sampling_params"]
    # Explicit defaults avoid model generation_config changing the distribution.
    for key, default in (("temperature", args.rollout_temperature), ("top_p", 1.0), ("top_k", -1), ("min_p", 0.0)):
        if sampling.get(key) is None:
            sampling[key] = default
    validate_rollout_topk_logprobs_sampling(sampling, temperature=args.rollout_temperature, candidate_count=k)
    if args.rollout_sampling_logprobs_mode == "support":
        # SGLang's support mode returns the actual post-filter behavior distribution.
        request.pop("top_logprobs", None)
        request.pop("top_logprobs_num", None)
        request["sampling_logprobs_mode"] = "support"
    elif openai:
        request.pop("sampling_logprobs_mode", None)
        request["top_logprobs"] = k
    else:
        request.pop("sampling_logprobs_mode", None)
        request["top_logprobs_num"] = k


def append_rollout_topk_logprobs(sample: Sample, meta: Mapping[str, Any], k: int) -> None:
    """Append compact [response, k] arrays of ``output_top_logprobs`` after appending generated tokens.

    Selected mode only: in support mode the trainer derives the candidates from the sampling mask.
    No rescoring is allowed: with stale rollouts it would replace the behavior policy. Slots
    unavailable from the server are represented by ID -1/logp -inf. CI checks the whole sample
    with validate_rollout_topk_logprobs_sample.
    """
    if not k:
        return
    n = len(meta.get("output_token_logprobs") or [])
    rows = meta.get("output_top_logprobs")
    if n and (rows is None or len(rows) != n):
        raise ValueError(
            "Rollout top-k logprobs collection requires SGLang output_top_logprobs for every generated token"
        )
    ids = np.full((n, k), -1, dtype=np.int32)
    logps = np.full((n, k), -np.inf, dtype=np.float32)
    for i, entries in enumerate(rows or []):
        token_ids = [entry[1] for entry in entries[:k]]
        probabilities = [entry[0] for entry in entries[:k]]
        ids[i, : len(token_ids)] = token_ids
        logps[i, : len(token_ids)] = probabilities
    for name, values in (("rollout_topk_token_ids", ids), ("rollout_topk_log_probs", logps)):
        previous = getattr(sample, name)
        setattr(sample, name, values if previous is None else np.concatenate((previous, values)))


def pad_rollout_topk_logprobs(sample: Sample, count: int) -> None:
    """Pad non-trained tool/observation positions before extending the response."""
    for field, fill in (("rollout_topk_token_ids", -1), ("rollout_topk_log_probs", -np.inf)):
        values = getattr(sample, field)
        if values is not None:
            padding = np.full((count, values.shape[1]), fill, dtype=values.dtype)
            setattr(sample, field, np.concatenate((values, padding)))


def validate_rollout_topk_logprobs_sample(sample: Sample, k: int) -> None:
    """Validate a sample's recorded candidates; see ``validate_rollout_topk_logprobs``."""
    sample.validate()
    ids, logps = sample.rollout_topk_token_ids, sample.rollout_topk_log_probs
    if ids is None or logps is None or sample.rollout_log_probs is None:
        raise ValueError(
            "Rollout top-k logprobs collection requires candidate and sampled-token logprobs on every rollout"
        )
    if ids.shape[1] != k:
        raise ValueError("Rollout top-k logprobs candidates must have --rollout-top-logprobs-num columns")
    validate_rollout_topk_logprobs(
        ids,
        logps,
        sampled_log_probs=np.asarray(sample.rollout_log_probs),
        response_tokens=np.asarray(
            sample.tokens[-sample.response_length :] if sample.response_length else [], dtype=np.int64
        ),
        loss_mask=np.asarray(
            sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length, dtype=bool
        ),
        support=None,
    )


def merge_rollout_topk_logprobs_field(first: Sample, second: Sample, field: str, gap: int) -> np.ndarray | None:
    a, b = getattr(first, field), getattr(second, field)
    if a is None and b is None:
        return None
    if a is None or b is None or a.shape[1] != b.shape[1]:
        raise ValueError(f"Both turns must carry matching {field} for rollout top-k logprobs collection")
    fill = -1 if field == "rollout_topk_token_ids" else -np.inf
    return np.concatenate((a, np.full((gap, a.shape[1]), fill, dtype=a.dtype), b))
