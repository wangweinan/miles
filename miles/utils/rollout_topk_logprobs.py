"""Rollout top-k logprobs: the sampler's candidate log-probs recorded for each generated token.

``--rollout-top-logprobs-num`` sets the width of ``Sample.rollout_topk_token_ids`` /
``rollout_topk_log_probs``; it records the sampling distribution and does not change it.
``arguments.py`` derives ``rollout_sampling_logprobs_mode`` from the rollout filters: ``selected`` reads
``output_top_logprobs``, the top-K of the full-vocabulary distribution (unfiltered sampling), on the
rollout side; ``support`` is the whole realized support (filtered sampling), which the trainer derives
from the sampling mask of sampling-support replay (``Sample.replay_index``).
"""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def validate_rollout_topk_logprobs_args(args: Any) -> None:
    """Reject rollout top-k logprobs settings whose recorded candidates cannot match the sampler."""
    k = args.rollout_top_logprobs_num
    if k < 0:
        raise ValueError(f"--rollout-top-logprobs-num must be non-negative, got {k}")
    if not k:
        return
    if args.rollout_sampling_logprobs_mode == "support" and k < args.rollout_top_k:
        raise ValueError(
            "Filtered rollout sampling requires --rollout-top-logprobs-num >= --rollout-top-k "
            "so the recorded candidates can hold the whole sampling support"
        )
    opd_student_top_k = args.use_opd and args.opd_log_prob_top_k > 0 and args.opd_top_k_strategy != "only-teacher"
    if opd_student_top_k:
        raise ValueError("--rollout-top-logprobs-num cannot be combined with OPD student top-k log-probs")


def validate_rollout_topk_logprobs_sampling(
    sampling: Mapping[str, Any], *, temperature: float, candidate_count: int
) -> None:
    """Require per-call sampling whose distribution the recorded candidates describe.

    Top-p/top-k bounds are checked by the rollout args and sampling-support replay;
    this adds what recorded candidates need on top of them.
    """
    if sampling.get("temperature", temperature) != temperature or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Rollout top-k logprobs collection requires the same positive rollout temperature on every generation call"
        )
    if sampling.get("top_k", -1) > candidate_count:
        raise ValueError(
            "Rollout top-k logprobs collection requires top_k <= --rollout-top-logprobs-num to hold the whole support"
        )
    if sampling.get("min_p", 0.0) != 0.0:
        raise ValueError("Rollout top-k logprobs collection requires min_p=0.0")
    for key in ("json_schema", "regex", "ebnf", "structural_tag", "custom_logit_processor", "logit_bias"):
        if sampling.get(key):
            raise ValueError(f"Rollout top-k logprobs collection does not support constrained/custom sampling ({key})")
    response_format = sampling.get("response_format")
    if response_format and (not isinstance(response_format, Mapping) or response_format.get("type", "text") != "text"):
        raise ValueError("Rollout top-k logprobs collection does not support constrained response_format")
    tool_choice = sampling.get("tool_choice", "auto")
    if tool_choice not in (None, "auto", "none"):
        raise ValueError("Rollout top-k logprobs collection does not support constrained tool_choice")
    if tool_choice != "none" and any(tool.get("function", {}).get("strict") for tool in sampling.get("tools") or []):
        raise ValueError("Rollout top-k logprobs collection does not support strict tool schemas")


def validate_rollout_topk_logprobs(
    ids: np.ndarray,
    logps: np.ndarray,
    *,
    sampled_log_probs: np.ndarray,
    response_tokens: np.ndarray,
    loss_mask: np.ndarray,
    support: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    """Validate one sample's candidate distributions, including custom producers and restored rollouts.

    ``support`` is the sample's sampling mask in CSR form (ids, offsets), when it has one. Full-sample
    sorting and support matching have significant CPU and memory costs; training callers must enable
    this validation only under --ci-test.
    """
    valid = ids >= 0
    sorted_ids = np.sort(ids, axis=-1)
    if ((sorted_ids[:, 1:] >= 0) & (sorted_ids[:, 1:] == sorted_ids[:, :-1])).any():
        raise ValueError("Duplicate rollout top-k logprobs candidate token IDs")
    if (loss_mask & ~valid.any(-1)).any():
        raise ValueError("Every trained token needs rollout top-k logprobs candidates")
    if support is not None:
        in_support, lengths = _support_membership(ids, *support)
        if ((valid != in_support) & loss_mask[:, None]).any() or ((valid.sum(-1) != lengths) & loss_mask).any():
            raise ValueError("Rollout top-k logprobs candidates must equal the sampling support")
    matches = (ids == response_tokens[:, None]) & valid & loss_mask[:, None]
    sampled = np.broadcast_to(np.asarray(sampled_log_probs)[:, None], logps.shape)
    if not np.allclose(sampled[matches], logps[matches], atol=1e-5, rtol=1e-5):
        raise ValueError("Sampled and candidate probabilities must come from the same sampler distribution")


def _support_membership(
    ids: np.ndarray, support_ids: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match candidate IDs to ragged support IDs without a per-token Python loop."""
    n = len(ids)
    if not n:
        return np.zeros_like(ids, dtype=bool), np.empty(0, dtype=np.int64)
    lengths = np.diff(offsets[: n + 1])
    support_ids = support_ids[: offsets[n]]
    stride = 1 << 32  # Token IDs are nonnegative int32; rows remain distinct.
    support_rows = np.repeat(np.arange(n, dtype=np.int64), lengths)
    support_keys = np.sort(support_rows * stride + support_ids.astype(np.int64))
    candidate_keys = np.arange(n, dtype=np.int64)[:, None] * stride + ids.astype(np.int64)
    positions = np.searchsorted(support_keys, candidate_keys)
    membership = (ids >= 0) & (support_keys[np.minimum(positions, len(support_keys) - 1)] == candidate_keys)
    return membership, lengths
