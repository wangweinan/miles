"""Trainer side of the replay data: rebuild R3 and indexer top-k, and derive the sampling-mask columns.

Runs once per train attempt on this rank's DP shard, after the shard is selected
(``process_rollout_data``), and writes the train dict columns training consumes today:
``rollout_routed_experts``, ``rollout_indexer_topk``, ``rollout_sampling_mask_ids`` /
``rollout_sampling_mask_offsets`` and, in support mode, ``rollout_topk_*``. ``rollout_log_probs``
comes from the rollout, which reads the sampled tokens' log-probs from the responses.
The inputs are only this shard's samples, so a retry with another DP size derives again from scratch.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from miles.utils.replay_index import (
    INDEXER_TOPK,
    ROUTED_EXPERTS,
    MaskPiece,
    ReplayDataError,
    ReplayIndex,
    ReplaySegments,
    pop_train_columns,
)
from miles.utils.rollout_topk_logprobs import validate_rollout_topk_logprobs

_ROWS_COLUMNS = ((ROUTED_EXPERTS, "rollout_routed_experts"), (INDEXER_TOPK, "rollout_indexer_topk"))


def materialize_replay_columns(args: Any, rollout_data: dict[str, Any]) -> None:
    """Replace this shard's replay index columns with the columns training reads."""
    indexes = pop_train_columns(rollout_data)
    if indexes is None:
        return
    # Inline rows only: no segment needs arrays read from elsewhere.
    fetched: dict[tuple[str, str], np.ndarray] = {}
    for name, column in _ROWS_COLUMNS:
        fields = _complete_field(indexes, name)
        if fields is not None:
            rollout_data[column] = [_materialize_rows(field, fetched) for field in fields]
    if (masks := _complete_field(indexes, "sampling_mask")) is not None:
        _write_sampling_mask_columns(args, rollout_data, [mask.mask_pieces(fetched) for mask in masks])


def _complete_field(indexes: Sequence[ReplayIndex], name: str) -> list[ReplaySegments] | None:
    fields = [getattr(index, name) for index in indexes]
    if all(field is None for field in fields):
        return None
    missing = [i for i, field in enumerate(fields) if field is None]
    if missing:
        raise ReplayDataError(f"{name} is missing for shard rows {missing[:8]} while other rows carry it")
    return fields


def _materialize_rows(field: ReplaySegments, fetched: Mapping[tuple[str, str], np.ndarray]) -> np.ndarray:
    rows = field.materialize_rows(fetched)
    if field.name == ROUTED_EXPERTS and rows.size and not rows.any():
        raise ReplayDataError(
            "routed_experts payload is all zeros: the sglang engine did not capture routed experts "
            "(topk-bypassing --moe-runner-backend such as flashinfer_trtllm?)."
        )
    return rows


# ============================ sampling mask (D9) ============================


@dataclass(frozen=True)
class DerivedSamplingMask:
    ids: np.ndarray  # int32 [sum(lengths)], every row's support in order
    offsets: np.ndarray  # int64 [rows + 1]
    # float32 [sum(lengths)] when every SGLang row carries its support log-probs, else None.
    support_logprobs: np.ndarray | None
    forced: np.ndarray  # bool [rows], rows of tokens Miles inserted


def derive_sampling_mask(pieces: Sequence[MaskPiece], sampled_tokens: np.ndarray) -> DerivedSamplingMask:
    """Validate a sample's sampling mask against its sampled tokens and derive its columns.

    Forced rows support exactly the inserted token, with log-prob 0.
    """
    num_rows = len(sampled_tokens)
    if (total := sum(piece.num_rows for piece in pieces)) != num_rows:
        raise ReplayDataError(f"sampling mask has {total} rows for {num_rows} response tokens")
    parts = []
    cursor = 0
    for piece in pieces:
        parts.append(_derive_piece(piece, sampled_tokens[cursor : cursor + piece.num_rows]))
        cursor += piece.num_rows
    lengths, ids, support, forced = (list(column) for column in zip(*parts, strict=True)) if parts else ([],) * 4
    offsets = np.zeros(num_rows + 1, dtype=np.int64)
    np.cumsum(_concat(lengths, np.int64), out=offsets[1:])
    return DerivedSamplingMask(
        ids=_concat(ids, np.int32),
        offsets=offsets,
        support_logprobs=None if any(value is None for value in support) else _concat(support, np.float32),
        forced=_concat(forced, np.bool_),
    )


def _derive_piece(piece: MaskPiece, tokens: np.ndarray) -> tuple:
    """(lengths, ids, support log-probs or None, forced) of one piece."""
    n = piece.num_rows
    if piece.forced:
        return np.ones(n, dtype=np.int64), tokens.astype(np.int32), np.zeros(n, np.float32), np.ones(n, np.bool_)
    lengths = piece.lengths.astype(np.int64)
    ids = piece.token_ids.astype(np.int32, copy=False)
    if (lengths <= 0).any():
        raise ReplayDataError("sampling support must contain at least one token")
    entry_rows = np.repeat(np.arange(n), lengths)
    counts = np.bincount(entry_rows[ids.astype(np.int64) == tokens.astype(np.int64)[entry_rows]], minlength=n)
    if (counts == 0).any():
        row = int(np.flatnonzero(counts == 0)[0])
        raise ReplayDataError(f"sampled token {int(tokens[row])} is absent from its sampling support")
    if piece.logprobs is None:
        return lengths, ids, None, np.zeros(n, np.bool_)
    if (counts != 1).any():
        raise ReplayDataError("Sampled token must occur exactly once in its sampling support")
    if len(piece.logprobs) != len(ids):
        raise ReplayDataError(f"{len(piece.logprobs)} support log-probs for {len(ids)} support entries")
    return lengths, ids, piece.logprobs.astype(np.float32, copy=False), np.zeros(n, np.bool_)


def derive_support_topk(derived: DerivedSamplingMask, *, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``[rows, k]`` candidates of the realized support; forced rows have none (ID -1, log-prob -inf)."""
    if derived.support_logprobs is None:
        raise ReplayDataError("rollout top-k candidates need support log-probs (sampling_logprobs_mode=support)")
    num_rows = len(derived.offsets) - 1
    lengths = np.diff(derived.offsets)
    if (lengths[~derived.forced] > k).any():
        raise ReplayDataError("Rollout top-k logprobs sampling support exceeds --rollout-top-logprobs-num")
    topk_ids = np.full((num_rows, k), -1, dtype=np.int32)
    topk_logprobs = np.full((num_rows, k), -np.inf, dtype=np.float32)
    entry_rows = np.repeat(np.arange(num_rows), lengths)
    columns = np.arange(len(derived.ids)) - derived.offsets[entry_rows]
    keep = ~derived.forced[entry_rows]
    topk_ids[entry_rows[keep], columns[keep]] = derived.ids[keep]
    topk_logprobs[entry_rows[keep], columns[keep]] = derived.support_logprobs[keep]
    return topk_ids, topk_logprobs


def _write_sampling_mask_columns(args: Any, rollout_data: dict[str, Any], pieces: list[list[MaskPiece]]) -> None:
    support = args.rollout_sampling_logprobs_mode == "support"
    k = args.rollout_top_logprobs_num
    derive_topk = bool(k) and support and "rollout_topk_token_ids" not in rollout_data
    columns: dict[str, list[Any]] = {"rollout_sampling_mask_ids": [], "rollout_sampling_mask_offsets": []}
    if derive_topk:
        columns |= {"rollout_topk_token_ids": [], "rollout_topk_log_probs": []}
    for i, sample_pieces in enumerate(pieces):
        tokens, response_length = rollout_data["tokens"][i], rollout_data["response_lengths"][i]
        sampled = np.asarray(tokens[len(tokens) - response_length :], dtype=np.int64)
        try:
            derived = derive_sampling_mask(sample_pieces, sampled)
            if derive_topk:
                topk_ids, topk_logprobs = derive_support_topk(derived, k=k)
        except ReplayDataError as e:
            raise ReplayDataError(f"shard row {i} (sample_index={_sample_index(rollout_data, i)}): {e}") from e
        columns["rollout_sampling_mask_ids"].append(torch.from_numpy(derived.ids))
        columns["rollout_sampling_mask_offsets"].append(torch.from_numpy(derived.offsets))
        if derive_topk:
            columns["rollout_topk_token_ids"].append(topk_ids)
            columns["rollout_topk_log_probs"].append(topk_logprobs)
    rollout_data.update(columns)
    if k and support and args.ci_test:
        _validate_support_topk(rollout_data)
    if k and args.loss_type == "score_centering" and support:
        compact_candidate_ids(rollout_data)


def _validate_support_topk(rollout_data: dict[str, Any]) -> None:
    for i, (ids, logprobs) in enumerate(
        zip(rollout_data["rollout_topk_token_ids"], rollout_data["rollout_topk_log_probs"], strict=True)
    ):
        tokens, response_length = rollout_data["tokens"][i], rollout_data["response_lengths"][i]
        validate_rollout_topk_logprobs(
            np.asarray(ids),
            np.asarray(logprobs),
            sampled_log_probs=np.asarray(rollout_data["rollout_log_probs"][i]),
            response_tokens=np.asarray(tokens[len(tokens) - response_length :], dtype=np.int64),
            loss_mask=np.asarray(rollout_data["loss_masks"][i], dtype=bool),
            support=(
                rollout_data["rollout_sampling_mask_ids"][i].numpy(),
                rollout_data["rollout_sampling_mask_offsets"][i].numpy(),
            ),
        )


def compact_candidate_ids(data: dict[str, Any]) -> None:
    """Share ordered support IDs in training batches; preserve other producer layouts."""
    lengths = []
    for dense, logps, support, offsets in zip(
        data["rollout_topk_token_ids"],
        data["rollout_topk_log_probs"],
        data["rollout_sampling_mask_ids"],
        data["rollout_sampling_mask_offsets"],
        strict=True,
    ):
        dense, logps = torch.as_tensor(dense), torch.as_tensor(logps)
        if dense.ndim != 2 or dense.shape != logps.shape or dense.size(0) != offsets.numel() - 1:
            return
        valid = dense >= 0
        count = valid.sum(-1, dtype=torch.int32)
        columns = torch.arange(dense.size(1))
        prefix = columns < count[:, None]
        if not torch.equal(valid, prefix) or not (dense[~prefix] == -1).all():
            return
        if not (count[count > 0] == offsets.diff()[count > 0]).all():
            return
        positions = offsets[:-1, None] + columns
        if not torch.equal(dense[prefix], support[positions[prefix]]):
            return
        if not torch.isneginf(logps[count == 0]).all():
            return
        lengths.append(count)
    data["rollout_topk_lengths"] = lengths
    del data["rollout_topk_token_ids"]


def _sample_index(rollout_data: Mapping[str, Any], row: int) -> Any:
    indices = rollout_data.get("sample_indices")
    return indices[row] if indices is not None else None


def _concat(parts: list[np.ndarray], dtype: type) -> np.ndarray:
    return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)
