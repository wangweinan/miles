"""Per-token replay data of one sample: routed experts (R3), indexer top-k and the sampling mask.

A ``ReplayIndex`` holds one ``ReplaySegments`` per field. A field is an ordered tuple of
row segments; each segment is a row range of one source:

- inline: arrays that came in the response's ``meta_info``;
- forced (sampling mask only): rows of tokens Miles inserted itself, such as tool observations.

Rollout-side edits (truncation, multi-turn merge, additional R3) only rearrange segments and never
read data. The trainer materializes the arrays (``miles.backends.training_utils.data.replay``).

Rows: R3 and indexer top-k count prompt and response tokens minus the last token
(``len(tokens) - 1``); the sampling mask counts response tokens.

The sampling mask holds each row's support and, in support log-prob mode, a log-prob per support
entry. The sampled tokens' log-probs, one per token, are not replay data: SGLang returns them in
``meta_info`` in both modes, and the rollout keeps them as ``rollout_log_probs``
(``sampled_logprobs_from_meta_info``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Any

import numpy as np
import pybase64

from miles.utils.retry_utils import NonRetryableError

ROUTED_EXPERTS = "routed_experts"
INDEXER_TOPK = "indexer_topk"
SAMPLING_MASK = "sampling_mask"

# Names of the arrays a segment holds, one per stored field.
MASK_LENGTHS = "output_token_sampling_mask_lengths"
MASK_TOKEN_IDS = "output_token_sampling_mask_token_ids"
# Support log-prob mode only: a log-prob per support entry.
MASK_LOGPROBS = "output_token_sampling_logprobs"

# meta_info keys of the sampled tokens' log-probs, one per output token, by log-prob mode.
_SAMPLED_LOGPROBS_KEYS = {False: "output_token_sampling_logprobs", True: "output_token_sampling_selected_logprobs"}


class ReplayDataError(NonRetryableError, ValueError):
    """Replay data is missing, inconsistent, or has the wrong shape.

    Retrying cannot fix it: the data a retry would read is the same. A ``ValueError``, so the session
    server answers a malformed response with 422 as before.
    """


@dataclass(frozen=True)
class _FieldLayout:
    name: str
    # Stored field names this field may have; the first one's leading dimension counts its rows.
    # A segment has the ones its source carries (the mask has MASK_LOGPROBS in support mode only).
    stored: tuple[str, ...]


_LAYOUTS: tuple[_FieldLayout, ...] = (
    _FieldLayout(ROUTED_EXPERTS, (ROUTED_EXPERTS,)),
    _FieldLayout(INDEXER_TOPK, (INDEXER_TOPK,)),
    _FieldLayout(SAMPLING_MASK, (MASK_LENGTHS, MASK_TOKEN_IDS, MASK_LOGPROBS)),
)
_LAYOUT_BY_NAME = {layout.name: layout for layout in _LAYOUTS}

INLINE = "inline"
FORCED = "forced"


@dataclass(frozen=True, eq=False)
class ReplaySegment:
    """Rows ``[start, stop)`` of one source; forced rows have no ``inline`` arrays."""

    start: int
    stop: int
    # Stored field name -> the whole inline array the rows index into.
    inline: Mapping[str, np.ndarray] | None = field(default=None, repr=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReplaySegment):
            return NotImplemented
        if (self.start, self.stop) != (other.start, other.stop):
            return False
        if self.inline is None or other.inline is None:
            return self.inline is other.inline
        return self.inline.keys() == other.inline.keys() and all(
            np.array_equal(self.inline[name], other.inline[name]) for name in self.inline
        )

    __hash__ = None

    @property
    def kind(self) -> str:
        if self.inline is not None:
            return INLINE
        return FORCED

    @property
    def num_rows(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class ReplaySegments:
    """One field of one sample: segments in row order."""

    name: str
    segments: tuple[ReplaySegment, ...] = ()

    @property
    def num_rows(self) -> int:
        return sum(segment.num_rows for segment in self.segments)

    def prefix(self, num_rows: int) -> ReplaySegments:
        if not 0 <= num_rows <= self.num_rows:
            raise ReplayDataError(f"{self.name} prefix of {num_rows} rows is outside [0, {self.num_rows}]")
        segments, left = [], num_rows
        for segment in self.segments:
            if left == 0 and segments:
                break
            take = min(left, segment.num_rows)
            # An empty prefix keeps one empty segment, so the rows keep their trailing shape.
            segments.append(replace(segment, stop=segment.start + take))
            left -= take
        return ReplaySegments(self.name, tuple(segments))

    def concat(self, *others: ReplaySegments) -> ReplaySegments:
        segments = self.segments + tuple(chain.from_iterable(other.segments for other in others))
        if self.name != SAMPLING_MASK:
            _check_row_shapes(self.name, segments)
        return ReplaySegments(self.name, segments)

    def materialize_rows(self, fetched: Mapping[tuple[str, str], np.ndarray]) -> np.ndarray:
        """R3 or indexer top-k rows: ``[num_rows, num_layers, topk]``.

        ``fetched`` maps (key, stored field name) to arrays of segments that are not inline; an index
        of inline rows passes ``{}``.
        """
        layout = _LAYOUT_BY_NAME[self.name]
        (stored,) = layout.stored
        parts = [
            _source_arrays(segment, layout, fetched)[stored][segment.start : segment.stop] for segment in self.segments
        ]
        if not parts:
            raise ReplayDataError(f"{self.name} has no rows to materialize")
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def mask_pieces(self, fetched: Mapping[tuple[str, str], np.ndarray]) -> list[MaskPiece]:
        """The sampling mask as pieces in response order."""
        pieces = []
        for segment in self.segments:
            if segment.kind == FORCED:
                pieces.append(MaskPiece(num_rows=segment.num_rows))
                continue
            arrays = _slice_mask(_source_arrays(segment, _LAYOUT_BY_NAME[SAMPLING_MASK], fetched), segment)
            pieces.append(
                MaskPiece(
                    num_rows=segment.num_rows,
                    lengths=arrays[MASK_LENGTHS],
                    token_ids=arrays[MASK_TOKEN_IDS],
                    logprobs=arrays.get(MASK_LOGPROBS),
                )
            )
        return pieces


@dataclass(frozen=True, eq=False)
class MaskPiece:
    """Sampling-mask rows of one segment; forced rows carry no arrays."""

    num_rows: int
    lengths: np.ndarray | None = field(default=None, repr=False)
    token_ids: np.ndarray | None = field(default=None, repr=False)
    # Per support entry; None in selected log-prob mode.
    logprobs: np.ndarray | None = field(default=None, repr=False)

    @property
    def forced(self) -> bool:
        return self.lengths is None


@dataclass(frozen=True)
class ReplayIndex:
    """The replay fields of one sample; a field is None when the sample has none."""

    routed_experts: ReplaySegments | None = None
    indexer_topk: ReplaySegments | None = None
    sampling_mask: ReplaySegments | None = None

    def fields(self) -> Iterator[ReplaySegments]:
        for layout in _LAYOUTS:
            if (segments := getattr(self, layout.name)) is not None:
                yield segments

    def is_empty(self) -> bool:
        return next(self.fields(), None) is None

    # ----------------------------- rollout-side edits -----------------------------

    def strip_tail(self, num_tokens: int) -> ReplayIndex:
        """Drop the last ``num_tokens`` tokens' rows of every field."""
        return ReplayIndex(
            **{segments.name: segments.prefix(_rows_after_strip(segments, num_tokens)) for segments in self.fields()}
        )

    def append_response(self, new: ReplayIndex, *, response_length: int) -> ReplayIndex:
        """Add a later generation call on the same token sequence.

        R3 and indexer top-k cover the whole sequence in every call, so ``new``'s replace them;
        sampling-mask rows of ``new`` follow the ``response_length`` rows already recorded.
        """
        sampling_mask = self.sampling_mask
        if new.sampling_mask is not None:
            sampling_mask = _append_mask_rows(self.sampling_mask, new.sampling_mask, response_length=response_length)
        return ReplayIndex(
            routed_experts=new.routed_experts, indexer_topk=new.indexer_topk, sampling_mask=sampling_mask
        )

    def append_forced_tokens(self, num_tokens: int, *, response_length: int) -> ReplayIndex:
        """Add ``num_tokens`` tokens Miles inserted (tool observations) after ``response_length`` rows."""
        if self.sampling_mask is None:
            return self
        forced = ReplaySegments(SAMPLING_MASK, (ReplaySegment(start=0, stop=num_tokens),))
        return replace(
            self, sampling_mask=_append_mask_rows(self.sampling_mask, forced, response_length=response_length)
        )

    def merge_turn(self, later: ReplayIndex, *, observation_tokens: int) -> ReplayIndex:
        """Merge ``later``, the next turn of the same trajectory, after ``observation_tokens`` inserted tokens."""
        first, second = self.sampling_mask, later.sampling_mask
        if (first is None) != (second is None):
            raise ReplayDataError("cannot merge samples unless both turns carry a complete rollout sampling mask")
        sampling_mask = None
        if first is not None:
            forced = ReplaySegments(SAMPLING_MASK, (ReplaySegment(start=0, stop=observation_tokens),))
            sampling_mask = first.concat(forced, second)
        return ReplayIndex(
            routed_experts=later.routed_experts, indexer_topk=later.indexer_topk, sampling_mask=sampling_mask
        )

    @classmethod
    def from_arrays(
        cls, *, routed_experts: np.ndarray | None = None, indexer_topk: np.ndarray | None = None
    ) -> ReplayIndex:
        """An index of inline ``[rows, layers, topk]`` int32 arrays, for producers that decode them themselves."""
        fields = {}
        for name, array in ((ROUTED_EXPERTS, routed_experts), (INDEXER_TOPK, indexer_topk)):
            if array is not None:
                if array.ndim != 3 or array.dtype != np.int32:
                    raise ReplayDataError(f"{name} must be an int32 [rows, layers, topk] array")
                fields[name] = ReplaySegments(name, (ReplaySegment(start=0, stop=len(array), inline={name: array}),))
        return cls(**fields)

    # ------------------------------- parse a response -------------------------------

    @classmethod
    def from_meta_info(
        cls,
        meta_info: Mapping[str, Any],
        *,
        routed_experts_rows: int | None,
        num_layers: int | None,
        indexer_topk_rows: int | None,
        indexer_num_streams: int | None,
        return_sampling_mask: bool,
        support_logprobs: bool,
        num_output_tokens: int,
        aborted: bool,
        allow_extra_routed_experts_row: bool = False,
    ) -> ReplayIndex:
        """The only parser of a response's replay fields.

        A field whose expected rows are None is not parsed. Structural checks that need no data (rows, layers, the
        sampling-mask length) run here, so a malformed response fails on arrival; checks that read
        data run on the trainer.
        ``allow_extra_routed_experts_row`` accepts R3 with one row past ``routed_experts_rows``
        (the legacy rollout's stop edge) and drops it.
        """
        routed_experts = indexer_topk = None
        if routed_experts_rows is not None:
            routed_experts = _parse_rows_field(
                meta_info,
                ROUTED_EXPERTS,
                rows=routed_experts_rows,
                num_layers=num_layers,
                allow_extra_row=allow_extra_routed_experts_row,
            )
        if indexer_topk_rows is not None:
            indexer_topk = _parse_rows_field(
                meta_info,
                INDEXER_TOPK,
                rows=indexer_topk_rows,
                num_layers=_indexer_num_layers(meta_info, expected=indexer_num_streams),
                allow_extra_row=False,
            )
        sampling_mask = None
        if return_sampling_mask:
            sampling_mask = _parse_sampling_mask(
                meta_info,
                support_logprobs=support_logprobs,
                num_output_tokens=num_output_tokens,
                aborted=aborted,
            )
        return cls(routed_experts=routed_experts, indexer_topk=indexer_topk, sampling_mask=sampling_mask)

    # --------------------------------- wire format ---------------------------------

    def to_wire(self) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
        """Split into a builtins descriptor and the inline arrays.

        Inline rows of each stored field are concatenated in segment order; the descriptor names
        each segment's kind and row count.
        """
        descriptor: dict[str, list[dict[str, Any]]] = {}
        arrays: dict[str, np.ndarray] = {}
        for segments in self.fields():
            layout = _LAYOUT_BY_NAME[segments.name]
            entries, inline_parts = [], []
            for segment in segments.segments:
                if segment.kind == INLINE:
                    entries.append({INLINE: segment.num_rows})
                    inline_parts.append(_slice_inline(layout, segment))
                else:
                    entries.append({FORCED: segment.num_rows})
            descriptor[segments.name] = entries
            for stored in inline_parts[0] if inline_parts else ():
                parts = [part[stored] for part in inline_parts]
                arrays[stored] = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return descriptor, arrays

    @classmethod
    def from_wire(
        cls, descriptor: Mapping[str, Sequence[Mapping[str, Any]]], arrays: Mapping[str, np.ndarray]
    ) -> ReplayIndex:
        fields = {}
        for name, entries in descriptor.items():
            layout = _LAYOUT_BY_NAME[name]
            inline = {stored: arrays[stored] for stored in layout.stored if stored in arrays} or None
            segments, cursor = [], 0
            for entry in entries:
                if INLINE in entry:
                    if inline is None:
                        raise ReplayDataError(f"{name} descriptor names inline rows but the payload has no arrays")
                    rows = int(entry[INLINE])
                    segments.append(ReplaySegment(start=cursor, stop=cursor + rows, inline=inline))
                    cursor += rows
                else:
                    segments.append(ReplaySegment(start=0, stop=int(entry[FORCED])))
            if inline is not None and cursor != _num_source_rows(layout, inline):
                raise ReplayDataError(
                    f"{name} descriptor uses {cursor} inline rows of {_num_source_rows(layout, inline)}"
                )
            fields[name] = ReplaySegments(name, tuple(segments))
        return cls(**fields)


def sampled_logprobs_from_meta_info(
    meta_info: Mapping[str, Any], *, support_logprobs: bool, num_output_tokens: int, aborted: bool
) -> list[float]:
    """The sampled tokens' log-probs under the sampling distribution, one per output token.

    With a sampling mask they replace ``output_token_logprobs`` as ``rollout_log_probs``. SGLang
    returns them in ``meta_info`` in both log-prob modes.
    """
    key = _SAMPLED_LOGPROBS_KEYS[support_logprobs]
    values = meta_info.get(key)
    if values is None:
        if _is_no_output_abort(meta_info, aborted=aborted, num_output_tokens=num_output_tokens):
            return []
        raise ReplayDataError(
            f"SGLang response is missing {key}; use an SGLang build that returns the sampled tokens' "
            "sampling log-probs inline"
        )
    if len(values) != num_output_tokens:
        raise ReplayDataError(f"{key} has {len(values)} values for {num_output_tokens} output tokens")
    if any(isinstance(value, (list, tuple)) for value in values):
        raise ReplayDataError(f"{key} must hold one scalar per output token")
    return [float(value) for value in values]


def response_has_routed_experts(meta_info: Mapping[str, Any]) -> bool:
    """Whether a response carries R3 rows; reads no data."""
    return ROUTED_EXPERTS in meta_info


# ============================== train dict columns ==============================

# One builtins descriptor per sample (``ReplayIndex.to_wire``); inline arrays go in typed columns.
REPLAY_INDEX_COLUMN = "replay_index"
INLINE_TRAIN_COLUMNS: dict[str, tuple[str, np.dtype]] = {
    ROUTED_EXPERTS: ("rollout_routed_experts", np.dtype(np.int32)),
    INDEXER_TOPK: ("rollout_indexer_topk", np.dtype(np.int32)),
    MASK_LENGTHS: ("rollout_sampling_mask_lengths", np.dtype(np.int32)),
    MASK_TOKEN_IDS: ("rollout_sampling_mask_token_ids", np.dtype(np.int32)),
    MASK_LOGPROBS: ("rollout_sampling_mask_logprobs", np.dtype(np.float32)),
}


def compute_train_columns(indexes: Sequence[ReplayIndex]) -> dict[str, list[Any]]:
    """Train dict columns of a batch's replay indexes; empty when no sample has replay data.

    A sample without inline rows of a stored field gets an empty array in that column.
    """
    wires = [index.to_wire() for index in indexes]
    if not any(descriptor for descriptor, _ in wires):
        return {}
    columns: dict[str, list[Any]] = {REPLAY_INDEX_COLUMN: [descriptor for descriptor, _ in wires]}
    for stored, (column, dtype) in INLINE_TRAIN_COLUMNS.items():
        if not any(stored in arrays for _, arrays in wires):
            continue
        values = []
        for _, arrays in wires:
            array = arrays.get(stored, np.empty(0, dtype=dtype))
            if array.dtype != dtype:
                raise ReplayDataError(f"inline {stored} must have dtype {dtype}, got {array.dtype}")
            values.append(array)
        columns[column] = values
    return columns


def pop_train_columns(rollout_data: dict[str, Any]) -> list[ReplayIndex] | None:
    """Rebuild the replay indexes of a train dict shard, removing their columns."""
    descriptors = rollout_data.pop(REPLAY_INDEX_COLUMN, None)
    if descriptors is None:
        return None
    inline = {
        stored: rollout_data.pop(column)
        for stored, (column, _) in INLINE_TRAIN_COLUMNS.items()
        if column in rollout_data
    }
    return [
        ReplayIndex.from_wire(descriptor, {stored: np.asarray(values[i]) for stored, values in inline.items()})
        for i, descriptor in enumerate(descriptors)
    ]


def stored_field_names(name: str) -> tuple[str, ...]:
    return _LAYOUT_BY_NAME[name].stored


# ================================ helpers ================================


def _check_row_shapes(name: str, segments: Sequence[ReplaySegment]) -> None:
    """R3 and indexer rows of one sample share their [layers, topk] shape; known without reading data."""
    shapes = set()
    for segment in segments:
        if segment.inline is not None:
            shapes.add(tuple(segment.inline[name].shape[1:]))
    if len(shapes) > 1:
        raise ReplayDataError(f"{name} segments of one sample have different row shapes {sorted(shapes)}")


def _append_mask_rows(current: ReplaySegments | None, new: ReplaySegments, *, response_length: int) -> ReplaySegments:
    if current is None:
        if response_length != 0:
            raise ReplayDataError("cannot initialize a sampling mask after response tokens have already been appended")
        return new
    if current.num_rows != response_length:
        raise ReplayDataError(
            f"sampling mask length {current.num_rows} is not aligned with response_length {response_length} "
            "before appending"
        )
    return current.concat(new)


def _rows_after_strip(segments: ReplaySegments, num_tokens: int) -> int:
    rows = segments.num_rows - num_tokens
    if rows < 0:
        raise ReplayDataError(f"cannot strip {num_tokens} tokens from {segments.num_rows} {segments.name} rows")
    return rows


def _num_source_rows(layout: _FieldLayout, arrays: Mapping[str, np.ndarray]) -> int:
    return int(arrays[layout.stored[0]].shape[0])


def _source_arrays(
    segment: ReplaySegment, layout: _FieldLayout, fetched: Mapping[tuple[str, str], np.ndarray]
) -> Mapping[str, np.ndarray]:
    if segment.inline is None:
        raise ReplayDataError(f"{layout.name} segment has no arrays")
    return segment.inline


def _slice_inline(layout: _FieldLayout, segment: ReplaySegment) -> dict[str, np.ndarray]:
    if layout.name == SAMPLING_MASK:
        return _slice_mask(segment.inline, segment)
    (stored,) = layout.stored
    return {stored: segment.inline[stored][segment.start : segment.stop]}


def _slice_mask(arrays: Mapping[str, np.ndarray], segment: ReplaySegment) -> dict[str, np.ndarray]:
    """The segment's rows of a sampling mask in CSR form; support log-probs follow the token IDs."""
    lengths = arrays[MASK_LENGTHS]
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    id_start, id_stop = int(offsets[segment.start]), int(offsets[segment.stop])
    sliced = {MASK_LENGTHS: lengths[segment.start : segment.stop]}
    for name in (MASK_TOKEN_IDS, MASK_LOGPROBS):
        if name in arrays:
            sliced[name] = arrays[name][id_start:id_stop]
    return sliced


def _decode_inline_rows(info: str, *, rows: int, num_layers: int, allow_extra_row: bool) -> np.ndarray:
    # validate=True: SGLang sends plain base64; pybase64 skips its slow non-alphabet filtering pass.
    x = np.frombuffer(pybase64.b64decode(info, validate=True), dtype=np.int32)
    if rows <= 0:
        if len(x):
            raise ReplayDataError(f"inline buffer of {len(x)} values for zero rows")
        return np.empty((0, num_layers, 0), dtype=np.int32)
    topk = len(x) // (rows * num_layers)
    if topk == 0:
        raise ReplayDataError(f"inline buffer of {len(x)} values cannot hold {rows} rows of {num_layers} layers")
    if allow_extra_row and topk and len(x) == (rows + 1) * num_layers * topk:
        x = x[: rows * num_layers * topk]
    if len(x) != rows * num_layers * topk:
        raise ReplayDataError(
            f"inline buffer of {len(x)} values does not hold {rows} rows of {num_layers} layers x {topk}"
        )
    return x.reshape(rows, num_layers, topk)


def _parse_rows_field(
    meta_info: Mapping[str, Any],
    name: str,
    *,
    rows: int,
    num_layers: int | None,
    allow_extra_row: bool,
) -> ReplaySegments | None:
    if (info := meta_info.get(name)) is None:
        return None
    if num_layers is None:
        raise ReplayDataError(f"cannot decode inline {name} without its layer count")
    array = _decode_inline_rows(info, rows=rows, num_layers=num_layers, allow_extra_row=allow_extra_row)
    return ReplaySegments(name, (ReplaySegment(start=0, stop=rows, inline={name: array}),))


def _indexer_num_layers(meta_info: Mapping[str, Any], *, expected: int | None) -> int | None:
    if INDEXER_TOPK in meta_info:
        num_layers = meta_info.get("indexer_topk_num_layers")
        if num_layers is None:
            raise ReplayDataError(
                "Server returned indexer_topk without indexer_topk_num_layers; "
                "sglang-miles must include the layer count in meta_info."
            )
    else:
        return None
    if expected is not None and num_layers != expected:
        raise ReplayDataError(
            f"Server returned indexer_topk with {num_layers} streams but the model has "
            f"{expected} indexer layers; replaying it would map streams to the wrong layers."
        )
    return num_layers


def _parse_sampling_mask(
    meta_info: Mapping[str, Any],
    *,
    support_logprobs: bool,
    num_output_tokens: int,
    aborted: bool,
) -> ReplaySegments:
    if "output_token_sampling_mask" in meta_info:
        segment = _inline_mask_segment(
            meta_info, support_logprobs=support_logprobs, num_output_tokens=num_output_tokens
        )
    elif _is_no_output_abort(meta_info, aborted=aborted, num_output_tokens=num_output_tokens):
        return ReplaySegments(SAMPLING_MASK)
    else:
        raise ReplayDataError(
            "SGLang response is missing output_token_sampling_mask; use an SGLang build with the "
            "native return_sampling_mask primitive"
        )
    return ReplaySegments(SAMPLING_MASK, (segment,))


def _is_no_output_abort(meta_info: Mapping[str, Any], *, aborted: bool, num_output_tokens: int) -> bool:
    finish_reason = meta_info.get("finish_reason") or {}
    return (aborted or finish_reason.get("type") == "abort") and num_output_tokens == 0


def _inline_mask_segment(
    meta_info: Mapping[str, Any], *, support_logprobs: bool, num_output_tokens: int
) -> ReplaySegment:
    supports = meta_info["output_token_sampling_mask"]
    if len(supports) != num_output_tokens:
        raise ReplayDataError(f"sampling support length {len(supports)} != token length {num_output_tokens}")
    lengths = np.fromiter((len(support) for support in supports), dtype=np.int32, count=len(supports))
    token_ids = np.fromiter(chain.from_iterable(supports), dtype=np.int32, count=int(lengths.sum()))
    inline = {MASK_LENGTHS: lengths, MASK_TOKEN_IDS: token_ids}
    if support_logprobs:
        log_probs = meta_info.get("output_token_sampling_logprobs")
        if log_probs is None or len(log_probs) != num_output_tokens:
            raise ReplayDataError("support mode needs output_token_sampling_logprobs with a row per output token")
        for support, row in zip(supports, log_probs, strict=True):
            if not isinstance(row, (list, tuple)) or len(row) != len(support):
                raise ReplayDataError("SGLang support log-probs must align with the sampling support IDs")
        inline[MASK_LOGPROBS] = np.fromiter(chain.from_iterable(log_probs), dtype=np.float32, count=len(token_ids))
    return ReplaySegment(start=0, stop=num_output_tokens, inline=inline)
