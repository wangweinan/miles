"""ReplayIndex: one parser for a response's replay fields, and segment-only rollout edits."""

import numpy as np
import pybase64
import pytest

from miles.utils.replay_index import (
    ReplayDataError,
    ReplayIndex,
    compute_train_columns,
    pop_train_columns,
    response_has_routed_experts,
    sampled_logprobs_from_meta_info,
)

LAYERS, TOPK = 3, 2


def _rows(n: int, start: int = 1) -> np.ndarray:
    return np.arange(start, start + n * LAYERS * TOPK, dtype=np.int32).reshape(n, LAYERS, TOPK)


def _b64(array: np.ndarray) -> str:
    return pybase64.b64encode(np.ascontiguousarray(array, dtype=np.int32).tobytes()).decode("ascii")


def _parse(meta_info, *, rows=4, num_output_tokens=2, support=False, mask=True, extra_row=False):
    return ReplayIndex.from_meta_info(
        meta_info,
        routed_experts_rows=rows,
        num_layers=LAYERS,
        indexer_topk_rows=rows,
        indexer_num_streams=None,
        return_sampling_mask=mask,
        support_logprobs=support,
        num_output_tokens=num_output_tokens,
        aborted=False,
        allow_extra_routed_experts_row=extra_row,
    )


def _response(*, support: bool, start: int = 1) -> dict:
    """A response as SGLang returns it in each log-prob mode."""
    meta_info = {
        "routed_experts": _b64(_rows(4, start)),
        "indexer_topk": _b64(_rows(4, start + 100)),
        "indexer_topk_num_layers": LAYERS,
        "output_token_sampling_mask": [[7, 3], [8]],
    }
    if support:
        meta_info["output_token_sampling_logprobs"] = [[-0.1, -2.0], [-0.3]]
        meta_info["output_token_sampling_selected_logprobs"] = [-0.1, -0.3]
    else:
        meta_info["output_token_sampling_logprobs"] = [-0.1, -0.3]
    return meta_info


def _mask_arrays(index: ReplayIndex) -> list[tuple]:
    return [
        (
            piece.num_rows,
            *(None if a is None else a.tolist() for a in (piece.lengths, piece.token_ids, piece.logprobs)),
        )
        for piece in index.sampling_mask.mask_pieces({})
    ]


@pytest.mark.parametrize("support", [False, True])
def test_response_fields_become_inline_segments(support):
    index = _parse(_response(support=support), support=support)

    np.testing.assert_array_equal(index.routed_experts.materialize_rows({}), _rows(4))
    np.testing.assert_array_equal(index.indexer_topk.materialize_rows({}), _rows(4, 101))
    # Only support mode keeps a log-prob per support entry in the mask.
    logprobs = pytest.approx([-0.1, -2.0, -0.3]) if support else None
    assert _mask_arrays(index) == [(2, [2, 1], [7, 3, 8], logprobs)]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(rows=5), "does not hold 5 rows"),
        (dict(num_output_tokens=3), "sampling support length 2 != token length 3"),
    ],
)
def test_structure_is_checked_on_arrival(kwargs, match):
    with pytest.raises(ReplayDataError, match=match):
        _parse(_response(support=False), **kwargs)


def test_legacy_stop_edge_row_is_dropped():
    routed = _rows(5)
    index = _parse({"routed_experts": _b64(routed)}, mask=False, extra_row=True)
    np.testing.assert_array_equal(index.routed_experts.materialize_rows({}), routed[:4])


def test_edits_rearrange_segments():
    first, second = _parse(_response(support=False)), _parse(_response(support=False, start=50))

    merged = first.merge_turn(second, observation_tokens=3)
    assert merged.routed_experts is second.routed_experts
    assert merged.sampling_mask.num_rows == 2 + 3 + 2
    assert [s.kind for s in merged.sampling_mask.segments] == ["inline", "forced", "inline"]
    stripped = merged.strip_tail(3)
    assert stripped.sampling_mask.num_rows == 4 and stripped.routed_experts.num_rows == 1
    added = first.routed_experts.concat(second.routed_experts).prefix(6)
    assert [(s.start, s.stop) for s in added.segments] == [(0, 4), (0, 2)]
    np.testing.assert_array_equal(added.materialize_rows({}), np.concatenate([_rows(4), _rows(2, 50)]))


def test_wire_round_trip_keeps_inline_and_forced_segments():
    index = _parse(_response(support=False)).append_forced_tokens(2, response_length=2).strip_tail(1)

    descriptor, arrays = index.to_wire()
    restored = ReplayIndex.from_wire(descriptor, arrays)
    assert restored.to_wire()[0] == descriptor
    assert _mask_arrays(restored) == _mask_arrays(index)
    assert [list(entry) for entry in descriptor["sampling_mask"]] == [["inline"], ["forced"]]


def test_train_columns_hold_inline_rows_and_empty_samples():
    indexes = [_parse(_response(support=True), support=True), ReplayIndex()]

    columns = compute_train_columns(indexes)
    assert [len(a) for a in columns["rollout_routed_experts"]] == [4, 0]
    assert columns["rollout_sampling_mask_logprobs"][0].dtype == np.float32
    restored = pop_train_columns(dict(columns))
    np.testing.assert_array_equal(restored[0].routed_experts.materialize_rows({}), _rows(4))
    assert restored[1].is_empty()
    assert compute_train_columns([ReplayIndex(), ReplayIndex()]) == {}


def test_response_has_routed_experts_reads_no_data():
    assert response_has_routed_experts(_response(support=False))
    assert not response_has_routed_experts({"indexer_topk": _b64(_rows(2))})


@pytest.mark.parametrize("support", [False, True])
def test_sampled_logprobs_come_from_the_response(support):
    values = sampled_logprobs_from_meta_info(
        _response(support=support), support_logprobs=support, num_output_tokens=2, aborted=False
    )
    assert values == pytest.approx([-0.1, -0.3])


def test_missing_sampled_logprobs_fail_unless_the_request_aborted_empty():
    with pytest.raises(ReplayDataError, match="missing output_token_sampling_selected_logprobs"):
        sampled_logprobs_from_meta_info({}, support_logprobs=True, num_output_tokens=1, aborted=False)
    assert sampled_logprobs_from_meta_info({}, support_logprobs=False, num_output_tokens=0, aborted=True) == []
