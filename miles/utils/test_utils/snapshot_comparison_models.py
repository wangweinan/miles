from pydantic import Field, JsonValue

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class SnapshotFieldMove(FrozenStrictBaseModel):
    source: list[str]
    target: list[str]
    op: str
    reason: str


class SnapshotValue(FrozenStrictBaseModel):
    present: bool
    value: JsonValue


class SnapshotDifference(FrozenStrictBaseModel):
    path: list[str]
    before: SnapshotValue
    after: SnapshotValue


class ApprovedSnapshotDifference(SnapshotDifference):
    human_decision: str
    op: str


class SnapshotListSort(FrozenStrictBaseModel):
    path: list[str]
    human_decision: str
    op: str


class SnapshotComparisonRules(FrozenStrictBaseModel):
    moves: list[SnapshotFieldMove]
    approved_differences: list[ApprovedSnapshotDifference]
    list_sorts: list[SnapshotListSort] = Field(default_factory=list)


class SnapshotComparisonReport(FrozenStrictBaseModel):
    before_directory: str
    after_directory: str
    moves: list[SnapshotFieldMove]
    approved_differences: list[ApprovedSnapshotDifference]
    unexplained: list[SnapshotDifference]
    unused_approvals: list[ApprovedSnapshotDifference]
    list_sorts: list[SnapshotListSort]
    sorting_differences: list[SnapshotDifference]
