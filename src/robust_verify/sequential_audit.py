"""Core state machine for the three-action sequential-audit MVP.

The public state in this module contains only observations that a policy is
allowed to receive: the current (possibly noisy) task labels and responses to
already purchased actions.  Deployment utilities and test-group fields belong
in :mod:`robust_verify.trajectory_oracle` and are intentionally not imported
here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal, Mapping, Sequence

import numpy as np


ActionKind = Literal["verify_task", "annotate_group", "stop"]


def _fraction(value: Fraction | int | float | str) -> Fraction:
    """Convert user-facing numeric costs without binary float accumulation."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


@dataclass(frozen=True)
class AuditAction:
    """One legal batch action.

    ``sample_ids`` are positions in the corresponding action pool.  A stop is
    represented explicitly and has zero cost, making it an absorbing state
    transition rather than an after-the-fact analysis choice.
    """

    kind: ActionKind
    sample_ids: tuple[int, ...] = ()
    cost: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if self.kind not in {"verify_task", "annotate_group", "stop"}:
            raise ValueError(f"unknown action kind: {self.kind}")
        ids = tuple(int(sample_id) for sample_id in self.sample_ids)
        if any(sample_id < 0 for sample_id in ids):
            raise ValueError("sample IDs must be non-negative")
        if len(set(ids)) != len(ids):
            raise ValueError("an action cannot contain duplicate sample IDs")
        object.__setattr__(self, "sample_ids", ids)
        cost = _fraction(self.cost)
        if cost < 0:
            raise ValueError("action cost must be non-negative")
        if self.kind == "stop":
            if ids or cost != 0:
                raise ValueError("stop must have no sample IDs and zero cost")
        elif not ids:
            raise ValueError("continuation actions must contain at least one ID")
        elif cost <= 0:
            raise ValueError("continuation actions must have positive cost")
        object.__setattr__(self, "cost", cost)

    @classmethod
    def stop(cls) -> "AuditAction":
        return cls(kind="stop")


@dataclass(frozen=True)
class AuditEvent:
    """A replayable action and the feedback legally returned for it."""

    round_index: int
    action: AuditAction
    responses: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        responses = tuple(int(response) for response in self.responses)
        if len(responses) != len(self.action.sample_ids):
            raise ValueError("the number of responses must match action IDs")
        if self.action.kind in {"verify_task", "annotate_group"} and any(
            response not in (0, 1) for response in responses
        ):
            raise ValueError("task labels and group attributes must be binary in the MVP")
        object.__setattr__(self, "responses", responses)


@dataclass(frozen=True)
class SequentialAuditConfig:
    """Fixed pools, costs and safety limits for one sequential trajectory."""

    budget: Fraction | int | float | str
    task_label_cost: Fraction | int | float | str
    attribute_cost: Fraction | int | float | str
    task_pool_size: int
    attribute_pool_size: int
    max_rounds: int = 4

    def __post_init__(self) -> None:
        budget = _fraction(self.budget)
        task_cost = _fraction(self.task_label_cost)
        attribute_cost = _fraction(self.attribute_cost)
        if budget < 0 or task_cost <= 0 or attribute_cost <= 0:
            raise ValueError("budget must be non-negative and action costs positive")
        if self.task_pool_size < 0 or self.attribute_pool_size < 0:
            raise ValueError("pool sizes must be non-negative")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "task_label_cost", task_cost)
        object.__setattr__(self, "attribute_cost", attribute_cost)


@dataclass
class SequentialAuditState:
    """Mutable public state; transitions always return a defensive copy."""

    current_labels: np.ndarray
    verified_ids: frozenset[int] = frozenset()
    audited_attributes: dict[int, int] = field(default_factory=dict)
    spent_cost: Fraction = Fraction(0)
    round_index: int = 0
    stopped: bool = False
    transcript: tuple[AuditEvent, ...] = ()

    def __post_init__(self) -> None:
        labels = np.asarray(self.current_labels)
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("current_labels must be a one-dimensional integer array")
        if np.any((labels < 0) | (labels > 1)):
            raise ValueError("current_labels must be binary in the MVP")
        self.current_labels = labels.astype(np.int64, copy=True)
        self.verified_ids = frozenset(int(value) for value in self.verified_ids)
        self.audited_attributes = {
            int(sample_id): int(value) for sample_id, value in self.audited_attributes.items()
        }
        if any(sample_id < 0 for sample_id in self.verified_ids | set(self.audited_attributes)):
            raise ValueError("state IDs must be non-negative")
        if any(value not in (0, 1) for value in self.audited_attributes.values()):
            raise ValueError("audited attributes must be binary")
        self.spent_cost = _fraction(self.spent_cost)
        if self.spent_cost < 0 or self.round_index < 0:
            raise ValueError("state cost and round index must be non-negative")
        self.transcript = tuple(self.transcript)
        self.current_labels.setflags(write=False)

    @classmethod
    def initial(cls, labels: Sequence[int] | np.ndarray) -> "SequentialAuditState":
        return cls(current_labels=np.asarray(labels, dtype=np.int64))

    @property
    def audited_attribute_ids(self) -> frozenset[int]:
        return frozenset(self.audited_attributes)

    def remaining_budget(self, config: SequentialAuditConfig) -> Fraction:
        return config.budget - self.spent_cost


def make_batch_action(
    kind: Literal["verify_task", "annotate_group"],
    sample_ids: Sequence[int],
    config: SequentialAuditConfig,
) -> AuditAction:
    """Create a cost-consistent continuation action."""

    ids = tuple(int(sample_id) for sample_id in sample_ids)
    unit_cost = config.task_label_cost if kind == "verify_task" else config.attribute_cost
    return AuditAction(kind=kind, sample_ids=ids, cost=unit_cost * len(ids))


def available_sample_ids(
    state: SequentialAuditState,
    config: SequentialAuditConfig,
    kind: Literal["verify_task", "annotate_group"],
) -> tuple[int, ...]:
    """Return deterministic, legal IDs for a continuation action."""

    if kind == "verify_task":
        used = state.verified_ids
        size = config.task_pool_size
    else:
        used = state.audited_attribute_ids
        size = config.attribute_pool_size
    return tuple(index for index in range(size) if index not in used)


def apply_action(
    state: SequentialAuditState,
    action: AuditAction,
    config: SequentialAuditConfig,
    *,
    responses: Sequence[int] = (),
) -> SequentialAuditState:
    """Apply one legal action and return the next state.

    The function never mutates the input state.  A stopped state rejects all
    future actions, and each action pool is checked for duplicate queries.
    """

    if state.stopped:
        raise RuntimeError("cannot act after Stop")
    if state.round_index >= config.max_rounds and action.kind != "stop":
        raise RuntimeError("maximum decision depth reached; only Stop is allowed")
    event = AuditEvent(state.round_index, action, tuple(int(value) for value in responses))

    if action.kind == "stop":
        return SequentialAuditState(
            current_labels=state.current_labels,
            verified_ids=state.verified_ids,
            audited_attributes=state.audited_attributes,
            spent_cost=state.spent_cost,
            round_index=state.round_index + 1,
            stopped=True,
            transcript=state.transcript + (event,),
        )

    pool_size = config.task_pool_size if action.kind == "verify_task" else config.attribute_pool_size
    if any(sample_id >= pool_size for sample_id in action.sample_ids):
        raise ValueError(f"{action.kind} contains an ID outside its action pool")
    used = state.verified_ids if action.kind == "verify_task" else state.audited_attribute_ids
    if used.intersection(action.sample_ids):
        raise ValueError(f"{action.kind} repeats an already queried ID")
    expected_cost = (
        config.task_label_cost if action.kind == "verify_task" else config.attribute_cost
    ) * len(action.sample_ids)
    if action.cost != expected_cost:
        raise ValueError(f"{action.kind} cost does not match its batch size")
    if state.spent_cost + action.cost > config.budget:
        raise ValueError("action would exceed the total budget")

    labels = state.current_labels
    verified = state.verified_ids
    attributes = state.audited_attributes
    if action.kind == "verify_task":
        labels = np.array(state.current_labels, copy=True)
        for sample_id, response in zip(action.sample_ids, event.responses):
            labels[sample_id] = response
        labels.setflags(write=False)
        verified = frozenset(verified | set(action.sample_ids))
    else:
        attributes = dict(attributes)
        attributes.update(zip(action.sample_ids, event.responses))

    return SequentialAuditState(
        current_labels=labels,
        verified_ids=verified,
        audited_attributes=attributes,
        spent_cost=state.spent_cost + action.cost,
        round_index=state.round_index + 1,
        stopped=False,
        transcript=state.transcript + (event,),
    )


def replay(
    initial_labels: Sequence[int] | np.ndarray,
    config: SequentialAuditConfig,
    events: Sequence[AuditEvent],
) -> SequentialAuditState:
    """Replay a transcript and verify round ordering and all invariants."""

    state = SequentialAuditState.initial(initial_labels)
    for event in events:
        if event.round_index != state.round_index:
            raise ValueError("transcript round indices are not consecutive")
        state = apply_action(state, event.action, config, responses=event.responses)
    return state


def state_to_dict(state: SequentialAuditState) -> dict[str, object]:
    """Return a JSON-safe, deterministic public-state representation."""

    return {
        "current_labels": state.current_labels.tolist(),
        "verified_ids": sorted(state.verified_ids),
        "audited_attributes": {
            str(sample_id): value for sample_id, value in sorted(state.audited_attributes.items())
        },
        "spent_cost": {
            "numerator": state.spent_cost.numerator,
            "denominator": state.spent_cost.denominator,
        },
        "round_index": state.round_index,
        "stopped": state.stopped,
        "transcript": [
            {
                "round_index": event.round_index,
                "action": {
                    "kind": event.action.kind,
                    "sample_ids": list(event.action.sample_ids),
                    "cost": {
                        "numerator": event.action.cost.numerator,
                        "denominator": event.action.cost.denominator,
                    },
                },
                "responses": list(event.responses),
            }
            for event in state.transcript
        ],
    }


def state_from_dict(payload: Mapping[str, object]) -> SequentialAuditState:
    """Load a state previously emitted by :func:`state_to_dict`."""

    transcript: list[AuditEvent] = []
    for raw_event in payload.get("transcript", []):
        if not isinstance(raw_event, Mapping):
            raise ValueError("invalid transcript event")
        raw_action = raw_event.get("action")
        if not isinstance(raw_action, Mapping):
            raise ValueError("invalid transcript action")
        raw_cost = raw_action.get("cost")
        if not isinstance(raw_cost, Mapping):
            raise ValueError("invalid transcript cost")
        action = AuditAction(
            kind=raw_action["kind"],  # type: ignore[arg-type]
            sample_ids=tuple(raw_action.get("sample_ids", ())),
            cost=Fraction(int(raw_cost["numerator"]), int(raw_cost["denominator"])),
        )
        transcript.append(
            AuditEvent(
                round_index=int(raw_event["round_index"]),
                action=action,
                responses=tuple(raw_event.get("responses", ())),
            )
        )
    raw_spent = payload.get("spent_cost", {})
    if not isinstance(raw_spent, Mapping):
        raise ValueError("invalid spent_cost")
    return SequentialAuditState(
        current_labels=np.asarray(payload.get("current_labels", ()), dtype=np.int64),
        verified_ids=frozenset(payload.get("verified_ids", ())),
        audited_attributes={
            int(sample_id): int(value)
            for sample_id, value in dict(payload.get("audited_attributes", {})).items()
        },
        spent_cost=Fraction(int(raw_spent["numerator"]), int(raw_spent["denominator"])),
        round_index=int(payload.get("round_index", 0)),
        stopped=bool(payload.get("stopped", False)),
        transcript=tuple(transcript),
    )


def state_to_json(state: SequentialAuditState) -> str:
    return json.dumps(state_to_dict(state), sort_keys=True, separators=(",", ":"))


def state_from_json(payload: str) -> SequentialAuditState:
    return state_from_dict(json.loads(payload))
