import numpy as np

from robust_verify.cavr import (
    allocation_regret,
    audited_panel_value,
    select_candidate,
    soft_group_value,
    wilson_lower_bound,
)


def test_wilson_lower_bound_is_conservative() -> None:
    assert 0.0 < wilson_lower_bound(8, 10, alpha=0.05) < 0.8
    assert wilson_lower_bound(0, 10, alpha=0.05) == 0.0


def test_panel_value_uses_every_expected_group() -> None:
    value = audited_panel_value(
        predictions=np.array([0, 1, 1, 0]),
        labels=np.array([0, 1, 0, 0]),
        groups=np.array([0, 0, 1, 1]),
        expected_groups=(0, 1),
        simultaneous_comparisons=2,
    )
    assert value.wga == 0.5
    assert value.group_count.tolist() == [2, 2]
    assert value.lower_bound <= value.wga


def test_panel_value_marks_missing_group_as_unresolved() -> None:
    value = audited_panel_value(
        predictions=np.array([0, 1]),
        labels=np.array([0, 1]),
        groups=np.array([0, 0]),
        expected_groups=(0, 1),
    )
    assert np.isnan(value.wga)
    assert value.lower_bound == 0.0


def test_selector_breaks_value_ties_toward_fewer_audits() -> None:
    assert select_candidate([0.7, 0.7, 0.6], [20, 0, 10]) == 1


def test_soft_group_value_uses_probabilistic_attribute_weights() -> None:
    value = soft_group_value(
        predictions=np.array([0, 0, 1, 0]),
        labels=np.array([0, 0, 1, 1]),
        attribute_probability=np.array([0.1, 0.9, 0.2, 0.8]),
        num_classes=2,
    )
    assert value.group_mass.shape == (4,)
    assert np.isfinite(value.wga)
    assert value.group_accuracy[0] == 1.0
    assert value.group_accuracy[1] == 1.0
    assert value.group_accuracy[2] > value.group_accuracy[3]


def test_allocation_regret_uses_candidate_oracle() -> None:
    np.testing.assert_allclose(allocation_regret([0.5, 0.7, 0.6], 2), 0.1)
