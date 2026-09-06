from fractions import Fraction

import pytest

from robust_verify.three_action_theory import (
    CapacityBatchThreeWorldWGATemplate,
    FiniteHorizonBellmanTree,
    FiniteTreeRegretGame,
    PerLabelBatchThreeWorldWGATemplate,
    SHARP_POLICIES,
    SharpThreeWorldWGATemplate,
    ThreeWorldTemplate,
    ThreeWorldWGATemplate,
    verify_capacity_batch_three_world_extension,
    verify_finite_horizon_bellman_recovery,
    verify_per_label_batch_three_world_extension,
    verify_sharp_three_world_wga_template,
    verify_three_world_wga_template,
)


def test_three_world_template_has_opposite_strict_oracles() -> None:
    witness = ThreeWorldTemplate()
    assert witness.oracle_value("R") == 1
    assert witness.oracle_value("G") == 1
    assert witness.oracle_value("S") == 0
    assert witness.utility("R", "R") > witness.utility("R", "G")
    assert witness.utility("G", "G") > witness.utility("G", "R")
    assert witness.utility("S", "S") > witness.utility("S", "R")


def test_three_world_uniform_mixture_is_tight() -> None:
    witness = ThreeWorldTemplate()
    uniform = (Fraction(1, 3),) * 3
    regrets = witness.regrets(uniform)
    assert regrets == {"R": Fraction(2, 3), "G": Fraction(2, 3), "S": Fraction(2, 3)}
    assert witness.minimax_regret() == Fraction(2, 3)
    assert sum(regrets.values()) == 2


def test_three_world_rejects_invalid_mixtures() -> None:
    witness = ThreeWorldTemplate()
    with pytest.raises(ValueError):
        witness.regrets((Fraction(1, 2), Fraction(1, 2)))
    with pytest.raises(ValueError):
        witness.regrets((Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)))


def test_three_world_wga_witness_preserves_xy_and_pairwise_bound() -> None:
    result = verify_three_world_wga_template()
    assert result["deterministic_policy_count"] == 905
    assert result["same_test_xy_marginal"] is True
    assert result["hidden_group_assignment_changes"] is True
    assert result["maximum_normalized_pairwise_sum"] == "1"
    assert result["inherited_pairwise_regret_bound"] == "2400/5251"
    assert result["pairwise_mixture_regrets"]["S"] == "0"


def test_three_world_wga_rejects_underanchored_instance() -> None:
    with pytest.raises(ValueError):
        ThreeWorldWGATemplate(anchor_mass=47)


def test_sharp_three_world_wga_realizes_exact_raw_and_net_tables() -> None:
    result = verify_sharp_three_world_wga_template()
    assert result["scope"] == "finite_batch_state_dependent_menu"
    assert result["deterministic_policy_count"] == 6
    assert result["policies"] == SHARP_POLICIES
    assert result["raw_plan_table"] == {
        "R": {"R": "2/3", "G": "1/3", "S": "0"},
        "G": {"R": "1/3", "G": "2/3", "S": "0"},
        "S": {"R": "0", "G": "0", "S": "0"},
    }
    assert result["net_plan_table"] == {
        "R": {"R": "1/3", "G": "0", "S": "0"},
        "G": {"R": "0", "G": "1/3", "S": "0"},
        "S": {"R": "-1/3", "G": "-1/3", "S": "0"},
    }


def test_sharp_three_world_wga_has_strict_oracles_and_tight_bound() -> None:
    result = verify_sharp_three_world_wga_template()
    assert result["strict_oracle_policies"] == {"R": "R", "G": "G11", "S": "S"}
    assert result["pure_policy_regret_sums"] == {
        "S": "2/3",
        "R": "2/3",
        "G00": "1",
        "G10": "5/6",
        "G01": "5/6",
        "G11": "2/3",
    }
    assert result["regret_sum_lower_bound"] == "2/3"
    assert result["tight_minimax_regret"] == "2/9"
    assert result["tight_mixture_regrets"] == {"R": "2/9", "G": "2/9", "S": "2/9"}
    assert result["two_world_minimax_regret"] == "1/6"


def test_sharp_three_world_wga_enumerates_partial_g_policies() -> None:
    witness = SharpThreeWorldWGATemplate()
    assert witness.expected_normalized_cost("G00") == Fraction(1, 3)
    assert witness.expected_normalized_cost("G10") == Fraction(2, 3)
    assert witness.expected_normalized_cost("G01") == Fraction(2, 3)
    assert witness.expected_normalized_cost("G11") == 1
    assert witness.utility("R", "G10") == Fraction(-1, 18)
    assert witness.utility("G", "G10") == Fraction(1, 9)
    assert witness.utility("S", "G10") == Fraction(-2, 9)


def test_sharp_three_world_logistic_sign_embedding_and_same_xy() -> None:
    witness = SharpThreeWorldWGATemplate()
    supports = {
        witness.test_support_xy(world, response)
        for world in ("R", "G", "S")
        for response in (0, 1)
    }
    assert len(supports) == 1
    initial = dict(witness.training_labels(()))
    repaired = dict(witness.training_labels(("a", "d0")))
    assert initial["a"] == initial["d0"] == initial["D"] == 0
    assert repaired["a"] == repaired["d0"] == 1
    assert repaired["D"] == 0
    signs = witness.logistic_sign_certificate(("a", "d0"))
    assert signs["a"] == signs["d0"] == signs["P"] == 1
    assert signs["D"] == signs["N0"] == signs["N1"] == -1
    result = witness.verify()
    assert result["logistic_dimension"] == 12
    assert result["one_hot_sign_realization"] is True
    assert result["finite_state_local_stability"] is True
    assert result["one_hot_margin"] == pytest.approx(0.4010581375, abs=1e-10)
    assert result["explicit_spectral_radius"] == pytest.approx(0.0731899006, abs=1e-10)
    assert witness.explicit_stability_radius() > 0


def test_sharp_three_world_rejects_parameters_outside_exact_certificate() -> None:
    with pytest.raises(ValueError):
        SharpThreeWorldWGATemplate(anchor_mass=11)
    with pytest.raises(ValueError):
        SharpThreeWorldWGATemplate(cost_multiplier=Fraction(1, 4))
    with pytest.raises(ValueError):
        SharpThreeWorldWGATemplate(l2_coefficient=0)


def test_finite_tree_regret_game_dual_certificate() -> None:
    game = FiniteTreeRegretGame.from_mapping(
        {
            "R": {"S": 0, "R": Fraction(1, 3), "G": 0},
            "G": {"S": 0, "R": 0, "G": Fraction(1, 3)},
            "S": {"S": 0, "R": Fraction(-1, 3), "G": Fraction(-1, 3)},
        }
    )
    assert game.oracle_values() == {
        "R": Fraction(1, 3),
        "G": Fraction(1, 3),
        "S": Fraction(0),
    }
    assert game.certified_minimax_lower_bound() == Fraction(2, 9)
    assert game.weighted_regret_lower_bound(
        {"R": Fraction(1, 3), "G": Fraction(1, 3), "S": Fraction(1, 3)}
    ) == Fraction(2, 9)
    certificate = game.certify_tight_mixture(
        {"S": Fraction(1, 3), "R": Fraction(1, 3), "G": Fraction(1, 3)}
    )
    assert certificate["upper_bound"] == Fraction(2, 9)
    assert certificate["regrets"] == {
        "R": Fraction(2, 9),
        "G": Fraction(2, 9),
        "S": Fraction(2, 9),
    }


def test_finite_tree_regret_game_requires_common_transcript_for_certificate() -> None:
    game = FiniteTreeRegretGame(
        worlds=("R", "G"),
        policies=("p",),
        utility_table=((Fraction(0),), (Fraction(0),)),
        common_transcript=False,
    )
    with pytest.raises(ValueError):
        game.certified_minimax_lower_bound()


def test_capacity_batch_three_world_extension_is_exact() -> None:
    result = verify_capacity_batch_three_world_extension()
    assert result["scope"] == "capacity_priced_arbitrary_batch_menu"
    assert result["root_batch_count"] == 92
    assert result["post_g_batch_option_count"] == 37
    assert result["deterministic_policy_count"] == 1462
    assert result["oracle_values"] == {"R": "1/3", "G": "1/3", "S": "0"}
    assert result["dual_world_weights"] == {"R": "1/5", "G": "2/5", "S": "2/5"}
    assert result["dual_lower_bound"] == "1/5"
    assert result["tight_minimax_regret"] == "1/5"
    assert result["tight_mixture_regrets"] == {
        "R": "1/5",
        "G": "1/5",
        "S": "1/5",
    }


def test_capacity_batch_extension_reuses_sharp_public_support() -> None:
    witness = CapacityBatchThreeWorldWGATemplate()
    assert len(
        {
            witness.witness.test_support_xy(world, response)
            for world in ("R", "G", "S")
            for response in (0, 1)
        }
    ) == 1
    assert witness.witness.raw_wga_for_query("R", ("a", "b"), 0) == Fraction(2, 3)


def test_per_label_batch_three_world_extension_is_exact() -> None:
    result = verify_per_label_batch_three_world_extension()
    assert result["scope"] == "per_label_priced_single_audit_interleaved_menu"
    assert result["repairable_atom_count"] == 7
    assert result["root_batch_count"] == 63
    assert result["post_g_batch_option_count"] == 29
    assert result["deterministic_policy_count"] == 905
    assert result["deterministic_normal_form_count"] == 905
    assert result["interleaved_trace_skeleton_count"] == 1205
    assert result["arbitrary_r_g_ordering"] is True
    assert result["maximum_audit_count"] == 1
    assert result["normal_form_complete"] is True
    assert result["oracle_values"] == {"R": "4/9", "G": "1/3", "S": "0"}
    assert result["oracle_policy_counts"] == {"R": 3, "G": 9, "S": 1}
    assert result["oracle_action_types"] == {"R": "R", "G": "G", "S": "S"}
    assert result["dual_world_weights"] == {
        "R": "1/9",
        "G": "4/9",
        "S": "4/9",
    }
    assert result["dual_lower_bound"] == "16/81"
    assert result["tight_minimax_regret"] == "16/81"
    assert result["tight_mixture_regrets"] == {
        "R": "16/81",
        "G": "16/81",
        "S": "16/81",
    }
    assert result["max_root_resource_slack"] == "0"
    assert result["max_branch_resource_slack"] == "0"


def test_per_label_bridge_geometry_preserves_public_marginal() -> None:
    witness = PerLabelBatchThreeWorldWGATemplate()
    assert len(
        {
            witness.test_support_xy(world, response)
            for world in ("R", "G", "S")
            for response in (0, 1)
        }
    ) == 1
    assert len(
        {
            witness.hidden_group_support(world, response)
            for world in ("R", "G", "S")
            for response in (0, 1)
        }
    ) == 4
    assert witness.raw_wga_for_query("R", ("a", "b"), 0) == Fraction(2, 3)
    assert witness.raw_wga_for_query("G", ("a", "d0"), 0) == Fraction(2, 3)
    assert witness.raw_wga_for_query("G", ("a", "d0"), 1) == 0
    assert witness.raw_wga_for_query("S", ("a", "b", "c"), 0) == 0


def test_per_label_interleaved_traces_have_complete_normal_forms() -> None:
    witness = PerLabelBatchThreeWorldWGATemplate()
    skeletons = witness.interleaved_trace_skeletons
    normal_forms = {
        witness.normal_form_for_interleaved_trace(skeleton)
        for skeleton in skeletons
    }
    advertised = {spec for spec in witness.policy_specs if spec[0] == "G"}
    assert len(skeletons) == 1205
    assert len(normal_forms) == 29**2
    assert normal_forms == advertised
    assert witness.normal_form_for_interleaved_trace(
        (frozenset(("a",)), frozenset(("d0",)), frozenset(("d1",)))
    ) == ("G", frozenset(("a", "d0")), frozenset(("a", "d1")))


def test_per_label_certificate_rejects_calibration_changes() -> None:
    with pytest.raises(ValueError):
        PerLabelBatchThreeWorldWGATemplate(anchor_mass=11)
    with pytest.raises(ValueError):
        PerLabelBatchThreeWorldWGATemplate(cost_multiplier=Fraction(1, 4))


def test_finite_horizon_bellman_recovery_is_exact() -> None:
    result = verify_finite_horizon_bellman_recovery()
    assert result["scope"] == "finite_horizon_common_transcript_bellman_recovery"
    assert result["node_count"] == 4
    assert result["interleaved_actions"] is True
    assert result["multi_branch_feedback"] is True
    assert result["oracle_values"] == {"R": "3/20", "G": "3/20", "S": "0"}
    assert result["regrets"] == {"R": "0", "G": "1/10", "S": "1/20"}
    assert result["expected_error_mass"] == "1/5"
    assert result["trajectory_regret_bound"] == "2/5"
    assert result["factor_two_tight"] is True


def test_finite_horizon_bellman_tree_checks_stop_and_error_assumptions() -> None:
    tree = FiniteHorizonBellmanTree(
        worlds=("w",),
        root="h",
        nodes_by_depth=(("h",),),
        actions={"h": ("S", "R")},
        transitions={},
        rewards={("w", "h", "S"): 0, ("w", "h", "R"): 1},
    )
    q_star, values = tree.optimal_bellman_values()
    assert q_star[("w", "h", "R")] == 1
    assert values[("w", "h")] == 1
    with pytest.raises(AssertionError, match="error certificate fails"):
        tree.certify_greedy_recovery(
            {("h", "S"): 0, ("h", "R"): 0},
            Fraction(0),
        )
    with pytest.raises(ValueError, match="zero remaining value"):
        FiniteHorizonBellmanTree(
            worlds=("w",),
            root="h",
            nodes_by_depth=(("h",),),
            actions={"h": ("S", "R")},
            transitions={},
            rewards={("w", "h", "S"): 1, ("w", "h", "R"): 0},
        )
