"""Die zwölf Verfahren: Handwerte der Mini-Instanzen, eingefrorene Konstanten, feste Referenz."""

import numpy as np
import pytest

import rku_constants as C
import rku_methods as me
import rku_score as sc


def test_every_method_has_a_label_a_class_and_a_function():
    assert set(C.METHODS) == set(C.METHOD_LABELS) == set(C.METHOD_SHORT) == set(C.METHOD_CLASS) == set(me.FUNCTIONS)


def test_ahp_distributive_by_hand_with_and_without_the_clone(ahp_clone):
    V, w = ahp_clone
    s, ref = me.ahp_priorities(V, w, "dist")
    assert ref.tolist() == [14.0, 14.0]
    assert s.tolist() == pytest.approx([10 / 28, 11 / 28, 7 / 28])
    s2, _ = me.ahp_priorities(np.vstack([V, V[1]]), w, "dist")
    assert s2.tolist() == pytest.approx([(3 / 23 + 7 / 16) / 2, (9 / 23 + 2 / 16) / 2, (2 / 23 + 5 / 16) / 2, (9 / 23 + 2 / 16) / 2])
    assert np.argsort(-s).tolist() == [1, 0, 2] and np.argsort(-s2[:3]).tolist() == [0, 1, 2]


def test_ahp_ideal_divides_by_the_maximum_and_ignores_a_dominated_clone(ahp_clone):
    V, w = ahp_clone
    s, ref = me.ahp_priorities(V, w, "ideal")
    assert ref.tolist() == [9.0, 7.0]
    assert s.tolist() == pytest.approx([(3 / 9 + 7 / 7) / 2, (9 / 9 + 2 / 7) / 2, (2 / 9 + 5 / 7) / 2])
    s2, _ = me.ahp_priorities(np.vstack([V, V[1]]), w, "ideal")
    assert s2[:3].tolist() == pytest.approx(s.tolist())


def test_percentile_sum_of_the_borda_example(borda_d):
    V, d, w = borda_d
    s, _ = me.sum_perz(V, w)
    assert (s * 2).tolist() == pytest.approx([1.0, 0.8, 1.2])                     # Ränge 0 bis 2, durch n - 1 = 2 geteilt: Summe der Ränge / 2
    s2, _ = me.sum_perz(np.vstack([V, d]), w)
    assert (s2 * 3).tolist() == pytest.approx([1.5, 0.8, 1.2, 2.5])               # Ränge 0 bis 3, durch 3 geteilt
    assert np.argsort(-s).tolist() == [2, 0, 1] and np.argsort(-s2[:3]).tolist() == [0, 2, 1]


def test_minmax_sum_with_an_outlier_by_hand(outlier):
    S, d, w = outlier
    s, ref = me.sum_minmax(S, w)
    assert s.tolist() == pytest.approx([0.0, 0.65, 0.6])
    s2, _ = me.sum_minmax(np.vstack([S, d]), w)
    assert s2[:3].tolist() == pytest.approx([0.0, (1 / 3 + 0.3) / 2, (1 / 15 + 1.0) / 2])
    s3, _ = me.sum_minmax(np.vstack([S, d]), w, ref)                              # eingefroren: die alten Grenzen
    assert s3[:3].tolist() == pytest.approx(s.tolist())
    assert s3[3] == pytest.approx((3.0 + 0.0) / 2)                                # (30 - 0) / 10 = 3 im ersten Kriterium


def test_frozen_constants_leave_old_scores_unchanged_for_constant_based_methods():
    rng = np.random.default_rng(4)
    S_old = rng.random((8, 4)) * 10 + 1
    S_old[:, 1] *= -1
    S_new = np.vstack([S_old, rng.random(4) * 30 - 5])
    w = np.array([0.4, 0.3, 0.2, 0.1])
    for k in ("sum_minmax", "sum_perz", "sum_z", "sum_vec", "topsis_vec", "ahp_dist", "ahp_ideal"):
        s_old, ref = me.score(k, S_old, w)
        s_frozen, _ = me.score(k, S_new, w, ref)
        assert s_frozen[:8].tolist() == pytest.approx(s_old.tolist()), k


def test_fixed_reference_methods_score_each_alternative_alone():
    rng = np.random.default_rng(5)
    S = np.column_stack([rng.random(6) * 100, -(rng.random(6) * 4000 + 900), rng.random(6) * 100])
    w = np.array([0.5, 0.3, 0.2])
    for k in ("sum_nom", "topsis_nom"):
        full, ref = me.score(k, S, w)
        assert ref is None
        part, _ = me.score(k, S[:4], w)
        assert part.tolist() == pytest.approx(full[:4].tolist()), k
        assert me.score(k, S[2:3], w)[0].tolist() == pytest.approx([full[2]])


def test_promethee_usual_equals_the_weighted_rank_sum_up_to_an_affine_map(borda_d):
    V, d, w = borda_d
    N = np.vstack([V, d])
    flows = me.prom_usual(N, w)[0]
    ranks = sc.percentile_apply(N, sc.percentile_ref(N))
    assert flows.tolist() == pytest.approx((2 * (ranks @ w) - 1.0).tolist())


def test_electre_copeland_on_a_three_alternative_example():
    N = np.array([[0.9, 0.9, 0.1], [0.5, 0.5, 0.9], [0.1, 0.1, 0.5]])
    w = np.array([0.4, 0.3, 0.3])
    s, _ = me.electre(N, w, None, 0.6)
    # Konkordanz (Schwelle 0,6): A über B 0,7, A über C 0,7, B über C 1,0; alle Gegenrichtungen höchstens 0,3: A übertrifft B und C, B übertrifft C
    assert s.tolist() == [2, 0, -2]


def test_topsis_methods_return_one_half_when_all_alternatives_are_equal():
    S = np.tile(np.array([[50.0, -2000.0, 40.0]]), (3, 1))
    w = np.array([0.4, 0.4, 0.2])
    assert me.topsis_vec(S, w)[0].tolist() == [0.5, 0.5, 0.5]


def test_ahp_values_invert_cost_criteria_and_keep_benefit_criteria():
    S = np.array([[40.0, -1000.0], [80.0, -500.0]])             # K1 ↑ (Punkte), K2 ↓ (Euro, mit Vorzeichen)
    v = me._ahp_values(S)
    assert v.ravel().tolist() == pytest.approx([40.0, 1 / 1000, 80.0, 1 / 500])
    s, ref = me.ahp_dist(S, np.array([0.5, 0.5]))
    assert s.tolist() == pytest.approx([(40 / 120 + 1 / 3) / 2, (80 / 120 + 2 / 3) / 2])


def test_sum_nominal_by_hand():
    """K1 (Punkte 0 bis 100): 50 -> 0,5; K2 (Euro 800 bis 5000, ↓, mit Vorzeichen): -2000 -> (5000 - 2000)/4200 = 0,7143; w = 0,5 / 0,5."""
    s, ref = me.sum_nom(np.array([[50.0, -2000.0]]), np.array([0.5, 0.5]))
    assert ref is None
    assert s.tolist() == pytest.approx([0.5 * 0.5 + 0.5 * 3000 / 4200])


def test_electre_threshold_is_inclusive_and_ties_are_concordant():
    N = np.array([[1.0, 0.0], [0.0, 1.0]])
    s, _ = me.electre(N, np.array([0.6, 0.4]), None, 0.6)
    assert s.tolist() == [1, -1]                               # Konkordanz genau 0,6 = c*: übertrifft
    s2, _ = me.electre(N, np.array([0.6, 0.4]), None, 0.61)
    assert s2.tolist() == [0, 0]
    ties = np.array([[0.5, 1.0], [0.5, 0.0]])
    s3, _ = me.electre(ties, np.array([0.5, 0.5]), None, 0.6)  # K1 Gleichstand zählt für beide: A über B 1,0, B über A 0,5
    assert s3.tolist() == [1, -1]


def test_degenerate_inputs_raise_no_numerical_warnings():
    S = np.tile(np.array([[50.0, -2000.0, 40.0]]), (3, 1))
    w = np.array([0.4, 0.4, 0.2])
    with np.errstate(all="raise"):
        assert me.topsis_vec(S, w)[0].tolist() == [0.5, 0.5, 0.5]
        assert me.topsis_nom(S, w)[0].tolist() == pytest.approx([me.topsis_nom(S[:1], w)[0][0]] * 3)


def test_promethee_net_flow_of_two_alternatives_with_a_tie():
    """A = (1, 0), B = (1, 1), w = 0,5 / 0,5: gewöhnlich: pi(A, B) = 0, pi(B, A) = 0,5 (K2); Phi = -0,5 / 0,5 (ein Gleichstand zählt in keiner Richtung)."""
    f = me.net_flows(np.array([[1.0, 0.0], [1.0, 1.0]]), np.array([0.5, 0.5]), "usual")
    assert f.tolist() == pytest.approx([-0.5, 0.5])


def test_ahp_priorities_of_a_zero_column_are_zero_and_without_warnings():
    V = np.array([[0.0, 2.0], [0.0, 6.0]])
    with np.errstate(all="raise"):
        s, ref = me.ahp_priorities(V, np.array([0.5, 0.5]), "dist")
    assert ref.tolist() == [0.0, 8.0]
    assert s.tolist() == pytest.approx([0.5 * 0.25, 0.5 * 0.75])
    with np.errstate(all="raise"):
        s2, _ = me.ahp_priorities(V, np.array([0.5, 0.5]), "ideal")
    assert s2.tolist() == pytest.approx([0.5 * 2 / 6, 0.5])
