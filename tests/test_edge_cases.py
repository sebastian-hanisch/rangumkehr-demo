"""Randfälle der Kernroutinen (DEMO-PLAYBOOK §12): n = 2, Gleichstände, konstante Spalten, Klon, Perzentil bei n = 2, leere Mengen gekippter Paare, inklusive Grenzen."""

import numpy as np
import pytest

import rku_evaluation as ev
import rku_methods as me
import rku_protocol as pr
import rku_score as sc


def test_exactly_two_old_alternatives():
    """n = 2: ein einziges Paar; PROMETHEE teilt durch n - 1 = 1, die Perzentile durch n_ref - 1 = 1."""
    S = np.array([[60.0, -1000.0, 30.0], [40.0, -2000.0, 50.0]])
    w = np.array([0.5, 0.3, 0.2])
    for k in ("sum_perz", "prom_usual", "prom_v", "electre", "topsis_vec", "ahp_dist"):
        s, _ = me.score(k, S, w)
        assert s.shape == (2,) and np.isfinite(s).all(), k
    out = pr.run("sum_perz", S, np.vstack([S, [50.0, -1500.0, 40.0]]), w)
    assert out.flipped.shape == (1,)


def test_a_clone_creates_a_tie_and_ties_do_not_count_as_flips():
    S = np.array([[10.0, -100.0, 5.0], [20.0, -200.0, 15.0], [30.0, -150.0, 25.0]])
    w = np.array([0.4, 0.3, 0.3])
    S_new = np.vstack([S, S[0]])
    s, _ = me.score("sum_minmax", S_new, w)
    assert s[3] == s[0]                                                      # Klon = gleiche Wertung
    out = pr.run("sum_perz", S, S_new, w)
    assert out.top_new in (0, 1, 2)


def test_all_alternatives_equal_gives_no_strict_pair_and_no_flip():
    S = np.tile(np.array([[50.0, -2000.0, 40.0]]), (4, 1))
    w = np.array([0.4, 0.4, 0.2])
    for k in ("sum_minmax", "sum_perz", "sum_z", "sum_vec", "topsis_vec", "prom_usual", "electre", "ahp_dist", "ahp_ideal"):
        out = pr.run(k, S, np.vstack([S, S[0]]), w)
        assert not out.flipped.any(), k
        assert np.isfinite(out.new).all(), k


def test_constant_column_does_not_divide_by_zero():
    S = np.column_stack([np.linspace(10, 90, 6), np.full(6, -500.0), np.linspace(5, 50, 6)])
    w = np.array([0.5, 0.3, 0.2])
    for k in ("sum_minmax", "sum_z", "sum_vec", "topsis_vec", "ahp_dist", "ahp_ideal"):
        s, _ = me.score(k, S, w)
        assert np.isfinite(s).all(), k


def test_a_zero_value_in_an_up_criterion_and_a_cost_criterion_stay_finite():
    S = np.array([[0.0, -800.0, 10.0], [50.0, -1200.0, 0.0], [100.0, -3000.0, 80.0]])
    w = np.array([0.4, 0.4, 0.2])
    for k in ("ahp_dist", "ahp_ideal"):
        assert np.isfinite(me.score(k, S, w)[0]).all()


def test_percentile_with_two_reference_values_and_with_one():
    ref = sc.percentile_ref(np.array([[1.0], [3.0]]))
    assert sc.percentile_apply(np.array([[0.0], [2.0], [4.0]]), ref).ravel().tolist() == pytest.approx([-0.5, 0.5, 1.5])        # außerhalb und zwischen der Referenz
    assert sc.percentile_apply(np.array([[1.0], [3.0]]), ref).ravel().tolist() == [0.0, 1.0]
    assert sc.percentile_apply(np.array([[5.0]]), np.array([[2.0]])).tolist() == [[0.0]]


def test_the_empty_set_of_flipped_pairs_gives_zero_counts_and_no_division_error():
    a = ev.analyse(ev.Settings(method="sum_nom", op="klon", seed=3))
    assert not a.outcome.flipped.any() and a.toward == 0 and a.away == 0
    rows = ev.experiment_protocol(ops=("klon",), methods=("sum_nom",), seeds=range(5))
    assert rows[0]["flipped_pairs"] == 0 and rows[0]["share"] == 0.0


def test_the_outlier_moves_exactly_one_criterion_outside_the_range():
    c = ev.make_case(5, 3, 4, "ausreisser_gut")
    hi = c.S_old.max(axis=0)
    assert c.S_new.shape == (6, 3) and (c.S_new[5] > hi + 1e-12).sum() == 1
    c2 = ev.make_case(5, 3, 4, "ausreisser_schlecht")
    lo = c2.S_old.min(axis=0)
    assert (c2.S_new[5] < lo - 1e-12).sum() == 1


def test_ahp_values_of_a_cost_criterion_with_zero_and_one():
    """↓-Kriterium mit Wert 0 (Kehrwert über die Schranke 1e-9, endlich) und genau 1 (Kehrwert 1, nicht die Schranke)."""
    S = np.array([[40.0, 0.0], [60.0, -1.0], [20.0, -4.0]])
    with np.errstate(all="raise"):
        v = me._ahp_values(S)
    assert v[:, 0].tolist() == [40.0, 60.0, 20.0]
    assert v[0, 1] == pytest.approx(1e9) and v[1, 1] == pytest.approx(1.0) and v[2, 1] == pytest.approx(0.25)
