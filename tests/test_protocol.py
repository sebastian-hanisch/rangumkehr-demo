"""Protokoll: neue Alternative, gekippte Paare, Ausgang eines Laufs, Richtung, relativer Entscheider (Handwerte)."""

import numpy as np
import pytest

import rku_constants as C
import rku_methods as me
import rku_protocol as pr
from entscheidungsmatrix_generator import SplitMix64


def test_pair_signs_and_pair_index():
    assert pr.pair_signs(np.array([3.0, 1.0, 2.0])).tolist() == [1.0, 1.0, -1.0]        # (0,1) (0,2) (1,2)
    assert pr.pair_index(3) == [(0, 1), (0, 2), (1, 2)]
    assert pr.pair_signs(np.array([1.0, 1.0])).tolist() == [0.0]


def test_flipped_needs_strict_and_opposite_orders():
    old = np.array([3.0, 2.0, 1.0])
    assert pr.flipped(old, np.array([3.0, 2.0, 1.0])).tolist() == [False, False, False]
    assert pr.flipped(old, np.array([2.0, 3.0, 1.0])).tolist() == [True, False, False]
    assert pr.flipped(old, np.array([2.0, 2.0, 1.0])).tolist() == [False, False, False]   # neuer Gleichstand zählt nicht als gekippt
    assert pr.flipped(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])).tolist() == [False, False, False]


def test_borda_example_flips_a_and_c(borda_d):
    V, d, w = borda_d
    out = pr.run("sum_perz", V, np.vstack([V, d]), w)
    assert out.flipped.tolist() == [False, True, False]            # Paare (A,B) (A,C) (B,C): nur A und C
    assert out.top_old == 2 and out.top_new == 0                   # C vorher, A nachher unter den alten
    assert out.new_alt == pytest.approx(2.5 / 3)                   # D hat die Rangsumme 2,5 bei Ränge 0 bis 3


def test_outlier_example_flips_b_and_c_and_the_frozen_constants_do_not(outlier):
    S, d, w = outlier
    out = pr.run("sum_minmax", S, np.vstack([S, d]), w)
    assert out.flipped.tolist() == [False, False, True]            # (A,B) (A,C) (B,C)
    assert out.top_old == 1 and out.top_new == 2
    fixed = pr.run("sum_minmax", S, np.vstack([S, d]), w, frozen=True)
    assert not fixed.flipped.any() and fixed.top_new == 1


def test_clone_example_flips_a_and_b(ahp_clone):
    V, w = ahp_clone
    S = V * np.array([1.0, 1.0])
    f = lambda S_, w_, ref=None: me.ahp_priorities(S_, w_, "dist", ref)       # noqa: E731
    s_old, ref = f(S, w)
    s_new, _ = f(np.vstack([S, S[1]]), w)
    assert pr.flipped(s_old, s_new[:3]).tolist() == [True, False, False]


def test_new_alternative_dominated_clone_and_outlier():
    S = np.array([[10.0, -100.0, 5.0], [30.0, -300.0, 25.0], [20.0, -200.0, 15.0]])
    lo, hi = S.min(axis=0), S.max(axis=0)
    rng = SplitMix64(3)
    k = int(SplitMix64(3).uniform(1)[0] * 3)
    assert pr.new_alternative("klon", S, SplitMix64(3)).tolist() == S[k].tolist()
    dom = pr.new_alternative("dominiert", S, SplitMix64(3))
    assert (dom <= S[k] + 1e-12).all() and (dom >= lo - 1e-12).all()
    assert dom.tolist() == pytest.approx(np.maximum(S[k] - C.DOMINATED_SHARE * (hi - lo), lo).tolist())
    up = pr.new_alternative("ausreisser_gut", S, SplitMix64(3))
    down = pr.new_alternative("ausreisser_schlecht", S, SplitMix64(3))
    j_up = int(np.argmax(up > hi + 1e-9))
    assert up[j_up] == pytest.approx(hi[j_up] + C.OUTLIER_SHARE * (hi[j_up] - lo[j_up]))
    assert (np.abs(up - S[k]) > 1e-12).sum() == 1
    j_dn = int(np.argmax(down < lo - 1e-9))
    assert down[j_dn] == pytest.approx(lo[j_dn] - C.OUTLIER_SHARE * (hi[j_dn] - lo[j_dn]))
    with pytest.raises(ValueError):
        pr.new_alternative("zufall", S, rng)


def test_direction_counts_flips_toward_and_away_from_the_decider(outlier):
    S, d, w = outlier
    out = pr.run("sum_minmax", S, np.vstack([S, d]), w)
    # der Entscheider bevorzugt B vor C: das gekippte Paar (B, C) führt vom Entscheider weg
    assert pr.direction(out, np.array([0.0, 0.9, 0.5])) == (0, 1)
    assert pr.direction(out, np.array([0.0, 0.5, 0.9])) == (1, 0)
    assert pr.direction(out, np.array([0.0, 0.7, 0.7])) == (0, 0)               # Gleichstand beim Entscheider


def test_relative_utility_is_the_minmax_sum_of_the_current_set(outlier):
    S, d, w = outlier
    assert pr.relative_utility(S, w).tolist() == pytest.approx([0.0, 0.65, 0.6])
    assert pr.relative_utility(np.vstack([S, d]), w)[:3].tolist() == pytest.approx([0.0, 0.3167, 0.5333], abs=1e-4)
