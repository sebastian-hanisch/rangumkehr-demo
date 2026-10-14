"""Normierungen mit Konstanten, Nenn-Spannweite, Kendalls tau-b, Platz 1 (Handwerte)."""

import numpy as np
import pytest

import rku_score as sc


def test_minmax_ref_and_apply():
    S = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    ref = sc.minmax_ref(S)
    assert sc.minmax_apply(S, ref).tolist() == [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]          # konstante Spalte: lauter Nullen
    assert sc.minmax_apply(np.array([[7.0, 5.0]]), ref).tolist() == [[1.5, 0.0]]              # Wert außerhalb der alten Spannweite


def test_percentile_apply_against_the_reference_sample():
    ref = sc.percentile_ref(np.array([[10.0], [20.0], [20.0], [30.0]]))
    assert sc.percentile_apply(np.array([[10.0], [20.0], [30.0]]), ref).ravel().tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert sc.percentile_apply(np.array([[15.0], [25.0], [99.0]]), ref).ravel().tolist() == pytest.approx([1 / 6, 5 / 6, 7 / 6])        # (links + rechts - 1) / 2 / (n - 1)
    assert sc.percentile_apply(np.array([[1.0]]), np.array([[5.0]])).tolist() == [[0.0]]    # Referenz mit einem Wert


def test_zscore_and_vector_constants():
    S = np.array([[1.0, 4.0], [3.0, 4.0]])
    ref = sc.zscore_ref(S)
    assert sc.zscore_apply(S, ref).tolist() == [[-1.0, 0.0], [1.0, 0.0]]                      # Streuung 0: Teiler 1
    v = sc.vector_ref(np.array([[3.0], [4.0]]))
    assert v.tolist() == [5.0]
    assert sc.vector_apply(np.array([[3.0], [4.0]]), v).ravel().tolist() == pytest.approx([0.6, 0.8])
    assert sc.vector_apply(np.array([[2.0]]), np.array([0.0])).tolist() == [[2.0]]


def test_nominal_bounds_follow_the_direction():
    lo, hi = sc.nominal_bounds(2)
    assert lo.tolist() == [0.0, -5000.0] and hi.tolist() == [100.0, -800.0]
    assert sc.directions(3).tolist() == [1.0, -1.0, 1.0]


def test_top1_takes_the_lowest_index_at_a_tie():
    assert sc.top1(np.array([0.3, 0.9, 0.9])) == 1


def test_kendall_tau_b_by_hand():
    assert sc.kendall_tau_b(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == pytest.approx(1.0)
    assert sc.kendall_tau_b(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)
    assert sc.kendall_tau_b(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) == 0.0
    assert sc.kendall_tau_b(np.array([1.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(2 / np.sqrt(2 * 3))
