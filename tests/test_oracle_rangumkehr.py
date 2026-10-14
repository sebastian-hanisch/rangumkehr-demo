"""Orakel: Kernroutinen gegen einen unabhängigen Rechenweg (DEMO-PLAYBOOK §12).

1. Konstanten eingefroren = keine Umkehr (Summe in vier Normierungen, TOPSIS, AHP): die alten Wertungen sind exakt unverändert.
2. Feste Referenz = nie eine Umkehr; dominierte Alternative und Klon innerhalb der Spannweite kippen die Min-Max-Summe und AHP ideal nie.
3. Perzentil-Summe = PROMETHEE gewöhnlich (Identität aus Stück 7), gleiche gekippte Paare.
4. Entfernen = Hinzufügen rückwärts: dieselben Paare kippen.
5. Der relative Entscheider kippt genau die Paare der Min-Max-Summe.
6. TOPSIS und PROMETHEE gegen Definitionsschleifen.
"""

import numpy as np
import pytest

import rku_constants as C
import rku_evaluation as ev
import rku_methods as me
import rku_protocol as pr
import rku_score as sc

CONSTANT_BASED = ("sum_minmax", "sum_perz", "sum_z", "sum_vec", "topsis_vec", "ahp_dist", "ahp_ideal")
SEEDS = tuple(range(300000, 300060))


def cases(op, n=12, m=5):
    for s in SEEDS:
        yield ev.make_case(n, m, s, op)


def test_frozen_constants_never_flip_for_constant_based_methods():
    for op in C.OPERATIONS:
        for c in cases(op):
            for k in CONSTANT_BASED:
                out = pr.run(k, c.S_old, c.S_new, c.w, frozen=True)
                s_old, _ = me.score(k, c.S_old, c.w)
                assert out.new.tolist() == pytest.approx(s_old.tolist()), (op, k)
                assert not out.flipped.any()


def test_the_fixed_reference_never_flips():
    for op in C.OPERATIONS:
        for c in cases(op):
            for k in ("sum_nom", "topsis_nom"):
                assert not pr.run(k, c.S_old, c.S_new, c.w).flipped.any()


def test_a_dominated_alternative_and_a_clone_leave_minmax_and_ahp_ideal_unchanged():
    for op in ("dominiert", "klon"):
        for c in cases(op):
            for k in ("sum_minmax", "ahp_ideal"):
                out = pr.run(k, c.S_old, c.S_new, c.w)
                assert out.new.tolist() == pytest.approx(out.old.tolist()), (op, k)


def test_percentile_sum_and_promethee_usual_give_the_same_flipped_pairs_and_order():
    for op in C.OPERATIONS:
        for c in cases(op):
            a = pr.run("sum_perz", c.S_old, c.S_new, c.w)
            b = pr.run("prom_usual", c.S_old, c.S_new, c.w)
            assert a.flipped.tolist() == b.flipped.tolist()
            assert pr.pair_signs(a.old).tolist() == pr.pair_signs(b.old).tolist()


def test_removing_an_alternative_flips_the_same_pairs_as_adding_it():
    for c in cases("zufall"):
        for k in ("sum_minmax", "sum_perz", "topsis_vec", "ahp_dist", "prom_v", "electre"):
            added = pr.run(k, c.S_old, c.S_new, c.w)
            s_big, _ = me.score(k, c.S_new, c.w)
            s_small, _ = me.score(k, c.S_old, c.w)
            removed = pr.flipped(s_big[: c.n], s_small)                       # von der großen Menge zur kleinen
            assert removed.tolist() == added.flipped.tolist()


def test_the_relative_decider_flips_exactly_the_pairs_of_the_minmax_sum():
    for op in ("zufall", "ausreisser_gut", "ausreisser_schlecht"):
        for c in cases(op):
            u_old = pr.relative_utility(c.S_old, c.w)
            u_new = pr.relative_utility(c.S_new, c.w)[: c.n]
            out = pr.run("sum_minmax", c.S_old, c.S_new, c.w)
            assert pr.flipped(u_old, u_new).tolist() == out.flipped.tolist()


def test_topsis_follows_the_definition():
    rng = np.random.default_rng(21)
    for _ in range(50):
        S = np.column_stack([rng.random(7) * 90 + 5, -(rng.random(7) * 4000 + 900), rng.random(7) * 90 + 5])
        w = rng.dirichlet(np.ones(3))
        s, _ = me.topsis_vec(S, w)
        norm = np.sqrt((S ** 2).sum(axis=0))
        V = S / norm * w
        ideal, anti = V.max(axis=0), V.min(axis=0)
        for i in range(7):
            dp = np.sqrt(sum((V[i, j] - ideal[j]) ** 2 for j in range(3)))
            dm = np.sqrt(sum((V[i, j] - anti[j]) ** 2 for j in range(3)))
            assert s[i] == pytest.approx(dm / (dp + dm))


def test_promethee_flows_follow_the_definition_loop():
    rng = np.random.default_rng(22)
    for kind, p in (("usual", 1.0), ("v", 0.5)):
        for _ in range(30):
            n, m = int(rng.integers(3, 9)), 4
            N = rng.random((n, m))
            w = rng.dirichlet(np.ones(m))
            flows = me.net_flows(N, w, kind, p)
            for a in range(n):
                plus = minus = 0.0
                for b in range(n):
                    if a == b:
                        continue
                    for j in range(m):
                        d = N[a, j] - N[b, j]
                        pa = (1.0 if d > 0 else 0.0) if kind == "usual" else min(1.0, max(0.0, d / p))
                        pb = (1.0 if -d > 0 else 0.0) if kind == "usual" else min(1.0, max(0.0, -d / p))
                        plus += w[j] * pa
                        minus += w[j] * pb
                assert flows[a] == pytest.approx((plus - minus) / (n - 1))
            assert flows.sum() == pytest.approx(0.0, abs=1e-12)
