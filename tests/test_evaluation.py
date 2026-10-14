"""Auswertung: Fall der App, Grenzfälle und die Verdikt-Texte."""

import numpy as np
import pytest

import rku_constants as C
import rku_evaluation as ev


def test_de_uses_a_decimal_comma():
    assert ev.de(3.14159, 2) == "3,14"
    assert ev.de(2.0, 0) == "2"


def test_make_case_is_deterministic_and_the_random_trigger_is_the_next_row():
    a = ev.make_case(10, 5, 7, "zufall")
    b = ev.make_case(10, 5, 7, "zufall")
    assert (a.S_old == b.S_old).all() and (a.S_new == b.S_new).all() and a.S_new.shape == (11, 5)
    assert (a.S_new[:10] == a.S_old).all()
    k = ev.make_case(10, 5, 7, "klon")
    assert k.S_new.shape == (11, 5) and any((k.S_new[10] == row).all() for row in k.S_old)
    assert a.w.sum() == pytest.approx(1.0) and len(a.utility) == 10


def test_analysis_of_the_default_case_is_consistent():
    a = ev.analyse(ev.Settings())
    n = a.settings.n
    assert len(a.outcome.old) == n and len(a.outcome.new) == n and set(a.methods) == set(C.METHODS)
    assert a.outcome.flipped.shape == (n * (n - 1) // 2,)
    assert a.ranges_old.shape == (2, 5) and (a.ranges_new[0] <= a.ranges_old[0] + 1e-12).all() and (a.ranges_new[1] >= a.ranges_old[1] - 1e-12).all()
    assert a.toward + a.away <= int(a.outcome.flipped.sum())


def test_frozen_setting_changes_the_main_outcome_and_both_variants_are_kept():
    free = ev.analyse(ev.Settings(method="sum_perz", op="dominiert", seed=1, frozen="frei"))
    frz = ev.analyse(ev.Settings(method="sum_perz", op="dominiert", seed=1, frozen="eingefroren"))
    assert free.outcome.flipped.sum() == free.free.flipped.sum() and frz.outcome.flipped.sum() == frz.fixed.flipped.sum()
    assert free.fixed.flipped.sum() == 0 and free.free.flipped.sum() >= 1                    # Perzentil: kippt frei, eingefroren nicht
    assert frz.outcome.flipped.sum() == 0


def test_fixed_reference_never_flips_and_the_relative_decider_does():
    a = ev.analyse(ev.Settings(method="sum_nom", op="ausreisser_gut", decider="relativ", seed=2))
    assert not a.outcome.flipped.any() and a.decider_flipped.any()


def test_every_method_operation_and_decider_analyses_without_error():
    for method in C.METHODS:
        for op in C.OPERATIONS:
            for dec in C.DECIDERS:
                a = ev.analyse(ev.Settings(n=6, m=4, method=method, op=op, decider=dec, seed=3))
                assert 0 <= a.outcome.top_new < 6


def test_two_alternatives_analyse():
    a = ev.analyse(ev.Settings(n=2, m=3, method="prom_usual", op="zufall", seed=1))
    assert a.outcome.flipped.shape == (1,)


def test_experiments_accept_a_small_set_of_seeds():
    seeds = C.SWEEP_SEEDS[:6]
    rows = ev.experiment_protocol(ops=("zufall", "klon"), seeds=seeds, methods=("sum_minmax", "sum_perz"))
    assert len(rows) == 4 and all(0.0 <= r["any"] <= 1.0 and r["followed"] + r["unfollowed"] == r["flipped_pairs"] for r in rows)
    ref = ev.experiment_reference(seeds=seeds)
    assert len(ref) == len(ev.REFERENCE_METHODS) * len(ev.WORLDS)
    sizes = ev.experiment_sizes(ns=(6, 9), seeds=seeds)
    assert {r["n"] for r in sizes} == {6, 9}


def test_verdicts_name_their_numbers():
    proto = ev.experiment_protocol(seeds=C.SWEEP_SEEDS[:30])
    kind, text = ev.verdict_triggers(proto)
    assert kind == "warning" and "hängt am Auslöser" in text and "Ausreißer" in text
    kind, text = ev.verdict_sources(proto)
    assert kind == "info" and "eingefroren" in text and "Paarzählungen" in text
    kind, text = ev.verdict_legitimacy(proto)
    assert kind == "warning" and "entscheidet der Entscheider" in text and "verpasst" in text
    kind, text = ev.verdict_reference(ev.experiment_reference(seeds=C.SWEEP_SEEDS[:30]))
    assert kind == "info" and "feste Referenz kostet" in text and "Rohwert-Welt" in text


# ----------------------------------------------------------------------------- Verdikte an von Hand gesetzten Zeilen (jede Zahl im Text)

def _proto_rows(over=None):
    """Zeilen für alle Auslöser und Verfahren mit Nullen; `over` (Dict) setzt einzelne Felder: over[(op, method)] = {feld: wert}."""
    over = over or {}
    rows = []
    for op in C.EXP_OPS:
        for m in C.METHODS:
            r = {"op": op, "method": m, "any": 0.0, "share": 0.0, "top": 0.0, "any_f": 0.0, "share_f": 0.0, "top_f": 0.0, "toward": 0, "away": 0, "dtau": 0.0,
                 "flipped_pairs": 0, "followed": 0, "unfollowed": 0, "missed": 0, "decider_pairs": 0}
            r.update(over.get((op, m), {}))
            rows.append(r)
    return rows


def test_verdict_triggers_names_each_trigger_value():
    rows = _proto_rows({("dominiert", "sum_minmax"): {"any": 0.0}, ("dominiert", "sum_perz"): {"any": 0.895}, ("klon", "ahp_dist"): {"any": 0.695},
                          ("ausreisser_gut", "sum_minmax"): {"any": 0.885, "top": 0.11}, ("zufall", "sum_nom"): {"any": 0.0}})
    kind, text = ev.verdict_triggers(rows)
    assert kind == "warning"
    assert "dominierten Alternative kippt die Min-Max-Summe in 0,0 % der Matrizen, die Perzentil-Summe in 89,5 %" in text
    assert "AHP verteilend in 69,5 %" in text and "Min-Max-Summe in 88,5 % (Platz 1 der alten wechselt in 11,0 %)" in text
    assert "kippt in keinem Protokoll (0,0 %)" in text
    # die feste Referenz mit einem Wert über 0: das Maximum über die Auslöser steht im Text
    rows = _proto_rows({("klon", "sum_nom"): {"any": 0.025}})
    assert "(2,5 %)" in ev.verdict_triggers(rows)[1]


def test_verdict_sources_takes_the_worst_frozen_value_of_constant_methods_only():
    rows = _proto_rows({("klon", "sum_perz"): {"any_f": 0.02}, ("zufall", "ahp_dist"): {"any_f": 0.01}, ("zufall", "prom_usual"): {"any": 0.89, "any_f": 0.89},
                          ("zufall", "prom_v"): {"any": 0.52, "any_f": 0.495}, ("zufall", "electre"): {"any": 0.13, "any_f": 0.13}})
    kind, text = ev.verdict_sources(rows)
    assert kind == "info" and "(höchstens 2,0 % der Matrizen)" in text
    assert "PROMETHEE gewöhnlich 89,0 % statt 89,0 %" in text and "PROMETHEE V 0,5 49,5 % statt 52,0 %" in text and "ELECTRE Copeland 13,0 % statt 13,0 %" in text
    assert "AHP" not in text.split("eingefroren:")[1]                    # nur die Paarzählungen stehen hinter dem Doppelpunkt


def test_verdict_legitimacy_ranges_over_the_methods_that_flip_something():
    rows = _proto_rows({("zufall", "sum_minmax"): {"flipped_pairs": 75, "followed": 75, "decider_pairs": 75},
                          ("zufall", "prom_v"): {"flipped_pairs": 100, "followed": 25, "decider_pairs": 75, "missed": 50},
                          ("zufall", "sum_perz"): {"flipped_pairs": 200, "followed": 4, "decider_pairs": 75, "missed": 71},
                          ("zufall", "sum_nom"): {"flipped_pairs": 0, "missed": 75, "decider_pairs": 75},
                          ("ausreisser_gut", "sum_minmax"): {"toward": 10, "away": 30, "dtau": -0.018}})
    kind, text = ev.verdict_legitimacy(rows)
    assert kind == "warning"
    assert "kippt er 75 Paare, und die Min-Max-Summe vollzieht 75 davon mit" in text
    assert "vollziehen 2,0 bis 25,0 % ihrer Kipper mit" in text
    assert "verpasst alle 75" in text and "30 der 40 Kipper führen von ihr weg (Δτ -0,018)" in text


def test_verdict_reference_gives_the_costs_in_points():
    rows = []
    hits = {"sum_minmax": (0.74, 0.74, 0.71), "sum_nom": (0.72, 0.73, 0.70), "topsis_mm": (0.635, 0.65, 0.635), "topsis_nom": (0.61, 0.605, 0.575)}
    for m, vals in hits.items():
        for w, v in zip(ev.WORLDS[:3], vals):
            rows.append({"method": m, "world": w, "hit": v, "se": 0.03})
    kind, text = ev.verdict_reference(rows)
    assert kind == "info" and "bei der Summe 1,0 bis 2,0 Punkte" in text and "bei TOPSIS 2,5 bis 6,0 Punkte" in text


def test_direction_toward_and_away_for_the_relative_decider():
    """Der relative Entscheider ist die Min-Max-Summe der aktuellen Menge: jedes gekippte Paar der Min-Max-Summe folgt seiner neuen Ordnung."""
    a = ev.analyse(ev.Settings(method="sum_minmax", op="ausreisser_gut", decider="relativ", seed=13))
    n_flips = int(a.outcome.flipped.sum())
    assert n_flips > 0 and a.toward == n_flips and a.away == 0
    assert (a.decider_flipped == a.outcome.flipped).all()
    for seed in range(40):
        b = ev.analyse(ev.Settings(method="sum_perz", op="ausreisser_gut", decider="relativ", seed=seed))
        if b.away >= 1 and b.toward >= 1:
            break
    assert b.toward >= 1 and b.away >= 1 and b.toward + b.away == int(b.outcome.flipped.sum())      # beide Richtungen kommen vor, keine Lücke
    c = ev.analyse(ev.Settings(method="sum_minmax", op="ausreisser_gut", decider="fest", seed=13))
    assert c.toward + c.away <= int(c.outcome.flipped.sum()) and c.away >= 1 and not (c.decider_flipped.any())


def test_verdict_legitimacy_counts_a_method_with_exactly_one_flipped_pair():
    rows = _proto_rows({("zufall", "sum_minmax"): {"flipped_pairs": 75, "followed": 75, "decider_pairs": 75},
                        ("zufall", "prom_v"): {"flipped_pairs": 1, "followed": 1, "decider_pairs": 75, "missed": 74},
                        ("zufall", "sum_perz"): {"flipped_pairs": 100, "followed": 10, "decider_pairs": 75, "missed": 65},
                        ("zufall", "sum_nom"): {"missed": 75, "decider_pairs": 75}})
    assert "vollziehen 10,0 bis 100,0 % ihrer Kipper mit" in ev.verdict_legitimacy(rows)[1]


def test_topsis_on_minmax_by_hand_and_without_division_by_zero():
    """Min-Max macht aus S die Zeilen (0, 0), (0,5; 0,5), (1, 1); w = 0,5 / 0,5: Ideal (0,5; 0,5), Antiideal (0, 0): C = 0 / 0,5 / 1."""
    S = np.array([[0.0, -2000.0], [5.0, -1500.0], [10.0, -1000.0]])
    s, ref = ev.topsis_minmax_sample(S, np.array([0.5, 0.5]))
    assert s.tolist() == pytest.approx([0.0, 0.5, 1.0])
    with np.errstate(all="raise"):
        eq, _ = ev.topsis_minmax_sample(np.tile(np.array([[50.0, -1500.0]]), (3, 1)), np.array([0.5, 0.5]))
    assert eq.tolist() == [0.0, 0.0, 0.0]                        # alle gleich: Min-Max macht lauter Nullen, das Antiideal
