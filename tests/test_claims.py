"""Jede Zahl aus README.md und App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SWEEP_SEEDS (200 feste Matrizen), 20 alte Alternativen, 5 Kriterien, nie über einen einzelnen Seed. Prozentwerte mit Toleranz 0,5 Punkte (Anteil gekippter Paare
0,02), damit Rundung und Bibliotheksversionen (CI installiert ungepinnt) nichts kippen. Der Standardfehler einer Trefferquote ist 1,3 bis 3,5 Punkte: Unterschiede unter etwa 6 Punkten
sind NICHT belastbar und stehen nirgends als Aussage.
"""

import pytest

import rku_constants as C
import rku_evaluation as ev


@pytest.fixture(scope="module")
def proto():
    return ev.experiment_protocol()


@pytest.fixture(scope="module")
def ref():
    return ev.experiment_reference()


@pytest.fixture(scope="module")
def sizes():
    return ev.experiment_sizes()


def cell(rows, **keys):
    hits = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(hits) == 1, (keys, len(hits))
    return hits[0]


def pct(x):
    return 100 * x


# ----------------------------------------------------------------------------- Experiment 1: wer kippt wobei

@pytest.mark.parametrize("method,expected", [
    ("sum_minmax", (13.5, 0.20, 0.5)), ("sum_perz", (89.0, 1.13, 5.0)), ("sum_z", (49.0, 0.49, 2.5)), ("sum_vec", (51.0, 0.53, 2.0)), ("sum_nom", (0.0, 0.0, 0.0)),
    ("topsis_vec", (58.0, 0.67, 1.5)), ("topsis_nom", (0.0, 0.0, 0.0)), ("prom_usual", (89.0, 1.13, 5.0)), ("prom_v", (52.0, 0.46, 3.5)), ("electre", (13.0, 0.09, 3.5)),
    ("ahp_dist", (59.5, 0.63, 3.5)), ("ahp_ideal", (5.5, 0.14, 0.5)),
])
def test_random_new_alternative(proto, method, expected):
    r = cell(proto, op="zufall", method=method)
    assert (pct(r["any"]), pct(r["share"]), pct(r["top"])) == pytest.approx(expected, abs=0.51)


def test_a_dominated_alternative_leaves_the_minmax_sum_and_ahp_ideal_untouched(proto):
    for method in ("sum_minmax", "ahp_ideal", "sum_nom", "topsis_nom"):
        assert cell(proto, op="dominiert", method=method)["any"] == 0.0
    assert [round(pct(cell(proto, op="dominiert", method=m)["any"]), 1) for m in ("sum_perz", "sum_z", "sum_vec", "topsis_vec", "prom_usual", "prom_v", "electre", "ahp_dist")] \
        == pytest.approx([89.5, 50.5, 55.0, 53.5, 89.5, 41.0, 12.0, 60.5], abs=0.51)


def test_a_clone_flips_rank_and_set_based_methods(proto):
    for method in ("sum_minmax", "ahp_ideal", "sum_nom", "topsis_nom"):
        assert cell(proto, op="klon", method=method)["any"] == 0.0
    assert [round(pct(cell(proto, op="klon", method=m)["any"]), 1) for m in ("sum_perz", "sum_z", "sum_vec", "topsis_vec", "prom_usual", "prom_v", "electre", "ahp_dist")] \
        == pytest.approx([85.0, 50.0, 54.5, 56.5, 85.0, 38.5, 5.5, 69.5], abs=0.51)


def test_an_outlier_in_one_criterion_is_the_worst_trigger(proto):
    expected = {"sum_minmax": (88.5, 3.20, 11.0), "sum_perz": (80.0, 0.93, 2.0), "sum_z": (83.0, 1.39, 4.0), "sum_vec": (65.5, 0.84, 1.0), "topsis_vec": (85.5, 2.89, 17.0),
                "prom_usual": (80.0, 0.93, 2.0), "prom_v": (84.5, 1.86, 6.5), "electre": (3.5, 0.05, 1.5), "ahp_dist": (71.0, 0.83, 2.5), "ahp_ideal": (52.0, 1.84, 5.5)}
    for method, e in expected.items():
        r = cell(proto, op="ausreisser_gut", method=method)
        assert (pct(r["any"]), pct(r["share"]), pct(r["top"])) == pytest.approx(e, abs=0.51), method
    for method in ("sum_nom", "topsis_nom"):
        assert cell(proto, op="ausreisser_gut", method=method)["any"] == 0.0


def test_percentile_sum_and_promethee_usual_flip_the_same_pairs(proto):
    for op in C.EXP_OPS:
        a = cell(proto, op=op, method="sum_perz")
        b = cell(proto, op=op, method="prom_usual")
        assert a["any"] == b["any"] and a["share"] == pytest.approx(b["share"]) and a["flipped_pairs"] == b["flipped_pairs"]


# ----------------------------------------------------------------------------- Experiment 2: woher die Umkehr kommt

def test_constant_based_methods_do_not_flip_with_frozen_constants(proto):
    for op in C.EXP_OPS:
        for method in ("sum_minmax", "sum_perz", "sum_z", "sum_vec", "topsis_vec", "ahp_dist", "ahp_ideal"):
            assert cell(proto, op=op, method=method)["any_f"] == 0.0, (op, method)


def test_pair_counting_methods_keep_flipping_with_frozen_constants(proto):
    frozen = lambda m: [round(pct(cell(proto, op=op, method=m)["any_f"]), 1) for op in C.EXP_OPS]  # noqa: E731
    assert frozen("prom_usual") == pytest.approx([89.0, 89.5, 85.0, 80.0], abs=0.51)
    assert frozen("prom_v") == pytest.approx([49.5, 41.0, 38.5, 57.0], abs=0.51)
    assert frozen("electre") == pytest.approx([13.0, 12.0, 5.5, 3.5], abs=0.51)


# ----------------------------------------------------------------------------- Experiment 3: wann ist eine Umkehr ein Fehler

def test_the_relative_decider_flips_what_the_minmax_sum_flips(proto):
    mm = cell(proto, op="zufall", method="sum_minmax")
    assert mm["decider_pairs"] == 75 and mm["followed"] == 75 and mm["flipped_pairs"] == 75 and mm["missed"] == 0
    assert cell(proto, op="ausreisser_gut", method="sum_minmax")["followed"] == cell(proto, op="ausreisser_gut", method="sum_minmax")["flipped_pairs"] == 1216


def test_other_methods_follow_few_of_the_deciders_flips(proto):
    expected = {"sum_perz": (3, 429), "sum_z": (25, 187), "sum_vec": (5, 202), "topsis_vec": (2, 253), "prom_usual": (3, 429), "prom_v": (28, 176), "electre": (0, 35),
                "ahp_dist": (2, 238), "ahp_ideal": (2, 52)}
    for method, (followed, total) in expected.items():
        r = cell(proto, op="zufall", method=method)
        assert (r["followed"], r["flipped_pairs"]) == (followed, total), method
        assert r["missed"] == 75 - followed


def test_the_fixed_reference_misses_all_of_them(proto):
    for method in ("sum_nom", "topsis_nom"):
        r = cell(proto, op="zufall", method=method)
        assert r["flipped_pairs"] == 0 and r["missed"] == 75
    for op in ("dominiert", "klon"):
        assert cell(proto, op=op, method="sum_minmax")["decider_pairs"] == 0
        assert all(r["followed"] == 0 for r in proto if r["op"] == op)


def test_an_outlier_worsens_the_order_for_a_decider_with_a_fixed_utility(proto):
    r = cell(proto, op="ausreisser_gut", method="sum_minmax")
    assert (r["toward"], r["away"]) == (435, 781)
    assert r["dtau"] == pytest.approx(-0.018, abs=0.002)
    ahp = cell(proto, op="ausreisser_gut", method="ahp_ideal")
    assert pct(ahp["away"] / (ahp["toward"] + ahp["away"])) == pytest.approx(65.4, abs=0.5)
    assert max(abs(cell(proto, op=op, method=m)["dtau"]) for op in ("zufall", "dominiert", "klon") for m in C.METHODS) <= 0.0035


# ----------------------------------------------------------------------------- Experiment 4: was die feste Referenz kostet

@pytest.mark.parametrize("method,expected", [
    ("sum_minmax", [74.0, 74.0, 71.0, 95.0]), ("sum_nom", [72.0, 73.0, 70.0, 100.0]), ("topsis_mm", [63.5, 65.0, 63.5, 80.5]), ("topsis_nom", [61.0, 60.5, 57.5, 80.0]),
    ("topsis_vec", [70.5, 74.5, 73.5, 78.5]), ("sum_perz", [75.0, 72.5, 68.0, 60.0]), ("ahp_ideal", [81.0, 73.0, 65.0, 57.0]), ("ahp_dist", [65.5, 60.0, 52.5, 44.5]),
])
def test_first_place_in_four_worlds(ref, method, expected):
    assert [pct(cell(ref, method=method, world=w)["hit"]) for w in ev.WORLDS] == pytest.approx(expected, abs=0.51)


def test_the_nominal_range_is_exactly_right_for_the_raw_value_decider(ref):
    assert cell(ref, method="sum_nom", world="rohwerte")["hit"] == 1.0


def test_flips_shrink_in_share_with_n_and_rise_in_frequency_for_rank_methods(sizes):
    mm = {n: cell(sizes, n=n, method="sum_minmax") for n in (8, 20, 40)}
    pz = {n: cell(sizes, n=n, method="sum_perz") for n in (8, 20, 40)}
    assert [pct(mm[n]["share"]) for n in (8, 20, 40)] == pytest.approx([1.29, 0.20, 0.06], abs=0.02)
    assert [pct(pz[n]["any"]) for n in (8, 20, 40)] == pytest.approx([56.0, 89.0, 96.5], abs=0.51)
    assert [pct(mm[n]["any"]) for n in (8, 20, 40)] == pytest.approx([21.5, 13.5, 10.0], abs=0.51)


# ----------------------------------------------------------------------------- Felder, die die Tabellen und Verdikte zusätzlich benutzen

def test_frozen_shares_and_first_place_changes(proto):
    frozen = lambda op, m, f: round(pct(cell(proto, op=op, method=m)[f]), 2)  # noqa: E731
    assert [frozen(op, "prom_usual", "share_f") for op in C.EXP_OPS] == pytest.approx([1.13, 1.31, 1.02, 0.93], abs=0.02)
    assert [frozen("zufall", m, "share_f") for m in ("prom_v", "electre")] == pytest.approx([0.39, 0.09], abs=0.02)
    assert [frozen(op, "prom_usual", "top_f") for op in C.EXP_OPS] == pytest.approx([5.0, 3.0, 2.5, 2.0], abs=0.51)
    assert [frozen("ausreisser_gut", m, "top_f") for m in ("prom_v", "electre")] == pytest.approx([2.5, 1.5], abs=0.51)
    for m in ("sum_minmax", "sum_perz", "topsis_vec", "ahp_dist", "ahp_ideal"):
        assert frozen("ausreisser_gut", m, "top_f") == 0.0 and frozen("ausreisser_gut", m, "share_f") == 0.0


def test_decider_pairs_are_the_followed_and_the_missed_ones(proto):
    for r in proto:
        assert r["decider_pairs"] == r["followed"] + r["missed"]
        assert r["flipped_pairs"] == r["followed"] + r["unfollowed"]
    assert cell(proto, op="zufall", method="sum_perz")["decider_pairs"] == 75
    assert cell(proto, op="ausreisser_gut", method="sum_perz")["decider_pairs"] == 1216


def test_standard_errors_of_the_hit_rates(ref):
    r = cell(ref, method="sum_minmax", world="guete_linear")
    assert r["hit"] == pytest.approx(0.74) and r["se"] == pytest.approx((0.74 * 0.26 / 200) ** 0.5)
    assert cell(ref, method="sum_nom", world="rohwerte")["se"] == 0.0
