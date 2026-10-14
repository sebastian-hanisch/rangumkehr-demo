"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt."""

import rku_constants as C
import rku_evaluation as ev
import rku_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_and_choices():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["method"] in C.METHODS and p["op"] in C.OPERATIONS and p["decider"] in C.DECIDERS and p["frozen"] in C.FROZEN, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Summe Min-Max, zufällig)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_labels_cover_every_choice():
    assert set(C.METHOD_LABELS) == set(C.METHODS) == set(C.METHOD_SHORT) == set(C.METHOD_CLASS)
    assert set(C.OPERATION_LABELS) == set(C.OPERATIONS) == set(C.OPERATION_SHORT)
    assert set(C.DECIDER_LABELS) == set(C.DECIDERS) == set(C.DECIDER_SHORT) and set(C.FROZEN_LABELS) == set(C.FROZEN)


def test_sweep_seeds_are_the_documented_fixed_set():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200 and C.SWEEP_SEEDS[0] == 200000


def _analysis(name, **override):
    p = dict(C.PRESETS[name])
    p.update(override)
    return ev.analyse(ev.Settings(**p))


def test_every_preset_shows_the_effect_it_is_named_for():
    """Die Presets sind auf einen Seed abgestimmt, bei dem der Effekt sichtbar ist; ändert sich Generator oder Auswertung, soll das hier auffallen."""
    a = _analysis("Standardfall (Summe Min-Max, zufällig)")
    assert a.outcome.flipped.any() and a.outcome.top_old == a.outcome.top_new

    a = _analysis("Ausreißer")
    assert a.settings.method == "sum_minmax" and a.outcome.top_old != a.outcome.top_new and not a.fixed.flipped.any()

    a = _analysis("Klon")
    assert a.settings.method == "ahp_dist" and a.outcome.top_old != a.outcome.top_new and not a.methods["sum_minmax"].flipped.any()

    a = _analysis("Ränge")
    assert a.outcome.flipped.any() and not a.methods["sum_minmax"].flipped.any() and not a.fixed.flipped.any()

    a = _analysis("Feste Referenz")
    assert not a.outcome.flipped.any() and a.decider_flipped.any()
