"""Konstanten: Regler-Grenzen, Verfahren und Auslöser, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 5, 40, 10          # n = Zahl der ALTEN Alternativen; dazu kommt eine neue
M_MIN, M_MAX, DEFAULT_M = 3, 8, 5
SEED_MAX, DEFAULT_SEED = 999999, 9           # je Preset geprüft (tests/test_presets.py)

METHODS = ("sum_minmax", "sum_perz", "sum_z", "sum_vec", "sum_nom", "topsis_vec", "topsis_nom", "prom_usual", "prom_v", "electre", "ahp_dist", "ahp_ideal")
METHOD_LABELS = {
    "sum_minmax": "Summe, Min-Max (Grenzen der Menge)",
    "sum_perz": "Summe, Perzentil-Ränge (Ränge in der Menge)",
    "sum_z": "Summe, z-Wert (Mittel und Streuung der Menge)",
    "sum_vec": "Summe, Vektor-Normierung (Norm der Menge)",
    "sum_nom": "Summe, Nenn-Spannweite (feste Referenz)",
    "topsis_vec": "TOPSIS, Vektor-Normierung, Ideal der Stichprobe",
    "topsis_nom": "TOPSIS, Nenn-Spannweite, Nenn-Ideal (feste Referenz)",
    "prom_usual": "PROMETHEE II, gewöhnliche Funktion (Paarzählung)",
    "prom_v": "PROMETHEE II, V-Form p = 0,5 (Paarfluss)",
    "electre": "ELECTRE I, Copeland-Zahl (c* = 0,6, ohne Veto)",
    "ahp_dist": "AHP verteilend (Spalte auf Summe 1)",
    "ahp_ideal": "AHP ideal (Spalte durch Maximum)",
}
METHOD_SHORT = {
    "sum_minmax": "Summe Min-Max", "sum_perz": "Summe Perzentil", "sum_z": "Summe z-Wert", "sum_vec": "Summe Vektor", "sum_nom": "Summe Nenn-Spannweite",
    "topsis_vec": "TOPSIS Vektor", "topsis_nom": "TOPSIS Nenn", "prom_usual": "PROMETHEE gewöhnlich", "prom_v": "PROMETHEE V 0,5", "electre": "ELECTRE Copeland",
    "ahp_dist": "AHP verteilend", "ahp_ideal": "AHP ideal",
}
# Ursache der Umkehr nach Klasse (Experiment 2)
METHOD_CLASS = {
    "sum_minmax": "Konstanten der Menge", "sum_perz": "Ränge", "sum_z": "Konstanten der Menge", "sum_vec": "Konstanten der Menge", "sum_nom": "feste Referenz",
    "topsis_vec": "Konstanten der Menge", "topsis_nom": "feste Referenz", "prom_usual": "Paarzählung", "prom_v": "Paarzählung", "electre": "Paarzählung",
    "ahp_dist": "Konstanten der Menge", "ahp_ideal": "Konstanten der Menge",
}
DEFAULT_METHOD = "sum_minmax"

OPERATIONS = ("zufall", "dominiert", "klon", "ausreisser_gut", "ausreisser_schlecht")
OPERATION_LABELS = {
    "zufall": "Eine zufällige neue Alternative",
    "dominiert": "Eine dominierte Alternative innerhalb der Spannweite",
    "klon": "Ein Klon einer alten Alternative",
    "ausreisser_gut": "Ein Ausreißer nach oben in einem Kriterium (+50 % Spannweite)",
    "ausreisser_schlecht": "Ein Ausreißer nach unten in einem Kriterium (+50 % Spannweite)",
}
OPERATION_SHORT = {"zufall": "zufällig", "dominiert": "dominiert", "klon": "Klon", "ausreisser_gut": "Ausreißer oben", "ausreisser_schlecht": "Ausreißer unten"}
DEFAULT_OPERATION = "zufall"
DOMINATED_SHARE = 0.05                         # dominierte Alternative: je Kriterium um 5 % der Spannweite schlechter als ihre Vorlage, nicht unter das Minimum
OUTLIER_SHARE = 0.5                            # Ausreißer: ein Kriterium um 50 % der Spannweite über (unter) dem Extrem
OFFSET = 7919                                  # Strom der Auslöser: SplitMix64(seed + OFFSET)

DECIDERS = ("fest", "relativ")
DECIDER_LABELS = {
    "fest": "Fester Nutzen (Güte, linear): hängt nicht an der Menge",
    "relativ": "Relativer Nutzen: Min-Max der aktuellen Menge",
}
DECIDER_SHORT = {"fest": "fester Nutzen", "relativ": "relativer Nutzen"}
DEFAULT_DECIDER = "fest"

FROZEN = ("frei", "eingefroren")
FROZEN_LABELS = {"frei": "Konstanten neu aus der Menge", "eingefroren": "Konstanten der alten Menge beibehalten"}
DEFAULT_FROZEN = "frei"

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
EXP_N = 20                                     # Experimente: 20 alte Alternativen, unabhängig vom Regler
EXP_M = 5
EXP_N_OTHER = (8, 40)
EXP_OPS = ("zufall", "dominiert", "klon", "ausreisser_gut")

COLOR_FLIP = "#c0392b"
COLOR_STAY = "#8A96A6"
COLOR_NEW = "#D68A2E"
COLOR_OK = "#2F6B65"
COLOR_LINE = "#4c78a8"

PRESETS = {
    "Standardfall (Summe Min-Max, zufällig)": {"n": DEFAULT_N, "m": DEFAULT_M, "method": DEFAULT_METHOD, "op": DEFAULT_OPERATION, "decider": DEFAULT_DECIDER,
                                              "frozen": DEFAULT_FROZEN, "seed": DEFAULT_SEED},
    "Ausreißer": {"n": 10, "m": 5, "method": "sum_minmax", "op": "ausreisser_gut", "decider": "fest", "frozen": "frei", "seed": 13},
    "Klon": {"n": 10, "m": 5, "method": "ahp_dist", "op": "klon", "decider": "fest", "frozen": "frei", "seed": 12},
    "Ränge": {"n": 10, "m": 5, "method": "sum_perz", "op": "dominiert", "decider": "fest", "frozen": "frei", "seed": 1},
    "Feste Referenz": {"n": 10, "m": 5, "method": "sum_nom", "op": "ausreisser_gut", "decider": "relativ", "frozen": "frei", "seed": 2},
}
PRESET_HELP = {
    "Standardfall (Summe Min-Max, zufällig)": "10 alte Alternativen, 5 Kriterien, eine zufällige neue Alternative, Min-Max-Summe: ein Paar der alten Alternativen kippt, Platz 1 bleibt.",
    "Ausreißer": "Die neue Alternative hat in einem Kriterium einen Ausreißer: Die Spannweite wächst, die Min-Max-Summe gewichtet dieses Kriterium weniger, und Platz 1 der alten wechselt.",
    "Klon": "Ein Klon einer alten Alternative kommt dazu: Bei AHP mit Spalten auf Summe 1 ändert das jede Priorität; Platz 1 der alten wechselt, ohne dass sich ein Wert geändert hat.",
    "Ränge": "Eine dominierte Alternative innerhalb der Spannweite: Die Min-Max-Summe bleibt, aber die Perzentil-Ränge verschieben sich, und ein Paar kippt.",
    "Feste Referenz": "Nenn-Spannweite als feste Referenz: Nichts kippt. Bewertet der Entscheider aber relativ zur Menge, verpasst das Verfahren seine Kipper.",
}
