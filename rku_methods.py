"""Die zwölf Verfahren als reine Funktionen (S, w, ref) -> (Wertungen, ref).

`S` sind die vorzeichenbehafteten Rohwerte (höher ist besser), `w` die Gewichte. `ref` sind die Normierungskonstanten: None heißt, sie werden aus der übergebenen Menge berechnet,
sonst werden sie angewendet (eingefroren). Verfahren mit fester Referenz (Nenn-Spannweite) haben kein ref (None). Höhere Wertung ist besser."""

from __future__ import annotations

import numpy as np

import rku_score as sc


def _weighted(apply_fn, ref_fn):
    def f(S, w, ref=None):
        ref = ref_fn(S) if ref is None else ref
        return apply_fn(S, ref) @ np.asarray(w, dtype=float), ref
    return f


sum_minmax = _weighted(sc.minmax_apply, sc.minmax_ref)
sum_perz = _weighted(sc.percentile_apply, sc.percentile_ref)
sum_z = _weighted(sc.zscore_apply, sc.zscore_ref)
sum_vec = _weighted(sc.vector_apply, sc.vector_ref)


def sum_nom(S, w, ref=None):
    """Summe auf der festen Nenn-Spannweite: jede Alternative wird für sich bewertet."""
    lo, hi = sc.nominal_bounds(np.asarray(S).shape[1])
    return ((np.asarray(S, dtype=float) - lo) / (hi - lo)) @ np.asarray(w, dtype=float), None


def topsis_vec(S, w, ref=None):
    """TOPSIS klassisch: Vektor-Normierung, gewichtet; Ideal und Antiideal aus der Stichprobe; Abstände in der 2-Norm."""
    w = np.asarray(w, dtype=float)
    if ref is None:
        norm = sc.vector_ref(S)
        V = sc.vector_apply(S, norm) * w
        ref = (norm, V.max(axis=0), V.min(axis=0))
    norm, ideal, anti = ref
    V = sc.vector_apply(S, norm) * w
    dp = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    dm = np.sqrt(((V - anti) ** 2).sum(axis=1))
    tot = dp + dm
    return np.where(tot == 0, 0.5, dm / np.where(tot == 0, 1.0, tot)), ref


def topsis_nom(S, w, ref=None):
    """TOPSIS auf der Nenn-Spannweite mit Nenn-Ideal (alle Werte 1) und Nenn-Antiideal (alle 0): feste Referenz."""
    w = np.asarray(w, dtype=float)
    lo, hi = sc.nominal_bounds(np.asarray(S).shape[1])
    V = ((np.asarray(S, dtype=float) - lo) / (hi - lo)) * w
    dp = np.sqrt(((V - w) ** 2).sum(axis=1))
    dm = np.sqrt((V ** 2).sum(axis=1))
    return dm / (dp + dm), None                      # dp + dm >= Norm von w > 0 (Dreiecksungleichung): kein Teiler 0


def net_flows(N: np.ndarray, w: np.ndarray, kind: str, p: float = 1.0) -> np.ndarray:
    """Nettofluss PROMETHEE II: gewöhnliche Funktion (kind = 'usual') oder V-Form mit Schwelle p (Anteil der Spannweite)."""
    N = np.asarray(N, dtype=float)
    n = len(N)
    D = N[:, None, :] - N[None, :, :]
    P = (D > 0).astype(float) if kind == "usual" else np.clip(D / p, 0.0, 1.0)
    pi = (P * np.asarray(w, dtype=float)).sum(axis=-1)
    np.fill_diagonal(pi, 0.0)
    return pi.sum(axis=1) / (n - 1) - pi.sum(axis=0) / (n - 1)


def _promethee(kind: str, p: float = 1.0):
    def f(S, w, ref=None):
        ref = sc.minmax_ref(S) if ref is None else ref
        return net_flows(sc.minmax_apply(S, ref), w, kind, p), ref
    return f


prom_usual = _promethee("usual")
prom_v = _promethee("v", 0.5)


def electre(S, w, ref=None, cstar: float = 0.6):
    """ELECTRE I ohne Veto: a übertrifft b, wenn die Gewichte der Kriterien, in denen a mindestens so gut ist wie b, zusammen mindestens c* betragen; Copeland-Zahl = übertroffen − übertreffend."""
    ref = sc.minmax_ref(S) if ref is None else ref
    N = sc.minmax_apply(S, ref)
    conc = ((N[:, None, :] >= N[None, :, :]) * np.asarray(w, dtype=float)).sum(axis=-1)
    R = conc >= cstar - 1e-12
    np.fill_diagonal(R, False)
    return R.sum(axis=1) - R.sum(axis=0), ref


def _ahp_values(S: np.ndarray) -> np.ndarray:
    """Rohwerte (positiv) je Kriterium: ↑ direkt, ↓ als Kehrwert (Verhältnisskala)."""
    S = np.asarray(S, dtype=float)
    raw = np.abs(S)
    return np.where(sc.directions(S.shape[1]) > 0, raw, 1.0 / np.where(raw == 0, 1e-9, raw))


def ahp_priorities(v: np.ndarray, w: np.ndarray, mode: str, ref=None):
    """Alternativenprioritäten aus positiven Werten je Kriterium: verteilend (Spalte durch Summe) oder ideal (Spalte durch Maximum), gewichtet summiert; (Prioritäten, Konstanten)."""
    v = np.asarray(v, dtype=float)
    if ref is None:
        ref = v.sum(axis=0) if mode == "dist" else v.max(axis=0)
    return (v / np.where(ref == 0, 1.0, ref)) @ np.asarray(w, dtype=float), ref


def _ahp(mode: str):
    def f(S, w, ref=None):
        return ahp_priorities(_ahp_values(S), w, mode, ref)
    return f


ahp_dist = _ahp("dist")
ahp_ideal = _ahp("ideal")

FUNCTIONS = {
    "sum_minmax": sum_minmax, "sum_perz": sum_perz, "sum_z": sum_z, "sum_vec": sum_vec, "sum_nom": sum_nom, "topsis_vec": topsis_vec, "topsis_nom": topsis_nom,
    "prom_usual": prom_usual, "prom_v": prom_v, "electre": electre, "ahp_dist": ahp_dist, "ahp_ideal": ahp_ideal,
}


def score(method: str, S: np.ndarray, w: np.ndarray, ref=None):
    """(Wertungen, ref) des Verfahrens `method`."""
    return FUNCTIONS[method](S, w, ref)
