"""Normierungen mit Konstanten (berechnen und anwenden), Nenn-Spannweite, Kendalls tau-b, Platz 1. Nur numpy; eigene Kopie der kleinen Helfer (kein Import über Repos).
Alle Funktionen arbeiten im vorzeichenbehafteten Raum, in dem HÖHER BESSER gilt."""

from __future__ import annotations

import numpy as np

from entscheidungsmatrix_generator import CRITERIA


def nominal_bounds(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Nenn-Spannweite je Kriterium im vorzeichenbehafteten Raum: ↑ [lo, hi], ↓ [-hi, -lo]."""
    lo = np.array([CRITERIA[k][3] if CRITERIA[k][1] > 0 else -CRITERIA[k][4] for k in range(m)], dtype=float)
    hi = np.array([CRITERIA[k][4] if CRITERIA[k][1] > 0 else -CRITERIA[k][3] for k in range(m)], dtype=float)
    return lo, hi


def directions(m: int) -> np.ndarray:
    return np.array([CRITERIA[k][1] for k in range(m)], dtype=float)


def minmax_ref(S: np.ndarray):
    S = np.asarray(S, dtype=float)
    return S.min(axis=0), S.max(axis=0)


def minmax_apply(S: np.ndarray, ref) -> np.ndarray:
    lo, hi = ref
    return (np.asarray(S, dtype=float) - lo) / np.where(hi - lo == 0, 1.0, hi - lo)


def percentile_ref(S: np.ndarray) -> np.ndarray:
    """Referenzstichprobe: die sortierten Werte je Kriterium."""
    return np.sort(np.asarray(S, dtype=float), axis=0)


def percentile_apply(S: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """(mittlerer Rang - 1) / (n_ref - 1) gegen die Referenzstichprobe; für die Referenz selbst die Perzentil-Ränge, für neue Werte der Rang, den sie einnähmen."""
    S = np.asarray(S, dtype=float)
    n = len(ref)
    out = np.empty_like(S, dtype=float)
    for j in range(S.shape[1]):
        left = np.searchsorted(ref[:, j], S[:, j], side="left")
        right = np.searchsorted(ref[:, j], S[:, j], side="right")
        out[:, j] = ((left + right - 1) / 2.0) / (n - 1) if n > 1 else 0.0
    return out


def zscore_ref(S: np.ndarray):
    S = np.asarray(S, dtype=float)
    return S.mean(axis=0), S.std(axis=0)


def zscore_apply(S: np.ndarray, ref) -> np.ndarray:
    mu, sd = ref
    return (np.asarray(S, dtype=float) - mu) / np.where(sd == 0, 1.0, sd)


def vector_ref(S: np.ndarray) -> np.ndarray:
    return np.sqrt((np.asarray(S, dtype=float) ** 2).sum(axis=0))


def vector_apply(S: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.asarray(S, dtype=float) / np.where(ref == 0, 1.0, ref)


def top1(score: np.ndarray) -> int:
    """Index des höchsten Scores; bei Gleichstand die kleinste Nummer."""
    return int(np.argmax(score))


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendalls tau-b (Gleichstände berücksichtigt); 0, wenn eine Seite keine Unterschiede hat."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.sign(a[:, None] - a[None, :])
    sb = np.sign(b[:, None] - b[None, :])
    denom = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return 0.0 if denom == 0 else float((sa * sb).sum() / denom)
