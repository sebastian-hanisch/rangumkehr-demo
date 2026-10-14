"""Protokoll der Rangumkehr: neue Alternative, gekippte Paare, Zerlegung (frei gegen eingefroren), Richtung gegen den Entscheider. Nur numpy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import rku_constants as C
import rku_methods as me
import rku_score as sc
from entscheidungsmatrix_generator import SplitMix64


def new_alternative(op: str, S_old: np.ndarray, rng: SplitMix64) -> np.ndarray:
    """Eine zusätzliche Zeile (vorzeichenbehaftet) zu S_old für die Auslöser dominiert, klon, ausreisser_gut, ausreisser_schlecht (zufall kommt aus der Matrix)."""
    S_old = np.asarray(S_old, dtype=float)
    k = int(rng.uniform(1)[0] * len(S_old))
    base = S_old[k]
    lo, hi = S_old.min(axis=0), S_old.max(axis=0)
    if op == "klon":
        return base.copy()
    if op == "dominiert":
        return np.maximum(base - C.DOMINATED_SHARE * (hi - lo), lo)          # von der Vorlage schwach dominiert, innerhalb der Spannweite
    if op in ("ausreisser_gut", "ausreisser_schlecht"):
        j = int(rng.uniform(1)[0] * S_old.shape[1])
        z = base.copy()
        z[j] = hi[j] + C.OUTLIER_SHARE * (hi[j] - lo[j]) if op == "ausreisser_gut" else lo[j] - C.OUTLIER_SHARE * (hi[j] - lo[j])
        return z
    raise ValueError(f"unbekannter Auslöser: {op}")


def pair_signs(score: np.ndarray) -> np.ndarray:
    """Vorzeichen der Differenzen je Paar i < j (obere Dreiecksreihenfolge): +1 i vor j, -1 j vor i, 0 gleich."""
    score = np.asarray(score, dtype=float)
    iu = np.triu_indices(len(score), 1)
    return np.sign(score[:, None] - score[None, :])[iu]


def flipped(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Maske über die Paare i < j: in beiden Ordnungen strikt, aber in entgegengesetzter Richtung."""
    so, sn = pair_signs(old), pair_signs(new)
    return (so != 0) & (sn != 0) & (so != sn)


def pair_index(n: int) -> list[tuple[int, int]]:
    iu = np.triu_indices(n, 1)
    return list(zip(iu[0].tolist(), iu[1].tolist()))


@dataclass
class Outcome:
    """Ergebnis eines Protokolllaufs für ein Verfahren."""
    old: np.ndarray            # Wertungen der alten Alternativen auf der alten Menge
    new: np.ndarray            # Wertungen der alten Alternativen auf der neuen Menge
    new_alt: float             # Wertung der neuen Alternative
    flipped: np.ndarray        # Maske über die Paare i < j
    top_old: int
    top_new: int               # Platz 1 unter den alten Alternativen nach dem Hinzufügen


def run(method: str, S_old: np.ndarray, S_new: np.ndarray, w: np.ndarray, frozen: bool = False) -> Outcome:
    """Wertungen vorher und nachher (nachher mit den Konstanten der alten Menge, wenn `frozen`), gekippte Paare unter den ALTEN Alternativen."""
    n = len(S_old)
    s_old, ref = me.score(method, S_old, w)
    s_all, _ = me.score(method, S_new, w, ref if frozen else None)
    s_new = s_all[:n]
    return Outcome(s_old, s_new, float(s_all[n]), flipped(s_old, s_new), sc.top1(s_old), sc.top1(s_new))


def direction(out: Outcome, utility_old: np.ndarray) -> tuple[int, int]:
    """(zur Ordnung des Entscheiders hin, von ihr weg): gekippte Paare, die der Entscheider strikt ordnet, nach der Richtung des Kippens."""
    du = pair_signs(utility_old)
    sn = pair_signs(out.new)
    so = pair_signs(out.old)
    both = out.flipped & (du != 0)
    return int((both & (sn == du)).sum()), int((both & (so == du)).sum())


def relative_utility(S: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Nutzen eines Entscheiders, der auf der Min-Max-Skala der aktuellen Menge bewertet."""
    return sc.minmax_apply(S, sc.minmax_ref(S)) @ np.asarray(w, dtype=float)
