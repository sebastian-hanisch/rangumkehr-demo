"""Auswertung: Fall der App, Experimente über feste Matrizen (SWEEP_SEEDS) und Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import rku_constants as C
import rku_methods as me
import rku_protocol as pr
import rku_score as sc
from entscheidungsmatrix_generator import SplitMix64, make_matrix


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    method: str = C.DEFAULT_METHOD
    op: str = C.DEFAULT_OPERATION
    decider: str = C.DEFAULT_DECIDER
    frozen: str = C.DEFAULT_FROZEN
    seed: int = C.DEFAULT_SEED


@dataclass
class Case:
    """Alte Menge, neue Menge (alte plus eine Alternative), Gewichte und Nutzen des festen Entscheiders (Güte, linear)."""
    n: int
    m: int
    S_old: np.ndarray
    S_new: np.ndarray
    w: np.ndarray
    goodness: np.ndarray       # Güte der alten Alternativen
    utility: np.ndarray        # fester Nutzen der alten Alternativen


def make_case(n: int, m: int, seed: int, op: str) -> Case:
    """n alte Alternativen und eine neue (die (n + 1)-te Zeile derselben Matrix beim Auslöser 'zufall', sonst aus den alten Zeilen erzeugt)."""
    x = make_matrix(n + 1, m, 0.0, SplitMix64(seed))
    S_all = np.asarray(x.signed, dtype=float)
    S_old = S_all[:n]
    if op == "zufall":
        S_new = S_all
    else:
        z = pr.new_alternative(op, S_old, SplitMix64(seed + C.OFFSET))
        S_new = np.vstack([S_old, z])
    g = np.asarray(x.goodness, dtype=float)[:n]
    return Case(n, m, S_old, S_new, np.asarray(x.weights, dtype=float), g, g @ np.asarray(x.weights, dtype=float))


@dataclass
class Analysis:
    settings: Settings
    case: Case
    outcome: pr.Outcome
    free: pr.Outcome               # dasselbe Verfahren, Konstanten neu aus der Menge
    fixed: pr.Outcome              # dasselbe Verfahren mit den Konstanten der alten Menge (Ursache)
    utility_old: np.ndarray        # Nutzen des gewählten Entscheiders vorher
    utility_new: np.ndarray        # Nutzen des gewählten Entscheiders nachher (alte Alternativen)
    decider_flipped: np.ndarray    # Maske der Paare, die der Entscheider selbst kippt
    toward: int                    # gekippte Paare zum Entscheider hin
    away: int                      # gekippte Paare vom Entscheider weg
    methods: dict                  # alle Verfahren: Outcome (mit der gewählten Einstellung frei / eingefroren)
    ranges_old: np.ndarray         # Min-Max-Grenzen der alten Menge (2 x m)
    ranges_new: np.ndarray


def analyse(settings: Settings) -> Analysis:
    s = settings
    case = make_case(s.n, s.m, s.seed, s.op)
    frozen = s.frozen == "eingefroren"
    methods = {k: pr.run(k, case.S_old, case.S_new, case.w, frozen) for k in C.METHODS}
    out = methods[s.method]
    free = pr.run(s.method, case.S_old, case.S_new, case.w, False)
    fixed = pr.run(s.method, case.S_old, case.S_new, case.w, True)
    if s.decider == "relativ":
        u_old = pr.relative_utility(case.S_old, case.w)
        u_new = pr.relative_utility(case.S_new, case.w)[: s.n]
    else:
        u_old = case.utility
        u_new = case.utility
    dflip = pr.flipped(u_old, u_new)
    toward, away = pr.direction(out, u_old) if s.decider == "fest" else _direction_relative(out, u_old, u_new)
    lo0, hi0 = sc.minmax_ref(case.S_old)
    lo1, hi1 = sc.minmax_ref(case.S_new)
    return Analysis(s, case, out, free, fixed, u_old, u_new, dflip, toward, away, methods, np.vstack([lo0, hi0]), np.vstack([lo1, hi1]))


def _direction_relative(out: pr.Outcome, u_old: np.ndarray, u_new: np.ndarray) -> tuple[int, int]:
    """Beim relativen Entscheider ändert sich sein Nutzen selbst: ein gekipptes Paar ist 'zum Entscheider hin', wenn die neue Ordnung seiner neuen Ordnung folgt."""
    du_new = pr.pair_signs(u_new)
    so = pr.pair_signs(out.old)
    sn = pr.pair_signs(out.new)
    both = out.flipped & (du_new != 0)
    return int((both & (sn == du_new)).sum()), int((both & (so == du_new)).sum())


# ----------------------------------------------------------------------------- Experimente (Protokoll über alle Verfahren)

def experiment_protocol(ops=C.EXP_OPS, n=C.EXP_N, m=C.EXP_M, seeds=C.SWEEP_SEEDS, methods=C.METHODS) -> list[dict]:
    """Je Auslöser und Verfahren: Matrizen mit gekipptem Paar, Anteil gekippter alter Paare, Platz 1 der alten wechselt (frei und mit eingefrorenen Konstanten);
    Richtung der Kipper gegen den festen Entscheider; Paare, die der relative Entscheider selbst kippt und das Verfahren mitvollzieht."""
    rows = []
    for op in ops:
        acc = {k: {f: [] for f in ("any", "share", "top", "any_f", "share_f", "top_f", "toward", "away", "dtau", "m", "both", "m_only", "d_only")} for k in methods}
        for seed in seeds:
            c = make_case(n, m, seed, op)
            u_rel_old = pr.relative_utility(c.S_old, c.w)
            u_rel_new = pr.relative_utility(c.S_new, c.w)[:n]
            dflip = pr.flipped(u_rel_old, u_rel_new)
            for k in methods:
                free = pr.run(k, c.S_old, c.S_new, c.w, False)
                fixed = pr.run(k, c.S_old, c.S_new, c.w, True)
                a = acc[k]
                strict = int((pr.pair_signs(free.old) != 0).sum())
                a["any"].append(bool(free.flipped.any()))
                a["share"].append(free.flipped.sum() / max(1, strict))
                a["top"].append(free.top_old != free.top_new)
                a["any_f"].append(bool(fixed.flipped.any()))
                a["share_f"].append(fixed.flipped.sum() / max(1, strict))
                a["top_f"].append(fixed.top_old != fixed.top_new)
                tw, aw = pr.direction(free, c.utility)
                a["toward"].append(tw)
                a["away"].append(aw)
                a["dtau"].append(sc.kendall_tau_b(free.new, c.utility) - sc.kendall_tau_b(free.old, c.utility))
                a["m"].append(int(free.flipped.sum()))
                a["both"].append(int((free.flipped & dflip).sum()))
                a["m_only"].append(int((free.flipped & ~dflip).sum()))
                a["d_only"].append(int((dflip & ~free.flipped).sum()))
        for k in methods:
            a = acc[k]
            rows.append({
                "op": op, "method": k, "any": float(np.mean(a["any"])), "share": float(np.mean(a["share"])), "top": float(np.mean(a["top"])),
                "any_f": float(np.mean(a["any_f"])), "share_f": float(np.mean(a["share_f"])), "top_f": float(np.mean(a["top_f"])),
                "toward": int(np.sum(a["toward"])), "away": int(np.sum(a["away"])), "dtau": float(np.mean(a["dtau"])),
                "flipped_pairs": int(np.sum(a["m"])), "followed": int(np.sum(a["both"])), "unfollowed": int(np.sum(a["m_only"])), "missed": int(np.sum(a["d_only"])),
                "decider_pairs": int(np.sum(a["both"]) + np.sum(a["d_only"])),
            })
    return rows


# ----------------------------------------------------------------------------- Experiment: feste Referenz, Umkehr gegen Qualität, andere n

REFERENCE_METHODS = C.METHODS + ("topsis_mm",)


def topsis_minmax_sample(S, w, ref=None):
    """TOPSIS auf Min-Max der Stichprobe (Ideal und Antiideal sind dann stets 1 und 0): Gegenstück zu TOPSIS Nenn-Spannweite."""
    ref = sc.minmax_ref(S) if ref is None else ref
    w = np.asarray(w, dtype=float)
    V = sc.minmax_apply(S, ref) * w
    dp = np.sqrt(((V - w) ** 2).sum(axis=1))
    dm = np.sqrt((V ** 2).sum(axis=1))
    return dm / (dp + dm), ref                       # dp + dm >= Norm von w > 0: kein Teiler 0


WORLDS = ("guete_linear", "guete_konkav", "guete_cobb", "rohwerte")


def experiment_reference(n=C.EXP_N, m=C.EXP_M, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Platz 1 getroffen in vier Welten (Güte linear, konkav, Cobb-Douglas, Rohwerte linear auf der Nenn-Spannweite) je Verfahren."""
    lo, hi = sc.nominal_bounds(m)
    hits = {k: {w: [] for w in WORLDS} for k in REFERENCE_METHODS}
    for seed in seeds:
        x = make_matrix(n + 1, m, 0.0, SplitMix64(seed))
        S = np.asarray(x.signed, dtype=float)[:n]
        g = np.asarray(x.goodness, dtype=float)[:n]
        w = np.asarray(x.weights, dtype=float)
        us = {"guete_linear": g @ w, "guete_konkav": np.sqrt(g) @ w, "guete_cobb": np.log(np.clip(g, 1e-12, None)) @ w, "rohwerte": ((S - lo) / (hi - lo)) @ w}
        for k in REFERENCE_METHODS:
            s = topsis_minmax_sample(S, w)[0] if k == "topsis_mm" else me.score(k, S, w)[0]
            for world, u in us.items():
                hits[k][world].append(sc.top1(s) == sc.top1(u))
    rows = []
    for k in REFERENCE_METHODS:
        for world in WORLDS:
            p = float(np.mean(hits[k][world]))
            rows.append({"method": k, "world": world, "hit": p, "se": float(np.sqrt(p * (1 - p) / len(seeds)))})
    return rows


def experiment_sizes(ns=C.EXP_N_OTHER + (C.EXP_N,), m=C.EXP_M, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Zufällige neue Alternative bei n = 8, 20, 40: Matrizen mit gekipptem Paar und Anteil gekippter Paare je Verfahren."""
    rows = []
    for n in ns:
        for r in experiment_protocol(ops=("zufall",), n=n, m=m, seeds=seeds):
            rows.append({"n": n, "method": r["method"], "any": r["any"], "share": r["share"], "top": r["top"]})
    return rows


# ----------------------------------------------------------------------------- Verdikte (live aus den Zeilen)

def _row(rows, **keys):
    return next(r for r in rows if all(r[k] == v for k, v in keys.items()))


def verdict_triggers(rows: list[dict]) -> tuple[str, str]:
    mm_dom = _row(rows, op="dominiert", method="sum_minmax")["any"]
    mm_out = _row(rows, op="ausreisser_gut", method="sum_minmax")
    perz = _row(rows, op="dominiert", method="sum_perz")["any"]
    ahp = _row(rows, op="klon", method="ahp_dist")["any"]
    nom = max(_row(rows, op=op, method="sum_nom")["any"] for op in C.EXP_OPS)
    return "warning", (f"Die Häufigkeit der Umkehr hängt am Auslöser, nicht nur am Verfahren: Bei einer dominierten Alternative kippt die Min-Max-Summe in {de(100 * mm_dom, 1)} % der Matrizen, "
                       f"die Perzentil-Summe in {de(100 * perz, 1)} %; bei einem Klon AHP verteilend in {de(100 * ahp, 1)} %; bei einem Ausreißer in einem Kriterium die Min-Max-Summe in "
                       f"{de(100 * mm_out['any'], 1)} % (Platz 1 der alten wechselt in {de(100 * mm_out['top'], 1)} %). Die feste Referenz kippt in keinem Protokoll "
                       f"({de(100 * nom, 1)} %). Gezählt sind Matrizen mit mindestens einem gekippten Paar, bei {C.EXP_N} alten Alternativen und {C.EXP_M} Kriterien.")


def verdict_sources(rows: list[dict]) -> tuple[str, str]:
    consts = [m for m in C.METHODS if C.METHOD_CLASS[m] == "Konstanten der Menge" or C.METHOD_CLASS[m] == "Ränge"]
    worst_frozen = max(_row(rows, op=op, method=m)["any_f"] for op in C.EXP_OPS for m in consts)
    pair = [m for m in C.METHODS if C.METHOD_CLASS[m] == "Paarzählung"]
    keep = {m: _row(rows, op="zufall", method=m) for m in pair}
    parts = ", ".join(f"{C.METHOD_SHORT[m]} {de(100 * r['any_f'], 1)} % statt {de(100 * r['any'], 1)} %" for m, r in keep.items())
    return "info", (f"Mit den Konstanten der alten Menge kippt bei Summe, TOPSIS und AHP nichts mehr (höchstens {de(100 * worst_frozen, 1)} % der Matrizen): Die Umkehr kommt aus den "
                    f"Konstanten der Menge, nicht aus der Summenform. Bei den Paarzählungen bleibt sie (zufällige neue Alternative, eingefroren: {parts}); sie liegt im Verfahren selbst.")


def verdict_legitimacy(rows: list[dict]) -> tuple[str, str]:
    mm = _row(rows, op="zufall", method="sum_minmax")
    others = [r for r in rows if r["op"] == "zufall" and r["method"] not in ("sum_minmax", "sum_nom", "topsis_nom") and r["flipped_pairs"] > 0]
    lo = min(r["followed"] / r["flipped_pairs"] for r in others)
    hi = max(r["followed"] / r["flipped_pairs"] for r in others)
    nom = _row(rows, op="zufall", method="sum_nom")
    out = _row(rows, op="ausreisser_gut", method="sum_minmax")
    return "warning", (f"Ob eine Umkehr ein Fehler ist, entscheidet der Entscheider: Bewertet er relativ zur Menge, kippt er {mm['decider_pairs']} Paare, und die Min-Max-Summe vollzieht "
                       f"{mm['followed']} davon mit; die übrigen Verfahren vollziehen {de(100 * lo, 1)} bis {de(100 * hi, 1)} % ihrer Kipper mit. Die feste Referenz kippt nichts und verpasst "
                       f"alle {nom['missed']}. Hat der Entscheider einen festen Nutzen, verschlechtert ein Ausreißer die Ordnung der Min-Max-Summe: {out['away']} der {out['toward'] + out['away']} "
                       f"Kipper führen von ihr weg (Δτ {de(out['dtau'], 3)}).")


def verdict_reference(ref: list[dict]) -> tuple[str, str]:
    cost = [_row(ref, method="sum_nom", world=w)["hit"] - _row(ref, method="sum_minmax", world=w)["hit"] for w in WORLDS[:3]]
    t_cost = [_row(ref, method="topsis_nom", world=w)["hit"] - _row(ref, method="topsis_mm", world=w)["hit"] for w in WORLDS[:3]]
    return "info", (f"Die feste Referenz verändert bei der Summe Platz 1 um {de(-100 * max(cost), 1)} bis {de(-100 * min(cost), 1)} Punkte in den drei Güte-Welten (nicht von Null unterscheidbar; "
                    f"Standardfehler 1,3 bis 3,5 Punkte), bei TOPSIS um {de(-100 * max(t_cost), 1)} bis {de(-100 * min(t_cost), 1)} Punkte (nur in der Cobb-Douglas-Welt belastbar). In der Rohwert-Welt ist sie per Definition richtig "
                    "(der Entscheider ist auf der Nenn-Spannweite definiert); sie ist falsch für einen Entscheider, der relativ zur Menge bewertet.")
