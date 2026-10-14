"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import rku_constants as C
import rku_protocol as pr
from rku_evaluation import WORLDS, Analysis, de

WORLD_LABELS = {"guete_linear": "Güte/linear", "guete_konkav": "Güte/konkav", "guete_cobb": "Güte/Cobb-Douglas", "rohwerte": "Rohwerte/linear"}


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _ranks(score: np.ndarray) -> np.ndarray:
    """Rang 1 = beste Wertung, Gleichstand nach Nummer."""
    return np.argsort(np.argsort(-np.asarray(score), kind="stable"), kind="stable") + 1


def build_slope(a: Analysis) -> go.Figure:
    """Rang der alten Alternativen vor und nach dem Hinzufügen; rot: beteiligt an einem gekippten Paar."""
    n = a.case.n
    names = alt_names(n)
    o = a.outcome
    r_old, r_new = _ranks(o.old), _ranks(o.new)
    pairs = pr.pair_index(n)
    involved = np.zeros(n, dtype=bool)
    for (i, j), f in zip(pairs, o.flipped):
        if f:
            involved[i] = involved[j] = True
    fig = go.Figure()
    for i in range(n):
        color = C.COLOR_FLIP if involved[i] else C.COLOR_STAY
        fig.add_trace(go.Scatter(x=["vorher", "nachher"], y=[r_old[i], r_new[i]], mode="lines+markers+text", text=[names[i], names[i]], textposition=["middle left", "middle right"],
                                 line=dict(color=color, width=3 if involved[i] else 1.5), marker=dict(size=8), showlegend=False,
                                 hovertemplate=f"{names[i]}: Rang %{{y}}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Rang unter den alten Alternativen", dtick=1)
    fig.update_layout(title=dict(text="Rang vorher und nachher (rot: an einem gekippten Paar beteiligt)", font=dict(size=13)))
    return _fixed(fig, 420)


def flipped_table(a: Analysis) -> pd.DataFrame:
    """Je gekipptem Paar: Ordnung vorher und nachher, Ordnung des Entscheiders und die Richtung."""
    n = a.case.n
    names = alt_names(n)
    o = a.outcome
    u_old, u_new = a.utility_old, a.utility_new
    rows = []
    for (i, j), f in zip(pr.pair_index(n), o.flipped):
        if not f:
            continue
        before = f"{names[i]} vor {names[j]}" if o.old[i] > o.old[j] else f"{names[j]} vor {names[i]}"
        after = f"{names[i]} vor {names[j]}" if o.new[i] > o.new[j] else f"{names[j]} vor {names[i]}"
        if a.settings.decider == "fest":
            ref = u_old
            follows = "zum Entscheider hin" if (o.new[i] - o.new[j]) * (ref[i] - ref[j]) > 0 else "vom Entscheider weg"
            judge = f"{names[i]} vor {names[j]}" if ref[i] > ref[j] else f"{names[j]} vor {names[i]}"
        else:
            ref = u_new
            follows = "folgt dem neuen Nutzen" if (o.new[i] - o.new[j]) * (ref[i] - ref[j]) > 0 else "gegen den neuen Nutzen"
            judge = f"{names[i]} vor {names[j]}" if ref[i] > ref[j] else f"{names[j]} vor {names[i]}"
        rows.append({"Paar": f"{names[i]} / {names[j]}", "vorher": before, "nachher": after, "Entscheider": judge, "Richtung": follows})
    return pd.DataFrame(rows, columns=["Paar", "vorher", "nachher", "Entscheider", "Richtung"])


def constants_table(a: Analysis) -> pd.DataFrame:
    """Min-Max-Grenzen je Kriterium vor und nach dem Hinzufügen und die Änderung der Spannweite."""
    m = a.case.m
    lo0, hi0 = a.ranges_old
    lo1, hi1 = a.ranges_new
    r0, r1 = hi0 - lo0, hi1 - lo1
    return pd.DataFrame({"Kriterium": [f"K{k + 1}" for k in range(m)], "Minimum vorher": lo0, "Maximum vorher": hi0, "Minimum nachher": lo1, "Maximum nachher": hi1,
                         "Spannweite nachher / vorher": np.where(r0 == 0, 1.0, r1 / np.where(r0 == 0, 1.0, r0))})


def style_constants(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 2)) for c in df.columns if c != "Kriterium"})


def method_table(a: Analysis) -> pd.DataFrame:
    n = a.case.n
    rows = []
    for k in C.METHODS:
        o = a.methods[k]
        rows.append({"Verfahren": C.METHOD_LABELS[k], "Klasse": C.METHOD_CLASS[k], "gekippte Paare": int(o.flipped.sum()), "Platz 1 vorher": f"A{o.top_old + 1}",
                     "Platz 1 nachher": f"A{o.top_new + 1}", "Platz 1 wechselt": "ja" if o.top_old != o.top_new else "nein"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Experimente

def trigger_table(rows: list[dict]) -> pd.DataFrame:
    ops = list(dict.fromkeys(r["op"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    out = []
    for k in methods:
        row = {"Verfahren": C.METHOD_SHORT[k], "Klasse": C.METHOD_CLASS[k]}
        for op in ops:
            r = next(x for x in rows if x["op"] == op and x["method"] == k)
            row[C.OPERATION_SHORT[op]] = f"{de(100 * r['any'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)


def build_trigger_heatmap(rows: list[dict]) -> go.Figure:
    ops = list(dict.fromkeys(r["op"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    z = [[100 * next(x for x in rows if x["op"] == op and x["method"] == k)["any"] for op in ops] for k in methods]
    fig = go.Figure(go.Heatmap(z=z, x=[C.OPERATION_SHORT[o] for o in ops], y=[C.METHOD_SHORT[k] for k in methods], zmin=0, zmax=100, colorscale="Reds", text=[[f"{v:.1f}" for v in r] for r in z],
                               texttemplate="%{text}", colorbar=dict(title="%", thickness=10), hovertemplate="%{y}, %{x}: %{z:.1f} % der Matrizen<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=dict(text="Matrizen mit mindestens einem gekippten alten Paar (%)", font=dict(size=13)))
    return _fixed(fig, 440)


def detail_table(rows: list[dict], op: str) -> pd.DataFrame:
    out = []
    for r in rows:
        if r["op"] != op:
            continue
        out.append({"Verfahren": C.METHOD_SHORT[r["method"]], "Matrizen mit Kipp": f"{de(100 * r['any'], 1)} %", "gekippte Paare": f"{de(100 * r['share'], 2)} %",
                    "Platz 1 wechselt": f"{de(100 * r['top'], 1)} %"})
    return pd.DataFrame(out)


def frozen_table(rows: list[dict], ops=("zufall", "dominiert", "klon", "ausreisser_gut")) -> pd.DataFrame:
    methods = list(dict.fromkeys(r["method"] for r in rows))
    out = []
    for k in methods:
        row = {"Verfahren": C.METHOD_SHORT[k], "Klasse": C.METHOD_CLASS[k]}
        for op in ops:
            r = next(x for x in rows if x["op"] == op and x["method"] == k)
            row[C.OPERATION_SHORT[op]] = f"{de(100 * r['any'], 1)} % → {de(100 * r['any_f'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)


def legitimacy_table(rows: list[dict], op: str = "zufall") -> pd.DataFrame:
    out = []
    for r in rows:
        if r["op"] != op:
            continue
        m = r["flipped_pairs"]
        out.append({"Verfahren": C.METHOD_SHORT[r["method"]], "gekippte Paare": m, "vom relativen Entscheider mitvollzogen": f"{r['followed']} ({de(100 * r['followed'] / m, 1)} %)" if m else "-",
                    "ohne den Entscheider": r["unfollowed"], "Kipper des Entscheiders verpasst": r["missed"]})
    return pd.DataFrame(out)


def direction_table(rows: list[dict], op: str) -> pd.DataFrame:
    out = []
    for r in rows:
        if r["op"] != op:
            continue
        t = r["toward"] + r["away"]
        out.append({"Verfahren": C.METHOD_SHORT[r["method"]], "zum Entscheider hin": f"{r['toward']} ({de(100 * r['toward'] / t, 1)} %)" if t else "-",
                    "vom Entscheider weg": f"{r['away']} ({de(100 * r['away'] / t, 1)} %)" if t else "-", "mittleres Δτ": de(r["dtau"], 3)})
    return pd.DataFrame(out)


def reference_table(rows: list[dict]) -> pd.DataFrame:
    methods = list(dict.fromkeys(r["method"] for r in rows))
    out = []
    for k in methods:
        row = {"Verfahren": C.METHOD_SHORT.get(k, "TOPSIS Min-Max, Stichprobe")}
        for w in WORLDS:
            row[WORLD_LABELS[w]] = f"{de(100 * next(x for x in rows if x['method'] == k and x['world'] == w)['hit'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)


def build_quality_scatter(rows: list[dict], proto: list[dict]) -> go.Figure:
    """Umkehr gegen Qualität: Matrizen mit gekipptem Paar bei einer zufälligen neuen Alternative gegen Platz 1 getroffen (Güte/linear)."""
    xs, ys, labels = [], [], []
    for k in C.METHODS:
        h = next(x for x in rows if x["method"] == k and x["world"] == "guete_linear")["hit"]
        a = next(x for x in proto if x["op"] == "zufall" and x["method"] == k)["any"]
        xs.append(100 * a)
        ys.append(100 * h)
        labels.append(C.METHOD_SHORT[k])
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers+text", text=labels, textposition="top center", textfont=dict(size=9), marker=dict(size=10, color=C.COLOR_LINE)))
    fig.update_xaxes(title="Matrizen mit gekipptem Paar (%)", range=[-5, 105])
    fig.update_yaxes(title="Platz 1 getroffen, Güte/linear (%)", range=[30, 90])
    return _fixed(fig, 420)


def size_table(rows: list[dict]) -> pd.DataFrame:
    ns = sorted({r["n"] for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    out = []
    for k in methods:
        row = {"Verfahren": C.METHOD_SHORT[k]}
        for n in ns:
            r = next(x for x in rows if x["n"] == n and x["method"] == k)
            row[f"n = {n}"] = f"{de(100 * r['any'], 1)} % / {de(100 * r['share'], 2)} %"
        out.append(row)
    return pd.DataFrame(out)
