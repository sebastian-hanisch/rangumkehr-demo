"""PDF-Export des aktuellen Falls (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import rku_constants as C
from rku_evaluation import Analysis

_REPLACE = {"↑": "+", "↓": "-", "→": "->", "↔": "<->", "–": "-", "—": "-", "τ": "tau", "σ": "sigma", "Δ": "Delta", "≥": ">=", "≤": "<=", "−": "-", "€": "EUR", "·": "*"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    n = s.n
    names = [f"A{i + 1}" for i in range(n)]
    o = a.outcome
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Rangumkehr: eine Alternative kommt dazu, die alte Ordnung kippt"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{n} alte Alternativen, {s.m} Kriterien. Verfahren: {C.METHOD_LABELS[s.method]}. Neue Alternative: {C.OPERATION_LABELS[s.op]}. "
                            f"Entscheider: {C.DECIDER_LABELS[s.decider]}. Konstanten: {C.FROZEN_LABELS[s.frozen]}. Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(f"Platz 1 der alten Alternativen: vorher {names[o.top_old]}, nachher {names[o.top_new]}. Gekippte alte Paare: {int(o.flipped.sum())} "
                            f"(zum Entscheider hin: {a.toward}, von ihm weg: {a.away}). Kipper des Entscheiders selbst: {int(a.decider_flipped.sum())}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Verfahren im Vergleich (gekippte Paare, Platz 1 vorher / nachher)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for k in C.METHODS:
        m = a.methods[k]
        pdf.cell(0, 5, _s(f"{C.METHOD_SHORT[k]} ({C.METHOD_CLASS[k]}): {int(m.flipped.sum())} Paare, Platz 1 {names[m.top_old]} -> {names[m.top_new]}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Der Entscheider ist ein simuliertes Orakel, kein Mensch. Quelle: rangumkehr-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
