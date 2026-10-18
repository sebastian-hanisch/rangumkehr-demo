"""Rangumkehr: eine Alternative kommt dazu, die alte Ordnung kippt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Linie Mehrkriterien-Ranking (Kind von AHP, Idee von TOPSIS). Dieselbe neutrale Entscheidungsmatrix; ein simulierter Entscheider kennt seinen Nutzen. Eine Alternative
kommt dazu (zufällig, dominiert, als Klon oder als Ausreißer), und gefragt wird, ob die ALTEN Alternativen untereinander in anderer Ordnung stehen, obwohl sich an ihren Werten nichts
geändert hat: woher das kommt (Konstanten der Menge, Ränge, Paarzählung), und wann es ein Fehler ist (das entscheidet der Entscheider).
Kern: rku_methods.py, rku_protocol.py, rku_score.py; Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import rku_constants as C
import rku_protocol as pr
from rku_evaluation import (
    Settings,
    analyse,
    de,
    experiment_protocol,
    experiment_reference,
    experiment_sizes,
    verdict_legitimacy,
    verdict_reference,
    verdict_sources,
    verdict_triggers,
)
from rku_pdf_export import build_pdf
from rku_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from rku_visualization import (
    alt_names,
    build_quality_scatter,
    build_slope,
    build_trigger_heatmap,
    constants_table,
    detail_table,
    direction_table,
    flipped_table,
    frozen_table,
    legitimacy_table,
    method_table,
    reference_table,
    size_table,
    style_constants,
    trigger_table,
)

st.set_page_config(page_title="Rangumkehr – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Die vier Experimente rechnen …")
def _experiments():
    return {"protocol": experiment_protocol(), "reference": experiment_reference(), "sizes": experiment_sizes()}


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("↕️ Rangumkehr: eine Alternative kommt dazu, die alte Ordnung kippt")
st.markdown(
    """
**Eine Umkehr entsteht dort, wo ein Verfahren die Alternativen mit etwas bewertet, das aus der ganzen Menge berechnet wird: Normierungskonstanten, Ränge, Paarzählungen.** Dies ist das neunte
Stück der Linie **Mehrkriterien-Ranking**: Kommt eine Alternative dazu, ändern sich die Rohwerte der alten nicht, ihre Ordnung untereinander aber vielleicht doch. Die Demo führt ein
**Protokoll** aus (zufällige neue Alternative, dominierte, Klon, Ausreißer), zerlegt die Ursache (Konstanten der Menge gegen das Verfahren selbst) und fragt, **wann eine Umkehr ein Fehler
ist**: Das entscheidet der Entscheider, nicht das Verfahren. Der Entscheider ist simuliert.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Umkehr", expanded=True):
    st.markdown(
        """
1. **Protokoll:** n alte Alternativen, eine kommt dazu. Verglichen wird die Ordnung der **alten** Alternativen untereinander vorher und nachher; ein Paar ist *gekippt*, wenn es in beiden
   Ordnungen strikt, aber entgegengesetzt geordnet ist.
2. **Drei Ursachen:** *Konstanten der Menge* (Min-Max-Grenzen, Mittel und Streuung, Vektornorm, Spaltensumme, Ideal), *Ränge* (die Perzentil-Summe ist Borda) und *Paarzählung*
   (PROMETHEE, ELECTRE). Mit **eingefrorenen** Konstanten der alten Menge verschwindet die erste Ursache: Was dann noch kippt, kommt aus dem Verfahren selbst.
3. **Feste Referenz:** Die Nenn-Spannweite bewertet jede Alternative für sich; es kippt nie etwas.
4. **Fehler oder nicht:** Ein Entscheider mit **festem Nutzen** (Annahme der Demo) ändert seine Ordnung nicht; jede Umkehr kippt dann ohne Grund. Ein Entscheider, der **relativ** zur vorhandenen Spannweite
   bewertet, kippt selbst, wenn die Spannweite sich ändert: Für ihn ist die Umkehr der Min-Max-Summe richtig.
5. **Auslöser:** Die Häufigkeit hängt am Auslöser. Eine dominierte Alternative lässt die Min-Max-Summe unberührt; ein Ausreißer in einem Kriterium kippt sie bei 20 alten Alternativen in neun von zehn, bei 10 in sieben von zehn Matrizen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Matrix**")
    n = st.slider("Alte Alternativen n", *bounds("n_slider"), key="n_slider", help="Wie viele Alternativen vor dem Hinzufügen zur Wahl stehen.")
    m = st.slider("Kriterien m", *bounds("m_slider"), key="m_slider", help="Nach wie vielen Kriterien jede Alternative gemessen ist.")
    st.markdown("**Verfahren und Auslöser**")
    method = st.selectbox("Verfahren", C.METHODS, key="method_select", format_func=lambda k: C.METHOD_LABELS[k],
                          help="Wie die Alternativen bewertet werden. Die Klasse nennt, woher eine Umkehr käme: Konstanten der Menge, Ränge, Paarzählung oder keine (feste Referenz).")
    op = st.selectbox("Neue Alternative", C.OPERATIONS, key="op_select", format_func=lambda k: C.OPERATION_LABELS[k],
                      help="Zufällig: eine weitere Zeile derselben Matrix. Dominiert: schlechter als eine alte, innerhalb der Spannweite. Klon: Kopie einer alten. Ausreißer: ein Kriterium weit außen.")
    frozen = st.selectbox("Konstanten", C.FROZEN, key="frozen_select", format_func=lambda k: C.FROZEN_LABELS[k],
                          help="Eingefroren: Die Normierungskonstanten der alten Menge werden beibehalten. Was dann noch kippt, kommt aus dem Verfahren selbst.")
    st.markdown("**Entscheider (simuliert)**")
    decider = st.selectbox("Entscheider", C.DECIDERS, key="decider_select", format_func=lambda k: C.DECIDER_LABELS[k],
                           help="Fester Nutzen: hängt nicht an der Menge. Relativ: bewertet auf der Min-Max-Skala der aktuellen Menge und kippt selbst, wenn sich die Spannweite ändert.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "method_select": method, "op_select": op, "decider_select": decider, "frozen_select": frozen, "seed_input": int(seed)})

settings = Settings(int(n), int(m), method, op, decider, frozen, int(seed))
a = _analysis(settings)
names = alt_names(settings.n)
o = a.outcome
pairs_total = settings.n * (settings.n - 1) // 2

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Platz 1 der alten Alternativen", f"{names[o.top_old]} → {names[o.top_new]}", delta="wechselt" if o.top_old != o.top_new else "bleibt", delta_color="off",
          help="Beste alte Alternative nach dem Verfahren vorher und nachher (nur unter den alten verglichen).")
c2.metric("Gekippte alte Paare", f"{int(o.flipped.sum())} von {pairs_total}", delta=f"Neue Alternative: Wertung {de(o.new_alt, 3)}", delta_color="off",
          help="Paare der alten Alternativen, die vorher und nachher strikt, aber entgegengesetzt geordnet sind.")
c3.metric("Mit eingefrorenen Konstanten", f"{int(a.fixed.flipped.sum())} gekippt", delta=f"frei: {int(a.free.flipped.sum())}; was bleibt, liegt im Verfahren", delta_color="off",
          help="Gekippte Paare, wenn die Normierungskonstanten der alten Menge beibehalten werden: Was bleibt, kommt nicht aus den Konstanten, sondern aus dem Verfahren.")
c4.metric("Kipper des Entscheiders", f"{int(a.decider_flipped.sum())} Paare", delta=f"Verfahren: {a.toward} hin, {a.away} weg", delta_color="off",
          help="Paare, die der gewählte Entscheider selbst kippt (nur beim relativen Entscheider möglich), und die gekippten Paare des Verfahrens nach ihrer Richtung.")

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(build_slope(a), width="stretch", key="slope_chart")
with right:
    st.markdown("**Normierungskonstanten (Min-Max) vorher und nachher**")
    st.dataframe(style_constants(constants_table(a)), hide_index=True, width="stretch")
    st.caption("Verändert die neue Alternative die Spannweite eines Kriteriums, verändert sich sein wirksames Gewicht in allen Verfahren, die an der Spannweite der Menge normieren.")

tab_p, tab_m = st.tabs(["Gekippte Paare", "Verfahren im Vergleich"])
with tab_p:
    ft = flipped_table(a)
    if len(ft):
        st.dataframe(ft, hide_index=True, width="stretch")
    else:
        st.info("Kein Paar der alten Alternativen kippt (unter den gewählten Einstellungen).")
with tab_m:
    st.dataframe(method_table(a), hide_index=True, width="stretch")
    st.caption("Dieselbe Matrix, dieselbe neue Alternative, alle zwölf Verfahren (mit der gewählten Einstellung der Konstanten).")

if st.button("📄 PDF des aktuellen Falls erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="rangumkehr.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Woher kommt die Umkehr, und ist sie ein Fehler?")
klass = C.METHOD_CLASS[settings.method]
if int(o.flipped.sum()) == 0:
    if klass == "feste Referenz":
        st.info("Nichts kippt: Das Verfahren bewertet jede Alternative für sich gegen eine feste Referenz. "
                + ("Der relative Entscheider kippt aber selbst " + str(int(a.decider_flipped.sum())) + " Paare, die dieses Verfahren verpasst." if int(a.decider_flipped.sum()) else
                   "Für einen Entscheider, der relativ zur Menge bewertet, wäre das ein Mangel."))
    else:
        st.info(f"Bei diesem Auslöser kippt kein Paar ({klass}). Andere Auslöser (Ausreißer, Klon) können das ändern, Experiment 1 zeigt, wie oft.")
else:
    cause = {"Konstanten der Menge": "den Konstanten der Menge", "Ränge": "den Rängen in der Menge", "Paarzählung": "der Paarzählung"}[klass]
    st.warning(f"{int(o.flipped.sum())} Paare kippen; die Ursache liegt bei {cause}. "
               + ("Mit eingefrorenen Konstanten bleiben " + str(int(a.fixed.flipped.sum())) + "." if klass != "Paarzählung" else "Auch mit eingefrorenen Konstanten bleiben "
                  + str(int(a.fixed.flipped.sum())) + ": Das Verfahren zählt selbst gegen die anderen Alternativen."))
    if settings.decider == "fest":
        st.caption(f"Der Entscheider hat einen festen Nutzen: {a.toward} der gekippten Paare führen zu seiner Ordnung hin, {a.away} von ihr weg. Im Mittel ist das Rauschen; ein Ausreißer "
                   "verschlechtert die Ordnung bei mengenabhängigen Min-Max-Normierungen (Experiment 3), nicht bei allen Verfahren.")
    else:
        st.caption(f"Der Entscheider bewertet relativ zur Menge und kippt selbst {int(a.decider_flipped.sum())} Paare; von den {int(o.flipped.sum())} gekippten Paaren des Verfahrens folgen "
                   f"{a.toward} seiner neuen Ordnung und {a.away} nicht.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die vier Experimente")
st.caption("200 feste Matrizen (SWEEP_SEEDS), 20 alte Alternativen, 5 Kriterien, unabhängig von den Reglern. Die Experimente rechnen zusammen rund 15 Sekunden und starten erst auf Klick. "
           "Standardfehler 1,3 bis 3,5 Punkte: Unterschiede unter etwa 6 Punkten sind nicht belastbar.")
if st.button("▶ Experimente rechnen", key="exp_start"):
    st.session_state["exp_rows"] = _experiments()
exp = st.session_state.get("exp_rows")
if exp:
    proto = exp["protocol"]
    st.markdown("### Experiment 1: Wer kippt wobei")
    st.caption("Matrizen mit mindestens einem gekippten alten Paar, je Verfahren und Auslöser. Eine Quote ohne Auslöser ist irreführend.")
    st.plotly_chart(build_trigger_heatmap(proto), width="stretch", key="trigger_chart")
    verdict(verdict_triggers(proto))
    st.dataframe(trigger_table(proto), hide_index=True, width="stretch")
    st.markdown("**Ausreißer in einem Kriterium** (Matrizen mit Kipp / Anteil gekippter alter Paare / Platz 1 der alten wechselt)")
    st.dataframe(detail_table(proto, "ausreisser_gut"), hide_index=True, width="stretch")

    st.markdown("### Experiment 2: Woher die Umkehr kommt")
    st.caption("Matrizen mit gekipptem Paar frei → mit eingefrorenen Konstanten der alten Menge. Was eingefroren noch kippt, kommt aus dem Verfahren selbst.")
    verdict(verdict_sources(proto))
    st.dataframe(frozen_table(proto), hide_index=True, width="stretch")

    st.markdown("### Experiment 3: Wann ist eine Umkehr ein Fehler")
    st.caption("Der relative Entscheider kippt Paare, wenn sich die Spannweite ändert; gezählt werden die gekippten Paare des Verfahrens, die er mitvollzieht, die ohne ihn und seine "
               "Kipper, die das Verfahren verpasst (zufällige neue Alternative). Darunter die Richtung gegenüber dem festen Entscheider.")
    verdict(verdict_legitimacy(proto))
    st.dataframe(legitimacy_table(proto, "zufall"), hide_index=True, width="stretch")
    st.markdown("**Richtung gegenüber dem festen Entscheider bei einem Ausreißer in einem Kriterium** (gekippte Paare zum Entscheider hin / von ihm weg)")
    st.dataframe(direction_table(proto, "ausreisser_gut"), hide_index=True, width="stretch")

    st.markdown("### Experiment 4: Was die feste Referenz verändert")
    st.caption("Platz 1 getroffen in vier Welten (Güte/linear, konkav, Cobb-Douglas, Rohwerte; in der Rohwert-Welt ist die Nenn-Spannweite per Definition richtig), daneben Umkehr gegen Qualität "
               "und der Einfluss von n.")
    verdict(verdict_reference(exp["reference"]))
    st.dataframe(reference_table(exp["reference"]), hide_index=True, width="stretch")
    st.plotly_chart(build_quality_scatter(exp["reference"], proto), width="stretch", key="quality_chart")
    st.markdown("**Zufällige neue Alternative bei n = 8, 20, 40** (Matrizen mit Kipp / Anteil gekippter alter Paare)")
    st.dataframe(size_table(exp["sizes"]), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Zwei Entscheider: fest und relativ** | Der feste Entscheider ist so gebaut, dass jede Umkehr ein Kippen ohne Grund ist; der relative so, dass die Min-Max-Summe richtig ist. Echte Kontexteffekte (Lockvogel, Kompromiss) sind nicht modelliert. | – |
| **Eine neue Alternative je Lauf** | Folgen von Änderungen und kumulative Umkehr sind nicht gemessen; die Richtungsanteile bei wenigen gekippten Paaren sind unsicher. | – |
| **Der Auslöser ist gesetzt** | Dominiert 5 % der Spannweite, Ausreißer 50 %: Die Matrix zeigt die Abhängigkeit vom Auslöser, nicht „die“ Umkehrquote. | – |
| **AHP aus Rohwertverhältnissen** | Echte AHP-Urteile (Paarvergleichsmatrizen) haben eigene Quellen von Inkonsistenz; hier ist die Normierung der Prioritäten die isolierte Quelle. | AHP |
| **Die Nenn-Spannweite ist bekannt** | Die feste Referenz setzt voraus, dass die Grenzen des Merkmals bekannt sind; wie man sie wählt, sagt die Demo nicht. Gegen feste Profile statt untereinander zu bewerten, ist die Entscheidung für diese Annahme. | Zuordnung gegen Profile |
| **Gleiche Zufallsverteilung** | Neue Alternativen stammen aus derselben Verteilung wie die alten; ein Markt, der sich verschiebt, kippt die Spannweiten anders. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gekipptes Paar.** Wertungen $s_i$ auf der alten Menge, $s'_i$ auf der Menge mit der neuen Alternative, $i, j$ alt. Das Paar $\{i, j\}$ ist gekippt, wenn
$\operatorname{sign}(s_i - s_j) \cdot \operatorname{sign}(s'_i - s'_j) < 0$; Gleichstände zählen nicht.

**Konstanten der Menge.** Min-Max: $n_{ij} = (x_{ij} - \min_k x_{kj}) / (\max_k x_{kj} - \min_k x_{kj})$; z-Wert: $(x_{ij} - \mu_j)/\sigma_j$; Vektor: $x_{ij} / \lVert x_{\cdot j} \rVert$;
AHP verteilend: $x_{ij} / \sum_k x_{kj}$, AHP ideal: $x_{ij} / \max_k x_{kj}$; Perzentil: $(\text{Rang}_{ij} - 1)/(n - 1)$. **Eingefroren** heißt: die Konstanten der alten Menge
bleiben für die neue Menge (die Wertung der neuen Alternative bezieht sich auf sie).

**Wann kippt nichts.** Wird jede Alternative $i$ durch eine Funktion $f(x_i)$ bewertet, die nur von $x_i$ und festen Konstanten abhängt, ändert eine neue Alternative keine Wertung der alten:
Es kippt nie ein Paar (Nenn-Spannweite, eingefrorene Konstanten bei Summe, TOPSIS und AHP). Mit der Min-Max-Summe kann eine neue Alternative die Ordnung nur ändern, wenn sie die
Grenzen eines Kriteriums so verschiebt, dass sich die wirksamen Gewichte $w_j / (\max_k x_{kj} - \min_k x_{kj})$ ungleich ändern; ob ein Paar wirklich kippt, hängt von den Werten ab.

**Borda und Paarzählung.** Die Perzentil-Summe ist (bis auf Gleichstände) die gewichtete Borda-Zählung, und PROMETHEE mit der gewöhnlichen Funktion ist gleich ihr: Eine neue Alternative
zwischen zwei alten in einem Kriterium ändert deren Rangdifferenz dort um 1 und damit die Gesamtwertung. Die Copeland-Zahl einer alten Alternative ändert sich um 1, wenn die neue Alternative sie übertrifft oder von ihr übertroffen wird.

**Entscheider.** Fest: $u_i = \sum_j w_j g_{ij}$ (Güte), unabhängig von der Menge. Relativ: $u_i = \sum_j w_j\, n_{ij}$ mit der Min-Max-Normierung der aktuellen Menge, ändert sich mit ihr.
Ein Kipper des Verfahrens heißt *mitvollzogen*, wenn auch der Nutzen des Entscheiders das Paar umkehrt, *verpasst*, wenn nur der Entscheider es umkehrt.

**Literatur:** Die Rangumkehr in der Mehrkriterien-Analyse wurde für AHP von Belton und Gear (1983) beschrieben; Titel und Fundstellen sind hier nicht im Original geprüft und deshalb
nicht angegeben.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html)."
)
