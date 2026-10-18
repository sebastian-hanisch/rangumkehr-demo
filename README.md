# Rangumkehr: eine Alternative kommt dazu, die alte Ordnung kippt – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-rangumkehr-demo.streamlit.app/)**

Neuntes Stück der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Kind von
[AHP](https://github.com/sebastian-hanisch/ahp-demo), Idee von [TOPSIS](https://github.com/sebastian-hanisch/topsis-demo). Kommt eine Alternative dazu, ändern sich die Rohwerte der
alten nicht, ihre Ordnung untereinander aber vielleicht doch. Die Demo führt ein **Protokoll** aus (zufällige neue Alternative, dominierte, Klon, Ausreißer in einem Kriterium), zerlegt die
**Ursache** (Konstanten der Menge gegen das Verfahren selbst) und fragt, **wann eine Umkehr ein Fehler ist**.

Vehikel ist dieselbe **neutrale, simulierte Entscheidungsmatrix** wie in den Vorgängern (Generator `entscheidungsmatrix_generator.py`, wortgleich). Der **Entscheider** kennt seinen Nutzen:
fest (Güte, linear; hängt nicht an der Menge) oder relativ (Min-Max der aktuellen Menge); er ist ein Orakel, kein Mensch.

## Warum dieses Problem

**Eine Umkehr entsteht dort, wo ein Verfahren die Alternativen mit etwas bewertet, das aus der ganzen Menge berechnet wird: Normierungskonstanten, Ränge, Paarzählungen.** Bewertet es jede
Alternative für sich gegen eine feste Referenz, kippt nie etwas. Die Kernfrage der Demo: Woher kommt die Umkehr, wie oft, und wann ist sie ein Fehler?

## Befunde und Korrekturen gegenüber dem Plan

- **Die Häufigkeit der Umkehr hängt am Auslöser, nicht nur am Verfahren.** Die Min-Max-Summe kippt bei einer dominierten Alternative oder einem Klon nie (0,0 %), bei einem Ausreißer in einem
  Kriterium bei n = 20, m = 5 in 88,5 % der Matrizen (mindestens ein Paar; bei n = 10 sind es 72 %). Eine Tabelle „Verfahren gegen Umkehrquote“ ohne den Auslöser ist irreführend.
- **Die Ursache liegt in den Konstanten der Menge, in den Rängen oder in der Paarzählung.** Mit eingefrorenen Konstanten der alten Menge kippt bei Summe, TOPSIS und AHP in keinem Protokoll
  etwas; PROMETHEE und ELECTRE kippen weiter (Struktur des Verfahrens).
- **Ob eine Umkehr ein Fehler ist, entscheidet der Entscheider.** Ein Entscheider, der per Definition wie die Min-Max-Summe der aktuellen Menge bewertet, wird von dieser vollständig nachvollzogen (75 von 75 Paare);
  die meisten anderen Verfahren kippen weit mehr Paare, als er es tut (176 bis 429 gegen 75), ELECTRE (35) und AHP ideal (52) weniger, die Nenn-Verfahren keines. Die feste Referenz kippt nichts
  und verpasst alle 75. Bei einem festen Nutzen verschlechtert nur der Ausreißer die Ordnung der Min-Max-Summe messbar (64 % der Kipper weg vom Entscheider, Δτ −0,018, klein); bei den anderen
  drei Auslösern ist die Richtung Rauschen.
- **Ein „neuer Idealpunkt“ allein genügt nicht:** Eine Alternative, die in jedem Kriterium um denselben Anteil der Spannweite über allen liegt, kippt die Min-Max-Summe (und die Perzentil-Summe) nicht; entscheidend ist die
  ungleiche Änderung der Spannweiten.
- **Entfernen ist Hinzufügen rückwärts:** dieselben Paare kippen; kein eigener Lauf.
- **Die Experimente starten auf Klick** (rund 15 Sekunden), nicht beim Laden.

## Modell

- **Matrix:** n alte Alternativen (5 bis 40, Standard 10), m Kriterien (3 bis 8), Rohwerte mit Richtung und Einheit; eine neue Alternative je Lauf.
- **Auslöser:** zufällig (die (n + 1)-te Zeile derselben Matrix), dominiert (je Kriterium um 5 % der Spannweite schlechter als eine alte, nicht unter das Minimum), Klon (Kopie einer alten),
  Ausreißer (ein Kriterium um 50 % der Spannweite über oder unter dem Extrem).
- **Verfahren (zwölf):** Summe mit Min-Max, Perzentil-Rängen, z-Wert, Vektor-Normierung und Nenn-Spannweite; TOPSIS (Vektor, Stichprobenideal) und TOPSIS (Nenn-Spannweite, Nenn-Ideal);
  PROMETHEE II gewöhnlich und V-Form p = 0,5; ELECTRE I mit Copeland-Zahl (c* = 0,6, ohne Veto); AHP verteilend (Spalte auf Summe 1) und ideal (Spalte durch Maximum, aus Rohwertverhältnissen).
- **Gekipptes Paar:** zwei alte Alternativen, die vorher und nachher strikt, aber entgegengesetzt geordnet sind; Gleichstände zählen nicht.
- **Eingefroren:** die Normierungskonstanten der alten Menge bleiben für die neue Menge.
- **Fester Rahmen der Experimente:** 200 feste Matrizen (`SWEEP_SEEDS`), 20 alte Alternativen, 5 Kriterien, wahre Gewichte, unabhängig von den Reglern.

Formeln im 📐-Expander der App.

## Methodik

1. **Dieselbe Matrix, dieselbe neue Alternative, alle Verfahren:** Vergleiche zwischen Verfahren sind gepaart.
2. **Zerlegung durch Einfrieren:** frei gegen eingefroren trennt die Konstanten der Menge vom Verfahren selbst.
3. **Zwei Entscheider:** fester Nutzen (Annahme: jede Umkehr kippt ohne Grund) und relativer Nutzen (die Umkehr der Min-Max-Summe ist seine); Maß ist der Anteil der vom Entscheider mitvollzogenen
   Kipper.
4. **Gegenproben:** feste Referenz (nie eine Umkehr), dominierte Alternative und Klon bei Min-Max-Summe und AHP ideal (nie eine Umkehr).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: `tests/test_claims.py`. 200 feste Matrizen, 20 alte Alternativen, 5 Kriterien, wahre Gewichte. **Der Standardfehler einer Trefferquote ist 1,3 bis 3,5 Punkte; Unterschiede
unter etwa 6 Punkten sind nicht belastbar und stehen nicht als Aussage.**

| Frage | Befund | Test |
|---|---|---|
| Zufällige neue Alternative: Matrizen mit gekipptem Paar / Anteil gekippter alter Paare / Platz 1 der alten wechselt | Summe Min-Max **13,5 / 0,20 / 0,5 %**; Perzentil und PROMETHEE gewöhnlich **89,0 / 1,13 / 5,0 %**; z-Wert 49,0 / 0,49 / 2,5; Vektor 51,0 / 0,53 / 2,0; TOPSIS (Vektor) 58,0 / 0,67 / 1,5; PROMETHEE V 0,5: 52,0 / 0,46 / 3,5; ELECTRE Copeland 13,0 / 0,09 / 3,5; AHP verteilend 59,5 / 0,63 / 3,5; AHP ideal 5,5 / 0,14 / 0,5; **Nenn-Spannweite und TOPSIS (Nenn) 0,0**. | `test_random_new_alternative` |
| Dominierte Alternative innerhalb der Spannweite | **Summe Min-Max, AHP ideal, Nenn-Verfahren 0,0 %**; Perzentil / gewöhnlich 89,5 %; AHP verteilend 60,5 %; z-Wert 50,5; Vektor 55,0; TOPSIS 53,5; V 0,5: 41,0; ELECTRE 12,0. | `test_a_dominated_alternative_leaves_the_minmax_sum_and_ahp_ideal_untouched` |
| Klon einer alten Alternative | Summe Min-Max, AHP ideal, Nenn-Verfahren 0,0 %; Perzentil 85,0 %; **AHP verteilend 69,5 %**; z-Wert 50,0; Vektor 54,5; TOPSIS 56,5; V 0,5: 38,5; ELECTRE 5,5. | `test_a_clone_flips_rank_and_set_based_methods` |
| Ausreißer in einem Kriterium (Spannweite +50 %) | **Summe Min-Max 88,5 / 3,20 / 11,0 %**; TOPSIS (Vektor) 85,5 / 2,89 / **17,0 %**; V 0,5: 84,5 / 1,86 / 6,5; z-Wert 83,0 / 1,39 / 4,0; Perzentil 80,0 / 0,93 / 2,0; AHP verteilend 71,0 / 0,83 / 2,5; Vektor 65,5; **AHP ideal 52,0 / 1,84 / 5,5**; ELECTRE 3,5; Nenn-Verfahren 0,0. | `test_an_outlier_in_one_criterion_is_the_worst_trigger` |
| Perzentil-Summe gegen PROMETHEE gewöhnlich | Gleiche gekippte Paare in jedem Protokoll (Borda-Identität aus Stück 7). | `test_percentile_sum_and_promethee_usual_flip_the_same_pairs` |
| Ursache: Matrizen mit gekipptem Paar frei → mit eingefrorenen Konstanten | **Summe (vier Normierungen), TOPSIS, AHP beide Modi: 0,0 % in jedem Protokoll**; PROMETHEE gewöhnlich 89,0 / 89,5 / 85,0 / 80,0 % (zufällig / dominiert / Klon / Ausreißer), V 0,5: 49,5 / 41,0 / 38,5 / 57,0 %; ELECTRE 13,0 / 12,0 / 5,5 / 3,5 %. | `test_constant_based_methods_do_not_flip_with_frozen_constants`, `test_pair_counting_methods_keep_flipping_with_frozen_constants` |
| Relativer Entscheider, zufällige neue Alternative | Er kippt 75 Paare; die **Min-Max-Summe vollzieht alle 75 mit**; mitvollzogen: V 0,5 28 von 176, z-Wert 25 von 187, Vektor 5 von 202, Perzentil / gewöhnlich 3 von 429, AHP verteilend 2 von 238, TOPSIS 2 von 253, AHP ideal 2 von 52, ELECTRE 0 von 35; **feste Referenz: 0 gekippt, verpasst alle 75**. Bei Ausreißer: Min-Max 1 216 von 1 216. | `test_the_relative_decider_flips_what_the_minmax_sum_flips`, `test_other_methods_follow_few_of_the_deciders_flips`, `test_the_fixed_reference_misses_all_of_them` |
| Dominierte Alternative / Klon beim relativen Entscheider | Er kippt nie; jedes gekippte Paar jedes Verfahrens ist ohne ihn. | `test_the_fixed_reference_misses_all_of_them` |
| Richtung bei festem Entscheider | Bei Ausreißer: Min-Max 435 zum Entscheider hin, **781 weg (Δτ −0,018)**; AHP ideal 65,4 % weg. Bei zufälliger, dominierter und Klon-Alternative: Δτ höchstens ±0,0035. | `test_an_outlier_worsens_the_order_for_a_decider_with_a_fixed_utility` |
| Platz 1 getroffen (Güte/linear, konkav, Cobb-Douglas, Rohwerte) | **Summe Min-Max 74,0 / 74,0 / 71,0 / 95,0 %, Nenn-Spannweite 72,0 / 73,0 / 70,0 / 100,0 %** (Rohwerte: der Entscheider ist auf der Nenn-Spannweite definiert, Tautologie); TOPSIS Min-Max, Stichprobe 63,5 / 65,0 / 63,5 / 80,5, Nenn 61,0 / 60,5 / 57,5 / 80,0; TOPSIS Vektor 70,5 / 74,5 / 73,5 / 78,5; Perzentil 75,0 / 72,5 / 68,0 / 60,0; AHP ideal 81,0 / 73,0 / 65,0 / 57,0; AHP verteilend 65,5 / 60,0 / 52,5 / 44,5. | `test_first_place_in_four_worlds` |
| Andere n (zufällige neue Alternative; Matrizen mit Kipp / Anteil gekippter Paare) | n = 8: Summe Min-Max 21,5 % / 1,29 %, Perzentil 56,0 %; n = 20: 13,5 % / 0,20 %, 89,0 %; **n = 40:** 10,0 % / 0,06 %, 96,5 %. | `test_flips_shrink_in_share_with_n_and_rise_in_frequency_for_rank_methods` |

## Ehrliche Grenzen

- **Der Auslöser prägt das Ergebnis.** Dominiert 5 %, Ausreißer 50 % sind Setzungen; die Matrix Verfahren × Auslöser zeigt die Abhängigkeit, nicht „die“ Umkehrquote.
- **Zwei Entscheider.** Der feste ist so gebaut, dass jede Umkehr ein Kippen ohne Grund ist; der relative so, dass die Min-Max-Summe richtig ist. Echte Kontexteffekte des Menschen (Lockvogel,
  Kompromiss) sind nicht modelliert.
- **AHP aus Rohwertverhältnissen.** Echte AHP-Urteile (Paarvergleichsmatrizen) haben eigene Quellen von Inkonsistenz; die Normierung der Prioritäten ist die isolierte Quelle.
- **Eine Alternative je Lauf, 200 Matrizen.** Folgen von Änderungen und kumulative Umkehr sind nicht gemessen; Richtungsanteile bei wenigen gekippten Paaren sind unsicher.
- **Die Nenn-Spannweite als feste Referenz** setzt voraus, dass die Grenzen des Merkmals bekannt sind; die Demo sagt nicht, wie man sie wählt. In der Rohwert-Welt ist sie per Definition richtig.
- **Quellenangaben nur aus Hintergrundwissen:** Die Rangumkehr in AHP (Belton und Gear, 1983) ist hier nicht im Original geprüft.

## Verifikation

- **Kern:** die zwölf Verfahren, die Protokolloperationen, gekippte Paare und die Richtung gegen Handrechnung auf den Mini-Instanzen (`tests/conftest.py`: AHP verteilend mit Klon,
  Perzentil-Summe mit einer neuen Alternative, Min-Max-Summe mit Ausreißer; je mit den Werten vorher und nachher).
- **Orakel** (`tests/test_oracle_rangumkehr.py`): eingefrorene Konstanten kippen nie; die feste Referenz kippt nie; dominierte Alternative und Klon lassen Min-Max-Summe und AHP ideal
  unverändert; Perzentil-Summe und PROMETHEE gewöhnlich kippen dieselben Paare; Entfernen kippt dieselben Paare wie Hinzufügen; der relative Entscheider kippt genau die Paare der
  Min-Max-Summe; TOPSIS und PROMETHEE gegen Definitionsschleifen.
- **Randfälle** (`tests/test_edge_cases.py`): n = 2, Klon als Gleichstand, lauter gleiche Alternativen, konstante Spalte, Nullwerte bei AHP, Perzentil mit kleiner Referenz, leere Menge
  gekippter Paare, Ausreißer in genau einem Kriterium.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula, Richtungen und Bereiche.
- **Aussagen:** jede Tabellenzahl dieser README in `tests/test_claims.py`; Presets in `test_presets.py` (jedes zeigt seinen Effekt).
- **Mutationstest** vor dem ersten Commit (`tools/mutate_demo.py` der Website): 184 Stellen, davon 16 durch wortgleiche Zeilen der Nachbar-Demos gesichert; erster Lauf 122 von 168 Mutanten gefangen (73 %), die Überlebenden waren fast alle Texte und Zählfelder der Verdikte und zwei unerreichbare Absicherungen gegen Teiler 0 (entfernt). Danach ist jede Stelle gefangen oder als äquivalent festgehalten (4: Vorzeichen der Richtung ±1, konstante Verschiebung des Rohwert-Nutzens, Gleichstände im Nettofluss); Stand in `tests/mutation_state.json`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), alle Presets, jedes Verfahren, jeder Auslöser, jeder Entscheider, Permalink, PDF und Experimente erst
  nach Klick, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `rku_constants.py` | Regler-Grenzen, Verfahren und Auslöser, feste Annahmen, Farben, Presets |
| `rku_score.py` | Normierungen mit Konstanten (Min-Max, Perzentil, z-Wert, Vektor), Nenn-Spannweite, Kendalls τ-b, Platz 1 |
| `rku_methods.py` | Die zwölf Verfahren als reine Funktionen mit eingefrorener Referenz |
| `rku_protocol.py` | Neue Alternative (vier Auslöser), gekippte Paare, Lauf, Richtung, relativer Entscheider |
| `rku_evaluation.py` | Fall der App, vier Experimente, Verdikte |
| `rku_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `rku_visualization.py` | Steigungsdiagramm, Tabellen, Matrix Verfahren × Auslöser, Plotly-Abbildungen (Achsen gesperrt) |
| `rku_pdf_export.py` | PDF-Export des aktuellen Falls (fpdf2, erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Mini-Instanzen, Verfahren, Protokoll, Score, Orakel, Randfälle, Generator, Auswertung, Aussagen, Presets, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Folgen von Änderungen und kumulative Umkehr:** eine Alternative je Lauf.
- **Echte AHP-Urteile (Paarvergleichsmatrizen):** AHP nur aus Rohwertverhältnissen, die Urteilsmatrix steht im AHP-Stück.
- **Menschliche Kontexteffekte (Lockvogel, Kompromiss):** zwei Entscheider, kein Mensch.
- **Gegen feste Profile zuordnen:** die feste Referenz ist die Annahme, die Zuordnung gegen Profile hat ein eigenes Stück.
- **Schwellen je Kriterium (PROMETHEE) und Veto (ELECTRE):** ein Regler für alle Kriterien, kein Veto.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html).
