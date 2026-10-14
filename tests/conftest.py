"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def ahp_clone():
    """AHP verteilend (Spalte auf Summe 1), w = 0,5 / 0,5, Werte als Verhältnisskala (beide Kriterien ↑): A = (3, 7), B = (9, 2), C = (2, 5). Spaltensummen 14 / 14:
    A = 10/28 = 0,3571, B = 11/28 = 0,3929, C = 7/28 = 0,25 -> B > A > C. Mit dem Klon B' = B: Spaltensummen 23 / 16: A = (3/23 + 7/16)/2 = 0,2840,
    B = (9/23 + 2/16)/2 = 0,2582, C = 0,1997 -> A > B > C: A und B kehren sich um, ohne dass sich ein Wert geändert hat."""
    return np.array([[3.0, 7.0], [9.0, 2.0], [2.0, 5.0]]), np.array([0.5, 0.5])


@pytest.fixture
def borda_d():
    """Summe der Perzentil-Ränge (0-basiert), w = 0,5 / 0,3 / 0,2: A = (1, 9, 8), B = (7, 4, 2), C = (8, 2, 3). Ränge A = (0, 2, 2), B = (1, 1, 0), C = (2, 0, 1);
    Summen A = 1,0, B = 0,8, C = 1,2 -> C > A > B. Mit D = (8,5; 6,5; 3,5): Ränge A = (0, 3, 3), B = (1, 1, 0), C = (2, 0, 1); A = 1,5, B = 0,8, C = 1,2 -> A > C > B (D = 2,5 führt)."""
    return np.array([[1.0, 9.0, 8.0], [7.0, 4.0, 2.0], [8.0, 2.0, 3.0]]), np.array([8.5, 6.5, 3.5]), np.array([0.5, 0.3, 0.2])


@pytest.fixture
def outlier():
    """Min-Max-Summe, w = 0,5 / 0,5: A = (0, 0), B = (10, 3), C = (2, 10): A = 0, B = 0,65, C = 0,6 -> B > C > A. Mit dem Ausreißer D = (30, 0): erstes Kriterium durch 30 statt 10:
    B = (0,3333 + 0,3)/2 = 0,3167, C = (0,0667 + 1)/2 = 0,5333 -> C > B > A. Mit fester Nenn-Spannweite bliebe B > C."""
    return np.array([[0.0, 0.0], [10.0, 3.0], [2.0, 10.0]]), np.array([30.0, 0.0]), np.array([0.5, 0.5])
