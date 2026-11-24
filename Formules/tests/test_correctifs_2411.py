"""Correctif v0 du 24/11/2026 : « de 3 à 5 » était épelé (« dé-e-trois-à-cinq »)."""

from __future__ import annotations

import pytest

from lectura_formules import lire_formule, lire_intervalle


@pytest.mark.parametrize(
    "texte, display, phone",
    [
        ("de 3 à 5", "de-trois-à-cinq", "də tʁwa a sɛ̃k"),
        ("3 à 5", "trois-à-cinq", "tʁwa a sɛ̃k"),
        ("de 3 a 5", "de-trois-à-cinq", "də tʁwa a sɛ̃k"),
        ("de -3 à 12", "de-moins-trois-à-douze", "də mwɛ̃ tʁwa a duz"),
    ],
)
def test_intervalle_forme_ecrite(texte, display, phone):
    r = lire_intervalle(texte)
    assert r.display_fr == display
    assert r.phone == phone


def test_intervalle_forme_ecrite_composants():
    r = lire_intervalle("de 3 à 5")
    assert [(e.ortho, e.composant) for e in r.events] == [
        ("de", 1), ("trois", 0), ("à", 1), ("cinq", 2),
    ]
    # Le surlignage pointe sur le texte d'origine
    assert [e.span_num for e in r.events] == [(0, 2), (3, 4), (5, 6), (7, 8)]


def test_intervalle_par_lire_formule():
    assert lire_formule("intervalle", "de 3 à 5").phone == "də tʁwa a sɛ̃k"


def test_crochets_inchanges():
    assert lire_intervalle("[2 ; 5]").phone == "də dø a sɛ̃k"
    assert lire_intervalle("]0 ; 1[").display_fr == "de-zéro-à-un"


def test_bornes_non_numeriques_toujours_epelees():
    # Pas d'intervalle reconnu : on garde le repli (épellation)
    assert lire_intervalle("de x à y").display_fr != "de-x-à-y"
