"""Tests des corrections v0 du 23/11/2026 (défauts relevés par la documentation).

Les tests qui passent par le modèle ont besoin des poids P2G déjà présents
(``LECTURA_MODELS_DIR`` ou ``~/.lectura/models/p2g``) ; sinon ils sont sautés.
"""

from __future__ import annotations

import pytest

import lectura_p2g
from lectura_p2g import _charger_corrections, analyser


@pytest.fixture(scope="module")
def engine():
    from lectura_graphemiseur import _resoudre_modeles_dir
    if _resoudre_modeles_dir(telecharger=False) is None:
        pytest.skip("poids P2G absents")
    return lectura_p2g.creer_engine()


# ── Défaut 3 : cache des corrections, une table par option ───────────

def test_cache_corrections_par_option():
    lectura_p2g._P2G_CORRECTIONS.clear()
    base = _charger_corrections(etendu=False)
    etendu = _charger_corrections(etendu=True)
    assert base and etendu
    assert len(etendu) > len(base)  # le second appel n'est plus figé sur le premier
    assert _charger_corrections(etendu=False) is base
    assert _charger_corrections(etendu=True) is etendu


# ── Défaut 1 : une formule n'avale plus « est » ──────────────────────

def test_formule_n_avale_pas_est(engine):
    r = analyser(["il", "ɛ", "kɛ̃z", "œʁ"], engine=engine)
    assert r["ortho"] == ["il", "est", "quinze", "heures"]
    r = analyser(["il", "ɛ", "kɛ̃z", "œʁ"], engine=engine, number_mode="num")
    assert r["ortho"] == ["il", "est", "15", "heures"]


# ── Défaut 2 : cases vides documentées, et retirées sur demande ──────

def test_cases_vides_par_defaut(engine):
    r = analyser(["il", "a", "vɛ̃", "e", "ɛ̃", "ɑ̃"], engine=engine)
    assert r["ortho"] == ["il", "a", "21", "", "", "ans"]
    assert len(r["ortho"]) == len(r["ipa_words"]) == len(r["pos"]) == 6
    assert "Cases vides" in analyser.__doc__


def test_compacter_retire_les_cases_vides(engine):
    ipa = ["oʒuʁ", "dɥi", ",", "il", "a", "vɛ̃", "e", "ɛ̃", "ɑ̃"]
    r = analyser(ipa, engine=engine, compacter=True)
    assert r["ortho"] == ["aujourd'hui", ",", "il", "a", "21", "ans"]
    assert r["ipa_words"] == ["oʒuʁ dɥi", ",", "il", "a", "vɛ̃ e ɛ̃", "ɑ̃"]
    n = len(r["ortho"])
    assert len(r["pos"]) == n and r["pos"][1] == "PUNCT"
    assert all(len(v) == n for v in r["morpho"].values())


def test_compacter_sans_case_vide(engine):
    ipa = ["le", "ɑ̃fɑ̃", "ʒu"]
    assert analyser(ipa, engine=engine, compacter=True) == analyser(ipa, engine=engine)


def test_compacter_unitaire():
    r = {"ipa_words": ["a", "b", "c"], "ortho": ["", "X", ""], "pos": ["P1", "P2", "P3"],
         "morpho": {"Number": ["1", "2", "3"]}, "alignments": [None, "al", None]}
    lectura_p2g._compacter_cases_vides(r)
    assert r == {"ipa_words": ["a b c"], "ortho": ["X"], "pos": ["P2"],
                 "morpho": {"Number": ["2"]}, "alignments": ["al"]}
