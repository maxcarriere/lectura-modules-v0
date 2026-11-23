"""Tests des corrections v0 du 23/11/2026 (défauts relevés par la documentation)."""

from __future__ import annotations

import lectura_formules
from lectura_formules import (
    OptionsLecture,
    detect_formule_spans_stt,
    detect_number_spans,
    lire_formule,
    lire_heure,
    lire_intervalle,
    lire_nombre,
)
from lectura_formules.lecture_formules import _LECTEURS


# ── Intervalles : « à » accentué ─────────────────────────────────────

def test_intervalle_a_accentue():
    r = lire_intervalle("[0 ; 1]")
    assert r.display_fr == "de-zéro-à-un"
    assert r.phone == "də zeʁo a ɛ̃"


# ── Heures : pas de minutes nulles lues ──────────────────────────────

def test_midi_sans_zero():
    mm = OptionsLecture(heure_minuit_midi=True)
    assert lire_heure("12h00", options=mm).display_fr == "midi"
    assert lire_heure("12:00", options=mm).display_fr == "midi"
    assert lire_heure("0h00", options=mm).display_fr == "minuit"
    assert lire_heure("12h30", options=mm).display_fr == "midi-trente"


def test_heure_pile_sans_zero():
    assert lire_heure("14h00").display_fr == "quatorze-heures"
    assert lire_heure("14h05").display_fr == "quatorze-heures-cinq"
    # Avec le mot « minutes » demandé, la forme complète reste lue
    r = lire_heure("14h00", options=OptionsLecture(heure_mot_minutes=True))
    assert "minute" in r.display_fr


# ── Exports et docstring ─────────────────────────────────────────────

def test_lecteurs_exportes():
    for nom in ("lire_abv", "lire_romain", "lire_unite"):
        assert callable(getattr(lectura_formules, nom))
    assert lectura_formules.lire_romain("XIV").display_fr == "quatorze"


def test_docstring_liste_les_18_types():
    doc = lire_formule.__doc__
    assert len(_LECTEURS) == 18
    for ftype in _LECTEURS:
        assert ftype in doc, ftype


def test_decimal_m3_chiffre_par_chiffre():
    r = lire_nombre("3,14", options=OptionsLecture(decimal_method="m3"))
    assert r.display_fr.endswith("virgule-un-quatre")
    assert '"m3"' in lire_formule.__doc__


# ── Reconnaissance : le connecteur « et » n'ouvre ni ne ferme un span ──

def test_nombre_n_englobe_pas_est():
    # « il est quinze heures » : « ɛ » (est) ne fait pas partie du nombre
    for ipa in (["il", "ɛ", "kɛ̃z", "œʁ"], ["il", "e", "kɛ̃z", "œʁ"]):
        for mode in ("auto", "num"):
            spans = detect_number_spans(ipa, min_span=1, max_span=15, mode=mode)
            assert all(s > 1 for s, _e, _r in spans), (ipa, mode, spans)
        for permissif in (False, True):
            spans = detect_formule_spans_stt(ipa, permissive=permissif)
            assert all(s > 1 for s, _e, _r in spans), (ipa, permissif, spans)


def test_nombre_n_englobe_pas_et_final():
    spans = detect_formule_spans_stt(["tʁwa", "øʁo", "e", "sɛ̃kɑ̃t"])
    assert all(e <= 2 for _s, e, _r in spans), spans


def test_et_interne_garde():
    # « vingt et un », « soixante et onze » : « et » au milieu reste dans le nombre
    spans = detect_number_spans(["vɛ̃", "e", "ɛ̃"], min_span=1, mode="auto")
    assert [(s, e, r.valeur) for s, e, r in spans] == [(0, 3, 21)]
    spans = detect_number_spans(["swasɑ̃t", "ɛ", "ɔ̃z"], min_span=1, mode="auto")
    assert [(s, e, r.valeur) for s, e, r in spans] == [(0, 3, 71)]
