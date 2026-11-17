"""Changement de timbre : timbres autorisés, filigrane exigé, chemin sans contournement.

Aucun modèle n'est chargé : lectura_vc_zeroshot est remplacé par un faux module.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from lectura_diphone import _retimbre
from lectura_diphone._retimbre import (
    FiligraneAbsent,
    RetimbreEngine,
    TimbreIndisponible,
    verifier_voix,
)

MARQUE = 0.001  # « filigrane » factice ajouté par le faux convertisseur


class _FauxEngine:
    """Imite ZeroShotEngine : convert_from_se rend un son « marqué » à 22050 Hz."""

    def __init__(self, presets):
        self.presets = list(presets)
        self.appels: list[tuple] = []

    def available_presets(self):
        return self.presets

    def resolve_target_se(self, ref, ref_sr=None, sr_override=None, variante=None):
        self.appels.append(("resolve", ref, sr_override) if variante is None else ("resolve", ref, "variante", variante))
        if not isinstance(ref, str) or ref not in self.presets:
            raise AssertionError(f"référence inattendue : {ref!r}")
        return np.ones((1, 256, 1), np.float32)

    def extract_se(self, audio, sr=None):
        self.appels.append(("extract", sr))
        return np.zeros((1, 256, 1), np.float32)

    def convert_from_se(self, audio, src_se, tgt_se, sr_in=None, tau=0.3):
        self.appels.append(("convert", sr_in, tau))
        n = int(len(audio) * 22050 / sr_in)
        return np.full(n, MARQUE, np.float32), 22050


def _faux_vc(monkeypatch, presets=("fr-f1",), marque_ia=True):
    mod = types.ModuleType("lectura_vc_zeroshot")
    eng = _FauxEngine(presets)
    mod.creer_engine = lambda models_dir=None: eng
    if marque_ia is not None:
        mod.MARQUE_IA = marque_ia
    monkeypatch.setitem(sys.modules, "lectura_vc_zeroshot", mod)
    return eng


# ── Timbres autorisés ──────────────────────────────────────────────


def test_trois_timbres_neutres():
    assert set(_retimbre.TIMBRES) == {"fr-f1", "melange-f", "melange-h"}
    import lectura_diphone
    assert set(lectura_diphone.TIMBRES_VOIX) == set(_retimbre.TIMBRES)


@pytest.mark.parametrize("voix", ["fr-f1", "melange-f", "melange-h"])
def test_timbres_acceptes(voix):
    assert verifier_voix(voix) == voix


@pytest.mark.parametrize("voix", [
    "voix.wav", "/tmp/ref.wav", ["fr-f1", "melange-f"], {"fr-f1": 0.5, "melange-f": 0.5},
    "inconnue", "FR-F1", np.zeros((1, 256, 1)), 3,
])
def test_autres_references_refusees(voix):
    """Ni fichier audio, ni liste, ni mélange, ni vecteur, ni autre préréglage."""
    with pytest.raises(TimbreIndisponible):
        verifier_voix(voix)


def test_melanges_pas_encore_livres(monkeypatch):
    """Point d'attache : tant que TIMBRES[...] est vide, erreur claire, aucun repli."""
    _faux_vc(monkeypatch, presets=("fr-f1",))
    if _retimbre.TIMBRES["melange-f"]:
        pytest.skip("voix mélangées branchées : voir test_passes.py")
    r = RetimbreEngine()
    with pytest.raises(TimbreIndisponible, match="pas encore livré"):
        r.retimbre(np.zeros(4410, np.float32), 44100, "melange-f")


def test_points_d_attache_branches():
    """Les trois timbres pointent vers les préréglages de même nom de lectura-vc-zeroshot 1.3."""
    assert _retimbre.TIMBRES == {"fr-f1": ("fr-f1",), "melange-f": ("melange-f",), "melange-h": ("melange-h",)}


def test_point_d_attache_branchement(monkeypatch):
    """Une fois le nom de préréglage écrit dans TIMBRES, le timbre est résolu par ce nom."""
    eng = _faux_vc(monkeypatch, presets=("fr-f1", "vecteur-f"))
    monkeypatch.setitem(_retimbre.TIMBRES, "melange-f", ("vecteur-f",))
    out, sr = RetimbreEngine().retimbre(np.zeros(4410, np.float32), 44100, "melange-f")
    assert sr == 22050 and ("resolve", "vecteur-f", None) in eng.appels


def test_fr_f1_resolu_par_prereglage(monkeypatch):
    eng = _faux_vc(monkeypatch, presets=("fr-f1",))
    RetimbreEngine().retimbre(np.zeros(4410, np.float32), 44100, "fr-f1")
    assert ("resolve", "fr-f1", None) in eng.appels


def test_ancien_nom_de_prereglage_ignore(monkeypatch):
    """Seul l'identifiant public fr-f1 est cherché dans lectura-vc-zeroshot."""
    _faux_vc(monkeypatch, presets=("ancien-nom",))
    with pytest.raises(TimbreIndisponible):
        RetimbreEngine().retimbre(np.zeros(4410, np.float32), 44100, "fr-f1")


def test_fr_f1_absent_de_vc(monkeypatch):
    _faux_vc(monkeypatch, presets=())
    with pytest.raises(TimbreIndisponible, match="absent"):
        RetimbreEngine().retimbre(np.zeros(4410, np.float32), 44100, "fr-f1")


# ── Filigrane exigé ────────────────────────────────────────────────


@pytest.mark.parametrize("marque", [None, False])
def test_vc_sans_filigrane_refuse(monkeypatch, marque):
    """Une ancienne lectura-vc-zeroshot (sans MARQUE_IA) est refusée."""
    eng = _faux_vc(monkeypatch, marque_ia=marque)
    with pytest.raises(FiligraneAbsent):
        RetimbreEngine().retimbre(np.zeros(4410, np.float32), 44100, "fr-f1")
    assert not any(a[0] == "convert" for a in eng.appels)


def test_vc_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "lectura_vc_zeroshot", None)
    with pytest.raises(ImportError, match=r"lectura-diphone\[vc\]"):
        RetimbreEngine().retimbre(np.zeros(4410, np.float32), 44100, "fr-f1")


def test_conversion_passe_par_convert_from_se(monkeypatch):
    """Le son rendu est celui de convert_from_se (point où vc-zeroshot pose le filigrane)."""
    eng = _faux_vc(monkeypatch)
    out, sr = RetimbreEngine().retimbre(np.zeros(44100, np.float32), 44100, "fr-f1", tau=0.2)
    assert [a[0] for a in eng.appels][-1] == "convert"
    assert np.allclose(out, MARQUE)


def test_voix_refusee_avant_synthese():
    """synthesize_groups refuse un timbre interdit avant de charger la banque."""
    from lectura_diphone.engine import DiphoneEngine

    e = DiphoneEngine()
    with pytest.raises(TimbreIndisponible):
        e.synthesize_groups([{"phones": ["a"], "boundary": "period"}], voix="ref.wav")
    assert e.loaded is False


def test_apply_retimbre_ne_fait_que_reechantillonner(monkeypatch):
    """Après la conversion : suréchantillonnage x2 et gain linéaire, rien d'autre."""
    pytest.importorskip("scipy")
    from lectura_diphone.engine import DiphoneEngine
    from lectura_diphone._world import SIWIS_SR

    _faux_vc(monkeypatch)
    e = DiphoneEngine()
    x = np.zeros(SIWIS_SR, np.float32)
    y = e._apply_retimbre(x, SIWIS_SR, "fr-f1", 0.0, 0.3, None)
    assert y.dtype == np.float32
    assert abs(len(y) - SIWIS_SR) <= 2
    # sortie = suréchantillonnage polyphasé x2 du son converti, à un gain près
    from scipy.signal import resample_poly
    ref = resample_poly(np.full(SIWIS_SR // 2, MARQUE, np.float32), 2, 1)
    assert len(ref) == len(y)
    gain = y / ref
    assert np.allclose(gain, gain[len(gain) // 2], rtol=1e-4)
    assert abs(np.abs(y).max() - 0.9) < 1e-6


def test_source_sans_contournement():
    """Aucun appel direct au réseau de conversion qui court-circuiterait convert_from_se."""
    import inspect

    import ast

    arbre = ast.parse(inspect.getsource(_retimbre))
    noms = {n.attr for n in ast.walk(arbre) if isinstance(n, ast.Attribute)}
    noms |= {n.id for n in ast.walk(arbre) if isinstance(n, ast.Name)}
    assert "convert_from_se" in noms
    assert not noms & {"vc_sess", "OpenVoiceConverter", "_filigrane", "convert", "run"}


@pytest.mark.parametrize("variante,attendu", [(-1.0, -1.0), (0.5, 0.5), (1.7, 1.0), (0.004, None)])
def test_variante_transmise(monkeypatch, variante, attendu):
    """Le curseur est transmis à lectura-vc-zeroshot (paramètre variante), borné à [-1, 1] ; ~0 = préréglage."""
    eng = _faux_vc(monkeypatch, presets=("fr-f1",))
    RetimbreEngine().retimbre(np.zeros(4410, np.float32), 44100, "fr-f1", variante=variante)
    if attendu is None:
        assert ("resolve", "fr-f1", None) in eng.appels
    else:
        assert ("resolve", "fr-f1", "variante", attendu) in eng.appels
