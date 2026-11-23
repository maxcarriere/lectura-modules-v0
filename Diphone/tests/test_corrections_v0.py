"""Corrections de la v0 (23/11/2026) : moteur gardé par synthetiser(), exports, voix_tau sans
effet, bornes d'ap_cleanup / formant_sharpening, phonèmes inconnus signalés.

Banque factice (aucun téléchargement) ; le test sur la banque réelle demande LECTURA_PASSES=1.
"""

from __future__ import annotations

import hashlib
import os
import warnings

import numpy as np
import pytest

pytest.importorskip("pyworld")

import lectura_diphone as ld  # noqa: E402
from lectura_diphone.engine import DiphoneEngine  # noqa: E402

GROUPES = [
    {"phones": ["b", "a", "b", "a"], "boundary": "comma"},
    {"phones": ["a", "b", "a"], "boundary": "period"},
]


def _banque_factice() -> dict:
    rng = np.random.default_rng(0)
    banque = {}
    for a in ("#", "a", "b"):
        for b in ("#", "a", "b"):
            if a == b == "#":
                continue
            banque[f"{a}-{b}"] = {
                "f0": np.full(12, 150.0) * (b != "b"),
                "sp": np.abs(rng.standard_normal((12, 1025))) * 1e-3 + 1e-6,
                "ap": np.full((12, 1025), 0.2),
            }
    return banque


@pytest.fixture
def chargements(monkeypatch, tmp_path):
    """DiphoneEngine.load remplacé par une banque factice ; compte les chargements."""
    compte = []
    banque = _banque_factice()

    def load(self, models_dir=None):
        compte.append(models_dir)
        self.diphones = banque
        self._phones_connus = None
        self.loaded = True

    monkeypatch.setattr(DiphoneEngine, "load", load)
    monkeypatch.setenv("LECTURA_MODELS_DIR", str(tmp_path))
    ld.vider_cache()
    yield compte
    ld.vider_cache()


def _g2p(texte):
    return [dict(g) for g in GROUPES]


def _empreinte(r) -> str:
    return hashlib.sha256(np.ascontiguousarray(r.samples).tobytes()).hexdigest()


# ---------------------------------------------------------------- moteur gardé par synthetiser()

def test_synthetiser_ne_recharge_pas_la_banque(chargements):
    r1 = ld.synthetiser("x", g2p=_g2p, seed=3)
    r2 = ld.synthetiser("x", g2p=_g2p, seed=3)
    ld.synthetiser("x", g2p=_g2p, seed=4, synth_mode="SYLLABES", base_f0=120)
    assert len(chargements) == 1
    assert _empreinte(r1) == _empreinte(r2)


def test_synthetiser_meme_son_qu_un_moteur_neuf(chargements):
    ld.synthetiser("x", g2p=_g2p, seed=1)            # met le moteur en cache
    en_cache = ld.synthetiser("x", g2p=_g2p, seed=5, prosody_style="corpus")
    neuf = ld.creer_engine(g2p=_g2p).synthesize_groups(_g2p("x"), seed=5, prosody_style="corpus")
    assert _empreinte(en_cache) == _empreinte(neuf)


def test_cle_du_cache(chargements, tmp_path):
    autre_g2p = lambda t: _g2p(t)  # noqa: E731
    ld.synthetiser("x", g2p=_g2p, seed=1)
    ld.synthetiser("x", g2p=autre_g2p, seed=1)                  # autre g2p -> autre moteur
    ld.synthetiser("x", g2p=_g2p, seed=1, models_dir=tmp_path / "b")  # autre banque -> autre moteur
    assert len(chargements) == 3
    ld.synthetiser("x", g2p=_g2p, seed=1)
    assert len(chargements) == 3
    ld.vider_cache()
    ld.synthetiser("x", g2p=_g2p, seed=1)
    assert len(chargements) == 4


def test_mode_toujours_verifie(chargements):
    with pytest.raises(ValueError, match="synth_mode"):
        ld.synthetiser("x", g2p=_g2p, mode="FLUIDE")


@pytest.mark.skipif(os.environ.get("LECTURA_PASSES") != "1", reason="banque réelle : LECTURA_PASSES=1")
def test_banque_reelle_meme_empreinte():
    from lectura_diphone import _poids
    if _poids.fichiers_manquants(_poids.dossier_destination()):
        pytest.skip("banque réelle absente")
    groupes = [{"phones": ["b", "ɔ̃", "ʒ", "u", "ʁ"], "boundary": "comma"},
               {"phones": ["l", "ə", "m", "ɔ̃", "d"], "boundary": "period"}]
    g2p = lambda t: [dict(g) for g in groupes]  # noqa: E731
    ld.vider_cache()
    try:
        a = ld.synthetiser("x", g2p=g2p, seed=1)
        b = ld.synthetiser("x", g2p=g2p, seed=1)
        neuf = ld.creer_engine(g2p=g2p).synthesize_groups(groupes, seed=1)
    finally:
        ld.vider_cache()
    assert _empreinte(a) == _empreinte(b) == _empreinte(neuf)


# ---------------------------------------------------------------- exports

def test_diphone_engine_et_synth_mode_exportes():
    from lectura_diphone import engine as e
    for nom in ("DiphoneEngine", "SynthMode", "vider_cache"):
        assert nom in ld.__all__
    assert ld.DiphoneEngine is e.DiphoneEngine and ld.SynthMode is e.SynthMode


# ---------------------------------------------------------------- voix_tau

def test_voix_tau_avertit(chargements):
    moteur = ld.creer_engine(g2p=_g2p)
    with pytest.warns(DeprecationWarning, match="voix_tau est sans effet"):
        moteur.synthesize_groups(GROUPES, seed=1, voix_tau=0.0)
    with pytest.warns(DeprecationWarning, match="voix_tau est sans effet") as releves:
        ld.synthetiser("x", g2p=_g2p, seed=1, voix_tau=0.5)
    assert len(releves) == 1 and releves[0].filename == __file__


def test_sans_voix_tau_aucun_avertissement(chargements):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ld.synthetiser("x", g2p=_g2p, seed=1)


# ---------------------------------------------------------------- bornes

@pytest.mark.parametrize("nom, maximum", [("ap_cleanup", 3.0), ("formant_sharpening", 2.0)])
def test_bornes_appliquees(chargements, nom, maximum):
    moteur = ld.creer_engine(g2p=_g2p)
    borne = moteur.synthesize_groups(GROUPES, seed=2, **{nom: maximum})
    with pytest.warns(UserWarning, match=f"{nom}=10.0 dépasse le maximum {maximum}"):
        trop = moteur.synthesize_groups(GROUPES, seed=2, **{nom: 10.0})
    assert _empreinte(trop) == _empreinte(borne)
    with pytest.warns(UserWarning, match=nom):
        a = moteur.synthesize_phones(["b", "a"], **{nom: 10.0})
    b = moteur.synthesize_phones(["b", "a"], **{nom: maximum})
    assert np.array_equal(a.samples, b.samples)


# ---------------------------------------------------------------- phonèmes inconnus

def test_phoneme_inconnu_signale(chargements):
    moteur = ld.creer_engine(g2p=_g2p)
    with pytest.warns(UserWarning, match=r"Phonèmes inconnus.*'x'"):
        moteur.synthesize_groups([{"phones": ["b", "x", "a"], "boundary": "period"}], seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        moteur.synthesize_groups([{"phones": ["b", "a"], "boundary": "period"}], seed=1)
    assert moteur.phones_connus == frozenset({"#", "a", "b"})
