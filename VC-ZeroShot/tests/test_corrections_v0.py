"""Corrections de la v0 (23/11/2026) : tau sans effet, détection et métadonnées publiques,
dossier des préréglages du moteur.

Les tests qui posent ou détectent le filigrane demandent les modèles AudioSeal (voir
test_filigrane.py) ; sinon ils sont sautés. Aucun téléchargement.
"""

from __future__ import annotations

import io
import warnings
import wave
from unittest.mock import MagicMock

import numpy as np
import pytest

import lectura_vc_zeroshot as Z
from conftest import openvoice_factices, presets_factices
from lectura_vc_zeroshot import _filigrane as F
from lectura_vc_zeroshot import _poids
from lectura_vc_zeroshot._openvoice import OV_SR, OpenVoiceConverter
from lectura_vc_zeroshot.engine import ZeroShotEngine

SE = np.zeros((1, 256, 1), np.float32)


@pytest.fixture(autouse=True)
def _isolement(tmp_path, monkeypatch):
    monkeypatch.setenv("LECTURA_HORS_LIGNE", "1")
    monkeypatch.setenv("LECTURA_MODELS_DIR", str(tmp_path / "defaut"))


def _voix(sr: int, d: float = 2.0) -> np.ndarray:
    t = np.arange(int(sr * d)) / sr
    s = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 15)) * (0.5 * (1 + np.sin(2 * np.pi * 4 * t)))
    s += 0.01 * np.random.default_rng(0).standard_normal(len(t))
    return (s / np.abs(s).max() * 0.3).astype(np.float32)


def _moteur(tmp_path):
    e = ZeroShotEngine(models_dir=openvoice_factices(tmp_path / "vc"))
    conv = MagicMock()
    conv.extract_se.return_value = SE
    conv.convert.return_value = (np.zeros(OV_SR, np.float32), OV_SR)
    e._converter = conv
    return e, conv


# ---------------------------------------------------------------- 1. tau : sans effet, DeprecationWarning

def test_tau_donne_avertit_et_n_est_pas_transmis(tmp_path):
    e, conv = _moteur(tmp_path)
    with pytest.warns(DeprecationWarning, match="sans effet"):
        e.convert(np.zeros(OV_SR, np.float32), SE, sr_in=OV_SR, tau=0.9)
    assert "tau" not in conv.convert.call_args.kwargs
    with pytest.warns(DeprecationWarning, match="sans effet"):
        e.convert_from_se(np.zeros(OV_SR, np.float32), SE, SE, sr_in=OV_SR, tau=0.0)
    assert "tau" not in conv.convert.call_args.kwargs


def test_sans_tau_aucun_avertissement(tmp_path):
    e, _ = _moteur(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        e.convert(np.zeros(OV_SR, np.float32), SE, sr_in=OV_SR)
        e.convert_from_se(np.zeros(OV_SR, np.float32), SE, SE, sr_in=OV_SR)


def test_convertir_tau_un_seul_avertissement(tmp_path, monkeypatch):
    e, conv = _moteur(tmp_path)
    monkeypatch.setattr(Z, "creer_engine", lambda models_dir=None: e)
    with pytest.warns(DeprecationWarning) as releves:
        Z.convertir(np.zeros(OV_SR, np.float32), SE, sr_in=OV_SR, tau=0.5)
    assert len([w for w in releves if "tau" in str(w.message)]) == 1
    assert releves[0].filename == __file__   # l'avertissement pointe vers l'appelant


def test_signatures_tau_none():
    import inspect
    for f in (OpenVoiceConverter.convert, ZeroShotEngine.convert, ZeroShotEngine.convert_from_se, Z.convertir):
        assert inspect.signature(f).parameters["tau"].default is None, f


def test_convertisseur_tau_absent_du_modele_et_filigrane_pose(monkeypatch):
    """OpenVoiceConverter.convert : tau n'atteint pas la session ONNX ; la sortie passe toujours par le filigrane."""
    brut = _voix(OV_SR)
    c = object.__new__(OpenVoiceConverter)
    c.sr = OV_SR
    c.vc_sess = MagicMock()
    c.vc_sess.run.return_value = [brut[None, None, :]]
    marques = []
    monkeypatch.setattr(F, "marquer", lambda a, sr, message=None: marques.append(sr) or a + 0.5)
    with pytest.warns(DeprecationWarning, match="sans effet"):
        a, _ = c.convert(brut, SE, SE, sr=OV_SR, tau=0.3)
    entrees = c.vc_sess.run.call_args.args[1]
    assert set(entrees) == {"spec", "spec_lengths", "sid_src", "sid_tgt"}
    b, _ = c.convert(brut, SE, SE, sr=OV_SR)
    assert marques == [OV_SR, OV_SR] and np.array_equal(a, b) and np.allclose(a, brut + 0.5)


# ---------------------------------------------------------------- 2. détection et métadonnées publiques

def test_detection_et_metadonnees_exportees():
    for nom in ("detecter", "marque_retrouvee", "metadonnees_wav"):
        assert nom in Z.__all__
        assert getattr(Z, nom) is getattr(F, nom)
    from lectura_vc_zeroshot import detecter, marque_retrouvee, metadonnees_wav  # noqa: F401


def test_marque_retrouvee_regle():
    assert Z.marque_retrouvee(0.9, F.MESSAGE)
    assert not Z.marque_retrouvee(0.4, F.MESSAGE)
    assert not Z.marque_retrouvee(0.9, [1 - b for b in F.MESSAGE])


def test_metadonnees_wav_public_audio_inchange():
    pcm = (np.sin(np.arange(8000) / 10) * 8000).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    brut = buf.getvalue()
    avant = F.lire_metadonnees(brut)
    assert not avant["xmp_ia"] and not avant["info"]
    apres = F.lire_metadonnees(Z.metadonnees_wav(brut))
    assert apres["xmp_ia"] and apres["info"] and apres["pcm_sha256"] == avant["pcm_sha256"]


def test_conversion_ne_pose_pas_de_metadonnees(tmp_path):
    """Les conversions rendent un tableau numpy (filigrane seul) : les métadonnées WAV ne sont pas automatiques."""
    e, _ = _moteur(tmp_path)
    audio, sr = e.convert(np.zeros(OV_SR, np.float32), SE, sr_in=OV_SR)
    assert isinstance(audio, np.ndarray) and sr == OV_SR


def test_detecter_public_retrouve_la_marque():
    if not (F.chemin_generateur().exists() and F.chemin_detecteur().exists()):
        pytest.skip("modèles AudioSeal absents (LECTURA_FILIGRANE_ONNX / LECTURA_FILIGRANE_DETECTEUR_ONNX)")
    brut = _voix(OV_SR)
    marque = Z.marquer_ia(brut, OV_SR)
    assert Z.marque_retrouvee(*Z.detecter(marque, OV_SR))
    assert not Z.marque_retrouvee(*Z.detecter(brut, OV_SR))


# ---------------------------------------------------------------- 3. préréglages : dossier du moteur

def test_get_preset_se_lit_le_dossier_du_moteur(tmp_path):
    d = openvoice_factices(tmp_path / "vc_moteur")
    vecteurs = presets_factices(d)
    e = ZeroShotEngine(models_dir=d)
    assert np.array_equal(e.get_preset_se("fr-f1"), vecteurs["fr-f1"])
    attendu = 0.25 * vecteurs["fr-f1"] + 0.75 * vecteurs["melange-h"]
    assert np.allclose(e.blend_preset_se({"fr-f1": 1, "melange-h": 3}), attendu)


def test_get_preset_se_sur_la_classe_dossier_par_defaut(tmp_path):
    d = openvoice_factices(tmp_path / "vc_moteur")
    presets_factices(d)
    # dossier par défaut ($LECTURA_MODELS_DIR/vc) vide, hors ligne -> erreur
    with pytest.raises(_poids.LecturaModelesIndisponibles):
        ZeroShotEngine.get_preset_se("fr-f1")
    with pytest.raises(_poids.LecturaModelesIndisponibles):
        ZeroShotEngine.blend_preset_se({"fr-f1": 1.0})
    # dossier explicite accepté sur la classe comme sur l'instance
    assert ZeroShotEngine.get_preset_se("melange-f", models_dir=d).shape == (1, 256, 1)
    assert ZeroShotEngine.available_presets() == Z.PRESET_SPEAKERS
