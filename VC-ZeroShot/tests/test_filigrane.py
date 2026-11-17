"""Marquage IA des sorties (AI Act, art. 50 §2) : filigrane AudioSeal posé dans OpenVoiceConverter.convert.

Les tests qui posent le filigrane demandent generator_base.onnx ($LECTURA_FILIGRANE_ONNX ou
~/.lectura/models/filigrane/) ; ceux de détection demandent detector_base.onnx ($LECTURA_FILIGRANE_DETECTEUR_ONNX ou
le même dossier). Sinon ils sont sautés (ces tests ne téléchargent rien).
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import lectura_vc_zeroshot as Z
from lectura_vc_zeroshot import _filigrane as F
from lectura_vc_zeroshot._openvoice import OV_SR, OpenVoiceConverter


def _voix(sr: int, d: float = 3.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * d)) / sr
    ph = 2 * np.pi * np.cumsum(140 * (1 + 0.08 * np.sin(2 * np.pi * 3 * t))) / sr
    s = sum(np.sin(k * ph) / k for k in range(1, 20)) * (0.5 * (1 + np.sin(2 * np.pi * 4 * t))) ** 2
    s += 0.01 * np.random.default_rng(0).standard_normal(len(t))
    return (s / np.abs(s).max() * amp).astype(np.float32)


@pytest.fixture
def generateur():
    if not F.chemin_generateur().exists():
        pytest.skip("generator_base.onnx absent (LECTURA_FILIGRANE_ONNX)")


def _detecteur_present():
    if not F.chemin_detecteur().exists():
        pytest.skip("detector_base.onnx absent (LECTURA_FILIGRANE_DETECTEUR_ONNX)")


def _convertisseur_factice(sortie: np.ndarray) -> OpenVoiceConverter:
    """Convertisseur sans modèles OpenVoice : la session VC rend `sortie` telle quelle."""
    c = object.__new__(OpenVoiceConverter)
    c.sr = OV_SR
    c.vc_sess = MagicMock()
    c.vc_sess.run.return_value = [sortie[None, None, :]]
    return c


def test_declaration():
    assert Z.MARQUE_IA is True
    assert Z.marquer_ia is F.marquer
    assert F.MESSAGE == [int(b) for b in f"{0x4C01:016b}"]


def test_textes_sans_corpus_ni_nom():
    for texte in (F.OUTIL, F.DESCRIPTION, F.CREDIT):
        assert "corpus" not in texte.lower() and "Syllabes" not in texte


def test_convert_pose_le_filigrane(generateur):
    brut = _voix(OV_SR)
    c = _convertisseur_factice(brut)
    sortie, sr = c.convert(brut, np.zeros((1, 256, 1), np.float32), np.zeros((1, 256, 1), np.float32), sr=OV_SR)
    assert sr == OV_SR and sortie.shape == brut.shape and sortie.dtype == np.float32
    assert np.array_equal(sortie, F.marquer(brut, OV_SR))
    assert np.sqrt(np.mean((sortie - brut) ** 2)) > 1e-4


def test_convert_sortie_detectee(generateur):
    _detecteur_present()
    brut = _voix(OV_SR)
    c = _convertisseur_factice(brut)
    sortie, sr = c.convert(brut, np.zeros((1, 256, 1), np.float32), np.zeros((1, 256, 1), np.float32), sr=OV_SR)
    assert F.marque_retrouvee(*F.detecter(sortie, sr))
    assert not F.marque_retrouvee(*F.detecter(brut, OV_SR))


@pytest.mark.parametrize("contenu", [None, b"faux modele"])
def test_modele_absent_ou_altere_refuse(tmp_path, monkeypatch, contenu):
    chemin = tmp_path / "generator_base.onnx"
    if contenu is not None:
        chemin.write_bytes(contenu)
    monkeypatch.setenv("LECTURA_FILIGRANE_ONNX", str(chemin))
    monkeypatch.setattr(F._Modele, "_session", None)
    brut = _voix(OV_SR, 1.0)
    c = _convertisseur_factice(brut)
    with pytest.raises(F.FiligraneIndisponible) as e:
        c.convert(brut, np.zeros((1, 256, 1), np.float32), np.zeros((1, 256, 1), np.float32), sr=OV_SR)
    assert "filigrane" in str(e.value)
    assert isinstance(e.value, RuntimeError)


def test_chemin_par_defaut(monkeypatch):
    monkeypatch.delenv("LECTURA_FILIGRANE_ONNX", raising=False)
    assert F.chemin_generateur() == Path.home() / ".lectura" / "models" / "filigrane" / "generator_base.onnx"


def test_metadonnees_wav():
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OV_SR)
        w.writeframes((_voix(OV_SR, 1.0) * 32767).astype("<i2").tobytes())
    m = F.metadonnees_wav(b.getvalue())
    meta = F.lire_metadonnees(m)
    assert meta["xmp_ia"] and meta["info"] and meta["pcm_sha256"] == F.lire_metadonnees(b.getvalue())["pcm_sha256"]


def test_court_d_un_seul_tenant(generateur):
    """Jusqu'à 30 s : pose d'un seul tenant (référence) ; au-delà : par blocs de 30 s, marque continue."""
    x = _voix(16000, 2.0)
    wm = F.filigrane_16k(x)
    assert np.allclose(F.marquer(x, 16000), np.clip(x + wm, -1, 1), atol=1e-6)


def test_long_par_blocs_detecte_partout(generateur):
    _detecteur_present()
    sr = 16000
    x = np.tile(_voix(sr, 5.0), 13)                     # 65 s : blocs de 30 + 30 + 5 s
    y = F.marquer(x, sr)
    assert y.shape == x.shape
    for debut in (0, 27, 31, 58, 61):                  # de part et d'autre des jointures
        assert F.marque_retrouvee(*F.detecter(y[debut * sr:(debut + 4) * sr], sr)), debut


# -- pas d'option pour désactiver le filigrane ; modèle manquant -> FiligraneIndisponible --------------------------


def test_aucune_option_pour_desactiver():
    import inspect
    from lectura_vc_zeroshot.engine import ZeroShotEngine
    for f in (OpenVoiceConverter.convert, ZeroShotEngine.convert, ZeroShotEngine.convert_from_se, Z.convertir):
        params = set(inspect.signature(f).parameters)
        assert not {p for p in params if any(m in p.lower() for m in ("filigrane", "watermark", "marqu", "mark"))}, f


@pytest.mark.parametrize("variable", ["LECTURA_FILIGRANE", "LECTURA_SANS_FILIGRANE", "LECTURA_FILIGRANE_DESACTIVE"])
def test_variables_sans_effet(tmp_path, monkeypatch, variable):
    """Aucune variable ne permet d'obtenir un son non marqué quand le modèle manque."""
    monkeypatch.setenv(variable, "0")
    monkeypatch.setenv("LECTURA_FILIGRANE_ONNX", str(tmp_path / "absent.onnx"))
    monkeypatch.setattr(F._Modele, "_session", None)
    c = _convertisseur_factice(_voix(OV_SR, 1.0))
    with pytest.raises(F.FiligraneIndisponible):
        c.convert(_voix(OV_SR, 1.0), np.zeros((1, 256, 1), np.float32), np.zeros((1, 256, 1), np.float32), sr=OV_SR)


def test_dossier_par_defaut_vide_hors_ligne(tmp_path, monkeypatch):
    """Chemin par défaut, modèle absent, téléchargement interdit -> FiligraneIndisponible."""
    monkeypatch.delenv("LECTURA_FILIGRANE_ONNX", raising=False)
    monkeypatch.setenv("LECTURA_HORS_LIGNE", "1")
    monkeypatch.setattr(F, "DOSSIER_MODELES", tmp_path / "filigrane")
    monkeypatch.setattr(F._Modele, "_session", None)
    with pytest.raises(F.FiligraneIndisponible, match="LECTURA_HORS_LIGNE"):
        F.marquer(_voix(OV_SR, 1.0), OV_SR)


def test_telecharge_altere_refuse(tmp_path, monkeypatch):
    """Faux Hugging Face qui sert un mauvais générateur : refusé (empreinte), FiligraneIndisponible."""
    from conftest import faux_hf
    from lectura_vc_zeroshot import _poids
    d = tmp_path / "depot" / "filigrane"
    d.mkdir(parents=True)
    taille = _poids.FICHIERS["filigrane/generator_base.onnx"][1]
    with open(d / "generator_base.onnx", "wb") as f:
        f.truncate(taille)                       # bonne taille, mauvais contenu
    monkeypatch.delenv("LECTURA_FILIGRANE_ONNX", raising=False)
    monkeypatch.delenv("LECTURA_HORS_LIGNE", raising=False)
    monkeypatch.setattr(F, "DOSSIER_MODELES", tmp_path / "filigrane")
    monkeypatch.setattr(F._Modele, "_session", None)
    with faux_hf(tmp_path / "depot", tmp_path / "hf") as base:
        monkeypatch.setenv("LECTURA_HF_BASE", base)
        with pytest.raises(F.FiligraneIndisponible):
            F.marquer(_voix(OV_SR, 1.0), OV_SR)
    assert not (tmp_path / "filigrane" / "generator_base.onnx").exists()
