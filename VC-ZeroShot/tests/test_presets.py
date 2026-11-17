"""Préréglages et téléchargement depuis un faux Hugging Face local (LECTURA_HF_BASE)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import lectura_vc_zeroshot as Z
from conftest import faux_hf, table_variantes, vecteur
from lectura_vc_zeroshot import _poids
from lectura_vc_zeroshot import _presets as P


@pytest.fixture(autouse=True)
def _isolement(tmp_path, monkeypatch):
    monkeypatch.setenv("LECTURA_MODELS_DIR", str(tmp_path / "lectura_models"))
    monkeypatch.delenv("LECTURA_HORS_LIGNE", raising=False)
    monkeypatch.delenv("LECTURA_HF_BASE", raising=False)


@pytest.fixture
def depot(tmp_path, monkeypatch):
    """Faux dépôt avec trois vecteurs ; la table d'empreintes est remplacée par la leur."""
    d = tmp_path / "depot"
    (d / "presets").mkdir(parents=True)
    table = dict(_poids.FICHIERS)
    vecs = {}
    for i, nom in enumerate(P.PRESET_SPEAKERS):
        f = d / "presets" / f"{nom}.npy"
        vecs[nom] = vecteur(10 + i)
        np.save(f, vecs[nom])
        table[f"presets/{nom}.npy"] = (hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_size)
        f = d / "presets" / f"{nom}-variantes.npy"
        vecs[f"{nom}-variantes"] = table_variantes(vecs[nom], i)
        np.save(f, vecs[f"{nom}-variantes"])
        table[f"presets/{nom}-variantes.npy"] = (hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_size)
    monkeypatch.setattr(_poids, "FICHIERS", table)
    with faux_hf(d, tmp_path / "hf") as base:
        monkeypatch.setenv("LECTURA_HF_BASE", base)
        yield d, vecs


def test_exactement_trois_preregles():
    assert Z.PRESET_SPEAKERS == ["fr-f1", "melange-f", "melange-h"]
    assert P.list_presets() == Z.PRESET_SPEAKERS
    assert Z.ZeroShotEngine.available_presets() == Z.PRESET_SPEAKERS
    assert sorted(c for c in _poids.FICHIERS if c.startswith("presets/")) == [
        "presets/fr-f1-variantes.npy", "presets/fr-f1.npy", "presets/melange-f-variantes.npy",
        "presets/melange-f.npy", "presets/melange-h-variantes.npy", "presets/melange-h.npy"]
    assert all(c.endswith(".npy") for c in _poids.FICHIERS if c.startswith("presets/"))


def test_aucun_fichier_lourd_dans_le_paquet():
    from pathlib import Path
    racine = Path(Z.__file__).parent
    lourds = [p for ext in ("*.npy", "*.wav", "*.onnx", "*.pth", "*.enc") for p in racine.rglob(ext)]
    assert lourds == []


@pytest.mark.parametrize("nom", ["voix-retiree", "", "FR-F1", "../fr-f1", "presets/fr-f1"])
def test_nom_inconnu_refuse(nom):
    assert not P.has_preset(nom)
    with pytest.raises(P.PresetInconnu, match="Préréglages disponibles"):
        P.load_preset(nom)


def test_telechargement_au_premier_usage(depot, tmp_path, capsys):
    _, vecs = depot
    for nom in P.PRESET_SPEAKERS:
        se = P.load_preset(nom)
        assert se.shape == (1, 256, 1) and se.dtype == np.float32
        np.testing.assert_array_equal(se, vecs[nom])
    dossier = tmp_path / "lectura_models" / "vc" / "presets"
    # variante 0 : seul le vecteur du préréglage est téléchargé, pas la table du curseur
    assert sorted(p.name for p in dossier.iterdir()) == ["fr-f1.npy", "melange-f.npy", "melange-h.npy"]
    assert "Licences" in capsys.readouterr().err
    # deuxième appel : rien n'est retéléchargé
    assert P.load_preset("fr-f1").shape == (1, 256, 1)
    assert "Téléchargement" not in capsys.readouterr().err


def test_blend_preregles(depot):
    _, vecs = depot
    np.testing.assert_allclose(P.blend_presets({"fr-f1": 1, "melange-h": 3}),
                               vecs["fr-f1"] * 0.25 + vecs["melange-h"] * 0.75, atol=1e-6)


def test_empreinte_fausse_refusee(depot, tmp_path, monkeypatch):
    d, _ = depot
    table = dict(_poids.FICHIERS)
    sha, taille = table["presets/melange-f.npy"]
    table["presets/melange-f.npy"] = ("0" * 64, taille)
    monkeypatch.setattr(_poids, "FICHIERS", table)
    with pytest.raises(_poids.LecturaModelesIndisponibles, match="refusé"):
        P.load_preset("melange-f")
    dossier = tmp_path / "lectura_models" / "vc" / "presets"
    assert list(dossier.iterdir()) == []  # ni fichier installé ni fichier temporaire laissé


def test_hors_ligne(monkeypatch):
    monkeypatch.setenv("LECTURA_HORS_LIGNE", "1")
    with pytest.raises(_poids.LecturaModelesIndisponibles, match="LECTURA_HORS_LIGNE"):
        P.load_preset("fr-f1")


def test_serveur_injoignable(monkeypatch):
    monkeypatch.setenv("LECTURA_HF_BASE", "http://127.0.0.1:9")
    with pytest.raises(_poids.LecturaModelesIndisponibles, match="Échec du téléchargement"):
        P.load_preset("melange-h")


def test_url_et_revision(monkeypatch):
    monkeypatch.setenv("LECTURA_HF_BASE", "https://miroir.example/")
    assert _poids.url_fichier("presets/fr-f1.npy") == (
        f"https://miroir.example/{_poids.DEPOT_HF}/resolve/{_poids.REVISION_HF}/presets/fr-f1.npy")
    assert _poids.DEPOT_HF == "lecturaworld/lectura-vc-zeroshot-v0"


def test_table_coherente():
    for c, (sha, taille) in _poids.FICHIERS.items():
        assert len(sha) == 64 and int(sha, 16) >= 0 and taille > 0, c
    assert set(c.split("/")[0] for c in _poids.FICHIERS) == {"openvoice", "filigrane", "presets"}


def test_modeles_openvoice_telecharges(tmp_path, monkeypatch):
    """Les modèles OpenVoice suivent le même chemin (petits fichiers factices à la place des vrais)."""
    d = tmp_path / "depot_ov"
    (d / "openvoice").mkdir(parents=True)
    table = dict(_poids.FICHIERS)
    for c in _poids.OPENVOICE:
        f = d / c
        f.write_bytes(c.encode() * 3)
        table[c] = (hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_size)
    monkeypatch.setattr(_poids, "FICHIERS", table)
    with faux_hf(d, tmp_path / "hf2") as base:
        monkeypatch.setenv("LECTURA_HF_BASE", base)
        e = Z.ZeroShotEngine()
    assert e.models_dir == tmp_path / "lectura_models" / "vc"
    assert all(_poids.verifier(e.models_dir, _poids.OPENVOICE).values())


# -- curseur grave/aigu ------------------------------------------------------------------------------------------


def test_positions():
    assert P.POSITIONS_VARIANTE == (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)


def test_variante_zero_identique(depot):
    _, vecs = depot
    for nom in P.PRESET_SPEAKERS:
        np.testing.assert_array_equal(P.load_preset(nom, variante=0.0), vecs[nom])
        np.testing.assert_array_equal(P.load_preset(nom, variante=0.0), P.load_preset(nom))


@pytest.mark.parametrize("i", range(9))
def test_variante_positions_exactes(depot, i):
    _, vecs = depot
    t = vecs["melange-h-variantes"]
    np.testing.assert_array_equal(P.load_preset("melange-h", variante=P.POSITIONS_VARIANTE[i]), t[i])


def test_variante_interpolee(depot):
    _, vecs = depot
    t = vecs["fr-f1-variantes"].astype(np.float64)
    se = P.load_preset("fr-f1", variante=0.1).astype(np.float64)       # entre 0 (i=4) et 0,25 (i=5), t = 0,4
    m = 0.6 * t[4] + 0.4 * t[5]
    assert np.dot(se.ravel(), m.ravel()) / np.linalg.norm(se) / np.linalg.norm(m) == pytest.approx(1, abs=1e-6)
    assert np.linalg.norm(se) == pytest.approx(0.6 * np.linalg.norm(t[4]) + 0.4 * np.linalg.norm(t[5]), rel=1e-5)


def test_variante_continue_pres_de_zero(depot):
    a, b = P.load_preset("melange-f", variante=1e-4), P.load_preset("melange-f")
    assert np.abs(a - b).max() < 1e-2 * np.abs(b).max()


def test_variante_bornee(depot, caplog):
    _, vecs = depot
    np.testing.assert_array_equal(P.load_preset("fr-f1", variante=3.0), vecs["fr-f1-variantes"][8])
    np.testing.assert_array_equal(P.load_preset("fr-f1", variante=-2.0), vecs["fr-f1-variantes"][0])
    assert "bornée" in caplog.text
    with pytest.raises(ValueError):
        P.load_preset("fr-f1", variante=float("nan"))
