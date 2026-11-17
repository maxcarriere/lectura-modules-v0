"""Tests pour engine.py --- ZeroShotEngine (sans modèles réels : convertisseur simulé)."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from conftest import openvoice_factices, presets_factices
from lectura_vc_zeroshot import _poids
from lectura_vc_zeroshot._openvoice import OV_SR
from lectura_vc_zeroshot._presets import PresetInconnu
from lectura_vc_zeroshot.engine import ZeroShotEngine, blend_se, _is_se


@pytest.fixture(autouse=True)
def _isolement(tmp_path, monkeypatch):
    """Aucun accès réseau ni au dossier de l'utilisateur."""
    monkeypatch.setenv("LECTURA_HORS_LIGNE", "1")
    monkeypatch.setenv("LECTURA_MODELS_DIR", str(tmp_path / "lectura_models"))


@pytest.fixture
def models_dir(tmp_path):
    return openvoice_factices(tmp_path / "vc")


@pytest.fixture
def vecteurs(models_dir):
    return presets_factices(models_dir)


def _moteur(models_dir, se=None):
    engine = ZeroShotEngine(models_dir=models_dir)
    conv = MagicMock()
    conv.extract_se.return_value = np.zeros((1, 256, 1), np.float32) if se is None else se
    conv.convert.return_value = (np.zeros(OV_SR, np.float32), OV_SR)
    engine._converter = conv
    return engine, conv


class TestZeroShotEngineInit:
    def test_basic_init(self, models_dir):
        engine = ZeroShotEngine(models_dir=models_dir)
        assert engine.models_dir == models_dir
        assert engine._converter is None  # lazy

    def test_dossier_par_defaut(self, tmp_path):
        d = openvoice_factices(tmp_path / "lectura_models" / "vc")
        assert ZeroShotEngine().models_dir == d

    def test_modeles_absents_hors_ligne(self, tmp_path):
        """Modèles absents et téléchargement interdit -> erreur explicite, marche à suivre."""
        with pytest.raises(_poids.LecturaModelesIndisponibles, match="LECTURA_HORS_LIGNE") as e:
            ZeroShotEngine(models_dir=tmp_path / "vide")
        assert "openvoice_vc.onnx" in str(e.value) and _poids.DEPOT_HF in str(e.value)

    def test_modele_tronque_hors_ligne(self, models_dir):
        """Fichier de taille inattendue = manquant."""
        (models_dir / "openvoice_vc.onnx").write_bytes(b"x")
        with pytest.raises(_poids.LecturaModelesIndisponibles):
            ZeroShotEngine(models_dir=models_dir)


class TestConvert:
    def test_convert_reference_audio(self, models_dir):
        """Une référence audio (ndarray) fournie par l'utilisateur est acceptée (zero-shot)."""
        engine, conv = _moteur(models_dir)
        audio = np.random.randn(16000).astype(np.float32)
        ref = np.random.randn(16000).astype(np.float32)
        _, sr = engine.convert(audio, ref, sr_in=16000, ref_sr=16000)
        assert sr == OV_SR
        assert conv.extract_se.call_count == 2  # src + tgt
        assert conv.convert.call_count == 1

    def test_convert_fichier_audio(self, models_dir, tmp_path):
        """Un chemin de fichier audio existant est accepté comme référence."""
        engine, conv = _moteur(models_dir)
        f = tmp_path / "ma_voix.wav"
        f.write_bytes(b"RIFF")
        engine.convert(np.zeros(16000, np.float32), str(f), sr_in=16000)
        assert conv.extract_se.call_count == 2
        assert conv.extract_se.call_args_list[1].args[0] == str(f)

    def test_convert_with_sr_override(self, models_dir):
        engine, conv = _moteur(models_dir)
        audio = np.random.randn(16000).astype(np.float32)
        ref = np.random.randn(16000).astype(np.float32)
        _, sr = engine.convert(audio, ref, sr_in=16000, ref_sr=16000, sr_override=11025)
        assert sr == OV_SR and conv.convert.call_count == 1

    def test_convert_preregle(self, models_dir, vecteurs):
        engine, conv = _moteur(models_dir)
        engine.convert(np.zeros(16000, np.float32), reference="melange-h", sr_in=16000)
        assert conv.extract_se.call_count == 1  # source seulement : la cible vient du préréglage
        np.testing.assert_array_equal(conv.convert.call_args.args[2], vecteurs["melange-h"])

    def test_convert_dict(self, models_dir, vecteurs):
        engine, conv = _moteur(models_dir)
        engine.convert(np.zeros(16000, np.float32), reference={"fr-f1": 0.5, "melange-f": 0.5}, sr_in=16000)
        np.testing.assert_allclose(conv.convert.call_args.args[2],
                                   (vecteurs["fr-f1"] + vecteurs["melange-f"]) / 2, atol=1e-6)

    def test_convert_from_se_delegates(self, models_dir):
        engine, conv = _moteur(models_dir)
        _, sr = engine.convert_from_se(np.zeros(16000, np.float32), np.zeros((1, 256, 1), np.float32),
                                       np.ones((1, 256, 1), np.float32), sr_in=16000)
        assert sr == OV_SR
        conv.convert.assert_called_once()

    def test_extract_se_delegates(self, models_dir):
        engine, conv = _moteur(models_dir, se=np.ones((1, 256, 1), np.float32))
        se = engine.extract_se(np.zeros(16000, np.float32), sr=16000)
        np.testing.assert_array_equal(se, np.ones((1, 256, 1), np.float32))


class TestResolveTargetSe:
    def test_preregles(self, models_dir, vecteurs):
        engine, _ = _moteur(models_dir)
        for nom, v in ((k, x) for k, x in vecteurs.items() if not k.endswith("-variantes")):
            np.testing.assert_array_equal(engine.resolve_target_se(nom), v)

    def test_preregle_curseur(self, models_dir, vecteurs):
        """Préréglage + curseur : vecteurs précalculés (aucune extraction) ; sr_override et variante équivalents."""
        engine, conv = _moteur(models_dir)
        t = vecteurs["fr-f1-variantes"]
        np.testing.assert_array_equal(engine.resolve_target_se("fr-f1", sr_override=11025), t[8])   # +1
        np.testing.assert_array_equal(engine.resolve_target_se("fr-f1", sr_override=44100), t[0])   # -1
        np.testing.assert_array_equal(engine.resolve_target_se("fr-f1", variante=-0.5), t[2])
        np.testing.assert_array_equal(engine.resolve_target_se("fr-f1", variante=0.0), vecteurs["fr-f1"])
        np.testing.assert_array_equal(engine.resolve_target_se("fr-f1", sr_override=22050), vecteurs["fr-f1"])
        np.testing.assert_allclose(engine.resolve_target_se("fr-f1", sr_override=15592),
                                   engine.resolve_target_se("fr-f1", variante=0.5), atol=1e-3)
        assert conv.extract_se.call_count == 0

    def test_convert_variante(self, models_dir, vecteurs):
        engine, conv = _moteur(models_dir)
        engine.convert(np.zeros(16000, np.float32), "melange-h", sr_in=16000, variante=1.0)
        np.testing.assert_array_equal(conv.convert.call_args.args[2], vecteurs["melange-h-variantes"][8])

    def test_reference_audio_variante(self, models_dir, monkeypatch):
        """Référence audio + variante : truc SR à la volée (sr_override = 22050 / 2**variante)."""
        engine, _ = _moteur(models_dir)
        vus = []
        monkeypatch.setattr(engine, "_extract_se_with_sr_trick",
                            lambda ref, ref_sr, sro: vus.append(sro) or np.zeros((1, 256, 1), np.float32))
        engine.resolve_target_se(np.random.randn(16000).astype(np.float32), ref_sr=16000, variante=1.0)
        engine.resolve_target_se(np.random.randn(16000).astype(np.float32), ref_sr=16000, variante=-0.5)
        assert vus == [11025, 31183]

    def test_conversions_variante_sr(self):
        from lectura_vc_zeroshot.engine import sr_vers_variante, variante_vers_sr
        assert variante_vers_sr(0.0) is None and sr_vers_variante(None) == 0.0 and sr_vers_variante(22050) == 0.0
        assert variante_vers_sr(1.0) == 11025 and variante_vers_sr(-1.0) == 44100
        assert sr_vers_variante(11025) == pytest.approx(1.0) and sr_vers_variante(44100) == pytest.approx(-1.0)

    @pytest.mark.parametrize("nom", ["voix-retiree", "Fr-F1", "fr_f1", "f1", "melange"])
    def test_nom_inconnu_refuse(self, models_dir, vecteurs, nom):
        """Un nom qui n'est ni un préréglage ni un fichier existant est refusé, sans lecture ni téléchargement."""
        engine, conv = _moteur(models_dir)
        with pytest.raises(PresetInconnu) as e:
            engine.resolve_target_se(nom)
        assert "fr-f1, melange-f, melange-h" in str(e.value)
        assert conv.extract_se.call_count == 0

    def test_nom_inconnu_dans_un_melange(self, models_dir, vecteurs):
        engine, _ = _moteur(models_dir)
        with pytest.raises(PresetInconnu):
            engine.resolve_target_se({"fr-f1": 0.5, "voix-retiree": 0.5})

    def test_direct_se(self, models_dir):
        engine, _ = _moteur(models_dir)
        direct = np.random.randn(1, 256, 1).astype(np.float32)
        np.testing.assert_array_equal(engine.resolve_target_se(direct), direct)

    def test_list(self, models_dir, vecteurs):
        engine, _ = _moteur(models_dir)
        se = engine.resolve_target_se(["fr-f1", "melange-h"])
        np.testing.assert_allclose(se, blend_se([vecteurs["fr-f1"], vecteurs["melange-h"]]), atol=1e-6)

    def test_audio_array(self, models_dir):
        engine, conv = _moteur(models_dir, se=np.ones((1, 256, 1), np.float32))
        result = engine.resolve_target_se(np.random.randn(16000).astype(np.float32), ref_sr=16000)
        np.testing.assert_array_equal(result, np.ones((1, 256, 1), np.float32))


class TestBlendSe:
    def test_blend_equal_weights(self):
        se1 = np.ones((1, 256, 1), dtype=np.float32) * 2
        se2 = np.ones((1, 256, 1), dtype=np.float32) * 4
        np.testing.assert_allclose(blend_se([se1, se2]), np.full((1, 256, 1), 3.0), atol=1e-6)

    def test_blend_custom_weights(self):
        se1 = np.zeros((1, 256, 1), dtype=np.float32)
        se2 = np.ones((1, 256, 1), dtype=np.float32) * 10
        np.testing.assert_allclose(blend_se([se1, se2], weights=[0.8, 0.2]), se2 * 0.2, atol=1e-6)

    def test_blend_single(self):
        se = np.random.randn(1, 256, 1).astype(np.float32)
        np.testing.assert_array_equal(blend_se([se]), se)

    def test_blend_mismatched_lengths_raises(self):
        se = np.zeros((1, 256, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Nombre de poids"):
            blend_se([se, se], weights=[1.0])


class TestIsSe:
    def test_shapes(self):
        assert _is_se(np.zeros((1, 256, 1))) is True
        assert _is_se(np.zeros(16000)) is False
        assert _is_se(np.zeros((256,))) is False
        assert _is_se(np.zeros((1, 128, 1))) is False


class TestSrOverrideFactor:
    def test_factors(self):
        assert OV_SR / 11025 == pytest.approx(2.0)
        assert OV_SR / 44100 == pytest.approx(0.5)


def test_entree_quelconque_documentee():
    """convert accepte tout son (synthèse ou enregistrement) : c'est dit dans sa docstring."""
    doc = ZeroShotEngine.convert.__doc__
    assert "quel qu'il soit" in doc and "synthese" in doc and "variante" in doc


@pytest.mark.parametrize("sr_in", [8000, 16000, 22050, 44100, 48000])
def test_entree_toute_frequence(models_dir, vecteurs, sr_in):
    engine, conv = _moteur(models_dir)
    son = np.random.default_rng(0).standard_normal(sr_in).astype(np.float32) * 0.1
    _, sr = engine.convert(son, "fr-f1", sr_in=sr_in)
    assert sr == OV_SR
    assert conv.extract_se.call_args.kwargs.get("sr") == sr_in and conv.convert.call_args.kwargs.get("sr") == sr_in
