"""Passes lourdes : modèles réels téléchargés depuis un faux Hugging Face, conversion, filigrane.

Sautées par défaut. Mémoire : environ 1 Go au pic. Exemple :

    LECTURA_PASSES=1 LECTURA_VC_HF_LOCAL=/chemin/lectura-vc-zeroshot-v0 \\
    systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 -p CPUQuota=100% \\
        python -m pytest tests/test_passes.py -v

Variables :
- LECTURA_PASSES=1 : active les passes ;
- LECTURA_VC_HF_LOCAL : dossier préparé du dépôt Hugging Face ; un serveur
  HTTP local l'expose, et tout est téléchargé (modèles OpenVoice, filigrane,
  préréglages) dans un dossier temporaire, avec vérification SHA-256 ;
- LECTURA_VC_DEV1_SRC : dossier src/ de lectura-vc-zeroshot 1.3.0.dev1, pour
  vérifier que la sortie fr-f1 est identique (sautée sinon).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import faux_hf

pytestmark = pytest.mark.skipif(
    os.environ.get("LECTURA_PASSES") != "1" or not os.environ.get("LECTURA_VC_HF_LOCAL"),
    reason="passe lourde : LECTURA_PASSES=1 et LECTURA_VC_HF_LOCAL, avec l'accord du guichet mémoire",
)

SR = 22050


def _voix(sr: int, d: float = 4.0, amp: float = 0.3) -> np.ndarray:
    """Voix synthétique (harmoniques, f0 et amplitude modulées), déterministe."""
    t = np.arange(int(sr * d)) / sr
    f0 = 130 * (1 + 0.15 * np.sin(2 * np.pi * 0.7 * t))
    ph = 2 * np.pi * np.cumsum(f0) / sr
    s = sum(np.sin(k * ph) / k for k in range(1, 25)) * (0.5 * (1 + np.sin(2 * np.pi * 3.5 * t))) ** 2
    s += 0.005 * np.random.default_rng(0).standard_normal(len(t))
    return (s / np.abs(s).max() * amp).astype(np.float32)


@pytest.fixture(scope="module")
def installation(tmp_path_factory):
    """Télécharge tout depuis le faux Hugging Face : (dossier vc, dossier filigrane)."""
    from lectura_vc_zeroshot import _filigrane as F
    from lectura_vc_zeroshot import _poids

    racine = tmp_path_factory.mktemp("hf")
    sauve = {k: os.environ.get(k) for k in ("LECTURA_HF_BASE", "LECTURA_FILIGRANE_ONNX",
                                            "LECTURA_FILIGRANE_DETECTEUR_ONNX", "LECTURA_HORS_LIGNE")}
    ancien_dossier = F.DOSSIER_MODELES
    vc = tmp_path_factory.mktemp("modeles") / "vc"
    fil = vc.parent / "filigrane"
    with faux_hf(Path(os.environ["LECTURA_VC_HF_LOCAL"]), racine) as base:
        for k in sauve:
            os.environ.pop(k, None)
        os.environ["LECTURA_HF_BASE"] = base
        F.DOSSIER_MODELES = fil
        F._Modele._session = None
        F._Detecteur._session = None
        try:
            _poids.assurer_openvoice(vc)
            from lectura_vc_zeroshot._presets import PRESET_SPEAKERS, load_preset
            for nom in PRESET_SPEAKERS:
                load_preset(nom, vc)
                load_preset(nom, vc, variante=0.5)             # table du curseur
            F._Modele.session()                       # télécharge et vérifie le générateur
            F.detecter(_voix(16000, 1.0), 16000)      # télécharge et vérifie le détecteur
            yield vc, fil
        finally:
            F.DOSSIER_MODELES = ancien_dossier
            F._Modele._session = None
            F._Detecteur._session = None
            for k, v in sauve.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def test_tout_telecharge_et_verifie(installation):
    from lectura_vc_zeroshot import _poids
    vc, fil = installation
    assert all(_poids.verifier(vc, _poids.OPENVOICE).values())
    pre = [c for c in _poids.FICHIERS if c.startswith("presets/")]
    assert len(pre) == 6 and all(_poids.verifier(vc / "presets", pre).values())
    assert all(_poids.verifier(fil, [c for c in _poids.FICHIERS if c.startswith("filigrane/")]).values())


def test_fr_f1_identique_a_la_dev1(installation, tmp_path):
    """fr-f1 : même sortie que lectura-vc-zeroshot 1.3.0.dev1 (même vecteur), filigrane compris et hors filigrane.
    Lancé avant le chargement du moteur dans ce processus (mémoire)."""
    dev1 = os.environ.get("LECTURA_VC_DEV1_SRC")
    if not dev1:
        pytest.skip("LECTURA_VC_DEV1_SRC non défini")
    vc, fil = installation
    entree = tmp_path / "entree.npy"
    np.save(entree, _voix(SR))
    ref = tmp_path / "dev1.npz"
    code = (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, {dev1!r})\n"
        "import lectura_vc_zeroshot as Z\n"
        "assert Z.__version__ == '1.3.0.dev1', Z.__version__\n"
        "from lectura_vc_zeroshot import _filigrane as F\n"
        "e = Z.creer_engine(models_dir=sys.argv[1])\n"
        "x = np.load(sys.argv[2]); se = np.load(sys.argv[3])\n"
        "marque, _ = e.convert(x, se, sr_in=22050, tau=0.0)\n"
        "F.marquer = lambda a, sr, message=None: a\n"
        "brut, _ = e.convert(x, se, sr_in=22050, tau=0.0)\n"
        "np.savez(sys.argv[4], marque=marque, brut=brut)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["LECTURA_FILIGRANE_ONNX"] = str(fil / "generator_base.onnx")
    subprocess.run([sys.executable, "-I", "-c", code, str(vc), str(entree), str(vc / "presets" / "fr-f1.npy"),
                    str(ref)], check=True, env=env)
    attendu = np.load(ref)

    import lectura_vc_zeroshot as Z
    from lectura_vc_zeroshot import _filigrane as F
    assert Z.__version__ == "1.3.0"
    e = Z.creer_engine(models_dir=vc)
    marque, sr = e.convert(np.load(entree), "fr-f1", sr_in=SR, tau=0.0)
    assert sr == SR
    np.testing.assert_allclose(marque, attendu["marque"], atol=1e-5)
    vrai = F.marquer
    try:
        F.marquer = lambda a, sr, message=None: a
        brut, _ = e.convert(np.load(entree), "fr-f1", sr_in=SR, tau=0.0)
    finally:
        F.marquer = vrai
    np.testing.assert_allclose(brut, attendu["brut"], atol=1e-5)
    print(f"écart max : marqué {np.abs(marque - attendu['marque']).max():.2e}, "
          f"brut {np.abs(brut - attendu['brut']).max():.2e}")


@pytest.fixture(scope="module")
def moteur(installation):
    import lectura_vc_zeroshot as Z
    return Z.creer_engine(models_dir=installation[0])


@pytest.mark.parametrize("voix", ["fr-f1", "melange-f", "melange-h"])
def test_conversion_filigranee(moteur, voix):
    from lectura_vc_zeroshot import _filigrane as F
    sortie, sr = moteur.convert(_voix(SR), voix, sr_in=SR, tau=0.3)
    assert sr == SR and sortie.dtype == np.float32 and np.isfinite(sortie).all()
    assert 0.01 < np.abs(sortie).max() <= 1.0
    p, bits = F.detecter(sortie, sr)
    assert F.marque_retrouvee(p, bits), (voix, p, bits)
    # après passage en PCM 16 bits
    p2, bits2 = F.detecter(np.clip(np.round(sortie * 32768), -32768, 32767).astype(np.int16), sr)
    assert F.marque_retrouvee(p2, bits2), (voix, p2, bits2)


def test_reference_audio_utilisateur_filigranee(moteur, tmp_path):
    """Référence fournie par l'utilisateur (fichier WAV) : acceptée, sortie filigranée."""
    import wave
    from lectura_vc_zeroshot import _filigrane as F
    f = tmp_path / "ma_voix.wav"
    with wave.open(str(f), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((_voix(16000, 3.0, 0.5) * 32767).astype("<i2").tobytes())
    sortie, sr = moteur.convert(_voix(SR), str(f), sr_in=SR)
    assert F.marque_retrouvee(*F.detecter(sortie, sr))


def test_temoin_non_marque(moteur):
    """Témoin : la même conversion sans filigrane (fonction neutralisée dans le test) n'est pas détectée."""
    from lectura_vc_zeroshot import _filigrane as F
    vrai = F.marquer
    try:
        F.marquer = lambda a, sr, message=None: a
        brut, sr = moteur.convert(_voix(SR), "melange-h", sr_in=SR, tau=0.0)
    finally:
        F.marquer = vrai
    assert not F.marque_retrouvee(*F.detecter(brut, sr))


# -- curseur grave/aigu -----------------------------------------------------------------------------------------


def _f0_mediane(y, sr):
    import librosa
    f0, vo, _ = librosa.pyin(y, fmin=50, fmax=600, sr=sr, frame_length=1024)
    return float(np.nanmedian(f0[vo])) if vo.any() else float("nan")


@pytest.mark.parametrize("voix", ["fr-f1", "melange-h"])
def test_variante_zero_identique(moteur, voix):
    """variante=0 (ou sr_override=22050) : même sortie qu'avant le curseur."""
    x = _voix(SR)
    a, _ = moteur.convert(x, voix, sr_in=SR, tau=0.0)
    b, _ = moteur.convert(x, voix, sr_in=SR, tau=0.0, variante=0.0)
    c, _ = moteur.convert(x, voix, sr_in=SR, tau=0.0, sr_override=22050)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


@pytest.mark.parametrize("voix", ["fr-f1", "melange-f", "melange-h"])
def test_variante_grave_aigu_filigrane(moteur, voix):
    """-1 donne une voix plus grave, +1 plus aiguë (f0 médiane de la sortie) ; filigrane toujours retrouvé."""
    from lectura_vc_zeroshot import _filigrane as F
    x = _voix(SR, 5.0)
    f0 = {}
    for v in (-1.0, 0.0, 1.0):
        y, sr = moteur.convert(x, voix, sr_in=SR, tau=0.0, variante=v)
        f0[v] = _f0_mediane(y, sr)
        assert F.marque_retrouvee(*F.detecter(y, sr)), (voix, v)
    print(voix, {k: round(v, 1) for k, v in f0.items()})
    assert f0[-1.0] < f0[0.0] < f0[1.0], (voix, f0)
