"""Passes lourdes : banque réelle, synthèse, changement de timbre, filigrane.

Sautées par défaut. À lancer seulement avec l'accord du guichet mémoire, par ex. :

    LECTURA_PASSES=1 LECTURA_HF_BASE=http://127.0.0.1:<port> \\
    systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 -p CPUQuota=100% \\
        python -m pytest tests/test_passes.py -v

Variables :
- LECTURA_PASSES=1 : active les passes ;
- LECTURA_DIPHONE_HF_LOCAL : dossier préparé du dépôt Hugging Face
  (lectura-diphone-v0/) ; s'il est donné, un serveur HTTP local l'expose et
  le téléchargement réel est testé (sinon LECTURA_MODELS_DIR doit contenir
  tts_diphone/ avec la banque) ;
- LECTURA_DIPHONE_V0_SRC : dossier src/ de lectura-diphone 2.0.2, pour la
  comparaison avec la v0 (sautée sinon) ;
- passes du changement de timbre : lectura-vc-zeroshot >= 1.3 installé, ses
  modèles OpenVoice et ceux du filigrane (LECTURA_FILIGRANE_ONNX,
  LECTURA_FILIGRANE_DETECTEUR_ONNX ou ~/.lectura/models/filigrane/).
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LECTURA_PASSES") != "1",
    reason="passe lourde (banque réelle / synthèse) : LECTURA_PASSES=1, avec l'accord du guichet",
)

GROUPES = [
    {"phones": ["b", "ɔ̃", "ʒ", "u", "ʁ"], "boundary": "comma"},
    {"phones": ["l", "ə", "m", "ɔ̃", "d"], "boundary": "period"},
]


@pytest.fixture(scope="module")
def dossier_banque(tmp_path_factory):
    """Dossier de la banque : téléchargée depuis un faux Hugging Face local si possible."""
    from lectura_diphone import _poids

    local = os.environ.get("LECTURA_DIPHONE_HF_LOCAL")
    if not local:
        d = _poids.dossier_destination()
        if _poids.fichiers_manquants(d):
            pytest.skip("ni LECTURA_DIPHONE_HF_LOCAL ni banque locale")
        yield d
        return

    racine = tmp_path_factory.mktemp("hf")
    # arborescence /<dépôt>/resolve/<révision>/<fichier> -> liens vers le dossier préparé
    cible = racine / _poids.DEPOT_HF / "resolve" / _poids.REVISION_HF
    cible.parent.mkdir(parents=True)
    cible.symlink_to(Path(local).resolve())
    class _Silencieux(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    handler = functools.partial(_Silencieux, directory=str(racine))
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    ancien = os.environ.get("LECTURA_HF_BASE")
    os.environ["LECTURA_HF_BASE"] = f"http://127.0.0.1:{httpd.server_port}"
    try:
        dest = tmp_path_factory.mktemp("modeles") / "tts_diphone"
        assert _poids.assurer_modeles(dest) == dest
        assert all(_poids.verifier(dest).values())
        yield dest
    finally:
        httpd.shutdown()
        if ancien is None:
            os.environ.pop("LECTURA_HF_BASE", None)
        else:
            os.environ["LECTURA_HF_BASE"] = ancien


@pytest.fixture(scope="module")
def moteur(dossier_banque):
    from lectura_diphone import creer_engine
    return creer_engine(models_dir=dossier_banque, g2p=lambda t: GROUPES)


def test_banque_une_seule_voix(dossier_banque):
    """Métadonnées internes : 1290 diphones, 44,1 kHz, aucun champ de locuteur."""
    import gzip
    import pickle

    with gzip.open(dossier_banque / "diphones.dpk.gz", "rb") as f:
        p = pickle.load(f)
    assert p["metadata"]["n_diphones"] == len(p["diphones"]) == 1290
    champs = {k for e in p["diphones"].values() for k in e}
    assert champs == {"f0", "log_sp", "ap_u8", "sr", "frame_period", "n_frames"}


@pytest.mark.parametrize("mode", ["FLUIDE", "MOT_A_MOT", "SYLLABES"])
def test_synthese_phonemes(moteur, mode, tmp_path):
    from lectura_diphone import _marquage

    r = moteur.synthesize_groups(GROUPES, mode=mode, seed=1)
    assert r.sample_rate == 44100 and r.samples.dtype == np.float32
    assert np.isfinite(r.samples).all() and 0.3 < len(r.samples) / 44100 < 10
    assert 0.01 < np.abs(r.samples).max() <= 1.0
    assert not r.timbre_converti
    m = _marquage.lire_metadonnees(r.vers_wav(tmp_path / f"{mode}.wav"))
    assert m["xmp_ia"] and m["info_ia"] and "SIWIS" in m["description"]


def test_prosodie_corpus(moteur):
    r = moteur.synthesize_groups(GROUPES, prosody_style="corpus", seed=2)
    assert len(r.samples) > 0


def test_synthetiser_avec_callable(dossier_banque):
    from lectura_diphone import synthetiser

    r = synthetiser("bonjour le monde", models_dir=dossier_banque, g2p=lambda t: GROUPES, seed=3)
    assert len(r.samples) > 0


def test_cli_phonemes(dossier_banque, tmp_path):
    from lectura_diphone import _cli, _marquage

    sortie = tmp_path / "cli.wav"
    assert _cli.main(["--phonemes", "b ɔ̃ ʒ u ʁ", "-o", str(sortie),
                      "--models-dir", str(dossier_banque), "--seed", "4"]) == 0
    assert _marquage.lire_metadonnees(sortie.read_bytes())["xmp_ia"]


def test_identique_a_la_v0(moteur, dossier_banque, tmp_path):
    """Même graine, mêmes phonèmes : même son que lectura-diphone 2.0.2 (code de synthèse inchangé)."""
    v0 = os.environ.get("LECTURA_DIPHONE_V0_SRC")
    if not v0:
        pytest.skip("LECTURA_DIPHONE_V0_SRC non défini")
    ref = tmp_path / "v0.npy"
    code = (
        "import sys, json, numpy as np\n"
        f"sys.path.insert(0, {v0!r})\n"
        "from lectura_diphone.engine import DiphoneEngine\n"
        "e = DiphoneEngine(); e.load(models_dir=sys.argv[1])\n"
        "r = e.synthesize_groups(json.loads(sys.argv[2]), seed=5)\n"
        "np.save(sys.argv[3], r.samples)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run([sys.executable, "-I", "-c", code, str(dossier_banque),
                    json.dumps(GROUPES, ensure_ascii=False), str(ref)], check=True, env=env)
    r = moteur.synthesize_groups(GROUPES, seed=5)
    np.testing.assert_allclose(r.samples, np.load(ref), atol=1e-6)


# ── Changement de timbre : filigrane présent sur la sortie finale ──


def _vc_ou_saut():
    vc = pytest.importorskip("lectura_vc_zeroshot")
    if getattr(vc, "MARQUE_IA", False) is not True:
        pytest.fail("lectura-vc-zeroshot installé sans MARQUE_IA")
    from lectura_vc_zeroshot import _filigrane
    return _filigrane


@pytest.mark.parametrize("voix", ["fr-f1", "melange-f", "melange-h"])
def test_retimbre_filigrane(moteur, voix, tmp_path):
    from lectura_diphone import _marquage, _retimbre

    if not _retimbre.TIMBRES[voix]:
        pytest.skip(f"{voix} : vecteurs mélangés pas encore branchés (point d'attache)")
    fil = _vc_ou_saut()
    import lectura_vc_zeroshot as vc
    presents = set(vc.ZeroShotEngine.available_presets())
    if not presents & set(_retimbre.TIMBRES[voix]):
        pytest.skip(f"{voix} : préréglage absent de la lectura-vc-zeroshot installée")
    groupes = GROUPES * 3  # quelques secondes, pour la détection
    r = moteur.synthesize_groups(groupes, seed=6, voix=voix)
    assert r.timbre_converti and r.sample_rate == 44100
    p, bits = fil.detecter(r.samples, r.sample_rate)
    assert fil.marque_retrouvee(p, bits), (p, bits)
    # après passage en WAV 16 bits (fichier livré à l'utilisateur)
    import io
    import wave
    octets = r.vers_wav(tmp_path / "conv.wav")
    with wave.open(io.BytesIO(octets)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    p2, bits2 = fil.detecter(pcm, 44100)
    assert fil.marque_retrouvee(p2, bits2), (p2, bits2)
    assert "filigrane" in _marquage.lire_metadonnees(octets)["description"]


def test_sans_retimbre_pas_de_filigrane(moteur):
    """Témoin : la synthèse par règles seule n'est pas filigranée (métadonnées seules)."""
    fil = _vc_ou_saut()
    r = moteur.synthesize_groups(GROUPES * 3, seed=7)
    p, bits = fil.detecter(r.samples, r.sample_rate)
    assert not fil.marque_retrouvee(p, bits)


@pytest.mark.parametrize("voix,variante", [("fr-f1", -1.0), ("melange-h", 1.0)])
def test_retimbre_variante_filigrane(moteur, voix, variante):
    """Curseur grave/aigu rebranché (vecteurs précalculés de lectura-vc-zeroshot) : la sortie change, filigrane présent."""
    from lectura_diphone import _retimbre

    fil = _vc_ou_saut()
    r0 = moteur.synthesize_groups(GROUPES * 3, seed=8, voix=voix)
    r = moteur.synthesize_groups(GROUPES * 3, seed=8, voix=voix, voix_variante=variante)
    assert r.timbre_converti
    n = min(len(r.samples), len(r0.samples))
    assert np.abs(r.samples[:n] - r0.samples[:n]).mean() > 1e-3
    assert fil.marque_retrouvee(*fil.detecter(r.samples, r.sample_rate))
