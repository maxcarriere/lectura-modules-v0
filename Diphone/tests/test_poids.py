"""Téléchargement de la banque au premier usage (module _poids), sans réseau extérieur.

Un serveur HTTP local (127.0.0.1) imite Hugging Face : chemin
/<dépôt>/resolve/<révision>/<fichier>. La table des empreintes est remplacée par
un petit fichier factice : la banque n'est ni téléchargée ni chargée ici.
"""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lectura_diphone import _poids, creer_engine
from lectura_diphone._poids import LecturaModelesIndisponibles

CONTENU = b"banque factice pour le test\n" * 100
SHA = hashlib.sha256(CONTENU).hexdigest()


class _FauxHF(BaseHTTPRequestHandler):
    fichiers: dict[str, bytes] = {}
    recues: list[str] = []

    def do_GET(self):  # noqa: N802
        _FauxHF.recues.append(self.path)
        corps = _FauxHF.fichiers.get(self.path)
        if corps is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def log_message(self, *args):
        pass


@pytest.fixture()
def faux_hf(monkeypatch, tmp_path):
    """Serveur local + table d'empreintes factice + dossier de modèles vide."""
    httpd = HTTPServer(("127.0.0.1", 0), _FauxHF)
    fil = threading.Thread(target=httpd.serve_forever, daemon=True)
    fil.start()
    chemin = f"/{_poids.DEPOT_HF}/resolve/{_poids.REVISION_HF}/poids.bin"
    _FauxHF.fichiers = {chemin: CONTENU}
    _FauxHF.recues = []
    monkeypatch.setattr(_poids, "FICHIERS", {"poids.bin": (SHA, len(CONTENU))})
    monkeypatch.setenv("LECTURA_HF_BASE", f"http://127.0.0.1:{httpd.server_port}")
    monkeypatch.setenv("LECTURA_MODELS_DIR", str(tmp_path / "modeles"))
    monkeypatch.delenv("LECTURA_HORS_LIGNE", raising=False)
    yield chemin, tmp_path / "modeles" / _poids.SOUS_DOSSIER
    httpd.shutdown()
    httpd.server_close()


def test_url_par_defaut(monkeypatch):
    monkeypatch.delenv("LECTURA_HF_BASE", raising=False)
    assert _poids.url_fichier("diphones.dpk.gz") == (
        "https://huggingface.co/lecturaworld/lectura-diphone-v0/resolve/" + _poids.REVISION_HF + "/diphones.dpk.gz")


def test_url_surchargee(monkeypatch):
    monkeypatch.setenv("LECTURA_HF_BASE", "http://127.0.0.1:9/")
    assert _poids.url_fichier("x") == "http://127.0.0.1:9/lecturaworld/lectura-diphone-v0/resolve/" + _poids.REVISION_HF + "/x"


def test_table_des_empreintes():
    assert set(_poids.FICHIERS) == {"diphones.dpk.gz", "diphone_statistics.pkl"}
    # copie nettoyée (champ « source » sans chemin de la machine)
    assert _poids.FICHIERS["diphone_statistics.pkl"] == (
        "771bcfccbbf0fdd4e1440268a00b2f645204d0808dc404ff0cd61d771aa15fcc", 190916)
    for sha, taille in _poids.FICHIERS.values():
        assert len(sha) == 64 and int(sha, 16) >= 0 and taille > 0
    # empreinte relevée dans le brain (LISTE-V1-VPS) : 380aaf6d3e2735ce…
    assert _poids.FICHIERS["diphones.dpk.gz"][0].startswith("380aaf6d3e2735ce")


def test_sous_dossier_historique():
    assert _poids.SOUS_DOSSIER == "tts_diphone"


def test_telechargement_et_verification(faux_hf, capsys):
    chemin, dossier = faux_hf
    assert _poids.assurer_modeles() == dossier
    assert (dossier / "poids.bin").read_bytes() == CONTENU
    assert not list(dossier.glob("*.part"))
    assert _FauxHF.recues == [chemin]
    err = capsys.readouterr().err
    assert "CC BY 4.0" in err and "SIWIS" in err and "Mo" in err
    assert "BY-NC" not in err
    assert _poids.verifier() == {"poids.bin": True}
    # Deuxième appel : rien n'est retéléchargé
    _poids.assurer_modeles()
    assert _FauxHF.recues == [chemin]


def test_models_dir_explicite(faux_hf, tmp_path):
    cible = tmp_path / "ailleurs"
    assert _poids.assurer_modeles(cible) == cible
    assert (cible / "poids.bin").read_bytes() == CONTENU


def test_empreinte_fausse_refusee(faux_hf):
    chemin, dossier = faux_hf
    _FauxHF.fichiers[chemin] = CONTENU[:-1] + b"X"   # même taille, autre contenu
    with pytest.raises(LecturaModelesIndisponibles) as e:
        _poids.assurer_modeles()
    assert "empreinte" in str(e.value)
    assert not (dossier / "poids.bin").exists()
    assert not list(dossier.glob("*.part"))


def test_taille_fausse_refusee(faux_hf):
    chemin, dossier = faux_hf
    _FauxHF.fichiers[chemin] = CONTENU + b"en trop"
    with pytest.raises(LecturaModelesIndisponibles):
        _poids.assurer_modeles()
    assert not (dossier / "poids.bin").exists()
    assert not list(dossier.glob("*.part"))


def test_fichier_absent_du_depot(faux_hf):
    _chemin, dossier = faux_hf
    _FauxHF.fichiers = {}
    with pytest.raises(LecturaModelesIndisponibles) as e:
        _poids.assurer_modeles()
    assert "404" in str(e.value) and "à la main" in str(e.value)
    assert not (dossier / "poids.bin").exists()


def test_hors_ligne(faux_hf, monkeypatch):
    _chemin, dossier = faux_hf
    monkeypatch.setenv("LECTURA_HORS_LIGNE", "1")
    with pytest.raises(LecturaModelesIndisponibles) as e:
        _poids.assurer_modeles()
    msg = str(e.value)
    assert "LECTURA_HORS_LIGNE" in msg and str(dossier) in msg and SHA in msg
    assert _FauxHF.recues == []


def test_hors_ligne_creer_engine(faux_hf, monkeypatch):
    pytest.importorskip("pyworld")
    monkeypatch.setenv("LECTURA_HORS_LIGNE", "1")
    with pytest.raises(LecturaModelesIndisponibles):
        creer_engine()
    assert _FauxHF.recues == []


def test_mode_api_retire():
    with pytest.raises(ValueError, match="retiré"):
        creer_engine(mode="api")


def test_chargeur_sans_telechargement(faux_hf):
    from lectura_diphone._chargeur import find_models_dir
    with pytest.raises(LecturaModelesIndisponibles):
        find_models_dir(telecharger=False)
    assert _FauxHF.recues == []


def test_plus_de_chiffrement_ni_api():
    """Plus de modèles chiffrés (.enc), plus d'API ni de clé dans le paquet."""
    import pathlib
    import re

    import lectura_diphone

    racine = pathlib.Path(lectura_diphone.__file__).parent
    assert not (racine / "_crypto.py").exists()
    assert not (racine / "modeles").exists()
    for f in racine.glob("*.py"):
        t = f.read_text(encoding="utf-8")
        assert not re.search(r"\.enc\b", t), f
        assert "api_url" not in t and "api.lectura" not in t and "api_key" not in t, f
