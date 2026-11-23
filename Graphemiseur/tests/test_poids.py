"""Téléchargement des poids au premier usage (module _poids), sans réseau extérieur.

Un serveur HTTP local (127.0.0.1) imite Hugging Face : chemin
/<dépôt>/resolve/<révision>/<fichier>. La table des empreintes est remplacée par
un petit fichier factice : aucun modèle n'est chargé ici.
"""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lectura_graphemiseur import _poids, creer_engine
from lectura_graphemiseur._poids import LecturaModelesIndisponibles

CONTENU = b"poids factices pour le test\n" * 100
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
    assert _poids.url_fichier("unifie_p2g_v7_int8.onnx") == (
        "https://huggingface.co/lecturaworld/lectura-p2g-v0/resolve/" + _poids.REVISION_HF + "/unifie_p2g_v7_int8.onnx")


def test_table_des_empreintes():
    assert set(_poids.FICHIERS) == {
        "unifie_p2g_v7_int8.onnx", "unifie_p2g_v7_vocab.json", "phone_lexicon.db",
    }
    for sha, taille in _poids.FICHIERS.values():
        assert len(sha) == 64 and int(sha, 16) >= 0 and taille > 0


def test_telechargement_et_verification(faux_hf, capsys):
    chemin, dossier = faux_hf
    assert _poids.assurer_modeles() == dossier
    assert (dossier / "poids.bin").read_bytes() == CONTENU
    assert not list(dossier.glob("*.part"))
    assert _FauxHF.recues == [chemin]
    err = capsys.readouterr().err
    assert "CC BY-NC 4.0" in err and "Mo" in err and "lectura.world/contact" in err
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
    monkeypatch.setenv("LECTURA_HORS_LIGNE", "1")
    monkeypatch.setattr("pathlib.Path.home", lambda: faux_hf[1].parent.parent)
    with pytest.raises(LecturaModelesIndisponibles):
        creer_engine()
    assert _FauxHF.recues == []


def test_mode_api_retire():
    with pytest.raises(ValueError, match="retire"):
        creer_engine(mode="api")
