"""Bout en bout en local : phonèmes -> texte, poids téléchargés depuis un faux Hugging Face.

Le test a besoin des vrais poids :
- soit ``LECTURA_TEST_HF_DIR`` désigne un dossier qui contient ``lectura-p2g-v0/``
  (contenu du futur dépôt Hugging Face) : un serveur HTTP local le sert, et les
  poids sont téléchargés dans un dossier temporaire (vérification SHA-256 comprise) ;
- soit les poids sont déjà présents (``LECTURA_MODELS_DIR``, ``~/.lectura/models``) ;
- sinon le test est sauté (aucun accès à Internet depuis les tests).
"""

from __future__ import annotations

import os
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

from lectura_graphemiseur import _poids

DEPOT = _poids.DEPOT_HF.split("/", 1)[1]


class _ServeurHF(SimpleHTTPRequestHandler):
    """/<organisation>/<dépôt>/resolve/<révision>/<fichier> -> <racine>/<dépôt>/<fichier>."""

    def translate_path(self, path):
        morceaux = path.split("?", 1)[0].strip("/").split("/")
        if len(morceaux) == 5 and morceaux[2] == "resolve":
            path = f"/{morceaux[1]}/{morceaux[4]}"
        return super().translate_path(path)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    racine = os.environ.get("LECTURA_TEST_HF_DIR")
    ancien = {k: os.environ.get(k) for k in ("LECTURA_HF_BASE", "LECTURA_MODELS_DIR", "LECTURA_HORS_LIGNE")}
    httpd = None
    try:
        if racine:
            if not (Path(racine) / DEPOT).is_dir():
                pytest.skip(f"{racine}/{DEPOT} absent")
            httpd = HTTPServer(("127.0.0.1", 0), partial(_ServeurHF, directory=racine))
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            os.environ["LECTURA_HF_BASE"] = f"http://127.0.0.1:{httpd.server_port}"
            os.environ["LECTURA_MODELS_DIR"] = str(tmp_path_factory.mktemp("modeles"))
            os.environ.pop("LECTURA_HORS_LIGNE", None)
        else:
            from lectura_graphemiseur import _resoudre_modeles_dir
            if _resoudre_modeles_dir(telecharger=False) is None:
                pytest.skip("poids P2G absents et LECTURA_TEST_HF_DIR non défini")
        from lectura_p2g import creer_engine
        e = creer_engine()
        if racine:
            d = Path(os.environ["LECTURA_MODELS_DIR"]) / _poids.SOUS_DOSSIER
            assert sorted(p.name for p in d.iterdir()) == sorted(_poids.FICHIERS)
            assert all(_poids.verifier(d).values())
        yield e
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        for k, v in ancien.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_moteur_onnx_et_correcteur(engine):
    assert "Onnx" in type(engine).__name__
    assert engine.correcteur is not None
    assert engine.phone_lexicon is not None


def test_mot(engine):
    from lectura_p2g import analyser

    assert analyser(["bɔ̃ʒuʁ"], engine=engine)["ortho"] == ["bonjour"]


def test_phrase(engine):
    from lectura_p2g import analyser

    r = analyser(["le", "ɑ̃fɑ̃", "ʒu", "dɑ̃", "lə", "ʒaʁdɛ̃"], engine=engine)
    assert r["ortho"] == ["les", "enfants", "jouent", "dans", "le", "jardin"]
