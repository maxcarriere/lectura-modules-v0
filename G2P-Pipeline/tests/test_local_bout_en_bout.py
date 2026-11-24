"""Bout en bout en local : phrase -> phonèmes, poids téléchargés depuis un faux Hugging Face.

Le test a besoin des vrais poids :
- soit ``LECTURA_TEST_HF_DIR`` désigne un dossier qui contient ``lectura-g2p-v0/``
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

from lectura_phonemiseur import _poids

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
            from lectura_phonemiseur import _resoudre_modeles_dir
            if _resoudre_modeles_dir(telecharger=False) is None:
                pytest.skip("poids G2P absents et LECTURA_TEST_HF_DIR non défini")
        from lectura_g2p import creer_engine
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


def test_moteur_onnx(engine):
    interne = getattr(engine, "_engine", engine)
    assert "Onnx" in type(interne).__name__


def test_mot(engine):
    r = engine.analyser(["bonjour"])
    assert r["g2p"] == ["bɔ̃ʒuʁ"]


def test_phrase(engine):
    from lectura_g2p import analyser_phrase

    r = analyser_phrase("Les enfants jouent dans le jardin.", engine=engine)
    phones = [m.phone for m in r.mots if not m.est_ponctuation]
    assert phones == ["le", "ɑ̃fɑ̃", "ʒu", "dɑ̃", "lə", "ʒaʁdɛ̃"]


def test_groupes_public(engine):
    """Correctif du 24/11/2026 : groupes de lecture accessibles par ``.groupes``."""
    from lectura_g2p import OptionsGroupes, analyser_phrase

    r = analyser_phrase("Les enfants jouent.", engine=engine)
    assert r.groupes is r._groupes  # l'ancien attribut privé reste lisible
    assert [g.text for g in r.groupes][:2] == ["Les enfants", "jouent"]
    assert r.groupes[0].jonctions == ["liaison_z"]

    sans = analyser_phrase(
        "Les enfants jouent.", engine=engine,
        options_groupes=OptionsGroupes(gerer_liaisons=False, gerer_enchainement=False),
    )
    assert [g.text for g in sans.groupes][:2] == ["Les", "enfants"]
