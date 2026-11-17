"""Outils communs : faux Hugging Face local (serveur HTTP) et modèles factices."""

from __future__ import annotations

import functools
import os
import threading
from contextlib import contextmanager
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np


class _Silencieux(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


@contextmanager
def faux_hf(dossier_depot: Path, racine: Path):
    """Expose ``dossier_depot`` comme https://huggingface.co/<DEPOT_HF>/resolve/<REVISION_HF>/ ;
    renvoie l'URL de base à mettre dans LECTURA_HF_BASE."""
    from lectura_vc_zeroshot import _poids

    cible = racine / _poids.DEPOT_HF / "resolve" / _poids.REVISION_HF
    cible.parent.mkdir(parents=True, exist_ok=True)
    cible.symlink_to(Path(dossier_depot).resolve())
    handler = functools.partial(_Silencieux, directory=str(racine))
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def vecteur(graine: int) -> np.ndarray:
    return np.random.default_rng(graine).standard_normal((1, 256, 1)).astype(np.float32)


def openvoice_factices(dossier: Path) -> Path:
    """Fichiers OpenVoice factices de la taille attendue (creux) : aucun téléchargement n'est tenté."""
    from lectura_vc_zeroshot import _poids

    dossier.mkdir(parents=True, exist_ok=True)
    for c in _poids.OPENVOICE:
        with open(dossier / _poids.nom_local(c), "wb") as f:
            f.truncate(_poids.FICHIERS[c][1])
    return dossier


def table_variantes(base: np.ndarray, graine: int) -> np.ndarray:
    """Table factice (9, 1, 256, 1) du curseur ; la position centrale (variante 0) est ``base``."""
    t = np.random.default_rng(100 + graine).standard_normal((9, 1, 256, 1)).astype(np.float32) * 3
    t[4] = base
    return t


def presets_factices(dossier_vc: Path) -> dict[str, np.ndarray]:
    """Vecteurs factices (mêmes tailles que les vrais) dans <dossier_vc>/presets ; renvoie aussi les tables
    du curseur sous les clés « <nom>-variantes »."""
    from lectura_vc_zeroshot._presets import PRESET_SPEAKERS

    d = dossier_vc / "presets"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, nom in enumerate(PRESET_SPEAKERS):
        v = vecteur(i)
        np.save(d / f"{nom}.npy", v)
        assert os.path.getsize(d / f"{nom}.npy") == 1152
        t = table_variantes(v, i)
        np.save(d / f"{nom}-variantes.npy", t)
        assert os.path.getsize(d / f"{nom}-variantes.npy") == 9344
        out[nom] = v
        out[f"{nom}-variantes"] = t
    return out
