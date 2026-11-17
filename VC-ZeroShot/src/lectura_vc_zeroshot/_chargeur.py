"""Localisation des modèles OpenVoice (ONNX, MIT).

Dossier : paramètre ``models_dir``, sinon ``$LECTURA_MODELS_DIR/vc``, sinon
``~/.lectura/models/vc``. Les modèles qui manquent y sont téléchargés depuis
Hugging Face au premier usage, vérifiés par SHA-256 (voir ``_poids``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from lectura_vc_zeroshot import _poids

log = logging.getLogger(__name__)

# Fichiers requis par OpenVoice
OPENVOICE_REQUIRED = [_poids.nom_local(c) for c in _poids.OPENVOICE]


def find_models_dir(models_dir: str | Path | None = None) -> Path:
    """Dossier des modèles OpenVoice (qu'ils y soient déjà ou non)."""
    return _poids.dossier_destination(models_dir)


def has_openvoice_models(directory: Path) -> bool:
    """Vrai si les deux modèles OpenVoice sont présents (taille attendue)."""
    return not _poids.fichiers_manquants(Path(directory), _poids.OPENVOICE)


def ensure_models(models_dir: str | Path | None = None) -> Path:
    """Dossier des modèles OpenVoice, après téléchargement s'ils manquent."""
    return _poids.assurer_openvoice(models_dir)


def load_model_bytes(models_dir: Path, filename: str) -> bytes | None:
    """Octets d'un modèle du dossier, ou None s'il est absent."""
    path = Path(models_dir) / filename
    return path.read_bytes() if path.exists() else None
