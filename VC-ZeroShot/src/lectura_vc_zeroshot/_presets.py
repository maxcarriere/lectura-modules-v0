"""Préréglages de voix : vecteurs de timbre (speaker embeddings) OpenVoice.

Trois préréglages, livrés comme vecteurs seuls (aucun son), téléchargés depuis
Hugging Face au premier usage (voir ``_poids``) :

- ``fr-f1``     : voix féminine du corpus SIWIS (CC BY 4.0, crédit dans NOTICE) ;
- ``melange-f`` : voix féminine mélangée (vecteur moyen de plusieurs voix, dont
  SIWIS ; CC BY 4.0) ;
- ``melange-h`` : voix masculine mélangée (vecteur moyen de plusieurs voix ;
  CC BY 4.0).

Les voix mélangées sont mesurées pour ne ressembler à aucune des voix dont
elles sont tirées. Chaque vecteur est un tableau (1, 256, 1) float32.

Curseur grave/aigu (``variante``, -1 à +1 : voix plus grave, « homme », à -1 ;
plus aiguë, « enfant », à +1). Pour chaque préréglage, 9 vecteurs précalculés
(positions -1, -0,75, ..., +1 ; fichier ``<identifiant>-variantes.npy``),
tirés par le « truc SR » d'OpenVoice (son rééchantillonné à
22050 / 2**variante puis lu comme à 22050 Hz). Entre deux positions, le
vecteur est interpolé linéairement, puis ramené à la norme interpolée.
``variante=0`` rend exactement le préréglage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from lectura_vc_zeroshot import _poids

logger = logging.getLogger(__name__)

# Identifiants publics des préréglages (seuls noms acceptés)
PRESET_SPEAKERS = ["fr-f1", "melange-f", "melange-h"]

# Positions précalculées du curseur grave/aigu
POSITIONS_VARIANTE = tuple(np.linspace(-1.0, 1.0, 9).tolist())
_EPS = 1e-9


class PresetInconnu(LookupError):
    """Nom de préréglage inconnu (et pas un fichier audio existant)."""


def _chemin_depot(name: str) -> str:
    return f"presets/{name}.npy"


def dossier_presets(models_dir: str | Path | None = None) -> Path:
    """Dossier local des préréglages : ``<dossier des modèles VC>/presets``."""
    return _poids.dossier_destination(models_dir) / "presets"


def list_presets() -> list[str]:
    """Identifiants des préréglages disponibles (téléchargés ou non)."""
    return list(PRESET_SPEAKERS)


def has_preset(name: str) -> bool:
    """Vrai si ``name`` est l'identifiant d'un préréglage."""
    return isinstance(name, str) and name in PRESET_SPEAKERS


def _refus(name: str) -> PresetInconnu:
    return PresetInconnu(
        f"Préréglage {name!r} inconnu. Préréglages disponibles : {', '.join(PRESET_SPEAKERS)}. "
        f"Pour une autre voix, donnez le chemin d'un fichier audio de référence "
        f"(votre propre voix, ou une voix dont vous avez l'accord écrit)."
    )


def _charger_npy(chemin: str, models_dir) -> np.ndarray:
    dossier = _poids.assurer([chemin], dossier_presets(models_dir))
    return np.load(dossier / _poids.nom_local(chemin), allow_pickle=False).astype(np.float32)


def interpoler_variante(table: np.ndarray, variante: float) -> np.ndarray:
    """Vecteur au point ``variante`` d'une table (9, 1, 256, 1) aux POSITIONS_VARIANTE.

    Interpolation linéaire entre les deux positions voisines, puis norme
    ramenée à l'interpolation linéaire des deux normes. Hors de [-1, 1] : borné.
    """
    pos = np.asarray(POSITIONS_VARIANTE)
    v = float(np.clip(variante, pos[0], pos[-1]))
    i = int(np.searchsorted(pos, v, side="right") - 1)
    i = min(max(i, 0), len(pos) - 2)
    t = (v - pos[i]) / (pos[i + 1] - pos[i])
    if t < _EPS:
        return table[i].copy()
    if t > 1 - _EPS:
        return table[i + 1].copy()
    a, b = table[i].astype(np.float64), table[i + 1].astype(np.float64)
    m = (1 - t) * a + t * b
    norme = (1 - t) * np.linalg.norm(a) + t * np.linalg.norm(b)
    return (m / np.linalg.norm(m) * norme).astype(np.float32)


def load_preset(name: str, models_dir: str | Path | None = None, variante: float = 0.0) -> np.ndarray:
    """Charge un préréglage (téléchargé au premier usage).

    Parameters
    ----------
    name : identifiant (``"fr-f1"``, ``"melange-f"`` ou ``"melange-h"``).
    models_dir : dossier des modèles VC (défaut : voir ``_poids``).
    variante : curseur grave/aigu, -1 (plus grave) à +1 (plus aigu) ; 0 = le
        préréglage lui-même. Hors de [-1, 1] : borné (avertissement).

    Returns
    -------
    np.ndarray shape (1, 256, 1) float32.

    Raises
    ------
    PresetInconnu : identifiant inconnu.
    _poids.LecturaModelesIndisponibles : fichier absent et téléchargement impossible.
    """
    if not has_preset(name):
        raise _refus(name)
    variante = float(variante)
    if not np.isfinite(variante):
        raise ValueError(f"variante non finie : {variante!r}")
    if abs(variante) < _EPS:
        se = _charger_npy(_chemin_depot(name), models_dir)
        if se.ndim == 2:
            se = se[:, :, np.newaxis]
        return se
    if abs(variante) > 1 + 1e-6:
        logger.warning("variante %.3f hors de [-1, 1] : bornée", variante)
    table = _charger_npy(f"presets/{name}-variantes.npy", models_dir)
    return interpoler_variante(table, variante)


def blend_presets(specs: dict[str, float], models_dir: str | Path | None = None) -> np.ndarray:
    """Mélange de préréglages avec des poids (normalisés pour sommer à 1).

    Ex. : ``{"fr-f1": 0.5, "melange-f": 0.5}``.
    """
    if not specs:
        raise ValueError("Au moins un preset requis.")

    total_weight = sum(specs.values())
    if total_weight <= 0:
        raise ValueError("Les poids doivent etre positifs.")

    result = np.zeros((1, 256, 1), dtype=np.float32)
    for name, weight in specs.items():
        se = load_preset(name, models_dir)
        result += se * (weight / total_weight)

    return result
