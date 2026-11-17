"""Localisation de la banque de diphones (mode local uniquement).

Ordre de recherche (voir ``_poids.dossier_destination``) :
1. paramètre explicite ``models_dir`` ;
2. variable d'environnement ``$LECTURA_MODELS_DIR/tts_diphone`` ;
3. dossier utilisateur ``~/.lectura/models/tts_diphone/``.

Si la banque manque, elle est téléchargée depuis Hugging Face au premier usage
et vérifiée par son empreinte SHA-256 (``_poids.assurer_modeles``). Aucun appel
à une API, aucune clé. ``LECTURA_HORS_LIGNE=1`` interdit le téléchargement.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lectura_diphone import _poids

log = logging.getLogger(__name__)

# Fichier requis pour la synthèse
REQUIRED_FILES = ["diphones.dpk.gz"]

# Fichier facultatif (statistiques prosodiques), livré avec la banque
OPTIONAL_FILES = ["diphone_statistics.pkl"]


def find_models_dir(models_dir: str | Path | None = None, telecharger: bool = True) -> Path:
    """Renvoie le dossier de la banque, après téléchargement si besoin.

    Parameters
    ----------
    models_dir : dossier explicite (sinon cascade ci-dessus).
    telecharger : False pour seulement chercher (lève une erreur si absente).

    Raises
    ------
    _poids.LecturaModelesIndisponibles
        Banque absente et téléchargement impossible ou désactivé.
    """
    if telecharger:
        dossier = _poids.assurer_modeles(models_dir)
    else:
        dossier = _poids.dossier_destination(models_dir)
        manquants = _poids.fichiers_manquants(dossier)
        if manquants:
            raise _poids.LecturaModelesIndisponibles(
                f"Banque de diphones absente de {dossier} : {', '.join(manquants)}."
            )
    log.debug("Banque de diphones : %s", dossier)
    return dossier
