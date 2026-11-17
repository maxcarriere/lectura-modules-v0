"""Changement de timbre facultatif (« retimbre ») par lectura-vc-zeroshot (OpenVoice).

Post-traitement qui remplace le timbre « moyen » de la banque de diphones par
l'un des timbres de référence ci-dessous. C'est une conversion vocale par réseau
de neurones : **toute sortie porte le filigrane IA** (AudioSeal), posé par
lectura-vc-zeroshot dans sa fonction de conversion. Ce module ne fait que
l'appeler ; il refuse une version de lectura-vc-zeroshot qui ne déclare pas le
filigrane (``MARQUE_IA``). Ne retirez pas le marquage : obligation de l'art. 50
du règlement européen sur l'IA (AI Act).

Timbres acceptés (identifiants neutres, aucune voix de personne identifiable) :

- ``"fr-f1"``     : voix native de la banque (corpus SIWIS, CC BY 4.0, voir NOTICE) ;
- ``"melange-f"`` : voix féminine mélangée (vecteur de timbre moyen) ;
- ``"melange-h"`` : voix masculine mélangée (vecteur de timbre moyen).

Aucun fichier audio ni vecteur fourni par l'utilisateur n'est accepté : le
paquet ne sert pas à imiter une personne.

Points d'attache des voix mélangées
-----------------------------------
Les trois timbres sont des préréglages de lectura-vc-zeroshot >= 1.3, qui les
expose sous les mêmes identifiants (``available_presets()``) et télécharge
leurs vecteurs depuis Hugging Face au premier usage. Un tuple vide dans
``TIMBRES`` signalerait un timbre annoncé mais pas livré (``TimbreIndisponible``).

Le curseur ``variante`` (-1 = plus grave, « homme » ; +1 = plus aigu,
« enfant ») est le « truc » de la fréquence d'échantillonnage d'OpenVoice
(sr_override = 22050 / 2**variante). Il est transmis tel quel à
lectura-vc-zeroshot (paramètre ``variante``), qui livre pour chaque
préréglage des vecteurs précalculés à 9 positions et les interpole ;
``variante=0`` donne le préréglage lui-même.

Installation : pip install "lectura-diphone[vc]"
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

OV_SR = 22050

# Identifiant du diphone -> noms de préréglage candidats dans lectura-vc-zeroshot,
# essayés dans l'ordre (le premier présent dans available_presets() est pris).
# Tuple vide : timbre annoncé mais pas encore livré.
TIMBRES: dict[str, tuple[str, ...]] = {
    "fr-f1": ("fr-f1",),
    "melange-f": ("melange-f",),  # préréglage « voix féminine mélangée » (lectura-vc-zeroshot 1.3)
    "melange-h": ("melange-h",),  # préréglage « voix masculine mélangée » (lectura-vc-zeroshot 1.3)
}

# Type accepté pour ``voix`` : un identifiant de TIMBRES (ou None : pas de retimbre)
VoixSpec = str | None


class TimbreIndisponible(LookupError):
    """Timbre inconnu, ou annoncé mais pas encore livré par lectura-vc-zeroshot."""


class FiligraneAbsent(RuntimeError):
    """La version installée de lectura-vc-zeroshot ne garantit pas le filigrane IA."""


def timbres_disponibles() -> list[str]:
    """Identifiants de timbre reconnus par le diphone (livrés ou non)."""
    return sorted(TIMBRES)


def verifier_voix(voix: object) -> str:
    """Contrôle ``voix`` (identifiant de TIMBRES) et renvoie l'identifiant.

    Lève ``TimbreIndisponible`` pour tout autre type ou identifiant (chemin de
    fichier, liste, dictionnaire de mélange, nom de préréglage non prévu).
    """
    if not isinstance(voix, str) or voix not in TIMBRES:
        raise TimbreIndisponible(
            f"Timbre {voix!r} non pris en charge. Timbres possibles : "
            f"{', '.join(timbres_disponibles())}. Les fichiers audio, listes et "
            f"mélanges personnalisés ne sont pas acceptés."
        )
    return voix


def variante_to_sr_override(variante: float) -> int | None:
    """Convertit un curseur variante en sr_override pour le truc SR.

    0.0 = neutre (None) ; +1 = formants montés (11025) ; -1 = formants baissés (44100).
    """
    if abs(variante) < 0.01:
        return None
    return max(8000, min(48000, int(OV_SR / (2 ** variante))))


def _make_cache_key(reference: str, sr_override: int | None) -> tuple:
    """Clé de cache du vecteur cible."""
    return (str(reference), sr_override)


def _importer_vc():
    """Importe lectura_vc_zeroshot et vérifie qu'il pose le filigrane IA."""
    try:
        import lectura_vc_zeroshot as vc
    except ImportError as e:
        raise ImportError(
            "lectura-vc-zeroshot est requis pour changer de timbre. "
            "Installez-le avec : pip install \"lectura-diphone[vc]\""
        ) from e
    if getattr(vc, "MARQUE_IA", False) is not True:
        raise FiligraneAbsent(
            "La version installée de lectura-vc-zeroshot ne déclare pas le filigrane IA "
            "(MARQUE_IA). Le changement de timbre est refusé : installez "
            "lectura-vc-zeroshot>=1.3 (pip install -U \"lectura-diphone[vc]\")."
        )
    return vc


class RetimbreEngine:
    """Garde le moteur de conversion et les vecteurs de timbre cibles en cache."""

    def __init__(self, vc_models_dir: str | Path | None = None):
        self._engine = None       # ZeroShotEngine (chargé au premier usage)
        self._vc_models_dir = vc_models_dir
        self._se_cache: dict[tuple, np.ndarray] = {}

    def _ensure_loaded(self):
        if self._engine is None:
            vc = _importer_vc()
            self._engine = vc.creer_engine(models_dir=self._vc_models_dir)
        return self._engine

    @staticmethod
    def _prereglage(engine, voix: str) -> str:
        """Nom du préréglage lectura-vc-zeroshot correspondant à ``voix``."""
        candidats = TIMBRES[voix]
        if not candidats:
            raise TimbreIndisponible(
                f"Le timbre {voix!r} n'est pas encore livré : il attend les vecteurs "
                f"de voix mélangées de lectura-vc-zeroshot."
            )
        presents = set(engine.available_presets())
        for nom in candidats:
            if nom in presents:
                return nom
        raise TimbreIndisponible(
            f"Le timbre {voix!r} est absent de la version installée de lectura-vc-zeroshot."
        )

    def _get_tgt_se(self, engine, voix: str, variante: float) -> np.ndarray:
        cache_key = _make_cache_key(voix, round(variante, 6))
        if cache_key not in self._se_cache:
            nom = self._prereglage(engine, voix)
            if variante:
                self._se_cache[cache_key] = engine.resolve_target_se(nom, variante=variante)
            else:
                self._se_cache[cache_key] = engine.resolve_target_se(nom, sr_override=None)
        return self._se_cache[cache_key]

    def retimbre(
        self,
        audio: np.ndarray,
        sr: int,
        reference: str,
        variante: float = 0.0,
        tau: float = 0.3,
    ) -> tuple[np.ndarray, int]:
        """Applique le changement de timbre.

        Parameters
        ----------
        audio : audio source float32 mono.
        sr : fréquence d'échantillonnage de la source.
        reference : identifiant de timbre (``"fr-f1"``, ``"melange-f"``, ``"melange-h"``).
        variante : curseur -1 (grave) à +1 (aigu).
        tau : paramètre d'OpenVoice (0 = déterministe).

        Returns
        -------
        (audio filigrané @ 22050 Hz, 22050). Le filigrane est posé par
        lectura-vc-zeroshot (``OpenVoiceConverter.convert``).
        """
        voix = verifier_voix(reference)
        engine = self._ensure_loaded()
        # curseur : 0 sous 0,01 (comme variante_to_sr_override), borné à [-1, 1]
        variante = 0.0 if abs(variante) < 0.01 else float(np.clip(variante, -1.0, 1.0))
        logger.info("Retimbre : voix=%s, variante=%.2f, tau=%.2f", voix, variante, tau)

        tgt_se = self._get_tgt_se(engine, voix, variante)
        src_se = engine.extract_se(audio, sr=sr)
        # convert_from_se -> OpenVoiceConverter.convert -> _filigrane.marquer (filigrane IA)
        return engine.convert_from_se(
            audio=audio, src_se=src_se, tgt_se=tgt_se, sr_in=sr, tau=tau,
        )
