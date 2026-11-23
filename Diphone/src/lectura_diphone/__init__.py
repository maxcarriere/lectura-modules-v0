"""lectura-diphone — Synthèse vocale du français par concaténation de diphones WORLD.

Copyright (C) 2025-2026 Maxime Carrière (Lectura)
Code : AGPL-3.0-or-later (voir LICENCE.txt). Licence commerciale sur demande :
https://www.lectura.world/contact/
Banque de diphones et données prosodiques : adaptations du corpus SIWIS,
CC BY 4.0 (voir NOTICE).

Tout tourne en local. La banque de diphones (~19 Mo) est téléchargée depuis
Hugging Face au premier usage et vérifiée par son empreinte SHA-256.

Marquage IA : les WAV écrits par ``ecrire_wav`` / ``DiphoneResult.vers_wav``
portent des métadonnées IPTC ``trainedAlgorithmicMedia`` ; un son passé par le
changement de timbre (``voix=...``) porte en plus le filigrane IA de
lectura-vc-zeroshot. Ne retirez pas le marquage.

Exports publics :
    - creer_engine(mode, g2p, models_dir) → DiphoneEngine
    - synthetiser(texte, **kwargs) → DiphoneResult (.samples float32 @ 44100 Hz, .vers_wav())
      (le moteur est gardé en mémoire d'un appel à l'autre, par models_dir et g2p)
    - DiphoneEngine, SynthMode, DiphoneResult, PhonemeTiming
    - ecrire_wav(chemin, audio, sr) → octets WAV marqués
    - TIMBRES_VOIX : timbres du changement de timbre ("fr-f1", "melange-f", "melange-h")
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from lectura_diphone._marquage import ecrire_wav
from lectura_diphone._poids import LecturaModelesIndisponibles


def __getattr__(nom: str):
    # Import paresseux : numpy n'est chargé qu'à l'usage.
    if nom in ("DiphoneResult", "PhonemeTiming", "DiphoneEngine", "SynthMode"):
        from lectura_diphone import engine as _engine
        return getattr(_engine, nom)
    raise AttributeError(nom)

log = logging.getLogger(__name__)

__version__ = "2.2.0"

# Timbres du changement de timbre (identifiants neutres ; voir _retimbre.TIMBRES).
TIMBRES_VOIX = ("fr-f1", "melange-f", "melange-h")


def _verifier_mode(mode: str) -> None:
    if mode not in ("auto", "local"):
        raise ValueError(
            f"Mode inconnu : {mode!r}. Seuls \"auto\" et \"local\" existent : "
            "le mode \"api\" a été retiré, tout tourne en local. (Le mode de "
            "synthèse FLUIDE / MOT_A_MOT / SYLLABES s'appelle synth_mode dans "
            "synthetiser().)"
        )


# Moteurs gardés en mémoire par synthetiser() : clé (dossier de la banque, id du g2p)
# -> (g2p, moteur). Le moteur ne garde aucun état qui change le son d'un appel à
# l'autre (la graine est reposée à chaque synthèse) : mêmes sorties qu'un moteur neuf.
_MOTEURS: dict[tuple[str, int], tuple[Any, Any]] = {}
_MOTEURS_MAX = 4
_MOTEURS_VERROU = threading.Lock()


def _moteur_en_cache(mode: str, g2p: Any, models_dir: str | Path | None):
    """Moteur de synthetiser() : réutilisé pour le même dossier de banque et le même g2p."""
    from lectura_diphone._poids import dossier_destination

    _verifier_mode(mode)
    cle = (str(dossier_destination(models_dir).resolve()), id(g2p))
    with _MOTEURS_VERROU:
        trouve = _MOTEURS.get(cle)
        if trouve is not None and trouve[0] is g2p:
            return trouve[1]
        engine = creer_engine(mode=mode, g2p=g2p, models_dir=models_dir)
        if engine._g2p_backend is not None:   # sans G2P : rien à garder
            if len(_MOTEURS) >= _MOTEURS_MAX:
                _MOTEURS.pop(next(iter(_MOTEURS)))
            _MOTEURS[cle] = (g2p, engine)
        return engine


def vider_cache() -> None:
    """Oublie les moteurs gardés par synthetiser() (rend la mémoire de la banque)."""
    with _MOTEURS_VERROU:
        _MOTEURS.clear()


def creer_engine(
    mode: str = "auto",
    g2p: Any = None,
    models_dir: str | Path | None = None,
):
    """Crée un moteur de synthèse diphone (local).

    Parameters
    ----------
    mode : str
        "auto" ou "local" (identiques : tout tourne en local ; le mode
        "api" n'existe plus).
    g2p : G2PBackend | callable | None
        Backend G2P. Si None, détection d'un G2P local
        (pip install "lectura-diphone[g2p]"). Peut être un
        callable(text) → list[dict]. Sans G2P, seules les méthodes qui
        prennent des phonèmes (synthesize_groups, synthesize_phones) servent.
    models_dir : Path | None
        Dossier de la banque (sinon $LECTURA_MODELS_DIR/tts_diphone, puis
        ~/.lectura/models/tts_diphone). Banque absente : téléchargement
        depuis Hugging Face au premier usage (désactivé par
        LECTURA_HORS_LIGNE=1).

    Returns
    -------
    DiphoneEngine

    Raises
    ------
    ImportError
        Si pyworld/numpy/scipy ne sont pas installés.
    LecturaModelesIndisponibles
        Si la banque manque et ne peut pas être téléchargée.
    """
    _verifier_mode(mode)
    try:
        import numpy  # noqa: F401
        import pyworld  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"Dépendance manquante : {e}. "
            "Installez avec : pip install lectura-diphone"
        ) from e

    from lectura_diphone.engine import DiphoneEngine

    engine = DiphoneEngine()
    engine.load(models_dir=models_dir)

    if g2p is not None:
        if callable(g2p) and not hasattr(g2p, "phonemize"):
            from lectura_diphone.g2p import CallableG2P
            engine._g2p_backend = CallableG2P(g2p)
        else:
            engine._g2p_backend = g2p
    else:
        from lectura_diphone.g2p import auto_detect_g2p
        try:
            engine._g2p_backend = auto_detect_g2p()
        except ImportError as e:
            log.info("%s", e)
            engine._g2p_backend = None

    return engine


def synthetiser(
    texte: str,
    mode: str = "auto",
    models_dir: str | Path | None = None,
    g2p: Any = None,
    synth_mode: str = "FLUIDE",
    duration_scale: float = 1.0,
    pause_scale: float = 1.0,
    macro_expressivity: float = 1.0,
    micro_expressivity: float = 1.0,
    seed: int | None = None,
    prosody_style: str = "regles",
    ap_cleanup: float = 1.5,
    formant_sharpening: float = 1.3,
    vtln_alpha: float = 1.0,
    timbre: str | None = None,
    base_f0: float = 175.0,
    sentence_pause_ms: float = 400.0,
    # -- Changement de timbre (OpenVoice, via lectura-vc-zeroshot) --
    voix: str | None = None,
    voix_variante: float = 0.0,
    voix_tau: float | None = None,
    vc_models_dir: str | Path | None = None,
) -> Any:
    """Texte → DiphoneResult (audio float32 mono @ 44100 Hz et timings).

    ``resultat.samples`` est le tableau numpy ; ``resultat.vers_wav(chemin)``
    écrit un WAV avec les métadonnées IA (à préférer pour tout fichier diffusé).

    Le moteur (banque de diphones, et moteur de conversion si ``voix=``) est
    créé au premier appel puis gardé en mémoire pour les appels suivants avec
    le même ``models_dir`` et le même ``g2p`` ; ``vider_cache()`` le libère.
    Les sorties sont identiques à celles d'un moteur neuf.

    Parameters
    ----------
    texte : str
        Texte français à synthétiser (demande un G2P local :
        pip install "lectura-diphone[g2p]", ou ``g2p=``).
    mode, models_dir, g2p :
        Paramètres de creer_engine() (``mode`` : "auto" ou "local" seulement).
    synth_mode : str
        Mode de synthèse : "FLUIDE", "MOT_A_MOT" ou "SYLLABES" (appelé
        ``mode`` dans ``DiphoneEngine.synthesize_groups``).
    duration_scale : float
        Facteur de vitesse (>1 = plus lent).
    pause_scale : float
        Facteur des pauses.
    macro_expressivity : float
        Gestes prosodiques (0 = plat, 1 = normal, 2 = exagéré).
    micro_expressivity : float
        Micro-variations (0 = robot, 1 = normal, 2 = très expressif).
        Actif en mode FLUIDE uniquement.
    seed : int | None
        Graine de la micro-prosodie (même graine = même résultat).
    prosody_style : str
        "regles" (défaut, règles LHiLH*) ou "corpus" (contours tirés du
        corpus SIWIS, plus variés).
    ap_cleanup : float
        Compression de l'apériodicité (1.0 = off, 1.5 = défaut, max 3.0 ;
        au-delà, ramené à 3.0 avec un avertissement).
    formant_sharpening : float
        Affûtage des formants (1.0 = off, 1.3 = défaut, max 2.0 ; au-delà,
        ramené à 2.0 avec un avertissement).
    vtln_alpha : float
        Déformation VTLN (0.8 = grave, 1.0 = neutre, 1.2 = aigu).
    timbre : str | None
        Signature de timbre paramétrique ("neutre", "homme", "enfant") ou
        chemin d'un fichier .json. Traitement du signal, sans réseau de neurones.
    base_f0 : float
        Hauteur de base en Hz (défaut 175 ; homme ~120, femme ~200, enfant ~280).
    sentence_pause_ms : float
        Pause entre deux phrases, en ms (défaut 400).
    voix : str | None
        Changement de timbre par OpenVoice (lectura-vc-zeroshot) :
        "fr-f1" (voix native SIWIS), "melange-f" ou "melange-h" (voix
        mélangées). None = pas de changement (défaut). La sortie porte le
        filigrane IA ; ne retirez pas le marquage.
        Requires: pip install "lectura-diphone[vc]"
    voix_variante : float
        Curseur -1 (grave) à +1 (aigu) du changement de timbre.
    voix_tau : None
        Sans effet (le modèle ONNX d'OpenVoice n'a pas d'entrée tau ; la
        conversion est déterministe). Gardé pour compatibilité :
        DeprecationWarning s'il est donné.
    vc_models_dir : str | Path | None
        Dossier des modèles de lectura-vc-zeroshot (défaut : détection).

    Returns
    -------
    DiphoneResult
        ``samples`` (float32 mono, 44100 Hz), ``sample_rate``,
        ``phoneme_timings``, ``timbre_converti``.
    """
    from lectura_diphone.engine import _avertir_voix_tau

    _avertir_voix_tau(voix_tau)
    engine = _moteur_en_cache(mode, g2p, models_dir)
    if engine._g2p_backend is None:
        from lectura_diphone.g2p import auto_detect_g2p
        auto_detect_g2p()  # lève l'ImportError explicite

    groups = engine._g2p_backend.phonemize(texte)
    return engine.synthesize_groups(
        groups,
        mode=synth_mode,
        duration_scale=duration_scale,
        pause_scale=pause_scale,
        macro_expressivity=macro_expressivity,
        micro_expressivity=micro_expressivity,
        seed=seed,
        prosody_style=prosody_style,
        ap_cleanup=ap_cleanup,
        formant_sharpening=formant_sharpening,
        vtln_alpha=vtln_alpha,
        timbre=timbre,
        base_f0=base_f0,
        sentence_pause_ms=sentence_pause_ms,
        voix=voix,
        voix_variante=voix_variante,
        vc_models_dir=vc_models_dir,
    )


__all__ = [
    "creer_engine",
    "synthetiser",
    "ecrire_wav",
    "TIMBRES_VOIX",
    "vider_cache",
    "DiphoneEngine",
    "SynthMode",
    "DiphoneResult",
    "PhonemeTiming",
    "LecturaModelesIndisponibles",
    "__version__",
]
