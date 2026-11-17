"""lectura-vc-zeroshot --- Conversion vocale zero-shot via OpenVoice (ONNX).

Tout tourne en local. Les modeles OpenVoice (MIT), le modele du filigrane
AudioSeal (MIT) et les vecteurs des preregles sont telecharges depuis Hugging
Face au premier usage, verifies par SHA-256 (LECTURA_HORS_LIGNE=1 pour
l'interdire). Preregles : fr-f1, melange-f, melange-h. Une reference audio
fournie par l'utilisateur est acceptee (conversion zero-shot) : n'utilisez que
votre propre voix, ou une voix dont vous avez l'accord ecrit.

API publique :
  creer_engine()  : fabrique un ZeroShotEngine
  convertir()     : fonction de commodite (cree un engine ephemere)
  blend_se()      : melange des speaker embeddings avec des poids
  OV_SR           : sample rate de sortie (22050 Hz)
  PRESET_SPEAKERS : identifiants des preregles ("fr-f1", "melange-f", "melange-h")
  PresetInconnu   : levee pour un nom qui n'est ni un preregle ni un fichier audio
  LecturaModelesIndisponibles : fichiers absents et telechargement impossible
  MARQUE_IA       : True — toute sortie de conversion porte le filigrane IA (AI Act, art. 50 §2)
  marquer_ia()    : la fonction de marquage appliquee a la sortie (controle par les services)
  FiligraneIndisponible : levee si le modele du filigrane manque ou est altere

Ne pas retirer le marquage : obligation de l'art. 50 de l'AI Act.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lectura_vc_zeroshot._filigrane import FiligraneIndisponible, marquer as marquer_ia
from lectura_vc_zeroshot._openvoice import OV_SR
from lectura_vc_zeroshot._poids import LecturaModelesIndisponibles
from lectura_vc_zeroshot._presets import PRESET_SPEAKERS, PresetInconnu
from lectura_vc_zeroshot.engine import ZeroShotEngine, blend_se

__version__ = "1.3.0"

# Convention Lectura (16/11/2026) : un paquet qui pose lui-meme le filigrane IA sur ses sorties le declare ici ;
# les services (API Lectura) ne remarquent pas ses sorties, apres un controle au demarrage (marquer_ia + detecteur).
MARQUE_IA = True

__all__ = [
    "MARQUE_IA",
    "marquer_ia",
    "FiligraneIndisponible",
    "creer_engine",
    "convertir",
    "ZeroShotEngine",
    "blend_se",
    "OV_SR",
    "PRESET_SPEAKERS",
    "PresetInconnu",
    "LecturaModelesIndisponibles",
]


def creer_engine(models_dir: str | Path | None = None) -> ZeroShotEngine:
    """Cree un moteur de conversion vocale zero-shot.

    Parameters
    ----------
    models_dir : dossier des modeles OpenVoice et des preregles (defaut :
        $LECTURA_MODELS_DIR/vc, sinon ~/.lectura/models/vc ; telechargement
        au premier usage).

    Returns
    -------
    ZeroShotEngine
    """
    return ZeroShotEngine(models_dir=models_dir)


def convertir(
    audio: np.ndarray | str | Path,
    reference: np.ndarray | str | Path | list | dict,
    sr_in: int | None = None,
    ref_sr: int | None = None,
    sr_override: int | None = None,
    tau: float = 0.3,
    variante: float | None = None,
) -> tuple[np.ndarray, int]:
    """Convertit un audio vers le timbre d'une reference (commodite).

    Cree un engine ephemere. Pour des conversions repetees,
    utiliser creer_engine() pour reutiliser les sessions ONNX.

    Parameters
    ----------
    audio : son a convertir, quel qu'il soit (voix de synthese ou enregistrement ;
        array float32 mono ou chemin fichier ; toute frequence).
    reference : voix cible — polymorphe :
        - str : preregle ("fr-f1", "melange-f", "melange-h") ou chemin d'un
          fichier audio (votre voix, ou une voix dont vous avez l'accord ecrit).
        - list : plusieurs references (poids egaux).
        - dict[str, float] : blend pondere (ex: {"fr-f1": 0.5, "melange-f": 0.5}).
    sr_in : sample rate source (auto-detecte si fichier).
    ref_sr : sample rate de la reference si ndarray.
    sr_override : trick SR pour decaler les formants.
        11025 → voix aigue/enfant, 44100 → voix grave/homme.
    tau : parametre OpenVoice (0 = deterministe).
    variante : curseur grave/aigu, -1 (plus grave) a +1 (plus aigu) ; 0 = preregle tel quel.

    Returns
    -------
    (audio_converti @ 22050 Hz, sample_rate) — audio marque du filigrane IA.

    Ne pas retirer le marquage : obligation de l'art. 50 de l'AI Act.
    """
    engine = creer_engine()
    return engine.convert(
        audio=audio,
        reference=reference,
        sr_in=sr_in,
        ref_sr=ref_sr,
        sr_override=sr_override,
        tau=tau,
        variante=variante,
    )
