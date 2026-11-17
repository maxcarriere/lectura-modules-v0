"""ZeroShotEngine --- conversion vocale zero-shot via OpenVoice.

Usage principal : modifier le timbre d'une voix de synthese (diphone,
monolocuteur...) ; tout son mono est accepte en entree (voix de synthese ou
enregistrement, n'importe quelle frequence d'echantillonnage).

Supporte :
  - Preregles fr-f1, melange-f, melange-h (vecteurs de timbre, telecharges au
    premier usage)
  - Curseur grave/aigu ``variante`` (-1 = plus grave, « homme » ; +1 = plus
    aigu, « enfant ») : « truc SR » d'OpenVoice, sr_override = 22050 / 2**variante.
    Pour un preregle, vecteurs precalcules a 9 positions et interpoles ; pour
    une reference audio, calcule a la volee.
  - Conversion vers une ou plusieurs references audio (fonction zero-shot :
    n'utilisez que votre propre voix, ou une voix dont vous avez l'accord ecrit)
  - Blend pondere de voix (preregles ou audio)

Toute sortie porte le filigrane IA AudioSeal. Ne pas retirer le marquage :
obligation de l'art. 50 de l'AI Act.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from lectura_vc_zeroshot._chargeur import ensure_models
from lectura_vc_zeroshot._openvoice import OpenVoiceConverter, OV_SR
from lectura_vc_zeroshot._presets import (
    PresetInconnu,
    has_preset,
    list_presets,
    load_preset,
    blend_presets,
)

logger = logging.getLogger(__name__)

# Type alias pour les references polymorphes
_RefSpec = (
    str                              # chemin audio ou identifiant de preregle
    | Path                           # chemin audio
    | np.ndarray                     # audio array ou SE (1,256,1)
    | list                           # liste de references (poids egaux)
    | dict                           # {ref: poids} pour blend pondere
)


def variante_vers_sr(variante: float) -> int | None:
    """Curseur grave/aigu -> sr_override du truc SR (None pour 0)."""
    if abs(variante) < 1e-9:
        return None
    return int(round(OV_SR / (2 ** variante)))


def sr_vers_variante(sr_override: int | None) -> float:
    """sr_override du truc SR -> curseur grave/aigu (0 pour None ou 22050)."""
    if sr_override is None or sr_override == OV_SR:
        return 0.0
    return float(np.log2(OV_SR / float(sr_override)))


def _curseur(sr_override: int | None, variante: float | None) -> tuple[int | None, float]:
    """(sr_override, variante) coherents ; ``variante`` l'emporte si les deux sont donnes."""
    if variante is not None:
        variante = float(variante)
        return variante_vers_sr(variante), variante
    return sr_override, sr_vers_variante(sr_override)


def _is_se(arr: np.ndarray) -> bool:
    """Detecte si un ndarray est un SE (1, 256, 1) vs audio (N,)."""
    return arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 256


def blend_se(
    se_list: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Melange des speaker embeddings avec des poids.

    Parameters
    ----------
    se_list : list[ndarray]
        Liste de SE (1, 256, 1).
    weights : list[float] | None
        Poids pour chaque SE. None = poids egaux.
        Normalises automatiquement pour sommer a 1.

    Returns
    -------
    np.ndarray shape (1, 256, 1) float32.
    """
    if not se_list:
        raise ValueError("Au moins un SE requis.")
    if len(se_list) == 1:
        return se_list[0].astype(np.float32)

    if weights is None:
        weights = [1.0] * len(se_list)
    if len(weights) != len(se_list):
        raise ValueError(
            f"Nombre de poids ({len(weights)}) != nombre de SE ({len(se_list)})"
        )

    total = sum(weights)
    if total <= 0:
        raise ValueError("Les poids doivent etre positifs.")

    result = np.zeros((1, 256, 1), dtype=np.float32)
    for se, w in zip(se_list, weights):
        result += se.astype(np.float32).reshape(1, 256, 1) * (w / total)
    return result


class ZeroShotEngine:
    """Moteur de conversion vocale zero-shot via OpenVoice.

    Supporte les presets de voix, le blend pondere, et le trick SR
    pour decaler les formants.
    """

    def __init__(self, models_dir: str | Path | None = None):
        # Modeles OpenVoice : telecharges au premier usage s'ils manquent (voir _poids),
        # leve _poids.LecturaModelesIndisponibles si c'est impossible.
        self.models_dir = ensure_models(models_dir)
        self._converter: OpenVoiceConverter | None = None

    def _ensure_loaded(self) -> OpenVoiceConverter:
        """Lazy-load le converter OpenVoice."""
        if self._converter is None:
            self._converter = OpenVoiceConverter(self.models_dir)
        return self._converter

    # -- Extraction SE ---------------------------------------------------------

    def extract_se(
        self,
        audio: np.ndarray | str | Path,
        sr: int | None = None,
    ) -> np.ndarray:
        """Extrait le speaker embedding d'un audio.

        Parameters
        ----------
        audio : audio de reference (array float32 ou chemin).
        sr : sample rate (auto-detecte si chemin).

        Returns
        -------
        np.ndarray shape (1, 256, 1) float32.
        """
        conv = self._ensure_loaded()
        return conv.extract_se(audio, sr=sr)

    def extract_se_multi(
        self,
        audios: list[np.ndarray | str | Path],
        sr: int | None = None,
        weights: list[float] | None = None,
    ) -> np.ndarray:
        """Extrait et melange les SE de plusieurs audios.

        Parameters
        ----------
        audios : liste d'audios (arrays ou chemins).
        sr : sample rate commun (auto si chemins).
        weights : poids pour chaque audio. None = poids egaux.

        Returns
        -------
        np.ndarray shape (1, 256, 1) float32.
        """
        ses = [self.extract_se(a, sr=sr) for a in audios]
        return blend_se(ses, weights)

    # -- Presets ---------------------------------------------------------------

    @staticmethod
    def get_preset_se(name: str) -> np.ndarray:
        """Charge un preregle par identifiant (dossier des modeles par defaut).

        Parameters
        ----------
        name : "fr-f1", "melange-f" ou "melange-h".

        Returns
        -------
        np.ndarray shape (1, 256, 1) float32.
        """
        return load_preset(name)

    @staticmethod
    def available_presets() -> list[str]:
        """Liste les presets disponibles."""
        return list_presets()

    @staticmethod
    def blend_preset_se(specs: dict[str, float]) -> np.ndarray:
        """Melange des presets avec des poids.

        Parameters
        ----------
        specs : {nom_preset: poids}.
            Ex: {"fr-f1": 0.5, "melange-f": 0.5}

        Returns
        -------
        np.ndarray shape (1, 256, 1) float32.
        """
        return blend_presets(specs)

    # -- Resolution de reference polymorphe ------------------------------------

    def resolve_target_se(
        self,
        reference: _RefSpec,
        ref_sr: int | None = None,
        sr_override: int | None = None,
        variante: float | None = None,
    ) -> np.ndarray:
        """Resout une reference polymorphe en speaker embedding cible.

        Parameters
        ----------
        reference : specification de la voix cible.
            - str : preregle ("fr-f1", "melange-f", "melange-h") ou chemin
              d'un fichier audio existant ; tout autre nom leve PresetInconnu.
            - Path : chemin fichier audio.
            - ndarray (1,256,1) : SE pre-calcule directement.
            - ndarray (N,) : audio array (necessite ref_sr).
            - list[str|Path|ndarray] : plusieurs references (poids egaux).
            - dict[str, float] : blend pondere de presets et/ou chemins.
        ref_sr : sample rate si reference est un ndarray audio.
        sr_override : truc SR pour decaler les formants (voir ``variante``).
        variante : curseur grave/aigu, -1 a +1 (l'emporte sur sr_override).

        Returns
        -------
        np.ndarray shape (1, 256, 1) float32.
        """
        sr_override, variante = _curseur(sr_override, variante)
        # -- dict : blend pondere --
        if isinstance(reference, dict):
            return self._resolve_dict_reference(reference, ref_sr, sr_override, variante)

        # -- list : plusieurs references (poids egaux) --
        if isinstance(reference, list):
            ses = [
                self._resolve_single_to_se(ref, ref_sr, sr_override, variante)
                for ref in reference
            ]
            return blend_se(ses)

        # -- scalaire --
        return self._resolve_single_to_se(reference, ref_sr, sr_override, variante)

    def _resolve_single_to_se(
        self,
        reference: np.ndarray | str | Path,
        ref_sr: int | None,
        sr_override: int | None,
        variante: float = 0.0,
    ) -> np.ndarray:
        """Resout une reference unique en SE."""
        # ndarray : SE directe ou audio
        if isinstance(reference, np.ndarray):
            if _is_se(reference):
                return reference.astype(np.float32)
            # audio array
            if sr_override is not None and sr_override != OV_SR:
                return self._extract_se_with_sr_trick(reference, ref_sr, sr_override)
            return self.extract_se(reference, sr=ref_sr)

        # str : preregle ou chemin de fichier audio
        ref_str = str(reference)
        if isinstance(reference, str) and has_preset(ref_str):
            # Curseur : vecteurs precalcules par le truc SR, interpoles (variante 0 = le preregle)
            return load_preset(ref_str, self.models_dir, variante=variante)

        # Chemin fichier (reference fournie par l'utilisateur)
        if not Path(ref_str).is_file():
            raise PresetInconnu(
                f"Reference {ref_str!r} : ni un preregle ({', '.join(list_presets())}), "
                f"ni un fichier audio existant."
            )
        if sr_override is not None and sr_override != OV_SR:
            return self._extract_se_with_sr_trick(reference, ref_sr, sr_override)
        return self.extract_se(reference, sr=ref_sr)

    def _resolve_dict_reference(
        self,
        specs: dict[str, float],
        ref_sr: int | None,
        sr_override: int | None,
        variante: float = 0.0,
    ) -> np.ndarray:
        """Resout un dict {ref: poids} en SE blende."""
        ses = []
        weights = []
        for ref, weight in specs.items():
            se = self._resolve_single_to_se(ref, ref_sr, sr_override, variante)
            ses.append(se)
            weights.append(weight)
        return blend_se(ses, weights)

    # -- Conversion ------------------------------------------------------------

    def convert_from_se(
        self,
        audio: np.ndarray | str | Path,
        src_se: np.ndarray,
        tgt_se: np.ndarray,
        sr_in: int | None = None,
        tau: float = 0.3,
    ) -> tuple[np.ndarray, int]:
        """Conversion avec speaker embeddings pre-calcules.

        Parameters
        ----------
        audio : audio source (array float32 ou chemin).
        src_se : speaker embedding source (1, 256, 1).
        tgt_se : speaker embedding cible (1, 256, 1).
        sr_in : sample rate source (auto si chemin).
        tau : parametre OpenVoice (0 = deterministe).

        Returns
        -------
        (audio_converti @ 22050 Hz, 22050)
        """
        conv = self._ensure_loaded()
        return conv.convert(audio, src_se, tgt_se, sr=sr_in, tau=tau)

    def convert(
        self,
        audio: np.ndarray | str | Path,
        reference: _RefSpec,
        sr_in: int | None = None,
        ref_sr: int | None = None,
        sr_override: int | None = None,
        tau: float = 0.3,
        variante: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Change le timbre d'un son (conversion zero-shot complete).

        Parameters
        ----------
        audio : son a convertir, quel qu'il soit : voix de synthese (diphone,
            monolocuteur...) ou enregistrement ; tableau float32 mono (donner
            sr_in) ou chemin de fichier ; toute frequence d'echantillonnage
            (reechantillonne a 22050 Hz).
        reference : voix cible — polymorphe :
            - str : preregle ("fr-f1", "melange-f", "melange-h") ou chemin
              d'un fichier audio (votre voix, ou une voix dont vous avez l'accord ecrit).
            - Path : chemin fichier audio.
            - ndarray (1,256,1) : SE pre-calcule.
            - ndarray (N,) : audio array (necessite ref_sr).
            - list : plusieurs references (poids egaux).
            - dict[str, float] : blend pondere.
        sr_in : sample rate source (auto si chemin).
        ref_sr : sample rate de la reference si ndarray audio.
        sr_override : trick SR — resample la reference vers ce SR avant
            extraction SE. OpenVoice traite comme 22050 → formants decales
            par facteur 22050/sr_override.
            Exemples: 11025 → voix aigue/enfant, 44100 → voix grave/homme.
        tau : parametre OpenVoice (0 = deterministe).
        variante : curseur grave/aigu, -1 (plus grave, « homme ») a +1 (plus
            aigu, « enfant ») ; 0 ou None = timbre du preregle tel quel.
            Equivaut a sr_override = 22050 / 2**variante (et l'emporte sur lui).

        Returns
        -------
        (audio_converti @ 22050 Hz, 22050) -- audio marque du filigrane IA.
        """
        conv = self._ensure_loaded()

        # Extraire SE source
        src_se = conv.extract_se(audio, sr=sr_in)

        # Resoudre SE cible (polymorphe)
        tgt_se = self.resolve_target_se(reference, ref_sr, sr_override, variante=variante)

        return conv.convert(audio, src_se, tgt_se, sr=sr_in, tau=tau)

    # -- Trick SR --------------------------------------------------------------

    def _extract_se_with_sr_trick(
        self,
        reference: np.ndarray | str | Path,
        ref_sr: int | None,
        sr_override: int,
    ) -> np.ndarray:
        """Extrait SE en trompant OpenVoice sur le SR.

        Resample la reference vers sr_override, puis passe a extract_se()
        sans declarer le SR → OpenVoice traite comme 22050 Hz.
        """
        conv = self._ensure_loaded()

        # Charger la reference a son SR natif
        if isinstance(reference, (str, Path)):
            ref_audio, actual_sr = librosa.load(str(reference), sr=None, mono=True)
        else:
            ref_audio = reference.astype(np.float32)
            actual_sr = ref_sr or OV_SR

        # Resample vers sr_override
        if actual_sr != sr_override:
            ref_audio = librosa.resample(
                ref_audio, orig_sr=actual_sr, target_sr=sr_override,
            )

        # Passer a OpenVoice SANS declarer le SR
        # → il traite comme 22050 → formants decales
        return conv.extract_se(ref_audio)
