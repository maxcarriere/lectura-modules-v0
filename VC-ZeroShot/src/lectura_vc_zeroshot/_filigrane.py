"""Marquage IA des sons de lectura-vc-zeroshot : filigrane AudioSeal (ONNX) + métadonnées WAV.

Ne pas retirer le marquage : obligation de l'art. 50 de l'AI Act.

AI Act, art. 50 §2 : les sons générés par IA doivent être marqués de façon lisible par machine. Toute sortie de
OpenVoiceConverter.convert passe par marquer() ; aucune option ne le désactive. Les modèles ne sont jamais dans la
roue : ils sont lus dans ~/.lectura/models/filigrane/, où ils sont téléchargés au premier usage depuis le dépôt
Hugging Face du paquet (empreinte SHA-256 vérifiée, voir _poids).
Licences : NOTICE (AudioSeal MIT, rééchantillonnage d'après torchaudio BSD-2).

Dépendances : numpy, onnxruntime (testé avec 1.25.1). Modèle : generator_base.onnx (AudioSeal 0.2.0, Meta, MIT),
cherché dans $LECTURA_FILIGRANE_ONNX, sinon ~/.lectura/models/filigrane/generator_base.onnx (téléchargé s'il manque,
sauf LECTURA_HORS_LIGNE=1 ; sha256 vérifié au chargement).

    from lectura_vc_zeroshot._filigrane import marquer, metadonnees_wav
    audio_marque = marquer(audio, sr)          # float32 (-1..1) ou int16, mono, même longueur, même type
    octets = metadonnees_wav(octets_wav)       # ajoute les chunks LIST/INFO et _PMX (XMP), audio inchangé

Pose (identique à Filigrane.poser) : le signal est rééchantillonné à 16 kHz (même noyau sinc/Hann que
torchaudio.functional.resample), le filigrane est calculé à 16 kHz avec le message fixe 0x4C01, ramené à la
fréquence d'origine, recalé à la même longueur, ajouté au signal, puis le tout est écrêté à [-1, 1].
Nombre de fils : $LECTURA_FILIGRANE_FILS (2 par défaut) ; $LECTURA_FILIGRANE_ARENE=0 pour rendre la mémoire.

Détection sans torch : detecter(audio, sr) -> (p, bits), avec detector_base.onnx cherché dans
$LECTURA_FILIGRANE_DETECTEUR_ONNX, sinon ~/.lectura/models/filigrane/detector_base.onnx (téléchargé s'il manque) ; marque_retrouvee(p, bits) applique la règle
p >= 0,5 et 16 bits justes (message 0x4C01).
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import struct
import threading
import wave
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ constantes du marquage Lectura
SOURCE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
OUTIL = "Lectura, conversion vocale par IA (lectura-vc-zeroshot, OpenVoice)"
DESCRIPTION = "Voix convertie par intelligence artificielle (Lectura)"
CREDIT = "Lectura"
# 16 bits du filigrane : 0x4C01 = « L » (Lectura) + 01 (sons v1), bit de poids fort en premier.
MESSAGE = [int(b) for b in f"{0x4C01:016b}"]
SR_MODELE = 16000
SHA256_ONNX = "8a9ea5fce3041482cf1268c2c6bfccfd451691bece3f440458c0426faa5fc778"
# détecteur AudioSeal (detector_base, exporté le 16/11/2026 par exporter_detecteur_onnx.py --legacy)
SHA256_DETECTEUR = "c540086101be6ca8a8cfd1049e9d7575087c9e769ecbba1c348429f52b32c383"


def _xmp(outil: str, description: str, credit: str) -> str:
    return f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/">
   <Iptc4xmpExt:DigitalSourceType>{SOURCE}</Iptc4xmpExt:DigitalSourceType>
   <xmp:CreatorTool>{outil}</xmp:CreatorTool>
   <photoshop:Credit>{credit}</photoshop:Credit>
   <dc:description><rdf:Alt><rdf:li xml:lang="x-default">{description}</rdf:li></rdf:Alt></dc:description>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="r"?>"""


XMP = _xmp(OUTIL, DESCRIPTION, CREDIT)
INFO = {b"INAM": DESCRIPTION, b"ISFT": OUTIL, b"ICMT": f"AI-Generated: true; DigitalSourceType: {SOURCE}",
        b"ICOP": CREDIT}

# ------------------------------------------------------------------ métadonnées RIFF (bibliothèque standard)


def _chunks(octets: bytes) -> list[tuple[bytes, bytes]]:
    assert octets[:4] == b"RIFF" and octets[8:12] == b"WAVE", "pas un WAV RIFF"
    out, i = [], 12
    while i + 8 <= len(octets):
        cid, n = octets[i:i + 4], struct.unpack("<I", octets[i + 4:i + 8])[0]
        out.append((cid, octets[i + 8:i + 8 + n]))
        i += 8 + n + (n & 1)
    return out


def _chunk(cid: bytes, data: bytes) -> bytes:
    return cid + struct.pack("<I", len(data)) + data + (b"\0" if len(data) & 1 else b"")


def _est_info(cid: bytes, data: bytes) -> bool:
    return cid == b"LIST" and data[:4] == b"INFO"


def metadonnees_wav(octets: bytes, outil: str = OUTIL, description: str = DESCRIPTION, credit: str = CREDIT) -> bytes:
    """Renvoie le WAV avec les chunks _PMX (XMP) et LIST/INFO de marquage ; les anciens sont remplacés, le reste
    (fmt, data, autres chunks) est recopié tel quel, dans le même ordre. Avec les valeurs par défaut, le résultat
    porte les textes de lectura-vc-zeroshot. Les textes sont paramétrables ; la mention IPTC trainedAlgorithmicMedia
    reste toujours."""
    if (outil, description, credit) == (OUTIL, DESCRIPTION, CREDIT):
        info_d, xmp = INFO, XMP
    else:
        info_d = {b"INAM": description, b"ISFT": outil, b"ICMT": INFO[b"ICMT"], b"ICOP": credit}
        xmp = _xmp(outil, description, credit)
    garde = [(c, d) for c, d in _chunks(octets) if c != b"_PMX" and not _est_info(c, d)]
    info = b"INFO" + b"".join(_chunk(k, v.encode("utf-8") + b"\0") for k, v in info_d.items())
    corps = b"WAVE" + b"".join(_chunk(c, d) for c, d in garde) + _chunk(b"LIST", info) + \
        _chunk(b"_PMX", xmp.encode("utf-8"))
    return b"RIFF" + struct.pack("<I", len(corps)) + corps


def lire_metadonnees(octets: bytes) -> dict:
    ch = _chunks(octets)
    xmp = next((d.decode("utf-8", "replace") for c, d in ch if c == b"_PMX"), "")
    pcm = next((d for c, d in ch if c == b"data"), b"")
    return {"xmp_ia": SOURCE in xmp, "info": any(_est_info(c, d) and SOURCE.encode() in d for c, d in ch),
            "pcm_sha256": hashlib.sha256(pcm).hexdigest()}

# ------------------------------------------------------------------ rééchantillonnage (copie numpy de torchaudio)

_NOYAUX: dict[tuple[int, int], tuple[np.ndarray, int, int, int]] = {}


def _noyau(sr_in: int, sr_out: int) -> tuple[np.ndarray, int, int, int]:
    """Noyau de torchaudio.functional._get_sinc_resample_kernel (sinc_interp_hann, largeur 6, rolloff 0.99),
    mêmes types intermédiaires (float32 / float64) que torchaudio 2.x."""
    cle = (sr_in, sr_out)
    if cle not in _NOYAUX:
        g = math.gcd(sr_in, sr_out)
        o, n = sr_in // g, sr_out // g
        base = min(o, n) * 0.99
        w = math.ceil(6 * o / base)
        idx = np.arange(-w, w + o, dtype=np.float64)[None, :] / o
        t = (np.arange(0, -n, -1, dtype=np.int64).astype(np.float32) / np.float32(n))[:, None].astype(np.float64) + idx
        t = np.clip(t * base, -6, 6)
        fen = np.cos(t * math.pi / 6 / 2) ** 2
        t = t * math.pi
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(t == 0, 1.0, np.sin(t) / t)
        k = (k * (fen * (base / o))).astype(np.float32)          # [n, 2w + o]
        _NOYAUX[cle] = (k, w, o, n)
    return _NOYAUX[cle]


def reechantillonner(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Équivalent numpy de torchaudio.functional.resample(x, sr_in, sr_out) pour un signal 1-D float32."""
    if sr_in == sr_out:
        return x
    k, w, o, n = _noyau(int(sr_in), int(sr_out))
    L = x.shape[-1]
    xp = np.pad(x.astype(np.float32, copy=False), (w, w + o))
    trames = np.lib.stride_tricks.sliding_window_view(xp, k.shape[1])[::o]   # [nb, 2w + o], sans copie
    y = (trames @ k.T).reshape(-1)                                          # [nb * n]
    return np.ascontiguousarray(y[: math.ceil(n * L / o)], dtype=np.float32)

# ------------------------------------------------------------------ filigrane AudioSeal (ONNX)


DOSSIER_MODELES = Path.home() / ".lectura" / "models" / "filigrane"


class FiligraneIndisponible(RuntimeError):
    """Modèle du filigrane absent ou altéré : la conversion est refusée plutôt que de rendre un son non marqué."""


def chemin_generateur() -> Path:
    return Path(os.environ.get("LECTURA_FILIGRANE_ONNX") or DOSSIER_MODELES / "generator_base.onnx")


def chemin_detecteur() -> Path:
    return Path(os.environ.get("LECTURA_FILIGRANE_DETECTEUR_ONNX") or DOSSIER_MODELES / "detector_base.onnx")


def _assurer(variable: str, chemin_depot: str) -> None:
    """Télécharge le modèle dans DOSSIER_MODELES s'il manque, sauf si son chemin est imposé par ``variable``."""
    if os.environ.get(variable):
        return
    from lectura_vc_zeroshot import _poids
    _poids.assurer([chemin_depot], DOSSIER_MODELES)


class _Modele:
    _session = None
    _verrou = threading.Lock()

    @classmethod
    def session(cls):
        if cls._session is None:
            with cls._verrou:
                if cls._session is None:
                    import onnxruntime as ort
                    from lectura_vc_zeroshot._poids import LecturaModelesIndisponibles
                    chemin = chemin_generateur()
                    try:
                        _assurer("LECTURA_FILIGRANE_ONNX", "filigrane/generator_base.onnx")
                    except LecturaModelesIndisponibles as e:
                        raise FiligraneIndisponible(
                            f"Modèle du filigrane IA indisponible ({chemin}) : la conversion vocale est refusée tant "
                            f"qu'il manque (AI Act, art. 50).\n{e}") from e
                    try:
                        octets = chemin.read_bytes()
                    except OSError as e:
                        raise FiligraneIndisponible(
                            f"Modèle du filigrane IA introuvable ({chemin}) : la conversion vocale est refusée tant "
                            "qu'il manque (AI Act, art. 50). Posez generator_base.onnx (sha256 "
                            f"{SHA256_ONNX[:16]}…) à cet endroit ou indiquez-le par LECTURA_FILIGRANE_ONNX.") from e
                    if hashlib.sha256(octets).hexdigest() != SHA256_ONNX:
                        raise FiligraneIndisponible(
                            f"Modèle du filigrane IA altéré ({chemin} : empreinte sha256 inattendue, attendue "
                            f"{SHA256_ONNX[:16]}…) : la conversion vocale est refusée.")
                    so = ort.SessionOptions()
                    so.intra_op_num_threads = int(os.environ.get("LECTURA_FILIGRANE_FILS", "2"))
                    so.inter_op_num_threads = 1
                    # arène mémoire : plus rapide, mais garde le pic (≈ 650 Mo pour 30 s) jusqu'à la fin du processus ;
                    # LECTURA_FILIGRANE_ARENE=0 rend la mémoire après chaque appel (≈ 30 % plus lent)
                    so.enable_cpu_mem_arena = os.environ.get("LECTURA_FILIGRANE_ARENE", "1") != "0"
                    cls._session = ort.InferenceSession(octets, so, providers=["CPUExecutionProvider"])
        return cls._session   # chargée une seule fois par processus ; run() est sûr entre fils


def filigrane_16k(x16k: np.ndarray, message: list[int] | None = None) -> np.ndarray:
    """Filigrane brut (à ajouter) d'un signal float32 mono à 16 kHz, même longueur (= get_watermark)."""
    msg = np.array([MESSAGE if message is None else message], dtype=np.int64)
    if msg.shape != (1, 16) or not np.isin(msg, (0, 1)).all():
        raise ValueError("message : 16 bits 0/1")
    x = np.ascontiguousarray(x16k, dtype=np.float32)[None, None, :]
    return _Modele.session().run(None, {"audio": x, "message": msg})[0][0, 0]


BLOC_S = 30.0       # mémoire : filigrane calculé par blocs de 30 s au plus (pic borné quelle que soit la durée)
CONTEXTE_S = 1.0    # contexte de chaque côté d'un bloc, pour une marque continue aux jointures


def _filigrane_bloc(x: np.ndarray, sr: int, message) -> np.ndarray:
    wm = filigrane_16k(reechantillonner(x, sr, SR_MODELE), message)
    if sr != SR_MODELE:
        wm = reechantillonner(wm, SR_MODELE, sr)
    n = x.shape[0]
    return np.pad(wm, (0, max(0, n - wm.shape[0])))[:n]


def _filigrane_signal(x: np.ndarray, sr: int, message) -> np.ndarray:
    """Filigrane à ajouter à x (float32) : d'un seul tenant jusqu'à 30 s, par blocs de 30 s (+ 1 s de contexte)
    au-delà."""
    n = x.shape[0]
    bloc = int(BLOC_S * sr)
    if n <= bloc:
        return _filigrane_bloc(x, sr, message)
    ctx = int(CONTEXTE_S * sr)
    wm = np.empty(n, dtype=np.float32)
    for a in range(0, n, bloc):
        b = min(n, a + bloc)
        a0, b0 = max(0, a - ctx), min(n, b + ctx)
        wm[a:b] = _filigrane_bloc(x[a0:b0], sr, message)[a - a0:b - a0]
    return wm


def marquer(audio: np.ndarray, sr: int, message: list[int] | None = None) -> np.ndarray:
    """Pose le filigrane AudioSeal sur un son mono et renvoie un tableau de même forme et même type.

    audio : 1-D (ou [n, 1]), float32/float64 dans [-1, 1], ou int16 (PCM). sr : fréquence quelconque (entière).
    Renvoie float32 écrêté à [-1, 1] pour une entrée flottante, int16 (arrondi) pour
    une entrée int16."""
    a = np.asarray(audio)
    forme = a.shape
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim != 1:
        raise ValueError(f"son mono attendu, forme reçue {forme}")
    entier = a.dtype == np.int16
    if entier:
        x = a.astype(np.float32) / 32768.0
    elif np.issubdtype(a.dtype, np.floating):
        x = a.astype(np.float32, copy=False)
    else:
        raise TypeError(f"type non géré : {a.dtype} (float ou int16)")
    sr = int(sr)
    n = x.shape[0]
    if n == 0:
        return np.array(audio, copy=True)
    wm = _filigrane_signal(x, sr, message)
    y = np.clip(x + wm, -1.0, 1.0).astype(np.float32)
    if entier:
        y = np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
    return y.reshape(forme)


def marquer_wav(octets: bytes, **meta) -> bytes:
    """WAV PCM 16 bits mono -> WAV filigrané + métadonnées (les autres chunks sont recopiés)."""
    with wave.open(io.BytesIO(octets)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("WAV mono 16 bits attendu")
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    nouveau = marquer(pcm, sr).astype("<i2").tobytes()
    ch = _chunks(octets)
    corps = b"WAVE" + b"".join(_chunk(c, nouveau if c == b"data" else d) for c, d in ch)
    return metadonnees_wav(b"RIFF" + struct.pack("<I", len(corps)) + corps, **meta)


# ------------------------------------------------------------------ détecteur AudioSeal (ONNX)


def _session_detecteur():
    import onnxruntime as ort
    _assurer("LECTURA_FILIGRANE_DETECTEUR_ONNX", "filigrane/detector_base.onnx")
    chemin = chemin_detecteur()
    octets = chemin.read_bytes()
    if hashlib.sha256(octets).hexdigest() != SHA256_DETECTEUR:
        raise RuntimeError(f"{chemin} : empreinte sha256 inattendue")
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("LECTURA_FILIGRANE_FILS", "2"))
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = os.environ.get("LECTURA_FILIGRANE_ARENE", "1") != "0"
    return ort.InferenceSession(octets, so, providers=["CPUExecutionProvider"])


class _Detecteur:
    _session = None
    _verrou = threading.Lock()

    @classmethod
    def session(cls):
        if cls._session is None:
            with cls._verrou:
                if cls._session is None:
                    cls._session = _session_detecteur()
        return cls._session

    @classmethod
    def liberer(cls) -> None:
        """Oublie la session (contrôle ponctuel : la mémoire est rendue au prochain ramasse-miettes)."""
        cls._session = None


def detecter(audio: np.ndarray, sr: int, seuil: float = 0.5) -> tuple[float, list[int]]:
    """Détecteur AudioSeal (= AudioSealDetector.detect_watermark, seuils 0,5) : renvoie (p, bits), où p est la part
    des échantillons jugés marqués et bits les 16 bits décodés. Son mono float (-1..1) ou int16, sr quelconque."""
    a = np.asarray(audio)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    x = a.astype(np.float32) / 32768.0 if a.dtype == np.int16 else a.astype(np.float32, copy=False)
    x16 = reechantillonner(x, int(sr), SR_MODELE)
    p_trames, msg = _Detecteur.session().run(None, {"audio": np.ascontiguousarray(x16)[None, None, :]})
    p = float((p_trames[0] > seuil).mean())
    return p, [int(b) for b in (msg[0] > 0.5)]


def marque_retrouvee(p: float, bits: list[int]) -> bool:
    """Règle de contrôle commune : p >= 0,5 et les 16 bits égaux au message 0x4C01."""
    return p >= 0.5 and list(bits) == MESSAGE
