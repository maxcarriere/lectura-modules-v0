"""Marquage IA des sons de lectura-diphone : métadonnées WAV (IPTC + LIST/INFO).

Ne pas retirer le marquage.

- Synthèse par diphones (règles, sans réseau de neurones) : métadonnées seules,
  type de source IPTC ``trainedAlgorithmicMedia`` (la banque est tirée
  d'enregistrements) et texte d'information.
- Changement de timbre (OpenVoice, réseau de neurones) : le son porte en plus le
  filigrane IA AudioSeal, posé par lectura-vc-zeroshot ; les métadonnées le
  disent.

Format (identique à lectura-vc-zeroshot) : chunk ``_PMX`` (XMP, avec
``Iptc4xmpExt:DigitalSourceType``) et chunk ``LIST/INFO`` (INAM, ISFT, ICMT,
ICOP). Bibliothèque standard et numpy seulement.

    from lectura_diphone import ecrire_wav
    ecrire_wav("sortie.wav", resultat.samples, resultat.sample_rate)
"""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

SOURCE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
CREDIT = "Lectura"
OUTIL = "Lectura, synthèse vocale par diphones (lectura-diphone)"
OUTIL_RETIMBRE = "Lectura, synthèse par diphones (lectura-diphone) et conversion de timbre par IA (lectura-vc-zeroshot, OpenVoice)"

# Texte d'information (Lectura Droit, V0-PUBLICATION §4). Aucun nom de personne.
DESCRIPTION = (
    "Voix de synthèse générée par Lectura (synthèse par diphones, à partir "
    "d'enregistrements SIWIS sous CC BY 4.0). Contenu audio généré par ordinateur."
)
DESCRIPTION_RETIMBRE = (
    "Voix de synthèse générée par Lectura (synthèse par diphones, à partir "
    "d'enregistrements SIWIS sous CC BY 4.0, puis timbre converti par intelligence "
    "artificielle). Contenu audio généré par ordinateur, porteur d'un filigrane IA."
)


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


def _chunks(octets: bytes) -> list[tuple[bytes, bytes]]:
    if octets[:4] != b"RIFF" or octets[8:12] != b"WAVE":
        raise ValueError("pas un WAV RIFF")
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


def metadonnees_wav(octets: bytes, retimbre: bool = False) -> bytes:
    """Renvoie le WAV avec les chunks ``_PMX`` (XMP) et ``LIST/INFO`` de marquage IA.

    Les anciens chunks de marquage sont remplacés ; ``fmt``, ``data`` et les
    autres chunks sont recopiés tels quels, dans le même ordre (audio inchangé).
    ``retimbre=True`` : textes du changement de timbre (filigrane présent).
    """
    outil = OUTIL_RETIMBRE if retimbre else OUTIL
    description = DESCRIPTION_RETIMBRE if retimbre else DESCRIPTION
    info_d = {
        b"INAM": description,
        b"ISFT": outil,
        b"ICMT": f"AI-Generated: true; DigitalSourceType: {SOURCE}"
                 + ("; Watermark: AudioSeal" if retimbre else ""),
        b"ICOP": CREDIT,
    }
    garde = [(c, d) for c, d in _chunks(octets) if c != b"_PMX" and not _est_info(c, d)]
    info = b"INFO" + b"".join(_chunk(k, v.encode("utf-8") + b"\0") for k, v in info_d.items())
    corps = (b"WAVE" + b"".join(_chunk(c, d) for c, d in garde)
             + _chunk(b"LIST", info) + _chunk(b"_PMX", _xmp(outil, description, CREDIT).encode("utf-8")))
    return b"RIFF" + struct.pack("<I", len(corps)) + corps


def lire_metadonnees(octets: bytes) -> dict:
    """Résumé du marquage d'un WAV : {"xmp_ia", "info_ia", "description"}."""
    ch = _chunks(octets)
    xmp = next((d.decode("utf-8", "replace") for c, d in ch if c == b"_PMX"), "")
    desc = ""
    if "<rdf:li" in xmp:
        desc = xmp.split('xml:lang="x-default">', 1)[-1].split("</rdf:li>", 1)[0]
    return {
        "xmp_ia": SOURCE in xmp,
        "info_ia": any(_est_info(c, d) and SOURCE.encode() in d for c, d in ch),
        "description": desc,
    }


def wav_pcm16(audio, sr: int) -> bytes:
    """Encode un son mono float (-1..1) en WAV PCM 16 bits, sans métadonnées."""
    import numpy as np

    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = np.clip(np.round(a * 32767.0), -32768, 32767).astype("<i2")
    tampon = io.BytesIO()
    with wave.open(tampon, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return tampon.getvalue()


def ecrire_wav(chemin: str | Path | None, audio, sr: int, retimbre: bool = False) -> bytes:
    """Écrit (si ``chemin``) et renvoie un WAV PCM 16 bits mono marqué (métadonnées IA).

    ``retimbre=True`` pour un son passé par le changement de timbre (filigrane
    déjà posé par lectura-vc-zeroshot ; cette fonction n'y touche pas).
    """
    octets = metadonnees_wav(wav_pcm16(audio, sr), retimbre=retimbre)
    if chemin is not None:
        Path(chemin).write_bytes(octets)
    return octets
