"""Métadonnées IA des WAV (IPTC trainedAlgorithmicMedia + texte d'information). Sans synthèse."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from lectura_diphone import _marquage, ecrire_wav

TEXTE_DROIT = (
    "Voix de synthèse générée par Lectura (synthèse par diphones, à partir "
    "d'enregistrements SIWIS sous CC BY 4.0). Contenu audio généré par ordinateur."
)


def test_texte_d_information_exact():
    assert _marquage.DESCRIPTION == TEXTE_DROIT
    assert _marquage.SOURCE.endswith("/digitalsourcetype/trainedAlgorithmicMedia")


def _son():
    t = np.arange(4410) / 44100
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.mark.parametrize("retimbre", [False, True])
def test_ecrire_wav(tmp_path, retimbre):
    p = tmp_path / "s.wav"
    octets = ecrire_wav(p, _son(), 44100, retimbre=retimbre)
    assert p.read_bytes() == octets
    m = _marquage.lire_metadonnees(octets)
    assert m["xmp_ia"] and m["info_ia"]
    if retimbre:
        assert "filigrane IA" in m["description"]
    else:
        assert m["description"] == TEXTE_DROIT
    with wave.open(io.BytesIO(octets)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 44100, 4410)


def test_metadonnees_n_alterent_pas_l_audio():
    brut = _marquage.wav_pcm16(_son(), 44100)
    marque = _marquage.metadonnees_wav(brut)
    data = lambda o: next(d for c, d in _marquage._chunks(o) if c == b"data")  # noqa: E731
    assert data(brut) == data(marque)
    # deux passages : un seul jeu de métadonnées
    deux = _marquage.metadonnees_wav(marque)
    assert sum(c == b"_PMX" for c, _ in _marquage._chunks(deux)) == 1


def test_diphoneresult_vers_wav(tmp_path):
    from lectura_diphone.engine import DiphoneResult

    r = DiphoneResult(samples=_son(), sample_rate=44100)
    m = _marquage.lire_metadonnees(r.vers_wav(tmp_path / "a.wav"))
    assert m["description"] == TEXTE_DROIT
    r2 = DiphoneResult(samples=_son(), sample_rate=44100, timbre_converti=True)
    assert "filigrane" in _marquage.lire_metadonnees(r2.vers_wav())["description"]
