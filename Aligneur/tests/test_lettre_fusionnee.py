"""Lettre fusionnée ² (correctif v0 du 24/11/2026) : « taxi » donnait « xx²i »."""

from __future__ import annotations

import pytest

from lectura_aligneur import LecturaSyllabeur


@pytest.mark.parametrize(
    "mot, phone, attendu",
    [
        ("taxi", "taksi", ["ta", "x²i"]),
        ("examen", "ɛɡzamɛ̃", ["e", "x²a", "men"]),
        ("exact", "ɛɡzakt", ["e", "x²act"]),
        ("boxer", "bɔkse", ["bo", "x²er"]),
        ("boxe", "bɔks", ["box²e°"]),
        ("extra", "ɛkstʁa", ["ex²", "tra"]),
        ("xylophone", "ɡzilɔfɔn", ["x²y", "lo", "phone°"]),
    ],
)
def test_x_ne_se_double_pas(mot, phone, attendu):
    res = LecturaSyllabeur().analyze(mot, phone=phone)
    orthos = [s.ortho for s in res.syllabes]
    assert orthos == attendu
    assert all("xx" not in o for o in orthos)


def test_spans_et_graphemes_inchanges():
    """La correction ne touche que l'affichage : les phonèmes gardent leur lettre."""
    res = LecturaSyllabeur().analyze("taxi", phone="taksi")
    s2 = res.syllabes[1]
    assert s2.span == (2, 4)
    assert [p.grapheme for p in s2.attaque.phonemes] == ["x", "x"]


def test_y_pont_inchange():
    """Le y partagé entre deux syllabes garde sa notation entre parenthèses."""
    res = LecturaSyllabeur().analyze("crayon", phone="kʁɛjɔ̃")
    assert [s.ortho for s in res.syllabes] == ["cra(y)", "yon"]
