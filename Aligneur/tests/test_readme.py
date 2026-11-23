"""Les exemples du README tournent (correction v0 du 23/11/2026 : MotAnalyse n'existait pas)."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

import lectura_aligneur
from lectura_aligneur import EspeakPhonemizer, LecturaSyllabeur

README = Path(__file__).resolve().parent.parent / "README.md"

pytestmark = pytest.mark.skipif(not README.is_file(), reason="README absent")


def _blocs_python() -> list[str]:
    return re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def test_imports_du_readme_existent():
    """Chaque nom importe de lectura_aligneur dans le README existe."""
    noms = []
    for bloc in _blocs_python():
        for ligne in re.findall(r"^from lectura_aligneur import (.+)$", bloc, re.M):
            noms += [n.strip() for n in ligne.split(",")]
    assert noms, "aucun import trouve dans le README"
    absents = [n for n in noms if not hasattr(lectura_aligneur, n)]
    assert absents == []


def test_exemple_analyser_complet():
    pytest.importorskip("lectura_phonemiseur.groupes_lecture")
    bloc = next(b for b in _blocs_python() if "analyser_complet(" in b)
    sortie = io.StringIO()
    with contextlib.redirect_stdout(sortie):
        exec(compile(bloc, "README.md", "exec"), {})  # noqa: S102
    texte = sortie.getvalue()
    assert "2 groupes, 4 syllabes" in texte
    assert "Syllabes : le.zɑ̃.fɑ̃ | ʒu" in texte
    assert "jonctions: liaison_z" in texte


def test_syllabify_ipa():
    assert LecturaSyllabeur().syllabify_ipa("ʃɔkɔla") == ["ʃɔ", "kɔ", "la"]


@pytest.mark.skipif(not EspeakPhonemizer().is_available(), reason="eSpeak NG absent")
def test_marques_muette_et_fusionnee():
    """° marque une lettre muette, ² une lettre qui produit deux phonemes (documentes au README)."""
    s = LecturaSyllabeur()
    choco = s.analyze("chocolat")
    assert choco.syllabes[-1].ortho == "lat°"
    taxi = s.analyze("taxi")
    assert any("²" in syl.ortho for syl in taxi.syllabes)
