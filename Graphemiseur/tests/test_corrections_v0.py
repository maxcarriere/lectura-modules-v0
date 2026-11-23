"""Tests des corrections v0 du 23/11/2026 (défauts relevés par la documentation)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from lectura_graphemiseur import corriger_phrase_v3, creer_engine

README = Path(__file__).resolve().parent.parent / "README.md"


def test_lectura_p2g_version_v6_message_clair(monkeypatch):
    from lectura_graphemiseur import _resoudre_modeles_dir
    d = _resoudre_modeles_dir(telecharger=False)
    if d is None:
        pytest.skip("poids P2G absents")
    if (d / "unifie_p2g_v6_int8.onnx").exists():
        pytest.skip("modèle V6 présent localement")
    monkeypatch.setenv("LECTURA_P2G_VERSION", "v6")
    with pytest.raises(FileNotFoundError, match="V6 n'est pas publie"):
        creer_engine()


def test_readme_aligne_sur_corriger_phrase_v3():
    texte = README.read_text(encoding="utf-8")
    assert "ipa_words" not in inspect.signature(corriger_phrase_v3).parameters
    assert "ipa_words=None" not in texte
    # Les formules ne sont pas appliquées par ce paquet (elles sont dans lectura-p2g)
    assert "via `lectura_formules`" not in texte
    assert "LECTURA_P2G_VERSION" in texte
