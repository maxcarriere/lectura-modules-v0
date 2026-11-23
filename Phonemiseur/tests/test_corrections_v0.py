"""Corrections v0 du 23/11/2026.

- Modes ``numpy`` et ``pure`` : avant, IsADirectoryError ; maintenant ValueError
  explicite, sans chercher ni telecharger de poids.
- ``lectura_formules`` : declare dans l'extra ``[formules]`` ; sans lui,
  ``analyser_phrase_complete`` leve une ImportError qui dit quoi installer.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from lectura_phonemiseur import creer_engine

RACINE = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("mode", ["numpy", "pure"])
def test_modes_numpy_pure_message_clair(mode, monkeypatch):
    import lectura_phonemiseur

    def _interdit(*_a, **_k):
        raise AssertionError("aucune resolution de poids attendue")

    monkeypatch.setattr(lectura_phonemiseur, "_resoudre_modeles_dir", _interdit)
    with pytest.raises(ValueError, match="pas disponible dans la v0"):
        creer_engine(mode=mode)


def test_mode_inconnu():
    with pytest.raises(ValueError, match="Mode inconnu"):
        creer_engine(mode="gpu")


def test_formules_declare_en_extra():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    meta = tomllib.loads((RACINE / "pyproject.toml").read_text(encoding="utf-8"))
    extras = meta["project"]["optional-dependencies"]
    assert any(d.startswith("lectura-formules") for d in extras["formules"])
    assert any(d.startswith("lectura-formules") for d in extras["all"])


def test_sans_formules_erreur_explicite():
    """lectura_formules masque : import du paquet OK, analyser_phrase_complete explicite."""
    code = textwrap.dedent("""
        import sys
        sys.modules["lectura_formules"] = None   # simule le paquet absent
        import lectura_phonemiseur
        assert lectura_phonemiseur.creer_engine
        try:
            from lectura_phonemiseur import analyser_phrase_complete
        except ImportError as exc:
            assert "lectura-formules" in str(exc), exc
            assert "[formules]" in str(exc), exc
            print("OK")
        else:
            raise SystemExit("ImportError attendue")
    """)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "OK"


def test_avec_formules_nom_disponible():
    pytest.importorskip("lectura_formules")
    from lectura_phonemiseur import analyser_phrase_complete

    assert callable(analyser_phrase_complete)
