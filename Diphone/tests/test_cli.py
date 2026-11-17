"""Interface en ligne de commande, sans banque ni synthèse."""

from __future__ import annotations

import pytest

from lectura_diphone import __version__, _cli, _poids


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        _cli.main(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_groupes_depuis_phonemes():
    g = _cli._groupes_depuis_phonemes("b ɔ̃ ʒ u ʁ | l ə m ɔ̃ d")
    assert g == [
        {"phones": ["b", "ɔ̃", "ʒ", "u", "ʁ"], "boundary": "comma"},
        {"phones": ["l", "ə", "m", "ɔ̃", "d"], "boundary": "period"},
    ]


def test_verifier_dossier_vide(tmp_path, capsys):
    assert _cli.main(["--verifier", "--models-dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert all(nom in out for nom in _poids.FICHIERS)


def test_voix_hors_liste_refusee():
    with pytest.raises(SystemExit):
        _cli.main(["bonjour", "--voix", "ref.wav"])


def test_sans_texte():
    with pytest.raises(SystemExit):
        _cli.main([])
