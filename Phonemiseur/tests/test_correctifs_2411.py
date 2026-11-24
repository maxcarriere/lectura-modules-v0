"""Correctif v0 du 24/11/2026 : propriété publique ``ResultatPhraseG2P.groupes``."""

from __future__ import annotations

from lectura_phonemiseur import (
    GroupeLecture,
    MotAnalyseG2P,
    ResultatPhraseG2P,
    construire_groupes_lecture,
)


def _phrase() -> ResultatPhraseG2P:
    return ResultatPhraseG2P(mots=[
        MotAnalyseG2P(text="Les", phone="le", pos="ART:def", liaison="Lz"),
        MotAnalyseG2P(text="enfants", phone="ɑ̃fɑ̃", pos="NOM"),
        MotAnalyseG2P(text="jouent", phone="ʒu", pos="VER"),
    ])


def test_groupes_calcules_a_la_demande():
    r = _phrase()
    groupes = r.groupes
    assert all(isinstance(g, GroupeLecture) for g in groupes)
    assert [g.text for g in groupes] == ["Les enfants", "jouent"]
    assert groupes[0].jonctions == ["liaison_z"]
    assert r.groupes is groupes  # calculés une seule fois


def test_groupes_attaches_prioritaires():
    """Les groupes attachés par lectura-g2p (avec leurs options) sont rendus tels quels."""
    r = _phrase()
    attaches = construire_groupes_lecture(r)[:1]
    r._groupes = attaches
    assert r.groupes is attaches


def test_groupes_hors_des_champs():
    """La propriété ne change pas les champs de la dataclass."""
    import dataclasses

    assert [f.name for f in dataclasses.fields(ResultatPhraseG2P)] == ["mots"]
