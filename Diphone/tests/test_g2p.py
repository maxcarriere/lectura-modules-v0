"""Tests pour lectura_diphone.g2p."""

import pytest


def test_g2p_protocol():
    """G2PBackend est un Protocol runtime-checkable."""
    from lectura_diphone.g2p import G2PBackend, CallableG2P

    def dummy(text):
        return [{"phones": ["a"], "boundary": "none"}]

    backend = CallableG2P(dummy)
    assert isinstance(backend, G2PBackend)


def test_callable_g2p():
    """CallableG2P wraps un callable."""
    from lectura_diphone.g2p import CallableG2P

    def my_g2p(text):
        return [{"phones": ["b", "a"], "boundary": "period"}]

    backend = CallableG2P(my_g2p)
    result = backend.phonemize("ba")
    assert len(result) == 1
    assert result[0]["phones"] == ["b", "a"]
    assert result[0]["boundary"] == "period"


def test_callable_g2p_empty():
    """CallableG2P retourne une liste vide."""
    from lectura_diphone.g2p import CallableG2P

    backend = CallableG2P(lambda t: [])
    assert backend.phonemize("") == []


def test_pas_de_backend_api():
    """Mode local seul : plus de backend API ni d'URL."""
    import lectura_diphone.g2p as g

    assert not hasattr(g, "LecturaApiG2P")
    src = open(g.__file__, encoding="utf-8").read()
    assert "urllib" not in src and "api.lectura" not in src


def test_preference_api_refusee():
    from lectura_diphone.g2p import auto_detect_g2p

    with pytest.raises(ValueError):
        auto_detect_g2p(preference="api")


def test_auto_detect_sans_g2p(monkeypatch):
    """Sans G2P local installé : ImportError explicite (pas de repli réseau)."""
    import builtins

    from lectura_diphone.g2p import auto_detect_g2p

    vrai_import = builtins.__import__

    def faux_import(nom, *a, **k):
        if nom.split(".")[0] in ("lecteur_syllabique", "lectura_g2p"):
            raise ImportError(nom)
        return vrai_import(nom, *a, **k)

    monkeypatch.setattr(builtins, "__import__", faux_import)
    with pytest.raises(ImportError, match="lectura-diphone\\[g2p\\]"):
        auto_detect_g2p()
