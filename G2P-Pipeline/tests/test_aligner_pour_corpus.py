"""aligner_pour_corpus() : retirée de la v0 (correction du 23/11/2026).

Avant : AttributeError/ImportError obscur (``align_for_corpus`` introuvable
dans lectura-aligneur 4.0.5). Maintenant : NotImplementedError explicite,
sans créer de moteur ni télécharger de poids.
"""

import pytest

import lectura_g2p


def test_aligner_pour_corpus_message_clair():
    with pytest.raises(NotImplementedError, match="pas disponible dans la v0"):
        lectura_g2p.aligner_pour_corpus("Les enfants jouent.")


def test_aucun_nom_interne_absent_de_l_aligneur():
    """Le paquet ne référence plus d'outil absent de lectura-aligneur."""
    import inspect

    source = inspect.getsource(lectura_g2p)
    assert "import align_for_corpus" not in source
    assert "align_for_corpus as" not in source
