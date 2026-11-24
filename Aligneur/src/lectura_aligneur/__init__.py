"""Lectura Aligneur-Syllabeur — Aligneur grapheme-phoneme et syllabeur phonologique du francais.

Pivot central du pipeline Lectura. Realise l'alignement lettre-par-lettre
entre orthographe et phonetique, et decompose chaque syllabe en
attaque/noyau/coda avec correspondance grapheme-phoneme.

La construction des groupes de lecture (E1 : elisions, liaisons,
enchainements) est desormais dans le module G2P (lectura_phonemiseur.groupes_lecture).

Tout tourne en local, sans appel reseau.

Copyright (C) 2025-2026 Maxime Carrière (Lectura)
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — contact via https://www.lectura.world
"""

# Types publics (toujours disponibles)
from lectura_aligneur._types import (
    Span,
    Phoneme,
    GroupePhonologique,
    Syllabe,
    ResultatAnalyse,
    EventFormule,
    LectureFormule,
    OptionsGroupes,
    GroupeLecture,
    ResultatGroupe,
    ResultatSyllabation,
)

# Utilitaires phonologiques (toujours disponibles, zero dependance)
from lectura_aligneur._utilitaires import (
    iter_phonemes,
    est_voyelle,
    est_consonne,
    est_semi_voyelle,
)

# Tout tourne en local (aucun appel reseau)
from lectura_aligneur.lectura_aligneur import (
    # Protocoles
    Phonemizer,
    EspeakPhonemizer,
    # Fonctions E2
    lecture_depuis_g2p,
    syllabifier_groupes,
    _valider_spans_formule,
    # Classe principale
    LecturaSyllabeur,
)
_MODE = "local"

# Re-export depuis le G2P pour retrocompatibilite
try:
    from lectura_phonemiseur.groupes_lecture import (
        construire_groupes_lecture,
        OptionsGroupes as _G2POptionsGroupes,
        ajouter_schwa_final,
    )
except ImportError:
    def construire_groupes_lecture(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "construire_groupes_lecture() requiert le module lectura_phonemiseur. "
            "Installez-le : pip install lectura-phonemiseur"
        )

    def ajouter_schwa_final(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError(
            "ajouter_schwa_final() requiert le module lectura_phonemiseur. "
            "Installez-le : pip install lectura-phonemiseur"
        )


__version__ = "4.0.7"
