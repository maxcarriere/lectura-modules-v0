"""G2P flexible — Protocol + backends locaux et personnalisé.

Convertit du texte en groupes prosodiques avec phones IPA. Tout tourne en
local : aucun appel à une API, aucune clé.

Backends (ordre de detection) :
  1. LecturaLocalG2P : lecteur_syllabique.Pipeline (Lectura Edition, s'il est installé)
  2. LecturaNlpG2P : lectura_g2p (pipeline unifie, pip install "lectura-diphone[g2p]")
  3. CallableG2P : enveloppe n'importe quel callable(text) → list[dict]

Chaque backend retourne une liste de groupes prosodiques :
  [{"phones": ["b", "ɔ̃"], "boundary": "comma"}, ...]
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol, runtime_checkable

log = logging.getLogger(__name__)

# Ponctuation → boundary type
_PUNCT_MAP = {
    "...": "suspensive", ",": "comma", ".": "period", "?": "question", "!": "exclamation",
    ";": "comma", ":": "comma",
    "\u2014": "comma", "\u2013": "comma",  # tirets cadratins/demi-cadratins
    "(": "comma", ")": "comma",            # parentheses
}
_PUNCT_BOUNDARIES = {"comma", "period", "question", "exclamation", "suspensive"}

# Ponctuation terminale de phrase
_SENTENCE_PUNCT = {".", "?", "!", "\u2026", "..."}


@runtime_checkable
class G2PBackend(Protocol):
    """Protocole pour un backend G2P."""

    def phonemize(self, text: str) -> list[dict]:
        """Convertit du texte en groupes prosodiques.

        Returns:
            Liste de dicts avec :
                - "phones": list[str] — phones IPA du groupe
                - "boundary": str — type de frontiere apres le groupe
                  ("comma", "period", "question", "exclamation", "none")
        """
        ...


class LecturaLocalG2P:
    """G2P via lecteur_syllabique.Pipeline (Lectura Edition complet)."""

    def __init__(self) -> None:
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from lecteur_syllabique.core.pipeline import Pipeline
        self._pipeline = Pipeline()

    def phonemize(self, text: str) -> list[dict]:
        """Texte → groupes prosodiques via Lectura Pipeline."""
        self._ensure_loaded()
        from lectura_diphone.phonemes import ipa_to_phones

        input_text = text.strip()
        if not input_text:
            return []
        # Ajouter ponctuation finale si absente
        if input_text[-1] not in ".?!":
            input_text += "."

        result = self._pipeline.run_steps(input_text)
        groups_out = result.presentation.syllabified

        # Collecter syllabes + boundaries par groupe
        syllables: list[str] = []
        boundaries: list[str] = []

        for gi, group in enumerate(groups_out):
            boundary = "none"

            if gi + 1 < len(groups_out):
                # Detecter ponctuation entre groupes
                next_g = groups_out[gi + 1]
                try:
                    cur_end = max(c.span[1] for c in group.components
                                  if hasattr(c, "span"))
                    nxt_start = min(c.span[0] for c in next_g.components
                                    if hasattr(c, "span"))
                    between = input_text[cur_end:nxt_start]
                except (ValueError, AttributeError):
                    between = ""

                for char, btype in _PUNCT_MAP.items():
                    if char in between:
                        boundary = btype
                        break

                if boundary == "none":
                    boundary = "word"
            else:
                # Dernier groupe : ponctuation terminale
                try:
                    cur_end = max(c.span[1] for c in group.components
                                  if hasattr(c, "span"))
                    trailing = input_text[cur_end:]
                    for char, btype in _PUNCT_MAP.items():
                        if char in trailing:
                            boundary = btype
                            break
                except (ValueError, AttributeError):
                    pass

            n_syls = len(group.syllabes)
            for si, syl in enumerate(group.syllabes):
                syllables.append(syl.phone)
                boundaries.append(boundary if si == n_syls - 1 else "none")

        # Regrouper en groupes prosodiques (split sur ponctuation)
        prosodic_groups: list[dict] = []
        current_phones: list[str] = []
        current_word_boundaries: list[int] = []

        for si, syl in enumerate(syllables):
            current_phones.extend(ipa_to_phones(syl))
            bnd = boundaries[si]

            is_punct = bnd in _PUNCT_BOUNDARIES
            is_last = si == len(syllables) - 1

            if bnd == "word" and not is_last:
                current_word_boundaries.append(len(current_phones))

            if is_punct or is_last:
                if current_phones:
                    prosodic_groups.append({
                        "phones": current_phones,
                        "boundary": bnd,
                        "word_boundaries": current_word_boundaries[:],
                    })
                    current_phones = []
                    current_word_boundaries = []

        return prosodic_groups


class LecturaNlpG2P:
    """G2P via lectura_g2p (pipeline unifie PyPI).

    Utilise le pipeline complet : tokeniseur → formules → phonemiseur
    → groupes de lecture avec liaisons et enchainements.
    """

    def __init__(self) -> None:
        self._engine = None

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        from lectura_g2p import creer_engine
        self._engine = creer_engine(mode="auto")

    def phonemize(self, text: str) -> list[dict]:
        """Texte → groupes prosodiques via le pipeline G2P unifie."""
        self._ensure_loaded()
        from lectura_g2p import texte_vers_phrases_ipa
        from lectura_diphone.phonemes import ipa_to_phones

        input_text = text.strip()
        if not input_text:
            return []

        phrases = texte_vers_phrases_ipa(input_text, engine=self._engine)

        # Convertir les phrases IPA au format diphone (groupes prosodiques)
        # Chaque phrase IPA contient des espaces (frontieres de mots) et
        # de la ponctuation (,  .  ?  !  …) comme separateurs de groupes.
        _boundary_map = {
            ",": "comma", ".": "period", "?": "question",
            "!": "exclamation", "\u2026": "suspensive",
        }

        groups: list[dict] = []

        for ipa_str, _phrase_type in phrases:
            current_phones: list[str] = []
            current_word_boundaries: list[int] = []

            # Parcourir les caracteres de l'IPA
            i = 0
            while i < len(ipa_str):
                ch = ipa_str[i]

                if ch in _boundary_map:
                    # Ponctuation → fermer le groupe
                    if current_phones:
                        groups.append({
                            "phones": current_phones,
                            "boundary": _boundary_map[ch],
                            "word_boundaries": current_word_boundaries,
                        })
                        current_phones = []
                        current_word_boundaries = []
                    i += 1
                elif ch == " ":
                    # Espace → frontiere de mot
                    if current_phones:
                        current_word_boundaries.append(len(current_phones))
                    i += 1
                else:
                    # Phone IPA — extraire via ipa_to_phones sur le segment
                    # jusqu'au prochain espace ou ponctuation
                    end = i
                    while end < len(ipa_str) and ipa_str[end] not in (" ", ",", ".", "?", "!", "\u2026"):
                        end += 1
                    segment = ipa_str[i:end]
                    phones = ipa_to_phones(segment)
                    current_phones.extend(phones)
                    i = end

            # Dernier groupe de la phrase
            if current_phones:
                groups.append({
                    "phones": current_phones,
                    "boundary": "period",
                    "word_boundaries": current_word_boundaries,
                })

        return groups


class CallableG2P:
    """Wrapper pour un callable custom (text) → list[dict]."""

    def __init__(self, fn: Callable[[str], list[dict]]) -> None:
        self._fn = fn

    def phonemize(self, text: str) -> list[dict]:
        return self._fn(text)


def auto_detect_g2p(preference: str | None = None) -> G2PBackend:
    """Auto-detecte le meilleur backend G2P local disponible.

    Cascade :
      1. lecteur_syllabique (Lectura Edition complet)
      2. lectura_g2p (modules PyPI : pip install "lectura-diphone[g2p]")

    Args:
        preference: "local" ou None (auto) ; les deux ont le même effet.

    Raises:
        ImportError: aucun G2P local installé.
        ValueError: préférence inconnue (le mode "api" n'existe plus).
    """
    if preference not in (None, "auto", "local"):
        raise ValueError(
            f"Préférence G2P inconnue : {preference!r}. Le mode \"api\" n'existe plus : "
            "tout tourne en local."
        )

    # 1. lecteur_syllabique (pipeline complet Lectura Edition)
    try:
        import lecteur_syllabique  # noqa: F401
        log.debug("G2P local disponible (lecteur_syllabique)")
        return LecturaLocalG2P()
    except ImportError:
        pass

    # 2. lectura_g2p (pipeline unifie PyPI)
    try:
        import lectura_g2p  # noqa: F401
        log.debug("G2P disponible (lectura_g2p)")
        return LecturaNlpG2P()
    except ImportError:
        raise ImportError(
            "Aucun G2P local installé. Installez-le avec : "
            "pip install \"lectura-diphone[g2p]\", ou passez g2p=<callable> "
            "(texte -> groupes de phones), ou synthétisez directement des phonèmes "
            "avec engine.synthesize_groups()."
        ) from None
