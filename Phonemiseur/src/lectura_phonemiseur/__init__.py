"""Lectura NLP — Modele unifie G2P+POS+Morpho+Liaison pour le francais.

Architecture : BiLSTM char-level + multi-tete (2.0M params, ONNX INT8 = 2.0 Mo)

Copyright (C) 2025-2026 Maxime Carrière (Lectura)
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — contact via https://www.lectura.world

Tout tourne en local (ONNX Runtime). Les poids ne sont pas dans la roue : ils
sont telecharges depuis Hugging Face au premier usage, verifies par SHA-256, et
ranges dans ~/.lectura/models/g2p/ (ou $LECTURA_MODELS_DIR/g2p/). Poids sous
licence CC BY-NC 4.0 (voir MODEL_LICENCE.md). Voir le module ``_poids``.

Exemple rapide::

    from lectura_phonemiseur import creer_engine
    engine = creer_engine()
    result = engine.analyser(["bonjour", "monde"])
"""

import os
from pathlib import Path

from lectura_phonemiseur import _poids
from lectura_phonemiseur._poids import LecturaModelesIndisponibles

__version__ = "4.2.3"

_MODELES_DIR = Path(__file__).parent / "modeles"


def get_model_path(filename: str) -> Path:
    """Retourne le chemin absolu vers un fichier modele embarque."""
    return _MODELES_DIR / filename


def _resoudre_modeles_dir(
    models_dir: str | Path | None = None, telecharger: bool = True,
) -> Path | None:
    """Cascade de resolution du dossier modeles.

    Ordre de priorite (premier dossier ou tous les poids sont presents) :
    1. Parametre explicite ``models_dir``
    2. Variable d'environnement ``LECTURA_MODELS_DIR`` / g2p
    3. Dossier utilisateur ``~/.lectura/models/g2p/``
    4. Site-packages (dossier ``modeles/`` du package installe)

    Si aucun dossier n'est complet et ``telecharger`` est vrai, les poids sont
    telecharges depuis Hugging Face dans le dossier 1, sinon 2, sinon 3
    (voir ``_poids.assurer_modeles``) ; sinon renvoie None.
    """
    candidats: list[Path] = []
    if models_dir is not None:
        candidats.append(Path(models_dir))
    env = os.environ.get("LECTURA_MODELS_DIR")
    if env:
        candidats.append(Path(env) / "g2p")
    candidats.append(Path.home() / ".lectura" / "models" / "g2p")
    candidats.append(_MODELES_DIR)

    for d in candidats:
        if not _poids.fichiers_manquants(d):
            return d
    if telecharger:
        return _poids.assurer_modeles(models_dir)
    return None


def _resoudre_lexique(
    lexicon_path: str | Path | None,
    models_dir: Path | None,
) -> str | Path | dict | None:
    """Cascade de resolution du lexique pour les lex_features.

    1. Chemin explicite (si passe)
    2. None (pas de lexique -> lex_features = zeros)
    """
    if lexicon_path is not None:
        return lexicon_path
    # Plus de construction automatique depuis lectura-lexique (retirée le 23/11/2026) : sans base fournie par
    # l'utilisateur, elle échouait toujours en silence ; mesurée avec un lexique fourni, l'aide est négligeable
    # (+0,01 point de mots justes sur le test du cadre). Un lexique de catégories reste utilisable par lexicon_path.
    return None


class _EngineAvecCorrections:
    """Wrapper qui applique le post-traitement (corrections, homographes,
    regles) automatiquement apres l'inference."""

    def __init__(self, engine: object, corrections_lexique: bool = False) -> None:
        self._engine = engine
        self._corrections_lexique = corrections_lexique
        self._loaded = False

    def _charger_tables(self) -> None:
        if self._loaded:
            return
        from lectura_phonemiseur.posttraitement import (
            charger_corrections, charger_corrections_supplementaires, charger_homographes,
        )
        data_dir = Path(__file__).parent / "data"
        # 1) Corrections de base (table plate)
        corr = data_dir / "g2p_corrections_unifie.json"
        if corr.exists():
            charger_corrections(corr)
        # 2) Corrections etendues du lexique (optionnel, priorite plus basse)
        if self._corrections_lexique:
            corr_lex = data_dir / "g2p_corrections_lexique.json"
            if corr_lex.exists():
                charger_corrections_supplementaires(corr_lex)
        # 3) Homographes (priorite la plus haute, retire les conflits)
        homo = data_dir / "homographes.json"
        if homo.exists():
            charger_homographes(homo)
        self._loaded = True

    def analyser(self, tokens: list[str], **kwargs) -> dict:
        self._charger_tables()
        result = self._engine.analyser(tokens, **kwargs)
        from lectura_phonemiseur.posttraitement import corriger_g2p
        keep_sep = bool(kwargs.get("sep_hyphen") or kwargs.get("sep_apos"))
        g2p_list = result.get("g2p", [])
        pos_list = result.get("pos", [])
        result["g2p"] = [
            corriger_g2p(tok, ipa, pos, keep_sep=keep_sep)
            for tok, ipa, pos in zip(tokens, g2p_list, pos_list)
        ]
        return result

    def __getattr__(self, name: str):
        return getattr(self._engine, name)


def creer_engine(
    mode: str = "auto",
    models_dir: str | Path | None = None,
    lexicon_path: str | Path | None = None,
    corrections_lexique: bool = True,
):
    """Factory pour creer un engine d'inference G2P.

    Le post-traitement (corrections, homographes, regles) est applique
    automatiquement sur les resultats de ``analyser()``.

    Parameters
    ----------
    mode : str
        "auto" (defaut), "local" ou "onnx" : ONNX Runtime, en local.
        Les modes "numpy" et "pure" ne sont pas disponibles dans la v0 : ils
        demandent des poids convertis (JSON) qui ne sont pas publies ; ils
        levent ValueError. Le mode "api" n'existe plus.
    models_dir : str | Path | None
        Chemin vers le dossier contenant les modeles. Si None, cascade
        automatique : LECTURA_MODELS_DIR → ~/.lectura/models/g2p/ → site-packages.
        Poids absents : telechargement depuis Hugging Face au premier usage
        (desactive par LECTURA_HORS_LIGNE=1).
    lexicon_path : str | Path | None
        Chemin vers le fichier lexique POS (JSON).
        Si None, resolution automatique (cascade).
    corrections_lexique : bool
        Si True (defaut), charge les corrections etendues du lexique
        (g2p_corrections_lexique.json, ~30K entrees).
        Ameliore la precision sur les mots rares et les anglicismes.
    """
    if mode == "api":
        raise ValueError(
            "Le mode 'api' a ete retire : lectura-phonemiseur tourne en local, avec des "
            "poids telecharges depuis Hugging Face au premier usage. Utilisez mode='auto'."
        )
    if mode in ("numpy", "pure"):
        raise ValueError(
            f"Le mode {mode!r} n'est pas disponible dans la v0 : il demande des poids "
            "convertis (JSON) qui ne sont pas publies. Utilisez mode='auto' ou 'onnx' "
            "(ONNX Runtime, installe avec le paquet)."
        )
    if mode not in ("auto", "local", "onnx"):
        raise ValueError(f"Mode inconnu : {mode!r}")

    resolved_dir = _resoudre_modeles_dir(models_dir)

    # Mode local — V4 uniquement
    base = resolved_dir
    model_onnx = base / "unifie_v4_int8.onnx"
    model_vocab = base / "unifie_v4_vocab.json"
    resolved_lexicon = _resoudre_lexique(lexicon_path, resolved_dir)

    engine = None
    erreur_onnx: Exception | None = None

    if mode in ("auto", "local", "onnx"):
        try:
            from lectura_phonemiseur.inference_onnx_v2 import OnnxInferenceEngineV2
            if isinstance(resolved_lexicon, dict):
                engine = OnnxInferenceEngineV2(
                    str(model_onnx), str(model_vocab),
                    lexicon=resolved_lexicon,
                )
            else:
                engine = OnnxInferenceEngineV2(
                    str(model_onnx), str(model_vocab),
                    lexicon_path=str(resolved_lexicon) if resolved_lexicon else None,
                )
        except Exception as exc:  # cause gardee pour l'erreur finale
            erreur_onnx = exc
            if mode == "onnx":
                raise

    if engine is None:
        raise RuntimeError(
            f"Aucun backend d'inference disponible (mode={mode!r}). "
            "Verifiez que onnxruntime est installe (pip install onnxruntime) "
            f"et que les poids sont presents dans {base}."
        ) from erreur_onnx

    return _EngineAvecCorrections(engine, corrections_lexique=corrections_lexique)


# API publique
from lectura_phonemiseur.tokeniseur import tokeniser, phrase_vers_chars
from lectura_phonemiseur.posttraitement import (
    appliquer_liaison,
    appliquer_regles_g2p,
    charger_corrections,
    charger_corrections_supplementaires,
    charger_homographes,
    corriger_g2p,
)
_NOMS_PIPELINE_FORMULES = ("analyser_phrase_complete", "ResultatPhraseG2P", "MotAnalyseG2P")
_ERREUR_FORMULES: ImportError | None = None
try:
    from lectura_phonemiseur.pipeline_formules import (
        analyser_phrase_complete,
        ResultatPhraseG2P,
        MotAnalyseG2P,
    )
except ImportError as _exc:
    # lectura-formules est optionnel (extra [formules]) : le reste du paquet
    # fonctionne sans lui ; l'acces a ces noms leve une erreur explicite.
    _ERREUR_FORMULES = _exc


def __getattr__(name: str):
    if name in _NOMS_PIPELINE_FORMULES and _ERREUR_FORMULES is not None:
        raise ImportError(
            f"{name} requiert le paquet lectura-formules. "
            "Installez-le : pip install 'lectura-phonemiseur[formules]' "
            f"(cause : {_ERREUR_FORMULES})"
        ) from _ERREUR_FORMULES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

try:
    from lectura_phonemiseur.groupes_lecture import (
        construire_groupes_lecture,
        GroupeLecture,
        OptionsGroupes,
        ajouter_schwa_final,
    )
except ImportError:
    pass  # groupes_lecture desactive si dependances manquantes
