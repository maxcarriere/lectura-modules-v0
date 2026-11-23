"""Lectura P2G — Modele unifie P2G+POS+Morpho V7 pour le francais (IPA -> orthographe).

Architecture : BiLSTM char-level + word feedback multi-tete V7
  avec phone_lex_features (28d) + lex_select (3.2M params, ONNX INT8 = 4.4 Mo)

Modele core : raw → lex_select → post-traitement (coherence morpho + accents)
  Les formules (nombres, sigles) et noms propres sont dans lectura-p2g
  (pipeline couche 2).

Copyright (C) 2025-2026 Maxime Carrière (Lectura)
Licence : AGPL-3.0-or-later — voir LICENCE.txt
Licence commerciale disponible — contact via https://www.lectura.world

Tout tourne en local (ONNX Runtime). Les poids ne sont pas dans la roue : ils
sont telecharges depuis Hugging Face au premier usage, verifies par SHA-256, et
ranges dans ~/.lectura/models/p2g/ (ou $LECTURA_MODELS_DIR/p2g/). Poids sous
licence CC BY-NC 4.0 (voir MODEL_LICENCE.md) ; phone_lexicon.db sous CC BY-SA 4.0
(voir NOTICE). Voir le module ``_poids``.

Exemple rapide::

    from lectura_graphemiseur import creer_engine
    engine = creer_engine()
    result = engine.analyser(["l\u0259", "\u0283a", "\u025b", "b\u0254\u0303"])
    print(result["ortho"])  # ['le', 'chat', 'est', 'bon']
"""

import logging
import os
from pathlib import Path

from lectura_graphemiseur import _poids
from lectura_graphemiseur._poids import LecturaModelesIndisponibles

__version__ = "4.4.1"

logger = logging.getLogger(__name__)

_MODELES_DIR = Path(__file__).parent / "modeles"


def get_model_path(filename: str) -> Path:
    """Retourne le chemin absolu vers un fichier modele embarque."""
    return _MODELES_DIR / filename


def _resoudre_modeles_dir(
    models_dir: str | Path | None = None, telecharger: bool = True,
) -> Path | None:
    """Cascade de resolution du dossier modeles.

    Si aucun dossier n'est complet (tous les fichiers de ``_poids.FICHIERS``) et
    ``telecharger`` est vrai, les poids sont telecharges depuis Hugging Face dans
    le dossier 1, sinon 2, sinon 3 (voir ``_poids.assurer_modeles``).

    Ordre de priorite (premier dossier complet) :
    1. Parametre explicite ``models_dir``
    2. Variable d'environnement ``LECTURA_MODELS_DIR`` / p2g
    3. Dossier utilisateur ``~/.lectura/models/p2g/``
    4. Site-packages (dossier ``modeles/`` du package installe)
    """
    candidats: list[Path] = []
    if models_dir is not None:
        candidats.append(Path(models_dir))
    env = os.environ.get("LECTURA_MODELS_DIR")
    if env:
        candidats.append(Path(env) / "p2g")
    candidats.append(Path.home() / ".lectura" / "models" / "p2g")
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
    2. Fichier JSON dans le dossier modeles resolu
    3. None (pas de lexique -> lex_features = zeros)
    """
    if lexicon_path is not None:
        return lexicon_path
    if models_dir is not None:
        json_path = models_dir / "lexique_pos_candidates.json"
        if json_path.exists():
            return json_path
    # Plus de construction automatique depuis lectura-lexique (retirée le 23/11/2026) : sans base fournie par
    # l'utilisateur, elle échouait toujours en silence ; mesurée avec un lexique fourni, l'aide est négligeable
    # (+0,01 point de mots justes sur le test du cadre). Un lexique de catégories reste utilisable par lexicon_path.
    return None


def _resoudre_phone_lexicon(models_dir: Path | None):
    """Cascade de resolution du phone_lexicon pour V6.

    1. phone_lexicon.db dans le dossier modeles
    2. lexique_correcteur.db dans le dossier modeles
    3. lectura_correcteur data dir (si installe)
    4. None (degradation gracieuse, features = zeros)
    """
    from lectura_graphemiseur._phone_lexicon import PhoneLexicon

    # 1. DB dediee dans le dossier modeles
    if models_dir is not None:
        db = models_dir / "phone_lexicon.db"
        if db.exists():
            try:
                return PhoneLexicon(db)
            except Exception as e:
                logger.warning("phone_lexicon.db present mais erreur: %s", e)

    # 2. DB correcteur dans le dossier modeles
    if models_dir is not None:
        db = models_dir / "lexique_correcteur.db"
        if db.exists():
            try:
                return PhoneLexicon(db)
            except Exception as e:
                logger.warning("lexique_correcteur.db present mais erreur: %s", e)

    # 3. DB correcteur installee (lectura_correcteur)
    try:
        from lectura_correcteur import _DATA_DIR
        db = Path(_DATA_DIR) / "lexique_correcteur.db"
        if db.exists():
            return PhoneLexicon(db)
    except (ImportError, Exception):
        pass

    # 4. Fallback : phone_lexicon.db embarque dans le package
    db = _MODELES_DIR / "phone_lexicon.db"
    if db.exists():
        try:
            return PhoneLexicon(db)
        except Exception as e:
            logger.warning("phone_lexicon.db embarque mais erreur: %s", e)

    logger.info("Aucun phone_lexicon trouve — V6 fonctionnera sans lex_features")
    return None


def creer_engine(
    mode: str = "auto",
    models_dir: str | Path | None = None,
    lexicon_path: str | Path | None = None,
):
    """Factory pour creer un engine d'inference P2G.

    Parameters
    ----------
    mode : str
        "auto" (defaut) ou "local" : ONNX > NumPy > Pure
        "onnx", "numpy", "pure" : force un backend local specifique
        (le mode "api" n'existe plus : tout tourne en local)
    models_dir : str | Path | None
        Chemin vers le dossier contenant les modeles. Si None, cascade
        automatique : LECTURA_MODELS_DIR → ~/.lectura/models/p2g/ → site-packages.
        Poids absents : telechargement depuis Hugging Face au premier usage
        (desactive par LECTURA_HORS_LIGNE=1).
    lexicon_path : str | Path | None
        Chemin vers le fichier lexique POS (JSON).
        Si None, resolution automatique (cascade).
    """
    if mode == "api":
        raise ValueError(
            "Le mode 'api' a ete retire : lectura-graphemiseur tourne en local, avec des "
            "poids telecharges depuis Hugging Face au premier usage. Utilisez mode='auto'."
        )
    if mode not in ("auto", "local", "onnx", "numpy", "pure"):
        raise ValueError(f"Mode inconnu : {mode!r}")

    resolved_dir = _resoudre_modeles_dir(models_dir)

    # Mode local — essayer les backends dans l'ordre de preference
    # V7 (self-attention) > V6 (phone_lex_features) > V5 > V4 > V3
    # V7 : lex_select desactive (char-level P2G meilleur sans, cf metrics)
    base = resolved_dir
    force_version = os.environ.get("LECTURA_P2G_VERSION", "").lower()
    if force_version == "v6":
        # Usage interne : le modele V6 n'est pas publie (seul V7 l'est)
        model_onnx = base / "unifie_p2g_v6_int8.onnx"
        model_vocab = base / "unifie_p2g_v6_vocab.json"
        if not model_onnx.exists():
            raise FileNotFoundError(
                f"LECTURA_P2G_VERSION=v6 demande {model_onnx.name}, absent de {base} : "
                "le modele V6 n'est pas publie. Retirez la variable LECTURA_P2G_VERSION "
                "pour utiliser le modele V7 publie."
            )
    elif (base / "unifie_p2g_v7_int8.onnx").exists():
        model_onnx = base / "unifie_p2g_v7_int8.onnx"
        model_vocab = base / "unifie_p2g_v7_vocab.json"
    elif (base / "unifie_p2g_v6_int8.onnx").exists():
        model_onnx = base / "unifie_p2g_v6_int8.onnx"
        model_vocab = base / "unifie_p2g_v6_vocab.json"
    elif (base / "unifie_p2g_v5_int8.onnx").exists():
        model_onnx = base / "unifie_p2g_v5_int8.onnx"
        model_vocab = base / "unifie_p2g_v5_vocab.json"
    elif (base / "unifie_p2g_v4_int8.onnx").exists():
        model_onnx = base / "unifie_p2g_v4_int8.onnx"
        model_vocab = base / "unifie_p2g_v4_vocab.json"
    else:
        model_onnx = base / "unifie_p2g_v3_int8.onnx"
        model_vocab = base / "unifie_p2g_v3_vocab.json"
    resolved_lexicon = _resoudre_lexique(lexicon_path, resolved_dir)

    engine = None
    erreur_onnx: Exception | None = None

    if mode in ("auto", "local", "onnx"):
        try:
            from lectura_graphemiseur.inference_onnx_v2 import OnnxInferenceEngineV2
            # Resoudre phone_lexicon pour V6/V7 (phone_lex_features 28d)
            phone_lex = None
            if "v6" in str(model_onnx) or "v7" in str(model_onnx):
                phone_lex = _resoudre_phone_lexicon(resolved_dir)
            if isinstance(resolved_lexicon, dict):
                engine = OnnxInferenceEngineV2(
                    str(model_onnx), str(model_vocab),
                    lexicon=resolved_lexicon,
                    phone_lexicon=phone_lex,
                )
            else:
                engine = OnnxInferenceEngineV2(
                    str(model_onnx), str(model_vocab),
                    lexicon_path=str(resolved_lexicon) if resolved_lexicon else None,
                    phone_lexicon=phone_lex,
                )
            # V7 : lex_select actif avec seuil conservateur (0.95)
            # Le benchmark montre +444 mots nets sur le dev set
            if "v7" in str(model_onnx):
                engine.apply_lex_select = True
                logger.info("V7: lex_select enabled (threshold=%.2f)", engine.LEX_SELECT_THRESHOLD)
        except Exception as exc:  # repli NumPy / pur Python ; cause gardee pour l'erreur finale
            erreur_onnx = exc
            if mode == "onnx":
                raise

    if engine is None and mode in ("auto", "local", "numpy"):
        try:
            from lectura_graphemiseur.inference_numpy import NumpyInferenceEngine
            weights_dir = Path(__file__).parent / "modeles_numpy"
            if not weights_dir.exists():
                weights_dir = resolved_dir if resolved_dir else _MODELES_DIR
            engine = NumpyInferenceEngine(
                str(weights_dir), str(model_vocab),
                lexicon_path=str(resolved_lexicon) if isinstance(resolved_lexicon, Path) else None,
                lexicon=resolved_lexicon if isinstance(resolved_lexicon, dict) else None,
            )
        except (ImportError, FileNotFoundError, Exception):
            if mode == "numpy":
                raise

    if engine is None and mode in ("auto", "local", "pure"):
        try:
            from lectura_graphemiseur.inference_pure import PureInferenceEngine
            weights_dir = Path(__file__).parent / "modeles_numpy"
            if not weights_dir.exists():
                weights_dir = resolved_dir if resolved_dir else _MODELES_DIR
            engine = PureInferenceEngine(
                str(weights_dir), str(model_vocab),
                lexicon_path=str(resolved_lexicon) if isinstance(resolved_lexicon, Path) else None,
                lexicon=resolved_lexicon if isinstance(resolved_lexicon, dict) else None,
            )
        except (ImportError, FileNotFoundError, Exception):
            if mode == "pure":
                raise

    if engine is None:
        raise RuntimeError(
            f"Aucun backend d'inference disponible (mode={mode!r}). "
            "Verifiez que onnxruntime est installe (pip install onnxruntime) "
            f"et que les poids sont presents dans {base}."
        ) from erreur_onnx

    return engine


# API publique
from lectura_graphemiseur.tokeniseur import tokeniser_ipa, ipa_phrase_vers_chars
from lectura_graphemiseur.posttraitement import corriger_p2g, corriger_phrase_v2, corriger_phrase_v3
