"""Banque de diphones : recherche locale et téléchargement au premier usage.

La banque (``diphones.dpk.gz``) et les statistiques prosodiques
(``diphone_statistics.pkl``) ne sont pas dans la roue. Au premier usage, s'ils
manquent, ils sont téléchargés depuis Hugging Face, vérifiés par leur empreinte
SHA-256 (table en dur ci-dessous) et écrits de façon atomique (fichier
temporaire puis renommage). Les deux fichiers sont des adaptations des
enregistrements SIWIS : licence CC BY 4.0, voir le NOTICE du paquet.

Dossier de destination, dans l'ordre :
1. paramètre ``models_dir`` ;
2. ``$LECTURA_MODELS_DIR/tts_diphone`` ;
3. ``~/.lectura/models/tts_diphone``.

Variables d'environnement :
- ``LECTURA_HORS_LIGNE=1`` : aucun téléchargement ; erreur explicite si des
  fichiers manquent, avec la marche à suivre pour les poser à la main ;
- ``LECTURA_HF_BASE`` : URL de base à la place de https://huggingface.co
  (miroir, ou serveur local pour les tests).

Bibliothèque standard seulement (urllib, hashlib).
Module repris de lectura-phonemiseur et lectura-graphemiseur (même mécanisme) ;
seules les constantes et les textes d'information changent (licence CC BY 4.0).
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# ── Constantes du modèle (faciles à changer) ─────────────────────────────────

NOM_MODELE = "banque de diphones Lectura v0 (voix fr-f1, SIWIS)"
SOUS_DOSSIER = "tts_diphone"
DEPOT_HF = "lecturaworld/lectura-diphone-v0"
# Révision du dépôt : une branche, ou mieux un identifiant de commit une fois le
# dépôt publié. Les empreintes ci-dessous protègent de toute façon le contenu.
REVISION_HF = "23401a1d1d2f7acdc30cf58a2b3735ec4df8bc29"

# fichier -> (sha256, taille en octets)
FICHIERS: dict[str, tuple[str, int]] = {
    "diphones.dpk.gz": (
        "380aaf6d3e2735cec2eb9c6a6979364837bfc42d55ada54c7645ab850c6f4aa2", 18918278),
    "diphone_statistics.pkl": (
        "771bcfccbbf0fdd4e1440268a00b2f645204d0808dc404ff0cd61d771aa15fcc", 190916),
}

# Les fichiers sont téléchargés puis chargés avec pickle : seule l'empreinte
# SHA-256 ci-dessus garantit qu'ils sont bien ceux publiés par Lectura.

LICENCE_DONNEES = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
CREDIT_COURT = (
    "adaptation de « The SIWIS French Speech Synthesis Database » (Yamagishi et al., "
    "University of Edinburgh, 2017, https://doi.org/10.7488/ds/1705), voir NOTICE"
)
CONTACT = "https://www.lectura.world/contact/"
HF_BASE_DEFAUT = "https://huggingface.co"
_TIMEOUT = 60
_BLOC = 1 << 20


class LecturaModelesIndisponibles(RuntimeError):
    """La banque de diphones est absente et n'a pas pu être téléchargée."""


# ── Chemins ──────────────────────────────────────────────────────────────────

def dossier_destination(models_dir: str | Path | None = None) -> Path:
    """Dossier où chercher puis, au besoin, télécharger la banque."""
    if models_dir is not None:
        return Path(models_dir).expanduser()
    env = os.environ.get("LECTURA_MODELS_DIR")
    if env:
        return Path(env).expanduser() / SOUS_DOSSIER
    return Path.home() / ".lectura" / "models" / SOUS_DOSSIER


def fichiers_manquants(dossier: Path) -> list[str]:
    """Fichiers de ``FICHIERS`` absents de ``dossier``, ou de taille inattendue."""
    manquants = []
    for nom, (_, taille) in FICHIERS.items():
        p = dossier / nom
        try:
            if p.stat().st_size != taille:
                manquants.append(nom)
        except OSError:
            manquants.append(nom)
    return manquants


def url_fichier(nom: str) -> str:
    base = (os.environ.get("LECTURA_HF_BASE") or HF_BASE_DEFAUT).rstrip("/")
    return f"{base}/{DEPOT_HF}/resolve/{REVISION_HF}/{nom}"


def page_depot() -> str:
    base = (os.environ.get("LECTURA_HF_BASE") or HF_BASE_DEFAUT).rstrip("/")
    return f"{base}/{DEPOT_HF}"


def hors_ligne() -> bool:
    return os.environ.get("LECTURA_HORS_LIGNE", "").strip().lower() in ("1", "true", "oui", "yes")


def _taille_lisible(octets: int) -> str:
    return f"{octets / 1e6:.1f} Mo".replace(".", ",")


# ── Vérification ─────────────────────────────────────────────────────────────

def sha256_fichier(chemin: Path) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(_BLOC), b""):
            h.update(bloc)
    return h.hexdigest()


def verifier(dossier: str | Path | None = None) -> dict[str, bool]:
    """Recalcule l'empreinte SHA-256 de chaque fichier présent ; {nom: conforme}."""
    d = dossier_destination(dossier)
    return {nom: (d / nom).is_file() and sha256_fichier(d / nom) == sha
            for nom, (sha, _) in FICHIERS.items()}


# ── Téléchargement ───────────────────────────────────────────────────────────

def _message_manuel(dossier: Path, manquants: list[str]) -> str:
    liste = "\n".join(f"  - {n} (sha256 {FICHIERS[n][0]})" for n in manquants)
    return (
        f"Pour poser les fichiers à la main : téléchargez-les depuis {page_depot()} "
        f"(onglet « Files »), vérifiez leur empreinte (fichier SHA256SUMS du dépôt) et "
        f"placez-les dans {dossier} ; ou indiquez un autre dossier avec models_dir=... "
        f"ou la variable LECTURA_MODELS_DIR (sous-dossier {SOUS_DOSSIER}/).\n"
        f"Fichiers attendus :\n{liste}\n"
        f"To install them by hand: download them from {page_depot()}, check their SHA-256 "
        f"and put them in {dossier} (or pass models_dir=..., or set LECTURA_MODELS_DIR)."
    )


def _telecharger_un(nom: str, dossier: Path) -> None:
    sha_attendu, taille_attendue = FICHIERS[nom]
    url = url_fichier(nom)
    req = urllib.request.Request(url, headers={"User-Agent": "lectura-modeles"})
    fd, tmp = tempfile.mkstemp(prefix=f".{nom}.", suffix=".part", dir=dossier)
    try:
        h = hashlib.sha256()
        recu = 0
        with os.fdopen(fd, "wb") as sortie, urllib.request.urlopen(req, timeout=_TIMEOUT) as rep:
            for bloc in iter(lambda: rep.read(_BLOC), b""):
                sortie.write(bloc)
                h.update(bloc)
                recu += len(bloc)
                if recu > taille_attendue:
                    break
            sortie.flush()
            os.fsync(sortie.fileno())
        sha = h.hexdigest()
        if recu != taille_attendue or sha != sha_attendu:
            raise LecturaModelesIndisponibles(
                f"Fichier {nom} téléchargé depuis {url} refusé : empreinte ou taille inattendue "
                f"({recu} octets, sha256 {sha} ; attendu {taille_attendue} octets, sha256 {sha_attendu}). "
                f"Le fichier n'a pas été installé.\n"
                f"Downloaded file {nom} rejected: unexpected checksum or size; it was not installed."
            )
        os.chmod(tmp, 0o644)
        os.replace(tmp, dossier / nom)
        tmp = None
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass



def _signaler_telechargement() -> None:
    """Demande ``config.json`` au dépôt (requête HEAD, sans effet si elle échoue) : Hugging Face ne compte un
    téléchargement que sur ce fichier (https://huggingface.co/docs/hub/models-download-stats). Ajouté le 23/11/2026."""
    try:
        req = urllib.request.Request(url_fichier("config.json"), method="HEAD",
                                     headers={"User-Agent": "lectura-modeles"})
        urllib.request.urlopen(req, timeout=5).close()
    except Exception:  # noqa: BLE001 — un simple comptage ne doit jamais gêner l'installation
        pass

def assurer_modeles(models_dir: str | Path | None = None) -> Path:
    """Renvoie le dossier de la banque, après l'avoir téléchargée si elle manque.

    Lève ``LecturaModelesIndisponibles`` si des fichiers manquent et que le
    téléchargement est désactivé (``LECTURA_HORS_LIGNE=1``) ou échoue.
    """
    dossier = dossier_destination(models_dir)
    manquants = fichiers_manquants(dossier)
    if not manquants:
        return dossier

    if hors_ligne():
        raise LecturaModelesIndisponibles(
            f"Fichiers de la {NOM_MODELE} absents de {dossier} : {', '.join(manquants)}. "
            f"Le téléchargement est désactivé (LECTURA_HORS_LIGNE=1).\n"
            f"Diphone bank missing and download disabled (LECTURA_HORS_LIGNE=1).\n"
            + _message_manuel(dossier, manquants)
        )

    total = sum(FICHIERS[n][1] for n in manquants)
    print(
        f"[lectura] Téléchargement de la {NOM_MODELE} "
        f"({len(manquants)} fichier(s), {_taille_lisible(total)}) depuis {page_depot()} "
        f"vers {dossier}.\n"
        f"[lectura] Données sous licence {LICENCE_DONNEES} : {CREDIT_COURT}.\n"
        f"[lectura] Downloading the Lectura v0 diphone bank ({_taille_lisible(total)}); "
        f"data licensed CC BY 4.0, adapted from the SIWIS French Speech Synthesis Database (see NOTICE).",
        file=sys.stderr, flush=True,
    )
    try:
        dossier.mkdir(parents=True, exist_ok=True)
        for nom in manquants:
            _telecharger_un(nom, dossier)
        _signaler_telechargement()
    except LecturaModelesIndisponibles:
        raise
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise LecturaModelesIndisponibles(
            f"Échec du téléchargement de la {NOM_MODELE} vers {dossier} : {exc}\n"
            f"Download of the diphone bank failed.\n" + _message_manuel(dossier, manquants)
        ) from exc
    print(f"[lectura] Banque installée et vérifiée (SHA-256) dans {dossier}.", file=sys.stderr, flush=True)
    return dossier
