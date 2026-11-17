"""Modèles et préréglages : recherche locale et téléchargement au premier usage.

Rien de lourd n'est dans la roue. Au premier usage, les fichiers qui manquent
sont téléchargés depuis Hugging Face (dépôt ``DEPOT_HF``), vérifiés par leur
empreinte SHA-256 (table en dur ci-dessous) et écrits de façon atomique
(fichier temporaire puis renommage).

Contenu du dépôt (licences : voir le NOTICE du paquet et celui du dépôt) :
- ``openvoice/openvoice_se.onnx``, ``openvoice/openvoice_vc.onnx`` : conversions
  ONNX des points de contrôle OpenVoice V2 (MyShell.ai, MIT) ;
- ``filigrane/generator_base.onnx``, ``filigrane/detector_base.onnx`` :
  conversions ONNX du générateur et du détecteur AudioSeal (Meta, MIT) ;
- ``presets/<identifiant>-variantes.npy`` : les 9 vecteurs du curseur grave/aigu
  de chaque préréglage (variante -1 à +1) ;
- ``presets/<identifiant>.npy`` : vecteurs de timbre des préréglages
  ``fr-f1``, ``melange-f`` et ``melange-h`` (CC BY 4.0).

Dossiers de destination :
- OpenVoice et préréglages : paramètre ``models_dir``, sinon
  ``$LECTURA_MODELS_DIR/vc``, sinon ``~/.lectura/models/vc`` (préréglages dans
  le sous-dossier ``presets/``) ;
- filigrane : ``~/.lectura/models/filigrane`` (voir ``_filigrane``).

Variables d'environnement :
- ``LECTURA_HORS_LIGNE=1`` : aucun téléchargement ; erreur explicite si des
  fichiers manquent, avec la marche à suivre pour les poser à la main ;
- ``LECTURA_HF_BASE`` : URL de base à la place de https://huggingface.co
  (miroir, ou serveur local pour les tests).

Bibliothèque standard seulement (urllib, hashlib). Même mécanisme que
lectura-phonemiseur, lectura-graphemiseur et lectura-diphone ; la table couvre
ici plusieurs dossiers de destination.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# ── Constantes (faciles à changer) ───────────────────────────────────────────

SOUS_DOSSIER = "vc"
DEPOT_HF = "lecturaworld/lectura-vc-zeroshot-v0"
# Révision du dépôt : une branche, ou mieux un identifiant de commit une fois le
# dépôt publié. Les empreintes ci-dessous protègent de toute façon le contenu.
REVISION_HF = "8b61248e1040259354ed0c08efc311eceabf19fc"

# chemin dans le dépôt -> (sha256, taille en octets)
FICHIERS: dict[str, tuple[str, int]] = {
    "openvoice/openvoice_se.onnx": (
        "9fc554414dc9bcd435df2526a0b08b6c39f82ce9d90cfba1d191d55fab836462", 3258294),
    "openvoice/openvoice_vc.onnx": (
        "2ee95d03c35460898d1b6175fc2b07f81715a03e9c5a1e31da7c8f9f1e5d2c61", 128052461),
    "filigrane/generator_base.onnx": (
        "8a9ea5fce3041482cf1268c2c6bfccfd451691bece3f440458c0426faa5fc778", 58910855),
    "filigrane/detector_base.onnx": (
        "c540086101be6ca8a8cfd1049e9d7575087c9e769ecbba1c348429f52b32c383", 34723894),
    "presets/fr-f1.npy": (
        "28fa0fba4ba20e08be40ba873f27d585bb2c3d49af036e2c1eb659aee2fbe61a", 1152),
    "presets/melange-f.npy": (
        "1bc44d552079b30b0c236a12a41f81a80d483c96096b31a3b3062875f4a7d185", 1152),
    "presets/melange-h.npy": (
        "5ad13fa6c17e65a5519dec8bcbb1054fce970821264210e6c494f49e337bcff5", 1152),
    # curseur grave/aigu : 9 vecteurs par préréglage (variante -1, -0,75, ..., +1)
    "presets/fr-f1-variantes.npy": (
        "0d0214f6b1313352b77d4d7b7deafe44e70529ed02bfee46a709a28ac72ef366", 9344),
    "presets/melange-f-variantes.npy": (
        "557eab930c35161047326cb733c832b26397e3be5e4ff5b687e5d1b108c20b35", 9344),
    "presets/melange-h-variantes.npy": (
        "925432eb5bc4f15914747acc3421cd38af8c367e183bebe36ae52a8e75c44a57", 9344),
}

OPENVOICE = ("openvoice/openvoice_se.onnx", "openvoice/openvoice_vc.onnx")

# Licence de chaque groupe de fichiers (affichée au téléchargement)
LICENCES = {
    "openvoice": "MIT (OpenVoice, MyShell.ai)",
    "filigrane": "MIT (AudioSeal, Meta)",
    "presets/fr-f1.npy": "CC BY 4.0 (corpus SIWIS, voir NOTICE)",
    "presets/fr-f1-variantes.npy": "CC BY 4.0 (corpus SIWIS, voir NOTICE)",
    "presets/melange-f-variantes.npy": "CC BY 4.0 (voix mélangée, dont SIWIS ; voir NOTICE)",
    "presets/melange-h-variantes.npy": "CC BY 4.0 (voix mélangée ; voir NOTICE)",
    "presets/melange-f.npy": "CC BY 4.0 (voix mélangée, dont SIWIS ; voir NOTICE)",
    "presets/melange-h.npy": "CC BY 4.0 (voix mélangée ; voir NOTICE)",
}

CONTACT = "https://www.lectura.world/contact/"
HF_BASE_DEFAUT = "https://huggingface.co"
_TIMEOUT = 60
_BLOC = 1 << 20


class LecturaModelesIndisponibles(RuntimeError):
    """Des fichiers manquent et n'ont pas pu être téléchargés."""


# ── Chemins ──────────────────────────────────────────────────────────────────

def dossier_destination(models_dir: str | Path | None = None) -> Path:
    """Dossier des modèles OpenVoice (préréglages dans ``presets/``)."""
    if models_dir is not None:
        return Path(models_dir).expanduser()
    env = os.environ.get("LECTURA_MODELS_DIR")
    if env:
        return Path(env).expanduser() / SOUS_DOSSIER
    return Path.home() / ".lectura" / "models" / SOUS_DOSSIER


def nom_local(chemin_depot: str) -> str:
    return chemin_depot.rsplit("/", 1)[-1]


def fichiers_manquants(dossier: Path, chemins: tuple[str, ...] | list[str]) -> list[str]:
    """Chemins du dépôt dont le fichier local est absent ou de taille inattendue."""
    manquants = []
    for c in chemins:
        p = Path(dossier) / nom_local(c)
        try:
            if p.stat().st_size != FICHIERS[c][1]:
                manquants.append(c)
        except OSError:
            manquants.append(c)
    return manquants


def url_fichier(chemin_depot: str) -> str:
    base = (os.environ.get("LECTURA_HF_BASE") or HF_BASE_DEFAUT).rstrip("/")
    return f"{base}/{DEPOT_HF}/resolve/{REVISION_HF}/{chemin_depot}"


def page_depot() -> str:
    base = (os.environ.get("LECTURA_HF_BASE") or HF_BASE_DEFAUT).rstrip("/")
    return f"{base}/{DEPOT_HF}"


def hors_ligne() -> bool:
    return os.environ.get("LECTURA_HORS_LIGNE", "").strip().lower() in ("1", "true", "oui", "yes")


def _taille_lisible(octets: int) -> str:
    return f"{octets / 1e6:.1f} Mo".replace(".", ",")


def _licence(chemin_depot: str) -> str:
    return LICENCES.get(chemin_depot) or LICENCES.get(chemin_depot.split("/", 1)[0], "voir NOTICE")


# ── Vérification ─────────────────────────────────────────────────────────────

def sha256_fichier(chemin: Path) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(_BLOC), b""):
            h.update(bloc)
    return h.hexdigest()


def verifier(dossier: str | Path, chemins: tuple[str, ...] | list[str]) -> dict[str, bool]:
    """Recalcule l'empreinte SHA-256 de chaque fichier ; {chemin du dépôt: conforme}."""
    d = Path(dossier)
    return {c: (d / nom_local(c)).is_file() and sha256_fichier(d / nom_local(c)) == FICHIERS[c][0]
            for c in chemins}


# ── Téléchargement ───────────────────────────────────────────────────────────

def _message_manuel(dossier: Path, manquants: list[str]) -> str:
    liste = "\n".join(f"  - {c} (sha256 {FICHIERS[c][0]})" for c in manquants)
    return (
        f"Pour poser les fichiers à la main : téléchargez-les depuis {page_depot()} "
        f"(onglet « Files »), vérifiez leur empreinte (fichier SHA256SUMS du dépôt) et "
        f"placez-les, sans leur sous-dossier, dans {dossier}.\n"
        f"Fichiers attendus :\n{liste}\n"
        f"To install them by hand: download them from {page_depot()}, check their SHA-256 "
        f"and put them in {dossier}."
    )


def _telecharger_un(chemin_depot: str, dossier: Path) -> None:
    sha_attendu, taille_attendue = FICHIERS[chemin_depot]
    nom = nom_local(chemin_depot)
    url = url_fichier(chemin_depot)
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
                f"Fichier {chemin_depot} téléchargé depuis {url} refusé : empreinte ou taille inattendue "
                f"({recu} octets, sha256 {sha} ; attendu {taille_attendue} octets, sha256 {sha_attendu}). "
                f"Le fichier n'a pas été installé.\n"
                f"Downloaded file {chemin_depot} rejected: unexpected checksum or size; it was not installed."
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


def assurer(chemins: tuple[str, ...] | list[str], dossier: str | Path) -> Path:
    """Renvoie ``dossier`` après y avoir téléchargé les fichiers manquants de ``chemins``.

    Lève ``LecturaModelesIndisponibles`` si des fichiers manquent et que le
    téléchargement est désactivé (``LECTURA_HORS_LIGNE=1``) ou échoue.
    """
    dossier = Path(dossier)
    manquants = fichiers_manquants(dossier, chemins)
    if not manquants:
        return dossier

    if hors_ligne():
        raise LecturaModelesIndisponibles(
            f"Fichiers de lectura-vc-zeroshot absents de {dossier} : "
            f"{', '.join(nom_local(c) for c in manquants)}. "
            f"Le téléchargement est désactivé (LECTURA_HORS_LIGNE=1).\n"
            f"Files missing and download disabled (LECTURA_HORS_LIGNE=1).\n"
            + _message_manuel(dossier, manquants)
        )

    total = sum(FICHIERS[c][1] for c in manquants)
    licences = " ; ".join(sorted({f"{nom_local(c)} : {_licence(c)}" for c in manquants}))
    print(
        f"[lectura] Téléchargement de {len(manquants)} fichier(s) de lectura-vc-zeroshot "
        f"({_taille_lisible(total)}) depuis {page_depot()} vers {dossier}.\n"
        f"[lectura] Licences : {licences}.\n"
        f"[lectura] Downloading {len(manquants)} lectura-vc-zeroshot file(s) "
        f"({_taille_lisible(total)}); licences listed in NOTICE.",
        file=sys.stderr, flush=True,
    )
    try:
        dossier.mkdir(parents=True, exist_ok=True)
        for c in manquants:
            _telecharger_un(c, dossier)
    except LecturaModelesIndisponibles:
        raise
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise LecturaModelesIndisponibles(
            f"Échec du téléchargement vers {dossier} : {exc}\n"
            f"Download failed.\n" + _message_manuel(dossier, manquants)
        ) from exc
    print(f"[lectura] Fichiers installés et vérifiés (SHA-256) dans {dossier}.", file=sys.stderr, flush=True)
    return dossier


def assurer_openvoice(models_dir: str | Path | None = None) -> Path:
    """Dossier des modèles OpenVoice, après téléchargement s'ils manquent."""
    return assurer(OPENVOICE, dossier_destination(models_dir))
