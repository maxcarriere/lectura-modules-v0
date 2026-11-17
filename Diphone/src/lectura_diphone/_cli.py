"""Interface en ligne de commande : ``lectura-diphone`` (ou ``python -m lectura_diphone``).

Exemples ::

    lectura-diphone "Bonjour le monde." -o bonjour.wav        # demande [g2p]
    lectura-diphone --phonemes "b ɔ̃ ʒ u ʁ | l ə m ɔ̃ d" -o b.wav  # sans G2P
    lectura-diphone "Bonjour." --voix melange-f -o b.wav       # demande [vc] ; filigrane IA
    lectura-diphone --telecharger                              # banque seule
    lectura-diphone --verifier                                 # empreintes SHA-256

Les WAV écrits portent les métadonnées IA (IPTC trainedAlgorithmicMedia).
Ne retirez pas le marquage.
"""

from __future__ import annotations

import argparse
import sys


def _groupes_depuis_phonemes(chaine: str) -> list[dict]:
    """"b ɔ̃ ʒ u ʁ | l ə m ɔ̃ d" -> groupes ; "|" sépare les groupes, le dernier finit en point."""
    morceaux = [m.split() for m in chaine.split("|")]
    morceaux = [m for m in morceaux if m]
    return [
        {"phones": m, "boundary": "period" if i == len(morceaux) - 1 else "comma"}
        for i, m in enumerate(morceaux)
    ]


def main(argv: list[str] | None = None) -> int:
    from lectura_diphone import TIMBRES_VOIX, __version__

    p = argparse.ArgumentParser(
        prog="lectura-diphone",
        description="Synthèse vocale du français par diphones (Lectura). "
                    "Les WAV portent les métadonnées IA ; ne retirez pas le marquage.",
    )
    p.add_argument("texte", nargs="?", help="texte français (demande un G2P : pip install \"lectura-diphone[g2p]\")")
    p.add_argument("-o", "--sortie", default="sortie.wav", help="fichier WAV de sortie (défaut : sortie.wav)")
    p.add_argument("--phonemes", help="phonèmes IPA séparés par des espaces, groupes séparés par « | » (sans G2P)")
    p.add_argument("--mode", default="FLUIDE", choices=["FLUIDE", "MOT_A_MOT", "SYLLABES"])
    p.add_argument("--vitesse", type=float, default=1.0, help="facteur de durée (>1 = plus lent)")
    p.add_argument("--prosodie", default="regles", choices=["regles", "corpus"])
    p.add_argument("--f0", type=float, default=175.0, help="hauteur de base en Hz")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--voix", choices=list(TIMBRES_VOIX), default=None,
                   help="changement de timbre par IA (lectura-vc-zeroshot, filigrane IA)")
    p.add_argument("--models-dir", default=None, help="dossier de la banque de diphones")
    p.add_argument("--telecharger", action="store_true", help="télécharger la banque puis quitter")
    p.add_argument("--verifier", action="store_true", help="vérifier les empreintes SHA-256 de la banque puis quitter")
    p.add_argument("--version", action="version", version=f"lectura-diphone {__version__}")
    a = p.parse_args(argv)

    from lectura_diphone import _poids

    if a.verifier:
        res = _poids.verifier(a.models_dir)
        for nom, ok in res.items():
            print(f"{'OK ' if ok else 'ÉCHEC'} {nom}")
        return 0 if all(res.values()) else 1
    if a.telecharger:
        print(_poids.assurer_modeles(a.models_dir))
        return 0
    if not a.texte and not a.phonemes:
        p.error("donnez un texte ou --phonemes")

    from lectura_diphone import creer_engine

    engine = creer_engine(models_dir=a.models_dir)
    if a.phonemes:
        groupes = _groupes_depuis_phonemes(a.phonemes)
    else:
        if engine._g2p_backend is None:
            from lectura_diphone.g2p import auto_detect_g2p
            auto_detect_g2p()
        groupes = engine._g2p_backend.phonemize(a.texte)
    res = engine.synthesize_groups(
        groupes, mode=a.mode, duration_scale=a.vitesse, prosody_style=a.prosodie,
        base_f0=a.f0, seed=a.seed, voix=a.voix,
    )
    res.vers_wav(a.sortie)
    duree = len(res.samples) / res.sample_rate if res.sample_rate else 0.0
    print(f"{a.sortie} : {duree:.2f} s, {res.sample_rate} Hz"
          + (" (timbre converti, filigrane IA)" if res.timbre_converti else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
