# lectura-diphone

Génération v0 (2026) de Lectura — https://www.lectura.world/developpement/modules/v0/

Synthèse vocale du français par concaténation de diphones dans le domaine du vocodeur WORLD. Tout tourne **en local** : aucune API, aucune clé.

*French diphone text-to-speech (WORLD vocoder), fully local. English summary at the end.*

## Installation

```bash
pip install lectura-diphone            # synthèse depuis des phonèmes (pyworld, numpy, scipy)
pip install "lectura-diphone[g2p]"     # + texte -> phonèmes (lectura-g2p, en local)
pip install "lectura-diphone[vc]"      # + changement de timbre par IA (lectura-vc-zeroshot)
```

## Banque de diphones : téléchargement au premier usage

La banque (`diphones.dpk.gz`, 18,9 Mo, et `diphone_statistics.pkl`, 0,2 Mo) n'est pas dans le paquet. Au premier usage, elle est téléchargée depuis Hugging Face ([lecturaworld/lectura-diphone-v0](https://huggingface.co/lecturaworld/lectura-diphone-v0)) et vérifiée par son empreinte SHA-256, inscrite dans le code.

Dossier, dans l'ordre : paramètre `models_dir`, puis `$LECTURA_MODELS_DIR/tts_diphone/`, puis `~/.lectura/models/tts_diphone/`.

| Variable | Effet |
|---|---|
| `LECTURA_HORS_LIGNE=1` | aucun téléchargement ; erreur claire si la banque manque |
| `LECTURA_HF_BASE` | autre adresse que https://huggingface.co (miroir, tests) |
| `LECTURA_MODELS_DIR` | autre dossier racine des modèles |

Pour poser la banque à la main : télécharger les fichiers du dépôt, vérifier `SHA256SUMS`, les placer dans l'un des dossiers ci-dessus. `lectura-diphone --telecharger` télécharge, `lectura-diphone --verifier` contrôle les empreintes.

## Utilisation

### En ligne de commande

```bash
lectura-diphone --phonemes "b ɔ̃ ʒ u ʁ | l ə m ɔ̃ d" -o bonjour.wav   # sans G2P
lectura-diphone "Bonjour le monde." -o bonjour.wav                    # demande [g2p]
lectura-diphone "Bonjour le monde." --mode SYLLABES --vitesse 1.2 -o lent.wav
lectura-diphone "Bonjour." --prosodie corpus --f0 120 --seed 1 -o grave.wav
lectura-diphone "Bonjour." --voix melange-f -o b.wav                  # demande [vc] ; filigrane IA
```

Options : `-o/--sortie` (défaut `sortie.wav`), `--phonemes` (groupes séparés par `|`), `--mode`
(`FLUIDE`, `MOT_A_MOT`, `SYLLABES`), `--vitesse` (= `duration_scale`), `--prosodie` (`regles`, `corpus`),
`--f0` (= `base_f0`, Hz), `--seed`, `--voix` (`fr-f1`, `melange-f`, `melange-h`), `--models-dir`,
`--telecharger`, `--verifier`, `--version`. Les autres réglages (`voix_variante`, `pause_scale`, `timbre`…) ne
sont accessibles que depuis Python.

### Depuis des phonèmes (API Python)

```python
from lectura_diphone import creer_engine

engine = creer_engine()
resultat = engine.synthesize_groups([
    {"phones": ["b", "ɔ̃", "ʒ", "u", "ʁ"], "boundary": "comma"},
    {"phones": ["l", "ə", "m", "ɔ̃", "d"], "boundary": "period"},
], seed=1)
resultat.samples          # numpy float32, 44 100 Hz
resultat.vers_wav("bonjour.wav")   # WAV 16 bits avec métadonnées IA
```

### Depuis du texte (`[g2p]`)

```python
from lectura_diphone import synthetiser

resultat = synthetiser("Bonjour le monde.", synth_mode="FLUIDE", seed=1)
resultat.vers_wav("bonjour.wav")
```

`g2p=` accepte aussi votre propre fonction `texte -> [{"phones": [...], "boundary": "period"}, ...]`.

`synthetiser()` crée le moteur (chargement de la banque, du G2P et, avec `voix=`, du moteur de conversion) au premier
appel, puis le garde en mémoire pour les appels suivants avec les mêmes `models_dir` et `g2p` : seul le premier appel
paie le chargement, et le son est identique à celui d'un moteur neuf (même graine, même son). `vider_cache()` libère
cette mémoire. Pour tout contrôler, créez le moteur vous-même avec `creer_engine()` (il renvoie un `DiphoneEngine`).

Un groupe peut porter, en plus de `phones` et `boundary`, la clé facultative `word_boundaries` : indices des phonèmes
qui commencent un mot (le premier mot est implicite), utilisés pour le découpage prosodique et les timings.
Un phonème absent de la banque est remplacé par un silence et signalé par un avertissement (`UserWarning`).

**`mode` a deux sens.** Dans `creer_engine()` et `synthetiser()`, `mode` vaut `"auto"` ou `"local"` (identiques :
tout tourne en local). Le mode de synthèse (`FLUIDE`, `MOT_A_MOT`, `SYLLABES`) s'appelle `synth_mode` dans
`synthetiser()` et `mode` dans `DiphoneEngine.synthesize_groups()` (valeur texte ou `SynthMode`).

## Réglages

| Paramètre | Défaut | Effet |
|---|---|---|
| `synth_mode` (`synthetiser`) / `mode` (`synthesize_groups`) | `"FLUIDE"` | `FLUIDE`, `MOT_A_MOT`, `SYLLABES` |
| `duration_scale` | 1.0 | vitesse (>1 = plus lent) |
| `pause_scale` | 1.0 | durée des pauses entre groupes |
| `sentence_pause_ms` | 400 | pause entre phrases (ms) |
| `macro_expressivity` | 1.0 | gestes prosodiques (0 = plat, 2 = exagéré) |
| `micro_expressivity` | 1.0 | micro-variations (mode FLUIDE) |
| `prosody_style` | `"regles"` | `"regles"` (LHiLH*) ou `"corpus"` (contours tirés du corpus SIWIS) |
| `base_f0` | 175.0 | hauteur de base (Hz) |
| `ap_cleanup` | 1.5 | compression de l'apériodicité (1.0 = sans) ; **maximum 3.0** : au-delà, ramené à 3.0 avec un avertissement |
| `formant_sharpening` | 1.3 | affûtage des formants (1.0 = sans) ; **maximum 2.0** : au-delà, ramené à 2.0 avec un avertissement |
| `vtln_alpha` | 1.0 | déformation du conduit vocal (0.8 = grave, 1.2 = aigu) |
| `spectral_contrast` | 1.3 | contraste spectral (1.0 = sans, 2.0 = fort) ; `synthesize_groups` seulement |
| `prosody` | `None` | `synthesize_groups` seulement : dict facultatif `{"f0_hz": ..., "duration_scale": ...}` appliqué à chaque groupe (F0 moyenne visée, facteur de durée) |
| `timbre` | `None` | signature paramétrique `"neutre"`, `"homme"`, `"enfant"`, ou chemin d'un fichier `.json` (liste de coefficients cepstraux, ou `{"signature": [...]}`) ; traitement du signal |
| `seed` | `None` | graine : même graine, même son |
| `voix` | `None` | changement de timbre par IA, voir ci-dessous |
| `voix_variante` | 0.0 | curseur grave/aigu du changement de timbre (-1 à +1) |
| `voix_tau` | `None` | **sans effet** (le modèle ONNX d'OpenVoice n'a pas cette entrée) ; gardé pour compatibilité, `DeprecationWarning` s'il est donné |

## Changement de timbre par IA (`[vc]`)

`voix=` convertit le son avec OpenVoice, par l'intermédiaire de [lectura-vc-zeroshot](https://pypi.org/project/lectura-vc-zeroshot/). Timbres possibles, identifiés de façon neutre :

| Identifiant | Timbre |
|---|---|
| `fr-f1` | voix native de la banque (corpus SIWIS) |
| `melange-f` | voix féminine mélangée (vecteur de timbre moyen de plusieurs voix) |
| `melange-h` | voix masculine mélangée (idem) |

Les trois timbres sont des préréglages de lectura-vc-zeroshot (>= 1.3), qui télécharge leurs vecteurs au premier usage (licences et crédits dans son NOTICE, tous sous CC BY 4.0). Le curseur `voix_variante` (-1 = plus grave, « homme » ; +1 = plus aigu, « enfant » ; 0 = timbre tel quel) est transmis à lectura-vc-zeroshot, qui l'applique par des vecteurs précalculés et interpolés. Aucun fichier audio de référence n'est accepté : ce paquet ne sert pas à imiter une personne.
Après la conversion (rendue à 22 050 Hz), `samples` est rééchantillonné à 44 100 Hz et renormalisé (crête 0,9).

**Filigrane.** Un son passé par le changement de timbre est produit par un réseau de neurones : il porte un filigrane IA inaudible (AudioSeal), posé par lectura-vc-zeroshot, et le paquet refuse une version de lectura-vc-zeroshot qui ne le garantit pas.

## Marquage IA : ne retirez pas le marquage

Les WAV écrits par `vers_wav`, `ecrire_wav` ou la ligne de commande portent des métadonnées lisibles par machine : type de source IPTC `trainedAlgorithmicMedia` (XMP) et chunk `LIST/INFO`, avec le texte :

> Voix de synthèse générée par Lectura (synthèse par diphones, à partir d'enregistrements SIWIS sous CC BY 4.0). Contenu audio généré par ordinateur.

Après un changement de timbre, le son porte en plus le filigrane IA. **Ne retirez pas le marquage** (métadonnées et filigrane) : il sert à informer les auditeurs que la voix est générée (règlement européen sur l'IA, art. 50). Si vous diffusez un son produit avec ce paquet, gardez les métadonnées ou indiquez qu'il s'agit d'une voix de synthèse.

`resultat.samples` est un tableau brut, sans métadonnées : pour un fichier diffusé, passez par `vers_wav` ou `ecrire_wav`.

## Fonctionnement

```
Texte → [G2P] → phonèmes IPA → chaîne de diphones
      → paramètres WORLD (F0, enveloppe spectrale, apériodicité)
      → étirement + concaténation → prosodie (F0, durées)
      → réglages du timbre → synthèse WORLD → audio 44 100 Hz
      → [facultatif] changement de timbre (OpenVoice, filigrane IA)
```

La synthèse elle-même est faite par règles et traitement du signal, sans réseau de neurones. Les diphones sont des paramètres WORLD tirés du corpus SIWIS, moyennés par type de transition.

## Licences

- **Code** : AGPL-3.0-or-later (`LICENCE.txt`), ou licence commerciale Lectura. Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Les données SIWIS restent sous CC BY 4.0 (ci-dessous). Autres cas sur demande : https://www.lectura.world/contact/
- **Banque de diphones, statistiques prosodiques et `_group_contour_bank.json`** : adaptations de « The SIWIS French Speech Synthesis Database » (J. Yamagishi, P.-E. Honnet, P. N. Garner, A. Lazaridis, University of Edinburgh, CSTR, 2017, https://doi.org/10.7488/ds/1705), sous **CC BY 4.0** (`LICENCE-DONNEES.txt`). Crédit complet et modifications : `NOTICE`. Cette réutilisation n'implique aucune approbation des auteurs ni de la personne enregistrée.
- Contributions de code : non acceptées (voir `CONTRIBUTING.md` du dépôt) ; les rapports de bogues sont bienvenus.

Copyright (C) 2025-2026 Maxime Carrière (Lectura).

---

## English summary

French diphone TTS using the WORLD vocoder; runs fully locally (no API, no key). The diphone bank (19 MB, CC BY 4.0, adapted from the SIWIS French Speech Synthesis Database) is downloaded from Hugging Face on first use and checked against a SHA-256 hash (`LECTURA_HORS_LIGNE=1` disables downloads; `LECTURA_HF_BASE` overrides the host). WAV files carry AI-disclosure metadata (IPTC `trainedAlgorithmicMedia`); optional timbre conversion (`voix="fr-f1" | "melange-f" | "melange-h"`, via lectura-vc-zeroshot/OpenVoice) also adds an AudioSeal AI watermark. **Do not remove the AI marking.** Code: AGPL-3.0-or-later or Lectura commercial licence (any valid Lectura v1 commercial licence also permits commercial use of v0, code and weights, within the same scope and term; other cases on request); data: CC BY 4.0 (see NOTICE).
