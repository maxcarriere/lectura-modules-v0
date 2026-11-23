# Lectura Phonemiseur

Génération v0 (2026) de Lectura — https://www.lectura.world/developpement/modules/v0/

**Phonemiseur neural du francais : G2P + POS + Morphologie + Liaison + Groupes de lecture**

Un seul modele BiLSTM char-level multi-tete (1.75M parametres, ONNX INT8 = 1.8 Mo) qui predit simultanement :

- **G2P** : transcription phonemique IPA
- **POS** : etiquetage morpho-syntaxique — 19 tags
- **Morphologie** : genre, nombre, temps, mode, personne, forme verbale
- **Liaison** : prediction des liaisons obligatoires/facultatives

Résultats mesurés : voir « Résultats ».

Tout tourne en local (ONNX Runtime). Les poids du modele (1,9 Mo) sont telecharges depuis Hugging Face au premier usage,
puis gardes sur le disque ; ils sont sous licence CC BY-NC 4.0 (voir « Modèles »).

## Demarrage rapide

### Installation

```bash
pip install lectura-phonemiseur             # installe aussi onnxruntime et numpy
pip install "lectura-phonemiseur[formules]" # + lectura-formules, requis par analyser_phrase_complete()
```

`analyser_phrase_complete()` (phrase tokenisee avec formules : nombres, dates, heures...) a besoin de
`lectura-formules` ; sans lui, l'importer leve une `ImportError` qui indique quoi installer. Le reste du paquet
fonctionne sans.

### Utilisation rapide (factory — recommande)

```python
from lectura_phonemiseur import creer_engine
from lectura_phonemiseur.tokeniseur import tokeniser

# Premier appel : telecharge les poids (1,9 Mo) dans ~/.lectura/models/g2p/
engine = creer_engine()

tokens = tokeniser("Les enfants sont arrivés à la maison.")
result = engine.analyser(tokens)

# Une case par token, ponctuation comprise (le point final : phonétique vide)
print(result["g2p"])      # ['le', 'ɑ̃fɑ̃', 'sɔ̃', 'aʁive', 'a', 'la', 'mɛzɔ̃', '']
print(result["pos"])      # ['ART:def', 'NOM', 'AUX', 'VER', 'PRE', 'ART:def', 'NOM', 'NOM']
print(result["liaison"])  # ['Lz', 'none', 'Lt', 'none', 'none', 'none', 'none', 'none']
print(result["morpho"])   # {'Number': ['Plur', ...], 'Gender': [...], ...}
```

Modes disponibles : `"auto"` (defaut, ONNX Runtime), `"local"`, `"onnx"`.

### Separateurs pour mots composes

Le modele V4 gere nativement les separateurs dans les mots composes et les elisions :

```python
# IPA plat (defaut)
result = engine.analyser(["abat-jour", "d'abord"])
print(result["g2p"])  # ['abaʒuʁ', 'dabɔʁ']

# Avec separateurs preserves
result = engine.analyser(["abat-jour", "d'abord"], sep_hyphen=True, sep_apos=True)
print(result["g2p"])  # ['aba-ʒuʁ', "d'abɔʁ"]
```

## Modèles

Tout tourne en local : aucun appel à un serveur Lectura.

Les poids du modèle G2P ne sont pas dans la roue. Au premier usage, `creer_engine()` les télécharge depuis
Hugging Face ([lecturaworld/lectura-g2p-v0](https://huggingface.co/lecturaworld/lectura-g2p-v0), 2 fichiers, 1,9 Mo), vérifie leur empreinte SHA-256
et les range dans `~/.lectura/models/g2p/` (ou `$LECTURA_MODELS_DIR/g2p/`, ou le dossier passé en `models_dir=`).
Les usages suivants n'utilisent plus le réseau.

- `LECTURA_HORS_LIGNE=1` : aucun téléchargement ; si des poids manquent, l'erreur `LecturaModelesIndisponibles`
  indique les fichiers attendus, leurs empreintes et comment les poser à la main.
- Poids sous licence CC BY-NC 4.0 (usage non commercial) ; usage commercial inclus dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande : https://www.lectura.world/contact/

## Backend d'inference

Le paquet tourne avec ONNX Runtime (environ 2 ms par phrase sur CPU). Les modes `"numpy"` et `"pure"` ne sont pas
disponibles dans la v0 : ils demandent des poids convertis (JSON) qui ne sont pas publiés, et
`creer_engine(mode="numpy")` ou `mode="pure"` lève une `ValueError` explicite.

## Features lexicales (optionnel)

Le modele accepte en entree optionnelle un vecteur de 24 dimensions par mot, construit a partir d'un lexique de candidats POS. Cela ameliore la prediction POS, la morphologie et les liaisons.

Aucun lexique n'est chargé par défaut. On peut en fournir un par le paramètre `lexicon_path` de `creer_engine()` : un fichier JSON `{"mot": ["NOM", "VER", ...]}` (catégories de Lexique 3). Mesure du 23/11/2026 sur le test du corpus du cadre (2 409 phrases) : l'aide est négligeable (+0,01 point de mots justes, +0,2 point sur les homographes). Sans lexique, le modèle fonctionne normalement (features = zéros).

```python
# Sans lexique (défaut)
engine = creer_engine()

# Avec un lexique de catégories fourni
engine = creer_engine(lexicon_path="lexique_pos.json")

# Desactiver les features lexicales
result = engine.analyser(tokens, use_lex=False)
```

## Corrections G2P

Le post-traitement applique automatiquement trois niveaux de corrections :

1. **Homographes POS-aware** (~2 700 entrees) : desambiguisation par categorie grammaticale (ex: jean/NOM = dʒin vs Jean/NOM PROPRE = ʒɑ̃)
2. **Table plate** (~3 500 corrections manuelles) : corrections ciblees avec separateurs
3. **Corrections lexique** (optionnel, ~30 000 entrees) : corrections etendues depuis le lexique de reference

```python
# Activer les corrections etendues du lexique
engine = creer_engine(corrections_lexique=True)
```

Les corrections stockent les separateurs dans l'IPA (`aba-ʒuʁ`, `d'abɔʁ`). Quand `sep_hyphen`/`sep_apos` ne sont pas actifs, les separateurs sont automatiquement retires.

## Résultats

Bancs du 23/11/2026, en local sur CPU, avec le code publié (`lectura-g2p` 4.1.5, `lectura-phonemiseur` 4.1.11) et
les poids publiés. Le même banc sert pour toutes les générations de Lectura ; espeak-ng 1.51 est mesuré comme point de
comparaison extérieur.

**1. Lecture de phrases, référence humaine indépendante : CSF18.** 238 phrases françaises dont la transcription
phonétique a été corrigée à la main. On mesure le taux d'erreur sur les phonèmes (PER), sur la meilleure des deux
lectures enregistrées de chaque phrase, sans frontière de mot.

| Système | PER strict | PER, timbres confondus (e/ɛ, o/ɔ, ø/œ) |
|---|---|---|
| **Lectura G2P v0** | **6,65 %** | **2,76 %** |
| espeak-ng 1.51 | 12,85 % | 8,90 % |

Une grande part de l'écart strict vient du timbre propre au locuteur (« aux », « nos » prononcés avec ɔ). Ces phrases
couvrent tous les sons du français mais contiennent peu d'homographes.

CSF18 : Liu, Hueber, Feng, Beautemps, Sankar, DOI [10.5281/zenodo.5554849](https://doi.org/10.5281/zenodo.5554849) (CC BY 4.0) ; Sankar et al., ICASSP 2022.

**2. Corpus d'évaluation de Lectura** (phrases annotées en Universal Dependencies, test de 2 409 phrases). La
référence phonétique vient d'un autre système de Lectura, avec ses conventions : à lire comme une indication.

| | Lectura G2P v0 |
|---|---|
| Mots justes, strict | 91,9 % |
| Mots justes, timbres et ə/ø confondus | 99,0 % |
| PER strict | 4,4 % |

**3. Analyse grammaticale**, test UD (2 409 phrases, ponctuation exclue), référence humaine UD (banc du 16/11, même
modèle) : catégorie du mot (UPOS) 94,6 % ; les 6 traits morphologiques justes ensemble 79,9 %.

Limites constatées :

- les homographes au sens strict (« est », « fils »…) ne sont justes que dans 62 à 73 % des cas (comparaison stricte) ;
- les liaisons obligatoires des inversions (« est-il ») sont souvent manquées ;
- le modèle ne donne pas de lemme.

## Interface Python

### `tokeniser(text) -> list[str]`

Tokenise une phrase francaise (gestion apostrophes, ponctuation, contractions).

### `creer_engine(mode, models_dir, lexicon_path, corrections_lexique)`

Factory pour creer un engine d'inference. Modes : `"auto"` (defaut), `"local"`, `"onnx"` (tous trois : ONNX Runtime).
Les modes `"numpy"` et `"pure"` levent `ValueError` dans la v0 (voir « Backend d'inference »).
`models_dir` permet de specifier le dossier des modeles (sinon cascade automatique, puis telechargement au premier usage).
`corrections_lexique=True` charge les corrections etendues du lexique (~30K entrees).

### `engine.analyser(tokens, *, use_lex=True, sep_hyphen=False, sep_apos=False) -> dict`

Analyse une liste de tokens et retourne un dictionnaire :
- `g2p` : transcription IPA par token
- `pos` : etiquette POS par token
- `liaison` : label liaison par token (`none`, `Lz`, `Lt`, `Ln`, `Lr`, `Lp`)
- `morpho` : dict de listes par trait (`Number`, `Gender`, `VerbForm`, `Mood`, `Tense`, `Person`)

Options :
- `use_lex=False` : desactive les features lexicales
- `sep_hyphen=True` : conserve les tirets dans l'IPA des mots composes
- `sep_apos=True` : conserve les apostrophes dans l'IPA des elisions

### `corriger_g2p(mot, ipa, pos, *, keep_sep=False) -> str`

Applique la table de corrections puis les regles (ex+consonne, ex+voyelle, yod).
`keep_sep=True` conserve les separateurs `-` et `'` dans l'IPA retourne.

### `appliquer_liaison(tokens, phones, liaisons) -> list[str]`

Applique les consonnes de liaison entre tokens consecutifs.

## Architecture du modele (V4)

```
Phrase → Char Embedding (64d) → Shared BiLSTM (2x160h → 320d)
                                        |
                    +-------------------+-------------------+
                    v                                       v
              G2P Head (per-char)         Word repr (320d) + Lex Features (24d)
              Linear(320→40)                        |
                                          Word BiLSTM (128h → 256d)
                                                |
                                    +---+---+---+---+---+---+---+
                                   POS Num Gen VF  Mood Tns Per Liaison
```

- **Entree** : sequence de caracteres avec `<BOS>`, `<SEP>`, `<EOS>`
- **Lex Features** : 24d par mot (21 POS one-hot + known + n_candidates + unambiguous)
- **G2P** : prediction par caractere avec labels `_CONT` (continuation) et separateurs `-`/`'`
- **Tetes mot** : representation mot = fwd[dernier_char] || bwd[premier_char] + lex_proj(24→24)

## Limites connues

- Les homographes au sens strict (« est », « fils »…) ne sont justes que dans 62 à 73 % des cas
- Sur le jeu hors domaine, seules 66,5 % des liaisons obligatoires sont faites (inversions du type « est-il »)
- Le G2P est evalue en contexte phrastique ; la performance sur mots isoles hors-vocabulaire est plus basse
- La liaison `Lp` est tres rare dans les donnees d'entrainement
- Les noms propres et neologismes peuvent produire des transcriptions approximatives

## Changelog

### 4.1.11
- Mode local seul : le mode API (`mode="api"`, `api_url`, `api_key`, `LECTURA_API_KEY`) est retiré
- Poids publiés en clair sur Hugging Face (`lecturaworld/lectura-g2p-v0`), téléchargés au premier usage et vérifiés par SHA-256 ; plus de fichiers chiffrés `.enc`
- `LECTURA_HORS_LIGNE=1` désactive le téléchargement ; `onnxruntime` et `numpy` deviennent des dépendances
- Tirets d'inversion (« est-ce », « dit-il ») recollés au pronom ; corrections « cet » et « monsieur »

### 4.1.1
- Fix pipeline_formules : elisions clitiques, propagation sep_hyphen aux compounds, neutralisation liaison avant ponctuation

### 4.1.0
- Fix separateurs : `corriger_g2p()` avec `keep_sep`, corrections enrichies avec separateurs
- Simplification : V4 uniquement, suppression cascades V2/V3
- Ajout `g2p_corrections_lexique.json` (30K corrections avec separateurs)
- Suppression dependance `lexique_pos_candidates.json` (fallback `lectura-lexique`)

### 4.0.2
- Corrections G2P etendues, homographes NOM/NOM PROPRE

### 4.0.0
- Renommage `lectura-g2p` → `lectura-phonemiseur`
- Modele V4 avec separateurs natifs

## Licence

Code sous AGPL-3.0-or-later, ou sous licence commerciale Lectura (intégration dans un produit fermé). Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Les tables de données listées dans le NOTICE restent sous CC BY-SA 4.0. Autres cas sur demande : https://www.lectura.world/contact/

Poids des modèles Lectura (`*.onnx`, `*.pt`, `*_weights.json`, `*_vocab.json`) : licence CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/), ou licence commerciale Lectura (incluse dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande) ; voir [MODEL_LICENCE.md](MODEL_LICENCE.md).

La licence commerciale couvre le code, pas les données listées dans le fichier NOTICE, qui restent sous CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
