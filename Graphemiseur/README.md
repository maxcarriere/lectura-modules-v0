# Lectura Graphemiseur

Génération v0 (2026) de Lectura — https://www.lectura.world/developpement/modules/v0/

**Graphemiseur neural du français : P2G + POS + Morphologie (IPA → orthographe)**

> Anciennement `lectura-p2g` (pip) / `lectura_p2g` (import).
> Renommé `lectura-graphemiseur` / `lectura_graphemiseur` à partir de la v4.0.0.

Un seul modèle BiLSTM char-level multi-tête V7 avec attention cross word-char, lex_select et phone_lex_features (ONNX INT8) qui prédit simultanément :

- **P2G** : transcription IPA vers orthographe (chaîne complète v0 : voir « Résultats »)
- **POS** : étiquetage morpho-syntaxique — 19 tags
- **Morphologie** : genre, nombre, temps, mode, personne, forme verbale

Le modèle V7 ajoute un mécanisme d'attention cross word-char et le lex_select (sélection lexicale parmi candidats phonétiques).

Tout tourne en local (ONNX Runtime). Les poids du modèle et le lexique phonétique (33,8 Mo en tout) sont téléchargés
depuis Hugging Face au premier usage, puis gardés sur le disque ; les poids sont sous licence CC BY-NC 4.0 (voir « Modèles »).

## Démarrage rapide

### Installation

```bash
pip install lectura-graphemiseur             # installe aussi onnxruntime et numpy
```

### Utilisation rapide (factory — recommandé)

```python
from lectura_graphemiseur import creer_engine

# Premier appel : télécharge les poids (33,8 Mo) dans ~/.lectura/models/p2g/
engine = creer_engine()

result = engine.analyser(["le", "ɑ̃fɑ̃", "sɔ̃", "aʁive", "a", "la", "mɛzɔ̃"])

print(result["ortho"])   # ['les', 'enfants', 'sont', 'arrivés', 'à', 'la', 'maison']
print(result["pos"])     # ['ART:def', 'NOM', 'AUX', 'VER', 'PRE', 'ART:def', 'NOM']
print(result["morpho"])  # {'Number': ['Plur', 'Plur', ...], 'Gender': [...], ...}
```

Modes disponibles : `"auto"` (défaut, ONNX Runtime), `"local"`, `"onnx"`.

## Modèles

Tout tourne en local : aucun appel à un serveur Lectura.

Les poids du modèle P2G ne sont pas dans la roue. Au premier usage, `creer_engine()` les télécharge depuis
Hugging Face ([lecturaworld/lectura-p2g-v0](https://huggingface.co/lecturaworld/lectura-p2g-v0), 3 fichiers, 33,8 Mo), vérifie leur empreinte SHA-256
et les range dans `~/.lectura/models/p2g/` (ou `$LECTURA_MODELS_DIR/p2g/`, ou le dossier passé en `models_dir=`).
Les usages suivants n'utilisent plus le réseau.

- `LECTURA_HORS_LIGNE=1` : aucun téléchargement ; si des poids manquent, l'erreur `LecturaModelesIndisponibles`
  indique les fichiers attendus, leurs empreintes et comment les poser à la main.
- Poids sous licence CC BY-NC 4.0 (usage non commercial) ; usage commercial inclus dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande : https://www.lectura.world/contact/
- Le lexique phonétique `phone_lexicon.db`, téléchargé avec les poids, est une table de données sous CC BY-SA 4.0 (voir NOTICE).
- `LECTURA_P2G_VERSION=v6` : usage interne seulement. Le modèle V6 n'est pas publié ; sans ses fichiers,
  `creer_engine()` lève `FileNotFoundError`. Laisser la variable vide pour le modèle V7 publié.
- Le correcteur neuronal des versions précédentes a été retiré (23/11/2026) : il abîmait des mots justes.
  Ses fichiers ne sont plus téléchargés.

## Backend d'inférence

Le paquet tourne avec ONNX Runtime (25 à 60 ms par phrase mesurés sur CPU, limité à 2 cœurs). Les backends NumPy et pur Python restent
dans le code, mais demandent des poids convertis (`modeles_numpy/`) qui ne sont pas publiés.

## Pipeline V7 (raw → lex_select)

`engine.analyser()` applique deux étapes :

1. **Raw** : prédiction brute du modèle BiLSTM
2. **Lex_select** : sélection lexicale par phone_lexicon — le modèle choisit parmi les candidats phonétiquement compatibles

Le post-traitement n'est pas appliqué automatiquement. Ce paquet fournit `corriger_phrase_v3()` (cohérence
morphologique et accents, à appeler soi-même). Les formules (nombres, sigles, dates), les élisions, les mots composés,
les noms propres et la chaîne complète sont dans `lectura-p2g` (fonction `analyser()`), qui dépend de ce paquet et de
`lectura-formules`.

## Phone_lex_features (V7)

Le modèle V7 utilise un vecteur de 28 dimensions par mot (`phone_lex_features`), construit à partir du `phone_lexicon.db` :

- 19d : POS one-hot (candidats POS du lexique phonétique)
- 3d : features morphologiques (genre, nombre)
- 6d : features lexicales (known, n_candidates, unambiguous, top_freq, has_verb, has_nom)

Le `phone_lexicon.db` est détecté automatiquement dans le dossier modèles. Sans lexique, le modèle fonctionne normalement (features = zéros).

## Résultats

Bancs du 23/11/2026, en local sur CPU, avec le code publié corrigé (sans le correcteur neuronal, retiré : il
abîmait des mots justes) et les poids publiés. Aucun outil public comparable n'existe à notre connaissance.

**1. Écriture de phrases : CSF18** (238 phrases ; entrée : la phonétique corrigée à la main, frontières de mots
posées par alignement). Mots faux (WER strict, casse et ponctuation ignorées) de la chaîne complète `lectura-p2g` :

| | Base naïve | **Lectura v0** |
|---|---|---|
| CSF18 | 41,2 % | **25,3 %** |

La base naïve écrit chaque mot avec l'orthographe la plus fréquente de sa prononciation, sans contexte. Les phrases de
CSF18 sont faites pour couvrir tous les sons du français : beaucoup de noms propres et de mots rares, peu de sens, et
la même notation pour « de » et « deux ». Le chiffre est donc sévère pour un outil qui s'appuie sur le contexte.

**2. Écriture de phrases ordinaires : corpus d'évaluation de Lectura** (test, 2 408 phrases naturelles). Entrée : la
phonétique de référence de chaque mot, tirée des données de Lectura (à lire comme une indication) ; mots faux (WER
strict) :

| | Base naïve | **Lectura v0** |
|---|---|---|
| Corpus Lectura | 28,9 % | **15,3 %** |

**3. Modèle de mot seul : WikiPron** (5 000 mots), sans lexique (`use_lex=False`, sélection lexicale coupée) :
34,2 % des mots isolés écrits exactement. Sans contexte, l'écriture d'un mot est ambiguë par nature (« ver, vert,
verre, vers ») : c'est la mesure du réseau seul, pas de l'usage réel.

CSF18 : Liu, Hueber, Feng, Beautemps, Sankar, DOI [10.5281/zenodo.5554849](https://doi.org/10.5281/zenodo.5554849) (CC BY 4.0) ; Sankar et al., ICASSP 2022. WikiPron : Lee et al., LREC 2020 (données du Wiktionnaire, CC BY-SA).

## Interface Python

### `creer_engine(mode, models_dir, lexicon_path)`

Factory pour créer un engine d'inférence. Modes : `"auto"`, `"local"`, `"onnx"` (et `"numpy"`, `"pure"`, voir plus haut).
`models_dir` permet de spécifier le dossier des modèles (sinon cascade automatique, puis téléchargement au premier usage).

### `engine.analyser(ipa_words, *, use_lex=True) -> dict`

Analyse une liste de mots IPA et retourne un dictionnaire :
- `ipa_words` : liste des mots IPA d'entrée
- `ortho` : orthographe reconstruite par mot
- `pos` : étiquette POS par mot
- `morpho` : dict de listes par trait (`Number`, `Gender`, `VerbForm`, `Mood`, `Tense`, `Person`)

Le paramètre `use_lex=False` désactive les features lexicales (utile pour le benchmarking).

### `tokeniser_ipa(text) -> list[str]`

Tokenise une phrase IPA (split sur espaces).

### `corriger_phrase_v3(ortho_words, pos_tags, morpho_features, lexique=None, lexique_index=None, freq_map=None, lex_candidates=None, skip_positions=None) -> list[str]`

Post-traitement : cohérence ortho-morpho (pluriel, féminin, conjugaison), accords déterminant/pronom, homophones par
POS (a/à, ou/où) et par morphologie (il/ils, au/aux). `skip_positions` protège des positions de toute correction.
Pas de formules ici : elles sont dans `lectura-p2g`.

### `corriger_phrase_v2(ortho_words, pos_tags, lexique) -> list[str]`

Post-traitement contextuel inter-mots : accord déterminant-nom, sujet-verbe.

## Architecture du modèle (V7)

```
Phrase IPA → Char Embedding (64d) → Shared BiLSTM (2x192h → 384d)
                                          |
                  +-----------------------+--------------------+
                  v                                             v
        Word representations              Word repr (384d) + Phone Lex Features (28d)
        (fwd[last] || bwd[first])                          |
                                                 Word BiLSTM (192h → 384d)
                                                       |
                                            +--------------+--------------+
                                           POS        Morpho (x6)    Attention Cross
                                                                    → P2G Head
                                                                    → Lex_Select Head
```

- **Entrée** : séquence de caractères IPA avec `<BOS>`, `<SEP>`, `<EOS>`
- **Phone Lex Features** : 28d par mot (19 POS one-hot + 3 morpho + 6 lex features)
- **Lex Select** : sélection parmi candidats phonétiques du phone_lexicon
- **Word Feedback** : les représentations mot sont diffusées aux positions char correspondantes
- **P2G** : prédiction par caractère IPA avec labels `_CONT` (continuation pour marques combinantes)

## Limites connues

- Le P2G est intrinsèquement ambigu pour les homophones (est/et, a/a, ses/ces) — résolution partielle par le contexte phrastique
- Les marques morphologiques muettes (-s pluriel, -e féminin) restent une source d'erreur importante
- Le modèle ne gère pas la ponctuation ni la casse (entrée = IPA pur)
- Performance sur mots hors-vocabulaire plus basse qu'en contexte

## Licence

Code sous AGPL-3.0-or-later, ou sous licence commerciale Lectura (intégration dans un produit fermé). Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Les tables de données listées dans le NOTICE restent sous CC BY-SA 4.0. Autres cas sur demande : https://www.lectura.world/contact/

Poids des modèles Lectura (`*.onnx`, `*.pt`, `*_weights.json`, `*_vocab.json`) : licence CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/), ou licence commerciale Lectura (incluse dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande) ; voir [MODEL_LICENCE.md](MODEL_LICENCE.md).

La licence commerciale couvre le code, pas les données listées dans le fichier NOTICE (`phone_lexicon.db`, téléchargé avec les poids), qui restent sous CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Crédits des sources : [ATTRIBUTION.md](ATTRIBUTION.md).
