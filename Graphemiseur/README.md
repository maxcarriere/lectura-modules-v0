# Lectura Graphemiseur

Génération v0 (2026) de Lectura — https://www.lectura.world/developpement/modules/v0/

**Graphemiseur neural du francais : P2G + POS + Morphologie (IPA → orthographe)**

> Anciennement `lectura-p2g` (pip) / `lectura_p2g` (import).
> Renomme `lectura-graphemiseur` / `lectura_graphemiseur` a partir de la v4.0.0.

Un seul modele BiLSTM char-level multi-tete V7 avec attention cross word-char, lex_select et phone_lex_features (ONNX INT8) qui predit simultanement :

- **P2G** : transcription IPA vers orthographe (chaîne complète v0 : 6,9 % de mots faux, voir « Résultats »)
- **POS** : etiquetage morpho-syntaxique — 19 tags
- **Morphologie** : genre, nombre, temps, mode, personne, forme verbale

Le modele V7 ajoute un mecanisme d'attention cross word-char et le lex_select (selection lexicale parmi candidats phonetiques).

Tout tourne en local (ONNX Runtime). Les poids du modele et le lexique phonetique (36,6 Mo en tout) sont telecharges
depuis Hugging Face au premier usage, puis gardes sur le disque ; les poids sont sous licence CC BY-NC 4.0 (voir « Modèles »).

## Demarrage rapide

### Installation

```bash
pip install lectura-graphemiseur             # installe aussi onnxruntime et numpy
```

### Utilisation rapide (factory — recommande)

```python
from lectura_graphemiseur import creer_engine

# Premier appel : telecharge les poids (36,6 Mo) dans ~/.lectura/models/p2g/
engine = creer_engine()

result = engine.analyser(["le", "ɑ̃fɑ̃", "sɔ̃", "aʁive", "a", "la", "mɛzɔ̃"])

print(result["ortho"])   # ['les', 'enfants', 'sont', 'arrives', 'a', 'la', 'maison']
print(result["pos"])     # ['ART:def', 'NOM', 'AUX', 'VER', 'PRE', 'ART:def', 'NOM']
print(result["morpho"])  # {'Number': ['Plur', 'Plur', ...], 'Gender': [...], ...}
```

Modes disponibles : `"auto"` (defaut, ONNX Runtime), `"local"`, `"onnx"`.

## Modèles

Tout tourne en local : aucun appel à un serveur Lectura.

Les poids du modèle P2G ne sont pas dans la roue. Au premier usage, `creer_engine()` les télécharge depuis
Hugging Face ([lecturaworld/lectura-p2g-v0](https://huggingface.co/lecturaworld/lectura-p2g-v0), 7 fichiers, 36,6 Mo), vérifie leur empreinte SHA-256
et les range dans `~/.lectura/models/p2g/` (ou `$LECTURA_MODELS_DIR/p2g/`, ou le dossier passé en `models_dir=`).
Les usages suivants n'utilisent plus le réseau.

- `LECTURA_HORS_LIGNE=1` : aucun téléchargement ; si des poids manquent, l'erreur `LecturaModelesIndisponibles`
  indique les fichiers attendus, leurs empreintes et comment les poser à la main.
- Poids sous licence CC BY-NC 4.0 (usage non commercial) ; usage commercial inclus dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande : https://www.lectura.world/contact/
- Le lexique phonétique `phone_lexicon.db`, téléchargé avec les poids, est une table de données sous CC BY-SA 4.0 (voir NOTICE).

## Backend d'inference

Le paquet tourne avec ONNX Runtime (environ 2 ms par phrase sur CPU). Les backends NumPy et pur Python restent
dans le code, mais demandent des poids convertis (`modeles_numpy/`) qui ne sont pas publiés.

## Pipeline V7 (raw → lex_select → post-traitement)

Le pipeline complet applique trois etapes successives :

1. **Raw** : prediction brute du modele BiLSTM
2. **Lex_select** : selection lexicale par phone_lexicon — le modele choisit parmi les candidats phonetiquement compatibles
3. **Post-traitement** :
   - **Formules** : reconnaissance deterministe de nombres, sigles, dates via `lectura_formules`
   - **Coherence morpho** : accord ortho-morpho (pluriel, feminin, conjugaison)
   - **Accents** : correction a/a, ou/ou par POS

## Phone_lex_features (V7)

Le modele V7 utilise un vecteur de 28 dimensions par mot (`phone_lex_features`), construit a partir du `phone_lexicon.db` :

- 19d : POS one-hot (candidats POS du lexique phonetique)
- 3d : features morphologiques (genre, nombre)
- 6d : features lexicales (known, n_candidates, unambiguous, top_freq, has_verb, has_nom)

Le `phone_lexicon.db` est detecte automatiquement dans le dossier modeles. Sans lexique, le modele fonctionne normalement (features = zeros).

## Résultats

Banc du 16/11/2026, en local sur CPU, chaîne complète P2G v0 (`lectura-p2g`, qui utilise ce paquet), sur le test
de FLEURS (français lu, 676 phrases). Entrée : la phonétique produite par le G2P v0 de Lectura à partir du texte de référence.

| Mesure | Score |
|---|---|
| Mots faux (WER tolérant, sans majuscules ni ponctuation) | 6,9 % |
| Phrases avec au moins une faute | 77 % |

La sortie n'a ni majuscules ni ponctuation. Les têtes POS et morphologie n'ont pas été mesurées sur ce banc.

## Interface Python

### `creer_engine(mode, models_dir, lexicon_path)`

Factory pour creer un engine d'inference. Modes : `"auto"`, `"local"`, `"onnx"` (et `"numpy"`, `"pure"`, voir plus haut).
`models_dir` permet de specifier le dossier des modeles (sinon cascade automatique, puis telechargement au premier usage).

### `engine.analyser(ipa_words, *, use_lex=True) -> dict`

Analyse une liste de mots IPA et retourne un dictionnaire :
- `ipa_words` : liste des mots IPA d'entree
- `ortho` : orthographe reconstruite par mot
- `pos` : etiquette POS par mot
- `morpho` : dict de listes par trait (`Number`, `Gender`, `VerbForm`, `Mood`, `Tense`, `Person`)

Le parametre `use_lex=False` desactive les features lexicales (utile pour le benchmarking).

### `tokeniser_ipa(text) -> list[str]`

Tokenise une phrase IPA (split sur espaces).

### `corriger_phrase_v3(ortho_words, pos_tags, morpho_features, ..., ipa_words=None) -> list[str]`

Pipeline post-traitement V6 complet : formules (via `ipa_words`) + coherence morpho + accents.
Le parametre `ipa_words` active la reconnaissance de formules (nombres, sigles).

### `corriger_phrase_v2(ortho_words, pos_tags, lexique) -> list[str]`

Post-traitement contextuel inter-mots : accord determinant-nom, sujet-verbe.

## Architecture du modele (V7)

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

- **Entree** : sequence de caracteres IPA avec `<BOS>`, `<SEP>`, `<EOS>`
- **Phone Lex Features** : 28d par mot (19 POS one-hot + 3 morpho + 6 lex features)
- **Lex Select** : selection parmi candidats phonetiques du phone_lexicon
- **Word Feedback** : les representations mot sont diffusees aux positions char correspondantes
- **P2G** : prediction par caractere IPA avec labels `_CONT` (continuation pour marques combinantes)

## Limites connues

- Le P2G est intrinsequement ambigu pour les homophones (est/et, a/a, ses/ces) — resolution partielle par le contexte phrastique
- Les marques morphologiques muettes (-s pluriel, -e feminin) restent une source d'erreur importante
- Le modele ne gere pas la ponctuation ni la casse (entree = IPA pur)
- Performance sur mots hors-vocabulaire plus basse qu'en contexte

## Licence

Code sous AGPL-3.0-or-later, ou sous licence commerciale Lectura (intégration dans un produit fermé). Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Les tables de données listées dans le NOTICE restent sous CC BY-SA 4.0. Autres cas sur demande : https://www.lectura.world/contact/

Poids des modèles Lectura (`*.onnx`, `*.pt`, `*_weights.json`, `*_vocab.json`) : licence CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/), ou licence commerciale Lectura (incluse dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande) ; voir [MODEL_LICENCE.md](MODEL_LICENCE.md).

La licence commerciale couvre le code, pas les données listées dans le fichier NOTICE (`phone_lexicon.db`, téléchargé avec les poids), qui restent sous CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Crédits des sources : [ATTRIBUTION.md](ATTRIBUTION.md).
