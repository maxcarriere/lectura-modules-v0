# lectura-p2g

Génération v0 (2026) de Lectura — https://www.lectura.world/developpement/modules/v0/

Pipeline P2G complet du francais : Graphemiseur + Formules + Noms propres (IPA -> orthographe).

Couche 2 du pipeline P2G, en miroir de `lectura-g2p` pour le G2P.

## Installation

```bash
pip install lectura-p2g              # pipeline complet
pip install lectura-p2g[aligneur]    # + alignement grapheme-phoneme
```

## Modèles

Tout tourne en local : aucun appel à un serveur Lectura.

Les poids des modèles ne sont pas dans les roues. Au premier usage, `creer_engine()` les télécharge depuis Hugging Face,
vérifie leur empreinte SHA-256 et les range dans `~/.lectura/models/p2g/` (ou `$LECTURA_MODELS_DIR/p2g/`) :
[lecturaworld/lectura-p2g-v0](https://huggingface.co/lecturaworld/lectura-p2g-v0), 3 fichiers, 33,8 Mo, dont le lexique phonétique `phone_lexicon.db` (table de données sous CC BY-SA 4.0, voir le NOTICE de `lectura-graphemiseur`). Les usages suivants n'utilisent plus le réseau.

- `LECTURA_HORS_LIGNE=1` : aucun téléchargement ; si des poids manquent, l'erreur `LecturaModelesIndisponibles`
  indique les fichiers attendus et comment les poser à la main.
- Poids sous licence CC BY-NC 4.0 (usage non commercial) ; usage commercial inclus dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande : https://www.lectura.world/contact/

Le paquet contient des tables de données (corrections P2G et entités nommées, au format JSON), pas de modèle.

## Usage

```python
from lectura_p2g import analyser, creer_engine

engine = creer_engine()
result = analyser(["le", "ɑ̃fɑ̃", "ʒu"], engine=engine)
print(result["ortho"])   # ['les', 'enfants', 'jouent']
```

`analyser()` rend un dictionnaire `{"ipa_words", "ortho", "pos", "morpho"}` (plus `"alignments"` avec `aligner=`),
avec une position par mot d'entrée. Les signes de ponctuation de l'entrée (`","`, `"."`...) sont recopiés tels
quels, avec l'étiquette `PUNCT`.

### Cases vides

Quand plusieurs mots en API donnent un seul mot écrit (nombre « vɛ̃ e ɛ̃ » → « 21 », élision « l ɔm » → « l'homme »,
mot composé « oʒuʁ dɥi » → « aujourd'hui », sigle « ɛs ɛn se ɛf » → « SNCF »), le mot écrit occupe la première
position et `ortho` vaut `""` aux positions suivantes, pour que toutes les listes restent alignées sur l'entrée :

```python
r = analyser(["il", "a", "vɛ̃", "e", "ɛ̃", "ɑ̃"], engine=engine)
r["ortho"]                                   # ['il', 'a', '21', '', '', 'ans']
" ".join(o for o in r["ortho"] if o)         # 'il a 21 ans'

r = analyser(["il", "a", "vɛ̃", "e", "ɛ̃", "ɑ̃"], engine=engine, compacter=True)
r["ortho"], r["ipa_words"]                   # ['il', 'a', '21', 'ans'], ['il', 'a', 'vɛ̃ e ɛ̃', 'ɑ̃']
```

Avec `compacter=True`, les cases vides sont retirées de toutes les listes et leur mot en API est rattaché à la
position écrite qui les précède : l'alignement entre listes reste cohérent, mais leur longueur peut être inférieure
à celle de l'entrée.

### Options d'`analyser()`

| Option | Défaut | Effet |
|---|---|---|
| `formule_mode` | `"num"` | formules en chiffres (`"num"` : 21, SNCF) ou en lettres (`"texte"`) |
| `formule_tolerance` | `"exact"` | `"stt"` : reconnaissance tolérante pour une phonétique issue d'une reconnaissance vocale |
| `number_mode` | `"auto"` | `"auto"` : un nombre d'un seul mot (« quinze », « sept ») reste en lettres ; `"num"` : tous les nombres en chiffres ; `"texte"` : aucun |
| `stt_mode` | `"auto"` | `"formule"` : tout détecter (formules mathématiques comprises) ; `"texte"` : sigles seulement |
| `corrections_p2g` | `True` | table de corrections (phonèmes sans ambiguïté → mot) : `True` (base), `"etendu"`, ou `False` |
| `aligner` | `None` | aligneur grapho-phonémique (ajoute `"alignments"`) |
| `compacter` | `False` | retirer les cases vides (voir plus haut) |

`corriger_phrase_pipeline(ortho, pos, morpho, ipa_words=..., phone_lexicon=..., ...)` applique les mêmes corrections
à une sortie brute du graphémiseur (`engine.analyser()`).

### Avec alignement

```python
from lectura_p2g import analyser, creer_engine
from lectura_aligneur import LecturaSyllabeur

engine = creer_engine()
syl = LecturaSyllabeur()
result = analyser(["bɔ̃ʒuʁ"], engine=engine, aligner=syl)
print(result["alignments"][0].syllabes)
```

## Architecture

- **Couche 1** : `lectura-graphemiseur` — modele P2G (poids telecharges au premier usage, voir « Modèles »)
- **Couche 2** : `lectura-p2g` — pipeline complet, dans cet ordre : formules (nombres, sigles, via `lectura-formules`),
  élisions, mots composés, table de corrections, cohérence morphologique et accents, noms propres, entités nommées.

Le correcteur neuronal des versions précédentes a été retiré (23/11/2026) : il abîmait des mots justes
(« il fait beau » → « baa »). Ses fichiers ne sont plus téléchargés.

## Licence

Code sous AGPL-3.0-or-later, ou sous licence commerciale Lectura (intégration dans un produit fermé). Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Les tables de données listées dans le NOTICE restent sous CC BY-SA 4.0. Autres cas sur demande : https://www.lectura.world/contact/

Poids des modèles Lectura (`*.onnx`, `*.pt`, `*_weights.json`, `*_vocab.json`) : licence CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/), ou licence commerciale Lectura (incluse dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande) ; voir [MODEL_LICENCE.md](MODEL_LICENCE.md).

La licence commerciale couvre le code, pas les données listées dans le fichier NOTICE, qui restent sous CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
