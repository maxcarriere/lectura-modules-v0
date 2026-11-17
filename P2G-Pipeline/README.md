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
[lecturaworld/lectura-p2g-v0](https://huggingface.co/lecturaworld/lectura-p2g-v0), 7 fichiers, 36,6 Mo, dont le lexique phonétique `phone_lexicon.db` (table de données sous CC BY-SA 4.0, voir le NOTICE de `lectura-graphemiseur`). Les usages suivants n'utilisent plus le réseau.

- `LECTURA_HORS_LIGNE=1` : aucun téléchargement ; si des poids manquent, l'erreur `LecturaModelesIndisponibles`
  indique les fichiers attendus et comment les poser à la main.
- Poids sous licence CC BY-NC 4.0 (usage non commercial) ; usage commercial inclus dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande : https://www.lectura.world/contact/

Le paquet contient des tables de données (corrections P2G et entités nommées, au format JSON), pas de modèle.

## Usage

```python
from lectura_p2g import analyser, creer_engine

engine = creer_engine()
result = analyser(["le", "ɑ̃fɑ̃", "ʒu"], engine=engine)
print(result["ortho"])
```

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
- **Couche 2** : `lectura-p2g` — pipeline complet (graphemiseur + formules + coherence morpho + noms propres)

## Licence

Code sous AGPL-3.0-or-later, ou sous licence commerciale Lectura (intégration dans un produit fermé). Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Les tables de données listées dans le NOTICE restent sous CC BY-SA 4.0. Autres cas sur demande : https://www.lectura.world/contact/

Poids des modèles Lectura (`*.onnx`, `*.pt`, `*_weights.json`, `*_vocab.json`) : licence CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/), ou licence commerciale Lectura (incluse dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande) ; voir [MODEL_LICENCE.md](MODEL_LICENCE.md).

La licence commerciale couvre le code, pas les données listées dans le fichier NOTICE, qui restent sous CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
