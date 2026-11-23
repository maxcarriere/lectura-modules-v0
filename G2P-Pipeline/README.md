# lectura-g2p

Génération v0 (2026) de Lectura — https://www.lectura.world/developpement/modules/v0/

Complete French grapheme-to-phoneme pipeline — text to IPA phonetics.
Combines tokenizer, formula reader, and neural phonemizer.

*Pipeline G2P complet du francais — texte vers phonetique IPA.
Combine tokeniseur, lecteur de formules et phonemiseur neural.*

## Installation

```bash
pip install lectura-g2p              # pipeline complet
pip install lectura-g2p[aligneur]    # + alignement grapheme-phoneme et syllabation
```

## Modèles

Tout tourne en local : aucun appel à un serveur Lectura.

Les poids des modèles ne sont pas dans les roues. Au premier usage, `creer_engine()` les télécharge depuis Hugging Face,
vérifie leur empreinte SHA-256 et les range dans `~/.lectura/models/g2p/` (ou `$LECTURA_MODELS_DIR/g2p/`) :
[lecturaworld/lectura-g2p-v0](https://huggingface.co/lecturaworld/lectura-g2p-v0), 2 fichiers, 1,9 Mo. Les usages suivants n'utilisent plus le réseau.

- `LECTURA_HORS_LIGNE=1` : aucun téléchargement ; si des poids manquent, l'erreur `LecturaModelesIndisponibles`
  indique les fichiers attendus et comment les poser à la main.
- Poids sous licence CC BY-NC 4.0 (usage non commercial) ; usage commercial inclus dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande : https://www.lectura.world/contact/

## Usage

```python
from lectura_g2p import analyser_phrase, creer_engine

engine = creer_engine()

# Texte → mots, phonetique IPA, categories, liaisons
result = analyser_phrase("Bonjour, il est 14h30.", engine=engine)
print(result.tokens)
print(result.phones)
```

### Phrases IPA (usage TTS)

```python
from lectura_g2p import texte_vers_phrases_ipa, creer_engine

engine = creer_engine()
for ipa, type_phrase in texte_vers_phrases_ipa("Bonjour. Comment vas-tu ?", engine=engine):
    print(type_phrase, ipa)
```

### Avec syllabation

```python
from lectura_g2p import analyser_phrase

result = analyser_phrase("Les enfants jouent.", syllabifier=True)   # necessite lectura-aligneur
print(result.format_ligne2())
```

`aligner_pour_corpus()` (preparation des corpus d'entrainement) n'est pas disponible dans la v0 : elle reposait
sur des outils internes absents de `lectura-aligneur` et leve `NotImplementedError`. Pour l'alignement
grapheme-phoneme, utiliser `lectura_aligneur.LecturaSyllabeur`.

## Architecture

- **Couche 1** : `lectura-phonemiseur` — modele G2P neural BiLSTM (poids telecharges au premier usage, voir « Modèles »)
- **Couche 2** : `lectura-g2p` — pipeline complet (tokeniseur + formules + phonemiseur + groupes de lecture)
- **Backend** : ONNX Runtime, en local

## Licence

Code sous AGPL-3.0-or-later, ou sous licence commerciale Lectura (intégration dans un produit fermé). Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Les tables de données sous CC BY-SA 4.0 des paquets utilisés (voir le NOTICE de lectura-tokeniseur et de lectura-phonemiseur) restent sous cette licence. Autres cas sur demande : https://www.lectura.world/contact/

Poids des modèles Lectura (`*.onnx`, `*.pt`, `*_weights.json`, `*_vocab.json`) : licence CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/), ou licence commerciale Lectura (incluse dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande) ; voir [MODEL_LICENCE.md](MODEL_LICENCE.md).
