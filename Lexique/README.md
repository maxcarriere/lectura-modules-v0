# lectura-lexique

Génération v0 (2026) de Lectura — https://www.lectura.world/developpement/modules/v0/

French lexical database access: lemmas, inflected forms with IPA phonetics, syllabification and morphology.

*Module d'acces a une base lexicale du francais : lemmes, formes flechies avec phonetique IPA, syllabation et morphologie.*

## Installation

```bash
pip install lectura-lexique
```

## Base lexicale non incluse

Ce paquet ne contient que le code d'acces. **Aucune base lexicale n'est incluse** : `Lexique` ouvre le fichier
que vous lui indiquez (SQLite, CSV ou TSV, detection par l'extension).

La base lexicale Lectura (lemmes, formes flechies, definitions, entites, relations semantiques) n'est pas distribuee
avec ce paquet : contact via https://www.lectura.world

## Usage

```python
from lectura_lexique import Lexique

lex = Lexique("/chemin/vers/lexique.db")   # ou un fichier .csv / .tsv

# Verifier l'existence d'un mot
lex.existe("bonjour")  # True

# Phonetique
lex.phone_de("bonjour")

# Conjugaison
lex.conjuguer("manger")  # {mode: {temps: {personne: forme}}}

# Formes flechies
lex.formes_de("chat")

# Definitions (si la base en contient)
lex.definitions("chat")
```

Les resultats dependent de la base fournie.

## Licence

Code sous AGPL-3.0-or-later, ou sous licence commerciale Lectura (intégration dans un produit fermé). Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Autres cas sur demande : https://www.lectura.world/contact/
