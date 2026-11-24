# lectura-lexique

Génération v0 (2026) de Lectura — https://www.lectura.world/developpement/modules/v0/

French lexical database access: lemmas, inflected forms with IPA phonetics, syllabification and morphology.

*Module d'accès à une base lexicale du français : lemmes, formes fléchies avec phonétique IPA, syllabation et morphologie.*

## Installation

```bash
pip install lectura-lexique
```

## Base lexicale non incluse

Ce paquet ne contient que le code d'accès. **Aucune base lexicale n'est incluse** : `Lexique` ouvre le fichier
que vous lui indiquez (SQLite, CSV ou TSV, détection par l'extension).

La base lexicale Lectura (lemmes, formes fléchies, définitions, entités, relations sémantiques) n'est pas distribuée
avec ce paquet : contact via https://www.lectura.world

## Usage

```python
from lectura_lexique import Lexique

lex = Lexique("/chemin/vers/lexique.db")   # ou un fichier .csv / .tsv

# Vérifier l'existence d'un mot
lex.existe("bonjour")  # True

# Phonétique
lex.phone_de("bonjour")

# Conjugaison
lex.conjuguer("manger")  # {mode: {temps: {personne: forme}}}

# Formes fléchies
lex.formes_de("chat")

# Définitions (si la base en contient)
lex.definitions("chat")
```

Les résultats dépendent de la base fournie.

### Filtrer, compter, exporter

```python
lex.filtrer(cgram="NOM", genre="f", limite=20)   # limite=0 : pas de limite
lex.compter(cgram="VER")
lex.exporter("noms.csv", cgram="NOM")            # ou format="sqlite"
```

- **SQLite** : la base est ouverte en **lecture seule** ; elle n'est jamais modifiée (aucun index n'y est
  créé) et peut se trouver sur un support en lecture seule.
- **CSV / TSV** : `filtrer()`, `compter()` et `exporter()` appliquent tous les critères (`cgram`, `genre`,
  `nombre`, `freq_min`/`freq_max`, `nb_syllabes`, `age_min`/`age_max`, `has_age`, `illustrable_min`,
  `categorie`, `has_phone` ; et `mode`, `temps`, `personne` pour `filtrer()`). Un critère qui porte sur une
  colonne absente du fichier déclenche un avertissement (`UserWarning`) et aucune entrée ne le satisfait.
- **Base v4/v5** : `exporter()` écrit toutes les colonnes de `formes` (dont `lemme_id`) et y ajoute `lemme`,
  `cgram`, `genre` et `age` du lemme. Les critères `nombre`, `illustrable_min` et `categorie` n'y sont pas
  gérés à l'export : ils sont ignorés avec un avertissement.

## Licence

Code sous AGPL-3.0-or-later, ou sous licence commerciale Lectura (intégration dans un produit fermé). Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Autres cas sur demande : https://www.lectura.world/contact/
