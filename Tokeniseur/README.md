# Lectura Tokeniseur

Génération v0 (2026) de Lectura — https://www.lectura.world/developpement/modules/v0/

**Normalisateur et tokeniseur complet pour le français**

Module autonome, zéro dépendance externe. Détecte et classifie les formules
(nombres, sigles, dates, téléphones, numéros, ordinaux, fractions, notations
scientifiques, expressions mathématiques).

## Installation

```bash
pip install lectura-tokeniseur
```

## Utilisation

```python
from lectura_tokeniseur import tokenise

tokens = tokenise("Le 25 décembre 2024, il faisait -3°C à Paris.")

for token in tokens:
    if token.type.name != "SEPARATEUR":
        print(f"{token.text:20s}  {token.type.name}")
```

```
Le                    MOT
25                    FORMULE
décembre              MOT
2024                  FORMULE
,                     PONCTUATION
il                    MOT
faisait               MOT
-3°C                  FORMULE
à                     MOT
Paris                 MOT
.                     PONCTUATION
```

`tokenise()` renvoie une liste plate de tokens (`Mot`, `Ponctuation`, `Separateur`, `Formule`), avec leur
position (`span`) dans le texte. `normalise(texte)` prépare le texte (typographie, espaces, Unicode) avant
la tokenisation.

## Fonctionnalités

- **Normalisation** : typographie, espaces, Unicode
- **Tokenisation** : mots, ponctuation, séparateurs
- **Détection de formules** : nombres (entiers, décimaux, négatifs), dates,
  heures, téléphones, sigles, ordinaux, fractions, pourcentages, monnaies,
  unités de mesure, expressions mathématiques, chiffres romains
- **API simple** : `tokenise(texte)` renvoie la liste des tokens

## Licence

Code sous AGPL-3.0-or-later, ou sous licence commerciale Lectura (intégration dans un produit fermé). Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Les tables de données listées dans le NOTICE restent sous CC BY-SA 4.0. Autres cas sur demande : https://www.lectura.world/contact/

La licence commerciale couvre le code, pas les données listées dans le fichier NOTICE, qui restent sous CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
