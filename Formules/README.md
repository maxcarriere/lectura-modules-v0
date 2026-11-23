# Lectura Formules

Génération v0 (2026) de Lectura — https://www.lectura.world/developpement/modules/v0/

Lecture algorithmique des formules pour le francais : nombres, sigles, dates, telephones, heures, monnaies, ordinaux, fractions, scientifiques, mathematiques, GPS, etc.

Module autonome, zero dependance externe.

## Installation

```bash
pip install lectura-formules
```

## Utilisation rapide

```python
from lectura_formules import lire_nombre, lire_formule

# Lire un nombre
result = lire_nombre("42")
print(result.display_fr)    # "quarante-deux"
print(result.phone)         # "kaʁɑ̃t dø"

# Lire une formule dont le type est connu
print(lire_formule("nombre", "42").display_fr)   # "quarante-deux"
```

### Enrichir les tokens du tokeniseur Lectura

`enrichir_formules()` complete, sur place, les tokens de type `FORMULE` produits par `lectura-tokeniseur`
(attributs `display_fr` et `lecture`) :

```python
from lectura_tokeniseur import tokenise
from lectura_formules import enrichir_formules

tokens = tokenise("Il a 3 chats.")
enrichir_formules(tokens)
print([(t.text, t.display_fr) for t in tokens if t.type.name == "FORMULE"])
# [('3', 'trois')]
```

## Sons (optionnel)

Les fichiers sons (WAV) des formules ne sont **pas** inclus dans ce paquet.
Si vous disposez d'un dossier de sons, indiquez son chemin :

```python
from lectura_formules import set_sounds_dir, get_sound_path

set_sounds_dir("/chemin/vers/sons_formules")

# identifiant de son (nom du fichier WAV sans extension, par exemple "num017")
wav = get_sound_path("num017")
if wav:
    print(f"Fichier son : {wav}")
```

## API principale

| Fonction | Description |
|---|---|
| `lire_formule(type, texte)` | Lit une formule dont le type est donne ("nombre", "date", "heure", "sigle"...) |
| `lire_nombre(texte)` | Nombres : "42" -> "quarante-deux" |
| `lire_date(texte)` | Dates : "25/12/2024" -> "vingt-cinq-décembre-deux-mille-vingt-quatre" |
| `lire_heure(texte)` | Heures : "14h30" -> "quatorze-heures-trente" |
| `lire_telephone(texte)` | Telephones : "06 12 34 56 78" |
| `lire_sigle(texte)` | Sigles : "SNCF" -> "esse-enne-cé-effe" |
| `lire_ordinal(texte)` | Ordinaux : "3e" -> "troisième" |
| `lire_fraction(texte)` | Fractions : "3/4" -> "trois-quarts" |
| `lire_monnaie(texte)` | Monnaies : "42 EUR" -> "quarante-deux-euros" |
| `lire_pourcentage(texte)` | Pourcentages : "50%" -> "cinquante-pour cent" |
| `lire_intervalle(texte)` | Intervalles : "[0 ; 1]" -> "de-zéro-à-un" |
| `lire_romain(texte)` | Nombres romains lus comme cardinaux : "XIV" -> "quatorze" |
| `lire_abv(texte)` | Abréviations : "M." -> "monsieur" (sinon épelées) |
| `lire_unite(texte)` | Unités : "km/h" -> "kilomètre-par-heure" |
| `lire_scientifique`, `lire_maths`, `lire_numero`, `lire_gps`, `lire_page_chapitre` | Autres types, aussi accessibles par `lire_formule` |
| `enrichir_formules(tokens)` | Enrichit les tokens FORMULE d'une phrase tokenisee |
| `int_to_roman(n)` / `roman_to_int(s)` | Chiffres romains |

`lire_formule` accepte 18 types : nombre, sigle, date, telephone, ordinal, fraction, scientifique, maths, numero,
heure, monnaie, pourcentage, intervalle, gps, page_chapitre, romain, abv, unite (un type inconnu est épelé).

Heures : les minutes nulles ne sont pas lues ("14h00" -> "quatorze-heures") ; avec
`OptionsLecture(heure_minuit_midi=True)`, "12h00" -> "midi" et "0h00" -> "minuit".

### Options (`OptionsLecture`)

| Option | Défaut | Effet |
|---|---|---|
| `fraction_mode` | `"hybride"` | `"hybride"`, `"ordinal"` ou `"standard"` |
| `decimal_method` | `"m2"` | partie décimale : `"m1"` (comme un entier), `"m2"` (groupes de 3), `"m3"` (chiffre par chiffre) |
| `heure_mot_minutes` | `False` | dire « minutes » après les heures (« 14h30 ») |
| `heure_minuit_midi` | `False` | 0h -> « minuit », 12h -> « midi » |
| `monnaie_dire_centimes` | `True` | lire les centimes |
| `romain_actif` | `True` | calculer `display_rom` pour les nombres |
| `auto_convert_sci` | `False` | convertir décimal <-> scientifique |

### Reconnaissance (phonétique -> formule)

`reconnaitre_ipa`, `reconnaitre_ipa_stt` (tolérant aux erreurs de transcription), `detect_number_spans`,
`detect_sigle_spans`, `detect_formule_spans_stt`, `detect_formula_spans` et `reconnaitre_maths_ipa*` repèrent les
formules dans une suite de mots en API ; `lectura-p2g` s'en sert. Le connecteur « et » (/e/, /ɛ/) n'est gardé qu'à
l'intérieur d'une formule (« vingt et un ») : en bord, c'est un mot de la phrase (« il est quinze heures »).

## Licence

Code sous AGPL-3.0-or-later, ou sous licence commerciale Lectura (intégration dans un produit fermé). Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Autres cas sur demande : https://www.lectura.world/contact/

Tables de données : voir le fichier NOTICE (tables écrites par Lectura, sous la licence du code).
