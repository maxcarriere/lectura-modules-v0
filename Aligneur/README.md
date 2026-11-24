# Lectura Aligneur-Syllabeur

Génération v0 (2026) de Lectura — https://www.lectura.world/developpement/modules/v0/

**Aligneur graphème-phonème et syllabeur phonologique du français**

Module autonome, **zéro dépendance** Python. Pivot central du pipeline Lectura, il réalise l'alignement lettre-par-lettre entre orthographe et phonétique, et décompose chaque syllabe en attaque/noyau/coda avec correspondance graphème-phonème.

> **Note v4.0.0 :** la construction des groupes de lecture (élisions, liaisons, enchaînements) a été transférée dans le module **lectura-phonemiseur**. L'Aligneur accepte les groupes préconstruits et se concentre sur l'alignement et la syllabation.

C'est grâce à cet alignement que les corpus d'entraînement des modèles G2P et P2G ont été préparés.

## Installation

```bash
pip install lectura-aligneur
```

## Fonctionnalités

| Fonction | Description |
|----------|-------------|
| **Alignement graphème-phonème** | Correspondance lettre-par-lettre entre orthographe et IPA via algorithme DFS |
| **Lettres muettes** | Détection et marquage des lettres silencieuses (e, s, t, d, h, x...), marquées `°` dans l'orthographe des syllabes |
| **Lettres fusionnées** | Identification des graphèmes multi-phonèmes (x = ks/gz, y...), marquées `²` dans l'orthographe des syllabes |
| **Groupes de lecture** | Accepte les groupes préconstruits par le phonémiseur (élision, liaison, enchaînement) |
| **Syllabation phonologique** | Découpage en syllabes par modèle de sonorité (IPA + orthographe) |
| **Décomposition attaque/noyau/coda** | Chaque syllabe décomposée en ses constituants avec phonèmes distribués |
| **Spans** | Positions caractère de chaque syllabe, groupe et composant dans le texte source |
| **Phonemiseur pluggable** | eSpeak-NG (défaut), Lectura G2P, ou tout objet avec `.phonemize()` / `.predict()` |
| **Formules** | Gestion des lectures de formules (nombres, dates, etc.) avec events alignés : paramètre `lectures_formules` d'`analyser_complet()` (index de **mot** -> `LectureFormule`), groupes marqués `est_formule` |

## Exemples

### Analyse d'un mot (API simple)

```python
from lectura_aligneur import LecturaSyllabeur

syllabeur = LecturaSyllabeur()    # eSpeak-NG par défaut
result = syllabeur.analyze("chocolat")

print(result.format_detail())
# chocolat -> /ʃokola/            (sortie avec eSpeak NG 1.51 ; elle dépend du phonémiseur)
#   σ1: /ʃo/ <<cho>> [0:3] att=ʃ noy=o cod=-
#   σ2: /ko/ <<co>> [3:5] att=k noy=o cod=-
#   σ3: /la/ <<lat°>> [5:8] att=l noy=a cod=-

# Chaque syllabe expose son alignement :
for s in result.syllabes:
    att = " ".join(f"{p.ipa}→{p.grapheme}" for p in s.attaque.phonemes)
    noy = " ".join(f"{p.ipa}→{p.grapheme}" for p in s.noyau.phonemes)
    cod = " ".join(f"{p.ipa}→{p.grapheme}" for p in s.coda.phonemes)
    print(f"  {s.ortho:6s} /{s.phone}/  att=[{att}] noy=[{noy}] cod=[{cod}]  span={s.span}")
# cho    /ʃo/  att=[ʃ→ch] noy=[o→o] cod=[]  span=(0, 3)
# co     /ko/  att=[k→c] noy=[o→o] cod=[]  span=(3, 5)
# lat°   /la/  att=[l→l] noy=[a→at] cod=[]  span=(5, 8)
```

### Marques dans l'orthographe des syllabes

L'orthographe affichée (`Syllabe.ortho`, `format_detail()`) porte deux marques :

| Marque | Sens | Exemple |
|--------|------|---------|
| `°` | la lettre qui précède est **muette** | `lat°` (chocolat), `grande°s°` (grandes), `joue°n°t°` (jouent) |
| `²` | **lettre fusionnée** : une même lettre produit deux phonèmes | `x²` dans taxi (/ks/), examen (/ɡz/) |

Les marques ne figurent que dans `ortho` ; les graphèmes des phonèmes (`Phoneme.grapheme`) et les `span` portent
sur le texte d'origine. Pour un affichage sans marque : `s.ortho.replace("°", "").replace("²", "")`.

### Analyse complète avec groupes de lecture

`analyser_complet()` construit les groupes de lecture (liaisons, enchaînements, élisions) puis syllabe chaque
groupe. La construction des groupes est faite par **`lectura-phonemiseur`**, qui doit être installé
(`pip install lectura-phonemiseur`) ; sans lui, `analyser_complet(mots)` lève `RuntimeError`.

Les mots sont lus par duck typing : tout objet avec les attributs `text`, `phone` et `liaison` convient
(par exemple les `MotAnalyseG2P` produits par `lectura_phonemiseur.analyser_phrase_complete()`). Les codes de
liaison sont `Lz`, `Lt`, `Ln`, `Lr`, `Lp` ou `none`.

```python
from dataclasses import dataclass

from lectura_aligneur import LecturaSyllabeur

@dataclass
class Mot:
    text: str
    phone: str
    liaison: str = "none"

syllabeur = LecturaSyllabeur()

mots = [
    Mot("les", "lez", liaison="Lz"),   # les (liaison en z)
    Mot("enfants", "ɑ̃fɑ̃"),
    Mot("jouent", "ʒu"),
]

result = syllabeur.analyser_complet(mots)   # requiert lectura-phonemiseur

print(f"{result.nb_groupes} groupes, {result.nb_syllabes} syllabes")
print(f"Groupes : {result.format_ligne1()}")
print(f"Syllabes : {result.format_ligne2()}")

# 2 groupes, 4 syllabes
# Groupes : les enfants | jouent
# Syllabes : le.zɑ̃.fɑ̃ | ʒu

# Détail des groupes :
for rg in result.groupes:
    g = rg.groupe
    jonc = ", ".join(g.jonctions) if g.jonctions else "-"
    print(f"  Groupe: /{g.phone_groupe}/  jonctions: {jonc}")
    for s in rg.syllabes:
        print(f"    σ /{s.phone}/ <<{s.ortho}>>  att={s.attaque.phone or '-'} noy={s.noyau.phone} cod={s.coda.phone or '-'}")
#   Groupe: /lezɑ̃fɑ̃/  jonctions: liaison_z
#     σ /le/ <<le>>  att=l noy=e cod=-
#     σ /zɑ̃/ <<sen>>  att=z noy=ɑ̃ cod=-
#     σ /fɑ̃/ <<fant°s°>>  att=f noy=ɑ̃ cod=-
#   Groupe: /ʒu/  jonctions: -
#     σ /ʒu/ <<joue°n°t°>>  att=ʒ noy=u cod=-
```

### API IPA directe (sans phonémiseur)

```python
syllabeur = LecturaSyllabeur()
syllabes = syllabeur.syllabify_ipa("ʃɔkɔla")
print(syllabes)  # ['ʃɔ', 'kɔ', 'la']
```

### Avec le phonémiseur Lectura

Le phonémiseur Lectura (`lectura-phonemiseur`) tourne en local : ses poids sont téléchargés depuis Hugging Face
au premier usage (1,9 Mo, licence CC BY-NC 4.0 ; usage commercial inclus dans toute licence commerciale Lectura v1 en cours de validité, autres cas sur demande : https://www.lectura.world/contact/).

```python
from lectura_aligneur import LecturaSyllabeur
from lectura_phonemiseur import creer_engine

g2p = creer_engine()   # premier appel : télécharge les poids dans ~/.lectura/models/g2p/

# Tout objet avec .predict(word) est accepté
class G2PPhonemizer:
    def predict(self, word):
        return g2p.analyser([word])['g2p'][0]

syllabeur = LecturaSyllabeur(phonemizer=G2PPhonemizer())
result = syllabeur.analyze("maison")
```

## Structures de données

| Classe | Rôle |
|--------|------|
| `ResultatAnalyse` | Analyse d'un mot : `mot`, `phone`, `syllabes[]` |
| `Syllabe` | Syllabe : `phone`, `ortho`, `span`, `attaque`, `noyau`, `coda` |
| `GroupePhonologique` | Attaque/noyau/coda : liste de `Phoneme` avec `ipa` + `grapheme` |
| `Phoneme` | Phonème individuel : `ipa` (ex: "ʃ") + `grapheme` (ex: "ch") |
| Mot (duck typing) | Tout objet avec `text`, `phone`, `liaison` (ex. `MotAnalyseG2P` de `lectura-phonemiseur`) |
| `GroupeLecture` | Groupe de lecture : `mots[]`, `phone_groupe`, `jonctions[]`, `span` |
| `ResultatGroupe` | Groupe syllabe : `groupe` + `syllabes[]` |
| `ResultatSyllabation` | Résultat complet : `groupes[]`, `nb_syllabes`, `nb_groupes` |
| `OptionsGroupes` | Options : `gerer_elisions`, `gerer_liaisons`, `gerer_enchainement` |

## Rôle dans le pipeline Lectura

L'Aligneur-Syllabeur est le **pivot central** de Lectura :

1. **Préparation des corpus** : l'alignement graphème-phonème a permis de constituer les données d'entraînement des modèles G2P et P2G
2. **Lecture assistée** : les groupes de lecture avec syllabes colorées sont la base de l'interface de lecture
3. **Synthèse vocale** : l'alignement et les spans permettent la synchronisation texte-audio

## Caractéristiques techniques

- **Zéro dépendance** Python
- **Alignement DFS** graphème-phonème avec gestion des lettres muettes et fusionnées
- **Modèle de sonorité** pour la syllabation (5 classes : obstruantes, nasales, liquides, semi-voyelles, voyelles)
- **Architecture E1/E2** : groupes de lecture (E1, depuis lectura-phonemiseur) puis syllabation (E2), utilisables séparément
- **Phonemiseur pluggable** : eSpeak-NG, Lectura G2P, ou tout objet compatible
- **Python 3.10+** avec type hints complets (PEP-561)
- **Licence** : AGPL-3.0-or-later, ou licence commerciale (voir « Licence »)

## Licence

Code sous AGPL-3.0-or-later, ou sous licence commerciale Lectura (intégration dans un produit fermé). Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Autres cas sur demande : https://www.lectura.world/contact/

Tables de données : voir le fichier NOTICE (tables écrites par Lectura, sous la licence du code).
