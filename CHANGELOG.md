# Journal des versions des modules Lectura v0

## 24/11/2026 : version de maintenance (correctifs mineurs)

Les poids publiés sur Hugging Face ne changent pas. Tokeniseur, Diphone et VC-ZeroShot sont inchangés.

| Paquet | Version | Changement |
|---|---|---|
| lectura-formules | 3.8.1 | `lire_intervalle` lit aussi la forme écrite : « de 3 à 5 » → « de trois à cinq » (elle était épelée « dé-e-trois-à-cinq ») ; README : accents |
| lectura-lexique | 1.6.1 | bases v4/v5 : `filtrer()` et `compter()` appliquent `genre`, `nombre` et `personne` même sans `cgram` (ils étaient ignorés sans message) ; `iter_sqlite` ouvre la base en lecture seule et lève `FileNotFoundError` sur un chemin absent (au lieu de créer une base vide) ; README : accents |
| lectura-aligneur | 4.0.7 | lettre fusionnée : « taxi » donne les syllabes `ta`, `x²i` (et non `xx²i`), de même « examen », « boxer », « xylophone » ; README : accents |
| lectura-phonemiseur | 4.2.3 | propriété publique `ResultatPhraseG2P.groupes` (groupes de lecture) ; README : accents |
| lectura-graphemiseur | 4.4.3 | README : accents |
| lectura-g2p | 4.2.1 | groupes de lecture accessibles par `result.groupes` (l'attribut `_groupes` reste lisible) ; dépend de lectura-phonemiseur >= 4.2.3 ; statut « Production/Stable » comme les autres paquets v0 ; README : accents et exemple des groupes |
| lectura-p2g | 4.7.1 | statut « Production/Stable » comme les autres paquets v0 ; README : accents |

## 23/11/2026, soir (2) : documentation seulement

phonemiseur 4.2.2 et graphemiseur 4.4.2 : résultats aussi sur le corpus d'évaluation de Lectura (phrases ordinaires), en plus de CSF18 (phrases difficiles) et de WikiPron.

## 23/11/2026, soir : documentation seulement

| Paquet | Version | Changement |
|---|---|---|
| lectura-phonemiseur | 4.2.1 | README : résultats mesurés sur CSF18 (chaîne complète, liaisons comprises : PER 6,72 %), WikiPron (modèle de mot seul) et Universal Dependencies |
| lectura-graphemiseur | 4.4.1 | README : résultats sur CSF18 et WikiPron ; plus de chiffre FLEURS (les modules NLP ne s'évaluent pas sur ce corpus de parole) |

## Corrections du 23/11/2026

Ces corrections font suite à une relecture complète, avec la documentation des 10 modules et des essais réels. Les poids publiés sur Hugging Face ne changent pas.

| Paquet | Version | Principaux changements |
|---|---|---|
| lectura-tokeniseur | 2.3.7 | inchangé |
| lectura-formules | 3.8.0 | « est » n'est plus englobé dans une heure en reconnaissance phonétique ; « [0 ; 1] » se lit « de zéro à un » ; midi et minuit ; minutes nulles non lues (« 14h00 » → « quatorze heures ») ; `lire_abv`, `lire_romain`, `lire_unite` exportés ; en-têtes de licence remis en AGPL |
| lectura-lexique | 1.6.0 | les bases SQLite sont ouvertes en lecture seule (jamais modifiées) ; filtres, comptage et export complets sur CSV ; export et recherche sur les bases v4/v5 ; étiquettes de sigles `Ys`/`Ya` ; fichier de test rétabli |
| lectura-aligneur | 4.0.6 | README corrigé (exemple, marques ° et ²) ; en-tête de licence remis en AGPL |
| lectura-phonemiseur | 4.2.0 | modes `numpy` et `pure` retirés (poids non publiés) ; extra `[formules]` ; plus de construction automatique d'un lexique de catégories (effet mesuré négligeable ; `lexicon_path` reste possible) ; comptage des téléchargements sur Hugging Face |
| lectura-graphemiseur | 4.4.0 | correcteur neuronal retiré (il abîmait des mots justes) ; 3 fichiers téléchargés au lieu de 7 ; message clair pour l'option V6 non publiée ; README aligné sur le code ; comptage des téléchargements |
| lectura-g2p | 4.2.0 | `aligner_pour_corpus` retiré (sans équivalent dans l'Aligneur publié) ; plus de dépendance à lectura-lexique |
| lectura-p2g | 4.7.0 | « il est quinze heures » ne perd plus « est » ; option `compacter` ; cache des corrections par option ; plus de dépendance à lectura-lexique |
| lectura-diphone | 2.2.0 | banque gardée en mémoire (2,3 s → 0,2 s par appel, même son) ; bornes appliquées ; phonèmes inconnus signalés ; `voix_tau` sans effet signalé ; comptage des téléchargements |
| lectura-vc-zeroshot | 1.4.0 | `detecter`, `marque_retrouvee` et `metadonnees_wav` publics ; `tau` sans effet signalé ; préréglages lus dans le dossier du moteur ; comptage des téléchargements. Le filigrane reste toujours appliqué. |
