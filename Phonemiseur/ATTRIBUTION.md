# Attribution et crédits

Ce projet repose sur les ressources linguistiques et logiciels suivants.

## Données linguistiques

### GLAFF 1.2

- **Description** : dictionnaire flexionnel du français avec transcriptions phonémiques
- **Auteurs** : Franck Sajous, Nabil Hathout, Basilio Calderone (CLLE-ERSS, CNRS & Université Toulouse Jean Jaurès)
- **Licence** : CC BY-SA 3.0
- **URL** : http://redac.univ-tlse2.fr/lexicons/glaff.html
- **Usage** : données G2P (graphème → phonème) pour l'entraînement Phase 1 (lexique)

### Lexique 3.83

- **Description** : base lexicale du français avec fréquences, phonologie, morphologie
- **Auteurs** : Boris New, Christophe Pallier
- **Licence** : CC BY-SA 4.0
- **URL** : http://www.lexique.org/
- **Usage** : données G2P complémentaires et table de corrections

### Universal Dependencies — French-GSD 2.x

- **Description** : corpus annoté morpho-syntaxique du français
- **Auteurs** : Marie-Catherine de Marneffe, Bruno Guillaume, Ryan McDonald, et al.
- **Licence** : CC BY-SA 4.0
- **URL** : https://universaldependencies.org/treebanks/fr_gsd/index.html
- **Usage** : données d'entraînement Phase 2 (POS, morphologie, contexte phrastique), enrichies avec phonèmes et liaisons

### Wiktionnaire français

- **Description** : dictionnaire collaboratif (formes, prononciations, catégories), y compris par l'extraction kaikki.org
- **Auteurs** : contributeurs du Wiktionnaire
- **Licence** : CC BY-SA 3.0 ou 4.0 selon la date de l'extraction
- **URL** : https://fr.wiktionary.org/
- **Usage** : données G2P complémentaires pour l'entraînement Phase 1 (lexique)

### ipa-dict

- **Description** : dictionnaires de prononciation en API (fichier fr_FR)
- **Auteurs** : open-dict-data (dohliam)
- **Licence** : MIT (notice reproduite en fin de fichier)
- **URL** : https://github.com/open-dict-data/ipa-dict
- **Usage** : données G2P complémentaires pour l'entraînement Phase 1 (lexique)

### Wikipédia en français

- **Description** : phrases de Wikipédia en français (vidage de 2026)
- **Auteurs** : contributeurs de Wikipédia
- **Licence** : CC BY-SA
- **URL** : https://fr.wikipedia.org/
- **Usage** : phase 1b de l'entraînement (500 000 phrases, phonétisées par Lectura)

### Universal Dependencies — French-Sequoia

- **Description** : corpus médical, presse, non-fiction, wiki
- **Auteurs** : Marie Candito, Djame Seddah, Guy Perrier, Bruno Guillaume
- **Référence** : Marie Candito et Djamé Seddah (2012), « Le corpus Sequoia : annotation syntaxique et exploitation pour l'adaptation d'analyseur par pont lexical », Actes de TALN 2012, Grenoble. Licence LGPL-LR.
- **Licence** : LGPL-LR
- **URL** : https://universaldependencies.org/treebanks/fr_sequoia/index.html
- **Usage** : données d'entraînement Phase 2 complémentaires (aucune table du paquet n'en reprend le contenu)

### Universal Dependencies — French-Rhapsodie

- **Description** : corpus oral du français
- **Auteurs** : Kim Gerdes, Sylvain Kahane, Mariam Nakhle, et al.
- **Licence** : CC BY-SA 4.0
- **URL** : https://universaldependencies.org/treebanks/fr_rhapsodie/index.html
- **Usage** : données d'entraînement Phase 2 (français oral)

## Logiciels et frameworks

### PyTorch

- **Licence** : BSD-3-Clause
- **URL** : https://pytorch.org/
- **Usage** : entraînement du modèle (non requis pour l'inférence)

### ONNX / ONNX Runtime

- **Licence** : Apache 2.0 / MIT
- **URL** : https://onnxruntime.ai/
- **Usage** : export et inférence optimisée (optionnel)

### NumPy

- **Licence** : BSD-3-Clause
- **URL** : https://numpy.org/
- **Usage** : backend d'inférence léger (optionnel)

## Notice de licence d'ipa-dict

```
MIT License

Copyright (c) 2016 dohliam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
