# Lectura — modules v0

Briques de Lectura pour le français écrit et parlé, génération v0. Tout tourne en local, sans clé ni API : les poids des modèles sont téléchargés depuis Hugging Face au premier usage et vérifiés par leur empreinte SHA-256.

Présentation : https://www.lectura.world/developpement/modules/

## Paquets

| Dossier | Paquet PyPI | Rôle | Poids (Hugging Face) |
|---|---|---|---|
| [Tokeniseur](Tokeniseur/) | `lectura-tokeniseur` | normalise et découpe le texte, repère les formules | — |
| [Formules](Formules/) | `lectura-formules` | lit nombres, dates, heures, monnaies, sigles… | — |
| [Lexique](Lexique/) | `lectura-lexique` | accès à une base lexicale (base non fournie) | — |
| [Aligneur](Aligneur/) | `lectura-aligneur` | aligne graphèmes et phonèmes, découpe en syllabes | — |
| [Phonemiseur](Phonemiseur/) | `lectura-phonemiseur` | écrit → phonétique (modèle neuronal) | [lecturaworld/lectura-g2p-v0](https://huggingface.co/lecturaworld/lectura-g2p-v0) |
| [Graphemiseur](Graphemiseur/) | `lectura-graphemiseur` | phonétique → écrit (modèle neuronal) | [lecturaworld/lectura-p2g-v0](https://huggingface.co/lecturaworld/lectura-p2g-v0) |
| [G2P-Pipeline](G2P-Pipeline/) | `lectura-g2p` | chaîne complète écrit → phonétique | via lectura-phonemiseur |
| [P2G-Pipeline](P2G-Pipeline/) | `lectura-p2g` | chaîne complète phonétique → écrit | via lectura-graphemiseur |
| [Diphone](Diphone/) | `lectura-diphone` | synthèse vocale par diphones (vocodeur WORLD) | [lecturaworld/lectura-diphone-v0](https://huggingface.co/lecturaworld/lectura-diphone-v0) |
| [VC-ZeroShot](VC-ZeroShot/) | `lectura-vc-zeroshot` | conversion de voix (OpenVoice, ONNX), sons filigranés | [lecturaworld/lectura-vc-zeroshot-v0](https://huggingface.co/lecturaworld/lectura-vc-zeroshot-v0) |

Installation, par exemple :

```bash
pip install lectura-g2p          # écrit → phonétique
pip install lectura-p2g          # phonétique → écrit
pip install "lectura-diphone[g2p,vc]"   # synthèse vocale, avec changement de timbre
```

Chaque dossier a son README (usage, variables d'environnement, hors ligne).

Ce dépôt ne contient aucun poids, aucun son et aucune base de données : seulement le code, les tests et les tables de données légères livrées dans les paquets.

## Marquage IA

Les sons produits par `lectura-diphone` portent des métadonnées qui indiquent une voix de synthèse ; après un changement de timbre (`lectura-vc-zeroshot`), ils portent aussi un filigrane inaudible (AudioSeal). **Ne retirez pas ce marquage** (règlement européen sur l'IA, art. 50).

## Licences

- **Code** : AGPL-3.0-or-later ([LICENCE.txt](LICENCE.txt)), ou licence commerciale Lectura.
- **Poids des modèles Lectura** (sur Hugging Face) : CC BY-NC 4.0, ou licence commerciale Lectura ; voir [MODEL_LICENCE.md](MODEL_LICENCE.md).
- **Tables de données** listées dans le fichier `NOTICE` de chaque paquet : CC BY-SA 4.0. Banque de diphones et préréglages de timbre : CC BY 4.0. Modèles de tiers (OpenVoice, AudioSeal) : MIT.
- Toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL ni la restriction non commerciale. Les tables de données listées dans le NOTICE restent sous CC BY-SA 4.0. Autres cas sur demande : https://www.lectura.world/contact/

Copyright (C) 2025-2026 Maxime Carrière (Lectura).

## Contributions

Les rapports de bogues et suggestions sont bienvenus (tickets). Les contributions de code ne sont pas acceptées : voir [CONTRIBUTING.md](CONTRIBUTING.md).

---

*English. Lectura v0 modules for written and spoken French: tokenizer, number/formula reader, aligner and syllabifier, neural G2P and P2G, diphone TTS and zero-shot voice conversion. Everything runs locally; model weights are downloaded from Hugging Face (`lecturaworld`) on first use and checked by SHA-256. This repository contains no weights, audio or databases. Code: AGPL-3.0-or-later, or a Lectura commercial licence. Lectura model weights: CC BY-NC 4.0, or a Lectura commercial licence (see MODEL_LICENCE.md). Data tables listed in each NOTICE: CC BY-SA 4.0; diphone bank and timbre presets: CC BY 4.0; third-party models (OpenVoice, AudioSeal): MIT. Any valid Lectura v1 commercial licence also permits commercial use of v0 (code and weights) within the same scope and term; other cases on request. Audio output carries AI-disclosure marking: do not remove it. Code contributions are not accepted; bug reports are welcome.*
