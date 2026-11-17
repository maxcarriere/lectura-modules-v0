# Changements

## 2.1.0 (republication v0, 2026)

- Tout en local : retrait du G2P par API, de l'URL d'API et des modèles chiffrés (`.enc`).
- Banque de diphones hors du paquet : téléchargée depuis Hugging Face
  (`lecturaworld/lectura-diphone-v0`) au premier usage, vérifiée par SHA-256
  (`LECTURA_HORS_LIGNE`, `LECTURA_HF_BASE`, `LECTURA_MODELS_DIR`).
- Marquage IA : WAV avec métadonnées IPTC `trainedAlgorithmicMedia`
  (`DiphoneResult.vers_wav`, `ecrire_wav`, ligne de commande).
- Changement de timbre limité aux timbres `fr-f1`, `melange-f`, `melange-h`
  (préréglages de lectura-vc-zeroshot 1.3, voix mélangées branchées) ;
  exige lectura-vc-zeroshot >= 1.3 (filigrane IA) ; plus de fichier audio de
  référence ni de mélange libre.
- Ligne de commande `lectura-diphone`.
- `pyworld`, `numpy`, `scipy` deviennent des dépendances obligatoires
  (l'extra `[local]` est conservé, vide) ; nouveaux extras `[g2p]` et `[vc]`.
- Retrait du post-filtre neuronal inutilisé (`_postfilter.py`).
- Licences : code AGPL-3.0-or-later ; données tirées du corpus SIWIS sous
  CC BY 4.0 (NOTICE, LICENCE-DONNEES.txt).
