# Changements

## 2.2.0 (23/11/2026, corrections)

- `synthetiser()` garde le moteur en mémoire (par `models_dir` et `g2p`) au lieu de recharger la banque, le G2P et
  le moteur de conversion à chaque appel ; sorties identiques (vérifié par empreinte, même graine). `vider_cache()`.
- `DiphoneEngine` et `SynthMode` ajoutés à `__all__` (avec `vider_cache`).
- `voix_tau` est sans effet (lectura-vc-zeroshot : pas d'entrée tau dans le modèle ONNX) : défaut `None`,
  `DeprecationWarning` s'il est donné, plus transmis.
- `ap_cleanup` et `formant_sharpening` sont bornés aux maxima annoncés (3.0 et 2.0), avec un avertissement.
- Un phonème absent de la banque (remplacé par un silence) est signalé par un `UserWarning`.
- README : les deux sens de `mode`, `spectral_contrast`, `prosody`, `word_boundaries`, `timbre=` fichier `.json`,
  options de la ligne de commande, cache de `synthetiser()`, rééchantillonnage après `voix=`.

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
