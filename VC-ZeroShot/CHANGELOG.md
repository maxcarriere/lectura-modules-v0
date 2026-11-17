# Changements

## 1.3.0 (republication v0, 2026)

- Marquage IA (AI Act, art. 50) : toute sortie porte le filigrane AudioSeal (message 16 bits `0x4C01`), sans option
  pour le désactiver ; `FiligraneIndisponible` si le modèle manque, est altéré ou ne peut pas être téléchargé.
  `MARQUE_IA = True`, `marquer_ia`, détecteur ONNX (`_filigrane.detecter`), métadonnées WAV (`metadonnees_wav`).
- Tout en local, rien dans la roue : modèles OpenVoice (MIT), modèles AudioSeal (MIT) et vecteurs des préréglages
  téléchargés depuis Hugging Face (`lecturaworld/lectura-vc-zeroshot-v0`) au premier usage, vérifiés par SHA-256
  (`LECTURA_HORS_LIGNE`, `LECTURA_HF_BASE`, `LECTURA_MODELS_DIR`). Nouvelle erreur `LecturaModelesIndisponibles`
  (remplace `FileNotFoundError` quand les modèles manquent).
- Retrait des modèles chiffrés (`.enc`) et de leur déchiffrement (`_crypto.py`).
- Préréglages : exactement `fr-f1` (SIWIS, CC BY 4.0), `melange-f` et `melange-h` (CC BY 4.0), en
  vecteurs seuls. Les préréglages des versions précédentes tirés de lecteurs identifiables sont retirés, ainsi que
  tous les sons de référence ; un ancien nom lève `PresetInconnu`.
- Curseur grave/aigu `variante` (-1 à +1, équivalent de `sr_override = 22050 / 2**variante`) : pour chaque
  préréglage, 9 vecteurs précalculés par le truc SR (`<identifiant>-variantes.npy`, téléchargés au besoin), interpolés
  linéairement ; `variante=0` rend le préréglage tel quel. Paramètre `variante` de `convert` et `convertir`.
- Une voix de référence fournie par l'utilisateur reste acceptée (conversion zero-shot) ; le README rappelle le
  consentement nécessaire.
- Licences : code AGPL-3.0-or-later ; NOTICE avec les textes MIT d'OpenVoice et d'AudioSeal et les crédits des
  préréglages. Titulaire : Maxime Carrière (Lectura).
