# lectura-vc-zeroshot

Génération v0 (2026) de Lectura — https://www.lectura.world/developpement/modules/v0/

Conversion vocale zero-shot du français par OpenVoice, en inférence ONNX pure (sans PyTorch), entièrement en local.
Toute sortie est marquée comme générée par IA (filigrane AudioSeal).

*Zero-shot voice conversion for French with OpenVoice, pure ONNX inference (no PyTorch), fully local.
Every output carries an AI watermark (AudioSeal). English summary at the end.*

## Installation

```bash
pip install lectura-vc-zeroshot
```

Au premier usage, le paquet télécharge depuis Hugging Face
([`lecturaworld/lectura-vc-zeroshot-v0`](https://huggingface.co/lecturaworld/lectura-vc-zeroshot-v0)) ce qui lui manque,
et vérifie chaque fichier par son empreinte SHA-256 :

| Fichier | Taille | Rôle | Licence |
|---|---|---|---|
| `openvoice_se.onnx`, `openvoice_vc.onnx` | 3,3 Mo, 128 Mo | extracteur de timbre et convertisseur OpenVoice V2 | MIT (MyShell.ai) |
| `generator_base.onnx` | 58,9 Mo | générateur du filigrane AudioSeal | MIT (Meta) |
| `detector_base.onnx` | 34,7 Mo | détecteur AudioSeal (seulement si vous appelez `detecter`) | MIT (Meta) |
| `fr-f1.npy`, `melange-f.npy`, `melange-h.npy` | 1 Ko chacun | vecteurs de timbre des préréglages, téléchargés un par un | voir « Préréglages » |
| `fr-f1-variantes.npy`, … | 9 Ko chacun | vecteurs du curseur grave/aigu (seulement si `variante` ≠ 0) | idem |

Dossiers : modèles OpenVoice dans `~/.lectura/models/vc/` (ou `$LECTURA_MODELS_DIR/vc/`, ou `models_dir=`), préréglages
dans son sous-dossier `presets/`, filigrane dans `~/.lectura/models/filigrane/`.
`LECTURA_HORS_LIGNE=1` interdit tout téléchargement (le message d'erreur dit quoi poser à la main) ;
`LECTURA_HF_BASE` remplace `https://huggingface.co` (miroir).

## Utilisation

Le module sert d'abord à **changer le timbre d'une voix de synthèse** (lectura-diphone, synthèse monolocuteur…).
Il accepte **n'importe quel son en entrée** : voix de synthèse ou enregistrement, fichier ou tableau float32 mono
(donner alors `sr_in`), à toute fréquence d'échantillonnage (rééchantillonné à 22 050 Hz).

```python
from lectura_vc_zeroshot import creer_engine, PRESET_SPEAKERS

engine = creer_engine()
print(PRESET_SPEAKERS)                      # ['fr-f1', 'melange-f', 'melange-h']

# vers un préréglage
audio, sr = engine.convert("source.wav", "melange-h")

# un son de synthèse en mémoire (tableau float32 mono à 44 100 Hz), timbre plus grave
audio, sr = engine.convert(son, "fr-f1", sr_in=44100, variante=-0.5)

# mélange pondéré de préréglages
audio, sr = engine.convert("source.wav", {"fr-f1": 0.5, "melange-f": 0.5})

# vers une voix de référence que vous fournissez (zero-shot)
audio, sr = engine.convert("source.wav", "ma_voix.wav")
```

`audio` est un tableau float32 à 22 050 Hz, déjà marqué. Pour des conversions répétées, gardez le même `engine`
(la fonction `convertir(...)` en crée un à chaque appel). Un nom qui n'est ni un préréglage ni un fichier existant lève
`PresetInconnu`.

**Voix de référence fournie par l'utilisateur.** C'est la fonction même d'un outil zero-shot : elle est acceptée.
N'utilisez que votre propre voix, ou une voix dont vous avez l'accord écrit. Imiter la voix d'une personne sans son consentement peut porter atteinte à ses droits (C. civ. art. 9). Diffuser un tel contenu sans qu'il soit évident, ou expressément indiqué, qu'il est généré par un algorithme peut aussi constituer un délit (C. pén. art. 226-8).

## Préréglages

Trois vecteurs de timbre (aucun son n'est livré) :

| Identifiant | Voix | Licence du vecteur |
|---|---|---|
| `fr-f1` | féminine, corpus SIWIS | CC BY 4.0 |
| `melange-f` | féminine, mélange de plusieurs voix, dont SIWIS | CC BY 4.0 |
| `melange-h` | masculine, mélange de plusieurs voix | CC BY 4.0 |

Les voix mélangées sont des vecteurs moyens, mesurés pour ne ressembler à aucune des voix dont elles sont tirées.
Crédits complets dans [NOTICE](NOTICE).

**Curseur grave/aigu** (`variante`, de -1 à +1) : -1 donne une voix plus grave (« homme »), +1 une voix plus aiguë
(« enfant ») ; 0 laisse le préréglage tel quel. C'est le « truc » de la fréquence d'échantillonnage d'OpenVoice
(`sr_override = 22050 / 2**variante`, paramètre toujours accepté). Pour un préréglage, le paquet télécharge
9 vecteurs précalculés (positions -1, -0,75, …, +1) et interpole entre les deux plus proches ; pour une référence
audio, le décalage est calculé à la volée. Mesure (17/11/2026, sur un extrait de parole) : f0 médiane de la sortie
d'environ 85-105 Hz à -1, 105-195 Hz à 0 et 225-375 Hz à +1 selon le préréglage (voix d'entrée féminine).

## Marquage IA (AI Act, art. 50)

**Ne retirez pas le marquage : c'est une obligation de l'art. 50 du règlement européen sur l'IA (AI Act).**

Toute sortie de conversion porte un filigrane inaudible **AudioSeal** (Meta, licence MIT), avec le message de 16 bits
`0x4C01`. Il est posé dans `OpenVoiceConverter.convert`, point de sortie commun de `convert`, `convert_from_se` et
`convertir`. Aucune option ne le désactive : si le modèle du filigrane manque, est altéré ou ne peut pas être téléchargé,
la conversion lève `FiligraneIndisponible` et ne rend aucun son. `lectura_vc_zeroshot.MARQUE_IA` vaut `True`.

**Vérifier un son**, avec le détecteur public d'AudioSeal (MIT, PyTorch) :

```python
# pip install audioseal torchaudio
import torchaudio
from audioseal import AudioSeal

wav, sr = torchaudio.load("sortie.wav")
wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, 16000)
detecteur = AudioSeal.load_detector("audioseal_detector_16bits")
p, message = detecteur.detect_watermark(wav.unsqueeze(0))
print(p, message)   # son de lectura-vc-zeroshot : p >= 0,5 et message 0100110000000001 (0x4C01)
```

ou, sans PyTorch, avec la conversion ONNX du même détecteur (téléchargée au premier appel) :

```python
import soundfile as sf
from lectura_vc_zeroshot._filigrane import detecter, marque_retrouvee

audio, sr = sf.read("sortie.wav", dtype="float32")
p, bits = detecter(audio, sr)
print(marque_retrouvee(p, bits))   # True : marque Lectura retrouvée
```

Pour écrire un WAV avec les métadonnées IA (IPTC `trainedAlgorithmicMedia`, chunks LIST/INFO et XMP) :
`from lectura_vc_zeroshot._filigrane import metadonnees_wav`.
Réglages : `LECTURA_FILIGRANE_FILS` (fils de calcul, 2 par défaut), `LECTURA_FILIGRANE_ARENE=0` (rend la mémoire
après chaque appel, un peu plus lent), `LECTURA_FILIGRANE_ONNX` / `LECTURA_FILIGRANE_DETECTEUR_ONNX` (chemins imposés,
sans téléchargement ; l'empreinte reste vérifiée).

## Licence

Code : AGPL-3.0-or-later, voir [LICENCE.txt](LICENCE.txt). Copyright (C) 2025-2026 Maxime Carrière (Lectura).
La roue ne contient aucun modèle ni vecteur. Les modèles OpenVoice et AudioSeal sont sous licence MIT ; les
trois préréglages sous CC BY 4.0. Textes et crédits : [NOTICE](NOTICE).

Licence commerciale du code : toute licence commerciale Lectura v1 en cours de validité permet aussi d'utiliser la v0 (code et poids) à des fins commerciales, dans le même périmètre et pour la même durée, sans les obligations de l'AGPL. Les modèles de tiers et les préréglages gardent leur licence (MIT, CC BY 4.0). Autres cas sur demande : https://www.lectura.world/contact/

---

*English. Local zero-shot voice conversion for French (OpenVoice V2, ONNX). Models (OpenVoice, MIT; AudioSeal, MIT)
and three preset timbre vectors (`fr-f1`, `melange-f`, `melange-h`, CC BY 4.0) are downloaded from
Hugging Face on first use and checked by SHA-256. A user-supplied reference recording is accepted. Only use your own voice, or a voice whose owner has given you written consent. Imitating a person's voice without their consent may infringe their rights (French Civil Code, art. 9). Distributing such content without it being obvious, or expressly stated, that it was generated by an algorithm may also be a criminal offence (French Criminal Code, art. 226-8). Every output carries an inaudible AudioSeal watermark (16-bit
message `0x4C01`) that cannot be switched off; check it with the public AudioSeal detector
(`audioseal_detector_16bits`) or `lectura_vc_zeroshot._filigrane.detecter`. **Do not remove the AI marking**
(EU AI Act, Art. 50). Code: AGPL-3.0-or-later, or Lectura commercial licence (any valid Lectura v1 commercial licence also permits commercial use of v0 within the same scope and term; other cases on request: https://www.lectura.world/contact/).*
