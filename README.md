# Sleep Detector

Sleep Detector est une application Python de vision par ordinateur qui detecte les signes de somnolence en temps reel via la webcam. Le programme analyse les points de repere du visage, calcule le ratio d'ouverture des yeux (EAR - Eye Aspect Ratio), puis declenche une alarme sonore lorsque les yeux restent fermes pendant plusieurs frames.

## Demo

Voici une demonstration de l'application en fonctionnement :

![Demo Sleep Detector](demo.gif)

Vous pouvez aussi telecharger la video complete :

[Telecharger la video](demo.mp4?raw=true)

## Fonctionnalites

- Detection du visage avec `dlib`
- Suivi des yeux avec les 68 points de repere faciaux
- Calcul automatique du niveau d'ouverture des yeux
- Detection de somnolence lorsque les yeux restent fermes
- Alarme sonore avec `pygame`
- Affichage en temps reel du statut, du nombre de detections et de la valeur EAR

## Technologies utilisees

- Python
- OpenCV
- dlib
- imutils
- scipy
- numpy
- pygame

## Installation

Installez les dependances du projet :

```bash
pip install -r requirements.txt
```

Le fichier `shape_predictor_68_face_landmarks.dat` doit etre present dans le dossier du projet.

## Utilisation

Lancez l'application avec :

```bash
python sleep_detector.py
```

Vous pouvez aussi specifier le chemin du modele manuellement :

```bash
python sleep_detector.py --shape-predictor shape_predictor_68_face_landmarks.dat
```

Appuyez sur `q` pour fermer la fenetre de detection.

## Objectif

Ce projet a pour objectif de montrer comment la vision par ordinateur peut etre utilisee pour construire un systeme simple d'aide a la detection de somnolence, notamment dans un contexte de securite routiere ou de surveillance de fatigue.
