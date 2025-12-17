# 🚀 Embedded-AI: Détection d’objets & Tracking en temps réel sur STM32MPU

![Demo](step4/block_diagram.png)

## Présentation

Ce projet propose une application complète de vision par ordinateur embarquée sur STM32MPU :
- Acquisition vidéo via **GStreamer**
- Inférence IA (SSD Mobilenet v2, YOLOv8, modèles custom)
- Post-traitement (NMS, filtrage, conversion coordonnées)
- Affichage **GTK** avec overlay (boîtes, scores, ID, trajectoires)
- Tracking temps réel avec **ByteTrack** (ID stable, traînée)

## Fonctionnalités principales

- 📷 Capture caméra double-branche (preview + IA)
- ⚡ Inférence rapide sur modèles quantifiés (`.nb`, `.tflite`)
- 🎨 Overlay graphique custom (couleurs, labels, trajectoires)
- 🏓 Détection ping-pong (modèle 1 classe) & multi-classes COCO
- 🧠 Tracking multi-objets (ByteTrack, historique centroïdes)
- 🛠️ Scripts de lancement & configuration automatique

## Schéma global

![Pipeline](block_diagram.png)

## Structure du projet

```
models/           # Modèles IA et labels
step1/            # Détection SSD Mobilenet v2 (base)
step2/            # Détection YOLOv8 COCO (80 classes)
step3/            # YOLOv8 pingpong (1 classe, overlay adapté)
step4/            # YOLOv8 pingpong + tracking ByteTrack
rapport.md        # Rapport détaillé (étapes, code, difficultés)
Makefile          # Génération PDF/image, automatisation
```

## Lancer une démo

1. **Pré-requis** : Python 3, GStreamer, GTK, PlantUML, Pandoc, modèles `.nb`/`.tflite`.
2. **Générer le rapport PDF** :
   ```sh
   make all
   ```
3. **Générer le schéma pipeline** :
   ```sh
   make image
   ```
4. **Lancer la détection** (exemple Step 4) :
   ```sh
   ./step4/launch_python_object_detection.sh
   ```

## Ressources utiles
- [Documentation STM32MPU](https://wiki.st.com/stm32mpu/)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [PlantUML](https://plantuml.com/fr/)
- [Netron (visualisation modèles)](https://netron.app/)

## Auteur

**theox33** — [GitHub](https://github.com/theox33)

---

> Projet pédagogique, adaptable à d’autres plateformes embarquées.
