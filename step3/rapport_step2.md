# Rapport Step 2 - YOLOv8 object detection

## 1. Objectifs et perimetre
- Porter l'application Step 1 (SSD Mobilenet v2) vers le modèle `yolov8s_integer_quant_256_fp32_io.nb` et conserver la détection multi-classes (COCO 80 classes).
- Ajuster le lanceur pour utiliser le nouveau modèle et le fichier de labels YOLOv8.
- Concevoir un post-traitement dédié YOLOv8 en s'appuyant sur les références existantes (post-traitement SSD Mobilenet v2 et post-traitement YOLOv8 pose estimation).
- Vérifier que les sorties sont compatibles avec l'overlay graphique existant.

## 2. Analyse des modeles (via Netron / specification fournie)


| Modèle | Entrée | Sortie brute | Décodage | Labels |
|---|---|---|---|---|
| SSD Mobilenet v2 | 1x256x256x3 | 3 tenseurs (scores, boxes, anchors) | Anchors à décoder puis NMS | labels COCO (ordre SSD) |
| YOLOv8 OD | 1x256x256x3 | 1 tenseur 1x84x1344 | Centre->coins puis NMS | labels COCO (ordre YOLOv8) |

Concrètement, SSD fournit les scores + offsets + anchors (il faut decoder).
Alors que YOLOv8 fournit directement centre/largeur/hauteur + scores classes, on convertit en coins et on passe au NMS. Même taille d entree, sorties simplifiees pour YOLOv8.

## 3. Comparaison des post-traitements existants

- **SSD Mobilenet v2 (`step1/ssd_mobilenet_pp.py`)**
  - Le modèle donne: scores + boxes relatives aux anchors + anchors. Il faut **décoder** les anchors pour obtenir des boites réelles, puis filtrer avec un **NMS**.
  _(Code clé dans `step1/ssd_mobilenet_pp.py` : `postprocess_predictions` appelle `decode_predictions` puis `non_max_supression`.)_
  - Avantage: pipeline UI déja en place, format (locations, classes, scores) attendu par l'overlay.
- **YOLOv8 pose estimation**
  - Sortie dense: 2100 détections, 56 valeurs chacune (box + 17 keypoints). On prend le score personne, on convertit centre->coins, on applique NMS, et on gère les keypoints (non utilisés pour l'OD).
  - Utile comme exemple: montre comment lire un gros tenseur, extraire la partie qui nous intéresse, et faire NMS.
- **Impacts pour YOLOv8 object detection**
  - Sortie unique 1x84x1344: pour chaque détection, on prend le **meilleur score de classe**, on convertit centre->coins, puis **NMS**. Pas d'anchors à décoder.
  - On sort au même format que SSD pour que l'overlay dessine sans changement.

### Recap post-traitement
| Modèle | Ce que le modele sort | Etapes de post-process | Pourquoi NMS ? |
|---|---|---|---|
| SSD Mobilenet v2 | scores + boxes relatives + anchors | Décode anchors → filtre confiance → NMS | Eviter les boites dupliquées sur le même objet |
| YOLOv8 pose | boxes centre + keypoints + score personne | Filtre score → centre->coins → NMS (et keypoints) | Garder une seule box/keypoints par personne |
| YOLOv8 OD | boxes centre + scores 80 classes | Meilleure classe → filtre confiance → centre->coins → NMS | Garder une seule box par objet détecté |

Toutes les variantes finissent par un **NMS** pour ne garder qu'une box par objet. La difference est surtout en amont: SSD a besoin de décoder les anchors; YOLOv8 donne directement centre/largeur/hauteur, donc le post-traitement est plus court.

## 4. Post-traitement YOLOv8 implémenté (`step2/yolov8_post_process.py`)
- Chargement labels COCO (`labels_coco_dataset_80_yolov8.txt`) et initialisation du modèle avec accéleration matérielle si `.nb`.
- `get_results()`:
  - Récupère la sortie unique, `postprocess_yolov8()` transpose en (1344, 84).
  - Pour chaque détection: best class + score; seuil `confidence_threshold`; conversion centre->coins (coordonnées normalisées).
  - NMS maison (`non_max_suppression`) sur boxes [x0, y0, x1, y1, score, class_id].
  - Retourne `boxes`, `classes`, `scores` shapes (1, N, 4), (1, N), (1, N) pour rester compatibles avec l'overlay et le flux GTK.
- `model_type = "ssd_mobilenet_v2"` pour réutiliser la voie d'affichage deja codifiée (mise a l'échelle des coordonnées et étiquettes).

### Procédure
1) **Récuperer la sortie du modèle**
  ```python
  output = self.stai_mpu_model.get_output(index=0)  # 1 x 84 x 1344
  detections = self.postprocess_yolov8(np.squeeze(output))
  ```
  `np.squeeze` enlève la dimension batch et on transpose dedans pour avoir 1344 lignes de détections.

2) **Parcourir chaque détection** (`postprocess_yolov8`)
  ```python
  x_c, y_c, w, h = det[:4]      # centre et taille (normalises 0..1)
  class_scores = det[4:]        # 80 scores de classes
  best_class = np.argmax(class_scores)
  best_score = class_scores[best_class]
  if best_score < self.confidence_threshold: continue  # on jette les faibles
  # conversion centre -> coins
  x0 = x_c - w/2; y0 = y_c - h/2; x1 = x_c + w/2; y1 = y_c + h/2
  candidates.append([x0, y0, x1, y1, best_score, best_class])
  ```
  On garde la classe la plus probable pour chaque détection, on applique un seuil, on convertit en coins pour dessiner une box.

3) **Nettoyer les doublons avec NMS**
  ```python
  final_dets = self.non_max_suppression(candidates, self.iou_threshold)
  ```
  NMS garde la meilleure box quand deux boxes se recouvrent trop (IoU >= seuil).

4) **Adapter le format pour l'overlay**
  ```python
  boxes = np.array(boxes)[np.newaxis, ...]      # (1, N, 4)
  classes = np.array(classes)[np.newaxis, ...]  # (1, N)
  scores = np.array(scores)[np.newaxis, ...]    # (1, N)
  ```
  L'overlay Step 1 attend cette forme; en ajoutant la dimension batch (1), on évite de changer le code d affichage.

5) **Rester compatible avec l'overlay existant**
  ```python
  self.model_type = "ssd_mobilenet_v2"
  ```
  Ce choix force l'overlay à emprunter la branche de rendu déja écrite pour SSD Mobilenet v2 (même rescaling des coords, même coloration), sans refactor UI.

## 5. Adaptations applicatives Step 2

### a. Lanceur (script shell)
- Fichier `step2/launch_python_object_detection.sh` :
  - Appelle le binaire Python Step 2 (`stai_mpu_yolov8_object_detection.py`).
  - Charge le modèle YOLOv8 (`yolov8s_integer_quant_256_fp32_io.nb`) et le fichier de labels YOLOv8 (80 classes COCO).
  - Reprend les variables d'écran/caméra (`DFPS`, `DWIDTH`, `DHEIGHT`, `CAMERA_SRC`) pour que la commande reste identique à Step 1 pour l'utilisateur.

### b. Application principale (GTK/GStreamer)
- Fichier `step2/stai_mpu_yolov8_object_detection.py` :
  - Pipeline vidéo et overlay **inchangés** par rapport à Step 1 (capture via GStreamer, affichage via GTK, overlay dessine les boxes).
  - Seul changement: on instancie `NeuralNetwork` depuis `yolov8_post_process.py` (nouveau post-traitement) à la place de la classe SSD.
  - Les coordonnées restent normalisées; l'overlay fait le rescaling pour l'écran comme avant. Pas besoin de modifier l'UI.

### c. Post-process legacy (reference)
- Fichier `step2/yolov8_object_detection_pp.py` :
  - Ancien squelette base SSD (decode anchors) conserve à titre de référence/fallback.
  - Le lanceur principal ne l'utilise pas; il est là pour comparaison ou debug si besoin.

## 6. Implications clé et couverture des attentes
- **Moins de décodage** : YOLOv8 est anchor-free, donc centre/largeur/hauteur → coins + NMS suffit.
- **80 classes COCO** : labels YOLOv8 chargés, on garde la classe la plus probable pour chaque détection.
- **Overlay réutilisé** : coords restent normalisées, l'overlay fait le rescaling comme en Step 1.
- **Seuils réglables** : `--conf_threshold` / `--iou_threshold` restent accessibles (0.65 / 0.45 par défaut).

## 7. Ce qui a été fait, concretement (Step 2)

### (rappel synthétique)
- Lanceur (`step2/launch_python_object_detection.sh`) mis à jour pour charger YOLOv8 + labels COCO avec les mêmes variables d'écran/caméra que Step 1.
- Appli (`step2/stai_mpu_yolov8_object_detection.py`) : pipeline et overlay inchangés, seule la classe NN importée passe à `yolov8_post_process.py`; l'overlay est réutilisé tel quel.
- Post-process (`step2/yolov8_post_process.py`) : lecture de la sortie unique, meilleure classe + seuil, centre→coins, NMS, format (1, N, 4)/(1, N)/(1, N), `model_type` conservé pour l'affichage.

### d. Ce qui change par rapport au Step 1
- SSD utilisait des anchors à décoder; YOLOv8 est anchor-free: on lit directement centre/largeur/hauteur, on convertit en coins, puis NMS.
- SSD sortait 3 tenseurs (scores, boxes, anchors); YOLOv8 sort un seul tenseur (84 x 1344). On lit juste `get_output(index=0)`.
- Labels: on passe aux labels YOLOv8 (80 classes) pour correspondre a l ordre des scores de sortie.
- Les seuils CLI restent les mêmes (`--conf_threshold`, `--iou_threshold`), donc l'utilisateur règle la sensibilité comme avant.
- La partie affichage/pipeline ne change pas: moins de risques de bugs visuels, transition douce pour l'utilisateur.
