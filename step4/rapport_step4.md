# Rapport Step 4 - YOLOv8 pingpong + tracking ByteTrack

## 1. Objectifs et périmètre
- Reprendre l'appli Step 3 (détection ping-pong) et y ajouter le suivi temps-réel avec ByteTrack (lib `supervision`).
- Affichage demandé : couleur stable par ID, score au-dessus de la boîte, trajectoire courte (mémoire 30 frames).

## 2. Modèle et ressources
- Modèle : `models/yolov8n_integer_quant_256_1c_pingpongball_2_fp32_io.nb` (1 classe, sortie 1 x (4+C) x N).
- Labels : `models/labels_pingpong_ball.txt` (contenu : `pingpong_ball`).
- Lanceur : `step4/launch_python_object_detection.sh` pointe vers l'appli Step 4 et ce modèle/label.

## 3. Post-traitement YOLOv8 (inchangé vs Step 3)
- Fichier : `step4/yolov8_post_process.py` (copie Step 3).
- Pipeline : sortie transposée, meilleure classe + seuil confiance, conversion centre→coins, NMS, format (1,N,4)/(1,N)/(1,N) pour réutiliser l'overlay.
```python
output = self.stai_mpu_model.get_output(index=0)
detections = self.postprocess_yolov8(np.squeeze(output))
output_data = np.transpose(outputs)  # (N, C+4)
best_class = int(np.argmax(class_scores))
if best_score < self.confidence_threshold:
	continue
```

## 4. Suivi ByteTrack
- Fichier : `step4/stai_mpu_yolov8_object_detection.py`.
- Création du tracker (version simple, sans paramètres) :
```python
import supervision as sv
self.byte_tracker = sv.ByteTrack()
```
- Nouvelle méthode dédiée `apply_tracking()` appelée juste après l'inférence pour encapsuler le suivi + le reset en cas d'absence de détection :
```python
self.app.nn_result_locations, self.app.nn_result_classes, self.app.nn_result_scores = self.nn.get_results()
self.app.apply_tracking()
```
- Conversion / ByteTrack / historique (coercition float32 + garde-fou forme) :
```python
boxes_norm = np.array(self.nn_result_locations[0], dtype=np.float32)
if boxes_norm.ndim != 2 or boxes_norm.shape[1] < 4:
	self.tracked_boxes = np.empty((0, 4))
	self.track_history = {}
	return

scale = np.array([self.frame_width, self.frame_height, self.frame_width, self.frame_height], dtype=np.float32)
boxes = boxes_norm[:, :4] * scale

detections = sv.Detections(
	xyxy=np.array(boxes, dtype=np.float32),
	confidence=np.array(self.nn_result_scores[0], dtype=np.float32),
	class_id=np.array(self.nn_result_classes[0], dtype=int)
)

tracked = self.byte_tracker.update_with_detections(detections)
new_history = {}
for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
	tid = int(track_id)
	cx = float((box[0] + box[2]) / 2.0)
	cy = float((box[1] + box[3]) / 2.0)
	history = self.track_history.get(tid, [])
	history.append((cx, cy))
	if len(history) > 30:
		history = history[-30:]
	new_history[tid] = history
self.track_history = new_history
```
  - Si aucune détection ou forme invalide : purge des boîtes/ids/historique pour éviter des exceptions sur la frame courante.

## 5. Overlay GTK : affichage ID, score, trajectoire
- `drawing()` modifiée pour s'appuyer sur les sorties du tracker : palette fixe de 32 couleurs indexée par `track_id` pour stabilité.
- Les boxes tracées utilisent `tracked_boxes/ids/scores` (déjà en pixels), re-scalées vers la zone d'affichage.
- Texte au-dessus : `label #id conf%`.
- Trajectoire : segments reliant les centroïdes mémorisés (histoire max 30 points) avec la même couleur que la boîte.
```python
color_idx = track_id % len(self.bbcolor_list)
cr.rectangle(int(x), int(y), width, height)
cr.move_to(x, (y - (self.ui_cairo_font_size / 2)))
text_to_display = f"{label} #{track_id} {int(accuracy)}%"
cr.show_text(text_to_display)

history = self.app.track_history.get(track_id, [])
for p_idx in range(1, len(history)):
	cr.move_to(x_prev, y_prev)
	cr.line_to(x_curr, y_curr)
cr.stroke()
```

## 6. Lanceur
- `step4/launch_python_object_detection.sh` appelle le script Step 4 et le modèle/label ping-pong :
```sh
/usr/local/x-linux-ai/workspace/step4/stai_mpu_yolov8_object_detection.py \
	-m /usr/local/x-linux-ai/workspace/models/yolov8n_integer_quant_256_1c_pingpongball_2_fp32_io.nb \
	-l /usr/local/x-linux-ai/workspace/models/labels_pingpong_ball.txt \
	--framerate $DFPS --frame_width $DWIDTH --frame_height $DHEIGHT --camera_src $CAMERA_SRC
```