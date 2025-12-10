# Rapport Step 4 - YOLOv8 pingpong + tracking

## 1. Objectifs et périmètre
- Repartir du Step 3 (détection pingpong YOLOv8) et ajouter un suivi simple des objets détectés (ID persistants, trajectoires).
- Utiliser le modèle pingpong en résolution 320 pour améliorer la stabilité : `yolov8n_integer_quant_320_1c_pingpongball_2_fp32_io.nb`.
- Conserver l’affichage des scores et les couleurs par boîte, mais les baser sur l’ID de suivi pour rester cohérent d’une frame à l’autre.

## 2. Modèle, labels et lanceur
- Modèle : `models/yolov8n_integer_quant_320_1c_pingpongball_2_fp32_io.nb` (sortie YOLOv8 : 1 x (4+C) x N, C=1 ici).
- Labels : `models/labels_pingpong_ball.txt` (contenu : `pingpong_ball`).
- Lanceur (`step4/launch_python_object_detection.sh`) mis à jour pour pointer vers l’appli Step 4 et ce modèle/label.
	```sh
	/usr/local/x-linux-ai/workspace/step4/stai_mpu_yolov8_object_detection.py \
		-m /usr/local/x-linux-ai/workspace/models/yolov8n_integer_quant_320_1c_pingpongball_2_fp32_io.nb \
		-l /usr/local/x-linux-ai/workspace/models/labels_pingpong_ball.txt \
		--framerate $DFPS --frame_width $DWIDTH --frame_height $DHEIGHT --camera_src $CAMERA_SRC
	```

## 3. Post-traitement YOLOv8
- Fichier `step4/yolov8_post_process.py` repris du Step 3 :
	- Lecture de la sortie unique, transposition `(C+4, N) -> (N, C+4)`.
	- Sélection de la meilleure classe par détection + seuil de confiance.
	- Conversion centre → coins, puis NMS pour filtrer les boxes qui se chevauchent.
	- Retour aux formats `(1, N, 4)`, `(1, N)`, `(1, N)` pour l’overlay.
- Le `model_type` reste "ssd_mobilenet_v2" pour réutiliser le code d’affichage existant.

## 4. Ajout du suivi (tracking) dans l’appli
Fichier `step4/stai_mpu_yolov8_object_detection.py` : ajout d’un suivi simple par centroïde.

### 4.1 État du tracker
- Variables ajoutées dans `Application.__init__` :
	```python
	self.tracks = {}              # dictionnaire des pistes actives
	self.track_id_counter = 0     # ID auto-incrémenté
	self.max_history = 30         # points de trajectoire mémorisés
	self.max_age = 10             # frames tolérées sans association
	self.match_thresh = 0.12      # distance max (coords normalisées) pour associer
	```

### 4.2 Mise à jour des pistes
- Appel du tracker juste après l’inférence :
	```python
	self.app.nn_result_locations, self.app.nn_result_classes, self.app.nn_result_scores = self.nn.get_results()
	self.app.update_tracks(self.app.nn_result_locations, self.app.nn_result_scores, self.app.nn_result_classes)
	```
- Logique principale (`update_tracks`):
	- Calcule le centroïde de chaque détection.
	- Associe chaque détection au track le plus proche si la distance < `match_thresh`.
	- Met à jour bbox/score/centre/historique et réinitialise l’âge des pistes associées.
	- Vieillit et supprime les pistes non associées au-delà de `max_age`.
	- Crée de nouvelles pistes pour les détections non associées, en leur attribuant une couleur aléatoire stable par ID.

Extrait clé :
```python
for det_idx, bb, cx, cy, sc, cid in detections_data:
		# association
		...
		if best_tid is not None:
				assignments.append((det_idx, best_tid))
...
for det_idx, tid in assignments:
		t = self.tracks[tid]
		t["bbox"] = bb
		t["center"] = (cx, cy)
		t["score"] = sc
		t["class"] = cid
		t["age"] = 0
		t["history"].append((cx, cy))
...
# création de nouvelles pistes
tid = self.track_id_counter; self.track_id_counter += 1
self.tracks[tid] = {"bbox": bb, "center": (cx, cy), "score": sc, "class": cid,
									 "age": 0, "history": deque(maxlen=self.max_history), "color": color}
self.tracks[tid]["history"].append((cx, cy))
```

## 5. Affichage overlay adapté au tracking
- L’overlay ne parcourt plus les detections brutes mais les `tracks` :
	- Bbox recalculée aux dimensions du preview.
	- Couleur basée sur la piste (stable d’une frame à l’autre).
	- Texte : `ID <tid> <label> <score%>` affiché au-dessus de la box.
	- Trajectoire : on dessine une polyline semi-transparente à partir de l’historique des centroïdes.

Extrait :
```python
for tid, track in tracks_items:
		x0, y0, x1, y1 = track["bbox"]
		color = track["color"]
		cr.rectangle(...)
		cr.show_text(f"ID {tid} {label} {int(accuracy)}%")
		history = list(track.get("history", []))
		if len(history) > 1:
				cr.set_source_rgba(color[0], color[1], color[2], 0.7)
				...  # polyline de la trajectoire
```

## 6. Points de réglage
- `match_thresh` (0.12 par défaut) : distance max (coords normalisées) pour associer une détection à une piste. Réduire si des fusions indésirables, augmenter si des IDs se perdent trop vite.
- `max_age` (10 frames) : tolérance avant de supprimer une piste sans association. À augmenter si la balle disparaît brièvement, diminuer pour nettoyer plus vite.
- `max_history` (30) : longueur de la trajectoire dessinée.

## 7. Résultat attendu
- Détection du pingpong ball identique au Step 3, mais avec un ID stable par objet.
- Les boîtes conservent leur couleur et leur ID d’une frame à l’autre tant que l’association reste sous le seuil.
- Une trajectoire (polyline) montre le déplacement récent de l’objet suivi.
