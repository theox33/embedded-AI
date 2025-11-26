# Step 1 - starting point: object detection with ssd_mobilenet model

## Analyse du code source

### Initialisation de l'application

La partie du code qui initialise l'application se trouve dans la classe `Application` du fichier `stai_mpu_object_detection_starting_point.py`, notamment dans le constructeur `__init__` et le bloc principal `if __name__ == '__main__'`. Elle instancie les objets nécessaires, configure les paramètres et lance l'interface graphique.

---
### Création du pipeline GStreamer

La création du pipeline GStreamer est réalisée dans la méthode `camera_dual_pipeline_creation` de la classe `GstWidget`. Cette méthode configure les éléments du pipeline pour la capture vidéo et le traitement par le NN.

---
### Capture de la frame caméra pour l'inférence NN

La capture de la frame caméra pour l'inférence du réseau de neurones est effectuée dans la méthode `new_sample` de la classe `GstWidget`. Elle récupère une image depuis le flux vidéo et la prépare pour l'inférence.

---
### Définition de la taille de la frame preview

La taille de la frame preview est définie par les paramètres `frame_width` et `frame_height` dans la classe `Application` et utilisés lors de la création du pipeline GStreamer (`caps_src`).

---
### Définition de la taille de la frame NN

La taille de la frame utilisée pour le NN est définie par `nn_input_width` et `nn_input_height` dans la classe `Application` et utilisée dans le pipeline GStreamer (`caps_src0`).

---
### Normalisation de la frame

La normalisation de la frame est réalisée dans la méthode `launch_inference` de la classe `NeuralNetwork` (fichier `ssd_mobilenet_pp.py`). Si le type d'entrée est `np.float32`, l'image est normalisée avec la moyenne et l'écart-type.

---
### Exécution de l'inférence NN

L'exécution de l'inférence du réseau de neurones est effectuée dans la méthode `launch_inference` de la classe `NeuralNetwork` (fichier `ssd_mobilenet_pp.py`).

---
### Post-traitement du NN

Le post-traitement des résultats du NN est réalisé dans la méthode `get_results` et la méthode `postprocess_predictions` de la classe `NeuralNetwork` (fichier `ssd_mobilenet_pp.py`).

---
### Affichage des résultats sur l'écran

L'affichage des résultats est géré dans la méthode `drawing` de la classe `OverlayWindow` (fichier `stai_mpu_object_detection_starting_point.py`). Cette méthode dessine les boîtes de détection et les labels sur la vidéo.


