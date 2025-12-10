#!/usr/bin/python3
#
# YOLOv8 object detection post-processing for STM32MP2
#

from stai_mpu import stai_mpu_network
import numpy as np
from timeit import default_timer as timer


class NeuralNetwork:
    """
    Class that handles YOLOv8 Object Detection inference
    """

    def __init__(self, model_file, label_file, input_mean, input_std,
                 confidence_thresh, iou_threshold):
        """
        :param model_file: .nb model to be executed
        :param label_file: labels file
        :param input_mean: input_mean
        :param input_std: input standard deviation
        :param confidence_thresh: confidence threshold
        :param iou_threshold: IoU threshold for NMS
        """

        self._model_file = model_file
        print("NN model used : ", self._model_file)
        self._label_file = label_file
        self._input_mean = input_mean
        self._input_std = input_std
        self.confidence_threshold = confidence_thresh
        self.iou_threshold = iou_threshold

        # On met un type de modèle (utilisé pour le dessin)
        # NOTE : on pourrait mettre "yolov8_od" et adapter OverlayWindow,
        # mais pour rester simple on réutilise la branche "ssd_mobilenet_v2"
        self.model_type = "ssd_mobilenet_v2"

        # Initialize NN model (HW accel si .nb)
        if self._model_file.endswith(".nb"):
            self.stai_mpu_model = stai_mpu_network(
                model_path=self._model_file,
                use_hw_acceleration=True
            )
        else:
            self.stai_mpu_model = stai_mpu_network(
                model_path=self._model_file,
                use_hw_acceleration=False
            )

        # Input / output infos
        self.num_inputs = self.stai_mpu_model.get_num_inputs()
        self.input_tensor_infos = self.stai_mpu_model.get_input_infos()

        self.num_outputs = self.stai_mpu_model.get_num_outputs()
        self.output_tensor_infos = self.stai_mpu_model.get_output_infos()

        # Labels
        self._labels = self._load_labels(self._label_file)

    # ---------- Utils ----------

    def _load_labels(self, filename):
        labels = []
        with open(filename, 'r') as f:
            for line in f:
                labels.append(line.strip())
        return labels

    def get_labels(self):
        return self._labels

    def get_img_size(self):
        """
        :return: (input_width, input_height, input_channel)
        """
        input_tensor_shape = self.input_tensor_infos[0].get_shape()
        # get_shape() typiquement [1, H, W, 3]
        input_width = input_tensor_shape[1]
        input_height = input_tensor_shape[2]
        input_channel = input_tensor_shape[3]
        return (input_width, input_height, input_channel)

    # ---------- Inference ----------

    def launch_inference(self, img):
        """
        Lance l'inférence sur une image RGB numpy (H, W, 3)
        """
        # add batch dim
        input_data = np.expand_dims(img, axis=0)

        # normalisation si float32
        if self.input_tensor_infos[0].get_dtype() == np.float32:
            input_data = (np.float32(input_data) - self._input_mean) / self._input_std

        # set input & run
        self.stai_mpu_model.set_input(0, input_data)
        start = timer()
        self.stai_mpu_model.run()
        end = timer()
        return end - start

    # ---------- Post-processing ----------

    def get_results(self):
        """
        Récupère la sortie YOLOv8 et applique :
        - filtrage sur confidence
        - conversion centre/largeur/hauteur -> xmin,ymin,xmax,ymax
        - NMS
        Retourne (locations, classes, scores) au format attendu par l'appli.
        """
        # Sortie unique : 1 x C+4 x N (ex: 1 x 84 x 1344 ou 1 x 5 x N pour 1 classe)
        output = self.stai_mpu_model.get_output(index=0)
        detections = self.postprocess_yolov8(np.squeeze(output))

        if len(detections) == 0:
            return np.array([]), np.array([]), np.array([])

        boxes = []
        classes = []
        scores = []
        for det in detections:
            box, cls_id, score = det
            boxes.append(box)
            classes.append(cls_id)
            scores.append(score)

        boxes = np.array(boxes)[np.newaxis, ...]      # (1, N, 4)
        classes = np.array(classes)[np.newaxis, ...]  # (1, N)
        scores = np.array(scores)[np.newaxis, ...]    # (1, N)

        return boxes, classes, scores

    def postprocess_yolov8(self, outputs):
        """
        outputs: (C+4, N) -> transpose to (N, C+4)
        Chaque détection:
          [0..3] : x_center, y_center, w, h   (normalisés 0..1)
          [4..]  : scores par classe (longueur variable selon le modèle)
        """
        output_data = np.transpose(outputs)  # (N, C+4)
        candidates = []

        for det in output_data:
            x_c, y_c, w, h = det[:4]

            class_scores = det[4:]
            if class_scores.size == 0:
                continue
            best_class = int(np.argmax(class_scores))
            best_score = float(class_scores[best_class])

            if best_score < self.confidence_threshold:
                continue

            # YOLO -> box centre -> box coins (toujours normalisé 0..1)
            x0 = x_c - w / 2.0
            y0 = y_c - h / 2.0
            x1 = x_c + w / 2.0
            y1 = y_c + h / 2.0

            # on stocke [x0, y0, x1, y1, score, class_id]
            candidates.append([x0, y0, x1, y1, best_score, best_class])

        final_dets = self.non_max_suppression(candidates, self.iou_threshold)
        results = []
        for det in final_dets:
            x0, y0, x1, y1, score, cls_id = det
            results.append(([x0, y0, x1, y1], int(cls_id), float(score)))
        return results

    # ---------- NMS & IoU ----------

    def intersection(self, a, b):
        """
        a, b: [x0, y0, x1, y1, ...]
        """
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        return inter_w * inter_h

    def union(self, a, b):
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return area_a + area_b - self.intersection(a, b)

    def iou(self, a, b):
        denom = self.union(a, b)
        if denom <= 0:
            return 0.0
        return self.intersection(a, b) / denom

    def non_max_suppression(self, detections, iou_thresh):
        """
        detections: liste de [x0,y0,x1,y1,score,class_id]
        """
        if len(detections) == 0:
            return []

        # tri décroissant par score
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [
                det for det in detections
                if self.iou(det, best) < iou_thresh
            ]
        return keep

    # ---------- Label helper ----------

    def get_label(self, idx, classes):
        """
        Même signature que ssd_mobilenet_pp : labels[classes[0][idx]]
        """
        labels = self.get_labels()
        return labels[int(classes[0][idx])]
