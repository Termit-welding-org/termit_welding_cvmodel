import cv2
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import os

# пути к моделям (лежат у скрипта)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_NANO = os.path.join(BASE_DIR, "best_nano.pt")
PATH_SMALL = os.path.join(BASE_DIR, "best_small.pt")

CLASS_NAMES = ['Geometric defect', 'Non-fusion defect', 'crack', 'porosity', 'spatters']
CONF_THRESH = 0.3
IOU_THRESH = 0.5

def load_models():
    if not os.path.exists(PATH_NANO):
        raise FileNotFoundError(f"Не найден файл {PATH_NANO}")
    if not os.path.exists(PATH_SMALL):
        raise FileNotFoundError(f"Не найден файл {PATH_SMALL}")
    return [YOLO(PATH_NANO), YOLO(PATH_SMALL)]

def predict_ensemble(image_path, models=None):
    if models is None:
        models = load_models()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []
    for model in models:
        results = model(img_rgb, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
        if results[0].boxes is None:
            continue
        boxes = results[0].boxes.xyxy.cpu().numpy() / np.array([w, h, w, h])
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy().astype(int)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
    if not all_boxes:
        return [], [], []
    boxes_f, scores_f, labels_f = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=None, iou_thr=IOU_THRESH, skip_box_thr=0.001
    )
    return boxes_f, scores_f, labels_f

def draw_predictions(image_path, boxes, scores, labels):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255)]
    for box, score, label in zip(boxes, scores, labels):
        label = int(label)
        x1, y1, x2, y2 = (box * np.array([w, h, w, h])).astype(int)
        color = colors[label % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{CLASS_NAMES[label]}: {score:.2f}"
        cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

if __name__ == "__main__":
    models = load_models()
    print("Ансамбль загружен.")
    while True:
        img_path = input("\nВведите путь к изображению (или 'q'): ").strip()
        if img_path.lower() == 'q':
            break
        if not os.path.exists(img_path):
            print("Файл не найден.")
            continue
        boxes, scores, labels = predict_ensemble(img_path, models)
        if len(boxes) == 0:
            print("Дефектов не обнаружено.")
        else:
            result_img = draw_predictions(img_path, boxes, scores, labels)
            cv2.imshow("Ensemble Detection", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            for i, (cls, conf) in enumerate(zip(labels, scores)):
                print(f"{i+1}. {CLASS_NAMES[int(cls)]}: {conf:.2f}")