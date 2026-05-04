"""
quick.py - Полное решение для детекции дефектов разных форм
Поддерживает квадраты, круги и прямоугольники
"""

import os
import cv2
import numpy as np
import torch
import random
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ============================================================
# 1. ПРОВЕРКА GPU
# ============================================================
print("=" * 60)
print("ПРОВЕРКА ОБОРУДОВАНИЯ")
print("=" * 60)

if torch.cuda.is_available():
    print(f"[OK] GPU ДОСТУПЕН: {torch.cuda.get_device_name(0)}")
    device = 'cuda'
    print(f"   CUDA версия: {torch.version.cuda}")
    print(f"   PyTorch версия: {torch.__version__}")
else:
    print("[X] GPU НЕ ДОСТУПЕН, используем CPU (будет медленно)")
    device = 'cpu'
    print("   Решение: установите PyTorch с CUDA")

# ============================================================
# 2. СОЗДАНИЕ ДАТАСЕТА С РАЗНЫМИ ДЕФЕКТАМИ
# ============================================================
print("\n" + "=" * 60)
print("СОЗДАНИЕ ДАТАСЕТА С ДЕФЕКТАМИ (КВАДРАТЫ, КРУГИ, ПРЯМОУГОЛЬНИКИ)")
print("=" * 60)


def create_square_defect(img, x, y, size):
    """Создание квадратного дефекта"""
    cv2.rectangle(img, (x, y), (x + size, y + size), (30, 30, 35), -1)
    # Добавляем тень
    cv2.rectangle(img, (x - 3, y - 3), (x + size + 3, y + size + 3), (80, 80, 80), 1)
    return (x, y, x + size, y + size)  # возвращаем bbox


def create_circle_defect(img, x, y, radius):
    """Создание круглого дефекта"""
    cv2.circle(img, (x + radius, y + radius), radius, (30, 30, 35), -1)
    # Добавляем тень
    cv2.circle(img, (x + radius, y + radius), radius + 3, (80, 80, 80), 1)
    # Для YOLO нужен прямоугольник вокруг круга
    return (x, y, x + radius * 2, y + radius * 2)


def create_rectangle_defect(img, x, y, width, height):
    """Создание прямоугольного дефекта"""
    cv2.rectangle(img, (x, y), (x + width, y + height), (30, 30, 35), -1)
    # Добавляем тень
    cv2.rectangle(img, (x - 3, y - 3), (x + width + 3, y + height + 3), (80, 80, 80), 1)
    return (x, y, x + width, y + height)


def create_defect_image_multi(width=640, height=640, num_defects=3):
    """
    Создание изображения с несколькими дефектами разных форм

    Args:
        width, height: размеры изображения
        num_defects: количество дефектов (по умолчанию 3)
    """
    # Металлический фон
    img = np.ones((height, width, 3), dtype=np.uint8) * 160

    # Добавляем градиент
    for i in range(width):
        gradient = int(30 * (i / width))
        img[:, i] = cv2.add(img[:, i], gradient)

    # Добавляем шум
    noise = np.random.normal(0, 8, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Сварной шов (светлая область)
    weld_width = width // 2
    weld_x1 = width // 2 - weld_width // 2
    weld_x2 = width // 2 + weld_width // 2
    weld_y1 = height // 4
    weld_y2 = 3 * height // 4

    # Создаём градиент для сварного шва
    for y in range(weld_y1, weld_y2):
        brightness = 200 - int(30 * abs(y - (weld_y1 + weld_y2) // 2) / ((weld_y2 - weld_y1) // 2))
        cv2.rectangle(img, (weld_x1, y), (weld_x2, y + 1), (brightness, brightness, brightness), -1)

    bboxes = []
    defect_types = []

    # Создаём дефекты
    for i in range(num_defects):
        # Случайный тип дефекта
        defect_type = random.choice(['square', 'circle', 'rectangle'])
        defect_types.append(defect_type)

        # Случайная позиция в области сварного шва
        x = random.randint(weld_x1 + 20, weld_x2 - 80)
        y = random.randint(weld_y1 + 20, weld_y2 - 80)

        if defect_type == 'square':
            size = random.randint(30, 70)
            bbox = create_square_defect(img, x, y, size)

        elif defect_type == 'circle':
            radius = random.randint(20, 40)
            bbox = create_circle_defect(img, x, y, radius)

        else:  # rectangle
            rect_w = random.randint(40, 80)
            rect_h = random.randint(25, 60)
            bbox = create_rectangle_defect(img, x, y, rect_w, rect_h)

        bboxes.append(bbox)

    return img, bboxes, defect_types


def create_dataset_multi(num_images=500, dataset_dir='data_multi'):
    """
    Создание датасета с несколькими дефектами на изображении

    Args:
        num_images: количество изображений
        dataset_dir: директория для сохранения
    """
    # Создаём структуру директорий
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{dataset_dir}/images/{split}', exist_ok=True)
        os.makedirs(f'{dataset_dir}/labels/{split}', exist_ok=True)

    print(f"Создание {num_images} изображений с множественными дефектами...")

    for i in range(num_images):
        # Количество дефектов на изображении (от 1 до 5)
        num_defects = random.randint(1, 5)

        # Создаём изображение
        img, bboxes, defect_types = create_defect_image_multi(num_defects=num_defects)

        # Сохраняем изображение
        if i < int(num_images * 0.7):  # 70% train
            split = 'train'
        elif i < int(num_images * 0.85):  # 15% val
            split = 'val'
        else:  # 15% test
            split = 'test'

        img_path = f'{dataset_dir}/images/{split}/defect_{i:04d}.jpg'
        cv2.imwrite(img_path, img)

        # Создаём YOLO аннотацию для всех дефектов
        h, w = img.shape[:2]
        label_path = f'{dataset_dir}/labels/{split}/defect_{i:04d}.txt'

        with open(label_path, 'w') as f:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox

                # Конвертация в YOLO формат
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width_norm = (x2 - x1) / w
                height_norm = (y2 - y1) / h

                # Класс 0 = дефект (все типы дефектов - один класс)
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        if i % 100 == 0:
            print(f"  Создано {i}/{num_images} изображений...")

    print(f"[OK] Датасет создан: {num_images} изображений")
    print(f"   train: {len(os.listdir(f'{dataset_dir}/images/train'))} изображений")
    print(f"   val: {len(os.listdir(f'{dataset_dir}/images/val'))} изображений")
    print(f"   test: {len(os.listdir(f'{dataset_dir}/images/test'))} изображений")

    return dataset_dir


# ============================================================
# 3. СОЗДАНИЕ КОНФИГУРАЦИОННОГО ФАЙЛА DATA.YAML
# ============================================================
def create_data_yaml(dataset_dir='data_multi'):
    """Создание data.yaml файла для YOLO"""

    data_yaml_content = f"""
# Dataset for weld defect detection (multi defects)
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 1

# Class names
names: ['defect']
"""

    data_yaml_path = 'data_multi.yaml'
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)

    print(f"[OK] Создан файл конфигурации: {data_yaml_path}")
    return data_yaml_path


# ============================================================
# 4. ВИЗУАЛИЗАЦИЯ ДАТАСЕТА
# ============================================================
def visualize_multi_defects(dataset_dir='data_multi', num_samples=3):
    """Визуализация изображений с множественными дефектами"""

    print(f"\nВизуализация {num_samples} примеров с несколькими дефектами...")

    train_images = sorted(os.listdir(f'{dataset_dir}/images/train'))[:num_samples]

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    if num_samples == 1:
        axes = [axes]

    for idx, img_name in enumerate(train_images):
        img_path = f'{dataset_dir}/images/train/{img_name}'
        label_path = f'{dataset_dir}/labels/train/{img_name.replace(".jpg", ".txt")}'

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Рисуем все дефекты из аннотации
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                print(f"  {img_name}: {len(lines)} дефектов")

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_c, y_c, box_w, box_h = map(float, parts)

                        # Конвертируем в пиксели
                        x1 = int((x_c - box_w / 2) * w)
                        y1 = int((y_c - box_h / 2) * h)
                        x2 = int((x_c + box_w / 2) * w)
                        y2 = int((y_c + box_h / 2) * h)

                        # Рисуем прямоугольник
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(img, f'DEFECT', (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        axes[idx].imshow(img)
        axes[idx].set_title(f'Image {idx + 1}')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('dataset_multi_visualization.png')
    print(f"[OK] Визуализация сохранена: dataset_multi_visualization.png")
    plt.show()


# ============================================================
# 5. ОБУЧЕНИЕ МОДЕЛИ
# ============================================================
def train_model(data_yaml='data_multi.yaml', epochs=5):
    """Обучение модели YOLO для детекции множественных дефектов"""

    print("\n" + "=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 60)

    print("Загрузка модели YOLOv8n...")
    model = YOLO('yolov8n.pt')

    training_params = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': 16,
        'imgsz': 640,
        'device': device,
        'workers': 4 if device == 'cuda' else 0,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,

        # Аугментации
        'hsv_h': 0.02,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10,
        'translate': 0.1,
        'scale': 0.5,
        'flipud': 0.1,
        'fliplr': 0.5,
        'mosaic': 0.5,
        'mixup': 0.2,

        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'verbose': True,
        'seed': 42
    }

    print(f"Параметры обучения:")
    print(f"  - Эпохи: {epochs}")
    print(f"  - Batch size: {training_params['batch']}")
    print(f"  - Устройство: {training_params['device']}")

    print("\nЗАПУСК ОБУЧЕНИЯ...")
    results = model.train(**training_params)

    print("[OK] ОБУЧЕНИЕ ЗАВЕРШЕНО")
    return model


# ============================================================
# 6. ТЕСТОВЫЙ ПРИМЕР С НЕСКОЛЬКИМИ ДЕФЕКТАМИ
# ============================================================
def create_test_multi_defects():
    """Создание специального тестового изображения с несколькими дефектами"""

    print("\n" + "=" * 60)
    print("СОЗДАНИЕ ТЕСТОВОГО ИЗОБРАЖЕНИЯ С 3 ДЕФЕКТАМИ")
    print("=" * 60)

    # Создаём изображение 800x600 для лучшей видимости
    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 180

    # Добавляем градиент
    for i in range(width):
        gradient = int(20 * (i / width))
        img[:, i] = cv2.add(img[:, i], gradient)

    # Добавляем шум
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Сварной шов
    weld_x1 = 100
    weld_x2 = width - 100
    weld_y1 = 100
    weld_y2 = height - 100

    for y in range(weld_y1, weld_y2):
        brightness = 210 - int(20 * abs(y - (weld_y1 + weld_y2) // 2) / ((weld_y2 - weld_y1) // 2))
        cv2.rectangle(img, (weld_x1, y), (weld_x2, y + 1), (brightness, brightness, brightness), -1)

    # Дефект 1: Квадрат (слева)
    cv2.rectangle(img, (150, 200), (220, 270), (20, 20, 25), -1)
    cv2.rectangle(img, (147, 197), (223, 273), (80, 80, 80), 2)
    cv2.putText(img, "SQUARE", (150, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Дефект 2: Круг (центр)
    center_x, center_y = 400, 350
    radius = 45
    cv2.circle(img, (center_x, center_y), radius, (20, 20, 25), -1)
    cv2.circle(img, (center_x, center_y), radius + 3, (80, 80, 80), 2)
    cv2.putText(img, "CIRCLE", (center_x - 35, center_y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Дефект 3: Прямоугольник (справа)
    cv2.rectangle(img, (580, 250), (680, 320), (20, 20, 25), -1)
    cv2.rectangle(img, (577, 247), (683, 323), (80, 80, 80), 2)
    cv2.putText(img, "RECTANGLE", (580, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Добавляем подписи
    cv2.putText(img, "TEST IMAGE WITH 3 DEFECTS", (width // 2 - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(img, "Square | Circle | Rectangle", (width // 2 - 180, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Сохраняем
    cv2.imwrite('test_multi_defects.jpg', img)
    print("[OK] Создано: test_multi_defects.jpg")
    print("  - Квадрат (слева, размер 70x70)")
    print("  - Круг (центр, радиус 45)")
    print("  - Прямоугольник (справа, размер 100x70)")

    # Создаём аннотацию для этого тестового изображения (для проверки)
    h, w = img.shape[:2]

    # BBox для квадрата: (150, 200) - (220, 270)
    square_bbox = (150, 200, 220, 270)
    # BBox для круга: (355, 305) - (445, 395) [при радиусе 45]
    circle_bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
    # BBox для прямоугольника: (580, 250) - (680, 320)
    rect_bbox = (580, 250, 680, 320)

    bboxes = [square_bbox, circle_bbox, rect_bbox]

    # Сохраняем аннотацию в YOLO формате
    with open('test_multi_defects.txt', 'w') as f:
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            width_norm = (x2 - x1) / w
            height_norm = (y2 - y1) / h
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

    print("[OK] Создана аннотация: test_multi_defects.txt")

    return img, bboxes


# ============================================================
# 7. ТЕСТИРОВАНИЕ МОДЕЛИ НА МНОЖЕСТВЕННЫХ ДЕФЕКТАХ
# ============================================================
def test_multi_defects(model_path='runs/detect/train/weights/best.pt'):
    """Тестирование модели на изображении с 3 дефектами"""

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ НА ИЗОБРАЖЕНИИ С 3 ДЕФЕКТАМИ")
    print("=" * 60)

    # Проверяем существование тестового изображения
    if not os.path.exists('test_multi_defects.jpg'):
        create_test_multi_defects()

    # Загружаем модель
    if os.path.exists(model_path):
        print(f"Загрузка модели: {model_path}")
        model = YOLO(model_path)
    else:
        print(f"Модель не найдена: {model_path}")
        import glob
        models = glob.glob('runs/detect/*/weights/best.pt')
        if models:
            model = YOLO(models[-1])
            print(f"Загружена: {models[-1]}")
        else:
            print("[X] Нет обученной модели! Сначала обучите модель.")
            return None, None

    # Тестируем с разными порогами
    results_dict = {}

    for conf_threshold in [0.1, 0.15, 0.2, 0.25, 0.3]:
        print(f"\nТестирование с confidence = {conf_threshold}")

        results = model.predict(
            'test_multi_defects.jpg',
            conf=conf_threshold,
            iou=0.3,
            augment=True,
            verbose=False
        )

        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                print(f"  [OK] НАЙДЕНО дефектов: {len(boxes)}")
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    xyxy = box.xyxy[0].tolist()
                    print(f"     Дефект {i + 1}: уверенность={conf:.3f}, позиция=({int(xyxy[0])}, {int(xyxy[1])})")

                # Сохраняем результат
                annotated = r.plot()
                output_path = f'test_multi_result_conf_{conf_threshold}.jpg'
                cv2.imwrite(output_path, annotated)
                print(f"  Сохранено: {output_path}")
                results_dict[conf_threshold] = (len(boxes), annotated)
            else:
                print(f"  [X] Дефекты НЕ НАЙДЕНЫ (conf={conf_threshold})")
                results_dict[conf_threshold] = (0, None)

    # Показываем лучший результат
    best_conf = None
    best_count = 0
    for conf, (count, _) in results_dict.items():
        if count > best_count:
            best_count = count
            best_conf = conf

    if best_conf and results_dict[best_conf][1] is not None:
        print(f"\nЛучший результат при confidence = {best_conf}: найдено {best_count} дефектов")
        cv2.imshow('Detection Result', results_dict[best_conf][1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return model, results_dict


# ============================================================
# 8. ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ
# ============================================================
def visualize_comparison():
    """Визуализация сравнения оригинального изображения и результатов детекции"""

    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ")
    print("=" * 60)

    if not os.path.exists('test_multi_defects.jpg'):
        print("Тестовое изображение не найдено")
        return

    # Загружаем оригинал
    original = cv2.imread('test_multi_defects.jpg')
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Ищем лучший результат
    import glob
    result_files = glob.glob('test_multi_result_conf_*.jpg')

    if result_files:
        # Берём результат с наименьшим confidence (обычно находит больше)
        best_result_file = min(result_files, key=lambda x: float(x.split('_')[-1].replace('.jpg', '')))
        result = cv2.imread(best_result_file)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Сравнение
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].imshow(original)
        axes[0].set_title('Оригинальное изображение\n(3 дефекта: квадрат, круг, прямоугольник)')
        axes[0].axis('off')

        axes[1].imshow(result)
        axes[1].set_title(f'Результат детекции\n{best_result_file}')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig('comparison_result.png')
        print("[OK] Сравнение сохранено: comparison_result.png")
        plt.show()
    else:
        print("Результаты детекции не найдены")


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
def main():
    """Главная функция"""

    print("=" * 60)
    print("СИСТЕМА ДЕТЕКЦИИ ДЕФЕКТОВ (КВАДРАТЫ, КРУГИ, ПРЯМОУГОЛЬНИКИ)")
    print("=" * 60)

    print("\nВыберите действие:")
    print("1. Полный цикл (создание датасета + обучение + тестирование)")
    print("2. Только создание тестового примера с 3 дефектами")
    print("3. Только тестирование существующей модели на 3 дефектах")
    print("4. Создание датасета + обучение")

    choice = input("\nВаш выбор (1-4): ").strip()

    if choice == '1':
        # Полный цикл
        dataset_dir = create_dataset_multi(num_images=500)
        create_data_yaml(dataset_dir)
        visualize_multi_defects(num_samples=3)
        model = train_model(epochs=5)
        create_test_multi_defects()
        test_multi_defects()
        visualize_comparison()

    elif choice == '2':
        # Только создание тестового примера
        create_test_multi_defects()
        print("\n[OK] Тестовое изображение с 3 дефектами создано")
        print("  Файлы: test_multi_defects.jpg, test_multi_defects.txt")

    elif choice == '3':
        # Только тестирование
        test_multi_defects()
        visualize_comparison()

    elif choice == '4':
        # Создание датасета и обучение
        dataset_dir = create_dataset_multi(num_images=500)
        create_data_yaml(dataset_dir)
        visualize_multi_defects(num_samples=3)
        model = train_model(epochs=5)

    else:
        print("Неверный выбор")

    print("\n" + "=" * 60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    main()