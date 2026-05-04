"""
quick.py - ПРОДВИНУТАЯ ДЕТЕКЦИЯ ДЕФЕКТОВ
Сложные тестовые примеры: дефекты на дефектах, перекрытия, градиенты
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
else:
    print("[X] GPU НЕ ДОСТУПЕН, используем CPU")
    device = 'cpu'


# ============================================================
# 2. СОЗДАНИЕ СЛОЖНЫХ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ
# ============================================================

def create_complex_defect_image_1():
    """
    ТЕСТ 1: Матрёшка - дефект внутри дефекта внутри дефекта
    5 дефектов, вложенных друг в друга
    """
    width, height = 800, 800
    img = np.ones((height, width, 3), dtype=np.uint8) * 200

    # Металлическая текстура
    for i in range(width):
        gradient = int(15 * (i / width))
        img[:, i] = cv2.add(img[:, i], gradient)

    # Сварной шов
    weld_region = np.ones((height - 100, width - 100, 3), dtype=np.uint8) * 180
    img[50:height - 50, 50:width - 50] = weld_region

    # Дефект 1: Большой прямоугольник (самый внешний)
    cv2.rectangle(img, (150, 150), (650, 650), (100, 100, 100), -1)
    cv2.rectangle(img, (150, 150), (650, 650), (50, 50, 50), 3)
    cv2.putText(img, "DEFECT 1: OUTER RECT", (160, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Дефект 2: Средний квадрат (внутри первого)
    cv2.rectangle(img, (250, 250), (550, 550), (60, 60, 60), -1)
    cv2.rectangle(img, (250, 250), (550, 550), (30, 30, 30), 3)
    cv2.putText(img, "DEFECT 2: MID SQUARE", (260, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Дефект 3: Круг (внутри квадрата)
    cv2.circle(img, (400, 400), 80, (40, 40, 40), -1)
    cv2.circle(img, (400, 400), 80, (20, 20, 20), 3)
    cv2.putText(img, "DEFECT 3: CIRCLE", (400 - 50, 400 - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Дефект 4: Маленький прямоугольник (внутри круга)
    cv2.rectangle(img, (370, 370), (430, 430), (25, 25, 25), -1)
    cv2.rectangle(img, (370, 370), (430, 430), (10, 10, 10), 2)
    cv2.putText(img, "DEFECT 4: INNER RECT", (372, 362), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Дефект 5: Точка/квадратик (центр матрёшки)
    cv2.rectangle(img, (395, 395), (405, 405), (10, 10, 10), -1)
    cv2.putText(img, "DEFECT 5: CENTER", (396, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Добавляем подписи
    cv2.putText(img, "TEST 1: RUSSIAN DOLL (5 nested defects)", (width // 2 - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img, [
        (150, 150, 650, 650),  # Defect 1
        (250, 250, 550, 550),  # Defect 2
        (320, 320, 480, 480),  # Defect 3 (circle bbox)
        (370, 370, 430, 430),  # Defect 4
        (395, 395, 405, 405)  # Defect 5
    ]


def create_complex_defect_image_2():
    """
    ТЕСТ 2: Шахматная доска дефектов - 9 дефектов в сетке 3x3
    Чередующиеся типы: квадрат, круг, прямоугольник
    """
    width, height = 800, 800
    img = np.ones((height, width, 3), dtype=np.uint8) * 190

    # Добавляем шум
    noise = np.random.normal(0, 3, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Сварной шов - фон
    for y in range(100, 700):
        brightness = 210 - int(15 * abs(y - 400) / 300)
        cv2.rectangle(img, (50, y), (750, y + 1), (brightness, brightness, brightness), -1)

    bboxes = []
    cell_size = 200
    start_x, start_y = 100, 100

    defect_types = ['square', 'circle', 'rectangle', 'square', 'circle', 'rectangle', 'square', 'circle', 'rectangle']

    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            x = start_x + j * cell_size
            y = start_y + i * cell_size
            defect_type = defect_types[idx]

            if defect_type == 'square':
                size = 80
                cv2.rectangle(img, (x, y), (x + size, y + size), (30, 30, 35), -1)
                cv2.rectangle(img, (x, y), (x + size, y + size), (10, 10, 10), 2)
                bboxes.append((x, y, x + size, y + size))
                cv2.putText(img, "SQ", (x + 25, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            elif defect_type == 'circle':
                radius = 45
                cx, cy = x + 50, y + 50
                cv2.circle(img, (cx, cy), radius, (30, 30, 35), -1)
                cv2.circle(img, (cx, cy), radius, (10, 10, 10), 2)
                bboxes.append((cx - radius, cy - radius, cx + radius, cy + radius))
                cv2.putText(img, "CI", (cx - 20, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            else:  # rectangle
                rect_w, rect_h = 100, 60
                cv2.rectangle(img, (x, y), (x + rect_w, y + rect_h), (30, 30, 35), -1)
                cv2.rectangle(img, (x, y), (x + rect_w, y + rect_h), (10, 10, 10), 2)
                bboxes.append((x, y, x + rect_w, y + rect_h))
                cv2.putText(img, "RC", (x + 30, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.putText(img, "TEST 2: CHESSBOARD (9 defects - square/circle/rectangle)", (width // 2 - 280, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img, bboxes


def create_complex_defect_image_3():
    """
    ТЕСТ 3: Цепная реакция - дефекты, соединённые линиями (6 дефектов)
    Разные формы, соединённые в композицию
    """
    width, height = 800, 800
    img = np.ones((height, width, 3), dtype=np.uint8) * 170

    # Градиентный фон
    for i in range(width):
        gradient = int(25 * (i / width))
        img[:, i] = cv2.add(img[:, i], gradient)

    # Сварной шов - сложная форма
    pts = np.array([[100, 200], [700, 200], [650, 600], [150, 600]], np.int32)
    cv2.fillPoly(img, [pts], (190, 190, 190))
    cv2.polylines(img, [pts], True, (150, 150, 150), 2)

    # Дефект 1: Большой треугольник (сварочный дефект)
    triangle_pts = np.array([[200, 300], [350, 500], [150, 500]], np.int32)
    cv2.fillPoly(img, [triangle_pts], (40, 40, 45))
    cv2.polylines(img, [triangle_pts], True, (10, 10, 10), 2)
    bbox1 = (150, 300, 350, 500)
    cv2.putText(img, "DEFECT 1: TRIANGLE", (160, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Дефект 2: Круг внутри треугольника
    cv2.circle(img, (250, 420), 40, (25, 25, 30), -1)
    cv2.circle(img, (250, 420), 40, (10, 10, 10), 2)
    bbox2 = (210, 380, 290, 460)
    cv2.putText(img, "DEFECT 2", (220, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Дефект 3: Прямоугольник (справа)
    cv2.rectangle(img, (500, 350), (620, 450), (35, 35, 40), -1)
    cv2.rectangle(img, (500, 350), (620, 450), (10, 10, 10), 2)
    bbox3 = (500, 350, 620, 450)
    cv2.putText(img, "DEFECT 3: RECT", (510, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Дефект 4: Ромб (внутри прямоугольника)
    rhombus_pts = np.array([[560, 380], [590, 400], [560, 420], [530, 400]], np.int32)
    cv2.fillPoly(img, [rhombus_pts], (20, 20, 25))
    cv2.polylines(img, [rhombus_pts], True, (5, 5, 5), 1)
    bbox4 = (530, 380, 590, 420)

    # Дефект 5: Маленький квадрат (соединён линией)
    cv2.rectangle(img, (450, 550), (490, 590), (30, 30, 35), -1)
    cv2.rectangle(img, (450, 550), (490, 590), (10, 10, 10), 2)
    bbox5 = (450, 550, 490, 590)
    cv2.line(img, (470, 550), (470, 500), (80, 80, 80), 2)
    cv2.putText(img, "DEFECT 5", (452, 542), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Дефект 6: Звезда (сложная форма)
    center_x, center_y = 600, 580
    star_pts = []
    for i in range(5):
        angle = i * 72 - 90
        rad = angle * np.pi / 180
        x1 = center_x + 30 * np.cos(rad)
        y1 = center_y + 30 * np.sin(rad)
        x2 = center_x + 15 * np.cos(rad + 36 * np.pi / 180)
        y2 = center_y + 15 * np.sin(rad + 36 * np.pi / 180)
        star_pts.append([int(x1), int(y1)])
        star_pts.append([int(x2), int(y2)])
    star_pts = np.array(star_pts, np.int32)
    cv2.fillPoly(img, [star_pts], (35, 35, 40))
    cv2.polylines(img, [star_pts], True, (10, 10, 10), 1)
    bbox6 = (570, 550, 630, 610)
    cv2.putText(img, "DEFECT 6: STAR", (572, 542), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.putText(img, "TEST 3: CHAIN REACTION (6 connected defects)", (width // 2 - 230, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img, [bbox1, bbox2, bbox3, bbox4, bbox5, bbox6]


def create_complex_defect_image_4():
    """
    ТЕСТ 4: Хаос - 12 случайных дефектов с перекрытиями
    """
    width, height = 800, 800
    img = np.ones((height, width, 3), dtype=np.uint8) * 160

    # Текстурированный фон
    for y in range(0, height, 10):
        cv2.line(img, (0, y), (width, y), (170, 170, 170), 1)
    for x in range(0, width, 10):
        cv2.line(img, (x, 0), (x, height), (170, 170, 170), 1)

    # Сварной шов - волнистая область
    for x in range(100, 700):
        y1 = int(200 + 50 * np.sin(x / 50))
        y2 = int(600 + 30 * np.cos(x / 40))
        cv2.line(img, (x, y1), (x, y2), (190, 190, 190), 2)

    bboxes = []
    colors = [(20, 20, 25), (25, 25, 30), (30, 30, 35), (35, 35, 40)]

    # Создаём 12 случайных дефектов
    for i in range(12):
        defect_type = random.choice(['square', 'circle', 'rectangle', 'ellipse', 'diamond'])
        x = random.randint(100, 650)
        y = random.randint(150, 650)

        if defect_type == 'square':
            size = random.randint(30, 70)
            cv2.rectangle(img, (x, y), (x + size, y + size), random.choice(colors), -1)
            cv2.rectangle(img, (x, y), (x + size, y + size), (5, 5, 5), 1)
            bboxes.append((x, y, x + size, y + size))

        elif defect_type == 'circle':
            radius = random.randint(20, 45)
            cv2.circle(img, (x + radius, y + radius), radius, random.choice(colors), -1)
            cv2.circle(img, (x + radius, y + radius), radius, (5, 5, 5), 1)
            bboxes.append((x, y, x + radius * 2, y + radius * 2))

        elif defect_type == 'rectangle':
            rect_w = random.randint(40, 90)
            rect_h = random.randint(30, 60)
            cv2.rectangle(img, (x, y), (x + rect_w, y + rect_h), random.choice(colors), -1)
            cv2.rectangle(img, (x, y), (x + rect_w, y + rect_h), (5, 5, 5), 1)
            bboxes.append((x, y, x + rect_w, y + rect_h))

        elif defect_type == 'ellipse':
            axes_w = random.randint(30, 60)
            axes_h = random.randint(20, 50)
            cv2.ellipse(img, (x + axes_w, y + axes_h), (axes_w, axes_h), 0, 0, 360, random.choice(colors), -1)
            cv2.ellipse(img, (x + axes_w, y + axes_h), (axes_w, axes_h), 0, 0, 360, (5, 5, 5), 1)
            bboxes.append((x, y, x + axes_w * 2, y + axes_h * 2))

        else:  # diamond
            center_x, center_y = x + 40, y + 40
            diamond_pts = np.array([
                [center_x, center_y - 40],
                [center_x + 40, center_y],
                [center_x, center_y + 40],
                [center_x - 40, center_y]
            ], np.int32)
            cv2.fillPoly(img, [diamond_pts], random.choice(colors))
            cv2.polylines(img, [diamond_pts], True, (5, 5, 5), 1)
            min_x = min(diamond_pts[:, 0])
            max_x = max(diamond_pts[:, 0])
            min_y = min(diamond_pts[:, 1])
            max_y = max(diamond_pts[:, 1])
            bboxes.append((min_x, min_y, max_x, max_y))

    cv2.putText(img, "TEST 4: CHAOS (12 overlapping random defects)", (width // 2 - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img, bboxes


def create_complex_defect_image_5():
    """
    ТЕСТ 5: Градиентные дефекты - 8 дефектов с разной интенсивностью
    От почти белого до чёрного
    """
    width, height = 800, 800
    img = np.ones((height, width, 3), dtype=np.uint8) * 200

    # Гладкий градиент
    for i in range(width):
        gradient = int(30 * (i / width))
        img[:, i] = cv2.subtract(img[:, i], gradient)

    bboxes = []
    intensities = [10, 30, 50, 70, 90, 110, 130, 150]

    for i, intensity in enumerate(intensities):
        x = 100 + (i % 4) * 150
        y = 150 + (i // 4) * 300

        # Квадрат с градиентом внутри
        for j in range(80):
            grad_intensity = intensity + int(20 * (j / 80))
            grad_intensity = min(255, grad_intensity)
            cv2.rectangle(img, (x + j, y), (x + j + 1, y + 80), (grad_intensity, grad_intensity, grad_intensity), -1)

        cv2.rectangle(img, (x, y), (x + 80, y + 80), (5, 5, 5), 2)
        bboxes.append((x, y, x + 80, y + 80))
        cv2.putText(img, f"LVL {i + 1}: {intensity}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.putText(img, "TEST 5: GRADIENT DEFECTS (8 intensity levels from white to black)",
                (width // 2 - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

    return img, bboxes


# ============================================================
# 3. СОЗДАНИЕ ВСЕХ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ
# ============================================================

def create_all_test_images():
    """Создание всех сложных тестовых изображений"""

    print("\n" + "=" * 60)
    print("СОЗДАНИЕ СЛОЖНЫХ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ")
    print("=" * 60)

    tests = [
        ("test_1_russian_doll", create_complex_defect_image_1, "5 nested defects (matryoshka)"),
        ("test_2_chessboard", create_complex_defect_image_2, "9 defects in 3x3 grid"),
        ("test_3_chain", create_complex_defect_image_3, "6 connected defects with different shapes"),
        ("test_4_chaos", create_complex_defect_image_4, "12 random overlapping defects"),
        ("test_5_gradient", create_complex_defect_image_5, "8 gradient intensity defects")
    ]

    test_images = {}

    for name, creator, description in tests:
        print(f"\nСоздание: {name}")
        print(f"  Описание: {description}")
        img, bboxes = creator()
        cv2.imwrite(f'{name}.jpg', img)

        # Сохраняем аннотацию
        h, w = img.shape[:2]
        with open(f'{name}.txt', 'w') as f:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width_norm = (x2 - x1) / w
                height_norm = (y2 - y1) / h
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        test_images[name] = {
            'path': f'{name}.jpg',
            'bboxes': bboxes,
            'count': len(bboxes),
            'description': description
        }
        print(f"  [OK] Создано: {name}.jpg ({len(bboxes)} дефектов)")

    return test_images


# ============================================================
# 4. ТЕСТИРОВАНИЕ МОДЕЛИ НА ВСЕХ СЛОЖНЫХ ПРИМЕРАХ
# ============================================================

def test_all_complex_images(model_path='runs/detect/train/weights/best.pt'):
    """Тестирование модели на всех сложных изображениях"""

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ НА СЛОЖНЫХ ИЗОБРАЖЕНИЯХ")
    print("=" * 60)

    # Проверяем существование тестовых изображений
    test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.jpg')]

    if not test_files:
        print("Тестовые изображения не найдены. Создаю...")
        test_images = create_all_test_images()
    else:
        test_images = {}
        for f in test_files:
            name = f.replace('.jpg', '')
            test_images[name] = {'path': f, 'count': 0, 'description': ''}

    # Загружаем модель
    if os.path.exists(model_path):
        print(f"Загрузка модели: {model_path}")
        model = YOLO(model_path)
    else:
        import glob
        models = glob.glob('runs/detect/*/weights/best.pt')
        if models:
            model = YOLO(models[-1])
            print(f"Загружена: {models[-1]}")
        else:
            print("[X] Нет обученной модели! Сначала обучите модель.")
            return None

    results_summary = []

    for name, info in test_images.items():
        print(f"\n" + "=" * 50)
        print(f"ТЕСТ: {name}")
        print(f"  {info.get('description', '')}")
        print("=" * 50)

        img_path = info['path']
        expected_count = info.get('count', 0)

        # Тестируем с разными порогами
        best_result = None
        best_conf = None
        best_count = 0

        for conf_threshold in [0.05, 0.1, 0.15, 0.2, 0.25]:
            results = model.predict(
                img_path,
                conf=conf_threshold,
                iou=0.3,
                augment=True,
                verbose=False
            )

            for r in results:
                boxes = r.boxes
                detected_count = len(boxes) if boxes is not None else 0

                if detected_count > best_count:
                    best_count = detected_count
                    best_conf = conf_threshold
                    best_result = r.plot()

                if detected_count > 0:
                    print(f"  conf={conf_threshold}: найдено {detected_count} дефектов")

                    if boxes is not None:
                        for i, box in enumerate(boxes):
                            conf = float(box.conf[0])
                            print(f"    Дефект {i + 1}: уверенность={conf:.3f}")

        # Сохраняем лучший результат
        if best_result is not None:
            output_path = f'result_{name}.jpg'
            cv2.imwrite(output_path, best_result)
            print(f"\n[OK] Лучший результат: conf={best_conf}, найдено {best_count} дефектов")
            print(f"  Сохранено: {output_path}")

            results_summary.append({
                'name': name,
                'expected': expected_count,
                'detected': best_count,
                'confidence': best_conf,
                'success_rate': (best_count / expected_count * 100) if expected_count > 0 else 0
            })
        else:
            print(f"\n[X] Дефекты не найдены ни при каком пороге")
            results_summary.append({
                'name': name,
                'expected': expected_count,
                'detected': 0,
                'confidence': None,
                'success_rate': 0
            })

    # Выводим сводку
    print("\n" + "=" * 60)
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 60)

    for res in results_summary:
        status = "[OK]" if res['detected'] >= res['expected'] * 0.7 else "[WARN]"
        print(f"{status} {res['name']}: {res['detected']}/{res['expected']} дефектов ({res['success_rate']:.1f}%)")

    return results_summary


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ КОЛЛАЖА
# ============================================================

def create_collage():
    """Создание коллажа из всех тестовых изображений"""

    print("\n" + "=" * 60)
    print("СОЗДАНИЕ КОЛЛАЖА")
    print("=" * 60)

    test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.jpg')]
    result_files = [f for f in os.listdir('.') if f.startswith('result_test_') and f.endswith('.jpg')]

    if not test_files:
        print("Тестовые изображения не найдены")
        return

    # Создаём коллаж 2x3
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, test_file in enumerate(sorted(test_files)[:6]):
        # Оригинал
        original = cv2.imread(test_file)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Результат (если есть)
        result_file = f"result_{test_file.replace('.jpg', '')}.jpg"
        if os.path.exists(result_file):
            result = cv2.imread(result_file)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            combined = np.hstack([original, result])
            title = f"{test_file} | RESULT"
        else:
            combined = original
            title = test_file

        axes[idx].imshow(combined)
        axes[idx].set_title(title, fontsize=10)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('test_collage.png', dpi=150)
    print("[OK] Коллаж сохранён: test_collage.png")
    plt.show()


# ============================================================
# 6. ИНТЕРАКТИВНЫЙ ПРОСМОТР
# ============================================================

def interactive_view():
    """Интерактивный просмотр всех тестовых изображений"""

    print("\n" + "=" * 60)
    print("ИНТЕРАКТИВНЫЙ ПРОСМОТР")
    print("=" * 60)

    test_files = sorted([f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.jpg')])

    if not test_files:
        print("Тестовые изображения не найдены")
        return

    print("\nДоступные тесты:")
    for i, f in enumerate(test_files):
        result_file = f"result_{f.replace('.jpg', '')}.jpg"
        has_result = "[Есть результат]" if os.path.exists(result_file) else "[Нет результата]"
        print(f"  {i + 1}. {f} {has_result}")

    while True:
        try:
            choice = input("\nВыберите тест (1-{}) или 'q' для выхода: ".format(len(test_files)))
            if choice.lower() == 'q':
                break

            idx = int(choice) - 1
            if 0 <= idx < len(test_files):
                test_file = test_files[idx]
                img = cv2.imread(test_file)

                # Показываем оригинал
                cv2.imshow(f'Original: {test_file}', img)

                # Показываем результат если есть
                result_file = f"result_{test_file.replace('.jpg', '')}.jpg"
                if os.path.exists(result_file):
                    result = cv2.imread(result_file)
                    cv2.imshow(f'Detection Result', result)

                print("\nУправление:")
                print("  Нажмите ESC - закрыть окна")
                print("  Нажмите любую другую клавишу - следующий тест")

                key = cv2.waitKey(0)
                cv2.destroyAllWindows()

                if key == 27:  # ESC
                    break
            else:
                print("Неверный номер")
        except ValueError:
            print("Введите число")

    cv2.destroyAllWindows()


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    """Главная функция"""

    print("=" * 60)
    print("СИСТЕМА ДЕТЕКЦИИ ДЕФЕКТОВ - СЛОЖНЫЕ ТЕСТОВЫЕ ПРИМЕРЫ")
    print("=" * 60)

    print("\nДоступные тестовые примеры:")
    print("  1. Матрёшка - 5 вложенных дефектов")
    print("  2. Шахматная доска - 9 дефектов в сетке 3x3")
    print("  3. Цепная реакция - 6 соединённых дефектов")
    print("  4. Хаос - 12 случайных перекрывающихся дефектов")
    print("  5. Градиент - 8 дефектов разной интенсивности")

    print("\nВыберите действие:")
    print("1. Создать все тестовые примеры")
    print("2. Создать тестовые примеры + протестировать модель")
    print("3. Только тестирование (если модель уже обучена)")
    print("4. Показать коллаж всех результатов")
    print("5. Интерактивный просмотр")

    choice = input("\nВаш выбор (1-5): ").strip()

    if choice == '1':
        create_all_test_images()
        print("\n[OK] Все тестовые примеры созданы")

    elif choice == '2':
        create_all_test_images()
        results = test_all_complex_images()
        create_collage()

    elif choice == '3':
        results = test_all_complex_images()
        create_collage()

    elif choice == '4':
        create_collage()

    elif choice == '5':
        interactive_view()

    else:
        print("Неверный выбор")

    print("\n" + "=" * 60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    main()