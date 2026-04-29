"""Главный скрипт Welding Inspection AI"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies(command=None):  # ← Добавлен параметр
    """Проверка зависимостей"""
    modules = {
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'cv2': 'opencv-python',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'streamlit': 'streamlit'
    }

    print("\n Проверка зависимостей:")
    all_ok = True

    for module, name in modules.items():
        try:
            __import__(module)
            print(f"  {name}")
        except ImportError:
            print(f"  {name}")
            all_ok = False

    if not all_ok:
        print("\nУстановите: pip install opencv-python")

    return all_ok


def predict(args):
    """Инференс"""
    if not check_dependencies('predict'):  # Теперь функция принимает аргумент
        return

    try:
        from ultralytics import YOLO

        model_path = args.model or 'yolov8n.pt'
        image_path = args.image

        if not image_path:
            print(" Укажите --image")
            return

        print(f"Модель: {model_path}")
        print(f"Изображение: {image_path}")

        model = YOLO(model_path)
        results = model.predict(image_path, conf=0.35, save=True)

        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                print(f"\n Найдено объектов: {len(boxes)}")
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    print(f"  - {cls_name}: {conf:.2f}")
            else:
                print("\n Объекты не найдены")

        print(f"\n Результат сохранен в: runs/detect/predict/")

    except ImportError as e:
        print(f" Ошибка импорта: {e}")
    except Exception as e:
        print(f" Ошибка: {e}")


def train(args):
    """Обучение модели"""
    print("\n" + "=" * 60)
    print("🚀 ЗАПУСК ОБУЧЕНИЯ YOLO")
    print("=" * 60)

    try:
        from ultralytics import YOLO

        # Проверяем data.yaml
        data_yaml = args.data_yaml or 'data.yaml'

        if not Path(data_yaml).exists():
            print(f"❌ Файл не найден: {data_yaml}")
            return

        print(f"📁 Датасет: {data_yaml}")
        print(f"⚙️ Эпохи: {args.epochs}")
        print(f"📦 Батч: {args.batch}")

        # Загружаем модель
        print("\n📥 Загрузка модели yolov8m.pt...")
        model = YOLO('yolov8m.pt')

        # Запускаем обучение
        print("\n🔥 Начинаем обучение...")
        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=640,
            batch=args.batch,
            device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
            workers=2,
            patience=20,
            save=True,
            save_period=10,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,

            # Аугментации
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            mosaic=1.0,
            mixup=0.1,
            close_mosaic=10,

            project='runs/detect',
            name='weld_detection',
            exist_ok=True,
            verbose=True
        )

        print("\n✅ Обучение завершено!")
        print(f"📁 Модель сохранена: runs/detect/weld_detection/weights/best.pt")

    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Установите: pip install ultralytics torch")
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")

def serve(args):
    """API сервер"""
    print("Запуск API...")


def demo(args):
    """Демо"""
    print("Запуск демо...")


def main():
    parser = argparse.ArgumentParser(description="Welding Inspection AI")
    subparsers = parser.add_subparsers(dest='command', help='Команды')

    train_parser = subparsers.add_parser('train', help='Обучение модели')
    train_parser.add_argument('--data-yaml', type=str)
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--batch', type=int, default=8)
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser('predict', help='Инференс')
    predict_parser.add_argument('--model', type=str, default='yolov8n.pt')
    predict_parser.add_argument('--image', type=str, required=True)
    predict_parser.add_argument('--conf', type=float, default=0.35)
    predict_parser.set_defaults(func=predict)

    serve_parser = subparsers.add_parser('serve', help='API сервер')
    serve_parser.add_argument('--model', type=str, default='yolov8n.pt')
    serve_parser.add_argument('--port', type=int, default=8000)
    serve_parser.set_defaults(func=serve)

    demo_parser = subparsers.add_parser('demo', help='Демо-интерфейс')
    demo_parser.add_argument('--port', type=int, default=8501)
    demo_parser.set_defaults(func=demo)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        check_dependencies()
    else:
        args.func(args)


if __name__ == "__main__":
    main()