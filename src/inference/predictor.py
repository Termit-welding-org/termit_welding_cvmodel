"""
Модуль для инференса моделей
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import time
from dataclasses import dataclass

from ..utils.logger import log
from ..utils.config import config, InferenceConfig


@dataclass
class Detection:
    """Класс для хранения результатов детекции"""

    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    area: float
    center: Tuple[float, float]

    def to_dict(self) -> Dict:
        """Конвертация в словарь"""
        return {
            'class': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'area': self.area,
            'center': self.center
        }


@dataclass
class InspectionResult:
    """Класс для хранения результатов инспекции"""

    status: str  # 'GOOD', 'BAD', 'REWORK', 'UNKNOWN'
    quality: str  # 'ACCEPT', 'REJECT', 'REWORK'
    detections: List[Detection]
    total_defects: int
    recommendation: str
    processing_time: float
    timestamp: str

    def to_dict(self) -> Dict:
        """Конвертация в словарь"""
        return {
            'status': self.status,
            'quality': self.quality,
            'detections': [d.to_dict() for d in self.detections],
            'total_defects': self.total_defects,
            'recommendation': self.recommendation,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp
        }


class WeldingInspector:
    """
    Класс для инспекции сварных швов

    Поддерживаемые модели:
    - YOLOv5
    - YOLOv8
    - ONNX
    - TensorRT
    """

    def __init__(
            self,
            model_path: str,
            conf_threshold: float = 0.35,
            iou_threshold: float = 0.5,
            device: Optional[str] = None,
            class_names: Optional[List[str]] = None
    ):
        """
        Инициализация инспектора

        Args:
            model_path: Путь к модели (.pt, .onnx, .engine)
            conf_threshold: Порог уверенности
            iou_threshold: Порог NMS
            device: Устройство для инференса
            class_names: Имена классов
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Классы
        self.class_names = class_names or config.data.classes

        # Цвета для визуализации
        self.colors = self._generate_colors(len(self.class_names))

        # Загрузка модели
        self.model = self._load_model()

        # Статистика
        self.total_inspections = 0
        self.total_time = 0.0

        log.info(f"WeldingInspector инициализирован")
        log.info(f"  - Модель: {model_path}")
        log.info(f"  - Устройство: {self.device}")
        log.info(f"  - Порог уверенности: {conf_threshold}")
        log.info(f"  - Классы: {self.class_names}")

    def _load_model(self) -> Any:
        """Загрузка модели"""
        try:
            if self.model_path.suffix == '.pt':
                return YOLO(str(self.model_path))
            elif self.model_path.suffix == '.onnx':
                return YOLO(str(self.model_path))
            elif self.model_path.suffix == '.engine':
                return YOLO(str(self.model_path))
            else:
                raise ValueError(f"Неподдерживаемый формат модели: {self.model_path.suffix}")
        except Exception as e:
            log.error(f"Ошибка загрузки модели: {e}")
            raise

    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Генерация цветов для классов"""
        colors = []
        for i in range(num_colors):
            hue = int(179 * i / num_colors)
            color = cv2.cvtColor(
                np.array([[[hue, 255, 255]]], dtype=np.uint8),
                cv2.COLOR_HSV2BGR
            )[0][0]
            colors.append(tuple(map(int, color)))
        return colors

    def inspect_image(
            self,
            image_path: str,
            save_result: bool = False,
            output_path: Optional[str] = None
    ) -> InspectionResult:
        """
        Инспекция одного изображения

        Args:
            image_path: Путь к изображению
            save_result: Сохранить результат
            output_path: Путь для сохранения

        Returns:
            InspectionResult: Результат инспекции
        """
        start_time = time.time()

        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        # Инференс
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        # Обработка результатов
        inspection_result = self._process_results(results[0], start_time)

        # Сохранение
        if save_result:
            annotated_image = self.draw_results(image, inspection_result)
            output_path = output_path or f"result_{Path(image_path).name}"
            cv2.imwrite(output_path, annotated_image)
            log.info(f"Результат сохранен: {output_path}")

        # Обновление статистики
        self.total_inspections += 1
        self.total_time += inspection_result.processing_time

        return inspection_result

    def inspect_frame(
            self,
            frame: np.ndarray,
            verbose: bool = False
    ) -> Tuple[np.ndarray, InspectionResult]:
        """
        Инспекция кадра в реальном времени

        Args:
            frame: Кадр (numpy array)
            verbose: Подробный вывод

        Returns:
            Tuple: (аннотированный кадр, результат инспекции)
        """
        start_time = time.time()

        # Инференс
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=verbose
        )

        # Обработка результатов
        inspection_result = self._process_results(results[0], start_time)

        # Отрисовка
        annotated_frame = results[0].plot()

        # Добавление текстовой информации
        annotated_frame = self._add_info_overlay(annotated_frame, inspection_result)

        return annotated_frame, inspection_result

    def inspect_video(
            self,
            video_path: str,
            output_path: Optional[str] = None,
            display: bool = True,
            callback: Optional[callable] = None
    ) -> List[InspectionResult]:
        """
        Инспекция видео

        Args:
            video_path: Путь к видео
            output_path: Путь для сохранения результата
            display: Отображать видео в реальном времени
            callback: Функция обратного вызова для каждого кадра

        Returns:
            List[InspectionResult]: Результаты инспекции для каждого кадра
        """
        log.info(f"Инспекция видео: {video_path}")

        cap = cv2.VideoCapture(video_path)

        # Параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Видео-райтер
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Инспекция кадра
            annotated_frame, result = self.inspect_frame(frame)

            results.append(result)

            # Сохранение
            if writer:
                writer.write(annotated_frame)

            # Отображение
            if display:
                cv2.imshow('Welding Inspection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Callback
            if callback:
                callback(frame_count, result)

            if frame_count % 100 == 0:
                log.info(f"Обработано кадров: {frame_count}")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        log.info(f"Инспекция видео завершена. Обработано {frame_count} кадров")

        return results

    def inspect_camera(
            self,
            camera_id: int = 0,
            display: bool = True
    ):
        """
        Инспекция с камеры в реальном времени

        Args:
            camera_id: ID камеры
            display: Отображать видео
        """
        log.info(f"Запуск инспекции с камеры {camera_id}")

        cap = cv2.VideoCapture(camera_id)

        # Установка разрешения
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fps_counter = 0
        fps_start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Инспекция
            annotated_frame, result = self.inspect_frame(frame)

            # FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - fps_start_time
                fps = 30 / elapsed
                fps_start_time = time.time()

                # Добавление FPS на кадр
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            # Отображение
            if display:
                cv2.imshow('Welding Inspection - Live', annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Сохранение текущего кадра
                    cv2.imwrite(f"capture_{time.time()}.jpg", annotated_frame)
                    log.info("Кадр сохранен")

        cap.release()
        cv2.destroyAllWindows()

    def _process_results(
            self,
            result: Any,
            start_time: float
    ) -> InspectionResult:
        """Обработка результатов детекции"""
        from datetime import datetime

        processing_time = time.time() - start_time
        timestamp = datetime.now().isoformat()

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return InspectionResult(
                status='NO_DETECTION',
                quality='UNKNOWN',
                detections=[],
                total_defects=0,
                recommendation='Требуется визуальный осмотр оператором',
                processing_time=processing_time,
                timestamp=timestamp
            )

        # Обработка детекций
        detections = []
        has_bad_weld = False
        has_defect = False

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else 'unknown'

            # Вычисление площади и центра
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            detections.append(Detection(
                class_name=class_name,
                confidence=conf,
                bbox=bbox,
                area=area,
                center=center
            ))

            # Определение статуса
            if class_name in ['bad_weld', 'crack', 'lack_of_fusion']:
                has_bad_weld = True
            elif class_name in ['pore', 'undercut', 'spatter']:
                has_defect = True

        # Определение качества и рекомендаций
        if has_bad_weld:
            quality = 'REJECT'
            status = 'BAD_WELD'
            recommendation = 'Немедленная отбраковка. Проверить параметры сварки.'
        elif has_defect:
            quality = 'REWORK'
            status = 'DEFECT_FOUND'
            recommendation = 'Обнаружены дефекты. Возможна зачистка и подварка.'
        else:
            quality = 'ACCEPT'
            status = 'GOOD_WELD'
            recommendation = 'Шов качественный. Продолжить производство.'

        return InspectionResult(
            status=status,
            quality=quality,
            detections=detections,
            total_defects=len(detections),
            recommendation=recommendation,
            processing_time=processing_time,
            timestamp=timestamp
        )

    def draw_results(
            self,
            image: np.ndarray,
            result: InspectionResult
    ) -> np.ndarray:
        """
        Отрисовка результатов на изображении

        Args:
            image: Исходное изображение
            result: Результат инспекции

        Returns:
            np.ndarray: Изображение с аннотациями
        """
        annotated = image.copy()

        # Отрисовка детекций
        for det in result.detections:
            class_id = self.class_names.index(det.class_name) if det.class_name in self.class_names else 0
            color = self.colors[class_id]

            # Прямоугольник
            cv2.rectangle(
                annotated,
                (int(det.bbox[0]), int(det.bbox[1])),
                (int(det.bbox[2]), int(det.bbox[3])),
                color,
                2
            )

            # Подпись
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(
                annotated,
                (int(det.bbox[0]), int(det.bbox[1]) - label_size[1] - 10),
                (int(det.bbox[0]) + label_size[0], int(det.bbox[1])),
                color,
                -1
            )

            cv2.putText(
                annotated,
                label,
                (int(det.bbox[0]), int(det.bbox[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        # Добавление информации
        annotated = self._add_info_overlay(annotated, result)

        return annotated

    def _add_info_overlay(
            self,
            image: np.ndarray,
            result: InspectionResult
    ) -> np.ndarray:
        """Добавление информационного оверлея"""
        h, w = image.shape[:2]

        # Полупрозрачный фон
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

        # Цвет статуса
        if result.quality == 'ACCEPT':
            color = (0, 255, 0)
        elif result.quality == 'REWORK':
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        # Текст
        texts = [
            f"Status: {result.status}",
            f"Quality: {result.quality}",
            f"Defects: {result.total_defects}",
            f"Time: {result.processing_time * 1000:.1f}ms"
        ]

        for i, text in enumerate(texts):
            cv2.putText(
                image,
                text,
                (10, 25 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return image

    def get_statistics(self) -> Dict:
        """Получение статистики инспекций"""
        return {
            'total_inspections': self.total_inspections,
            'total_time': self.total_time,
            'avg_time': self.total_time / max(1, self.total_inspections),
            'device': self.device,
            'conf_threshold': self.conf_threshold
        }