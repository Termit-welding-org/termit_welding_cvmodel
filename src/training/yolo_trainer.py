"""
Обучение YOLO моделей
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
from ultralytics import YOLO, settings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ..utils.logger import log
from ..utils.config import config


class YOLOTrainer:
    """Класс для обучения YOLO моделей"""

    def __init__(self, config_override: Optional[Dict] = None):
        """
        Инициализация тренера

        Args:
            config_override: Переопределение конфигурации
        """
        self.config = config.model
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)

        # Настройка Ultralytics
        self._setup_ultralytics()

        # Модель
        self.model = None
        self.results = None

        log.info(f"YOLOTrainer инициализирован с архитектурой {self.config.architecture}")

    def _setup_ultralytics(self):
        """Настройка Ultralytics"""
        settings.update({
            'raytune': False,
            'wandb': False,
            'clearml': False,
            'comet': False,
            'mlflow': False,
            'neptune': False,
            'tensorboard': True,
            'datasets_dir': config.data.prepared_data_path,
            'weights_dir': config.models_dir,
            'runs_dir': config.logs_dir,
            'sync_bn': False,
            'exist_ok': True,
            'plots': True,
            'save': True,
            'verbose': True
        })

    def train(
            self,
            data_yaml: str,
            model_name: Optional[str] = None
    ) -> Any:
        """
        Обучение модели

        Args:
            data_yaml: Путь к YAML файлу с данными
            model_name: Имя модели (если None, берется из конфига)

        Returns:
            Результаты обучения
        """
        log.info("=" * 60)
        log.info("ЗАПУСК ОБУЧЕНИЯ YOLO")
        log.info("=" * 60)

        # Загрузка модели
        model_path = model_name or self._get_model_path()
        log.info(f"Загрузка модели: {model_path}")
        self.model = YOLO(model_path)

        # Параметры обучения
        train_args = self._prepare_train_args(data_yaml)

        # Вывод параметров
        log.info("\n Параметры обучения:")
        log.info(f"  - Модель: {self.config.architecture}")
        log.info(f"  - Эпохи: {self.config.epochs}")
        log.info(f"  - Батч: {self.config.batch_size}")
        log.info(f"  - LR: {self.config.learning_rate}")
        log.info(f"  - Оптимизатор: {self.config.optimizer}")
        log.info(f"  - Mosaic: {self.config.mosaic}")
        log.info(f"  - Close mosaic: {self.config.close_mosaic}")

        # Обучение
        log.info("\n Начинаем обучение...")
        self.results = self.model.train(**train_args)

        log.info("\n Обучение завершено!")

        # Сохранение результатов
        self._save_results()

        return self.results

    def _get_model_path(self) -> str:
        """Получение пути к модели"""
        model_mapping = {
            'yolov5n': 'yolov5nu.pt',
            'yolov5s': 'yolov5su.pt',
            'yolov5m': 'yolov5mu.pt',
            'yolov5l': 'yolov5lu.pt',
            'yolov5x': 'yolov5xu.pt',
            'yolov8n': 'yolov8n.pt',
            'yolov8s': 'yolov8s.pt',
            'yolov8m': 'yolov8m.pt',
            'yolov8l': 'yolov8l.pt',
            'yolov8x': 'yolov8x.pt'
        }

        return model_mapping.get(self.config.architecture, 'yolov8m.pt')

    def _prepare_train_args(self, data_yaml: str) -> Dict:
        """Подготовка аргументов для обучения"""

        # Базовые аргументы
        args = {
            'data': data_yaml,
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': config.data.image_size,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 8,
            'patience': self.config.patience,
            'save': True,
            'save_period': self.config.save_period,
            'pretrained': self.config.pretrained,
            'optimizer': self.config.optimizer,
            'lr0': self.config.learning_rate,
            'lrf': 0.01,
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            'warmup_epochs': self.config.warmup_epochs,
            'warmup_momentum': self.config.warmup_momentum,
            'cos_lr': self.config.scheduler == 'cosine',
            'label_smoothing': self.config.label_smoothing,
            'dropout': self.config.dropout,

            # Аугментации
            'mosaic': self.config.mosaic,
            'mixup': self.config.mixup,
            'copy_paste': self.config.copy_paste,
            'close_mosaic': self.config.close_mosaic,
            'hsv_h': self.config.hsv_h,
            'hsv_s': self.config.hsv_s,
            'hsv_v': self.config.hsv_v,
            'degrees': self.config.degrees,
            'translate': self.config.translate,
            'scale': self.config.scale,
            'shear': self.config.shear,
            'perspective': self.config.perspective,
            'flipud': self.config.flipud,
            'fliplr': self.config.fliplr,

            # Системные
            'amp': True,
            'cache': True,
            'exist_ok': True,
            'verbose': True,
            'seed': config.seed,
            'project': config.models_dir,
            'name': f"{self.config.architecture}_welding"
        }

        return args

    def validate(self, model_path: str, data_yaml: str) -> Dict:
        """
        Валидация модели

        Args:
            model_path: Путь к модели
            data_yaml: Путь к YAML файлу с данными

        Returns:
            Метрики валидации
        """
        log.info(" Валидация модели...")

        model = YOLO(model_path)
        metrics = model.val(data=data_yaml, split='test')

        log.info("\nРезультаты валидации:")
        log.info(f"  - mAP50: {metrics.box.map50:.4f}")
        log.info(f"  - mAP50-95: {metrics.box.map:.4f}")
        log.info(f"  - Precision: {metrics.box.mp:.4f}")
        log.info(f"  - Recall: {metrics.box.mr:.4f}")

        return metrics

    def _save_results(self):
        """Сохранение результатов обучения"""
        if self.results is None:
            return

        # Сохранение метрик
        metrics_df = pd.DataFrame(self.results.results)
        metrics_path = Path(config.logs_dir) / f"{self.config.architecture}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        log.info(f"Метрики сохранены в {metrics_path}")

    def plot_training_results(self, results_dir: Optional[str] = None):
        """
        Построение графиков обучения

        Args:
            results_dir: Директория с результатами
        """
        if results_dir is None:
            results_dir = Path(config.models_dir) / f"{self.config.architecture}_welding"

        results_img = Path(results_dir) / 'results.png'

        if results_img.exists():
            img = plt.imread(results_img)
            plt.figure(figsize=(15, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Графики обучения', fontsize=14)
            plt.show()
        else:
            log.warning(f"График не найден: {results_img}")

    def export_model(
            self,
            model_path: str,
            formats: Optional[list] = None
    ) -> Dict[str, str]:
        """
        Экспорт модели в различные форматы

        Args:
            model_path: Путь к модели
            formats: Список форматов для экспорта

        Returns:
            Словарь с путями к экспортированным моделям
        """
        if formats is None:
            formats = ['onnx', 'tensorrt', 'openvino', 'tflite']

        log.info(" Экспорт модели...")

        model = YOLO(model_path)
        exported_paths = {}

        for fmt in formats:
            try:
                log.info(f"  Экспорт в {fmt}...")
                exported_path = model.export(
                    format=fmt,
                    imgsz=config.data.image_size,
                    half=True if fmt == 'tensorrt' else False,
                    optimize=True
                )
                exported_paths[fmt] = exported_path
                log.info(f"  {fmt}: {exported_path}")
            except Exception as e:
                log.error(f"  Ошибка экспорта в {fmt}: {e}")

        return exported_paths


class FasterRCNNTrainer:
    """Класс для обучения Faster R-CNN"""

    def __init__(self):
        # TODO: Реализовать обучение Faster R-CNN
        pass