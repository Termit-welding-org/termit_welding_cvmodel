"""
Конфигурация проекта
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from pathlib import Path


@dataclass
class DataConfig:
    """Конфигурация данных"""

    # Пути
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    prepared_data_path: str = "data/prepared"

    # Классы дефектов сварки
    classes: List[str] = field(default_factory=lambda: [
        'good_weld',
        'bad_weld',
        'crack',
        'pore',
        'undercut',
        'spatter',
        'lack_of_fusion'
    ])

    # Разделение данных
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1

    # Аугментации
    use_augmentation: bool = True
    augmentation_prob: float = 0.5

    # Размер изображений
    image_size: int = 640
    min_image_size: int = 320
    max_image_size: int = 1280


@dataclass
class ModelConfig:
    """Конфигурация модели"""

    # Архитектура
    architecture: str = "yolov8m"  # yolov5mu, yolov8m, faster_rcnn
    pretrained: bool = True
    num_classes: int = 7

    # Параметры обучения
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    momentum: float = 0.937

    # Оптимизатор
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8

    # Регуляризация
    dropout: float = 0.0
    label_smoothing: float = 0.0

    # Аугментации при обучении
    mosaic: float = 1.0
    mixup: float = 0.15
    copy_paste: float = 0.1
    close_mosaic: int = 10

    # HSV аугментации
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    # Геометрические аугментации
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0001
    flipud: float = 0.0
    fliplr: float = 0.5

    # Сохранение
    save_period: int = 10
    patience: int = 20


@dataclass
class InferenceConfig:
    """Конфигурация инференса"""

    # Пороги
    confidence_threshold: float = 0.35
    nms_threshold: float = 0.5
    max_detections: int = 100

    # Производительность
    device: str = "cuda"
    half_precision: bool = True
    batch_size: int = 1

    # Визуализация
    show_labels: bool = True
    show_conf: bool = True
    line_width: int = 2
    box_alpha: float = 0.3


@dataclass
class DeploymentConfig:
    """Конфигурация деплоймента"""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Мониторинг
    enable_metrics: bool = True
    log_level: str = "INFO"

    # Экспорт
    export_formats: List[str] = field(default_factory=lambda: [
        "onnx", "tensorrt", "openvino"
    ])


@dataclass
class ProjectConfig:
    """Главная конфигурация проекта"""

    # Версия
    version: str = "1.0.0"
    project_name: str = "welding_inspection_ai"

    # Компоненты
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    # Пути
    base_dir: str = "."
    models_dir: str = "models"
    logs_dir: str = "logs"
    outputs_dir: str = "outputs"

    # Случайность
    seed: int = 42

    def __post_init__(self):
        """Создание необходимых директорий"""
        dirs = [
            self.base_dir,
            self.models_dir,
            self.logs_dir,
            self.outputs_dir,
            self.data.raw_data_path,
            self.data.processed_data_path,
            self.data.prepared_data_path
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def save(self, path: str = "config.yaml"):
        """Сохранение конфигурации в YAML"""
        config_dict = {
            'version': self.version,
            'project_name': self.project_name,
            'seed': self.seed,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'inference': self.inference.__dict__,
            'deployment': self.deployment.__dict__
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        return path

    @classmethod
    def load(cls, path: str = "config.yaml"):
        """Загрузка конфигурации из YAML"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = cls()
        config.version = config_dict.get('version', '1.0.0')
        config.project_name = config_dict.get('project_name', 'welding_inspection_ai')
        config.seed = config_dict.get('seed', 42)

        # Загрузка подконфигураций
        for key, value in config_dict.get('data', {}).items():
            setattr(config.data, key, value)

        for key, value in config_dict.get('model', {}).items():
            setattr(config.model, key, value)

        for key, value in config_dict.get('inference', {}).items():
            setattr(config.inference, key, value)

        for key, value in config_dict.get('deployment', {}).items():
            setattr(config.deployment, key, value)

        return config


# Глобальная конфигурация
config = ProjectConfig()