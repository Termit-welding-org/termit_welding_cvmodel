"""
Конвертация датасетов между различными форматами
"""

import os
import cv2
import yaml
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
from PIL import Image

from ..utils.logger import log
from ..utils.config import config


class DatasetConverter:
    """Класс для конвертации датасетов между форматами"""

    def __init__(self):
        self.supported_formats = ['yolo', 'voc', 'coco', 'labelme']
        self.classes = config.data.classes

    def voc_to_yolo(
            self,
            xml_path: str,
            img_width: int,
            img_height: int
    ) -> List[str]:
        """
        Конвертация Pascal VOC в формат YOLO

        Args:
            xml_path: Путь к XML файлу
            img_width: Ширина изображения
            img_height: Высота изображения

        Returns:
            List[str]: Строки в формате YOLO
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            yolo_lines = []

            for obj in root.findall('object'):
                # Получаем класс
                class_name = obj.find('name').text
                if class_name not in self.classes:
                    continue

                class_id = self.classes.index(class_name)

                # Получаем bbox
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # Конвертация в YOLO формат
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # Валидация
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                if width > 0 and height > 0:
                    yolo_lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )

            return yolo_lines

        except Exception as e:
            log.error(f"Ошибка конвертации {xml_path}: {e}")
            return []

    def yolo_to_voc(
            self,
            yolo_line: str,
            img_width: int,
            img_height: int,
            class_name: str
    ) -> Dict:
        """
        Конвертация YOLO в Pascal VOC

        Args:
            yolo_line: Строка в формате YOLO
            img_width: Ширина изображения
            img_height: Высота изображения
            class_name: Имя класса

        Returns:
            Dict: Аннотация в формате VOC
        """
        parts = yolo_line.strip().split()
        if len(parts) < 5:
            return None

        class_id = int(parts[0])
        x_center = float(parts[1]) * img_width
        y_center = float(parts[2]) * img_height
        width = float(parts[3]) * img_width
        height = float(parts[4]) * img_height

        xmin = int(x_center - width / 2)
        ymin = int(y_center - height / 2)
        xmax = int(x_center + width / 2)
        ymax = int(y_center + height / 2)

        return {
            'class': class_name,
            'bbox': [xmin, ymin, xmax, ymax]
        }

    def convert_dataset(
            self,
            input_dir: str,
            output_dir: str,
            input_format: str = 'voc',
            output_format: str = 'yolo',
            num_workers: int = 8
    ):
        """
        Конвертация всего датасета

        Args:
            input_dir: Входная директория
            output_dir: Выходная директория
            input_format: Входной формат
            output_format: Выходной формат
            num_workers: Количество воркеров
        """
        log.info(f"Конвертация датасета из {input_format} в {output_format}")

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Поиск всех файлов аннотаций
        if input_format == 'voc':
            annotation_files = list(input_path.glob('**/*.xml'))
        elif input_format == 'yolo':
            annotation_files = list(input_path.glob('**/*.txt'))
        else:
            raise ValueError(f"Неподдерживаемый формат: {input_format}")

        log.info(f"Найдено {len(annotation_files)} аннотаций")

        # Конвертация
        tasks = []
        for ann_file in annotation_files:
            tasks.append((
                ann_file,
                output_path,
                input_format,
                output_format
            ))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(
                executor.map(self._convert_single, tasks),
                total=len(tasks),
                desc="Конвертация"
            ))

        log.info(f"Конвертация завершена. Результаты сохранены в {output_dir}")

    def _convert_single(self, args):
        """Конвертация одного файла"""
        ann_file, output_path, input_format, output_format = args

        try:
            if input_format == 'voc' and output_format == 'yolo':
                self._voc_to_yolo_single(ann_file, output_path)
            elif input_format == 'yolo' and output_format == 'voc':
                self._yolo_to_voc_single(ann_file, output_path)
        except Exception as e:
            log.error(f"Ошибка конвертации {ann_file}: {e}")

    def _voc_to_yolo_single(self, xml_path: Path, output_path: Path):
        """Конвертация одного VOC файла в YOLO"""
        # Получаем размеры изображения
        img_path = self._find_image(xml_path)
        if img_path:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        else:
            img_width, img_height = 640, 640

        # Конвертация
        yolo_lines = self.voc_to_yolo(str(xml_path), img_width, img_height)

        # Сохранение
        rel_path = xml_path.relative_to(xml_path.parent.parent)
        txt_path = output_path / rel_path.with_suffix('.txt')
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

    def _yolo_to_voc_single(self, txt_path: Path, output_path: Path):
        """Конвертация одного YOLO файла в VOC"""
        # TODO: Реализовать
        pass

    def _find_image(self, ann_path: Path) -> Optional[Path]:
        """Поиск соответствующего изображения"""
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for ext in img_extensions:
            img_path = ann_path.with_suffix(ext)
            if img_path.exists():
                return img_path

            # Поиск в соседних директориях
            for parent in ann_path.parents:
                img_path = parent / 'images' / ann_path.with_suffix(ext).name
                if img_path.exists():
                    return img_path

        return None

    def create_yolo_yaml(self, data_dir: str, output_path: str):
        """
        Создание YAML конфига для YOLO

        Args:
            data_dir: Директория с данными
            output_path: Путь для сохранения YAML
        """
        data_path = Path(data_dir)

        yaml_config = {
            'path': str(data_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }

        with open(output_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)

        log.info(f"YAML конфиг сохранен в {output_path}")
        return output_path


class DatasetSplitter:
    """Класс для разделения датасета на train/val/test"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def split_dataset(
            self,
            data_dir: str,
            output_dir: str,
            train_ratio: float = 0.7,
            val_ratio: float = 0.2,
            test_ratio: float = 0.1,
            copy_files: bool = True
    ):
        """
        Разделение датасета

        Args:
            data_dir: Исходная директория
            output_dir: Выходная директория
            train_ratio: Доля тренировочных данных
            val_ratio: Доля валидационных данных
            test_ratio: Доля тестовых данных
            copy_files: Копировать файлы (True) или перемещать (False)
        """
        log.info("Разделение датасета...")

        data_path = Path(data_dir)
        output_path = Path(output_dir)

        # Поиск всех изображений
        images = list(data_path.glob('**/*.jpg')) + \
                 list(data_path.glob('**/*.jpeg')) + \
                 list(data_path.glob('**/*.png'))

        log.info(f"Найдено {len(images)} изображений")

        # Перемешивание
        indices = np.random.permutation(len(images))

        # Разделение
        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)

        splits = {
            'train': [images[i] for i in indices[:train_end]],
            'valid': [images[i] for i in indices[train_end:val_end]],
            'test': [images[i] for i in indices[val_end:]]
        }

        # Создание структуры
        for split_name, split_images in splits.items():
            log.info(f"  {split_name}: {len(split_images)} изображений")

            img_dir = output_path / split_name / 'images'
            lbl_dir = output_path / split_name / 'labels'
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(split_images, desc=f"Копирование {split_name}"):
                # Копирование изображения
                dest_img = img_dir / img_path.name
                if copy_files:
                    shutil.copy2(img_path, dest_img)
                else:
                    shutil.move(img_path, dest_img)

                # Копирование аннотации
                txt_path = img_path.with_suffix('.txt')
                if txt_path.exists():
                    dest_txt = lbl_dir / txt_path.name
                    if copy_files:
                        shutil.copy2(txt_path, dest_txt)
                    else:
                        shutil.move(txt_path, dest_txt)

        log.info(f"Разделение завершено. Данные сохранены в {output_dir}")

        return splits