"""
Модуль логирования
"""

import sys
import logging
from pathlib import Path
from loguru import logger
from typing import Optional


class Logger:
    """Класс для управления логированием"""

    def __init__(
            self,
            name: str = "welding_ai",
            log_dir: str = "logs",
            level: str = "INFO",
            rotation: str = "100 MB",
            retention: str = "30 days"
    ):
        """
        Инициализация логгера

        Args:
            name: Имя логгера
            log_dir: Директория для логов
            level: Уровень логирования
            rotation: Ротация логов
            retention: Хранение логов
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Удаляем стандартный handler
        logger.remove()

        # Добавляем handler для консоли
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=level,
            colorize=True
        )

        # Добавляем handler для файла (все уровни)
        logger.add(
            self.log_dir / f"{name}_{{time:YYYY-MM-DD}}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation=rotation,
            retention=retention,
            compression="zip"
        )

        # Добавляем handler для ошибок
        logger.add(
            self.log_dir / f"{name}_errors_{{time:YYYY-MM-DD}}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation=rotation,
            retention=retention
        )

        self.logger = logger.bind(name=name)

    def get_logger(self):
        """Получить логгер"""
        return self.logger


# Глобальный логгер
log = Logger().get_logger()