"""
utils/logger.py
Structured logging — เขียนทั้ง console และ file (outputs/logs/)
"""

import logging
import os
from pathlib import Path


def get_logger(name: str, log_dir: str = "outputs/logs", level: int = logging.INFO) -> logging.Logger:
    """
    สร้าง logger ที่ส่ง output ไปทั้ง console และ file

    Args:
        name: ชื่อ logger (ใช้ __name__ จาก module ที่เรียก)
        log_dir: โฟลเดอร์สำหรับเก็บ log file
        level: logging level
    """
    logger = logging.getLogger(name)

    # ไม่ add handler ซ้ำถ้า logger นี้ถูก init แล้ว
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
