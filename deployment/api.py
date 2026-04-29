"""
REST API для сервиса инспекции сварных швов
"""

import io
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image

# Прямой импорт вместо относительного
from ultralytics import YOLO

# ============================================
# Pydantic модели
# ============================================

try:
    from pydantic import BaseModel, Field
except ImportError:
    pip
    install
    pydantic
    from pydantic import BaseModel, Field


class InspectionResponse(BaseModel):
    request_id: str
    status: str
    quality: str
    total_defects: int
    detections: List[Dict]
    recommendation: str
    processing_time: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime: float


# ============================================
# FastAPI приложение
# ============================================

app = FastAPI(
    title="Welding Inspection AI API",
    description="API для автоматической инспекции качества сварных швов",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
model = None
start_time = time.time()
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

CLASSES = ['Geometric defect', 'Non-fusion defect', 'crack', 'porosity', 'spatters']


@app.on_event("startup")
async def startup():
    global model
    model_path = os.environ.get("MODEL_PATH", "yolov8m.pt")

    try:
        model = YOLO(model_path)
        print(f" Модель загружена: {model_path}")
    except Exception as e:
        print(f" Модель не загружена: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h1> Welding Inspection API</h1>
    <p>API работает! Документация: <a href="/docs">/docs</a></p>
    """


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "uptime": time.time() - start_time
    }


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), conf: float = Form(0.35)):
    if model is None:
        return JSONResponse({"error": "Модель не загружена"}, status_code=503)

    # Чтение изображения
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Инференс
    results = model.predict(image, conf=conf, verbose=False)
    boxes = results[0].boxes

    detections = []
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class": CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown",
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })

    # Сохранение результата
    request_id = str(uuid.uuid4())
    result_path = output_dir / f"{request_id}.jpg"
    cv2.imwrite(str(result_path), results[0].plot())

    return {
        "request_id": request_id,
        "status": "defects_found" if detections else "no_defects",
        "quality": "REJECT" if detections else "ACCEPT",
        "total_defects": len(detections),
        "detections": detections,
        "recommendation": "Проверить дефекты" if detections else "Шов качественный",
        "processing_time": 0.1,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/images/{request_id}")
async def get_image(request_id: str):
    image_path = output_dir / f"{request_id}.jpg"
    if not image_path.exists():
        return JSONResponse({"error": "Не найдено"}, status_code=404)
    return FileResponse(image_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)