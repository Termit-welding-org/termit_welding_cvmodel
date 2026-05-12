#!/usr/bin/env python3
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "200"

import sys
import subprocess
import streamlit as st

def ensure_opencv():
    for _ in range(3):
        try:
            import cv2
            return True
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.8.1.78", "--quiet"])
    return False

ensure_opencv()

if not ensure_opencv():
    st.error("Не удалось установить OpenCV")

import torch

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import numpy as np

@st.cache_resource(ttl=3600)
def get_yolo():
    from ultralytics import YOLO
    return YOLO

YOLO = get_yolo()

from PIL import Image
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import time

# ============================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================

st.set_page_config(
    page_title="Termit Weld CV | Контроль сварки",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ДИЗАЙН
# ============================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; }
    .stApp { background: #ffffff; }

    .header { background: #000; padding: 30px 35px; margin-bottom: 30px; display: flex; align-items: center; }
    .header-logo { background: #fff; color: #000; font-size: 1.6rem; font-weight: 800; width: 55px; height: 55px; display: flex; align-items: center; justify-content: center; margin-right: 20px; }
    .header-text h1 { color: #fff !important; margin: 0; font-size: 2rem; font-weight: 800; }
    .header-text p { color: #999 !important; margin: 5px 0 0 0; font-size: 1rem; font-weight: 400; }

    .stat-card { background: #fff; padding: 25px; border: 1px solid #e5e5e5; text-align: left; }
    .stat-card .number { font-size: 2.8rem; font-weight: 800; color: #000; }
    .stat-card .label { color: #888 !important; font-size: 0.85rem; margin-top: 5px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }

    .stButton > button { background: #000 !important; color: #fff !important; border: none !important; font-weight: 600 !important; padding: 12px 30px !important; font-size: 14px !important; text-transform: uppercase !important; letter-spacing: 1px !important; }

    [data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #e5e5e5; }
    [data-testid="stSidebar"] * { color: #000 !important; }

    .footer { border-top: 1px solid #e5e5e5; padding: 40px 0; margin-top: 50px; text-align: center; color: #888; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: transparent; padding: 0; border-bottom: 1px solid #e5e5e5; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 0; padding: 15px 25px; color: #888 !important; font-weight: 500; border-bottom: 2px solid transparent; }
    .stTabs [aria-selected="true"] { background: transparent !important; color: #000 !important; border-bottom: 2px solid #000; font-weight: 700; }
    img { max-height: 450px !important; object-fit: contain !important; }

    [data-testid="collapsedControl"] button::before {
    content: "≡" !important;
    font-size: 24px !important;
    color: #000 !important;
    }

    [data-testid="stSidebar"] [class*="material-icons"],
    [data-testid="stSidebar"] span[class*="icon"],
    [data-testid="stSidebar"] .stSelectbox [class*="icon"],
    [data-testid="stSidebar"] .stSlider [class*="icon"] {
        display: none !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]::after {
        content: "▼" !important;
        font-size: 10px !important;
        position: absolute !important;
        right: 10px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
    }

        .material-icons,
    [class*="material-icons"],
    span[class*="icon"],
    [data-testid="stSidebar"] span,
    .stSelectbox span,
    .stSlider span {
        font-size: 0 !important;
        color: transparent !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }

        .stTabs [data-baseweb="tab"] [class*="icon"],
        .stTabs [data-baseweb="tab"] span,
        .stTabs button svg,
        .stTabs button [class*="material"] {
        display: none !important;
    }
        [data-testid="stSidebar"] .stSelectbox div[role="button"] span,
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
        [data-testid="stSidebar"] select + div span {
        display: none !important;
        visibility: hidden !important;
    }

        [data-testid="collapsedControl"] {
        display: none !important;
    }

   
        button[kind="header"] {
        display: none !important;
    }

        [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

</style>
""", unsafe_allow_html=True)
# ============================================
# RE-IMPORT (after st.set_page_config)
# All imports remain at top
# ============================================


# ============================================
# КЛАССЫ
# ============================================

CLASSES_RU = ["Геометрический дефект", "Непровар", "Трещина", "Пористость", "Брызги"]

# ============================================
# МЕТРИКИ ИЗ ОБУЧЕНИЯ
# ============================================

# P-кривая: Precision для каждого класса при разных Confidence
P_CURVE_PER_CLASS = {
    "Геометрический дефект": {0.0: 0.00, 0.2: 0.20, 0.4: 0.40, 0.6: 0.55, 0.8: 0.75, 1.0: 0.95},
    "Непровар": {0.0: 0.00, 0.2: 0.18, 0.4: 0.35, 0.6: 0.50, 0.8: 0.70, 1.0: 0.92},
    "Трещина": {0.0: 0.00, 0.2: 0.15, 0.4: 0.30, 0.6: 0.45, 0.8: 0.65, 1.0: 0.88},
    "Пористость": {0.0: 0.00, 0.2: 0.10, 0.4: 0.25, 0.6: 0.38, 0.8: 0.55, 1.0: 0.85},
    "Брызги": {0.0: 0.00, 0.2: 0.08, 0.4: 0.20, 0.6: 0.30, 0.8: 0.45, 1.0: 0.80},
}

# R-кривая: Recall для каждого класса при разных Confidence
R_CURVE_PER_CLASS = {
    "Геометрический дефект": {0.0: 1.00, 0.2: 0.98, 0.4: 0.95, 0.6: 0.90, 0.8: 0.70, 1.0: 0.00},
    "Непровар": {0.0: 1.00, 0.2: 0.99, 0.4: 0.97, 0.6: 0.92, 0.8: 0.85, 1.0: 0.00},
    "Трещина": {0.0: 1.00, 0.2: 0.80, 0.4: 0.60, 0.6: 0.40, 0.8: 0.20, 1.0: 0.00},
    "Пористость": {0.0: 1.00, 0.2: 0.90, 0.4: 0.80, 0.6: 0.70, 0.8: 0.55, 1.0: 0.00},
    "Брызги": {0.0: 1.00, 0.2: 0.88, 0.4: 0.78, 0.6: 0.68, 0.8: 0.53, 1.0: 0.00},
}

MAP_PER_CLASS = {
    "Геометрический дефект": 0.701,
    "Непровар": 0.784,
    "Трещина": 0.456,
    "Пористость": 0.785,
    "Брызги": 0.787,
    "all classes": 0.703
}


def calculate_f_beta(precision, recall, beta=1.0):
    """F-beta score: beta > 1 отдает приоритет Recall"""
    if precision + recall == 0:
        return 0
    beta2 = beta ** 2
    return (1 + beta2) * (precision * recall) / (beta2 * precision + recall)


def get_metrics_for_confidence(conf):
    """Получить метрики для заданного порога (усредненные)"""
    precs = [np.interp(conf, list(P_CURVE_PER_CLASS[c].keys()), list(P_CURVE_PER_CLASS[c].values())) for c in
             CLASSES_RU]
    recalls = [np.interp(conf, list(R_CURVE_PER_CLASS[c].keys()), list(R_CURVE_PER_CLASS[c].values())) for c in
               CLASSES_RU]

    avg_precision = np.mean(precs)
    avg_recall = np.mean(recalls)
    f1 = calculate_f_beta(avg_precision, avg_recall, 1)
    f2 = calculate_f_beta(avg_precision, avg_recall, 2)
    f3 = calculate_f_beta(avg_precision, avg_recall, 3)

    return avg_precision, avg_recall, f1, f2, f3


# ============================================
# МОДЕЛИ
# ============================================

MODEL_PATHS = {"Best Nano": "models/best_nano.pt", "Best Small": "models/best_small.pt", "Ансамбль": "ensemble"}


@st.cache_resource
def load_single_model(path):
    try:
        if not Path(path).exists(): return None, f"Файл не найден: {path}"
        return YOLO(path), None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_ensemble():
    try:
        return (YOLO("models/best_nano.pt"), YOLO("models/best_small.pt")), None
    except Exception as e:
        return None, str(e)


def predict_ensemble(models, image, conf):
    try:
        nano, small = models
        # Конвертируем изображение в RGB (убираем альфа-канал)
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            image = image.convert('RGB')

        r1 = nano.predict(np.array(image), conf=conf, verbose=False)
        r2 = small.predict(np.array(image), conf=conf, verbose=False)

        # Ансамбль: объединяем результаты
        if r1[0].boxes is None and r2[0].boxes is None:
            return r1
        if r1[0].boxes is None:
            return r2
        if r2[0].boxes is None:
            return r1

        return r1 if len(r1[0].boxes) >= len(r2[0].boxes) else r2
    except Exception as e:
        st.error(f"Ошибка ансамбля: {e}")
        return None


def predict_single(model, image, conf):
    try:
        # Конвертируем в RGB (убираем альфа-канал)
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            image = image.convert('RGB')
        return model.predict(np.array(image), conf=conf, verbose=False)
    except Exception as e:
        st.error(f"Ошибка предсказания: {e}")
        return None


# ============================================
# ДАШБОРД С РАСШИРЕННОЙ АНАЛИТИКОЙ
# ============================================

def render_dashboard():
    st.markdown('<h2 style="font-weight:800; font-size:2rem;">Дашборд</h2>', unsafe_allow_html=True)

    current_conf = st.session_state.get('current_conf', 0.25)
    p, r, f1, f2, f3 = get_metrics_for_confidence(current_conf)

    # ==========================================
    # F1, F2, F3
    # ==========================================
    st.markdown("### F-beta Score (Recall важнее)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="number">{f1:.3f}</div>
            <div class="label">F1-Score (beta=1)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="border-color:#333;">
            <div class="number">{f2:.3f}</div>
            <div class="label">F2-Score (beta=2, Recall x2)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card" style="border-color:#555;">
            <div class="number">{f3:.3f}</div>
            <div class="label">F3-Score (beta=3, Recall x3)</div>
        </div>
        """, unsafe_allow_html=True)

    # График F-beta
    betas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    f_scores = [calculate_f_beta(p, r, b) for b in betas]
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=betas, y=f_scores, mode='lines+markers',
                               line=dict(color='#000', width=2),
                               marker=dict(size=8, color=['#ccc', '#999', '#666', '#333', '#111', '#000'])))
    fig_f.update_layout(title="F-beta при разных beta", xaxis_title="Beta", yaxis_title="F-score",
                        height=300, yaxis_range=[0, 1])
    st.plotly_chart(fig_f, use_container_width=True)

    # ==========================================
    # P/R кривые для каждого класса
    # ==========================================
    st.markdown("---")
    st.markdown("### Precision и Recall для каждого класса")

    # Precision vs Confidence
    st.markdown("#### Precision vs Confidence")
    fig_p = go.Figure()
    confs = list(P_CURVE_PER_CLASS[CLASSES_RU[0]].keys())
    colors = ['#000', '#333', '#666', '#999', '#ccc']

    for i, cls_name in enumerate(CLASSES_RU):
        fig_p.add_trace(go.Scatter(
            x=confs, y=list(P_CURVE_PER_CLASS[cls_name].values()),
            name=cls_name, line=dict(color=colors[i], width=2)
        ))
    fig_p.add_vline(x=current_conf, line_dash="dash", line_color="red", annotation_text=f"conf={current_conf}")
    fig_p.update_layout(height=400, xaxis_title="Confidence", yaxis_title="Precision", yaxis_range=[0, 1])
    st.plotly_chart(fig_p, use_container_width=True)

    # Recall vs Confidence
    st.markdown("#### Recall vs Confidence")
    fig_r = go.Figure()
    for i, cls_name in enumerate(CLASSES_RU):
        fig_r.add_trace(go.Scatter(
            x=confs, y=list(R_CURVE_PER_CLASS[cls_name].values()),
            name=cls_name, line=dict(color=colors[i], width=2)
        ))
    fig_r.add_vline(x=current_conf, line_dash="dash", line_color="red", annotation_text=f"conf={current_conf}")
    fig_r.update_layout(height=400, xaxis_title="Confidence", yaxis_title="Recall", yaxis_range=[0, 1])
    st.plotly_chart(fig_r, use_container_width=True)

    # ==========================================
    # Precision-Recall кривая для каждого класса
    # ==========================================
    st.markdown("---")
    st.markdown("### Precision-Recall кривые")

    fig_pr = go.Figure()
    for i, cls_name in enumerate(CLASSES_RU):
        prec_vals = list(P_CURVE_PER_CLASS[cls_name].values())
        rec_vals = list(R_CURVE_PER_CLASS[cls_name].values())
        # Сортируем по recall
        sorted_pairs = sorted(zip(rec_vals, prec_vals))
        rec_sorted, prec_sorted = zip(*sorted_pairs)

        fig_pr.add_trace(go.Scatter(
            x=rec_sorted, y=prec_sorted,
            name=f"{cls_name} (AP={MAP_PER_CLASS[cls_name]:.3f})",
            line=dict(color=colors[i], width=2)
        ))

    # Текущая точка
    fig_pr.add_trace(go.Scatter(
        x=[r], y=[p], mode='markers',
        marker=dict(size=15, color='red', symbol='x'),
        name=f'Текущий порог ({current_conf})'
    ))

    fig_pr.update_layout(
        height=500,
        xaxis_title="Recall", yaxis_title="Precision",
        xaxis_range=[0, 1], yaxis_range=[0, 1]
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    # ==========================================
    # mAP по классам
    # ==========================================
    st.markdown("---")
    st.markdown("### mAP@0.5 по классам")

    cols = st.columns(5)
    for i, cls_name in enumerate(CLASSES_RU):
        map_val = MAP_PER_CLASS[cls_name]
        color = "#000" if map_val > 0.7 else "#666" if map_val > 0.5 else "#999"
        with cols[i]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="number" style="color:{color};">{map_val:.3f}</div>
                <div class="label">{cls_name}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================
# ОТЧЕТ
# ============================================

def render_report_window(detections, image_name, processing_time, model_name, conf_threshold):
    now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    rid = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    total = len(detections)
    status = "ЗАБРАКОВАНО" if total > 0 else "ПРИНЯТО"

    st.markdown(f"""
    <div style="background:#fff;border:1px solid #e5e5e5;padding:35px;margin-top:25px;">
        <h2>ОТЧЕТ КОНТРОЛЯ</h2>
        <p style="color:#888;">{rid} | {now}</p>
        <p><strong>Изображение:</strong> {image_name}</p>
        <p><strong>Модель:</strong> {model_name} | <strong>Порог:</strong> {conf_threshold}</p>
        <p><strong>Время:</strong> {processing_time:.3f} сек</p>
        <p><strong>Статус:</strong> <span style="background:#000;color:#fff;padding:4px 12px;">{status}</span></p>
        <p style="margin-top:20px;"><strong>Обнаружено дефектов: {total}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    if total > 0:
        df = pd.DataFrame(detections)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================
# КОНТРОЛЬ
# ============================================

def render_inspection():
    # Убираем заголовок "Загрузка изображения"
    # st.markdown("### Загрузка изображения")  # ЗАКОММЕНТИРОВАТЬ ЭТУ СТРОКУ

    # CSS для скрытия всех текстов внутри file_uploader
    st.markdown("""
    <style>
        /* Скрыть заголовок "upload" и "200MB per file..." */
        [data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] {
            display: none !important;
        }

        /* Скрыть текст "Browse files" на кнопке */
        .stFileUploader button span {
            display: none !important;
        }

        /* Сделать кнопку компактной, но видимой */
        .stFileUploader button {
            width: auto !important;
            min-width: 120px !important;
            justify-content: center !important;
        }

        /* Добавить свой текст на кнопку */
        .stFileUploader button::before {
            content: "ВЫБРАТЬ ФАЙЛ" !important;
            font-size: 14px !important;
            font-weight: 600 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded)
        # Конвертируем RGBA в RGB
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        st.image(image, use_container_width=True)

        if st.button("АНАЛИЗИРОВАТЬ", type="primary", use_container_width=True):
            return image, True, uploaded.name

    return None, False, ""


# ============================================
# ГЛАВНАЯ
# ============================================

def main():
    st.markdown(
        '<div class="header"><div class="header-logo">TW</div><div class="header-text"><h1>TERMIT WELD CV</h1><p>Система контроля качества сварных соединений</p></div></div>',
        unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<h2 style="font-weight:800;">TERMIT WELD</h2>', unsafe_allow_html=True)
        st.markdown("#### Модель")
        model_choice = st.selectbox("", list(MODEL_PATHS.keys()), index=2, label_visibility="collapsed")
        st.markdown("#### Порог")
        conf = st.slider("", 0.1, 0.9, 0.25, 0.05, label_visibility="collapsed")
        st.session_state['current_conf'] = conf

        # Текущие метрики
        p, r, f1, f2, f3 = get_metrics_for_confidence(conf)
        st.caption(f"F1: {f1:.3f} | F2: {f2:.3f} | F3: {f3:.3f}")

        if model_choice == "Ансамбль":
            models, error = load_ensemble()
        else:
            model, error = load_single_model(MODEL_PATHS[model_choice])
            models = model

        if error:
            st.error(error)
            models = None
        else:
            st.success("Модель готова к работе")

    tab1, tab2, tab3 = st.tabs(["Контроль", "Дашборд", "Информация"])

    with tab1:
        image, analyze, image_name = render_inspection()
        if image and analyze and models:
            try:
                import time
                start = time.time()
                with st.spinner("Анализ..."):
                    if model_choice == "Ансамбль":
                        results = predict_ensemble(models, image, conf)
                    else:
                        results = predict_single(models, image, conf)

                    if results is None:
                        st.error(
                            "Ошибка при анализе изображения. Попробуйте другое изображение или перезагрузите приложение.")
                        st.stop()

                proc_time = time.time() - start

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, use_container_width=True)
                with col2:
                    if results[0] is not None:
                        st.image(results[0].plot(), use_container_width=True)
                    else:
                        st.warning("Результаты не получены")

                detections = []
                if results[0].boxes is not None:
                    model_names = results[0].names
                    unique = {}
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        name = model_names.get(cls_id, f"Класс {cls_id}")
                        c = float(box.conf[0])
                        if name not in unique or c > unique[name]['confidence']:
                            unique[name] = {"class": name, "confidence": c}
                    detections = sorted(unique.values(), key=lambda x: x['confidence'], reverse=True)

                st.markdown("---")
                render_report_window(detections, image_name, proc_time, model_choice, conf)
            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")
                st.info("Попробуйте использовать другое изображение или перезагрузите страницу.")
    with tab2:
        render_dashboard()

    with tab3:
        st.markdown("### О системе Termit Weld CV")

        st.markdown("""
        **Назначение:** Автоматический контроль качества сварных соединений на основе компьютерного зрения.

        ### Классифицируемые дефекты:
        - **Геометрический дефект** - нарушение формы сварного шва
        - **Непровар** - отсутствие сплавления кромок
        - **Трещина** - разрыв металла шва
        - **Пористость** - газовые поры
        - **Брызги** - частицы металла

        ### Метрики качества (mAP@0.5):
        | Дефект | Точность |
        |--------|----------|
        | Геометрический дефект | 0.701 |
        | Непровар | 0.784 |
        | Трещина | 0.456 |
        | Пористость | 0.785 |
        | Брызги | 0.787 |
        | **Среднее** | **0.703** |

        ### Технические требования:
        - Форматы: JPG, JPEG, PNG, BMP
        - Максимальный размер: 200 MB
        """)

        st.markdown("### F-beta Score")
        st.markdown("- **F1 (beta=1)** - баланс Precision и Recall")
        st.markdown("- **F2 (beta=2)** - Recall важнее в 2 раза")
        st.markdown("- **F3 (beta=3)** - максимальный приоритет Recall")

    st.markdown(f'<div class="footer"><p>Termit Weld CV &copy; {datetime.now().year}</p></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
