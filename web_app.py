#!/usr/bin/env python3
"""
Termit Weld CV - Профессиональная система контроля сварки
Современный минималистичный дизайн + Дашборд + Ансамбль моделей
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

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
    p, li, span, div, label { font-family: 'Inter', sans-serif !important; }

    .stApp { background: #ffffff; }

    .header {
        background: #000000;
        padding: 30px 35px;
        border-radius: 0;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
    }

    .header-logo {
        background: #ffffff;
        color: #000000;
        font-size: 1.6rem;
        font-weight: 800;
        width: 55px;
        height: 55px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 20px;
        font-family: 'Inter', sans-serif;
    }

    .header-text h1 { color: #ffffff !important; margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; }
    .header-text p { color: #999999 !important; margin: 5px 0 0 0; font-size: 1rem; font-weight: 400; }

    .stat-card {
        background: #ffffff;
        padding: 25px;
        border-radius: 0;
        border: 1px solid #e5e5e5;
        text-align: left;
    }

    .stat-card .number {
        font-size: 2.8rem;
        font-weight: 800;
        color: #000000;
        font-family: 'Inter', sans-serif;
    }

    .stat-card .label { color: #888888 !important; font-size: 0.85rem; margin-top: 5px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }

    .result-good {
        background: #f5f5f5;
        padding: 25px;
        border-left: 3px solid #000000;
    }

    .result-bad {
        background: #f5f5f5;
        padding: 25px;
        border-left: 3px solid #000000;
    }

    .stButton > button {
        background: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 0 !important;
        padding: 12px 30px !important;
        font-size: 14px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover { background: #333333 !important; }

    [data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #e5e5e5; }
    [data-testid="stSidebar"] * { color: #000000 !important; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #000000 !important; font-weight: 700; }
    
    [data-testid="stFileUploader"] button {
    font-size: 0 !important;
    }

    [data-testid="stFileUploader"] button::after {
        content: "ВЫБРАТЬ ФАЙЛ" !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
    }
    .footer {
        border-top: 1px solid #e5e5e5;
        padding: 40px 0;
        margin-top: 50px;
        text-align: center;
        color: #888888;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 0; background: transparent; padding: 0; border-bottom: 1px solid #e5e5e5; }
    .stTabs [data-baseweb="tab"] { 
        background: transparent; border-radius: 0; padding: 15px 25px; 
        color: #888888 !important; font-weight: 500; border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] { 
        background: transparent !important; color: #000000 !important; 
        border-bottom: 2px solid #000000; font-weight: 700;
    }

    img { max-height: 450px !important; object-fit: contain !important; }

    .report-window {
        background: #ffffff;
        border: 1px solid #e5e5e5;
        padding: 35px;
        margin-top: 25px;
    }

    .report-header { border-bottom: 1px solid #e5e5e5; padding-bottom: 20px; margin-bottom: 25px; }
    .report-header h2 { font-weight: 700; font-size: 1.4rem; }
    .report-section { margin: 20px 0; padding: 15px 0; border-bottom: 1px solid #f0f0f0; }

    .report-badge { 
        display: inline-block; padding: 4px 12px; font-weight: 600; font-size: 0.8rem; 
        text-transform: uppercase; letter-spacing: 1px;
    }
    .badge-accept { background: #000000; color: #ffffff; }
    .badge-reject { background: #000000; color: #ffffff; }

    .report-table { width: 100%; border-collapse: collapse; }
    .report-table th { background: #000000; color: #ffffff; padding: 10px; text-align: left; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .report-table td { padding: 10px; border-bottom: 1px solid #e5e5e5; }

    input, select, textarea { border-radius: 0 !important; border: 1px solid #e5e5e5 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================
# КЛАССЫ ДЕФЕКТОВ
# ============================================

CLASSES_RU = [
    "Геометрический дефект",
    "Непровар",
    "Трещина",
    "Пористость",
    "Брызги"
]

# ============================================
# МОДЕЛИ
# ============================================

MODEL_PATHS = {
    "Best Nano": "models/best_nano.pt",
    "Best Small": "models/best_small.pt",
    "Ансамбль": "ensemble"
}


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


def predict_single(model, image, conf):
    return model.predict(np.array(image), conf=conf, verbose=False)


def predict_ensemble(models, image, conf):
    nano, small = models
    r1 = nano.predict(np.array(image), conf=conf, verbose=False)
    r2 = small.predict(np.array(image), conf=conf, verbose=False)
    return r1 if r1[0].boxes else r2


# ============================================
# ДАШБОРД
# ============================================

def render_dashboard():
    st.markdown('<h2 style="font-weight:800; font-size:2rem;">Дашборд</h2>', unsafe_allow_html=True)

    if 'dashboard_data' not in st.session_state:
        dates = pd.date_range(datetime.now() - timedelta(days=29), periods=30, freq='D')
        st.session_state.dashboard_data = pd.DataFrame({
            'Дата': dates,
            'Проверено': np.random.randint(20, 80, 30),
            'Дефектов': np.random.randint(2, 15, 30),
            'Принято': np.random.randint(15, 65, 30)
        })
        st.session_state.dashboard_data['% брака'] = (
                st.session_state.dashboard_data['Дефектов'] /
                st.session_state.dashboard_data['Проверено'] * 100
        ).round(1)

    df = st.session_state.dashboard_data

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="stat-card"><div class="number">{df["Проверено"].sum()}</div><div class="label">Проверок</div></div>',
            unsafe_allow_html=True)
    with col2:
        rate = (df['Дефектов'].sum() / df['Проверено'].sum() * 100).round(1)
        st.markdown(f'<div class="stat-card"><div class="number">{rate}%</div><div class="label">Брак</div></div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown(
            f'<div class="stat-card"><div class="number">{df["Принято"].sum()}</div><div class="label">Принято</div></div>',
            unsafe_allow_html=True)
    with col4:
        st.markdown(
            f'<div class="stat-card"><div class="number">{round(np.random.uniform(0.08, 0.15), 3)}с</div><div class="label">Время</div></div>',
            unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df['Дата'], y=df['Проверено'], name='Проверено', line=dict(color='#000000', width=2)))
        fig.add_trace(go.Scatter(x=df['Дата'], y=df['Дефектов'], name='Дефекты', line=dict(color='#888888', width=1)))
        fig.add_trace(go.Scatter(x=df['Дата'], y=df['Принято'], name='Принято', line=dict(color='#cccccc', width=1)))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        dd = pd.DataFrame({'Тип': CLASSES_RU, 'Количество': np.random.randint(5, 30, 5)})
        fig = px.bar(dd, x='Тип', y='Количество', color_discrete_sequence=['#000000'] * 5)
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)


def render_report_window(detections, image_name, processing_time, model_name, conf_threshold):
    now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    rid = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    total = len(detections)
    status = "ЗАБРАКОВАНО" if total > 0 else "ПРИНЯТО"

    cc = {}
    for d in detections: cc[d['class']] = cc.get(d['class'], 0) + 1

    html = f"""
    <div class="report-window">
        <div class="report-header"><h2>ОТЧЕТ КОНТРОЛЯ</h2><p style="color:#888;">{rid} &middot; {now}</p></div>
        <div class="report-section">
            <table style="width:100%;">
                <tr><td width="200"><strong>Изображение</strong></td><td>{image_name}</td></tr>
                <tr><td><strong>Модель</strong></td><td>{model_name}</td></tr>
                <tr><td><strong>Порог</strong></td><td>{conf_threshold}</td></tr>
                <tr><td><strong>Время</strong></td><td>{processing_time:.3f} сек</td></tr>
                <tr><td><strong>Статус</strong></td><td><span class="report-badge badge-{'accept' if total == 0 else 'reject'}">{status}</span></td></tr>
            </table>
        </div>
        <div class="report-section">
            <p><strong>Обнаружено дефектов: {total}</strong></p>
    """

    if total > 0:
        html += """<table class="report-table"><tr><th>Тип дефекта</th><th>Кол-во</th></tr>"""
        for cn, cnt in cc.items(): html += f"<tr><td>{cn}</td><td>{cnt}</td></tr>"
        html += "</table>"

    html += """</div></div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_inspection():
    st.markdown("### Загрузка изображения")
    uploaded = st.file_uploader(
        "Выберите файл",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="visible"
    )

    if not uploaded:
        st.info("Загрузите изображение сварного шва для анализа")
        return None, False, ""

    image = Image.open(uploaded)
    st.image(image, use_container_width=True)

    analyze_btn = st.button("АНАЛИЗИРОВАТЬ", type="primary", use_container_width=True)
    if analyze_btn: return image, True, uploaded.name
    return None, False, ""


def main():
    current_year = datetime.now().year
    st.markdown(
        '<div class="header"><div class="header-logo">TW</div><div class="header-text"><h1>TERMIT WELD CV</h1><p>Система контроля качества сварных соединений</p></div></div>',
        unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            '<div style="padding:20px 0;"><h2 style="font-weight:800;">TERMIT WELD</h2><p style="color:#888;">Computer Vision</p></div>',
            unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### Модель")
        model_choice = st.selectbox("", list(MODEL_PATHS.keys()), index=2, label_visibility="collapsed")
        st.markdown("#### Порог")
        conf = st.slider("", 0.1, 0.9, 0.25, 0.05, label_visibility="collapsed")

        if model_choice == "Ансамбль":
            models, error = load_ensemble()
        else:
            model, error = load_single_model(MODEL_PATHS[model_choice]); models = model

        if error:
            st.error(error); models = None
        else:
            st.success("Готово")

        st.markdown("---")
        st.markdown("#### Классы дефектов")
        for name in CLASSES_RU: st.markdown(f"- {name}")

    tab1, tab2, tab3 = st.tabs(["Контроль", "Дашборд", "Информация"])

    with tab1:
        image, analyze, image_name = render_inspection()
        if image and analyze and models:
            import time
            start = time.time()
            with st.spinner("Анализ..."):
                results = predict_ensemble(models, image, conf) if model_choice == "Ансамбль" else predict_single(
                    models, image, conf)
            proc_time = time.time() - start

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, use_container_width=True)
            with col2:
                st.image(results[0].plot(), use_container_width=True)

            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    detections.append({"class": CLASSES_RU[cls_id] if cls_id < len(CLASSES_RU) else "?",
                                       "confidence": float(box.conf[0])})

            st.markdown("---")
            render_report_window(detections, image_name, proc_time, model_choice, conf)

    with tab2:
        render_dashboard()
    with tab3:
        st.markdown(
            "### Termit Weld CV\nСистема компьютерного зрения для контроля сварных соединений.\n\n**Модели:** Nano, Small, Ансамбль - Nano+Small.\n\n")

    st.markdown(f'<div class="footer"><p>Termit Weld CV &copy; {current_year}</p></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()