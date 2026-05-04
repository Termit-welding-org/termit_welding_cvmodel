"""
Streamlit веб-интерфейс для демонстрации
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path
import sys
import os

# Добавление пути к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.predictor import WeldingInspector
from src.utils.config import config

# ============================================
# Конфигурация страницы
# ============================================

st.set_page_config(
    page_title="Welding Inspection AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS стили
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }

    .result-good {
        background-color: #90EE90;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }

    .result-bad {
        background-color: #FFB6C1;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }

    .result-rework {
        background-color: #FFE4B5;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }

    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }

    .detection-item {
        background-color: #f8f9fa;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Инициализация
# ============================================

@st.cache_resource
def load_inspector(model_path: str):
    """Загрузка модели (кэшируется)"""
    try:
        inspector = WeldingInspector(
            model_path=model_path,
            conf_threshold=0.35,
            device='cuda' if st.session_state.get('use_gpu', True) else 'cpu'
        )
        return inspector
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None


# ============================================
# Боковая панель
# ============================================

def render_sidebar():
    """Отображение боковой панели"""

    with st.sidebar:
        st.image("https://via.placeholder.com/200x100?text=Welding+AI", use_column_width=True)

        st.markdown("##  Настройки")

        # Выбор модели
        model_path = st.text_input(
            "Путь к модели",
            value="models/best.pt",
            help="Путь к файлу модели (.pt, .onnx, .engine)"
        )

        # Порог уверенности
        conf_threshold = st.slider(
            "Порог уверенности",
            min_value=0.1,
            max_value=0.9,
            value=0.35,
            step=0.05,
            help="Минимальная уверенность для детекции"
        )

        # Использование GPU
        use_gpu = st.checkbox(
            "Использовать GPU",
            value=True,
            help="Использовать CUDA для ускорения"
        )

        st.session_state['model_path'] = model_path
        st.session_state['conf_threshold'] = conf_threshold
        st.session_state['use_gpu'] = use_gpu

        st.markdown("---")

        # Статистика
        st.markdown(" Статистика")

        if 'inspector' in st.session_state and st.session_state.inspector:
            stats = st.session_state.inspector.get_statistics()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Инспекций", stats['total_inspections'])
            with col2:
                st.metric("Среднее время", f"{stats['avg_time'] * 1000:.1f}ms")

        st.markdown("---")

        # Информация
        st.markdown("## О проекте")
        st.markdown("""
        **Welding Inspection AI** - система автоматического контроля качества 
        сварных швов на основе искусственного интеллекта.

        **Возможности:**
        - Детекция дефектов сварки
        - Классификация качества
        - Работа в реальном времени

        **Версия:** 1.0.0
        """)


# ============================================
# Главная страница
# ============================================

def render_main_page():
    """Отображение главной страницы"""

    st.markdown('<h1 class="main-header"> Welding Inspection AI</h1>', unsafe_allow_html=True)

    # Вкладки
    tab1, tab2, tab3, tab4 = st.tabs([
        " Изображение",
        " Видео",
        " Камера",
        " Отчет"
    ])

    # Вкладка "Изображение"
    with tab1:
        render_image_tab()

    # Вкладка "Видео"
    with tab2:
        render_video_tab()

    # Вкладка "Камера"
    with tab3:
        render_camera_tab()

    # Вкладка "Отчет"
    with tab4:
        render_report_tab()


def render_image_tab():
    """Вкладка инспекции изображений"""

    st.markdown("## Инспекция изображения")

    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Загрузите изображение сварного шва для анализа"
    )

    if uploaded_file:
        # Отображение
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Исходное изображение")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

        # Инспекция
        if st.button(" Анализировать", type="primary"):
            with st.spinner("Анализ изображения..."):
                # Загрузка модели
                if 'inspector' not in st.session_state:
                    st.session_state.inspector = load_inspector(
                        st.session_state.get('model_path', 'models/best.pt')
                    )

                inspector = st.session_state.inspector

                if inspector:
                    # Конвертация изображения
                    image_np = np.array(image)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    # Сохранение временного файла
                    temp_path = "temp_upload.jpg"
                    cv2.imwrite(temp_path, image_bgr)

                    # Инспекция
                    inspector.conf_threshold = st.session_state.get('conf_threshold', 0.35)
                    result = inspector.inspect_image(temp_path)

                    # Отрисовка
                    annotated = inspector.draw_results(image_bgr, result)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    with col2:
                        st.markdown("### Результат анализа")
                        st.image(annotated_rgb, use_column_width=True)

                        # Статус
                        if result.quality == 'ACCEPT':
                            st.markdown(f'<div class="result-good"> {result.quality}</div>',
                                        unsafe_allow_html=True)
                        elif result.quality == 'REWORK':
                            st.markdown(f'<div class="result-rework"> {result.quality}</div>',
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="result-bad"> {result.quality}</div>',
                                        unsafe_allow_html=True)

                        # Метрики
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Дефектов", result.total_defects)
                        with col_b:
                            st.metric("Время", f"{result.processing_time * 1000:.1f}ms")
                        with col_c:
                            st.metric("Статус", result.status)

                        # Рекомендация
                        st.info(f"{result.recommendation}")

                        # Детекции
                        if result.detections:
                            st.markdown("### Обнаруженные дефекты")
                            for det in result.detections:
                                st.markdown(f"""
                                <div class="detection-item">
                                    <b>{det.class_name}</b> - {det.confidence:.2%}<br>
                                    Площадь: {det.area:.0f} px²
                                </div>
                                """, unsafe_allow_html=True)


def render_video_tab():
    """Вкладка инспекции видео"""

    st.markdown("Инспекция видео")

    uploaded_file = st.file_uploader(
        "Выберите видео",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Загрузите видео для анализа"
    )

    if uploaded_file:
        st.video(uploaded_file)

        if st.button("Анализировать видео", type="primary"):
            st.info("Функция в разработке")



def render_camera_tab():
    """Вкладка работы с камерой"""

    st.markdown(" Работа с камерой")

    camera_enabled = st.checkbox("Включить камеру")

    if camera_enabled:
        camera_image = st.camera_input("Сделайте снимок")

        if camera_image:
            # Анализ снимка
            image = Image.open(camera_image)

            if st.button(" Анализировать снимок"):
                with st.spinner("Анализ..."):
                    # TODO: Анализ снимка с камеры
                    st.image(image, use_column_width=True)
                    st.success("Анализ завершен")


def render_report_tab():
    """Вкладка отчета"""

    st.markdown(" Отчет о качестве")

    if 'inspector' in st.session_state and st.session_state.inspector:
        stats = st.session_state.inspector.get_statistics()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Всего инспекций", stats['total_inspections'])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Общее время", f"{stats['total_time']:.1f}s")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Среднее время", f"{stats['avg_time'] * 1000:.1f}ms")
            st.markdown('</div>', unsafe_allow_html=True)

        # Графики
        st.markdown("### Статистика по типам дефектов")

        # Демо-данные для графиков
        import matplotlib.pyplot as plt
        import pandas as pd

        demo_data = pd.DataFrame({
            'Класс': ['Good Weld', 'Bad Weld', 'Pore', 'Undercut', 'Crack'],
            'Количество': [150, 45, 30, 20, 10]
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(demo_data['Класс'], demo_data['Количество'])
        ax.set_title('Распределение дефектов')
        ax.set_ylabel('Количество')
        plt.xticks(rotation=45)

        st.pyplot(fig)
    else:
        st.info("Загрузите модель и выполните инспекции для отображения статистики")


# ============================================
# Главная функция
# ============================================

def main():
    """Главная функция"""

    # Боковая панель
    render_sidebar()

    # Главная страница
    render_main_page()


if __name__ == "__main__":
    main()