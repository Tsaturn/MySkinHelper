import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import albumentations as A
import tensorflow.keras.backend as K
import pandas as pd
import plotly.express as px
import os
import gdown
import keras.src.models.functional
from keras.models import load_model

st.set_page_config(
    page_title="My Skin Helper",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="auto"
)

IMG_SIZE = 380
MODEL_PATH = 'model_not_exists.keras' 
GOOGLE_DRIVE_FILE_ID = "1HAF2H6WdPtNd_FTIdFJGrei1wC8GV0KJ"
    
DISEASE_DESCRIPTIONS = {
    'Acitinic Keratosis': {
        'description': 'Актинический кератоз — предраковое состояние кожи, вызванное длительным воздействием ультрафиолета.',
        'symptoms': 'Шероховатые, чешуйчатые пятна на коже, чаще на лице, ушах, руках.',
        'treatment': 'Криотерапия, лазерное удаление, местные препараты (5-фторурацил).'
    },
    'Carcinoma': {
        'description': 'Базальноклеточная или плоскоклеточная карцинома — распространенные виды рака кожи.',
        'symptoms': 'Жемчужные узелки, язвы, красные чешуйчатые пятна.',
        'treatment': 'Хирургическое удаление, лучевая терапия, MOHS-хирургия.'
    },
    'Dermatofibroma': {
        'description': 'Доброкачественное образование кожи, чаще появляется после укусов или травм.',
        'symptoms': 'Твердый узелок коричневого или красного цвета, обычно на ногах.',
        'treatment': 'Не требует лечения, но можно удалить хирургически.'
    },
    'Melanoma': {
        'description': 'Самый опасный вид рака кожи, может метастазировать.',
        'symptoms': 'Асимметричная родинка с неровными краями, изменяющая цвет и размер.',
        'treatment': 'Срочное хирургическое удаление, иммунотерапия, таргетная терапия.'
    },
    'Nevus': {
        'description': 'Обычная родинка, доброкачественное скопление меланоцитов.',
        'symptoms': 'Коричневые, черные или розовые пятна, плоские или выпуклые.',
        'treatment': 'Обычно не требует лечения, но нужно наблюдать за изменениями.'
    },
    'Seborrheic Keratosis': {
        'description': 'Доброкачественные новообразования, часто появляются с возрастом.',
        'symptoms': 'Коричневые, черные или желтые "наросты" с восковой текстурой.',
        'treatment': 'Криотерапия, лазерное удаление, если мешают.'
    },
    'Vascular Lesion': {
        'description': 'Сосудистые образования (гемангиомы, ангиомы).',
        'symptoms': 'Красные или фиолетовые пятна, могут быть выпуклыми.',
        'treatment': 'Лазерная терапия, склеротерапия, наблюдение.'
    }
}

@tf.keras.utils.register_keras_serializable()
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha_list = tf.cast([0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25], tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_factor = y_true * alpha_list + (1 - y_true) * (1 - alpha_list)
    modulating_factor = (1.0 - p_t)**gamma
    loss = -alpha_factor * modulating_factor * K.log(p_t)
    return tf.reduce_sum(loss, axis=-1)

@st.cache_resource
def load_downloaded_model(model_path):
    url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
    except Exception as e:
        st.error(f"Ошибка при скачивании модели: {e}")
        return None

    try:
        custom_objects = {'focal_loss': focal_loss}  # если используется
        model = load_model(MODEL_PATH, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

model = load_downloaded_model(MODEL_PATH)
classes = list(DISEASE_DESCRIPTIONS.keys())

def preprocess_image(image):
    transformator = A.Compose([A.Resize(width=IMG_SIZE, height=IMG_SIZE)])
    image = np.array(image)
    transformed = transformator(image=image)
    return np.expand_dims(transformed['image'], axis=0)

st.title("My skin helper")
st.markdown("""Загрузите изображение кожи для анализа. Приложение поможет определить возможное* заболевание.""")
st.markdown("""&nbsp;&nbsp;&nbsp;&nbsp;* не является диагнозом и несёт ознакомительную информацию. Требуется консультация со специалистом""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Выберите файл", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image_container_width = 600
            image_width = int(image_container_width) * 0.5
            st.image(image, caption="Загруженное изображение", width=int(image_width))
            image_height = image.size[1] * (image_width / image.size[0])
        
        with col2:
            with st.spinner("Анализ изображения..."):
                processed_image = preprocess_image(image)
                if model is not None:
                    prediction = model.predict(processed_image)
                    predicted_class = classes[np.argmax(prediction)]
                    confidence = prediction[0][np.argmax(prediction)]
                    
                    df = pd.DataFrame({
                        'Заболевание': classes,
                        'Вероятность (%)': (prediction[0] * 100).round(2)  
                    })
                    
                    fig = px.bar(
                        df,
                        x='Вероятность (%)',
                        y='Заболевание',
                        orientation='h',
                        text='Вероятность (%)',
                        color='Вероятность (%)',
                        color_continuous_scale='Blues',
                        labels={'Вероятность (%)': 'Вероятность, %'},
                    )
                    
                    fig.update_traces(
                        texttemplate='<b>%{text:.2f}%</b>',
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Вероятность: <b>%{x:.2f}%</b>',
                        textfont_size=14,
                        marker_line_width=0,
                        textfont=dict(color='white', size=14),
                    )
                    
                    fig.update_layout(
                        height=int(image_height), 
                        showlegend=False,
                        xaxis_title=None,
                        yaxis_title=None,
                        margin=dict(l=0, r=0, t=30, b=0),
                        font=dict(
                            family="Arial",
                            size=14,  # Общий размер шрифта
                            color="black"
                        ),
                        yaxis=dict(
                            tickfont=dict(
                                size=14,  # Размер шрифта названий заболеваний
                                weight='bold'  # Жирный шрифт
                            )
                        ),
                        xaxis=dict(
                            tickfont=dict(
                                size=12  # Размер шрифта значений оси X
                            )
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',  # Прозрачный фон
                        paper_bgcolor='rgba(0,0,0,0)',
                        hoverlabel=dict(
                            bgcolor="white",
                            font_size=14,
                            font_family="Arial"
                        )
                    )
                    
                    fig.update_traces(width=0.8)
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"### Наиболее вероятный диагноз: **{predicted_class}** (Вероятность: {confidence:.2%})")
        
        if predicted_class in ['Melanoma', 'Carcinoma', 'Acitinic Keratosis']:
            st.warning("**Срочно обратитесь к дерматологу!**")
        
        st.markdown("---")
        st.subheader("Описание заболеваний")
        for disease, info in DISEASE_DESCRIPTIONS.items():
            with st.expander(f"**{disease}**"):
                st.markdown(f"**Описание:** {info['description']}")
                st.markdown(f"**Симптомы:** {info['symptoms']}")
                st.markdown(f"**Лечение:** {info['treatment']}")
    
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
