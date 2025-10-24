"""
🎯 Dashboard de Interpretabilidad NLP
Comparación interactiva de SHAP vs LIME para análisis de sentimientos
"""

from src.utils.dashboard import comparar_shap_lime, get_model_info, mostrar_prediccion_modelo, visualizar_lime, visualizar_shap
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
from lime.lime_text import LimeTextExplainer
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Interpretabilidad NLP - SHAP vs LIME",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; padding: 0px 24px;}
    .positive-word {color: green; font-weight: bold;}
    .negative-word {color: red; font-weight: bold;}
    .neutral-word {color: gray;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# INICIALIZACIÓN Y CACHÉ
# ============================================================


# @st.cache_resource
def load_model(model_number):
    """Carga el modelo y tokenizer con caché"""

    # Array de opciones de modelos
    model_options = {
        1: "distilbert-base-uncased-finetuned-sst-2-english",
        2: "cardiffnlp/twitter-roberta-base-sentiment",
        3: "j-hartmann/emotion-english-distilroberta-base",
        4: "bhadresh-savani/bert-base-uncased-emotion"
    }
    model_name = model_options[model_number]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Pipeline para predicciones
    classifier = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU
    )

    return model, tokenizer, classifier


def init_explainers(_model, _tokenizer, _classifier, model_number):
    """
    Inicializa SHAP y LIME - ahora depende del model_number y detecta clases dinámicamente

    Args:
        _model: modelo cargado
        _tokenizer: tokenizer del modelo
        _classifier: pipeline del modelo
        model_number: ID del modelo para configuración específica

    Returns:
        tuple: (shap_explainer, lime_explainer, num_classes, class_names)
    """
    # Obtener información del modelo
    num_classes, class_names = get_model_info(model_number, _model)

    # ============================================================
    # SHAP - funciona igual para todos los modelos
    # ============================================================
    shap_explainer = shap.Explainer(_classifier)

    # ============================================================
    # LIME - configuración dinámica según número de clases
    # ============================================================
    lime_explainer = LimeTextExplainer(
        class_names=class_names,  # Ahora dinámico
        split_expression=r'\s+',
        random_state=42
    )

    return shap_explainer, lime_explainer, num_classes, class_names


def predict_proba(texts):
    """Función de predicción para LIME"""
    if isinstance(texts, str):
        texts = [texts]

    predictions = st.session_state.classifier(list(texts))
    proba_array = []

    for pred in predictions:
        if pred['label'] == 'NEGATIVE' or pred['label'] == 'LABEL_0':
            prob_neg = pred['score']
            prob_pos = 1 - pred['score']
        else:
            prob_pos = pred['score']
            prob_neg = 1 - pred['score']
        proba_array.append([prob_neg, prob_pos])

    return np.array(proba_array)


# ============================================================
# HEADER Y DESCRIPCIÓN
# ============================================================
st.title("🔍 Dashboard de Interpretabilidad NLP")
st.markdown("### Comparación de SHAP vs LIME para Análisis de Sentimientos")

with st.expander("ℹ️ Acerca de este Dashboard", expanded=False):
    st.markdown("""
    Este dashboard permite comparar dos métodos de interpretabilidad:

    **SHAP (SHapley Additive exPlanations)**
    - ✅ Base matemática sólida (teoría de juegos)
    - ✅ Resultados determinísticos y consistentes
    - ❌ Más lento (~10-30 segundos por texto)

    **LIME (Local Interpretable Model-agnostic Explanations)**
    - ✅ Muy rápido (~1-3 segundos por texto)
    - ✅ Fácil de entender e interpretar
    - ❌ Resultados estocásticos (pueden variar)
    """)


# ============================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuración")

    # Diccionario con las opciones y sus valores asociados
    model_dict = {
        "DistilBERT (2 clases: positivo/negativo)": 1,
        "RoBERTa (3 clases: negativo/neutral/positivo)": 2,
        "DistilRoBERTa  base (7 emociones)": 3,
        "BERT Emotion (6 emociones)": 4
    }

    # Selector de modelos para pasar load_model(model_number)
    st.subheader("Modelo de Lenguaje")
    model_choice = st.selectbox(
        "Seleccionar modelo:", list(model_dict.keys()), index=0)
    # Obtengo el id del modelo
    model_number = model_dict[model_choice]

    # Método de explicación
    st.subheader("Método de Interpretabilidad")
    method = st.radio(
        "Seleccionar método:",
        ["Ambos (SHAP + LIME)", "Solo SHAP", "Solo LIME"],
        index=0
    )

    # Parámetros LIME (también aplican a SHAP para consistencia)
    st.subheader("📊 Parámetros de Visualización")

    # Número de características (común para ambos métodos)
    num_features_lime = st.slider(
        "Número de características:",
        min_value=5,
        max_value=20,
        value=10,
        help="Cuántas palabras mostrar en las visualizaciones"
    )

    # Parámetros específicos de LIME
    if "LIME" in method or method == "Ambos (SHAP + LIME)":
        num_samples_lime = st.slider(
            "Número de muestras LIME:",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Más muestras = más preciso pero más lento"
        )
    else:
        # Valor por defecto cuando no se usa LIME
        num_samples_lime = 1000

    # Ejemplos predefinidos
    st.subheader("📝 Textos de Ejemplo")
    example_texts = {
        "Positivo claro": "This movie is absolutely fantastic! Best film I've seen all year. Highly recommended!",
        "Negativo claro": "Terrible movie. Complete waste of time and money. Boring and poorly acted.",
        "Mixto/Ambiguo": "The movie had some good moments but overall it was disappointing.",
        "Sarcástico": "Oh great, another superhero movie. Just what the world needed.",
        "Neutral": "The movie was okay. Nothing special but watchable."
    }

    selected_example = st.selectbox(
        "Cargar ejemplo:",
        [""] + list(example_texts.keys())
    )
# ============================================================
# ÁREA PRINCIPAL
# ============================================================

# Verificar si el modelo cambió
if 'current_model' not in st.session_state or st.session_state.current_model != model_number:
    with st.spinner(f"🚀 Cargando modelo {model_choice}..."):
        model, tokenizer, classifier = load_model(model_number)
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.classifier = classifier
        st.session_state.current_model = model_number

        # IMPORTANTE: Recrear los explainers con el NUEVO modelo
        shap_exp, lime_exp, num_classes, class_names = init_explainers(
            model, tokenizer, classifier, model_number
        )
        st.session_state.shap_explainer = shap_exp
        st.session_state.lime_explainer = lime_exp

        # Guardar explainers y configuración en session_state
        st.session_state.shap_explainer = shap_exp
        st.session_state.lime_explainer = lime_exp
        st.session_state.num_classes = num_classes
        st.session_state.class_names = class_names

        st.success(f"✅ Modelo cargado: {model_choice} ({num_classes} clases)")


# Input de texto
col1, col2 = st.columns([3, 1])
with col1:
    if selected_example and selected_example in example_texts:
        default_text = example_texts[selected_example]
    else:
        default_text = ""

    input_text = st.text_area(
        "📝 Ingresa el texto a analizar:",
        value=default_text,
        height=100,
        placeholder="Escribe o pega aquí el texto que quieres analizar..."
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button(
        "🔍 Analizar", type="primary", use_container_width=True)

# ============================================================
# ANÁLISIS
# ============================================================

if analyze_button and input_text:
    # Predicción base
    with st.spinner("Realizando predicción..."):
        prediction = st.session_state.classifier(input_text)[0]
        proba = predict_proba(input_text)[0]

    # Mostrar predicción
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    mostrar_prediccion_modelo(input_text)

    st.markdown("---")

    # Tabs para explicaciones
    if method == "Ambos (SHAP + LIME)":
        tab1, tab2, tab3 = st.tabs(["📊 Comparación", "🔷 SHAP", "🔶 LIME"])
    elif method == "Solo SHAP":
        tab1 = st.tabs(["🔷 SHAP"])[0]
    else:
        tab1 = st.tabs(["🔶 LIME"])[0]

    # ============================================================
    # TAB COMPARATIVO
    # ============================================================
    if method == "Ambos (SHAP + LIME)":
        with tab1:
            # Llamar a la función de comparación
            shap_values, lime_explanation, shap_time, lime_time = comparar_shap_lime(
                input_text=input_text,
                predict_proba=predict_proba,
                num_features_lime=num_features_lime,
                num_samples_lime=num_samples_lime
            )

    # ============================================================
    # TAB SHAP SOLO
    # ============================================================
    if method == "Solo SHAP" or method == "Ambos (SHAP + LIME)":
        tab_shap = tab2 if method == "Ambos (SHAP + LIME)" else tab1
        with tab_shap:
            # Calcular SHAP si es modo "Solo SHAP"
            if method == "Solo SHAP":
                with st.spinner("Calculando SHAP..."):
                    shap_values = st.session_state.shap_explainer([input_text])

            # Llamar a la función de visualización
            visualizar_shap(
                shap_values=shap_values,
                input_text=input_text,
                model_choice=model_choice,
                model_number=model_number,
                method=method,
                num_features=num_features_lime
            )
    # ============================================================
    # TAB LIME SOLO
    # ============================================================
    if method == "Solo LIME" or method == "Ambos (SHAP + LIME)":
        tab_lime = tab3 if method == "Ambos (SHAP + LIME)" else tab1
        with tab_lime:
            # Calcular LIME si es modo "Solo LIME"
            if method == "Solo LIME":
                with st.spinner("Calculando LIME..."):
                    lime_explanation = st.session_state.lime_explainer.explain_instance(
                        input_text,
                        predict_proba,
                        num_features=num_features_lime,
                        num_samples=num_samples_lime
                    )

            # Llamar a la función de visualización
            visualizar_lime(
                lime_explanation=lime_explanation,
                num_features_lime=num_features_lime
            )
# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Dashboard de Interpretabilidad NLP | Módulo II</p>
    <p>Comparación de métodos SHAP y LIME para análisis de sentimientos</p>
</div>
""", unsafe_allow_html=True)
