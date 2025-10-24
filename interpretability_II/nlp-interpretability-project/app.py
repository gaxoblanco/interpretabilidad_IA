"""
🎯 Dashboard de Interpretabilidad NLP
Comparación interactiva de SHAP vs LIME para análisis de sentimientos
"""

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


def init_explainers(_model, _tokenizer, _classifier):
    """Inicializa SHAP y LIME - ahora depende del model_number"""
    # SHAP
    shap_explainer = shap.Explainer(_classifier)

    # LIME
    lime_explainer = LimeTextExplainer(
        class_names=['NEGATIVE', 'POSITIVE'],
        split_expression=r'\s+',
        random_state=42
    )

    return shap_explainer, lime_explainer


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

    # Parámetros LIME
    if "LIME" in method or method == "Ambos (SHAP + LIME)":
        st.subheader("📊 Parámetros LIME")
        num_samples_lime = st.slider(
            "Número de muestras:",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Más muestras = más preciso pero más lento"
        )
        num_features_lime = st.slider(
            "Número de características:",
            min_value=5,
            max_value=20,
            value=10,
            help="Cuántas palabras mostrar"
        )

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
        shap_exp, lime_exp = init_explainers(model, tokenizer, classifier)
        st.session_state.shap_explainer = shap_exp
        st.session_state.lime_explainer = lime_exp


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

    with col2:
        sentiment = prediction['label']
        confidence = prediction['score']

        # Color según sentimiento
        if sentiment == 'POSITIVE':
            color = "green"
            emoji = "😊"
        else:
            color = "red"
            emoji = "😔"

        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
            <h2 style='color: {color};'>{emoji} {sentiment}</h2>
            <h4 style='color: black;'>Confianza: {confidence:.1%}</h4>
            <p style='color: gray;'>Negativo: {proba[0]:.1%} | Positivo: {proba[1]:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Tabs para explicaciones
    if method == "Ambos (SHAP + LIME)":
        tab1, tab2, tab3 = st.tabs(["📊 Comparación", "🔷 SHAP", "🔶 LIME"])
    elif method == "Solo SHAP":
        tab1 = st.tabs(["🔷 SHAP"])[0]
    else:
        tab1 = st.tabs(["🔶 LIME"])[0]

    # ============================================================
    # TAB COMPARACIÓN
    # ============================================================
    if method == "Ambos (SHAP + LIME)":
        with tab1:
            col1, col2 = st.columns(2)

            # SHAP
            with col1:
                st.markdown("### 🔷 SHAP")
                with st.spinner("Calculando SHAP (puede tomar 10-30 segundos)..."):
                    start_time = time.time()
                    shap_values = st.session_state.shap_explainer([input_text])
                    shap_time = time.time() - start_time

                    # Extraer valores - detectar qué clase usar
                    tokens = shap_values[0].data
                    # Detectar si es un caso mayormente positivo o negativo
                    sum_positive = np.sum(shap_values[0].values[:, 1])
                    if sum_positive > 1:
                        # Caso positivo: usar valores para clase POSITIVE
                        values = shap_values[0].values[:, 1]
                        class_label = "POSITIVE"
                    else:
                        # Caso negativo: usar valores para clase NEGATIVE
                        values = shap_values[0].values[:, 0]
                        values = -values  # Ahora valores negativos serán rojos
                        class_label = "NEGATIVE"

                    # Top palabras
                    top_indices = np.argsort(
                        np.abs(values))[-num_features_lime:][::-1]

                    # Crear DataFrame
                    shap_df = pd.DataFrame({
                        'Palabra': [tokens[i] for i in top_indices],
                        'Importancia': [values[i] for i in top_indices]
                    })

                    # Visualizar
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['green' if v >
                              0 else 'red' for v in shap_df['Importancia']]
                    ax.barh(range(len(shap_df)),
                            shap_df['Importancia'], color=colors, alpha=0.7)
                    ax.set_yticks(range(len(shap_df)))
                    ax.set_yticklabels(shap_df['Palabra'])
                    ax.set_xlabel('Importancia SHAP')
                    ax.set_title(f'SHAP ({shap_time:.1f}s)')
                    ax.axvline(x=0, color='black',
                               linestyle='-', linewidth=0.5)
                    st.pyplot(fig)

                    st.info(f"⏱️ Tiempo: {shap_time:.1f} segundos")

            # LIME
            with col2:
                st.markdown("### 🔶 LIME")
                with st.spinner("Calculando LIME..."):
                    start_time = time.time()
                    lime_explanation = st.session_state.lime_explainer.explain_instance(
                        input_text,
                        predict_proba,
                        num_features=num_features_lime,
                        num_samples=num_samples_lime
                    )
                    lime_time = time.time() - start_time

                    # Extraer valores
                    exp_list = lime_explanation.as_list()[:num_features_lime]

                    # Crear DataFrame
                    lime_df = pd.DataFrame({
                        'Palabra': [x[0] for x in exp_list],
                        'Importancia': [x[1] for x in exp_list]
                    })

                    # Visualizar
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['green' if v >
                              0 else 'red' for v in lime_df['Importancia']]
                    ax.barh(range(len(lime_df)),
                            lime_df['Importancia'], color=colors, alpha=0.7)
                    ax.set_yticks(range(len(lime_df)))
                    ax.set_yticklabels(lime_df['Palabra'])
                    ax.set_xlabel('Importancia LIME')
                    ax.set_title(f'LIME ({lime_time:.1f}s)')
                    ax.axvline(x=0, color='black',
                               linestyle='-', linewidth=0.5)
                    st.pyplot(fig)

                    st.info(f"⏱️ Tiempo: {lime_time:.1f} segundos")

            # Resumen comparativo
            st.markdown("---")
            st.markdown("### 📊 Resumen Comparativo")

            comparison_df = pd.DataFrame({
                'Métrica': ['Tiempo de cómputo', 'Speedup'],
                'SHAP': [f"{shap_time:.1f}s", "1x"],
                'LIME': [f"{lime_time:.1f}s", f"{shap_time/lime_time:.1f}x"]
            })
            st.table(comparison_df)

    # ============================================================
    # TAB SHAP SOLO
    # ============================================================
    if method == "Solo SHAP" or method == "Ambos (SHAP + LIME)":
        tab_shap = tab2 if method == "Ambos (SHAP + LIME)" else tab1
        with tab_shap:
            st.markdown("### Análisis detallado con SHAP")
            st.markdown("#### Modelo utilizado: " + model_choice)
            # print del model_number
            st.markdown(f"#### ID del modelo: {model_number}")

            if method == "Solo SHAP":
                with st.spinner("Calculando SHAP..."):
                    shap_values = st.session_state.shap_explainer([input_text])

            # Waterfall plot con manejo robusto según modelo
            st.markdown("#### Contribución Acumulativa")

            try:
                # Determinar número de clases del modelo
                num_classes = shap_values[0].values.shape[1] if len(
                    shap_values[0].values.shape) > 1 else 1

                # Para modelos de 2 clases (binario)
                if num_classes == 2:
                    values_for_positive = shap_values[0].values[:, 1]
                    is_positive_case = np.sum(values_for_positive) < 0

                    fig, ax = plt.subplots(figsize=(10, 6))
                    if is_positive_case:
                        shap.plots.waterfall(
                            shap_values[0, :, 1], max_display=10, show=False)
                    else:
                        shap.plots.waterfall(
                            shap_values[0, :, 0], max_display=10, show=False)
                    st.pyplot(fig)
                    plt.close()

                # Para modelos multiclase (3+ clases)
                else:
                    # Usar gráfico de barras para multiclase
                    st.info(
                        f"Modelo con {num_classes} clases - Mostrando importancia promedio")

                    tokens = shap_values[0].data
                    # Promedio absoluto a través de todas las clases
                    avg_importance = np.mean(
                        np.abs(shap_values[0].values), axis=1)

                    # Top 10 tokens
                    top_indices = np.argsort(avg_importance)[-10:][::-1]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_tokens = [tokens[i] for i in top_indices]
                    top_values = [avg_importance[i] for i in top_indices]

                    ax.barh(range(len(top_tokens)), top_values,
                            color='steelblue', alpha=0.7)
                    ax.set_yticks(range(len(top_tokens)))
                    ax.set_yticklabels(top_tokens)
                    ax.set_xlabel('Importancia SHAP Promedio')
                    ax.set_title(
                        f'Top 10 Palabras - Modelo {num_classes} clases')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            except Exception as e:
                st.error(f"Error generando gráfico: {str(e)[:200]}")
                # Fallback simple
                st.write(
                    "Valores SHAP calculados pero visualización no disponible para esta configuración")
            # Información adicional
            st.markdown("#### ℹ️ Sobre SHAP")
            st.info("""
            SHAP utiliza valores de Shapley de la teoría de juegos para asignar
            importancia a cada palabra. Garantiza consistencia y aditividad en
            las explicaciones.
            """)

    # ============================================================
    # TAB LIME SOLO
    # ============================================================
    if method == "Solo LIME" or method == "Ambos (SHAP + LIME)":
        tab_lime = tab3 if method == "Ambos (SHAP + LIME)" else tab1
        with tab_lime:
            st.markdown("### Análisis detallado con LIME")

            if method == "Solo LIME":
                with st.spinner("Calculando LIME..."):
                    lime_explanation = st.session_state.lime_explainer.explain_instance(
                        input_text,
                        predict_proba,
                        num_features=num_features_lime,
                        num_samples=num_samples_lime
                    )

            # Tabla de palabras
            st.markdown("#### Tabla de Importancia")
            exp_df = pd.DataFrame(
                lime_explanation.as_list()[:num_features_lime],
                columns=['Palabra', 'Importancia']
            )
            exp_df['Impacto'] = exp_df['Importancia'].apply(
                lambda x: '🟢 Positivo' if x > 0 else '🔴 Negativo'
            )
            st.dataframe(exp_df, use_container_width=True)

            # Información adicional
            st.markdown("#### ℹ️ Sobre LIME")
            st.info("""
            LIME aproxima el modelo complejo con un modelo lineal local, 
            perturbando el texto de entrada y observando cómo cambian las 
            predicciones.
            """)

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
