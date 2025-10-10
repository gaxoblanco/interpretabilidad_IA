"""
Dashboard Interactivo de Análisis de Riesgo Crediticio
=======================================================

Objetivo: Comunicar interpretabilidad de modelos ML a usuarios no técnicos
Fase 4 del proyecto interpretability_I

Funcionalidades:
1. Evaluar nuevos clientes (predicción + explicación SHAP + counterfactuals
Dashboard Interactivo de Análisis de Riesgo Crediticio
=======================================================

Objetivo: Comunicar interpretabilidad de modelos ML a usuarios no técnicos
Fase 4 del proyecto interpretability_I

Funcionalidades:
1. Evaluar nuevos clientes (predicción + explicación SHAP + counterfactuals)
2. Explorar comportamiento del modelo (SHAP summary, dependence plots)
3. Análisis en batch (subir CSV con múltiples clientes)

Stack: Streamlit + SHAP + XGBoost + Plotly
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# ==============================================================================
# CONFIGURACIÓN INICIAL
# ==============================================================================

st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CARGA DE MODELO Y DATOS (con cache para performance)
# ==============================================================================


@st.cache_resource
def load_model():
    """
    Carga el modelo XGBoost entrenado y crea el explainer SHAP

    Returns:
        tuple: (modelo, explainer, label_encoders, feature_names)
    """
    try:
        # Intentar cargar desde la carpeta models/
        model = joblib.load('models/xgboost_model.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')

        # Crear explainer SHAP (TreeExplainer optimizado para XGBoost)
        explainer = shap.TreeExplainer(model)

        # Feature names (20 features del German Credit Data)
        # IMPORTANTE: Estos nombres deben coincidir EXACTAMENTE con los del modelo entrenado
        feature_names = [
            'checking_status', 'duration', 'credit_history', 'purpose',
            'credit_amount', 'savings_status', 'employment', 'installment_rate',
            'personal_status', 'other_parties', 'residence_since',
            'property_magnitude', 'age', 'other_payment_plans', 'housing',
            'existing_credits', 'job', 'num_dependents', 'own_telephone',
            'foreign_worker'
        ]

        st.sidebar.success("✅ Modelo cargado correctamente")
        return model, explainer, label_encoders, feature_names

    except FileNotFoundError as e:
        st.error(f"""
        ❌ **Error: No se encontraron los archivos del modelo**
        
        Archivos necesarios:
        - `models/xgboost_model.pkl`
        - `models/label_encoders.pkl`
        
        **Solución:**
        1. Asegúrate de estar en la carpeta `interpretability_I/`
        2. Ejecuta desde terminal:
           ```
           cd interpretability_I
           streamlit run app.py
           ```
        
        **Ruta actual de trabajo:** `{os.getcwd()}`
        
        **Detalle del error:** {e}
        """)
        st.stop()

    except Exception as e:
        st.error(f"""
        ❌ **Error inesperado cargando modelo**
        
        {type(e).__name__}: {e}
        
        Verifica que los archivos .pkl sean compatibles con las versiones actuales de:
        - xgboost
        - scikit-learn
        - joblib
        """)
        st.stop()


@st.cache_data
def load_test_data():
    """
    Carga datos de test para análisis exploratorio
    Si no existen archivos guardados, genera un split desde el dataset original
    """
    try:
        # Intentar cargar desde archivos guardados
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv')
        return X_test, y_test
    except FileNotFoundError:
        try:
            # Si no hay archivos, cargar dataset original y hacer split
            from sklearn.model_selection import train_test_split

            # Nombres de las columnas
            column_names = [
                'checking_status', 'duration', 'credit_history', 'purpose',
                'credit_amount', 'savings_status', 'employment', 'installment_rate',
                'personal_status', 'other_parties', 'residence_since',
                'property_magnitude', 'age', 'other_payment_plans', 'housing',
                'existing_credits', 'job', 'num_dependents', 'own_telephone',
                'foreign_worker', 'risk'
            ]

            # Cargar dataset
            data = pd.read_csv('../german_credit_data/german.data',
                               sep=' ',
                               header=None,
                               names=column_names)

            X = data.drop('risk', axis=1)
            y = data['risk']

            # Aplicar label encoding a categóricas
            categorical_cols = X.select_dtypes(include=['object']).columns

            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

            # Split estratificado (mismo que en entrenamiento)
            _, X_test, _, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            return X_test, y_test

        except Exception as e:
            st.warning(f"No se pudieron cargar datos de test: {e}")
            return None, None
# ==============================================================================
# DICCIONARIO DE FEATURES (para traducir a lenguaje natural)
# ==============================================================================


FEATURE_LABELS = {
    'checking_status': 'Estado de cuenta corriente',
    'duration': 'Duración del crédito (meses)',
    'credit_history': 'Historial crediticio',
    'purpose': 'Propósito del crédito',
    'credit_amount': 'Monto del crédito',
    'savings_status': 'Estado de ahorros',
    'employment': 'Tiempo de empleo',
    'installment_rate': 'Tasa de cuota (% ingreso)',
    'personal_status': 'Estado civil y sexo',
    'other_parties': 'Otros deudores/garantes',
    'residence_since': 'Años en residencia actual',
    'property_magnitude': 'Propiedad',
    'age': 'Edad (años)',
    'other_payment_plans': 'Otros planes de pago',
    'housing': 'Vivienda',
    'existing_credits': 'Créditos existentes',
    'job': 'Tipo de trabajo',
    'num_dependents': 'Personas a cargo',
    'own_telephone': 'Teléfono propio',
    'foreign_worker': 'Trabajador extranjero'
}

# Opciones categóricas (mapeo simplificado para el formulario)
CHECKING_STATUS_OPTIONS = {
    '< 0 DM': 0,
    '0-200 DM': 1,
    '>= 200 DM': 2,
    'Sin cuenta': 3
}

SAVINGS_STATUS_OPTIONS = {
    '< 100 DM': 0,
    '100-500 DM': 1,
    '500-1000 DM': 2,
    '>= 1000 DM': 3,
    'Desconocido/Sin ahorros': 4
}

CREDIT_HISTORY_OPTIONS = {
    'Sin créditos/Pagos al día': 0,
    'Todos pagados en este banco': 1,
    'Créditos pagados al día': 2,
    'Retrasos en el pasado': 3,
    'Cuenta crítica': 4
}

PURPOSE_OPTIONS = {
    'Auto nuevo': 0, 'Auto usado': 1, 'Muebles': 2, 'Radio/TV': 3,
    'Electrodomésticos': 4, 'Reparaciones': 5, 'Educación': 6,
    'Vacaciones': 7, 'Reentrenamiento': 8, 'Negocio': 9, 'Otros': 10
}

# Features modificables para counterfactuals (según LEARNINGS.md Fase 4)
MODIFIABLE_FEATURES = [
    'savings_status', 'employment', 'duration', 'credit_amount',
    'property_magnitude', 'housing', 'job', 'own_telephone',
    'other_payment_plans'
]

# ==============================================================================
# FUNCIONES DE PREDICCIÓN Y EXPLICACIÓN
# ==============================================================================


def predict_single(data, model):
    """
    Realiza predicción para un cliente individual

    Args:
        data: DataFrame con features del cliente
        model: Modelo XGBoost

    Returns:
        tuple: (predicción, probabilidad_bad_risk)
    """
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0]

    # proba[1] = probabilidad de Bad Risk (clase 1)
    return prediction, proba[1]


def explain_prediction_shap(data, explainer, feature_names):
    """
    Genera explicación SHAP waterfall para una predicción individual

    Args:
        data: DataFrame con features del cliente
        explainer: SHAP TreeExplainer
        feature_names: Lista de nombres de features

    Returns:
        shap_values: Valores SHAP para visualización
    """
    shap_values = explainer.shap_values(data)

    # Para XGBoost binario, shap_values tiene forma (n_samples, n_features)
    return shap_values[0]  # Primera (y única) muestra


def generate_counterfactual(data, model, feature_names):
    """
    Genera recomendaciones counterfactuales para cliente rechazado

    Algoritmo greedy (según LEARNINGS.md):
    1. Ordenar features modificables por importancia SHAP
    2. Probar cambios incrementales que mejoren probabilidad
    3. Limitar a máximo 3 cambios

    Args:
        data: DataFrame con features del cliente
        model: Modelo XGBoost
        feature_names: Lista de nombres de features

    Returns:
        dict: {'cambios': [...], 'nueva_proba': float, 'mejora': float}
    """
    current_proba = model.predict_proba(data)[0][1]

    # Si ya está aprobado, no necesita cambios
    if current_proba < 0.5:
        return None

    cambios = []
    data_modified = data.copy()

    # Intentar modificar features modificables
    for feature in MODIFIABLE_FEATURES[:3]:  # Limitar a 3 intentos
        if feature not in feature_names:
            continue

        feature_idx = feature_names.index(feature)
        original_value = data_modified.iloc[0, feature_idx]

        # Estrategia simple: aumentar/disminuir valor según feature
        if feature == 'duration':
            # Reducir duración mejora score
            new_value = max(6, original_value - 6)
        elif feature == 'credit_amount':
            # Reducir monto mejora score
            new_value = original_value * 0.8
        elif feature == 'savings_status':
            # Aumentar ahorros mejora score
            new_value = min(4, original_value + 1)
        else:
            continue

        # Probar cambio
        data_modified.iloc[0, feature_idx] = new_value
        new_proba = model.predict_proba(data_modified)[0][1]

        if new_proba < current_proba:  # Mejoró
            cambios.append({
                'feature': FEATURE_LABELS[feature],
                'original': original_value,
                'nuevo': new_value,
                'mejora': current_proba - new_proba
            })
            current_proba = new_proba

            # Si logró aprobación, detener
            if current_proba < 0.5:
                break

    if cambios:
        return {
            'cambios': cambios,
            'nueva_proba': current_proba,
            'aprobado': current_proba < 0.5
        }
    else:
        return None

# ==============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ==============================================================================


def plot_shap_waterfall(shap_values, base_value, feature_names, feature_values):
    """
    Crea gráfico waterfall SHAP explicando predicción individual

    Args:
        shap_values: Array de valores SHAP
        base_value: Valor base del modelo
        feature_names: Lista de nombres
        feature_values: Valores de features del cliente
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Ordenar por magnitud de impacto
    sorted_idx = np.argsort(np.abs(shap_values))[::-1][:10]  # Top 10

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=feature_values,
            feature_names=feature_names
        ),
        max_display=10,
        show=False
    )

    return fig


def plot_risk_gauge(proba):
    """
    Gauge chart mostrando nivel de riesgo

    Args:
        proba: Probabilidad de Bad Risk (0-1)
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Riesgo de Incumplimiento (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if proba > 0.5 else "darkgreen"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 50], 'color': "yellow"},
                {'range': [50, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def plot_shap_summary(shap_values, X_data, feature_names):
    """
    SHAP summary plot mostrando importancia global
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_data,
                      feature_names=feature_names, show=False)
    return fig

# ==============================================================================
# INTERFAZ PRINCIPAL
# ==============================================================================


def main():
    # Cargar modelo
    model, explainer, label_encoders, feature_names = load_model()
    X_test, y_test = load_test_data()

    # Título
    st.title("🏦 Credit Risk Analyzer")
    st.markdown(
        "**Dashboard de Interpretabilidad para Análisis de Riesgo Crediticio**")
    st.markdown("---")

    # Sidebar con métricas del modelo
    with st.sidebar:
        st.header("📊 Métricas del Modelo")
        st.metric("Accuracy (Test)", "78.0%")
        st.metric("Recall", "71.7%")
        st.metric("ROC-AUC", "0.809")
        st.markdown("---")
        st.caption("Modelo: XGBoost")
        st.caption("Dataset: German Credit (1000 registros)")

    # Tabs principales
    tab1, tab2, tab3 = st.tabs(
        ["📝 Evaluar Cliente", "🔍 Explorar Modelo", "📄 Análisis Batch"])

    # ==============================================================================
    # TAB 1: EVALUAR CLIENTE INDIVIDUAL
    # ==============================================================================
    with tab1:
        st.header("Evaluación Individual de Cliente")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Información del Cliente")

            # Formulario simplificado (features críticas)
            checking = st.selectbox(
                "Estado de cuenta corriente",
                options=list(CHECKING_STATUS_OPTIONS.keys())
            )

            duration = st.slider(
                "Duración del crédito (meses)",
                min_value=6, max_value=72, value=24
            )

            credit_amount = st.number_input(
                "Monto del crédito (DM)",
                min_value=250, max_value=20000, value=3000, step=100
            )

            savings = st.selectbox(
                "Estado de ahorros",
                options=list(SAVINGS_STATUS_OPTIONS.keys())
            )

            age = st.number_input(
                "Edad", min_value=18, max_value=80, value=35
            )

            # Botón de predicción
            if st.button("🔮 Evaluar Cliente", type="primary"):
                # Crear DataFrame con valores (simplificado para demo)
                # En producción, requerirías todas las 20 features
                client_data = pd.DataFrame({
                    'checking_status': [CHECKING_STATUS_OPTIONS[checking]],
                    'duration': [duration],
                    'credit_history': [2],  # Valor por defecto
                    'purpose': [2],
                    'credit_amount': [credit_amount],
                    'savings_status': [SAVINGS_STATUS_OPTIONS[savings]],
                    'employment': [3],
                    'installment_rate': [3],
                    'personal_status': [2],
                    'other_parties': [0],  # CORREGIDO: era other_debtors
                    'residence_since': [3],
                    'property_magnitude': [1],
                    'age': [age],
                    # CORREGIDO: era other_installment_plans
                    'other_payment_plans': [2],
                    'housing': [1],
                    # CORREGIDO: era num_existing_credits
                    'existing_credits': [1],
                    'job': [2],
                    'num_dependents': [1],  # CORREGIDO: era num_people_liable
                    'own_telephone': [1],
                    'foreign_worker': [0]
                })

                # Predicción
                prediction, proba = predict_single(client_data, model)

                with col2:
                    st.subheader("Resultado de la Evaluación")

                    # Gauge de riesgo
                    st.plotly_chart(plot_risk_gauge(proba),
                                    use_container_width=True)

                    # Decisión
                    if prediction == 0:
                        st.success(
                            f"✅ **APROBADO** - Riesgo: {proba*100:.1f}%")
                    else:
                        st.error(f"❌ **RECHAZADO** - Riesgo: {proba*100:.1f}%")

                # Explicación SHAP
                st.subheader("🔍 ¿Por qué esta decisión?")

                shap_vals = explain_prediction_shap(
                    client_data, explainer, feature_names)

                summary = generate_decision_summary(
                    shap_vals,
                    feature_names,
                    client_data.values[0],
                    prediction,
                    proba
                )
                st.markdown(summary)
                st.markdown("---")

                # Waterfall plot
                fig = plot_shap_waterfall(
                    shap_vals,
                    explainer.expected_value,
                    feature_names,
                    client_data.values[0]
                )
                st.pyplot(fig)

                st.caption(
                    "**Interpretación:** Las barras rojas aumentan el riesgo, las azules lo reducen.")

                # Counterfactual si rechazado
                if prediction == 1:
                    st.subheader("💡 ¿Qué cambiar para aprobar?")

                    counterfactual = generate_counterfactual(
                        client_data, model, feature_names
                    )

                    if counterfactual and counterfactual['cambios']:
                        st.info("**Recomendaciones para lograr aprobación:**")

                        for i, cambio in enumerate(counterfactual['cambios'], 1):
                            st.markdown(f"""
                            {i}. **{cambio['feature']}**: 
                               {cambio['original']} → {cambio['nuevo']} 
                               (Mejora: {cambio['mejora']*100:.1f}%)
                            """)

                        st.metric(
                            "Nuevo riesgo estimado",
                            f"{counterfactual['nueva_proba']*100:.1f}%",
                            delta=f"{(proba - counterfactual['nueva_proba'])*100:.1f}%"
                        )

                        if counterfactual['aprobado']:
                            st.success(
                                "✅ Con estos cambios, el cliente sería aprobado")
                    else:
                        st.warning(
                            "No se encontraron cambios simples que logren aprobación")

    # ==============================================================================
    # TAB 2: EXPLORAR MODELO
    # ==============================================================================
    with tab2:
        st.header("Exploración del Comportamiento del Modelo")

        if X_test is not None:
            st.subheader("📊 Importancia Global de Features (SHAP)")

            # Calcular SHAP values para dataset de test (muestra)
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(sample_size, random_state=42)

            with st.spinner("Calculando valores SHAP..."):
                shap_values = explainer.shap_values(X_sample)

            # Summary plot
            fig = plot_shap_summary(shap_values, X_sample, feature_names)
            st.pyplot(fig)

            st.caption("""
            **Interpretación:** 
            - Cada punto es un cliente
            - Color rojo = valor alto de la feature, azul = valor bajo
            - Posición en eje X = impacto en la predicción (derecha aumenta riesgo)
            """)

            # Top features
            st.subheader("🏆 Top 5 Features Más Importantes")

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[::-1][:5]

            top_features_df = pd.DataFrame({
                'Feature': [FEATURE_LABELS[feature_names[i]] for i in top_indices],
                'Impacto Promedio (SHAP)': mean_abs_shap[top_indices]
            })

            st.dataframe(top_features_df, hide_index=True)

        else:
            st.warning("No se encontraron datos de test para exploración")

    # ==============================================================================
    # TAB 3: ANÁLISIS BATCH
    # ==============================================================================
    with tab3:
        st.header("Análisis de Múltiples Clientes")

        st.markdown("""
        Analiza múltiples solicitudes de crédito simultáneamente.
        
        **Opciones:**
        1. Usa el dataset de ejemplo (German Credit Data completo)
        2. Sube tu propio archivo CSV con las 20 features requeridas
        """)

        # Opciones de carga
        col1, col2 = st.columns(2)

        with col1:
            use_sample = st.checkbox("✅ Usar dataset de ejemplo", value=True)

        with col2:
            uploaded_file = st.file_uploader(
                "O sube tu archivo CSV",
                type=['csv', 'txt'],
                disabled=use_sample
            )

        df = None

        # Cargar datos según opción seleccionada
        if use_sample:
            try:
                # Cargar dataset original de ejemplo
                column_names = [
                    'checking_status', 'duration', 'credit_history', 'purpose',
                    'credit_amount', 'savings_status', 'employment', 'installment_rate',
                    'personal_status', 'other_parties', 'residence_since',
                    'property_magnitude', 'age', 'other_payment_plans', 'housing',
                    'existing_credits', 'job', 'num_dependents', 'own_telephone',
                    'foreign_worker', 'risk'
                ]

                # Cargar dataset
                data = pd.read_csv(
                    '../german_credit_data/german.data',
                    sep=' ',
                    header=None,
                    names=column_names
                )

                # Aplicar label encoding a categóricas (igual que en entrenamiento)
                categorical_cols = data.select_dtypes(
                    include=['object']).columns

                from sklearn.preprocessing import LabelEncoder
                for col in categorical_cols:
                    if col != 'risk':  # No encodear el target
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col])

                df = data.drop('risk', axis=1)  # Solo features, sin target

                st.success(f"✅ Dataset de ejemplo cargado: {len(df)} clientes")

                # Mostrar muestra
                with st.expander("👀 Ver muestra de datos (primeras 5 filas)"):
                    st.dataframe(df.head())

            except Exception as e:
                st.error(f"❌ Error cargando dataset de ejemplo: {e}")
                st.stop()

        elif uploaded_file is not None:
            try:
                # Intentar leer como CSV normal
                df = pd.read_csv(uploaded_file)

                # Si no tiene las columnas esperadas, intentar con espacio como separador
                if len(df.columns) < 20:
                    uploaded_file.seek(0)  # Reiniciar puntero del archivo

                    column_names = [
                        'checking_status', 'duration', 'credit_history', 'purpose',
                        'credit_amount', 'savings_status', 'employment', 'installment_rate',
                        'personal_status', 'other_parties', 'residence_since',
                        'property_magnitude', 'age', 'other_payment_plans', 'housing',
                        'existing_credits', 'job', 'num_dependents', 'own_telephone',
                        'foreign_worker', 'risk'
                    ]

                    df = pd.read_csv(
                        uploaded_file,
                        sep=' ',
                        header=None,
                        names=column_names
                    )

                    # Si tiene columna 'risk', removerla
                    if 'risk' in df.columns:
                        df = df.drop('risk', axis=1)

                # Aplicar label encoding si hay categóricas
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    from sklearn.preprocessing import LabelEncoder
                    for col in categorical_cols:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col])

                st.success(f"✅ Archivo cargado: {len(df)} registros")

                # Mostrar muestra
                with st.expander("👀 Ver muestra de datos"):
                    st.dataframe(df.head())

            except Exception as e:
                st.error(f"❌ Error procesando archivo: {e}")
                st.info("""
                **Formato esperado:**
                - CSV con 20 columnas (las 20 features del modelo)
                - O archivo formato German Credit (.data con espacios como separador)
                """)
                df = None

        # Procesar batch si hay datos
        if df is not None and st.button("🚀 Procesar Batch", type="primary"):

            with st.spinner("Procesando predicciones..."):
                try:
                    # Verificar que tenga las 20 features requeridas
                    if len(df.columns) != 20:
                        st.error(f"""
                        ❌ **Error:** Se esperaban 20 features, pero el archivo tiene {len(df.columns)}
                        
                        **Features requeridas:**
                        {', '.join(feature_names)}
                        """)
                        st.stop()

                    # Asegurar que las columnas estén en el orden correcto
                    df = df[feature_names]

                    # Predicciones
                    predictions = model.predict(df)
                    probas = model.predict_proba(df)[:, 1]

                    # Crear DataFrame de resultados
                    results = df.copy()
                    results.insert(0, 'ID Cliente', range(1, len(results) + 1))
                    results['Decisión'] = ['✅ Aprobado' if p ==
                                           0 else '❌ Rechazado' for p in predictions]
                    results['Riesgo (%)'] = (probas * 100).round(1)
                    results['Clasificación'] = pd.cut(
                        probas,
                        bins=[0, 0.3, 0.5, 0.7, 1.0],
                        labels=['Riesgo Bajo', 'Riesgo Medio',
                                'Riesgo Alto', 'Riesgo Muy Alto']
                    )

                    # Mostrar resultados resumidos
                    st.subheader("📊 Resultados del Análisis")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        aprobados = (predictions == 0).sum()
                        st.metric("✅ Aprobados", aprobados,
                                  delta=f"{aprobados/len(predictions)*100:.1f}%")

                    with col2:
                        rechazados = (predictions == 1).sum()
                        st.metric("❌ Rechazados", rechazados,
                                  delta=f"{rechazados/len(predictions)*100:.1f}%")

                    with col3:
                        riesgo_promedio = probas.mean() * 100
                        st.metric("📈 Riesgo Promedio",
                                  f"{riesgo_promedio:.1f}%")

                    with col4:
                        riesgo_alto = (probas > 0.7).sum()
                        st.metric("⚠️ Riesgo Alto", riesgo_alto)

                    # Distribución de riesgo
                    st.subheader("📉 Distribución de Riesgo")

                    import plotly.express as px

                    fig = px.histogram(
                        x=probas * 100,
                        nbins=20,
                        labels={
                            'x': 'Probabilidad de Riesgo (%)', 'y': 'Cantidad de Clientes'},
                        title='Distribución de Niveles de Riesgo'
                    )
                    fig.add_vline(x=50, line_dash="dash", line_color="red",
                                  annotation_text="Threshold de Aprobación")
                    st.plotly_chart(fig, use_container_width=True)

                    # Tabla de resultados
                    st.subheader("📋 Detalle de Clientes")

                    # Mostrar solo columnas relevantes para no saturar
                    display_cols = ['ID Cliente', 'Decisión',
                                    'Riesgo (%)', 'Clasificación']
                    st.dataframe(
                        results[display_cols].style.apply(
                            lambda x: ['background-color: #d4edda' if v == '✅ Aprobado'
                                       else 'background-color: #f8d7da'
                                       for v in x],
                            subset=['Decisión']
                        ),
                        use_container_width=True,
                        height=400
                    )

                    # Descargar resultados
                    st.subheader("💾 Exportar Resultados")

                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Descargar CSV Completo",
                        data=csv,
                        file_name="resultados_credito_batch.csv",
                        mime="text/csv"
                    )

                    # Insights adicionales
                    with st.expander("🔍 Ver Insights del Batch"):
                        st.markdown(f"""
                        **Análisis del lote de {len(df)} solicitudes:**
                        
                        - **Tasa de aprobación:** {aprobados/len(predictions)*100:.1f}%
                        - **Tasa de rechazo:** {rechazados/len(predictions)*100:.1f}%
                        - **Riesgo promedio:** {riesgo_promedio:.1f}%
                        - **Clientes de riesgo alto (>70%):** {riesgo_alto} ({riesgo_alto/len(predictions)*100:.1f}%)
                        - **Clientes de riesgo bajo (<30%):** {(probas < 0.3).sum()} ({(probas < 0.3).sum()/len(predictions)*100:.1f}%)
                        
                        **Recomendaciones:**
                        - Revisar manualmente los {riesgo_alto} casos de riesgo alto
                        - Considerar ajustar el threshold si la tasa de rechazo es muy alta
                        """)

                except Exception as e:
                    st.error(f"❌ Error durante el procesamiento: {e}")
                    st.exception(e)


def generate_decision_summary(shap_values, feature_names, feature_values, prediction, proba):
    """
    Genera un resumen textual explicando la decisión del modelo

    Args:
        shap_values: Array de valores SHAP
        feature_names: Lista de nombres de features
        feature_values: Valores de las features del cliente
        prediction: 0 (aprobado) o 1 (rechazado)
        proba: Probabilidad de Bad Risk

    Returns:
        str: Resumen en lenguaje natural
    """
    # Obtener las top 3 features que más impactaron
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[::-1][:3]

    # Separar factores positivos (aumentan riesgo) y negativos (reducen riesgo)
    positive_factors = []  # Aumentan riesgo (SHAP > 0)
    negative_factors = []  # Reducen riesgo (SHAP < 0)

    for idx in top_indices:
        feature = feature_names[idx]
        shap_val = shap_values[idx]
        label = FEATURE_LABELS.get(feature, feature)

        if shap_val > 0:  # Aumenta riesgo
            positive_factors.append(label)
        else:  # Reduce riesgo
            negative_factors.append(label)

    # Construir el resumen según la decisión
    if prediction == 0:  # APROBADO
        summary = f"**Cliente Aprobado** (Riesgo estimado: {proba*100:.1f}%)\n\n"

        if negative_factors:
            summary += f"✅ **Factores favorables:** {', '.join(negative_factors)}"
            if positive_factors:
                summary += f"\n\n⚠️ **Puntos de atención:** {', '.join(positive_factors)}"

    else:  # RECHAZADO
        summary = f"**Cliente Rechazado** (Riesgo estimado: {proba*100:.1f}%)\n\n"

        if positive_factors:
            summary += f"❌ **Factores de riesgo principales:** {', '.join(positive_factors)}"
            summary += f"\n\n💡 **Para mejorar su solicitud, el cliente debería enfocarse en mejorar estas variables.**"

        if negative_factors:
            summary += f"\n\n✅ **Factores positivos identificados:** {', '.join(negative_factors)}"

    return summary


if __name__ == "__main__":
    main()
