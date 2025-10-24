# ============================================================
# FUNCIÓN: VISUALIZAR SHAP
# ============================================================
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import shap


def get_model_info(model_number, model):
    """
    Obtiene información del modelo: número de clases y nombres de las clases

    Args:
        model_number: ID del modelo (1-4)
        model: modelo cargado

    Returns:
        tuple: (num_classes, class_names)
    """
    # Detectar número de clases desde la configuración del modelo
    num_classes = model.config.num_labels

    # Mapeo de nombres de clases según el modelo
    class_names_map = {
        1: ['NEGATIVE', 'POSITIVE'],  # distilbert SST-2
        2: ['NEGATIVE', 'NEUTRAL', 'POSITIVE'],  # twitter-roberta
        # emotion-english
        3: ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'],
        4: ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']  # bert-emotion
    }

    # Obtener nombres de clases del mapeo
    class_names = class_names_map.get(
        model_number, [f'Class_{i}' for i in range(num_classes)])

    return num_classes, class_names

# ============================================================
# FUNCIÓN: VISUALIZAR SHAP
# ============================================================


def visualizar_shap(shap_values, input_text, model_choice, model_number, method, num_features=10):
    """
    Genera visualización completa de SHAP según el tipo de modelo.

    Args:
        shap_values: Valores SHAP calculados
        input_text: Texto de entrada analizado
        model_choice: Nombre del modelo seleccionado
        model_number: ID del modelo (1-4)
        method: Método de explicación seleccionado ("Solo SHAP" o "Ambos (SHAP + LIME)")
        num_features: Número de características a mostrar (default: 10)

    Returns:
        None (muestra directamente en Streamlit)
    """
    st.markdown("### Análisis detallado con SHAP")
    st.markdown("#### Modelo utilizado: " + model_choice)
    # st.markdown(f"#### ID del modelo: {model_number}")

    # ============================================================
    # WATERFALL / BAR PLOT CON MANEJO ROBUSTO SEGÚN MODELO
    # ============================================================
    st.markdown("#### Contribución Acumulativa")
    try:
        # Determinar número de clases del modelo
        num_classes = shap_values[0].values.shape[1] if len(
            shap_values[0].values.shape) > 1 else 1

        # Extraer tokens (común para todos los casos)
        tokens = shap_values[0].data

        # ============================================================
        # CASO: MODELO BINARIO (2 clases)
        # ============================================================
        if num_classes == 2:
            # Detectar clase predominante
            values_for_positive = shap_values[0].values[:, 1]
            sum_positive = np.sum(values_for_positive)

            # Determinar qué clase usar
            if sum_positive > 0:
                class_idx = 1
                class_label = "POSITIVE"
            else:
                class_idx = 0
                class_label = "NEGATIVE"

            st.info(f"Clase predicha: **{class_label}**")

            # Usar waterfall plot nativo de SHAP
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(
                shap_values[0, :, class_idx],
                max_display=num_features,
                show=False
            )
            st.pyplot(fig)
            plt.close()

        # ============================================================
        # CASO: MODELO MULTICLASE (3+ clases)
        # ============================================================
        else:
            # Obtener la predicción del modelo
            prediction = st.session_state.classifier(input_text)[0]
            predicted_label = prediction['label']
            predicted_score = prediction['score']

            # Obtener nombres de clases
            if hasattr(st.session_state, 'class_names') and st.session_state.class_names:
                class_names_shap = st.session_state.class_names
            else:
                class_names_shap = [f'Clase_{i}' for i in range(num_classes)]

            # CAMBIO: Crear dos columnas para mostrar lado a lado
            col1, col2 = st.columns(2)

            # ============================================================
            # COLUMNA 1: CLASE PREDICHA ESPECÍFICA
            # ============================================================
            with col1:
                st.markdown("##### 🎯 Clase Predicha")

                # Mapear label a índice
                try:
                    predicted_idx = class_names_shap.index(predicted_label)
                except ValueError:
                    predicted_idx = 0
                    st.warning(
                        f"No se pudo mapear '{predicted_label}', usando índice 0")

                st.info(
                    f"**{predicted_label}**\n\n(confianza: {predicted_score:.2%})")

                # Obtener valores para la clase predicha
                values = shap_values[0].values[:, predicted_idx]

                # Top tokens por valor absoluto
                top_indices = np.argsort(np.abs(values))[-num_features:][::-1]

                # Crear DataFrame
                top_tokens = [tokens[i] for i in top_indices]
                top_values = [values[i] for i in top_indices]

                # Crear figura con colores verde/rojo (tamaño ajustado para columna)
                fig, ax = plt.subplots(figsize=(6, 6))

                # Colores según signo
                colors = ['#2ecc71' if v >
                          0 else '#e74c3c' for v in top_values]

                ax.barh(range(len(top_tokens)), top_values,
                        color=colors, alpha=0.7, edgecolor='black', linewidth=1)
                ax.set_yticks(range(len(top_tokens)))
                ax.set_yticklabels(top_tokens, fontsize=9)
                ax.set_xlabel(
                    f'Importancia para "{predicted_label}"', fontsize=10)
                ax.set_title(f'Clase: {predicted_label}',
                             fontsize=11, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
                ax.grid(True, alpha=0.3, axis='x')

                # Agregar valores (opcional, puede hacer el gráfico muy denso)
                for i, (token, val) in enumerate(zip(top_tokens, top_values)):
                    x_pos = val + (0.002 if val > 0 else -0.002)
                    ha = 'left' if val > 0 else 'right'
                    ax.text(x_pos, i, f'{val:.3f}',
                            va='center', ha=ha, fontsize=8,
                            color='black', fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # ============================================================
            # COLUMNA 2: DISTRIBUCIÓN DE PROBABILIDADES POR CLASE
            # ============================================================
            with col2:
                st.markdown("##### 🌍 Distribución de Clases")

                st.info(f"Predicción entre\n\n{num_classes} clases posibles")

                # Obtener las probabilidades de todas las clases manualmente
                import torch

                try:
                    with torch.no_grad():
                        # Tokenizar el texto
                        inputs = st.session_state.tokenizer(
                            input_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512
                        )

                        # Obtener logits del modelo
                        outputs = st.session_state.model(**inputs)

                        # Calcular probabilidades con softmax
                        probabilities = torch.nn.functional.softmax(
                            outputs.logits, dim=-1)[0]

                        # Convertir a lista de Python
                        probs_list = probabilities.cpu().numpy().tolist()

                    # Crear lista de clases con sus probabilidades
                    all_predictions = []
                    for i in range(num_classes):
                        all_predictions.append({
                            'label': class_names_shap[i],
                            'score': probs_list[i]
                        })

                    # Ordenar de mayor a menor probabilidad
                    all_predictions_sorted = sorted(
                        all_predictions, key=lambda x: x['score'], reverse=True)

                    # Extraer labels y scores ordenados
                    class_labels_sorted = [pred['label']
                                           for pred in all_predictions_sorted]
                    class_scores_sorted = [pred['score']
                                           for pred in all_predictions_sorted]

                    # Crear figura
                    fig, ax = plt.subplots(figsize=(6, 6))

                    # Colores: verde para la clase predicha, azul para las demás
                    colors = []
                    for label in class_labels_sorted:
                        if label == predicted_label:
                            colors.append('#6cdb9b')  # Verde para la predicha
                        else:
                            colors.append("#6c9cdb")  # Azul para las demás

                    # Gráfico de barras horizontales
                    bars = ax.barh(range(len(class_labels_sorted)), class_scores_sorted,
                                   color=colors,
                                   alpha=0.9,
                                   edgecolor='black',
                                   linewidth=0.6)

                    ax.set_yticks(range(len(class_labels_sorted)))
                    ax.set_yticklabels(class_labels_sorted,
                                       fontsize=10, fontweight='normal')
                    ax.set_xlabel('Probabilidad', fontsize=11,
                                  fontweight='normal')
                    ax.set_title('Distribución de Clases\n Predicción',
                                 fontsize=12, fontweight='bold')
                    ax.set_xlim(0, 1)  # Probabilidades van de 0 a 1

                    # Grid
                    ax.grid(True, alpha=0.4, axis='x',
                            linestyle='--', linewidth=0.8)

                    # Agregar porcentajes al final de cada barra
                    for i, (label, score) in enumerate(zip(class_labels_sorted, class_scores_sorted)):
                        ax.text(score, i, f'  {score:.1%}',
                                va='center', ha='left', fontsize=9,
                                color='black', fontweight='bold')

                    # Agregar líneas horizontales sutiles
                    for i in range(len(class_labels_sorted) - 1):
                        ax.axhline(y=i + 0.5, color='gray',
                                   linestyle='-', linewidth=0.3, alpha=0.3)

                    # Bordes del gráfico
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_linewidth(1.5)
                    ax.spines['bottom'].set_linewidth(1.5)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                except Exception as e:
                    st.error(
                        f"Error generando distribución de clases: {str(e)[:200]}")
                    import traceback
                    st.code(traceback.format_exc())
            # ============================================================
            # EXPLICACIÓN COMPARATIVA (debajo de ambas columnas)
            # ============================================================
            st.markdown("---")
            with st.expander("ℹ️ ¿Qué significan estos dos gráficos?", expanded=False):
                st.markdown("""
                **🎯 Gráfico Izquierdo: Clase Predicha**
                - 🟢 Verde: aumenta la probabilidad de esta clase
                - 🔴 Rojo: disminuye la probabilidad de esta clase
                
                **🌍 Gráfico Derecho: Distribución de Clases**
                - Muestra la confianza del modelo en cada clase posible
                - 🟢 Verde: clase con mayor probabilidad (predicción final)
                - 🔵 Azul: clases alternativas consideradas por el modelo
                - Los porcentajes suman 100%
                """.replace('{num_classes}', str(num_classes)))

    except Exception as e:
        st.error(f"Error generando gráfico: {str(e)[:200]}")
        # Mostrar más información del error para debugging
        import traceback
        st.code(traceback.format_exc())
        st.write(
            "Valores SHAP calculados pero visualización no disponible para esta configuración")

    # ============================================================
    # INFORMACIÓN ADICIONAL
    # ============================================================
    st.markdown("#### ℹ️ Sobre SHAP")
    if num_classes == 2:
        st.info("""
        **SHAP (SHapley Additive exPlanations)** utiliza valores de Shapley de la teoría de juegos 
        para asignar importancia a cada palabra. 
        
        🔴 **Rojo**: palabras que favorecen la clase NEGATIVE  
        🟢 **Verde**: palabras que favorecen la clase POSITIVE
        
        ✅ Garantiza **consistencia** y **aditividad** en las explicaciones.
        """)
    else:
        st.info("""
        **SHAP (SHapley Additive exPlanations)** utiliza valores de Shapley de la teoría de juegos 
        para asignar importancia a cada palabra.
        
        **Dos modos disponibles:**
        - **Importancia Promedio**: Muestra qué palabras son más importantes globalmente (azul intenso = más importante)
        - **Clase Predicha**: Muestra contribución específica para la clase predicha (🟢 a favor | 🔴 en contra)
        
        ✅ Garantiza **consistencia** y **aditividad** en las explicaciones.
        """)


# ============================================================
# FUNCIÓN: VISUALIZAR LIME
# ============================================================


def visualizar_lime(lime_explanation, num_features_lime):
    """
    Genera visualización completa de LIME con tabla y gráficos.

    Args:
        lime_explanation: Explicación LIME calculada
        num_features_lime: Número de características a mostrar

    Returns:
        None (muestra directamente en Streamlit)
    """
    st.markdown("### Análisis detallado con LIME")

    # ============================================================
    # OBTENER INFORMACIÓN DEL MODELO
    # ============================================================
    # Obtener información del modelo desde session_state o calcularla
    if hasattr(st.session_state, 'num_classes') and st.session_state.num_classes is not None:
        num_classes_lime = st.session_state.num_classes
        class_names = st.session_state.class_names
    else:
        # Calcular dinámicamente si no existe
        num_classes_lime, class_names = get_model_info(
            st.session_state.current_model,
            st.session_state.model
        )
        # Guardar para uso futuro
        st.session_state.num_classes = num_classes_lime
        st.session_state.class_names = class_names

    # Determinar la clase que LIME está explicando
    predicted_class = lime_explanation.available_labels()[0]
    explained_class = class_names[predicted_class] if predicted_class < len(
        class_names) else f"Clase {predicted_class}"

    # ============================================================
    # TABLA DE IMPORTANCIA
    # ============================================================
    st.markdown("#### Tabla de Importancia")

    # Extraer valores de la explicación
    exp_list = lime_explanation.as_list()[:num_features_lime]

    # Crear DataFrame con información detallada
    exp_df = pd.DataFrame(
        exp_list,
        columns=['Palabra', 'Importancia']
    )
    exp_df['Impacto'] = exp_df['Importancia'].apply(
        lambda x: '🟢 Positivo' if x > 0 else '🔴 Negativo'
    )

    # Mostrar información de la clase explicada
    if num_classes_lime == 2:
        st.info(f"Explicación para clase: **{explained_class.upper()}**")
    else:
        st.info(
            f"Modelo con {num_classes_lime} clases - Explicando: **{explained_class}**")

    # Mostrar tabla
    st.dataframe(exp_df, use_container_width=True)

    # ============================================================
    # VISUALIZACIÓN CON GRÁFICO DE BARRAS
    # ============================================================
    st.markdown("#### Visualización Gráfica")

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colores según signo
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in exp_df['Importancia']]

    ax.barh(range(len(exp_df)), exp_df['Importancia'],
            color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(exp_df)))
    ax.set_yticklabels(exp_df['Palabra'], fontsize=10)
    ax.set_xlabel('Importancia LIME', fontsize=11)

    # Título según tipo de modelo
    if num_classes_lime == 2:
        ax.set_title(f'LIME - Clase: {explained_class.upper()}\n(Verde: a favor | Rojo: en contra)',
                     fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'LIME - Clase: {explained_class} ({num_classes_lime} clases)\n(Verde: a favor | Rojo: en contra)',
                     fontsize=12, fontweight='bold')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(True, alpha=0.3, axis='x')

    # Agregar valores al final de cada barra
    for i, (palabra, val) in enumerate(zip(exp_df['Palabra'], exp_df['Importancia'])):
        x_pos = val + (0.002 if val > 0 else -0.002)
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, i, f'{val:.3f}',
                va='center', ha=ha, fontsize=9,
                color='black', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ============================================================
    # INFORMACIÓN ADICIONAL
    # ============================================================
    st.markdown("#### ℹ️ Sobre LIME")

    if num_classes_lime == 2:
        st.info("""
        **LIME (Local Interpretable Model-agnostic Explanations)** aproxima el modelo complejo 
        con un modelo lineal local, perturbando el texto de entrada y observando cómo cambian 
        las predicciones.
        
        🔴 **Rojo**: palabras que reducen la probabilidad de la clase explicada  
        🟢 **Verde**: palabras que aumentan la probabilidad de la clase explicada
        
        ✅ Es **agnóstico al modelo** y trabaja con cualquier clasificador.
        """)
    else:
        st.info("""
        **LIME (Local Interpretable Model-agnostic Explanations)** aproxima el modelo complejo 
        con un modelo lineal local, perturbando el texto de entrada y observando cómo cambian 
        las predicciones.
        
        **Para modelos multiclase:**
        - LIME explica **una clase específica** (la predicha por el modelo)
        - 🟢 **Verde**: palabras que aumentan la probabilidad de esa clase
        - 🔴 **Rojo**: palabras que reducen la probabilidad de esa clase
        
        ✅ Es **agnóstico al modelo** y trabaja con cualquier clasificador.
        """)

# ============================================================
# FUNCIÓN: COMPARACIÓN SHAP VS LIME
# ============================================================


def comparar_shap_lime(input_text, predict_proba, num_features_lime, num_samples_lime):
    """
    Genera comparación lado a lado de SHAP y LIME con métricas de rendimiento.

    Args:
        input_text: Texto de entrada analizado
        predict_proba: Función de predicción para LIME
        num_features_lime: Número de características a mostrar
        num_samples_lime: Número de muestras para LIME

    Returns:
        tuple: (shap_values, lime_explanation, shap_time, lime_time)
    """
    col1, col2 = st.columns(2)

    # Variables para guardar tiempos
    shap_time = 0
    lime_time = 0
    shap_values = None
    lime_explanation = None

    # ============================================================
    # COLUMNA SHAP
    # ============================================================
    with col1:
        st.markdown("### 🔷 SHAP")
        with st.spinner("Calculando SHAP (puede tomar 10-30 segundos)..."):
            start_time = time.time()
            shap_values = st.session_state.shap_explainer([input_text])
            shap_time = time.time() - start_time

            try:
                # Determinar número de clases del modelo
                num_classes = shap_values[0].values.shape[1] if len(
                    shap_values[0].values.shape) > 1 else 1

                # Extraer tokens
                tokens = shap_values[0].data

                # ============================================================
                # CASO: MODELO BINARIO (2 clases)
                # ============================================================
                if num_classes == 2:
                    # Detectar clase predominante
                    values_for_positive = shap_values[0].values[:, 1]
                    sum_positive = np.sum(values_for_positive)

                    # Determinar clase y valores
                    if sum_positive > 0:
                        values = shap_values[0].values[:, 1]
                        class_label = "POSITIVE"
                    else:
                        values = shap_values[0].values[:, 0]
                        class_label = "NEGATIVE"

                    # Top palabras
                    top_indices = np.argsort(
                        np.abs(values))[-num_features_lime:][::-1]

                    # DataFrame
                    shap_df = pd.DataFrame({
                        'Palabra': [tokens[i] for i in top_indices],
                        'Importancia': [values[i] for i in top_indices]
                    })

                    # Visualización
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['#2ecc71' if v > 0 else '#e74c3c'
                              for v in shap_df['Importancia']]
                    ax.barh(range(len(shap_df)),
                            shap_df['Importancia'],
                            color=colors,
                            alpha=0.7,
                            edgecolor='black',
                            linewidth=1)
                    ax.set_yticks(range(len(shap_df)))
                    ax.set_yticklabels(shap_df['Palabra'], fontsize=10)
                    ax.set_xlabel('Importancia SHAP', fontsize=10)
                    ax.set_title(f'SHAP - {class_label}\n({shap_time:.1f}s)',
                                 fontsize=11, fontweight='bold')
                    ax.axvline(x=0, color='black',
                               linestyle='-', linewidth=1.5)
                    ax.grid(True, alpha=0.3, axis='x')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # ============================================================
                # CASO: MODELO MULTICLASE (3+ clases)
                # ============================================================
                else:
                    # Obtener la predicción del modelo
                    prediction = st.session_state.classifier(input_text)[0]
                    predicted_label = prediction['label']
                    predicted_score = prediction['score']

                    # Obtener nombres de clases
                    if hasattr(st.session_state, 'class_names') and st.session_state.class_names:
                        class_names_shap = st.session_state.class_names
                    else:
                        class_names_shap = [
                            f'Clase_{i}' for i in range(num_classes)]

                    # Mapear label a índice
                    try:
                        predicted_idx = class_names_shap.index(predicted_label)
                    except ValueError:
                        predicted_idx = 0
                        st.warning(
                            f"No se pudo mapear '{predicted_label}', usando índice 0")

                    st.info(
                        f"Modelo con {num_classes} clases - Clase: **{predicted_label}** (confianza: {predicted_score:.2%})")

                    # Obtener valores para la clase predicha
                    values = shap_values[0].values[:, predicted_idx]

                    # Top palabras
                    top_indices = np.argsort(
                        np.abs(values))[-num_features_lime:][::-1]

                    # DataFrame
                    shap_df = pd.DataFrame({
                        'Palabra': [tokens[i] for i in top_indices],
                        'Importancia': [values[i] for i in top_indices]
                    })

                    # Visualización con colores verde/rojo
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['#2ecc71' if v > 0 else '#e74c3c'
                              for v in shap_df['Importancia']]

                    ax.barh(range(len(shap_df)),
                            shap_df['Importancia'],
                            color=colors,
                            alpha=0.7,
                            edgecolor='black',
                            linewidth=1)
                    ax.set_yticks(range(len(shap_df)))
                    ax.set_yticklabels(shap_df['Palabra'], fontsize=10)
                    ax.set_xlabel(
                        f'Importancia para "{predicted_label}"', fontsize=10)
                    ax.set_title(f'SHAP - {predicted_label}\n({shap_time:.1f}s)',
                                 fontsize=11, fontweight='bold')
                    ax.axvline(x=0, color='black',
                               linestyle='-', linewidth=1.5)
                    ax.grid(True, alpha=0.3, axis='x')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # Tiempo
                st.info(f"⏱️ Tiempo: {shap_time:.1f}s")

            except Exception as e:
                st.error(f"Error SHAP: {str(e)[:150]}")
                st.info(f"⏱️ Tiempo: {shap_time:.1f}s")
    # ============================================================
    # COLUMNA LIME
    # ============================================================
    with col2:
        st.markdown("### 🔶 LIME")
        with st.spinner("Calculando LIME..."):
            start_time = time.time()

            try:
                # Obtener información del modelo
                if hasattr(st.session_state, 'num_classes') and st.session_state.num_classes:
                    num_classes_lime = st.session_state.num_classes
                    class_names = st.session_state.class_names
                else:
                    # Fallback
                    num_classes_lime, class_names = get_model_info(
                        st.session_state.current_model,
                        st.session_state.model
                    )
                    st.session_state.num_classes = num_classes_lime
                    st.session_state.class_names = class_names

                # Calcular LIME
                lime_explanation = st.session_state.lime_explainer.explain_instance(
                    input_text,
                    predict_proba,
                    num_features=num_features_lime,
                    num_samples=num_samples_lime
                )
                lime_time = time.time() - start_time

                # Clase explicada
                predicted_class = lime_explanation.available_labels()[0]
                explained_class = class_names[predicted_class] if predicted_class < len(
                    class_names) else f"Clase {predicted_class}"

                # ============================================================
                # CASO: MODELO BINARIO (2 clases)
                # ============================================================
                if num_classes_lime == 2:
                    # Extraer valores
                    exp_list = lime_explanation.as_list()[:num_features_lime]

                    # DataFrame
                    lime_df = pd.DataFrame({
                        'Palabra': [x[0] for x in exp_list],
                        'Importancia': [x[1] for x in exp_list]
                    })

                    # Visualización
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['#2ecc71' if v > 0 else '#e74c3c'
                              for v in lime_df['Importancia']]
                    ax.barh(range(len(lime_df)),
                            lime_df['Importancia'],
                            color=colors,
                            alpha=0.7,
                            edgecolor='black',
                            linewidth=1)
                    ax.set_yticks(range(len(lime_df)))
                    ax.set_yticklabels(lime_df['Palabra'], fontsize=10)
                    ax.set_xlabel('Importancia LIME', fontsize=10)
                    ax.set_title(f'LIME - {explained_class.upper()}\n({lime_time:.1f}s)',
                                 fontsize=11, fontweight='bold')
                    ax.axvline(x=0, color='black',
                               linestyle='-', linewidth=1.5)
                    ax.grid(True, alpha=0.3, axis='x')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # ============================================================
                # CASO: MODELO MULTICLASE (3+ clases)
                # ============================================================
                else:
                    st.info(
                        f"Modelo con {num_classes_lime} clases - Clase: **{explained_class}**")

                    # Extraer valores
                    exp_list = lime_explanation.as_list()[:num_features_lime]

                    # DataFrame
                    lime_df = pd.DataFrame({
                        'Palabra': [x[0] for x in exp_list],
                        'Importancia': [x[1] for x in exp_list]
                    })

                    # Visualización
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['#2ecc71' if v > 0 else '#e74c3c'
                              for v in lime_df['Importancia']]
                    ax.barh(range(len(lime_df)),
                            lime_df['Importancia'],
                            color=colors,
                            alpha=0.7,
                            edgecolor='black',
                            linewidth=1)
                    ax.set_yticks(range(len(lime_df)))
                    ax.set_yticklabels(lime_df['Palabra'], fontsize=10)
                    ax.set_xlabel('Importancia LIME', fontsize=10)
                    ax.set_title(f'LIME - {explained_class}\n({lime_time:.1f}s)',
                                 fontsize=11, fontweight='bold')
                    ax.axvline(x=0, color='black',
                               linestyle='-', linewidth=1.5)
                    ax.grid(True, alpha=0.3, axis='x')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # Tiempo
                st.info(f"⏱️ Tiempo: {lime_time:.1f}s")

            except Exception as e:
                lime_time = time.time() - start_time
                st.error(f"Error LIME: {str(e)[:150]}")
                st.info(f"⏱️ Tiempo: {lime_time:.1f}s")

    # ============================================================
    # RESUMEN COMPARATIVO
    # ============================================================
    st.markdown("---")
    st.markdown("### 📊 Resumen Comparativo")

    # Obtener número de clases para la tabla
    num_classes = shap_values[0].values.shape[1] if shap_values and len(
        shap_values[0].values.shape) > 1 else 2

    if hasattr(st.session_state, 'num_classes') and st.session_state.num_classes:
        num_classes_lime = st.session_state.num_classes
    else:
        num_classes_lime = num_classes

    # Tabla comparativa
    comparison_df = pd.DataFrame({
        'Métrica': [
            'Tiempo de cómputo',
            'Speedup',
            'Tipo de modelo',
            'Características mostradas'
        ],
        'SHAP': [
            f"{shap_time:.1f}s",
            "1x (base)",
            f"{num_classes} clase{'s' if num_classes > 1 else ''}",
            f"{num_features_lime} palabras"
        ],
        'LIME': [
            f"{lime_time:.1f}s",
            f"{shap_time/lime_time:.2f}x {'más rápido' if lime_time < shap_time else 'más lento'}",
            f"{num_classes_lime} clase{'s' if num_classes_lime > 1 else ''}",
            f"{num_features_lime} palabras"
        ]
    })

    st.table(comparison_df)

    # Notas explicativas según tipo de modelo
    if num_classes == 2 and num_classes_lime == 2:
        st.markdown("""
        **ℹ️ Interpretación (modelo binario):**
        - 🟢 **Verde**: palabras que favorecen la clase predicha
        - 🔴 **Rojo**: palabras que favorecen la clase contraria
        - Ambos métodos explican la **misma clase**
        """)
    elif num_classes > 2 or num_classes_lime > 2:
        st.markdown("""
        **ℹ️ Interpretación (modelo multiclase):**
        - **Ambos métodos** explican la **misma clase predicha** para una comparación justa
        - 🟢 **Verde**: palabras que aumentan la probabilidad de esa clase
        - 🔴 **Rojo**: palabras que disminuyen la probabilidad de esa clase
        
        💡 **Diferencias esperadas**: SHAP y LIME usan algoritmos distintos, por lo que pueden 
        identificar palabras diferentes, pero ambos están explicando la misma clase.
        """)

    return shap_values, lime_explanation, shap_time, lime_time


# ============================================================
# FUNCIÓN: MOSTRAR PREDICCIÓN DEL MODELO
# ============================================================
def mostrar_prediccion_modelo(input_text):
    """
    Muestra un análisis rápido de la predicción del modelo con diseño minimalista.

    Args:
        input_text: Texto de entrada analizado

    Returns:
        None (muestra directamente en Streamlit)
    """
    import streamlit.components.v1 as components

    st.markdown("---")

    # Obtener información del modelo y predicción
    prediction = st.session_state.classifier(input_text)[0]
    sentiment = prediction['label']
    confidence = prediction['score']

    # Obtener todas las probabilidades
    import torch
    with torch.no_grad():
        inputs = st.session_state.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        outputs = st.session_state.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        proba = probabilities.cpu().numpy().tolist()

    # Obtener número de clases y nombres
    num_classes = len(proba)
    if hasattr(st.session_state, 'class_names') and st.session_state.class_names:
        class_names = st.session_state.class_names
    else:
        class_names = [f'Clase_{i}' for i in range(num_classes)]

    # ============================================================
    # CONFIGURACIÓN DE COLORES
    # ============================================================
    color_map = {
        'POSITIVE': '#2ecc71',
        'NEGATIVE': '#e74c3c',
        'NEUTRAL': '#95a5a6',
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6',
        'joy': '#f1c40f',
        'sadness': '#3498db',
        'anger': '#e74c3c',
        'fear': '#9b59b6',
        'surprise': '#e67e22',
        'disgust': '#16a085',
        'love': '#e91e63'
    }

    main_color = color_map.get(sentiment, '#34495e')

    # ============================================================
    # OBTENER PALABRA MÁS INFLUYENTE
    # ============================================================
    shap_values = st.session_state.shap_explainer([input_text])
    tokens = shap_values[0].data

    if num_classes == 2:
        values_for_positive = shap_values[0].values[:, 1]
        sum_positive = np.sum(values_for_positive)
        if sum_positive > 0:
            values = shap_values[0].values[:, 1]
        else:
            values = shap_values[0].values[:, 0]
    else:
        try:
            predicted_idx = class_names.index(sentiment)
        except ValueError:
            predicted_idx = 0
        values = shap_values[0].values[:, predicted_idx]

    top_idx = np.argmax(np.abs(values))
    top_word = tokens[top_idx]
    top_value = values[top_idx]

    # ============================================================
    # CARD PRINCIPAL CON DOS COLUMNAS
    # ============================================================
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        # Filtrar clases con más de 20%
        relevant_classes = [(class_names[i], proba[i])
                            for i in range(num_classes) if proba[i] > 0.20]
        relevant_classes.sort(key=lambda x: x[1], reverse=True)

        # Construir HTML para las clases relevantes
        classes_pills = []
        for class_name, class_prob in relevant_classes:
            class_color = color_map.get(class_name, '#95a5a6')
            pill_html = f'''
            <div style="display: inline-block; margin: 5px 8px; padding: 8px 16px; background-color: {class_color}; border-radius: 20px;">
                <span style="color: white; font-weight: 600; font-size: 14px;">{class_name.upper()} {class_prob:.0%}</span>
            </div>
            '''
            classes_pills.append(pill_html)

        classes_html = ''.join(classes_pills)

        # Card principal con dos columnas
        html_content = f'''
        <div style="font-family: 'Source Sans Pro', sans-serif; background-color: #ffffff; border-radius: 6px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); border: 1px solid #e8e8e8; overflow: hidden;">
            <div style="display: flex; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 250px; padding: 30px 25px; border-right: 1px solid #e8e8e8; text-align: center;">
                    <p style="color: #7f8c8d; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Predicción</p>
                    <h1 style="color: #2c3e50; margin: 0 0 15px 0; font-size: 28px; font-weight: 600;">{sentiment.upper()}</h1>
                    <div>{classes_html}</div>
                </div>
                <div style="flex: 1; min-width: 250px; padding: 30px 25px; text-align: center;">
                    <p style="color: #7f8c8d; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Palabra clave</p>
                    <div style="margin-top: 10px;">
                        <div style="display: inline-block; padding: 10px 18px; background-color: {main_color}20; border-radius: 8px; border: 1px solid {main_color}40; margin-bottom: 10px;">
                            <span style="color: {main_color}; font-weight: 600; font-size: 20px;">"{top_word}"</span>
                        </div>
                        <p style="color: #95a5a6; font-size: 13px; margin-top: 8px;">Influencia: {abs(top_value):.3f}</p>
                    </div>
                </div>
            </div>
        </div>
        '''

        # Usar components.html en lugar de st.markdown
        components.html(html_content, height=200)

    st.markdown("---")
