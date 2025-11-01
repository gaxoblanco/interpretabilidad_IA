"""
🎯 Validación de Explicaciones - Métricas de Fidelidad
Implementa métricas para evaluar la calidad de explicaciones SHAP y LIME
"""

from src.utils.tokens import TokenFilter
import streamlit.components.v1 as components
import json
import numpy as np
import pandas as pd
import random
import torch
from typing import List, Dict, Tuple, Optional
import streamlit as st
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


class FidelityMetrics:
    """
    Clase para calcular métricas de fidelidad de explicaciones
    """

    def __init__(self, model, tokenizer):
        """
        Inicializa el analizador de fidelidad

        Args:
            model: Modelo de transformers
            tokenizer: Tokenizer correspondiente al modelo
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def predict_single(self, text: str) -> Tuple[int, float]:
        """
        Realiza predicción para un texto individual

        Args:
            text: Texto a predecir

        Returns:
            tuple: (clase_predicha, probabilidad_clase_original)
        """
        if not text.strip():
            return 0, 0.0

        # Tokenizar y predecir
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_class = probs.argmax().item()
        predicted_prob = probs[predicted_class].item()

        return predicted_class, predicted_prob

    def compute_fidelity_removal(self, text: str, importance_values: np.ndarray,
                                 top_k: int = 5, method: str = 'shap') -> float:
        """
        Calcula fidelidad eliminando las top-k palabras más importantes
        Usa Comprehensiveness: mide cuánto baja la prob de la clase ORIGINAL
        """

        # Predicción original
        original_class, original_prob = self.predict_single(text)

        # Tokenizar texto
        tokens = self.tokenizer.tokenize(text)

        # Validar que tenemos suficientes tokens
        if len(tokens) <= top_k:
            return 1.0

        # Ajustar si hay diferencia de longitud
        if len(importance_values) != len(tokens):
            min_len = min(len(importance_values), len(tokens))
            importance_values = importance_values[:min_len]
            tokens = tokens[:min_len]

        if method.lower() == 'lime':
            processed_values = np.abs(importance_values)
        else:
            processed_values = np.abs(importance_values)

        # Ya viene filtrado desde extraer_valores_shap_por_modelo
        # Obtener índices de las k palabras más importantes
        top_indices = np.argsort(processed_values)[-top_k:]

        # Crear texto sin las palabras más importantes
        modified_tokens = [token for i, token in enumerate(tokens)
                           if i not in top_indices]

        if not modified_tokens:
            return 1.0

        modified_text = self.tokenizer.convert_tokens_to_string(
            modified_tokens)

        # Predicción modificada
        if modified_text.strip():
            inputs = self.tokenizer(
                modified_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                modified_probs = torch.softmax(outputs.logits, dim=-1)[0]

            modified_prob_for_original_class = modified_probs[original_class].item(
            )

            # st.write(
            #     f"📊 **Predicción modificada:** Clase {original_class}, Prob {modified_prob_for_original_class:.4f}")

            comprehensiveness = original_prob - modified_prob_for_original_class

            # st.write(
            #     f"📉 **Comprehensiveness:** {comprehensiveness:.4f} (original - modificada)")

            # Normalizar a 0-1 (puede ser negativo si la prob SUBE)
            fidelity_score = max(0.0, min(comprehensiveness, 1.0))

            return fidelity_score
        else:
            return 1.0

    def compute_fidelity_correlation(self, text: str, importance_values: np.ndarray,
                                     n_perturbations: int = 50) -> float:
        """
        Calcula fidelidad mediante correlación entre importancia y cambio real

        Args:
            text: Texto original
            importance_values: Array con importancias de cada token
            n_perturbations: Número de perturbaciones a generar

        Returns:
            float: Correlación entre importancia SHAP/LIME y cambio real (0-1)
        """
        # Predicción original
        original_class, original_prob = self.predict_single(text)

        tokens = self.tokenizer.tokenize(text)

        if len(tokens) <= 1:
            return 0.0

        # Ajustar longitudes si es necesario
        if len(importance_values) != len(tokens):
            min_len = min(len(importance_values), len(tokens))
            importance_values = importance_values[:min_len]
            tokens = tokens[:min_len]

        importance_changes = []
        actual_changes = []

        # Generar perturbaciones aleatorias
        for _ in range(min(n_perturbations, 100)):  # Limitar para performance
            # Filtrar valores no semánticos
            importance_values = TokenFilter.filter_importance_values(
                tokens, importance_values
            )

            # Obtener solo índices semánticos para perturbaciones
            semantic_indices = TokenFilter.get_semantic_indices(tokens)

            if len(semantic_indices) <= 1:
                return 0.0
            # Seleccionar tokens aleatorios para eliminar
            n_remove = random.randint(1, min(3, len(tokens)))
            # Solo perturbar tokens semánticos
            n_remove_adjusted = min(n_remove, len(semantic_indices))
            indices_to_remove = random.sample(
                semantic_indices, n_remove_adjusted)

            # Suma de importancia de tokens eliminados
            importance_change = sum(
                abs(importance_values[i]) for i in indices_to_remove)

            # Crear texto perturbado
            perturbed_tokens = [token for i, token in enumerate(tokens)
                                if i not in indices_to_remove]

            if not perturbed_tokens:
                continue

            perturbed_text = self.tokenizer.convert_tokens_to_string(
                perturbed_tokens)

            if perturbed_text.strip():
                # Cambio real en predicción
                _, perturbed_prob = self.predict_single(perturbed_text)
                actual_change = abs(original_prob - perturbed_prob)

                importance_changes.append(importance_change)
                actual_changes.append(actual_change)

        # Calcular correlación
        if len(importance_changes) > 3:
            try:
                correlation, _ = spearmanr(importance_changes, actual_changes)
                return max(correlation, 0.0) if not np.isnan(correlation) else 0.0
            except:
                return 0.0
        return 0.0

    def evaluate_explanation_fidelity(self, text: str, shap_values: Optional[np.ndarray] = None,
                                      lime_values: Optional[np.ndarray] = None) -> Dict:
        """
        Evalúa fidelidad completa de explicaciones SHAP y/o LIME

        Args:
            text: Texto original
            shap_values: Valores SHAP (opcional)
            lime_values: Valores LIME (opcional)

        Returns:
            dict: Resultados de fidelidad para cada método
        """
        results = {}

        # Evaluar SHAP
        if shap_values is not None:
            shap_fidelity_removal = self.compute_fidelity_removal(
                text, shap_values)
            shap_fidelity_corr = self.compute_fidelity_correlation(
                text, shap_values)

            results['shap'] = {
                'removal_fidelity': shap_fidelity_removal,
                'correlation_fidelity': shap_fidelity_corr,
                'average_fidelity': (shap_fidelity_removal + shap_fidelity_corr) / 2
            }

        # Evaluar LIME
        if lime_values is not None:
            lime_fidelity_removal = self.compute_fidelity_removal(
                text, lime_values)
            lime_fidelity_corr = self.compute_fidelity_correlation(
                text, lime_values)

            results['lime'] = {
                'removal_fidelity': lime_fidelity_removal,
                'correlation_fidelity': lime_fidelity_corr,
                'average_fidelity': (lime_fidelity_removal + lime_fidelity_corr) / 2
            }

        return results

    def compare_fidelity(self, text: str, shap_values: np.ndarray,
                         lime_values: np.ndarray) -> Dict:
        """
        Compara fidelidad entre SHAP y LIME

        Args:
            text: Texto original
            shap_values: Valores SHAP
            lime_values: Valores LIME

        Returns:
            dict: Comparación detallada
        """
        results = self.evaluate_explanation_fidelity(
            text, shap_values, lime_values)

        if 'shap' in results and 'lime' in results:
            shap_avg = results['shap']['average_fidelity']
            lime_avg = results['lime']['average_fidelity']

            comparison = {
                'shap_fidelity': shap_avg,
                'lime_fidelity': lime_avg,
                'winner': 'SHAP' if shap_avg > lime_avg else 'LIME',
                'difference': abs(shap_avg - lime_avg),
                'detailed_results': results
            }

            return comparison

        return {'error': 'Necesita tanto SHAP como LIME para comparar'}


def extract_lime_values(lime_explanation, input_text: str, tokenizer) -> np.ndarray:
    """
    Extrae valores de importancia de una explicación LIME y los mapea a tokens
    Args:
        lime_explanation: Explicación LIME
        input_text: Texto original
        tokenizer: Tokenizer del modelo
    Returns:
        np.ndarray: Array de importancias por token
    """
    # Obtener la explicación como lista de (palabra, importancia)
    exp_list = lime_explanation.as_list()

    # Tokenizar el texto original
    tokens = tokenizer.tokenize(input_text)

    # Crear array de importancias (inicializado en 0)
    lime_values = np.zeros(len(tokens))

    # Mapear explicaciones LIME a tokens
    for word, importance in exp_list:
        word_clean = word.lower().strip()

        # Buscar coincidencias entre palabras LIME y tokens
        for i, token in enumerate(tokens):
            token_clean = token.lower().strip().replace('##', '').replace('Ġ', '')

            # Verificar si el token es semántico
            if not TokenFilter.is_semantic(token):
                continue

            # Diferentes estrategias de matching
            if (word_clean == token_clean or
                word_clean in token_clean or
                    token_clean in word_clean):
                lime_values[i] += importance  # ← CAMBIO: += en lugar de =
                # NO hacer break, puede haber múltiples tokens para una palabra

            lime_values = TokenFilter.filter_importance_values(
                tokens, lime_values)

    return lime_values


def extraer_valores_shap_por_modelo(shap_vals, model_number, original_pred, tokens):
    """
    Extrae valores SHAP según el tipo de modelo, filtrando puntuación
    """

    if shap_vals is None:
        return None

    if model_number in [3, 4]:  # Modelos multiclase
        if shap_vals.ndim == 1:
            combined_importance = np.abs(shap_vals[:len(tokens)])
        else:
            # Extraer clase predicha
            label = original_pred['label']

            if hasattr(st.session_state, 'class_names') and st.session_state.class_names:
                try:
                    class_idx = st.session_state.class_names.index(label)
                except ValueError:
                    class_idx = 0
            else:
                class_idx = 0

            # Extraer columna de la clase predicha
            combined_importance = np.abs(shap_vals[:len(tokens), class_idx])

        # Filtrar usando TokenFilter
        filtered_importance = TokenFilter.filter_importance_values(
            tokens, combined_importance
        )

        return filtered_importance

    else:  # Modelos binarios (1, 2)
        if shap_vals.ndim > 1:
            label = original_pred['label'].upper()

            if 'POSITIVE' in label or 'LABEL_2' in label:
                class_idx = 1
            elif 'NEGATIVE' in label or 'LABEL_0' in label:
                class_idx = 0
            elif 'NEUTRAL' in label or 'LABEL_1' in label:
                class_idx = 1
            else:
                class_idx = 1 if original_pred['score'] > 0.5 else 0

            values = shap_vals[:len(tokens), class_idx]
        else:
            values = shap_vals[:len(tokens)]

        # Filtrar usando TokenFilter
        filtered_values = TokenFilter.filter_importance_values(tokens, values)

        return filtered_values


def mostrar_validacion_explicaciones(input_text, shap_values=None, lime_explanation=None,
                                     method="Ambos (SHAP + LIME)"):
    """
    Muestra métricas de validación de las explicaciones en Streamlit
    """
    st.header("⚖️ Validación de Explicaciones")
    st.markdown("Evalúa qué tan confiables son las explicaciones generadas")

    # Inicializar métricas de fidelidad
    fidelity_analyzer = FidelityMetrics(
        st.session_state.model,
        st.session_state.tokenizer
    )

    # Extraer valores para SHAP
    shap_vals = None
    if shap_values is not None:
        if hasattr(shap_values, 'values'):
            raw_shap_vals = shap_values.values[0]
        else:
            raw_shap_vals = shap_values[0] if isinstance(
                shap_values, list) else shap_values

        # CRÍTICO: Procesar según tipo de modelo
        original_pred = st.session_state.classifier(input_text)[0]
        tokens = st.session_state.tokenizer.tokenize(input_text)
        model_number = st.session_state.current_model

        shap_vals = extraer_valores_shap_por_modelo(
            raw_shap_vals, model_number, original_pred, tokens
        )

    else:
        shap_vals = None

    # Extraer valores para LIME
    lime_vals = None
    if lime_explanation is not None:
        # Convertir explicación LIME a array de importancias
        lime_vals = extract_lime_values(
            lime_explanation, input_text, st.session_state.tokenizer)

    # === CALCULAR MÉTRICAS DE FIDELIDAD (FUERA DE LOS BLOQUES) ===
    shap_fidelity = None
    lime_fidelity = None

    if method in ["Solo SHAP", "Ambos (SHAP + LIME)"] and shap_vals is not None:
        with st.spinner("Calculando fidelidad SHAP..."):
            shap_fidelity = fidelity_analyzer.compute_fidelity_removal(
                input_text, shap_vals, top_k=5, method='shap'
            )

    if method in ["Solo LIME", "Ambos (SHAP + LIME)"] and lime_vals is not None:
        with st.spinner("Calculando fidelidad LIME..."):
            lime_fidelity = fidelity_analyzer.compute_fidelity_removal(
                input_text, lime_vals, top_k=5, method='lime'
            )

    # === MÉTRICAS INDIVIDUALES ===
    col1, col2 = st.columns(2)

    if method in ["Solo SHAP", "Ambos (SHAP + LIME)"] and shap_fidelity is not None:
        with col1:
            st.subheader("🔷 Fidelidad SHAP")

            # Mostrar score
            color = "green" if shap_fidelity > 0.7 else "orange" if shap_fidelity > 0.4 else "red"
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; border-radius: 10px; 
                        background-color: {color}20; border: 2px solid {color};'>
                <h3 style='color: {color}; margin: 0;'>{shap_fidelity:.3f}</h3>
                <p style='margin: 5px 0 0 0;'>Score de Fidelidad</p>
            </div>
            """, unsafe_allow_html=True)

            # Interpretación
            if shap_fidelity > 0.7:
                st.success(
                    "✅ **Muy confiable** - Las explicaciones SHAP reflejan bien el modelo")
            elif shap_fidelity > 0.4:
                st.warning(
                    "⚠️ **Moderado** - Las explicaciones SHAP son parcialmente confiables")
            else:
                st.error(
                    "❌ **Poco confiable** - Las explicaciones SHAP pueden ser engañosas")

    if method in ["Solo LIME", "Ambos (SHAP + LIME)"] and lime_fidelity is not None:
        with col2:
            st.subheader("🔶 Fidelidad LIME")

            # Mostrar score
            color = "green" if lime_fidelity > 0.7 else "orange" if lime_fidelity > 0.4 else "red"
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; border-radius: 10px; 
                        background-color: {color}20; border: 2px solid {color};'>
                <h3 style='color: {color}; margin: 0;'>{lime_fidelity:.3f}</h3>
                <p style='margin: 5px 0 0 0;'>Score de Fidelidad</p>
            </div>
            """, unsafe_allow_html=True)

            # Interpretación
            if lime_fidelity > 0.7:
                st.success("✅ **Muy confiable**")
            elif lime_fidelity > 0.4:
                st.warning("⚠️ **Moderado**")
            else:
                st.error("❌ **Poco confiable**")

    # === COMPARACIÓN DIRECTA (solo si ambos están disponibles) ===
    if method == "Ambos (SHAP + LIME)" and shap_fidelity is not None and lime_fidelity is not None:
        st.markdown("---")
        st.subheader("🔄 Comparación de Fidelidad")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # Gráfico de comparación
            comparison_data = pd.DataFrame({
                'Método': ['SHAP', 'LIME'],
                'Fidelidad': [shap_fidelity, lime_fidelity]
            })

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(comparison_data['Método'], comparison_data['Fidelidad'],
                          color=['#1f77b4', '#ff7f0e'], alpha=0.8)
            ax.set_ylabel('Score de Fidelidad')
            ax.set_title('Comparación de Fidelidad: SHAP vs LIME')
            ax.set_ylim(0, 1)

            # Agregar líneas de referencia
            ax.axhline(y=0.7, color='green', linestyle='--',
                       alpha=0.7, label='Alto')
            ax.axhline(y=0.4, color='orange', linestyle='--',
                       alpha=0.7, label='Moderado')

            # Agregar valores en las barras
            for bar, value in zip(bars, comparison_data['Fidelidad']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

            ax.legend()
            st.pyplot(fig, clear_figure=True)

        # Conclusión
        winner = "SHAP" if shap_fidelity > lime_fidelity else "LIME"
        difference = abs(shap_fidelity - lime_fidelity)

        if difference > 0.1:
            st.info(
                f"🏆 **{winner}** tiene significativamente mayor fidelidad (+{difference:.3f})")
        else:
            st.info(
                f"🤝 Ambos métodos tienen fidelidad similar (diferencia: {difference:.3f})")

    # === TEST DE ELIMINACIÓN CON TABS ===
    st.markdown("---")

    # Crear tabs en lugar de radio buttons
    tab_interactive, tab_automatic = st.tabs(
        ["🎯 Test Interactivo", "📊 Test Automático"])

    # === TAB INTERACTIVA ===
    with tab_interactive:
        st.subheader("Análisis de Impacto por Palabra")

        mostrar_analisis_impacto_palabras(
            input_text=input_text,
            shap_vals=shap_vals if method in [
                "Solo SHAP", "Ambos (SHAP + LIME)"] else None,
            lime_vals=lime_vals if method in [
                "Solo LIME", "Ambos (SHAP + LIME)"] else None,
            method=method
        )

    # === TAB AUTOMÁTICA ===
    with tab_automatic:
        st.subheader("Test Automático de Reemplazo de Palabras")
        with st.expander("Ver análisis detallado", expanded=True):
            if method in ["Solo SHAP", "Ambos (SHAP + LIME)"] and shap_vals is not None:
                mostrar_test_eliminacion(input_text, shap_vals, "SHAP")

            if method in ["Solo LIME", "Ambos (SHAP + LIME)"] and lime_vals is not None:
                mostrar_test_eliminacion(input_text, lime_vals, "LIME")


def detectar_tipo_tokenizador(model_number):
    """Detecta el tipo de tokenizador según el modelo"""
    tipo_tokenizador_map = {
        1: 'wordpiece',  # DistilBERT
        2: 'bpe',        # RoBERTa
        3: 'bpe',        # DistilRoBERTa
        4: 'wordpiece'   # BERT
    }
    return tipo_tokenizador_map.get(model_number, 'wordpiece')


def agrupar_tokens_en_palabras(tokens, model_number):
    """
    Agrupa tokens en palabras completas según el tipo de tokenizador

    Returns:
        list: Lista de diccionarios con información de palabras agrupadas
    """
    tipo_tokenizador = detectar_tipo_tokenizador(model_number)
    palabras_agregadas = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if tipo_tokenizador == 'bpe':
            # RoBERTa/DistilRoBERTa: 'Ġ' o '▁' marca inicio de palabra
            es_inicio = token.startswith('Ġ') or token.startswith('▁')
            clean_token = token.replace('Ġ', '').replace('▁', '')

            if es_inicio or i == 0:
                palabra_actual = clean_token
                indices_tokens = [i]

                # Buscar continuaciones (sin Ġ)
                j = i + 1
                while j < len(tokens):
                    siguiente = tokens[j]
                    if not siguiente.startswith('Ġ') and not siguiente.startswith('▁'):
                        palabra_actual += siguiente.replace(
                            'Ġ', '').replace('▁', '')
                        indices_tokens.append(j)
                        j += 1
                    else:
                        break

                palabras_agregadas.append({
                    'palabra': palabra_actual,
                    'tokens_originales': [tokens[idx] for idx in indices_tokens],
                    'indices': indices_tokens
                })
                i = j
            else:
                i += 1

        else:  # wordpiece (BERT/DistilBERT)
            # '##' marca continuación
            clean_token = token.replace('##', '')
            es_continuacion = token.startswith('##')

            if not es_continuacion:
                palabra_actual = clean_token
                indices_tokens = [i]

                # Buscar continuaciones (con ##)
                j = i + 1
                while j < len(tokens):
                    siguiente = tokens[j]
                    if siguiente.startswith('##'):
                        palabra_actual += siguiente.replace('##', '')
                        indices_tokens.append(j)
                        j += 1
                    else:
                        break

                palabras_agregadas.append({
                    'palabra': palabra_actual,
                    'tokens_originales': [tokens[idx] for idx in indices_tokens],
                    'indices': indices_tokens
                })
                i = j
            else:
                i += 1

    return palabras_agregadas


def mostrar_analisis_impacto_palabras(input_text, shap_vals, lime_vals, method):
    """
    Análisis de impacto con UI de dos niveles: palabras agregadas + detalle de tokens

    REEMPLAZA líneas 508-608 de fidelity_explanation.py
    """
    # ========================================================================
    # PASO 1: Tokenización y preparación
    # ========================================================================
    tokens = st.session_state.tokenizer.tokenize(input_text)
    original_pred = st.session_state.classifier(input_text)[0]
    model_number = st.session_state.current_model

    # Extraer valores SHAP/LIME
    shap_values_1d = extraer_valores_shap_por_modelo(
        shap_vals, model_number, original_pred, tokens)
    lime_values_1d = lime_vals[:len(tokens)] if lime_vals is not None else None

    # Info del modelo
    tipo_tok = detectar_tipo_tokenizador(model_number)
    st.write(
        f"**Modelo #{model_number}** ({tipo_tok.upper()}) - "
        f"{original_pred['label']} ({original_pred['score']:.3f})"
    )

    if shap_values_1d is not None:
        max_shap = np.max(np.abs(shap_values_1d))
        if max_shap < 0.001:
            st.warning(
                f"⚠️ Valores SHAP muy pequeños para modelo #{model_number}. "
                "Esto puede ser normal para este tipo de modelo."
            )

    # ========================================================================
    # PASO 2: AGRUPAR TOKENS EN PALABRAS
    # ========================================================================
    palabras_agregadas = agrupar_tokens_en_palabras(tokens, model_number)

    # ========================================================================
    # PASO 3: CALCULAR IMPORTANCIA AGREGADA POR PALABRA
    # ========================================================================
    word_analysis = []
    token_details = {}  # Para guardar detalles de tokens por palabra

    for palabra_info in palabras_agregadas:
        palabra = palabra_info['palabra']
        indices = palabra_info['indices']
        tokens_palabra = palabra_info['tokens_originales']

        # ====================================================================
        # 3.1: Agregar valores SHAP/LIME
        # ====================================================================
        shap_val_total = 0.0
        lime_val_total = 0.0
        detalles_tokens = []

        for idx in indices:
            shap_token = 0.0
            lime_token = 0.0

            if shap_values_1d is not None and idx < len(shap_values_1d):
                shap_token = shap_values_1d[idx]
                shap_val_total += shap_token

            if lime_values_1d is not None and idx < len(lime_values_1d):
                lime_token = lime_values_1d[idx]
                lime_val_total += lime_token

            # Guardar detalle del token
            detalles_tokens.append({
                'token': tokens[idx],
                'shap': shap_token,
                'lime': lime_token
            })

        # Guardar detalles para expandir después
        token_details[palabra] = detalles_tokens

        # ====================================================================
        # 3.2: FILTRAR palabras no semánticas usando TokenFilter
        # ====================================================================
        if not TokenFilter.is_semantic(palabra):
            impacto_real = 0.0
            nivel = "🚫 Ignorado"

            word_analysis.append({
                'Palabra': palabra,
                'Num_Tokens': len(indices),
                'SHAP': f"{shap_val_total:.4f}",
                'LIME': f"{lime_val_total:.4f}",
                'Impacto Real': f"{impacto_real:.4f}",
                'Importancia': nivel
            })
            continue

        # ====================================================================
        # 3.3: Calcular IMPACTO REAL reemplazando la PALABRA COMPLETA
        # ====================================================================
        tokens_modificados = tokens.copy()

        reemplazos_neutrales = {
            'terrible': 'okay', 'awful': 'okay', 'horrible': 'okay',
            'bad': 'okay', 'boring': 'okay', 'poor': 'okay',
            'worst': 'average', 'disappointing': 'okay',
            'fantastic': 'okay', 'amazing': 'okay', 'excellent': 'decent',
            'wonderful': 'okay', 'great': 'okay', 'best': 'average',
            'perfect': 'adequate', 'brilliant': 'okay',
            'waste': 'use', 'masterpiece': 'film', 'disaster': 'event',
            'poorly': 'adequately', 'extremely': 'somewhat',
            'very': 'somewhat', 'really': 'somewhat'
        }

        palabra_lower = palabra.lower()
        reemplazo = reemplazos_neutrales.get(palabra_lower, 'okay')

        # Reemplazar TODOS los tokens de esta palabra
        for idx_token, idx_palabra in enumerate(indices):
            if idx_token == 0:
                # Primer token: preservar formato
                if tokens[idx_palabra].startswith('Ġ'):
                    tokens_modificados[idx_palabra] = 'Ġ' + reemplazo
                elif tokens[idx_palabra].startswith('▁'):
                    tokens_modificados[idx_palabra] = '▁' + reemplazo
                else:
                    tokens_modificados[idx_palabra] = reemplazo
            else:
                # Tokens subsiguientes: eliminar
                tokens_modificados[idx_palabra] = ''

        texto_modificado = st.session_state.tokenizer.convert_tokens_to_string(
            [t for t in tokens_modificados if t != ''])

        if texto_modificado.strip() and len(texto_modificado) > 10:
            try:
                pred_modificada = st.session_state.classifier(texto_modificado)[
                    0]
                if original_pred['label'] == pred_modificada['label']:
                    impacto_real = abs(
                        original_pred['score'] - pred_modificada['score'])
                else:
                    impacto_real = original_pred['score']
            except Exception:
                impacto_real = 0.0
        else:
            impacto_real = 0.0

        # ====================================================================
        # 3.4: Clasificar importancia
        # ====================================================================
        max_importancia = max(abs(shap_val_total), abs(lime_val_total))
        if max_importancia > 0.05:
            nivel = "🔴 Alta"
        elif max_importancia > 0.01:
            nivel = "🟡 Media"
        else:
            nivel = "🟢 Baja"

        word_analysis.append({
            'Palabra': palabra,
            'Num_Tokens': len(indices),
            'SHAP': f"{shap_val_total:.4f}",
            'LIME': f"{lime_val_total:.4f}",
            'Impacto Real': f"{impacto_real:.4f}",
            'Importancia': nivel
        })

    # ========================================================================
    # PASO 4: ORDENAMIENTO INTELIGENTE
    # ========================================================================
    shap_disponible = shap_values_1d is not None
    lime_disponible = lime_values_1d is not None

    max_shap = max([abs(float(row['SHAP']))
                   for row in word_analysis]) if shap_disponible else 0
    max_lime = max([abs(float(row['LIME']))
                   for row in word_analysis]) if lime_disponible else 0

    if max_shap > 0.001 and max_lime > 0.001:
        sort_key = [(abs(float(row['SHAP'])) + abs(float(row['LIME']))) / 2
                    for row in word_analysis]
        metodo_texto = "SHAP + LIME"
    elif max_shap > 0.001:
        sort_key = [abs(float(row['SHAP'])) for row in word_analysis]
        metodo_texto = "SHAP"
    elif max_lime > 0.001:
        sort_key = [abs(float(row['LIME'])) for row in word_analysis]
        metodo_texto = "LIME"
    else:
        sort_key = [abs(float(row['Impacto Real'])) for row in word_analysis]
        metodo_texto = "Impacto Real Medido"

    df_analysis = pd.DataFrame(word_analysis)
    df_sorted = df_analysis.iloc[np.argsort(sort_key)[::-1]]

    # Filtrar ignorados
    df_filtrado = df_sorted[~df_sorted['Importancia'].str.contains('Ignorado')]

    if len(df_filtrado) == 0:
        df_filtrado = df_sorted
        st.warning("⚠️ Solo se encontraron tokens no semánticos")

    # ========================================================================
    # PASO 5: VISUALIZACIÓN CON DOS NIVELES
    # ========================================================================
    st.markdown("### 📊 Análisis por Palabras (Tokens Agregados)")

    # Mostrar tabla principal
    st.dataframe(df_filtrado, use_container_width=True)

    # ========================================================================
    # NIVEL 2: DESGLOSE DE TOKENS (EXPANDIBLE)
    # ========================================================================
    with st.expander("🔍 Ver desglose detallado por tokens", expanded=False):
        st.markdown("#### Detalle Token por Token")

        # Para cada palabra en la tabla filtrada, mostrar sus tokens
        for idx, row in df_filtrado.head(10).iterrows():  # Top 10 palabras
            palabra = row['Palabra']

            if palabra in token_details:
                detalles = token_details[palabra]

                st.markdown(
                    f"**📌 Palabra: `{palabra}`** (SHAP Total: {row['SHAP']} | LIME Total: {row['LIME']})")

                # Crear tabla de tokens
                token_data = []
                for detalle in detalles:
                    token_data.append({
                        'Token': detalle['token'],
                        'SHAP': f"{detalle['shap']:.4f}",
                        'LIME': f"{detalle['lime']:.4f}"
                    })

                df_tokens = pd.DataFrame(token_data)
                st.dataframe(df_tokens, use_container_width=True)
                st.markdown("---")

    # ========================================================================
    # EXPLICACIÓN Y ADVERTENCIAS
    # ========================================================================
    with st.expander("ℹ️ ¿Cómo se calculan estos valores?"):
        st.markdown(f"""
        **Agregación de Tokens a Palabras (Tokenizador: {tipo_tok.upper()}):**
        
        - 🔤 **Palabra completa**: Si "Terrible" se tokeniza como `['Ter', '##rible']` (WordPiece) 
          o `['B', 'oring']` (BPE), se agrupa como UNA palabra
        - ➕ **SHAP/LIME Total**: Se SUMAN los valores de todos los tokens que forman la palabra
        - 🎯 **Impacto Real**: Se mide reemplazando la PALABRA COMPLETA (no tokens individuales)
        - 📊 **Num_Tokens**: Cantidad de tokens que forman la palabra
        - 🚫 **Filtrado**: Se excluyen puntuación y conectores del análisis (usando TokenFilter)
        
        **Ejemplo:**
        - Palabra: "Boring"
        - Tokens: `['B', 'oring']` (2 tokens)
        """)

    if max_shap < 0.001 and shap_disponible:
        st.warning(f"⚠️ Valores SHAP muy pequeños (max: {max_shap:.6f})")

    if max_lime < 0.001 and lime_disponible:
        st.warning(f"⚠️ Valores LIME muy pequeños (max: {max_lime:.6f})")

    # ========================================================================
    # RECOMENDACIONES
    # ========================================================================
    st.markdown(f"### 💡 Recomendaciones (basado en {metodo_texto})")

    top_palabras = df_filtrado.head(3)['Palabra'].tolist()

    if len(top_palabras) > 0:
        st.success(f"**Palabras más influyentes:** {', '.join(top_palabras)}")

        palabra_mas_importante = df_filtrado.iloc[0]['Palabra']
        impacto_mas_importante = df_filtrado.iloc[0]['Impacto Real']
        num_tokens = df_filtrado.iloc[0]['Num_Tokens']

        st.info(
            f"💡 La palabra **'{palabra_mas_importante}'** (formada por {num_tokens} token(s)) "
            f"tiene un impacto estimado de **{impacto_mas_importante}** al ser reemplazada."
        )
    else:
        st.error("❌ No se pudieron identificar palabras influyentes")


def mostrar_test_eliminacion(input_text, importance_values, method_name):
    """
    Muestra test de eliminación paso a paso - reemplazando con palabras neutrales/opuestas
    """
    st.markdown(f"**Test automático para {method_name}:**")

    # Obtener predicción original
    original_pred = st.session_state.classifier(input_text)[0]
    original_prob = original_pred['score']
    original_label = original_pred['label']

    st.markdown(
        f"**Predicción original:** {original_label} ({original_prob:.3f})")

    # Diccionarios de reemplazos
    neutral_replacements = {
        # Adjetivos positivos → neutrales
        'fantastic': 'okay', 'amazing': 'okay', 'excellent': 'decent', 'wonderful': 'okay',
        'great': 'okay', 'best': 'average', 'perfect': 'adequate', 'brilliant': 'okay',
        'outstanding': 'decent', 'superb': 'okay', 'magnificent': 'decent',

        # Adjetivos negativos → neutrales
        'terrible': 'okay', 'awful': 'okay', 'horrible': 'okay', 'bad': 'okay',
        'boring': 'okay', 'worst': 'average', 'poor': 'okay', 'disappointing': 'okay',
        'waste': 'use', 'poorly': 'adequately',

        # Sustantivos emotivos → neutrales
        'masterpiece': 'film', 'disaster': 'movie', 'gem': 'film',

        # Intensificadores → neutrales
        'absolutely': 'somewhat', 'completely': 'somewhat', 'totally': 'somewhat',
        'extremely': 'somewhat', 'really': 'somewhat', 'very': 'somewhat'
    }

    # Tokenizar
    tokens = st.session_state.tokenizer.tokenize(input_text)

    # Preparar importance_values
    if importance_values.ndim > 1:
        importance_values = importance_values.flatten()

    importance_subset = importance_values[:len(tokens)]

    results = []
    for k in [1, 3, 5]:
        if k >= len(tokens):
            continue

        # Obtener top-k SEMÁNTICOS + valores filtrados
        top_indices, filtered_importance = TokenFilter.get_top_k_semantic_indices(
            tokens=tokens,
            importance_values=importance_subset,
            k=k
        )

        # Mostrar información de debug (útil para validar)
        if k == 1:  # Solo para el primer caso
            st.caption(f"🔍 Top-{k} tokens semánticos seleccionados:")
            for idx in top_indices:
                token = tokens[idx]
                importance = filtered_importance[idx]
                st.caption(
                    f"  • Token: '{token}' | Importancia: {importance:.4f}")

        # CREAR TEXTO CON REEMPLAZOS NEUTRALES
        modified_tokens = []
        replacements_made = []

        for i, token in enumerate(tokens):
            if i in top_indices:
                # Ya usamos get_top_k_semantic_indices que son semánticos

                # Limpiar token para buscar reemplazo
                clean_token = TokenFilter.clean_token(token).lower()

                if clean_token in neutral_replacements:
                    replacement = neutral_replacements[clean_token]
                    if token.startswith('Ġ'):  # GPT-2/RoBERTa
                        modified_tokens.append('Ġ' + replacement)
                    elif token.startswith('##'):  # BERT WordPiece
                        modified_tokens.append('##' + replacement)
                    else:
                        modified_tokens.append(replacement)
                    replacements_made.append(f"{clean_token}→{replacement}")
                else:
                    # Reemplazo genérico neutral
                    if token.startswith('Ġ'):
                        modified_tokens.append('Ġokay')
                    elif token.startswith('##'):
                        modified_tokens.append('##okay')
                    else:
                        modified_tokens.append('okay')
                    replacements_made.append(f"{clean_token}→okay")
            else:
                # Mantener token original
                modified_tokens.append(token)

        # Convertir tokens modificados a texto
        modified_text = st.session_state.tokenizer.convert_tokens_to_string(
            modified_tokens)

        if modified_text.strip():
            # Nueva predicción con texto modificado
            new_pred = st.session_state.classifier(modified_text)[0]
            change = abs(original_prob - new_pred['score'])

            # Agregar importancia promedio de tokens reemplazados
            avg_importance = np.mean([filtered_importance[idx]
                                     for idx in top_indices]) if top_indices else 0

            results.append({
                'Cantidad': k,
                'Reemplazos': ', '.join(replacements_made),
                'Importancia Promedio': f'{avg_importance:.4f}',
                'Cambio Probabilidad': f'{change:.3f}',
                'Nueva Predicción': f"{new_pred['label']} ({new_pred['score']:.3f})",
                'Texto Modificado': modified_text[:60] + "..." if len(modified_text) > 60 else modified_text
            })

    # Mostrar tabla de resultados
    if results:
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        # ⭐ CAMBIO 5: Análisis de fidelidad mejorado
        st.markdown("### 📊 Análisis de Fidelidad")

        if len(results) >= 2:
            # Comparar k=1 vs k=3
            change_k1 = float(results[0]['Cambio Probabilidad'])
            change_k3 = float(results[1]['Cambio Probabilidad']) if len(
                results) > 1 else 0

            if change_k1 > 0.1:
                st.success(
                    f"✅ **Alta fidelidad:** Reemplazar top-1 token causó cambio de {change_k1:.3f}")
            elif change_k1 > 0.05:
                st.info(
                    f"ℹ️ **Fidelidad moderada:** Reemplazar top-1 token causó cambio de {change_k1:.3f}")
            else:
                st.warning(
                    f"⚠️ **Baja fidelidad:** Reemplazar top-1 token solo causó cambio de {change_k1:.3f}")
                st.caption(
                    "Posibles causas: Tokens con baja importancia real o modelo robusto a perturbaciones")

        st.markdown(
            "**Estrategia:** Se reemplazan palabras semánticamente importantes con equivalentes neutrales "
            "para medir su impacto real en la predicción del modelo.")


def interpret_fidelity_score(score: float) -> Tuple[str, str, str]:
    """
    Interpreta un score de fidelidad

    Args:
        score: Score de fidelidad (0-1)

    Returns:
        tuple: (interpretacion, color, emoji)
    """
    if score >= 0.7:
        return "Muy confiable", "green", "✅"
    elif score >= 0.5:
        return "Moderadamente confiable", "orange", "⚠️"
    elif score >= 0.3:
        return "Poco confiable", "red", "❌"
    else:
        return "No confiable", "darkred", "🚫"


# Funciones de compatibilidad para mantener código existente
def evaluate_fidelity(model, tokenizer, texts: List[str],
                      explanations: List[np.ndarray]) -> Dict:
    """
    Función de compatibilidad - evalúa fidelidad en lote
    """
    fidelity_analyzer = FidelityMetrics(model, tokenizer)
    fidelities = []

    for text, explanations_array in zip(texts, explanations):
        try:
            fidelity = fidelity_analyzer.compute_fidelity_removal(
                text, explanations_array)
            fidelities.append(fidelity)
        except Exception as e:
            print(f"Error procesando texto: {e}")
            continue

    if fidelities:
        return {
            'mean_fidelity': np.mean(fidelities),
            'std_fidelity': np.std(fidelities),
            'median_fidelity': np.median(fidelities),
            'min_fidelity': np.min(fidelities),
            'max_fidelity': np.max(fidelities),
            'individual_scores': fidelities,
            'num_samples': len(fidelities)
        }
    else:
        return {
            'error': 'No se pudieron procesar los textos',
            'num_samples': 0
        }
