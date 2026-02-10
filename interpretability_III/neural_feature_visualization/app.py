"""
===================================================================
APP.PY - Neural Feature Visualization Dashboard (REFACTORIZADO)
===================================================================

Aplicación Streamlit refactorizada con:
- Manejo robusto de session state
- Sin st.rerun() innecesarios
- Caché de objetos pesados
- Separación lógica vs UI
- Persistencia correcta entre interacciones

Autor: Neural Viz Team
Fecha: 2025-01-15 (Refactorizado)
===================================================================
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Importar módulos custom
from modules.model_manager import ModelManager
from modules.image_processor import ImageProcessor
from modules.neuron_analyzer import NeuronAnalyzer
from modules.feature_generator import FeatureGenerator
from modules.visualizer import Visualizer
from utils.cache_manager import get_cache
from utils.helpers import format_number, format_layer_name, format_model_name

# Importar configuración
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, SIDEBAR_STATE,
    WELCOME_MESSAGE, MAX_NEURONS_DISPLAY,
    AVAILABLE_MODELS, IMAGE_SIZE, ROI_SIZE
)


# ===================================================================
# CONFIGURACIÓN DE PÁGINA
# ===================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE
)


# ===================================================================
# SESSION STATE - MANAGEMENT MEJORADO
# ===================================================================

def init_session_state():
    """Inicializa session state con valores por defecto."""

    defaults = {
        # Flags de estado
        'image_loaded': False,
        'model_loaded': False,
        'heatmap_generated': False,
        'comparison_generated': False,

        # Datos
        'image_tensor': None,
        'image_visual': None,
        'image_pil': None,  # Nuevo: guardar PIL original
        'current_image_id': None,  # Nuevo: ID para detectar cambios de imagen
        'model': None,
        'model_manager': None,
        'activations': None,
        'neuron_stats': None,
        'heatmap': None,
        'roi_center': None,
        'synthetic_image': None,
        'roi_real': None,

        # Objetos cacheados (evitar recrear)
        'analyzer': None,
        'generator': None,
        'processor': None,
        'visualizer': None,

        # Config
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'current_model_name': None,
        'selected_layer': None,
        'selected_neuron': 0,
        'available_layers': [],

        # UI state
        'debug_mode': False,
        'show_logs': False,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_state(key: str, default=None):
    """Helper seguro para obtener valores del session state."""
    return st.session_state.get(key, default)


def set_state(key: str, value):
    """Helper para setear valores en session state."""
    st.session_state[key] = value


# ===================================================================
# STATE RESET FUNCTIONS
# ===================================================================

def reset_all():
    """Resetea TODO el estado (como reiniciar la app)."""
    keys_to_keep = ['device']  # Mantener solo device
    keys_to_delete = [k for k in st.session_state.keys()
                      if k not in keys_to_keep]

    for key in keys_to_delete:
        del st.session_state[key]

    init_session_state()


def reset_from_image():
    """Resetea estado cuando cambia la imagen."""
    reset_keys = [
        'heatmap_generated', 'comparison_generated',
        'activations', 'neuron_stats', 'heatmap',
        'roi_center', 'synthetic_image', 'roi_real',
        'analyzer', 'generator', 'selected_neuron'
    ]

    for key in reset_keys:
        if key in st.session_state:
            st.session_state[key] = None if key not in [
                'selected_neuron'] else 0

    st.session_state.heatmap_generated = False
    st.session_state.comparison_generated = False


def reset_from_model():
    """Resetea estado cuando cambia modelo o capa."""
    reset_keys = [
        'heatmap_generated', 'comparison_generated',
        'activations', 'neuron_stats', 'heatmap',
        'roi_center', 'synthetic_image', 'roi_real',
        'analyzer', 'generator', 'selected_neuron'
    ]

    for key in reset_keys:
        if key in st.session_state:
            st.session_state[key] = None if key not in [
                'selected_neuron'] else 0

    st.session_state.heatmap_generated = False
    st.session_state.comparison_generated = False


# ===================================================================
# CORE BUSINESS LOGIC (Separado de UI)
# ===================================================================

def load_and_process_image(image_source, source_type='upload'):
    """
    Carga y procesa una imagen.

    Args:
        image_source: Archivo subido, PIL Image, o URL
        source_type: 'upload', 'pil', 'url'

    Returns:
        bool: True si tuvo éxito
    """
    try:
        # 1. Obtener PIL Image según fuente
        if source_type == 'upload':
            image_pil = Image.open(image_source).convert('RGB')
        elif source_type == 'pil':
            image_pil = image_source
        elif source_type == 'url':
            import requests
            from io import BytesIO
            response = requests.get(image_source, timeout=10)
            response.raise_for_status()
            image_pil = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            return False

        # 2. Procesar imagen
        if st.session_state.processor is None:
            st.session_state.processor = ImageProcessor(
                device=st.session_state.device)

        img_tensor, img_visual = st.session_state.processor.load_and_preprocess(
            image_pil)

        # 3. Guardar en session state
        st.session_state.image_pil = image_pil
        st.session_state.image_tensor = img_tensor
        st.session_state.image_visual = img_visual
        st.session_state.image_loaded = True

        # 4. Resetear análisis dependientes
        reset_from_image()

        return True

    except Exception as e:
        st.error(f"❌ Error al procesar imagen: {e}")
        if st.session_state.debug_mode:
            import traceback
            st.code(traceback.format_exc())
        return False


@st.cache_resource
def load_model_cached(model_name: str):
    """Carga modelo con caché de Streamlit."""
    manager = ModelManager()
    model = manager.load_model(model_name)
    return model, manager


def generate_heatmap():
    """
    Genera el mapa de calor de activaciones.

    Returns:
        bool: True si tuvo éxito
    """
    try:
        # 1. Crear o reutilizar analyzer
        if (st.session_state.analyzer is None or
                st.session_state.analyzer.target_layer != st.session_state.selected_layer):

            # Limpiar analyzer anterior si existe
            if st.session_state.analyzer is not None:
                st.session_state.analyzer.cleanup()

            # Crear nuevo analyzer
            st.session_state.analyzer = NeuronAnalyzer(
                st.session_state.model,
                st.session_state.selected_layer,
                st.session_state.device
            )

        analyzer = st.session_state.analyzer

        # 2. Extraer activaciones
        activations = analyzer.extract_activations(
            st.session_state.image_tensor)
        st.session_state.activations = activations

        # 3. Calcular estadísticas
        stats = analyzer.compute_neuron_statistics(activations)
        st.session_state.neuron_stats = stats

        # 4. Generar heatmap
        heatmap = analyzer.compute_heatmap(activations, method='max')
        heatmap_resized = analyzer.resize_heatmap(heatmap, IMAGE_SIZE)
        st.session_state.heatmap = heatmap_resized

        # 5. Encontrar ROI
        roi_center = analyzer.find_max_activation_region(heatmap)

        # Escalar coordenadas al tamaño de imagen
        scale_y = IMAGE_SIZE[0] / heatmap.shape[0]
        scale_x = IMAGE_SIZE[1] / heatmap.shape[1]
        roi_center_scaled = (
            int(roi_center[0] * scale_y),
            int(roi_center[1] * scale_x)
        )
        st.session_state.roi_center = roi_center_scaled

        # 6. Marcar como generado
        st.session_state.heatmap_generated = True

        return True

    except Exception as e:
        st.error(f"❌ Error al generar mapa de calor: {e}")
        if st.session_state.debug_mode:
            import traceback
            st.code(traceback.format_exc())
        return False


def generate_comparison(neuron_idx: int):
    """
    Genera la comparación Real vs Sintética.

    Args:
        neuron_idx: Índice de la neurona

    Returns:
        bool: True si tuvo éxito
    """
    try:
        analyzer = st.session_state.analyzer

        # 1. Calcular ROI ESPECÍFICO para esta neurona
        neuron_activation_map = analyzer.get_neuron_activation_map(
            st.session_state.activations,
            neuron_idx
        )

        # Encontrar máximo de ESTA neurona
        roi_center_neuron = analyzer.find_max_activation_region(
            neuron_activation_map)

        # Escalar coordenadas al tamaño de imagen
        # [H, W] de las activaciones
        act_shape = st.session_state.activations.shape[2:]
        scale_y = IMAGE_SIZE[0] / act_shape[0]
        scale_x = IMAGE_SIZE[1] / act_shape[1]

        roi_center_scaled = (
            int(roi_center_neuron[0] * scale_y),
            int(roi_center_neuron[1] * scale_x)
        )

        # Guardar ROI específico de esta neurona
        st.session_state.roi_center = roi_center_scaled

        # 2. Extraer ROI real usando el centro específico de esta neurona
        roi_real = analyzer.extract_roi(
            st.session_state.image_visual,
            roi_center_scaled,
            ROI_SIZE
        )

        # 3. Obtener activación real
        real_activation = st.session_state.neuron_stats[neuron_idx]['mean']

        # 4. Crear o reutilizar generator
        if (st.session_state.generator is None or
                st.session_state.generator.target_layer != st.session_state.selected_layer):

            # Limpiar generator anterior si existe
            if st.session_state.generator is not None:
                st.session_state.generator.cleanup()

            # Crear nuevo generator
            st.session_state.generator = FeatureGenerator(
                st.session_state.model,
                st.session_state.selected_layer,
                st.session_state.device
            )

        generator = st.session_state.generator

        # 5. Generar patrón sintético
        synthetic_img, history = generator.generate_pattern(
            neuron_idx=neuron_idx,
            verbose=False
        )

        synthetic_activation = history['activation'][-1]

        # 6. Redimensionar sintética a tamaño de ROI
        from skimage.transform import resize
        synthetic_resized = resize(
            synthetic_img / 255.0,
            ROI_SIZE,
            anti_aliasing=True
        )

        # 7. Guardar resultados
        # Guardar AMBAS versiones: completa (para tiling) y redimensionada (para comparación)
        st.session_state.synthetic_image_full = synthetic_img / 255.0
        st.session_state.synthetic_image = synthetic_resized
        st.session_state.roi_real = roi_real
        st.session_state.real_activation = real_activation
        st.session_state.synthetic_activation = synthetic_activation
        st.session_state.selected_neuron_for_comparison = neuron_idx
        st.session_state.roi_center = roi_center_scaled
        st.session_state.comparison_generated = True

        return True

    except Exception as e:
        st.error(f"❌ Error al generar comparación: {e}")
        if st.session_state.debug_mode:
            import traceback
            st.code(traceback.format_exc())
        return False

# ===================================================================
# UI COMPONENTS
# ===================================================================


def section_image_upload():
    """Sección de carga de imagen."""

    st.header("📤 1. Carga de Imagen")

    upload_option = st.radio(
        "Selecciona una opción:",
        ["Subir imagen", "Usar imagen de muestra"],
        horizontal=True,
        key="upload_option_radio"
    )

    image_to_load = None
    source_type = None

    if upload_option == "Subir imagen":
        uploaded_file = st.file_uploader(
            "Sube una imagen (JPG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            key="file_uploader"
        )

        if uploaded_file is not None:
            image_to_load = uploaded_file
            source_type = 'upload'

    else:
        # Botón para cargar imagen de muestra
        if st.button("📥 Cargar Imagen de Muestra", key="load_sample"):
            with st.spinner("Descargando imagen..."):
                url = 'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400'
                success = load_and_process_image(url, 'url')

                if success:
                    st.success("✅ Imagen de muestra cargada")
                else:
                    st.warning("⚠️ Fallo descarga, usando imagen de respaldo")
                    fallback = Image.new(
                        'RGB', (224, 224), color=(100, 150, 200))
                    load_and_process_image(fallback, 'pil')

    # Procesar imagen subida
    if image_to_load is not None and source_type is not None:
        # Solo procesar si es una imagen NUEVA
        # Comparar con el nombre del archivo anterior
        new_image_id = None

        if source_type == 'upload':
            new_image_id = f"upload_{image_to_load.name}_{image_to_load.size}"

        current_image_id = st.session_state.get('current_image_id', None)

        # Solo procesar si cambió la imagen
        if new_image_id != current_image_id:
            success = load_and_process_image(image_to_load, source_type)
            if success:
                st.session_state.current_image_id = new_image_id
                st.success("✅ Imagen procesada correctamente")

    # Mostrar preview si hay imagen cargada
    if st.session_state.image_loaded and st.session_state.image_visual is not None:
        st.write("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                st.session_state.image_visual,
                caption="Imagen Cargada",
                use_column_width=True
            )

        if st.session_state.debug_mode:
            with st.expander("🔍 Debug Info - Imagen"):
                st.write(
                    f"Tensor shape: {st.session_state.image_tensor.shape}")
                st.write(
                    f"Visual shape: {st.session_state.image_visual.shape}")
                st.write(f"Device: {st.session_state.image_tensor.device}")

    elif not st.session_state.image_loaded:
        st.info("👆 Sube una imagen o carga la muestra para comenzar")


def section_model_config():
    """Sección de configuración de modelo y capa."""

    st.header("⚙️ 2. Configuración del Modelo")

    if not st.session_state.image_loaded:
        st.warning("⚠️ Primero debes cargar una imagen")
        return

    col1, col2 = st.columns(2)

    # Columna 1: Selección de modelo
    with col1:
        st.subheader("Modelo")

        model_options = list(AVAILABLE_MODELS.keys())
        model_display = [format_model_name(m) for m in model_options]

        # Encontrar índice actual
        current_idx = 0
        if st.session_state.current_model_name:
            try:
                current_idx = model_options.index(
                    st.session_state.current_model_name)
            except ValueError:
                pass

        selected_model_idx = st.selectbox(
            "Selecciona el modelo:",
            range(len(model_options)),
            index=current_idx,
            format_func=lambda i: model_display[i],
            key="model_selector"
        )

        selected_model = model_options[selected_model_idx]

        # Cargar modelo si cambió
        if selected_model != st.session_state.current_model_name:
            with st.spinner(f"Cargando {format_model_name(selected_model)}..."):
                model, manager = load_model_cached(selected_model)
                st.session_state.model = model
                st.session_state.model_manager = manager
                st.session_state.current_model_name = selected_model
                st.session_state.model_loaded = True

                # Obtener capas
                layers = manager.get_conv_layers(model)
                st.session_state.available_layers = layers
                st.session_state.selected_layer = layers[0] if layers else None

                # Resetear análisis
                reset_from_model()

                st.success(f"✅ {format_model_name(selected_model)} cargado")

        # Info del modelo
        model_info = AVAILABLE_MODELS[selected_model]
        st.caption(f"📝 {model_info['description']}")
        st.caption(f"💾 {model_info['size']}")

    # Columna 2: Selección de capa
    with col2:
        st.subheader("Capa")

        if st.session_state.model_loaded and st.session_state.available_layers:
            layers = st.session_state.available_layers

            # Encontrar índice actual
            current_layer_idx = 0
            if st.session_state.selected_layer in layers:
                current_layer_idx = layers.index(
                    st.session_state.selected_layer)

            selected_layer_idx = st.selectbox(
                "Selecciona la capa:",
                range(len(layers)),
                index=current_layer_idx,
                format_func=lambda i: f"{format_layer_name(layers[i])} ({layers[i]})",
                key="layer_selector"
            )

            selected_layer = layers[selected_layer_idx]

            # Si cambió la capa, resetear
            if selected_layer != st.session_state.selected_layer:
                st.session_state.selected_layer = selected_layer
                reset_from_model()

            # Info de la capa
            if st.session_state.model_manager:
                layer_info = st.session_state.model_manager.get_layer_info(
                    st.session_state.model,
                    selected_layer
                )
                st.caption(
                    f"📊 Canales: {layer_info.get('num_channels', 'N/A')}")
                st.caption(f"🔢 Kernel: {layer_info.get('kernel_size', 'N/A')}")


def section_heatmap_generation():
    """Sección de generación de mapa de calor."""

    st.header("🔥 3. Mapa de Activación")

    if not st.session_state.image_loaded or not st.session_state.model_loaded:
        st.warning("⚠️ Completa las secciones anteriores primero")
        return

    # Botón para generar heatmap
    if st.button("🔍 Generar Mapa de Calor", type="primary", key="gen_heatmap_btn"):
        with st.spinner("Analizando activaciones..."):
            success = generate_heatmap()

            if success:
                st.success("✅ Mapa de calor generado")

    # Mostrar resultados si ya fueron generados
    if st.session_state.heatmap_generated:
        st.write("---")

        # Crear visualizador si no existe
        if st.session_state.visualizer is None:
            st.session_state.visualizer = Visualizer()

        viz = st.session_state.visualizer

        # NUEVA visualización: Con marcadores de neuronas
        fig = viz.create_heatmap_with_neuron_markers(
            st.session_state.image_visual,
            st.session_state.activations,
            st.session_state.neuron_stats,
            top_n=MAX_NEURONS_DISPLAY,
            title="Mapa de Activación Neuronal"
        )

        st.pyplot(fig)
        plt.close(fig)

        # Ranking de neuronas
        st.subheader("📊 Neuronas Más Activas")

        ranked = sorted(
            st.session_state.neuron_stats,
            key=lambda x: x['mean'],
            reverse=True
        )[:MAX_NEURONS_DISPLAY]

        # Normalizar para barras - ASEGURAR RANGO [0, 1]
        if ranked:
            # Evitar división por 0
            max_activation = max(ranked[0]['mean'], 1e-8)
            min_activation = min(s['mean'] for s in ranked)
            activation_range = max_activation - min_activation

            # Si todas las activaciones son iguales
            if activation_range < 1e-8:
                activation_range = 1.0
        else:
            max_activation = 1.0
            min_activation = 0.0
            activation_range = 1.0

        for i, stat in enumerate(ranked, 1):
            cols = st.columns([0.3, 2, 1, 1])

            with cols[0]:
                emoji = "⭐" if i == 1 else f"{i}."
                st.write(emoji)

            with cols[1]:
                # Barra de progreso visual - normalizar a [0, 1]
                if activation_range > 0:
                    progress = (stat['mean'] - min_activation) / \
                        activation_range
                else:
                    progress = 1.0 if i == 1 else 0.0

                # Asegurar rango válido [0, 1]
                progress = max(0.0, min(1.0, progress))

                st.progress(progress, text=f"Neurona {stat['neuron_idx']}")

            with cols[2]:
                st.caption(f"μ: {stat['mean']:.3f}")

            with cols[3]:
                st.caption(f"max: {stat['max']:.3f}")


def section_comparison():
    """Sección de comparación Real vs Sintética."""

    st.header("🔬 4. Comparación: Real vs Ideal")

    if not st.session_state.heatmap_generated:
        st.warning("⚠️ Primero debes generar el mapa de calor")
        return

    # Selector de neurona
    st.subheader("Selección de Neurona")

    # Top neuronas
    ranked = sorted(
        st.session_state.neuron_stats,
        key=lambda x: x['mean'],
        reverse=True
    )[:MAX_NEURONS_DISPLAY]

    neuron_options = [s['neuron_idx'] for s in ranked]
    neuron_labels = [
        f"Neurona {s['neuron_idx']} (Act: {s['mean']:.3f})"
        for s in ranked
    ]

    # Selector sin callback (no causa rerun)
    selected_idx = st.selectbox(
        "Selecciona una neurona:",
        range(len(neuron_options)),
        format_func=lambda i: neuron_labels[i],
        key="neuron_selector"
    )

    selected_neuron = neuron_options[selected_idx]

    st.caption(f"📍 ROI: {st.session_state.roi_center}")

    # Botón para generar comparación
    if st.button("🎨 Generar Comparación", type="primary", key="gen_comparison_btn"):
        with st.spinner("Generando patrón sintético (10-30 seg)..."):

            # Progress bar
            progress_bar = st.progress(0)
            progress_bar.progress(30)

            success = generate_comparison(selected_neuron)

            progress_bar.progress(100)

            if success:
                st.success("✅ Comparación generada")

    # Mostrar comparación si ya fue generada
    if st.session_state.comparison_generated:
        st.write("---")

        viz = st.session_state.visualizer

        # Usar ROI específico de la neurona (no el global)
        roi_to_use = st.session_state.get(
            'roi_center_used', st.session_state.roi_center)

        # Visualización 1: Comparación 4-panel clásica
        st.subheader("🔬 Comparación Detallada: ROI vs Patrón Ideal")

        fig1 = viz.create_4panel_comparison(
            st.session_state.image_visual,
            st.session_state.roi_real,
            st.session_state.synthetic_image,
            roi_to_use,  # ✅ USA ROI ESPECÍFICO
            st.session_state.selected_neuron_for_comparison,
            st.session_state.real_activation,
            st.session_state.synthetic_activation
        )

        st.pyplot(fig1)
        plt.close(fig1)

        # Visualización 2: Patrón repetido (NUEVO)
        st.write("---")
        st.subheader("🎨 Visualización Global: Patrón Superpuesto")
        st.caption(
            "El patrón ideal se repite sobre toda la imagen para mostrar coincidencias globales")

        # Redimensionar patrón sintético completo (224x224) para tiling
        from skimage.transform import resize
        synthetic_full = st.session_state.synthetic_image

        # Si el patrón es pequeño (32x32), usar el generado completo
        if hasattr(st.session_state, 'synthetic_image_full'):
            synthetic_for_tile = st.session_state.synthetic_image_full
        else:
            # Usar el ROI redimensionado
            synthetic_for_tile = synthetic_full

        fig2 = viz.create_pattern_overlay_comparison(
            st.session_state.image_visual,
            synthetic_for_tile,
            roi_to_use,  # ✅ USA ROI ESPECÍFICO
            st.session_state.selected_neuron_for_comparison,
            st.session_state.real_activation,
            st.session_state.synthetic_activation
        )

        st.pyplot(fig2)
        plt.close(fig2)

        # Métricas
        st.subheader("📈 Métricas")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Activación Real",
                f"{st.session_state.real_activation:.3f}"
            )

        with col2:
            st.metric(
                "Activación Sintética",
                f"{st.session_state.synthetic_activation:.3f}"
            )

        with col3:
            improvement = (
                st.session_state.synthetic_activation /
                max(st.session_state.real_activation, 1e-8)
            )
            st.metric("Mejora", f"{improvement:.2f}x")


def inject_custom_css():
    """
    Inyecta CSS personalizado para eliminar el scroll del contenido principal.

    Comportamiento:
    - El contenido principal NO tiene scroll (overflow hidden)
    - El sidebar mantiene su scroll natural
    - La altura del main se fija al 100% del viewport
    """
    st.markdown("""
        <style>
        /* Eliminar scroll del contenido principal */
        .main .block-container {
            overflow-y: hidden !important;
            max-height: 100vh !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Asegurar que el main ocupe todo el viewport sin scroll */
        section[data-testid="stMain"] {
            overflow-y: hidden !important;
            height: 100vh !important;
        }
        
        /* El sidebar mantiene su scroll natural */
        section[data-testid="stSidebar"] {
            overflow-y: auto !important;
        }
        
        /* Ajustar padding para mejor visualización */
        .main {
            padding: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ===================================================================
# SIDEBAR
# ===================================================================


def render_sidebar():
    """Renderiza el sidebar."""

    with st.sidebar:
        st.title("🧠 Neural Viz")
        st.caption("Feature Visualization")

        st.divider()

        # Estado del sistema
        st.subheader("📊 Estado")

        device_icon = "🟢" if torch.cuda.is_available() else "🔵"
        st.write(f"{device_icon} Device: `{st.session_state.device}`")

        status_icon = "✅" if st.session_state.image_loaded else "⏳"
        st.write(
            f"{status_icon} Imagen: {'Cargada' if st.session_state.image_loaded else 'Pendiente'}")

        status_icon = "✅" if st.session_state.model_loaded else "⏳"
        model_name = st.session_state.current_model_name or 'N/A'
        st.write(f"{status_icon} Modelo: `{model_name}`")

        if st.session_state.selected_layer:
            st.write(f"📍 Capa: `{st.session_state.selected_layer}`")

        st.divider()

        # Debug mode
        st.session_state.debug_mode = st.checkbox(
            "🐛 Modo Debug",
            value=st.session_state.debug_mode,
            help="Mostrar información detallada"
        )

        st.divider()

        # Ayuda
        with st.expander("❓ Ayuda"):
            st.markdown("""
            **Pasos:**
            1. Carga una imagen
            2. Selecciona modelo y capa
            3. Genera mapa de calor
            4. Selecciona neurona
            5. Genera comparación
            """)

        st.divider()

        # Reset
        if st.button("🔄 Reiniciar Todo"):
            reset_all()
            st.rerun()


# ===================================================================
# MAIN APP
# ===================================================================

def main():
    """Función principal."""

    # Inicializar
    init_session_state()

    # Inyectar estilos CSS personalizados (eliminar scroll principal)
    # inject_custom_css()

    # Sidebar
    render_sidebar()

    # Título
    st.title(PAGE_TITLE)

    # Bienvenida
    with st.expander("👋 Bienvenida", expanded=False):
        st.markdown(WELCOME_MESSAGE)

    st.divider()

    # Secciones
    section_image_upload()
    st.divider()

    section_model_config()
    st.divider()

    section_heatmap_generation()
    st.divider()

    section_comparison()

    # Footer
    st.divider()
    st.caption("Desarrollado por @gaxoblanco | PyTorch & Streamlit")


# ===================================================================
# ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    main()
