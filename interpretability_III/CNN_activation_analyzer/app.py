"""
Dashboard de Análisis de Activaciones de CNN
Aplicación Streamlit para visualizar activaciones de ResNet18 y AlexNet
"""

import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils_streamlit import (
    StreamlitImageAnalyzer,
    create_activation_heatmap,
    create_filter_grid,
    get_available_conv_layers,
    fig_to_image
)

# Configuración de la página
st.set_page_config(
    page_title="CNN Activation Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Título principal
st.title("🔬 Analizador de Activaciones de CNN")
st.markdown("""
Esta aplicación permite analizar las activaciones internas de redes neuronales convolucionales 
(ResNet18 y AlexNet) para entender qué patrones detecta cada capa.
""")

# Configuración del dispositivo


@st.cache_resource
def get_device():
    """Determina el dispositivo disponible (GPU o CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


device = get_device()

# Cargar modelo


@st.cache_resource
def load_model(model_name: str):
    """
    Carga el modelo pre-entrenado.

    Args:
        model_name: 'resnet18' o 'alexnet'

    Returns:
        Modelo PyTorch
    """
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    model.eval()
    return model


# Sidebar - Configuración
st.sidebar.header("⚙️ Configuración")

# Selección de modelo
model_name = st.sidebar.selectbox(
    "Selecciona el modelo",
    options=['alexnet', 'resnet18'],
    index=0,
    help="Elige entre ResNet18 (más moderno) o AlexNet (clásico)"
)

# Cargar modelo seleccionado
with st.spinner(f"Cargando modelo {model_name}..."):
    model = load_model(model_name)

st.sidebar.success(f"✅ Modelo {model_name} cargado")
st.sidebar.caption(f"🖥️ Dispositivo: {device}")

# Obtener capas disponibles
conv_layers = get_available_conv_layers(model, model_name)

# Selección de capa
st.sidebar.subheader("🎯 Capa a Analizar")
selected_layer = st.sidebar.selectbox(
    "Selecciona una capa convolucional",
    options=conv_layers,
    index=0,
    help="Capas más tempranas detectan patrones simples, capas profundas detectan conceptos complejos"
)

# Parámetros de visualización
# Parámetros de visualización
st.sidebar.subheader("📊 Parámetros de Visualización")

# Detectar número de filtros en la capa seleccionada


@st.cache_data
def get_layer_num_filters(_model, layer_name, _device):
    """Obtiene el número de filtros en una capa específica."""
    dummy_input = torch.randn(1, 3, 224, 224).to(_device)
    num_filters_found = [0]

    def hook_fn(module, input, output):
        num_filters_found[0] = output.shape[1]

    # Registrar hook temporal
    hook_handle = None
    for name, module in _model.named_modules():
        if name == layer_name:
            hook_handle = module.register_forward_hook(hook_fn)
            break

    # Forward pass para obtener dimensiones
    with torch.no_grad():
        _ = _model(dummy_input)

    # Limpiar hook
    if hook_handle:
        hook_handle.remove()

    return num_filters_found[0]


# Obtener número de filtros
num_filters_in_layer = get_layer_num_filters(model, selected_layer, device)

# Limitar slider al número real de filtros (máximo 24 para no saturar la UI)
max_selectable = min(num_filters_in_layer, 24)

top_k = st.sidebar.slider(
    f"Número de neuronas más activas",
    min_value=4,
    max_value=max_selectable,
    value=min(12, max_selectable),
    step=4,
    help=f"Capa '{selected_layer}' tiene {num_filters_in_layer} filtros totales"
)

criterion = st.sidebar.selectbox(
    "Criterio de selección",
    options=['balanced', 'mean', 'max', 'std'],
    index=0,
    help=(
        "balanced: Neuronas activas Y selectivas (RECOMENDADO) - "
        "evita filtros que se activan en todo (como fondos)\n"
        "mean: Solo por activación promedio\n"
        "max: Solo por activación máxima\n"
        "std: Solo por variabilidad"
    )
)

# Parámetros adicionales para balanced
if criterion == 'balanced':
    st.sidebar.markdown("**Ajustes de Balance:**")

    activation_weight = st.sidebar.slider(
        "Peso Activación vs Selectividad",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help=(
            "1.0 = Solo activación (puede incluir fondos)\n"
            "0.5 = Balance 50/50 (RECOMENDADO)\n"
            "0.0 = Solo selectividad (filtros muy específicos)"
        )
    )

    min_sparsity = st.sidebar.slider(
        "Sparsity Mínima Requerida",
        min_value=0.0,
        max_value=0.8,  # ← Aumentar rango
        value=0.5,  # ← Empezar en 0 por defecto
        step=0.05,
        help=(
            "Filtra neuronas con sparsity menor a este valor.\n\n"
            "**Recomendaciones por capa:**\n"
            "• Capas tempranas (conv1, layer1): 0.05-0.10\n"
            "• Capas medias (layer2, layer3): 0.10-0.20\n"
            "• Capas profundas (layer4): 0.20-0.40\n\n"
            "**Valores comunes:**\n"
            "• 0.0 = Sin filtro (incluye fondos)\n"
            "• 0.15 = Filtrar fondos uniformes (RECOMENDADO)\n"
            "• 0.30 = Solo neuronas muy selectivas"
        )
    )
    if min_sparsity == 0.0:
        st.sidebar.warning(
            "⚠️ Sin filtro de sparsity: puede incluir neuronas de fondo")
else:
    activation_weight = 0.6
    min_sparsity = 0.0

alpha = st.sidebar.slider(
    "Transparencia del heatmap",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

cmap = st.sidebar.selectbox(
    "Colormap",
    options=['jet', 'viridis', 'hot', 'cool', 'plasma'],
    index=0
)

# ========================================================================
# DETECCIÓN DE CAMBIOS EN SIDEBAR
# ========================================================================

# Detectar si hubo cambios en los parámetros del sidebar
# Inicializar valores previos si no existen
if 'prev_model' not in st.session_state:
    st.session_state['prev_model'] = model_name
    st.session_state['prev_layer'] = selected_layer
    st.session_state['prev_top_k'] = top_k
    st.session_state['prev_criterion'] = criterion
    st.session_state['prev_activation_weight'] = activation_weight
    st.session_state['prev_min_sparsity'] = min_sparsity
    st.session_state['prev_alpha'] = alpha
    st.session_state['prev_cmap'] = cmap

# Verificar si hubo cambios
params_changed = (
    st.session_state['prev_model'] != model_name or
    st.session_state['prev_layer'] != selected_layer or
    st.session_state['prev_top_k'] != top_k or
    st.session_state['prev_criterion'] != criterion or
    st.session_state['prev_activation_weight'] != activation_weight or
    st.session_state['prev_min_sparsity'] != min_sparsity or
    st.session_state['prev_alpha'] != alpha or
    st.session_state['prev_cmap'] != cmap
)
# Si hubo cambios, hacer scroll al inicio
if params_changed:
    # LIMPIAR RESULTADOS VIEJOS
    if 'results' in st.session_state:
        del st.session_state['results']

    # Agregar timestamp único para forzar ejecución del script
    import time
    timestamp = int(time.time() * 1000)

    st.markdown(f"""
        <script id="scroll-script-{timestamp}">
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
    """, unsafe_allow_html=True)

    # Actualizar valores previos
    st.session_state['prev_model'] = model_name
    st.session_state['prev_layer'] = selected_layer
    st.session_state['prev_top_k'] = top_k
    st.session_state['prev_criterion'] = criterion
    st.session_state['prev_activation_weight'] = activation_weight
    st.session_state['prev_min_sparsity'] = min_sparsity
    st.session_state['prev_alpha'] = alpha
    st.session_state['prev_cmap'] = cmap

# Sección principal - Carga de imagen
st.header("📸 Carga de Imagen")

# Si ya hay imagen, mostrarla con botón para cambiar
if 'current_image' in st.session_state:
    col_img, col_btn = st.columns([3, 1])

    with col_img:
        st.image(st.session_state['current_image'],
                 caption="Imagen a analizar",
                 width=400)

    with col_btn:
        if st.button("🔄 Cambiar imagen", use_container_width=True):
            st.session_state['show_uploader'] = True
            st.rerun()

# Mostrar uploader si no hay imagen o si se presionó "Cambiar"
if 'current_image' not in st.session_state or st.session_state.get('show_uploader', False):
    col1, col2 = st.columns([1, 1])

    with col1:
        image_option = st.radio(
            "Selecciona una opción:",
            options=["Usar imagen de ejemplo", "Subir mi propia imagen"],
            index=0
        )

    with col2:
        if image_option == "Usar imagen de ejemplo":
            example_url = st.text_input(
                "URL de imagen de ejemplo",
                value="https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400",
                help="Ingresa una URL de imagen o usa la predeterminada"
            )

            if st.button("🔄 Cargar imagen de ejemplo"):
                try:
                    from urllib.request import urlopen
                    pil_image = Image.open(urlopen(example_url))
                    st.session_state['current_image'] = pil_image
                    st.session_state['show_uploader'] = False
                    st.success("✅ Imagen cargada correctamente")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error al cargar imagen: {str(e)}")
        else:
            uploaded_file = st.file_uploader(
                "Sube una imagen",
                type=['jpg', 'jpeg', 'png'],
                help="Formatos soportados: JPG, JPEG, PNG"
            )

            if uploaded_file is not None:
                pil_image = Image.open(uploaded_file)
                st.session_state['current_image'] = pil_image
                st.session_state['show_uploader'] = False
                st.success("✅ Imagen subida correctamente")
                st.rerun()

# Mostrar imagen actual
if 'current_image' in st.session_state:
    # st.subheader("🖼️ Imagen Original")
    # st.image(st.session_state['current_image'],
    #          caption="Imagen a analizar", width=400)

    # Botón de análisis
    if st.button("🚀 Analizar Activaciones", type="primary"):

        # ═══════════════════════════════════════════════════════════════
        # LIMPIAR RESULTADOS ANTERIORES DE TAB 6
        # ═══════════════════════════════════════════════════════════════
        if 'ablation_results' in st.session_state:
            del st.session_state['ablation_results']
        if 'ablation_neurons' in st.session_state:
            del st.session_state['ablation_neurons']
        if 'ablation_amp_factor' in st.session_state:
            del st.session_state['ablation_amp_factor']
        if 'ablation_noise_level' in st.session_state:
            del st.session_state['ablation_noise_level']
        # Crear columnas para imagen + progreso
        col_img_prog, col_status_prog = st.columns([1, 1])
        with st.spinner("Analizando imagen... Esto puede tomar unos segundos..."):
            try:
                # Crear analizador
                analyzer = StreamlitImageAnalyzer(
                    model=model,
                    target_layer=selected_layer,
                    device=device
                )

                # Procesar imagen
                img_tensor, img_vis = analyzer.load_image_from_pil(
                    st.session_state['current_image']
                )

                # Analizar
                results = analyzer.analyze_image(img_tensor)
                activations = results['activations']

                # Calcular estadísticas
                stats = analyzer.get_neuron_statistics(activations)

                # Debug: mostrar distribución de sparsity
                if criterion == 'balanced' and min_sparsity > 0:
                    sparsity_values = [s['sparsity'] for s in stats]
                    num_above_threshold = sum(
                        1 for s in sparsity_values if s >= min_sparsity)
                    max_sparsity = max(sparsity_values)
                    st.info(f"🔍 **Filtrado de Neuronas**: De {len(stats)} totales, "
                            f"**{num_above_threshold}** tienen sparsity ≥ {min_sparsity:.0%}. "
                            f"| Sparsity promedio: {np.mean(sparsity_values):.1%} "
                            f"| Máxima: {max_sparsity:.1%}")

                top_neurons = analyzer.get_top_neurons(
                    stats,
                    top_k=top_k,
                    criterion=criterion,
                    activation_weight=activation_weight,
                    min_sparsity=min_sparsity
                )

                # Guardar resultados en session_state
                st.session_state['results'] = {
                    'activations': activations,
                    'prediction': results['prediction'],
                    'confidence': results['confidence'],
                    'stats': stats,
                    'top_neurons': top_neurons,
                    'img_vis': img_vis,
                    'layer_name': selected_layer,
                    'image_tensor': img_tensor
                }

                # Cleanup
                analyzer.cleanup()

                st.success("✅ Análisis completado!")

                # Scroll al inicio de la página
                st.markdown("""
                    <script>
                        window.parent.document.querySelector('section.main').scrollTo(0, 0);
                    </script>
                """, unsafe_allow_html=True)

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error durante el análisis: {str(e)}")
                st.exception(e)

# Mostrar resultados si existen
if 'results' in st.session_state:
    results = st.session_state['results']
    # Definir activations para usarlo en todos los tabs
    activations = results['activations']

    st.markdown("---")
    st.header("📊 Resultados del Análisis")

    # Crear tabs para organizar resultados
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Resultados Generales",
        "🔥 Heatmaps Superpuestos",
        "🎨 Grid de Filtros",
        "🔬 Análisis Detallado",
        "🎯 Visualización de Filtros",
        "🤖 Predicción del Modelo",
        "🧪 Experimentos de Ablación"
    ])

    # ===================================================================
    # TAB 0: RESULTADOS GENERALES
    # ===================================================================
    with tab0:
        st.info("💡 Vista general del análisis: información de la capa, predicción del modelo, y top neuronas más activas.")

        # Sección 1: Información General
        st.subheader("1️⃣ Información General")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Capa Analizada", results['layer_name'])

        with col2:
            st.metric("Clase Predicha", f"#{results['prediction']}")

        with col3:
            st.metric("Confianza", f"{results['confidence']:.2%}")

        with col4:
            shape = activations.shape
            st.metric("Neuronas Totales", shape[1])

        st.markdown("---")

        # Sección 2: Estadísticas Globales
        st.subheader("2️⃣ Estadísticas Globales de Activaciones")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Media", f"{activations.mean().item():.4f}")

        with col2:
            st.metric("Máximo", f"{activations.max().item():.4f}")

        with col3:
            st.metric("Desv. Estándar", f"{activations.std().item():.4f}")

        with col4:
            sparsity = (activations == 0).float().mean().item() * 100
            st.metric("Sparsity", f"{sparsity:.1f}%")

        # Información adicional de la capa
        with st.expander("📈 Ver información detallada de la capa"):
            st.write(f"**Shape completo:** {activations.shape}")
            st.write(f"**Batch size:** {activations.shape[0]}")
            st.write(f"**Número de neuronas:** {activations.shape[1]}")
            st.write(f"**Alto del mapa:** {activations.shape[2]} píxeles")
            st.write(f"**Ancho del mapa:** {activations.shape[3]} píxeles")

            total_values = activations.numel()
            st.write(f"**Total de valores:** {total_values:,}")
            st.write(
                f"**Memoria aprox:** {total_values * 4 / 1024 / 1024:.2f} MB")

            # Distribución de valores
            num_positive = (activations > 0).sum().item()
            num_zero = (activations == 0).sum().item()
            num_negative = (activations < 0).sum().item()

            st.write("**Distribución de valores:**")
            st.write(
                f"- Positivos: {num_positive:,} ({num_positive/total_values*100:.1f}%)")
            st.write(
                f"- Ceros: {num_zero:,} ({num_zero/total_values*100:.1f}%)")
            st.write(
                f"- Negativos: {num_negative:,} ({num_negative/total_values*100:.1f}%)")

        st.markdown("---")

        # Sección 3: Top Neuronas
        st.subheader(
            f"3️⃣ Top {len(results['top_neurons'])} Neuronas Más Activas")

        st.caption(f"Criterio de selección: {criterion}")

        # Crear tabla de top neuronas
        import pandas as pd

        top_data = []
        for rank, neuron_idx in enumerate(results['top_neurons'], 1):
            s = results['stats'][neuron_idx]
            top_data.append({
                'Rank': rank,
                'Neurona': neuron_idx,
                'Media': f"{s['mean']:.4f}",
                'Máxima': f"{s['max']:.4f}",
                'Std': f"{s['std']:.4f}",
                'Sparsity': f"{s['sparsity']*100:.1f}%"
            })

        df = pd.DataFrame(top_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.info(
            "💡 Usa los otros tabs para explorar visualizaciones detalladas de estas neuronas.")

    with tab1:
        st.write("**Visualización de las neuronas más interesantes**")

        # Toggle para mostrar/ocultar imagen de fondo
        col_toggle1, col_toggle2 = st.columns([3, 1])

        with col_toggle1:
            st.info(
                f"💡 Las activaciones fuertes (rojo/amarillo) indican dónde el filtro detectó patrones relevantes.")

        with col_toggle2:
            show_background = st.checkbox(
                "🖼️ Mostrar imagen",
                value=True,
                help="Activa/desactiva la imagen de fondo en los heatmaps"
            )

        # Mostrar los top 5 más interesantes por defecto
        num_heatmaps_to_show = min(top_k, len(results['top_neurons']))

        # Crear grid de heatmaps (2 columnas)
        cols_per_row = 3
        num_rows = (num_heatmaps_to_show + cols_per_row - 1) // cols_per_row

        for row_idx in range(num_rows):
            cols = st.columns(cols_per_row)

            for col_idx in range(cols_per_row):
                neuron_list_idx = row_idx * cols_per_row + col_idx

                if neuron_list_idx < num_heatmaps_to_show:
                    neuron_idx = results['top_neurons'][neuron_list_idx]
                    neuron_stats = results['stats'][neuron_idx]

                    with cols[col_idx]:
                        # Crear heatmap
                        act_map = activations[0,
                                              neuron_idx, :, :].cpu().numpy()

                        if show_background:
                            # Con imagen de fondo
                            fig = create_activation_heatmap(
                                image_vis=results['img_vis'],
                                activation_map=act_map,
                                title=f"#{neuron_list_idx + 1}: Filtro {neuron_idx}",
                                alpha=alpha,
                                cmap=cmap,
                                figsize=(3, 3)
                            )
                        else:
                            # Solo mapa de calor (sin imagen de fondo)
                            fig, ax = plt.subplots(figsize=(3, 3))

                            # Redimensionar mapa de activación
                            from scipy.ndimage import zoom
                            h, w = results['img_vis'].shape[:2]
                            h_act, w_act = act_map.shape

                            if (h_act, w_act) != (h, w):
                                zoom_factors = (h / h_act, w / w_act)
                                act_resized = zoom(
                                    act_map, zoom_factors, order=1)
                            else:
                                act_resized = act_map

                            # Mostrar solo el mapa de calor
                            im = ax.imshow(act_resized, cmap=cmap)
                            ax.set_title(f"#{neuron_list_idx + 1}: Filtro {neuron_idx}",
                                         fontsize=14, fontweight='bold')
                            ax.axis('off')
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            plt.tight_layout()

                        st.pyplot(fig)
                        plt.close(fig)

                        # Mostrar estadísticas debajo
                        st.caption(f"**Media:** {neuron_stats['mean']:.3f} | "
                                   f"**Máx:** {neuron_stats['max']:.3f} | "
                                   f"**Std:** {neuron_stats['std']:.3f}")

        # Separador
        st.markdown("---")

        # Opción para ver una neurona específica con más detalle
        with st.expander("🔍 Ver neurona específica en detalle"):
            selected_neuron_idx = st.selectbox(
                "Selecciona una neurona",
                options=results['top_neurons'],
                index=0,
                format_func=lambda x: f"Neurona {x} (Rank {results['top_neurons'].index(x) + 1})"
            )

            # Toggle individual para el detalle
            show_bg_detail = st.checkbox(
                "🖼️ Mostrar imagen de fondo",
                value=True,
                key="detail_bg"
            )

            # Crear heatmap grande
            act_map = activations[0, selected_neuron_idx, :, :].cpu().numpy()

            if show_bg_detail:
                # Con imagen de fondo
                fig = create_activation_heatmap(
                    image_vis=results['img_vis'],
                    activation_map=act_map,
                    title=f"Mapa de Activación - Neurona {selected_neuron_idx}",
                    alpha=alpha,
                    cmap=cmap,
                    figsize=(4, 4)
                )
            else:
                # Solo mapa de calor
                fig, ax = plt.subplots(figsize=(4, 4))

                from scipy.ndimage import zoom
                h, w = results['img_vis'].shape[:2]
                h_act, w_act = act_map.shape

                if (h_act, w_act) != (h, w):
                    zoom_factors = (h / h_act, w / w_act)
                    act_resized = zoom(act_map, zoom_factors, order=1)
                else:
                    act_resized = act_map

                im = ax.imshow(act_resized, cmap=cmap)
                ax.set_title(f"Mapa de Activación - Neurona {selected_neuron_idx}",
                             fontsize=14, fontweight='bold')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # Información detallada de la neurona seleccionada
            neuron_stats = results['stats'][selected_neuron_idx]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Media", f"{neuron_stats['mean']:.4f}")
            with col2:
                st.metric("Máxima", f"{neuron_stats['max']:.4f}")
            with col3:
                st.metric("Std", f"{neuron_stats['std']:.4f}")
            with col4:
                st.metric("Sparsity", f"{neuron_stats['sparsity']*100:.1f}%")

    with tab2:
        st.write("**Grid de todos los filtros más activos numerados**")

        # Crear grid
        fig_grid = create_filter_grid(
            activations=activations,
            neuron_indices=results['top_neurons'],
            image_vis=results['img_vis'],
            max_cols=4,
            cmap=cmap
        )

        st.pyplot(fig_grid)
        plt.close(fig_grid)

        st.caption("""
        Cada panel muestra el mapa de activación de un filtro específico. Los números en las esquinas 
        indican el ranking (1 = más activo). El número en el título corresponde al índice real de la 
        neurona en la capa.
        """)

    with tab3:
        st.write("**Análisis detallado: ¿Por qué se activó cada filtro?**")

        st.info(
            "💡 Analiza cada filtro individualmente para entender qué detectó y dónde.")

        # CSS para permitir wrap en tabs
        st.markdown("""
            <style>
            /* Permitir que los tabs hagan wrap en múltiples líneas */
            div[data-baseweb="tab-list"] {
                flex-wrap: wrap !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Importar funciones necesarias
        from filter_visualization import (
            compute_activation_regions,
            create_image_with_filter_patches,
            explain_filter_activation
        )

        # Pre-calcular todas las regiones de activación
        num_filtros_detalle = min(top_k, len(results['top_neurons']))
        all_regions = {}
        for neuron_idx in results['top_neurons'][:num_filtros_detalle]:
            act_map = activations[0, neuron_idx, :, :].cpu().numpy()
            all_regions[neuron_idx] = compute_activation_regions(
                act_map,
                results['img_vis'].shape[:2],
                threshold_percentile=75,
                min_regions=3
            )

        # Crear pestañas para cada uno de los 5 filtros
        filter_tabs = st.tabs([
            f"Filtro {results['top_neurons'][i]} (#{i+1})"
            for i in range(num_filtros_detalle)
        ])

        # Generar contenido para cada pestaña
        for tab_idx, filter_tab in enumerate(filter_tabs):
            neuron_idx = results['top_neurons'][tab_idx]

            with filter_tab:
                # Obtener mapa de activación
                act_map = activations[0, neuron_idx, :, :].cpu().numpy()
                regions = all_regions[neuron_idx]

                # Layout principal: Columna izquierda (visualizaciones) | Columna derecha (análisis)
                col_left, col_right = st.columns([1.3, 1])

                # ===================================================================
                # COLUMNA IZQUIERDA: Visualizaciones (Regiones + Heatmap)
                # ===================================================================
                with col_left:
                    # Sección 1: Regiones + Patrón del Filtro
                    st.markdown("### 🖼️ Regiones + Patrón del Filtro")

                    fig_combined = create_image_with_filter_patches(
                        image_vis=results['img_vis'],
                        activation_map=act_map,
                        model=model,
                        layer_name=results['layer_name'],
                        filter_idx=neuron_idx,
                        max_boxes=3
                    )

                    st.pyplot(fig_combined)
                    plt.close(fig_combined)

                    st.caption(
                        "🔴 Roja: Mayor | 🟡 Amarilla: Media | 🟢 Verde: Menor")

                    # Separador
                    st.markdown("---")

                    # Mostrar descomposición RGB del filtro
                    st.markdown(
                        "### 🎨 Patrón del Filtro")

                    # Importar función para extraer pesos del filtro
                    from filter_visualization import extract_filter_weights_rgb, create_rgb_channel_visualization

                    try:
                        # Intentar extraer pesos RGB
                        filter_weights = extract_filter_weights_rgb(
                            model=model,
                            layer_name=results['layer_name'],
                            filter_idx=neuron_idx
                        )

                        if filter_weights is not None:
                            # Capa RGB real - Mostrar descomposición completa
                            st.markdown("#### Descomposición por Canales RGB")

                            fig_channels = create_rgb_channel_visualization(
                                filter_weights=filter_weights,
                                filter_idx=neuron_idx
                            )

                            st.pyplot(fig_channels)
                            plt.close(fig_channels)

                            st.caption(
                                "💡 **Interpretación**: Esta es una capa temprana que opera directamente sobre píxeles RGB. "
                                "Cada canal muestra qué intensidad de ese color busca el filtro."
                            )
                        else:
                            # Capa profunda - Mostrar explicación
                            st.info(
                                f"ℹ️ **Capa profunda detectada**: `{results['layer_name']}`\n\n"
                                "Esta capa no opera sobre píxeles RGB directamente, sino sobre "
                                "representaciones abstractas aprendidas por capas anteriores.\n\n"
                                "**¿Qué significa esto?**\n"
                                "- Los filtros tienen cientos de canales de entrada (no solo 3 RGB)\n"
                                "- No se pueden visualizar como 'patrones de color'\n"
                                "- Las **regiones de activación** arriba son una forma de entender qué detecta este filtro"
                            )

                    except Exception as e:
                        st.error(f"❌ Error al visualizar filtro: {str(e)}")

                    # Sección 2: Heatmap completo (justo debajo)
                    # st.markdown("---")
                    # st.markdown("### 🌡️ Mapa de Calor Completo")

                    # fig_heat = create_activation_heatmap(
                    #     image_vis=results['img_vis'],
                    #     activation_map=act_map,
                    #     title=f"Filtro {neuron_idx}",
                    #     alpha=alpha,
                    #     cmap=cmap,
                    #     figsize=(3, 3)
                    # )
                    # st.pyplot(fig_heat)
                    # plt.close(fig_heat)

                # ===================================================================
                # COLUMNA DERECHA: Análisis (se mantiene a la derecha todo el tiempo)
                # ===================================================================
                with col_right:
                    st.markdown(f"### 🔍 Análisis")

                    # Estadísticas
                    neuron_stats = results['stats'][neuron_idx]

                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Media", f"{neuron_stats['mean']:.4f}")
                        st.metric("Desv. Std", f"{neuron_stats['std']:.4f}")
                    with metric_col2:
                        st.metric("Máxima", f"{neuron_stats['max']:.4f}")

                        # Indicador visual de sparsity
                        sparsity_pct = neuron_stats['sparsity']*100
                        if sparsity_pct < 10:
                            st.metric("Sparsity", f"{sparsity_pct:.1f}%",
                                      delta="Baja", delta_color="inverse")
                        elif sparsity_pct < 30:
                            st.metric("Sparsity", f"{sparsity_pct:.1f}%",
                                      delta="Media", delta_color="off")
                        else:
                            st.metric("Sparsity", f"{sparsity_pct:.1f}%",
                                      delta="Alta", delta_color="normal")

                    st.markdown("---")

                    # Explicación textual
                    explanation = explain_filter_activation(
                        filter_idx=neuron_idx,
                        regions=regions,
                        neuron_stats=neuron_stats
                    )
                    st.markdown(explanation)

                    # Interpretación simple
                    st.markdown("---")
                    st.markdown("#### 💡 Interpretación")
                    if len(regions) > 0:
                        if neuron_stats['sparsity'] > 0.5:
                            st.success(
                                f"✅ Filtro selectivo: {len(regions)} región(es) específica(s)")
                        else:
                            st.info(
                                f"ℹ️ Filtro general: {len(regions)} región(es) detectada(s)")

                        avg_intensity = np.mean(
                            [r['intensity'] for r in regions[:3]])
                        if avg_intensity > 0.7:
                            st.write("🔥 Coincidencia fuerte del patrón")
                        elif avg_intensity > 0.4:
                            st.write("⚡ Coincidencia moderada")
                        else:
                            st.write("💫 Coincidencia débil")
                    else:
                        st.warning("⚠️ Patrón no detectado en esta imagen")

    with tab4:
        st.write("**Visualización de Filtros: Patches detectados y Patrones RGB**")
        st.info(
            "💡 Explora qué partes de la imagen activaron cada filtro y qué patrón busca.")

        # Importar funciones
        from filter_visualization import (
            create_filter_grid_rgb,
            create_activation_patches_visualization
        )

        # ===================================================================
        # SECCIÓN 1: PESOS RGB DE LOS FILTROS
        # ===================================================================
        st.subheader("🎨 Patrones RGB de los Filtros")
        st.caption(
            "Estos son los pesos aprendidos (kernel de 7×7) que busca cada filtro")

        with st.spinner("Generando visualización de filtros RGB..."):
            try:
                fig_filters = create_filter_grid_rgb(
                    model=model,
                    layer_name=results['layer_name'],
                    filter_indices=results['top_neurons'][:top_k],
                    num_cols=6
                )
                st.pyplot(fig_filters)
                plt.close(fig_filters)

                st.info(
                    "💡 **Interpretación**: En capas tempranas (conv1, layer1), estos patrones representan "
                    "colores y texturas. En capas profundas son más abstractos."
                )
            except Exception as e:
                st.warning(
                    f"⚠️ No se pueden visualizar filtros RGB para esta capa: {str(e)}")
                st.caption(
                    "**Limitación**: Solo capas convolucionales tempranas (que reciben entrada RGB) "
                    "pueden visualizarse como patrones de color."
                )

        st.markdown("---")

        # ===================================================================
        # SECCIÓN 2: PATCHES DE IMAGEN QUE ACTIVARON CADA FILTRO
        # ===================================================================
        st.subheader("📸 Patches de Imagen que Activaron cada Filtro")
        st.caption("Fragmentos reales de la imagen que más activaron cada filtro")

        # Mostrar patches para los top 5 filtros
        for idx, neuron_idx in enumerate(results['top_neurons'][:top_k], 1):
            act_map = activations[0, neuron_idx, :, :].cpu().numpy()

            # Encabezado del filtro
            col_title, col_stats = st.columns([2, 1])
            with col_title:
                st.markdown(f"### Filtro {neuron_idx} (Rank #{idx})")
            with col_stats:
                neuron_stats = results['stats'][neuron_idx]
                st.caption(
                    f"Media: {neuron_stats['mean']:.3f} | "
                    f"Sparsity: {neuron_stats['sparsity']*100:.0f}%"
                )

            # Generar visualización de patches
            fig_patches = create_activation_patches_visualization(
                image_vis=results['img_vis'],
                activation_map=act_map,
                filter_idx=neuron_idx,
                num_patches=3
            )
            st.pyplot(fig_patches)
            plt.close(fig_patches)

            st.markdown("---")

        # Nota final
        st.info(
            "💡 **Tip**: Compara los patrones RGB (arriba) con los patches detectados (aquí) "
            "para entender qué está buscando y encontrando cada filtro."
        )

    # ===================================================================
    # TAB 5: PREDICCIÓN DEL MODELO
    # ===================================================================
    with tab5:
        st.write("**Información sobre la predicción del modelo**")
        st.info("💡 Esta pestaña muestra qué clase detectó el modelo en la imagen.")

        # Predicción principal
        pred = results['prediction']
        conf = results['confidence']

        # Diccionario completo de clases ImageNet (las más comunes)
        IMAGENET_CLASSES = {
            # Gatos
            281: "gato atigrado (tabby cat)",
            282: "gato tigre (tiger cat)",
            283: "gato persa (Persian cat)",
            284: "gato siamés (Siamese cat)",
            285: "gato egipcio (Egyptian cat)",

            # Perros
            207: "golden retriever",
            208: "labrador retriever",
            209: "pastor alemán (German shepherd)",
            235: "pastor belga (Belgian sheepdog)",
            236: "cocker spaniel",

            # Otros animales
            151: "chimpancé",
            388: "ballena jorobada (humpback whale)",
            33: "lobo gris (grey wolf)",
            334: "mono aullador (howler monkey)",

            # Objetos
            404: "avión de pasajeros (airliner)",
            436: "guitarra acústica (acoustic guitar)",
            511: "piscina (swimming pool)",
            779: "computadora portátil (laptop)",

            # Vehículos
            609: "jeep",
            656: "minivan",
            751: "racer (coche de carreras)",
        }

        # Obtener nombre de la clase
        class_name = IMAGENET_CLASSES.get(pred, f"Clase ImageNet #{pred}")

        # Banner principal
        st.markdown("### 🔍 Predicción Principal")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if conf > 0.7:
                st.success(f"### {class_name}")
            elif conf > 0.4:
                st.warning(f"### {class_name}")
            else:
                st.error(f"### {class_name}")

        with col2:
            st.metric("Confianza", f"{conf:.1%}")

        with col3:
            st.metric("Clase ID", f"#{pred}")

        st.markdown("---")

        # Interpretación de la confianza
        st.markdown("### 📈 Interpretación de la Confianza")

        if conf > 0.9:
            st.success(
                "✅ **Muy alta confianza** - El modelo está muy seguro de esta predicción.")
        elif conf > 0.7:
            st.info("👍 **Alta confianza** - El modelo está razonablemente seguro.")
        elif conf > 0.4:
            st.warning(
                "⚠️ **Confianza moderada** - El modelo tiene algunas dudas.")
        else:
            st.error(
                "❌ **Baja confianza** - El modelo no está seguro de la predicción.")

        # Barra de progreso visual
        st.progress(conf)

        st.markdown("---")

        # Información sobre ImageNet
        st.markdown("### ℹ️ Sobre las Clases de ImageNet")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **¿Qué es ImageNet?**
            
            ImageNet es un conjunto de datos con más de 14 millones de imágenes 
            organizadas en **1,000 categorías** diferentes.
            
            Los modelos ResNet18 y AlexNet fueron entrenados con este dataset 
            para reconocer objetos, animales, vehículos, y más.
            """)

        with col2:
            st.markdown("""
            **Categorías principales:**
            
            - 🐱 Animales (perros, gatos, aves, etc.)
            - 🚗 Vehículos (coches, aviones, barcos)
            - 🏠 Objetos cotidianos (muebles, instrumentos)
            - 🌿 Naturaleza (plantas, paisajes)
            - 🍎 Alimentos
            """)

        # Información adicional del modelo
        st.markdown("---")
        st.markdown("### 🤖 Información del Modelo")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Modelo Usado", model_name.upper())

        with col2:
            if model_name == 'resnet18':
                st.metric("Parámetros", "~11M")
            else:  # alexnet
                st.metric("Parámetros", "~61M")

        with col3:
            st.metric("Dataset", "ImageNet")

        # Nota sobre limitaciones
        with st.expander("⚠️ Limitaciones del modelo"):
            st.markdown("""
            **Ten en cuenta:**
            
            1. **Sesgos del dataset**: El modelo puede tener mejor desempeño 
               en categorías más representadas en ImageNet.
            
            2. **Clases específicas**: Si tu imagen no pertenece a ninguna 
               de las 1,000 clases, la predicción será la más cercana.
            
            3. **Confianza baja**: Una confianza <50% puede indicar que la 
               imagen no encaja bien en ninguna categoría.
            
            4. **Contexto importa**: El fondo y otros elementos pueden 
               influir en la predicción.
            """)

        # Enlace a documentación
        st.markdown("---")
        st.info(
            "📚 **Más información**: Para ver la lista completa de las 1,000 clases de ImageNet, "
            "visita [ImageNet Classes](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)"
        )

    # ===================================================================
    # TAB 6: EXPERIMENTOS DE ABLACIÓN
    # ===================================================================
    with tab6:
        st.markdown("## 🧪 Experimentos de Ablación")
        st.markdown(
            "Descubre qué tan importantes son las neuronas seleccionadas para la predicción del modelo. "
            "Ejecuta experimentos de **knockout**, **aislamiento**, **amplificación** y **ruido** para entender su rol."
        )

        # Importar funciones necesarias
        from utils_streamlit import run_ablation_experiment, get_imagenet_class_name
        from filter_visualization import (
            extract_filter_weights_rgb,
        )

        # ===================================================================
        # SECCIÓN 1: CONFIGURACIÓN Y BASELINE
        # ===================================================================
        st.markdown("---")
        st.markdown("### 🎯 Configuración del Experimento")

        # Baseline en columnas compactas
        col_base1, col_base2, col_base3 = st.columns(3)
        baseline_class = get_imagenet_class_name(results['prediction'])

        with col_base1:
            st.metric("📊 Predicción Original", baseline_class)
        with col_base2:
            st.metric("🎯 Confianza", f"{results['confidence']:.1%}")
        with col_base3:
            st.metric("🔢 Clase ID", f"#{results['prediction']}")

        st.markdown("")

        # ═══════════════════════════════════════════════════════════════
        # FORMULARIO: Evita recargas al cambiar configuración
        # ═══════════════════════════════════════════════════════════════
        with st.form("ablation_config_form"):
            st.markdown("#### ⚙️ Configuración de Experimentos")

            # Selector de neuronas y parámetros
            col_selector, col_params = st.columns([2, 1])

            with col_selector:
                # Crear opciones con información útil
                neuron_options = {
                    neuron_idx: f"Filtro {neuron_idx} (#{results['top_neurons'].index(neuron_idx) + 1}) - "
                    f"Media: {results['stats'][neuron_idx]['mean']:.2f}, "
                    f"Sparsity: {results['stats'][neuron_idx]['sparsity']*100:.0f}%"
                    for neuron_idx in results['top_neurons']
                }

                selected_neurons = st.multiselect(
                    "🎯 Selecciona neurona(s) para experimentar:",
                    options=list(neuron_options.keys()),
                    format_func=lambda x: neuron_options[x],
                    default=[results['top_neurons'][0]],
                    help="Selecciona una o múltiples neuronas para analizar su importancia"
                )

            with col_params:
                amp_factor = st.slider(
                    "⚡ Factor de amplificación:",
                    min_value=2.0,
                    max_value=10.0,
                    value=5.0,
                    step=1.0,
                    help="Factor de multiplicación para el experimento de amplificación"
                )

                noise_level = st.slider(
                    "🌫️ Nivel de ruido:",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    help="Intensidad del ruido gaussiano"
                )

            # Botón único: aplica Y ejecuta
            st.markdown("")

            # Calcular total de experimentos
            if selected_neurons:
                total_exp = len(selected_neurons) * 4 + \
                    (4 if len(selected_neurons) > 1 else 0)
                button_text = f"🧪 Ejecutar los Experimentos"
                button_help = f"Ejecutará {len(selected_neurons) * 4} individuales"
                if len(selected_neurons) > 1:
                    button_help += f" + 4 grupales"
            else:
                button_text = "🧪 Ejecutar Experimentos"
                button_help = "Selecciona al menos una neurona primero"

            execute_button = st.form_submit_button(
                button_text,
                type="primary",
                use_container_width=True,
                help=button_help
            )

        # ═══════════════════════════════════════════════════════════════
        # VALIDACIÓN FUERA DEL FORM
        # ═══════════════════════════════════════════════════════════════
        if not selected_neurons:
            st.warning("⚠️ Selecciona al menos una neurona en el formulario")
            st.stop()

        # Mostrar info de tipos de experimentos
        with st.expander("ℹ️ ¿Qué hace cada experimento?", expanded=False):
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.markdown("**🔴 Knockout**")
                st.caption("Apaga las neuronas para ver si son críticas")
            with col_info2:
                st.markdown("**🟢 Aislamiento**")
                st.caption("Activa SOLO estas neuronas")
            with col_info3:
                st.markdown("**⚡ Amplificación**")
                st.caption("Multiplica activaciones")
            with col_info4:
                st.markdown("**🌫️ Ruido**")
                st.caption("Agrega ruido para probar robustez")

        st.markdown("---")

        # ===================================================================
        # EJECUCIÓN DE EXPERIMENTOS
        # ===================================================================
        if execute_button:
            # Estructura para almacenar resultados
            experiments_results = {
                'individual': {},
                'group': {}
            }

            # Barra de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Calcular total de experimentos
            individual_experiments = len(selected_neurons) * 4
            group_experiments = 4 if len(selected_neurons) > 1 else 0
            total_experiments = individual_experiments + group_experiments

            current_step = 0

            # ---------------------------------------------------------------
            # EXPERIMENTOS INDIVIDUALES
            # ---------------------------------------------------------------
            for neuron_idx in selected_neurons:
                experiments_results['individual'][neuron_idx] = {}

                # Knockout individual
                status_text.text(f"🔴 Knockout - Filtro {neuron_idx}...")
                result_ko = run_ablation_experiment(
                    model=model,
                    image=results['image_tensor'],
                    target_layer=results['layer_name'],
                    neuron_idx=neuron_idx,
                    experiment_type='knockout',
                    device=device
                )
                experiments_results['individual'][neuron_idx]['knockout'] = result_ko
                current_step += 1
                progress_bar.progress(current_step / total_experiments)

                # Isolation individual
                status_text.text(f"🟢 Aislamiento - Filtro {neuron_idx}...")
                result_iso = run_ablation_experiment(
                    model=model,
                    image=results['image_tensor'],
                    target_layer=results['layer_name'],
                    neuron_idx=neuron_idx,
                    experiment_type='isolation',
                    device=device
                )
                experiments_results['individual'][neuron_idx]['isolation'] = result_iso
                current_step += 1
                progress_bar.progress(current_step / total_experiments)

                # Amplification individual
                status_text.text(f"⚡ Amplificación - Filtro {neuron_idx}...")
                result_amp = run_ablation_experiment(
                    model=model,
                    image=results['image_tensor'],
                    target_layer=results['layer_name'],
                    neuron_idx=neuron_idx,
                    experiment_type='amplify',
                    amplification_factor=amp_factor,
                    device=device
                )
                experiments_results['individual'][neuron_idx]['amplify'] = result_amp
                current_step += 1
                progress_bar.progress(current_step / total_experiments)

                # Experimento 4: Noise individual
                status_text.text(f"🌫️ Ruido - Filtro {neuron_idx}...")
                result_noise = run_ablation_experiment(
                    model=model,
                    image=results['image_tensor'],
                    target_layer=results['layer_name'],
                    neuron_idx=neuron_idx,
                    experiment_type='add_noise',
                    noise_level=noise_level,
                    device=device
                )
                experiments_results['individual'][neuron_idx]['noise'] = result_noise
                current_step += 1
                progress_bar.progress(current_step / total_experiments)

            # ---------------------------------------------------------------
            # EXPERIMENTOS GRUPALES (si hay múltiples neuronas)
            # ---------------------------------------------------------------
            if len(selected_neurons) > 1:
                # Group Knockout
                status_text.text(
                    f"🔴 Knockout grupal - {len(selected_neurons)} neuronas...")
                result_group_ko = run_ablation_experiment(
                    model=model,
                    image=results['image_tensor'],
                    target_layer=results['layer_name'],
                    neuron_idx=selected_neurons,
                    experiment_type='group_knockout',
                    device=device
                )
                experiments_results['group']['knockout'] = result_group_ko
                current_step += 1
                progress_bar.progress(current_step / total_experiments)

                # Group Isolation
                status_text.text(
                    f"🟢 Aislamiento grupal - {len(selected_neurons)} neuronas...")
                result_group_iso = run_ablation_experiment(
                    model=model,
                    image=results['image_tensor'],
                    target_layer=results['layer_name'],
                    neuron_idx=selected_neurons,
                    experiment_type='group_isolation',
                    device=device
                )
                experiments_results['group']['isolation'] = result_group_iso
                current_step += 1
                progress_bar.progress(current_step / total_experiments)

                # Group Amplification
                status_text.text(
                    f"⚡ Amplificación grupal - {len(selected_neurons)} neuronas...")
                result_group_amp = run_ablation_experiment(
                    model=model,
                    image=results['image_tensor'],
                    target_layer=results['layer_name'],
                    neuron_idx=selected_neurons,
                    experiment_type='group_amplify',
                    amplification_factor=amp_factor,
                    device=device
                )
                experiments_results['group']['amplify'] = result_group_amp
                current_step += 1
                progress_bar.progress(current_step / total_experiments)

                # Experimento grupal 4: Ruido del grupo completo
                status_text.text(
                    f"🌫️ Ruido grupal - {len(selected_neurons)} neuronas...")
                result_group_noise = run_ablation_experiment(
                    model=model,
                    image=results['image_tensor'],
                    target_layer=results['layer_name'],
                    neuron_idx=selected_neurons,  # Lista completa
                    experiment_type='group_noise',
                    noise_level=noise_level,
                    device=device
                )
                experiments_results['group']['noise'] = result_group_noise
                current_step += 1
                progress_bar.progress(current_step / total_experiments)

            # Limpiar y guardar
            progress_bar.empty()
            status_text.empty()

            st.session_state['ablation_results'] = experiments_results
            st.session_state['ablation_neurons'] = selected_neurons
            st.session_state['ablation_amp_factor'] = amp_factor
            st.session_state['ablation_noise_level'] = noise_level

            st.success(
                f"✅ Completado: {len(selected_neurons)*4} individuales" +
                (f" + 4 grupales" if len(selected_neurons) > 1 else "")
            )

        # ===================================================================
        # MOSTRAR RESULTADOS (si existen)
        # ===================================================================
        if 'ablation_results' in st.session_state:

            results_data = st.session_state['ablation_results']
            tested_neurons = st.session_state['ablation_neurons']
            amp_factor_used = st.session_state['ablation_amp_factor']
            noise_level_used = st.session_state['ablation_noise_level']

            st.markdown("---")
            st.markdown("## 📊 Resultados de los Experimentos")

            # ===============================================================
            # SECCIÓN 2: COMPARACIÓN VISUAL
            # ===============================================================
            st.markdown("### 🔬 Comparación Visual: Original vs Ruido")

            st.info(
                f"Comparación visual de activaciones originales vs con ruido gaussiano. "
                f"Nivel de ruido: ±{noise_level_used}"
            )

            # Crear figura con 2 filas: Original (arriba) y Con Ruido (abajo)
            num_neurons = len(tested_neurons)
            max_cols = min(4, num_neurons)

            fig, axes = plt.subplots(
                2, max_cols,
                figsize=(3 * max_cols, 6)  # ← Reducido de 4x8 a 3x6
            )

            # Asegurar que axes sea 2D
            if max_cols == 1:
                axes = axes.reshape(2, 1)

            # FILA 1: ACTIVACIONES ORIGINALES
            for idx, neuron_idx in enumerate(tested_neurons[:max_cols]):
                ax = axes[0, idx]

                # Obtener activación original
                act_map = results['activations'][0,
                                                 neuron_idx, :, :].cpu().numpy()

                # Normalizar activación
                if act_map.max() > act_map.min():
                    act_norm = (act_map - act_map.min()) / \
                        (act_map.max() - act_map.min())
                else:
                    act_norm = act_map

                # Mostrar heatmap
                im = ax.imshow(
                    act_norm,
                    cmap='hot',  # ← Sin variable, valor fijo
                    alpha=1.0
                )

                ax.set_title(
                    f"📊 Original - Filtro {neuron_idx}",
                    fontsize=12,
                    fontweight='bold'
                )
                ax.axis('off')

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax.text(
                    0.5, -0.1,
                    f"Max: {act_map.max():.2f} | Mean: {act_map.mean():.2f}",
                    transform=ax.transAxes,
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

            # FILA 2: ACTIVACIONES CON RUIDO
            for idx, neuron_idx in enumerate(tested_neurons[:max_cols]):
                ax = axes[1, idx]

                # Obtener activación original
                act_map = results['activations'][0,
                                                 neuron_idx, :, :].cpu().numpy()

                # Agregar ruido gaussiano
                act_std = act_map.std()
                noise = np.random.randn(
                    *act_map.shape) * act_std * noise_level_used
                act_map_noisy = act_map + noise

                # Normalizar activación con ruido
                if act_map_noisy.max() > act_map_noisy.min():
                    act_norm = (act_map_noisy - act_map_noisy.min()) / \
                        (act_map_noisy.max() - act_map_noisy.min())
                else:
                    act_norm = act_map_noisy

                # Mostrar heatmap
                im = ax.imshow(
                    act_norm,
                    cmap='hot',  # ← Sin variable, valor fijo
                    alpha=1.0
                )

                ax.set_title(
                    f"🌫️ Con Ruido (±{noise_level_used}) - Filtro {neuron_idx}",
                    fontsize=12,
                    fontweight='bold'
                )
                ax.axis('off')

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax.text(
                    0.5, -0.1,
                    f"Max: {act_map_noisy.max():.2f} | Ruido: ±{act_std * noise_level_used:.2f}",
                    transform=ax.transAxes,
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Mostrar resultado del experimento de ruido
            st.markdown("---")
            st.markdown("#### 🎯 Resultado del Experimento de Ruido")

            # Obtener resultado de ruido
            if len(tested_neurons) > 1:
                if 'group' in results_data and 'noise' in results_data['group']:
                    noise_result = results_data['group']['noise']
                    result_type = "Resultado grupal"
                else:
                    noise_result = None
            else:
                noise_result = results_data['individual'][tested_neurons[0]]['noise']
                result_type = "Resultado individual"

            if noise_result:
                noise_class = get_imagenet_class_name(
                    noise_result['prediction'])
                noise_change = noise_result['confidence'] - \
                    results['confidence']

                col_noise1, col_noise2, col_noise3 = st.columns(3)

                with col_noise1:
                    st.metric(
                        "Predicción Original",
                        baseline_class,
                        help="Predicción sin ruido"
                    )
                    st.caption(f"Confianza: {results['confidence']:.1%}")

                with col_noise2:
                    st.metric(
                        "Predicción con Ruido",
                        noise_class,
                        help="Predicción con ruido en activaciones"
                    )
                    st.caption(f"Confianza: {noise_result['confidence']:.1%}")

                with col_noise3:
                    st.metric(
                        "Cambio en Confianza",
                        f"{noise_result['confidence']:.1%}",
                        delta=f"{noise_change:+.1%}"
                    )

                    # Clasificación de robustez
                    if abs(noise_change) < 0.05:
                        st.success("🟢 **Muy Robusta**")
                    elif abs(noise_change) < 0.10:
                        st.info("🟡 **Robusta**")
                    elif abs(noise_change) < 0.15:
                        st.warning("🟠 **Sensible**")
                    else:
                        st.error("🔴 **Muy Sensible**")

                # Interpretación detallada
                st.markdown("")
                if noise_result['prediction'] == results['prediction']:
                    st.success(
                        f"✅ **Mantiene la clase:** Aunque se agregó ruido (±{noise_level_used}), "
                        f"el modelo sigue prediciendo '{noise_class}'. "
                        f"{'Las neuronas son robustas.' if abs(noise_change) < 0.10 else 'Hay cierta sensibilidad al ruido.'}"
                    )
                else:
                    st.error(
                        f"❌ **Cambió de clase:** El ruido (±{noise_level_used}) causó que la predicción "
                        f"cambiara de '{baseline_class}' a '{noise_class}'. "
                        "Las neuronas son muy sensibles a perturbaciones."
                    )
            # ---------------------------------------------------------------
            # VIZ TAB 2: Mapas Individuales (Grid)
            # ---------------------------------------------------------------
            # with viz_tab2:
            #     st.info(
            #         "Vista completa de los mapas de activación de todas las neuronas seleccionadas"
            #     )

            #     # Colormap selector
            #     im = ax.imshow(
            #         act_norm,
            #         cmap=viz_cmap_grid,  # ← Usa el selector
            #         alpha=1.0             # ← Sin transparencia
            #     )

            #     # Crear grid de activaciones
            #     from utils_streamlit import create_filter_grid

            #     fig_grid = create_filter_grid(
            #         activations=results['activations'],
            #         neuron_indices=tested_neurons,
            #         image_vis=results['img_vis'],
            #         max_cols=min(4, len(tested_neurons)),
            #         cmap=viz_cmap_grid
            #     )
            #     st.pyplot(fig_grid)
            #     plt.close(fig_grid)

            # ===============================================================
            # SECCIÓN 3: RESULTADOS DE EXPERIMENTOS
            # ===============================================================
            st.markdown("---")
            st.markdown("### 📈 Resultados de Experimentos")

            # Tabs internos para resultados
            if len(tested_neurons) > 1:
                results_tab1, results_tab2, results_tab3 = st.tabs([
                    "👥 Resultados Grupales",
                    "🔢 Resultados Individuales",
                    "📊 Comparación Detallada"
                ])
            else:
                # Si solo hay 1 neurona, no mostrar tab grupal
                results_tab2, results_tab3 = st.tabs([
                    "🔢 Resultados Individuales",
                    "📊 Comparación Detallada"
                ])
                results_tab1 = None

            # ---------------------------------------------------------------
            # RESULTS TAB 1: Resultados Grupales
            # ---------------------------------------------------------------
            if results_tab1 is not None:
                with results_tab1:
                    st.info(
                        f"📌 Analizando {len(tested_neurons)} neuronas como grupo: {tested_neurons}")

                    # Obtener resultados grupales
                    group_ko = results_data['group']['knockout']
                    group_iso = results_data['group']['isolation']
                    group_amp = results_data['group']['amplify']
                    group_noise = results_data['group']['noise']

                    # Tabla resumen de 4 experimentos
                    import pandas as pd

                    summary_data = {
                        'Experimento': ['🔵 Original', '🟢 Aislamiento', '⚡ Amplificación', '🌫️ Ruido', '🔴 Knockout'],
                        'Predicción': [
                            baseline_class,
                            get_imagenet_class_name(group_iso['prediction']),
                            get_imagenet_class_name(group_amp['prediction']),
                            get_imagenet_class_name(group_noise['prediction']),
                            get_imagenet_class_name(group_ko['prediction'])
                        ],
                        'Confianza': [
                            f"{results['confidence']:.1%}",
                            f"{group_iso['confidence']:.1%}",
                            f"{group_amp['confidence']:.1%}",
                            f"{group_noise['confidence']:.1%}",
                            f"{group_ko['confidence']:.1%}"
                        ],
                        'Δ Cambio': [
                            '-',
                            f"{(group_iso['confidence'] - results['confidence']):+.1%}",
                            f"{(group_amp['confidence'] - results['confidence']):+.1%}",
                            f"{(group_noise['confidence'] - results['confidence']):+.1%}",
                            f"{(group_ko['confidence'] - results['confidence']):+.1%}"
                        ],
                        'Estado': [
                            '🔵 Base',
                            '✅' if group_iso['prediction'] == results['prediction'] else '❌ Cambió',
                            '✅' if group_amp['prediction'] == results['prediction'] else '❌ Cambió',
                            '✅' if group_noise['prediction'] == results['prediction'] else '❌ Cambió',
                            '✅' if group_ko['prediction'] == results['prediction'] else '❌ Cambió'
                        ]
                    }

                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(
                        df_summary, use_container_width=True, hide_index=True)

                    st.markdown("---")

                    # Análisis de sinergia
                    st.markdown("#### 🔬 Análisis de Sinergia")

                    # Calcular suma de efectos individuales
                    sum_individual_ko_changes = sum([
                        abs(results_data['individual'][n]['knockout']
                            ['confidence'] - results['confidence'])
                        for n in tested_neurons
                    ])

                    ko_change = group_ko['confidence'] - results['confidence']
                    group_ko_change = abs(ko_change)

                    col_syn1, col_syn2 = st.columns(2)
                    with col_syn1:
                        st.metric("Efecto Grupal (Knockout)",
                                  f"{group_ko_change:.1%}")
                    with col_syn2:
                        st.metric("Suma Efectos Individuales",
                                  f"{sum_individual_ko_changes:.1%}")

                    # Interpretación
                    if group_ko_change > sum_individual_ko_changes * 1.2:
                        st.success(
                            "🔗 **Sinergia positiva**: El grupo tiene más impacto que la suma de individuales"
                        )
                    elif group_ko_change < sum_individual_ko_changes * 0.5:
                        st.warning(
                            "🔀 **Redundancia**: El grupo tiene menos impacto (neuronas redundantes)"
                        )
                    else:
                        st.info(
                            "➡️ **Efecto aditivo**: El grupo suma aproximadamente los efectos individuales"
                        )

                    # Interpretación de importancia
                    st.markdown("---")
                    st.markdown("#### 💡 Importancia del Grupo")

                    if abs(ko_change) > 0.2:
                        st.error(
                            "🔥 **Grupo crítico**: Eliminarlo cambia drásticamente la predicción")
                    elif abs(ko_change) > 0.1:
                        st.warning(
                            "⚡ **Grupo importante**: Tiene impacto significativo")
                    else:
                        st.info(
                            "💤 **Grupo redundante**: El modelo compensa fácilmente")

                    if group_iso['confidence'] > 0.5:
                        st.success(
                            "🎯 **Suficientes por sí solas**: Este grupo es altamente determinante")
                    elif group_iso['confidence'] > 0.3:
                        st.info(
                            "⚖️ **Contribuyentes fuertes**: Aportan información significativa")

            # ---------------------------------------------------------------
            # RESULTS TAB 2: Resultados Individuales
            # ---------------------------------------------------------------
            with results_tab2:
                st.markdown("#### 🔍 Tabla Comparativa por Neurona")

                # Tabla comparativa
                import pandas as pd

                comparison_data = []
                for neuron_idx in tested_neurons:
                    ko_result = results_data['individual'][neuron_idx]['knockout']
                    iso_result = results_data['individual'][neuron_idx]['isolation']
                    amp_result = results_data['individual'][neuron_idx]['amplify']

                    ko_change = ko_result['confidence'] - results['confidence']
                    amp_change = amp_result['confidence'] - \
                        results['confidence']

                    noise_result = results_data['individual'][neuron_idx]['noise']
                    noise_change = noise_result['confidence'] - \
                        results['confidence']

                    comparison_data.append({
                        'Filtro': neuron_idx,
                        'Rank': f"#{tested_neurons.index(neuron_idx) + 1}",
                        'Sparsity': f"{results['stats'][neuron_idx]['sparsity']*100:.0f}%",
                        'KO': f"{ko_result['confidence']:.1%}",
                        'Δ KO': f"{ko_change:+.1%}",
                        'ISO': f"{iso_result['confidence']:.1%}",
                        'AMP': f"{amp_result['confidence']:.1%}",
                        'Δ AMP': f"{amp_change:+.1%}",
                        'NOISE': f"{noise_result['confidence']:.1%}",
                        'Δ NOISE': f"{noise_change:+.1%}"
                    })

                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(
                    df_comparison, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("#### 📋 Detalles por Neurona")

                # Expanders con detalles
                for neuron_idx in tested_neurons:
                    with st.expander(f"🔍 Filtro {neuron_idx} - Análisis Detallado", expanded=False):
                        ko_result = results_data['individual'][neuron_idx]['knockout']
                        iso_result = results_data['individual'][neuron_idx]['isolation']
                        amp_result = results_data['individual'][neuron_idx]['amplify']
                        noise_result = results_data['individual'][neuron_idx]['noise']

                        col_det1, col_det2, col_det3, col_det4 = st.columns(4)

                        # Knockout
                        with col_det1:
                            st.markdown("**🔴 Knockout**")
                            ko_change = ko_result['confidence'] - \
                                results['confidence']
                            st.metric(
                                "Confianza", f"{ko_result['confidence']:.1%}", delta=f"{ko_change:.1%}")

                            if abs(ko_change) > 0.1:
                                st.caption("⚡ Importante")
                            else:
                                st.caption("💤 Redundante")

                        # Isolation
                        with col_det2:
                            st.markdown("**🟢 Aislamiento**")
                            st.metric(
                                "Confianza", f"{iso_result['confidence']:.1%}")

                            if iso_result['confidence'] > 0.3:
                                st.caption("🎯 Determinante")
                            else:
                                st.caption("⚖️ Contribuyente")

                        # Amplification
                        with col_det3:
                            st.markdown("**⚡ Amplificación**")
                            amp_change = amp_result['confidence'] - \
                                results['confidence']
                            st.metric(
                                "Confianza", f"{amp_result['confidence']:.1%}", delta=f"{amp_change:.1%}")

                            if amp_change > 0.05:
                                st.caption("📈 Efectiva")
                            else:
                                st.caption("➡️ Sin efecto")

                        # Noise
                        with col_det4:
                            st.markdown("**🌫️ Ruido**")
                            noise_change = noise_result['confidence'] - \
                                results['confidence']
                            st.metric(
                                "Confianza", f"{noise_result['confidence']:.1%}", delta=f"{noise_change:.1%}")

                            if abs(noise_change) > 0.1:
                                st.caption("🔴 Sensible")
                            else:
                                st.caption("🟢 Robusta")

            # ---------------------------------------------------------------
            # RESULTS TAB 3: Comparación Detallada
            # ---------------------------------------------------------------
            with results_tab3:
                st.markdown(
                    "#### 📊 Resumen Completo de Todos los Experimentos")

                import pandas as pd

                # Crear tabla completa con todos los datos
                detailed_data = []

                # Agregar datos individuales
                for neuron_idx in tested_neurons:
                    ko = results_data['individual'][neuron_idx]['knockout']
                    iso = results_data['individual'][neuron_idx]['isolation']
                    amp = results_data['individual'][neuron_idx]['amplify']

                    detailed_data.append({
                        'Tipo': 'Individual',
                        'Neurona(s)': f"Filtro {neuron_idx}",
                        'Experimento': 'Knockout',
                        'Predicción': get_imagenet_class_name(ko['prediction']),
                        'Confianza': f"{ko['confidence']:.1%}",
                        'Δ': f"{(ko['confidence'] - results['confidence']):+.1%}",
                        'Estado': '✅ Mantiene' if ko['prediction'] == results['prediction'] else '❌ Cambió'
                    })
                    detailed_data.append({
                        'Tipo': 'Individual',
                        'Neurona(s)': f"Filtro {neuron_idx}",
                        'Experimento': 'Aislamiento',
                        'Predicción': get_imagenet_class_name(iso['prediction']),
                        'Confianza': f"{iso['confidence']:.1%}",
                        'Δ': f"{(iso['confidence'] - results['confidence']):+.1%}",
                        'Estado': '✅ Mantiene' if ko['prediction'] == results['prediction'] else '❌ Cambió'
                    })
                    detailed_data.append({
                        'Tipo': 'Individual',
                        'Neurona(s)': f"Filtro {neuron_idx}",
                        'Experimento': 'Amplificación',
                        'Predicción': get_imagenet_class_name(amp['prediction']),
                        'Confianza': f"{amp['confidence']:.1%}",
                        'Δ': f"{(amp['confidence'] - results['confidence']):+.1%}",
                        'Estado': '✅ Mantiene' if ko['prediction'] == results['prediction'] else '❌ Cambió'
                    })

                # Agregar datos grupales si existen
                if len(tested_neurons) > 1 and 'group' in results_data:
                    group_ko = results_data['group']['knockout']
                    group_iso = results_data['group']['isolation']
                    group_amp = results_data['group']['amplify']

                    neurons_str = f"{len(tested_neurons)} neuronas"

                    detailed_data.append({
                        'Tipo': 'Grupal',
                        'Neurona(s)': neurons_str,
                        'Experimento': 'Knockout',
                        'Predicción': get_imagenet_class_name(group_ko['prediction']),
                        'Confianza': f"{group_ko['confidence']:.1%}",
                        'Δ': f"{(group_ko['confidence'] - results['confidence']):+.1%}",
                        'Estado': '✅ Mantiene' if ko['prediction'] == results['prediction'] else '❌ Cambió'
                    })
                    detailed_data.append({
                        'Tipo': 'Grupal',
                        'Neurona(s)': neurons_str,
                        'Experimento': 'Aislamiento',
                        'Predicción': get_imagenet_class_name(group_iso['prediction']),
                        'Confianza': f"{group_iso['confidence']:.1%}",
                        'Δ': f"{(group_iso['confidence'] - results['confidence']):+.1%}",
                        'Estado': '✅ Mantiene' if ko['prediction'] == results['prediction'] else '❌ Cambió'
                    })
                    detailed_data.append({
                        'Tipo': 'Grupal',
                        'Neurona(s)': neurons_str,
                        'Experimento': 'Amplificación',
                        'Predicción': get_imagenet_class_name(group_amp['prediction']),
                        'Confianza': f"{group_amp['confidence']:.1%}",
                        'Δ': f"{(group_amp['confidence'] - results['confidence']):+.1%}",
                        'Estado': '✅ Mantiene' if ko['prediction'] == results['prediction'] else '❌ Cambió'
                    })

                df_detailed = pd.DataFrame(detailed_data)
                st.dataframe(
                    df_detailed, use_container_width=True, hide_index=True)

                # Botón de descarga
                csv = df_detailed.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar resultados (CSV)",
                    data=csv,
                    file_name="ablation_results.csv",
                    mime="text/csv"
                )

            # ===============================================================
            # SECCIÓN 4: INTERPRETACIÓN Y CONCLUSIONES
            # ===============================================================
            st.markdown("---")
            st.markdown("### 💡 Interpretación y Conclusiones")

            # Análisis automático
            col_concl1, col_concl2 = st.columns(2)

            with col_concl1:
                st.markdown("#### 🏆 Neuronas Más Importantes")

                # Encontrar neurona con mayor impacto en knockout
                max_ko_impact = 0
                most_important = None

                for neuron_idx in tested_neurons:
                    ko_change = abs(
                        results_data['individual'][neuron_idx]['knockout']['confidence'] - results['confidence'])
                    if ko_change > max_ko_impact:
                        max_ko_impact = ko_change
                        most_important = neuron_idx

                if most_important:
                    st.success(f"🥇 **Filtro {most_important}**")
                    st.caption(f"Impacto en knockout: {max_ko_impact:.1%}")

                # Lista de neuronas críticas
                critical_neurons = [
                    n for n in tested_neurons
                    if abs(results_data['individual'][n]['knockout']['confidence'] - results['confidence']) > 0.1
                ]

                if critical_neurons:
                    st.info(f"⚡ **Neuronas críticas**: {critical_neurons}")
                else:
                    st.warning("💤 Ninguna neurona individual es crítica")

            with col_concl2:
                st.markdown("#### 📋 Resumen Ejecutivo")

                # Contar tipos de neuronas
                important = sum([
                    1 for n in tested_neurons
                    if abs(results_data['individual'][n]['knockout']['confidence'] - results['confidence']) > 0.1
                ])

                determinant = sum([
                    1 for n in tested_neurons
                    if results_data['individual'][n]['isolation']['confidence'] > 0.3
                ])

                effective_amp = sum([
                    1 for n in tested_neurons
                    if (results_data['individual'][n]['amplify']['confidence'] - results['confidence']) > 0.05
                ])

                st.metric("Neuronas Importantes",
                          f"{important}/{len(tested_neurons)}")
                st.metric("Neuronas Determinantes",
                          f"{determinant}/{len(tested_neurons)}")
                st.metric("Amplificación Efectiva",
                          f"{effective_amp}/{len(tested_neurons)}")

            # Recomendación final
            st.markdown("---")

            if len(tested_neurons) > 1 and 'group' in results_data:
                group_ko_change = abs(
                    results_data['group']['knockout']['confidence'] - results['confidence'])

                if group_ko_change > 0.2:
                    st.error(
                        "🔥 **Conclusión**: Este grupo de neuronas es **CRÍTICO** para la predicción. "
                        "Eliminarlas cambia drásticamente el resultado del modelo."
                    )
                elif group_ko_change > 0.1:
                    st.warning(
                        "⚡ **Conclusión**: Este grupo de neuronas es **IMPORTANTE**. "
                        "Contribuyen significativamente a la predicción final."
                    )
                else:
                    st.success(
                        "✅ **Conclusión**: Este grupo de neuronas es **REDUNDANTE**. "
                        "El modelo puede compensar su ausencia fácilmente."
                    )
            else:
                # Conclusión individual
                if most_important and max_ko_impact > 0.1:
                    st.warning(
                        f"⚡ **Conclusión**: El Filtro {most_important} es importante "
                        f"(impacto: {max_ko_impact:.1%}), pero no crítico para la predicción."
                    )
                else:
                    st.success(
                        "✅ **Conclusión**: Las neuronas seleccionadas son redundantes. "
                        "El modelo distribuye la información en múltiples neuronas."
                    )

else:
    st.info(
        "👆 Por favor, carga una imagen y presiona 'Analizar Activaciones' para comenzar.")

# Footer
st.markdown("---")
st.caption("""
**Interpretability III** - Dashboard de análisis de activaciones CNN  
Modelos disponibles: ResNet18, AlexNet | Desarrollado con Streamlit y PyTorch
""")
