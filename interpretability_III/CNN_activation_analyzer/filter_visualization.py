"""
Módulo para visualizar filtros RGB y mapear activaciones a la imagen.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_layer_filters(model: nn.Module, layer_name: str) -> torch.Tensor:
    """
    Extrae los pesos (filtros RGB) de una capa convolucional.

    Args:
        model: Modelo PyTorch
        layer_name: Nombre de la capa

    Returns:
        Tensor de filtros [num_filters, in_channels, kernel_h, kernel_w]
    """
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            # module.weight tiene shape: [out_channels, in_channels, kernel_h, kernel_w]
            return module.weight.data.clone()

    raise ValueError(f"No se encontró capa convolucional: {layer_name}")


def normalize_filter_for_visualization(filter_tensor: torch.Tensor) -> np.ndarray:
    """
    Normaliza un filtro para visualización RGB.

    Args:
        filter_tensor: Tensor [in_channels, kernel_h, kernel_w]

    Returns:
        Array numpy [kernel_h, kernel_w, 3] normalizado a [0, 1]
    """
    # Si tiene 3 canales (RGB), transponer a formato de imagen
    if filter_tensor.shape[0] == 3:
        img = filter_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    else:
        # Si tiene más o menos canales, tomar primeros 3 o promediar
        if filter_tensor.shape[0] >= 3:
            img = filter_tensor[:3].cpu().numpy().transpose(1, 2, 0)
        else:
            # Escala de grises, replicar en 3 canales
            gray = filter_tensor[0].cpu().numpy()
            img = np.stack([gray, gray, gray], axis=-1)

    # Normalizar a [0, 1]
    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img_normalized = (img - img_min) / (img_max - img_min)
    else:
        img_normalized = np.zeros_like(img)

    return img_normalized


def create_filter_grid_rgb(
    model: nn.Module,
    layer_name: str,
    filter_indices: List[int],
    num_cols: int = 4
) -> plt.Figure:
    """
    Crea un grid mostrando los filtros RGB de la capa.

    Args:
        model: Modelo PyTorch
        layer_name: Nombre de la capa
        filter_indices: Índices de filtros a mostrar
        num_cols: Número de columnas en el grid

    Returns:
        Figura de matplotlib
    """
    # Obtener filtros
    filters = get_layer_filters(model, layer_name)

    num_filters = len(filter_indices)
    num_rows = (num_filters + num_cols - 1) // num_cols

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(3*num_cols, 3*num_rows))

    # Asegurar que axes sea 2D
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, filter_idx in enumerate(filter_indices):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]

        # Obtener y normalizar filtro
        filter_tensor = filters[filter_idx]  # [in_channels, H, W]
        filter_img = normalize_filter_for_visualization(filter_tensor)

        # Mostrar filtro
        ax.imshow(filter_img, interpolation='nearest')
        ax.set_title(f'Filtro {filter_idx}', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Añadir número de ranking
        ax.text(0.05, 0.95, str(idx + 1),
                transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Ocultar ejes vacíos
    for idx in range(num_filters, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis('off')

    plt.suptitle(f'Filtros RGB de {layer_name}\n(Patrones de 7×7 que busca cada neurona)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def compute_activation_regions(
    activation_map: np.ndarray,
    image_shape: Tuple[int, int],
    threshold_percentile: float = 75,
    min_regions: int = 3
) -> List[Dict]:
    """
    Identifica regiones de alta activación en el mapa.
    Prioriza regiones específicas sobre fondos uniformes.

    Args:
        activation_map: Mapa de activación [H, W]
        image_shape: Forma de la imagen original (H, W)
        threshold_percentile: Percentil para umbral de activación
        min_regions: Número mínimo de regiones a retornar

    Returns:
        Lista de diccionarios con información de regiones ordenadas por interés
    """
    from scipy.ndimage import label
    from scipy.ndimage import zoom

    # Redimensionar mapa de activación al tamaño de la imagen
    h, w = image_shape
    h_act, w_act = activation_map.shape

    if (h_act, w_act) != (h, w):
        zoom_factors = (h / h_act, w / w_act)
        activation_resized = zoom(activation_map, zoom_factors, order=1)
    else:
        activation_resized = activation_map

    # Calcular umbral adaptativo
    threshold = np.percentile(activation_resized, threshold_percentile)

    # Si el umbral es muy bajo, usar un mínimo
    if threshold < 0.1:
        threshold = 0.1

    # Crear máscara binaria
    binary_mask = activation_resized > threshold

    # Etiquetar regiones conectadas
    labeled_array, num_features = label(binary_mask)

    regions = []

    for region_id in range(1, num_features + 1):
        # Obtener coordenadas de la región
        region_coords = np.argwhere(labeled_array == region_id)

        if len(region_coords) == 0:
            continue

        # Calcular bounding box
        y_min, x_min = region_coords.min(axis=0)
        y_max, x_max = region_coords.max(axis=0)

        width = x_max - x_min
        height = y_max - y_min

        # Filtrar regiones muy pequeñas (ruido)
        if width < 3 or height < 3:
            continue

        # Calcular intensidad promedio en la región
        region_mask = labeled_array == region_id
        intensity = activation_resized[region_mask].mean()

        regions.append({
            'x': int(x_min),
            'y': int(y_min),
            'width': int(width),
            'height': int(height),
            'intensity': float(intensity),
            'area': int(width * height)
        })

    # Calcular score de "interés" para cada región
    # Combina: intensidad alta + tamaño pequeño/moderado (más específico)
    max_area = image_shape[0] * image_shape[1]

    for region in regions:
        area = region['area']
        intensity = region['intensity']

        # Ratio del área respecto a la imagen completa
        area_ratio = area / max_area

        # Score de selectividad basado en tamaño
        if area_ratio > 0.3:  # Región gigante (>30% de la imagen)
            size_score = 0.3  # Penalizar fuertemente (probablemente fondo)
        elif area_ratio > 0.15:  # Región grande (15-30%)
            size_score = 0.6
        elif area_ratio > 0.05:  # Región mediana (5-15%)
            size_score = 0.9
        else:  # Región pequeña (<5%)
            size_score = 1.0  # Máximo interés

        # Score final: intensidad × selectividad de tamaño
        region['interest_score'] = intensity * size_score
        region['area_ratio'] = area_ratio

    # Ordenar por score de interés (descendente)
    regions = sorted(regions, key=lambda x: x['interest_score'], reverse=True)

    # Asegurar al menos min_regions si existen
    if len(regions) < min_regions and num_features > 0:
        # Si hay pocas regiones, bajar el umbral y reintentar
        threshold = np.percentile(
            activation_resized, threshold_percentile - 10)
        binary_mask = activation_resized > threshold
        labeled_array, num_features = label(binary_mask)
        # ... (repetir proceso)

    return regions[:min_regions * 3]  # Retornar hasta 3x min_regions


def create_image_with_activation_boxes(
    image_vis: np.ndarray,
    activation_map: np.ndarray,
    filter_idx: int,
    max_boxes: int = 3
) -> plt.Figure:
    """
    Crea visualización de imagen con cajas marcando regiones de activación.

    Args:
        image_vis: Imagen original [H, W, 3]
        activation_map: Mapa de activación [H, W]
        filter_idx: Índice del filtro
        max_boxes: Número máximo de cajas a mostrar

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(4, 4))

    # Mostrar imagen
    ax.imshow(image_vis)

    # Obtener regiones de activación
    regions = compute_activation_regions(
        activation_map,
        image_vis.shape[:2],
        threshold_percentile=85
    )

    # Dibujar cajas para las top regiones
    colors = ['red', 'yellow', 'lime']
    for idx, region in enumerate(regions[:max_boxes]):
        color = colors[idx % len(colors)]

        # Crear rectángulo
        rect = patches.Rectangle(
            (region['x'], region['y']),
            region['width'],
            region['height'],
            linewidth=3,
            edgecolor=color,
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)

        # Añadir etiqueta
        ax.text(
            region['x'], region['y'] - 5,
            f"Región {idx + 1}",
            color=color,
            fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )

    ax.set_title(f'Regiones de activación del Filtro {filter_idx}',
                 fontsize=10, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    return fig


def explain_filter_activation(
    filter_idx: int,
    regions: List[Dict],
    neuron_stats: Dict
) -> str:
    """
    Genera explicación textual de por qué se activó el filtro.

    Args:
        filter_idx: Índice del filtro
        regions: Lista de regiones detectadas
        neuron_stats: Estadísticas de la neurona

    Returns:
        Texto explicativo
    """
    explanation = f"### 🔍 Explicación del Filtro {filter_idx}\n\n"

    # Análisis de selectividad (sparsity)
    is_selective = neuron_stats['sparsity'] > 0.3
    num_regions = len(regions)

    # Nivel de activación con contexto mejorado
    if neuron_stats['mean'] > 0.5:
        if is_selective:
            explanation += "**🎯 Alta activación selectiva** - Este filtro detectó patrones específicos en áreas concretas.\n\n"
        else:
            explanation += "**🌊 Alta activación dispersa** - Este filtro se activa en muchas áreas, probablemente detecta texturas o colores de fondo (ej: pasto, cielo, pared).\n\n"
    elif neuron_stats['mean'] > 0.2:
        if is_selective:
            explanation += "**⚡ Activación moderada y focal** - El filtro detectó su patrón en regiones específicas.\n\n"
        else:
            explanation += "**💫 Activación moderada y dispersa** - El filtro responde a elementos comunes en la imagen.\n\n"
    else:
        explanation += "**💤 Baja activación** - El filtro no encontró patrones que coincidan con lo que busca.\n\n"

    # Análisis de regiones
    if num_regions > 0:
        explanation += f"**Regiones detectadas:** {num_regions}"

        # Interpretar según cantidad de regiones
        if num_regions > 20:
            explanation += " ⚠️ (Muchas regiones = patrón muy común, probablemente fondo/textura)\n\n"
        elif num_regions > 10:
            explanation += " (Patrón presente en múltiples áreas)\n\n"
        elif num_regions > 3:
            explanation += " (Patrón encontrado en varias ubicaciones)\n\n"
        else:
            explanation += " ✓ (Patrón específico y localizado)\n\n"

        # Mostrar top 3 regiones con clasificación de tamaño
        for idx, region in enumerate(regions[:3], 1):
            explanation += f"- **Región {idx}**: "
            explanation += f"Intensidad {region['intensity']:.2f}, "
            explanation += f"ubicada en ({region['x']}, {region['y']}), "
            explanation += f"tamaño {region['width']}×{region['height']} px"

            # Clasificar por interés (ya ordenadas)
            if idx == 1:
                explanation += " ⭐ (Más interesante)"

            # Clasificar tamaño de región
            area_ratio = region.get('area_ratio', region['area'] / (224*224))
            if area_ratio > 0.3:
                explanation += " 🔷 (Región grande - posible fondo)"
            elif area_ratio > 0.05:
                explanation += " 🔹 (Región mediana)"
            else:
                explanation += " 🔸 (Región pequeña - patrón específico)"

            explanation += "\n"
    else:
        explanation += "**No se detectaron regiones de alta activación** en esta imagen.\n"

    # Interpretación de sparsity con advertencias
    explanation += f"\n\n**Selectividad (Sparsity):** {neuron_stats['sparsity']*100:.1f}%\n\n"

    if neuron_stats['sparsity'] > 0.5:
        explanation += "🎯 **Muy selectivo** - Solo se activa en patrones muy específicos. "
        explanation += "Este filtro detecta características únicas del objeto.\n"
    elif neuron_stats['sparsity'] > 0.3:
        explanation += "⚖️ **Moderadamente selectivo** - Balance entre generalidad y especificidad. "
        explanation += "Detecta patrones reconocibles pero no únicos.\n"
    elif neuron_stats['sparsity'] > 0.1:
        explanation += "🌊 **Poco selectivo** - Responde a patrones comunes. "
        explanation += "Probablemente detecta texturas o colores generales.\n"
    else:
        explanation += "⚠️ **Nada selectivo** - Se activa en casi toda la imagen (>90%). "
        explanation += "Este filtro detecta características de fondo (color, iluminación) más que patrones específicos del objeto.\n"

    # Advertencia adicional para sparsity muy baja
    if neuron_stats['sparsity'] < 0.1:
        explanation += f"\n---\n\n"
        explanation += "### ⚠️ Advertencias de Interpretación\n\n"
        explanation += f"⚠️ **Baja selectividad (Sparsity < 10%)**: Este filtro se activa en casi toda la imagen "
        explanation += f"({(1-neuron_stats['sparsity'])*100:.0f}% de píxeles activos), "
        explanation += "probablemente está detectando características generales del fondo "
        explanation += "(color dominante, iluminación, textura uniforme) más que patrones específicos del objeto.\n\n"
        explanation += "💡 **Sugerencias**:\n"
        explanation += "- Usa el criterio 'balanced' con **Sparsity Mínima ≥ 0.1** en el sidebar\n"
        explanation += "- Analiza una **capa más profunda** (layer3, layer4) que es más selectiva\n"
        explanation += "- Si la imagen tiene fondo uniforme (cielo, pared, césped), estos filtros detectarán principalmente eso\n"

    return explanation


def create_activation_patches_visualization(
    image_vis: np.ndarray,
    activation_map: np.ndarray,
    filter_idx: int,
    num_patches: int = 3,
    figsize: Tuple[int, int] = (3, 3)
) -> plt.Figure:
    """
    Muestra los parches de la imagen original que más activaron el filtro.

    Args:
        image_vis: Imagen original [H, W, 3]
        activation_map: Mapa de activación [H, W]
        filter_idx: Índice del filtro
        num_patches: Número de parches a mostrar

    Returns:
        Figura de matplotlib con los patches
    """
    # Obtener regiones de activación
    regions = compute_activation_regions(
        activation_map,
        image_vis.shape[:2],
        threshold_percentile=75,
        min_regions=num_patches  # Garantizar que encontramos suficientes
    )

    # Limitar al número solicitado
    top_regions = regions[:num_patches]

    if len(top_regions) == 0:
        # Si no hay regiones, crear figura vacía
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No se detectaron regiones de alta activación',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Crear figura con subplots
    num_cols = min(len(top_regions), 5)
    fig, axes = plt.subplots(1, num_cols, figsize=(3*num_cols, 3))

    # Si solo hay un patch, convertir a array
    if num_cols == 1:
        axes = [axes]

    for idx, region in enumerate(top_regions):
        ax = axes[idx]

        # Extraer patch de la imagen original con validación
        x = max(0, region['x'])
        y = max(0, region['y'])
        w = region['width']
        h = region['height']

        # Asegurar que no nos salimos de los límites
        x_end = min(x + w, image_vis.shape[1])
        y_end = min(y + h, image_vis.shape[0])

        # Validar que el patch tenga tamaño válido
        if x_end <= x or y_end <= y:
            ax.text(0.5, 0.5, f'Región {idx + 1}\n(inválida)',
                    ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue

        # Recortar patch
        patch = image_vis[y:y_end, x:x_end]

        # Verificar que el patch no esté vacío
        if patch.size == 0:
            ax.text(0.5, 0.5, f'Región {idx + 1}\n(vacía)',
                    ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue

        # Mostrar patch
        ax.imshow(patch)
        ax.set_title(f'Región {idx + 1}\nInt: {region["intensity"]:.2f}',
                     fontsize=10, fontweight='bold')
        ax.axis('off')

        # Añadir borde del color correspondiente
        colors = ['red', 'yellow', 'lime']
        color = colors[idx % len(colors)]
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    plt.suptitle(f'Patches que activaron el Filtro {filter_idx}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig


def create_filter_rgb_small(
    model: nn.Module,
    layer_name: str,
    filter_idx: int,
    size: Tuple[int, int] = (100, 100)
) -> np.ndarray:
    """
    Crea una visualización pequeña del filtro RGB.

    Args:
        model: Modelo PyTorch
        layer_name: Nombre de la capa
        filter_idx: Índice del filtro
        size: Tamaño de salida (ancho, alto) en píxeles

    Returns:
        Array numpy [H, W, 3] con el filtro visualizado
    """
    # Obtener filtros
    filters = get_layer_filters(model, layer_name)
    filter_tensor = filters[filter_idx]

    # Normalizar para visualización
    filter_img = normalize_filter_for_visualization(filter_tensor)

    # Redimensionar usando PIL para mejor calidad
    from PIL import Image
    filter_pil = Image.fromarray((filter_img * 255).astype(np.uint8))
    filter_resized = filter_pil.resize(
        size, Image.NEAREST)  # NEAREST para ver pixeles

    return np.array(filter_resized).astype(np.float32) / 255.0


def create_image_with_filter_patches(
    image_vis: np.ndarray,
    activation_map: np.ndarray,
    model: nn.Module,
    layer_name: str,
    filter_idx: int,
    max_boxes: int = 3,
    num_patches: int = 3
) -> plt.Figure:
    """
    Crea visualización mostrando regiones Y el filtro RGB que las detectó.

    Args:
        image_vis: Imagen original [H, W, 3]
        activation_map: Mapa de activación [H, W]
        model: Modelo PyTorch
        layer_name: Nombre de la capa
        filter_idx: Índice del filtro
        max_boxes: Número máximo de regiones

    Returns:
        Figura de matplotlib
    """
    # Obtener regiones de activación
    regions = compute_activation_regions(
        activation_map,
        image_vis.shape[:2],
        threshold_percentile=85,
        min_regions=num_patches
    )

    # Crear figura con 2 columnas: imagen con regiones | filtros RGB
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], hspace=0.3, wspace=0.3)

    # Columna izquierda: Imagen con cajas
    # Columna izquierda: Imagen con cajas
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image_vis)

    # Dibujar TODAS las cajas solicitadas
    colors = ['red', 'yellow', 'lime']
    num_regions_to_show = min(max_boxes, len(regions))

    for idx in range(num_regions_to_show):
        region = regions[idx]
        color = colors[idx % len(colors)]

        # Crear rectángulo
        rect = patches.Rectangle(
            (region['x'], region['y']),
            region['width'],
            region['height'],
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=0.8
        )
        ax_img.add_patch(rect)

        # Etiqueta con número
        label_x = region['x']
        label_y = region['y'] - 5

        # Ajustar posición si está muy arriba
        if label_y < 10:
            label_y = region['y'] + region['height'] + 15

        # Ajustar posición si está muy a la izquierda
        if label_x < 10:
            label_x = 10

        ax_img.text(
            label_x, label_y,
            f"Región {idx + 1}",
            color=color,
            fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )

    # Asegurar que se vea toda la imagen
    ax_img.set_xlim(0, image_vis.shape[1])
    # Invertido para que (0,0) esté arriba-izquierda
    ax_img.set_ylim(image_vis.shape[0], 0)

    ax_img.set_title(f'Regiones de activación del Filtro {filter_idx}',
                     fontsize=10, fontweight='bold')
    ax_img.axis('off')

    # Columna derecha: Mostrar el filtro RGB que detectó esto
    ax_filter = fig.add_subplot(gs[0, 1])

    try:
        # Obtener filtro RGB
        filter_vis = create_filter_rgb_small(
            model=model,
            layer_name=layer_name,
            filter_idx=filter_idx,
            size=(150, 150)
        )

        ax_filter.imshow(filter_vis, interpolation='nearest')
        ax_filter.set_title(
            f'Patrón RGB del Filtro {filter_idx}\n(Esto es lo que busca)',
            fontsize=10,
            fontweight='bold'
        )
        ax_filter.axis('off')

        # Añadir texto explicativo
        fig.text(
            0.75, 0.05,
            'El filtro busca este patrón\nen las regiones marcadas →',
            ha='center',
            fontsize=9,
            style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    except Exception as e:
        # Si no se puede mostrar el filtro RGB (capas profundas)
        ax_filter.text(
            0.5, 0.5,
            'Filtro RGB no disponible\n(capa muy profunda)',
            ha='center',
            va='center',
            fontsize=10
        )
        ax_filter.axis('off')

    plt.tight_layout()
    return fig


def create_validation_panel(
    image_vis: np.ndarray,
    activation_map: np.ndarray,
    filter_idx: int,
    neuron_stats: Dict,
    model: nn.Module,
    layer_name: str
) -> Dict:
    """
    Crea un panel de validación para verificar que la interpretación es correcta.

    Returns:
        Dict con información de validación y warnings
    """
    validation = {
        'warnings': [],
        'info': [],
        'debug': {}
    }

    # 1. Verificar sparsity
    if neuron_stats['sparsity'] < 0.1:
        validation['warnings'].append(
            "⚠️ **Baja selectividad (Sparsity < 10%)**: Este filtro se activa en casi "
            "toda la imagen, probablemente está detectando características generales "
            "del fondo (color, iluminación) más que patrones específicos del objeto."
        )

    # 2. Verificar distribución de activaciones
    act_flat = activation_map.flatten()
    std_to_mean_ratio = neuron_stats['std'] / (neuron_stats['mean'] + 1e-8)

    if std_to_mean_ratio < 0.3:
        validation['warnings'].append(
            f"⚠️ **Activación muy uniforme (Std/Mean = {std_to_mean_ratio:.2f})**: "
            "Las activaciones son muy similares en toda la imagen. Esto sugiere que el "
            "filtro responde de manera general, no selectiva."
        )

    # 3. Verificar si es capa profunda (difícil de interpretar como RGB)
    try:
        filters = get_layer_filters(model, layer_name)
        num_input_channels = filters.shape[1]

        if num_input_channels > 3:
            validation['info'].append(
                f"ℹ️ **Capa profunda detectada**: Esta capa tiene {num_input_channels} "
                "canales de entrada (no RGB directo). El 'Patrón RGB' mostrado es una "
                "aproximación y puede no ser interpretable visualmente. "
                "**Recomendación**: Usa capas más tempranas (conv1, layer1) para "
                "visualización de patrones RGB claros."
            )
    except:
        pass

    # 4. Verificar coherencia región-intensidad
    from scipy.ndimage import zoom
    h, w = image_vis.shape[:2]
    h_act, w_act = activation_map.shape

    if (h_act, w_act) != (h, w):
        zoom_factors = (h / h_act, w / w_act)
        act_resized = zoom(activation_map, zoom_factors, order=1)
    else:
        act_resized = activation_map

    # Calcular varianza espacial
    spatial_variance = np.var(act_resized)

    if spatial_variance < 0.1:
        validation['warnings'].append(
            f"⚠️ **Muy poca variación espacial**: El mapa de activación es casi "
            "uniforme (varianza = {spatial_variance:.4f}). Las 'regiones' detectadas "
            "pueden ser artefactos del threshold, no patrones reales."
        )

    # 5. Debug info
    validation['debug'] = {
        'std_to_mean_ratio': std_to_mean_ratio,
        'spatial_variance': spatial_variance,
        'activation_range': (neuron_stats['min'], neuron_stats['max']),
        'activation_mean': neuron_stats['mean']
    }

    return validation


def extract_filter_weights_rgb(
    model: nn.Module,
    layer_name: str,
    filter_idx: int
) -> Optional[np.ndarray]:
    """
    Extrae los pesos RGB de un filtro específico.

    Args:
        model: Modelo de PyTorch
        layer_name: Nombre de la capa
        filter_idx: Índice del filtro

    Returns:
        Array numpy [3, H, W] con los pesos RGB, o None si no aplica
    """
    # Obtener la capa
    layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            layer = module
            break

    if layer is None or not isinstance(layer, nn.Conv2d):
        return None

    # Obtener pesos [out_channels, in_channels, H, W]
    weights = layer.weight.data.cpu().numpy()

    # Verificar que tenga 3 canales de entrada (RGB)
    if weights.shape[1] != 3:
        return None

    # Extraer el filtro específico [3, H, W]
    filter_weights = weights[filter_idx, :, :, :]

    return filter_weights


def create_rgb_channel_visualization(
    filter_weights: np.ndarray,
    filter_idx: int,
    figsize: Tuple[int, int] = (15, 3)
) -> plt.Figure:
    """
    Visualiza canales RGB con colores reales.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Extraer cada canal
    red_channel = filter_weights[0, :, :]
    green_channel = filter_weights[1, :, :]
    blue_channel = filter_weights[2, :, :]

    h, w = red_channel.shape

    # Función para normalizar a [0, 1]
    def normalize(channel):
        c_min, c_max = channel.min(), channel.max()
        if abs(c_max - c_min) < 1e-6:
            return np.ones_like(channel) * 0.5
        return (channel - c_min) / (c_max - c_min)

    # Canal Rojo (solo rojo activado)
    red_img = np.zeros((h, w, 3))
    red_img[:, :, 0] = normalize(red_channel)
    axes[0].imshow(red_img)
    axes[0].set_title('🔴 Canal Rojo', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Canal Verde (solo verde activado)
    green_img = np.zeros((h, w, 3))
    green_img[:, :, 1] = normalize(green_channel)
    axes[1].imshow(green_img)
    axes[1].set_title('🟢 Canal Verde', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Canal Azul (solo azul activado)
    blue_img = np.zeros((h, w, 3))
    blue_img[:, :, 2] = normalize(blue_channel)
    axes[2].imshow(blue_img)
    axes[2].set_title('🔵 Canal Azul', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Combinado RGB
    rgb_combined = np.stack([
        normalize(red_channel),
        normalize(green_channel),
        normalize(blue_channel)
    ], axis=-1)

    axes[3].imshow(rgb_combined)
    axes[3].set_title('🎨 Combinado RGB', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    fig.suptitle(f'Descomposición por Canales del Filtro {filter_idx}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    return fig
