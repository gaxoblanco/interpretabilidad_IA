"""
Módulo para visualizar mapas de activación neuronal.

Este módulo permite analizar qué neuronas se activan cuando el modelo
procesa una imagen específica, mostrando la "ruta de activación" a través
de las capas de la red.

Author: interpretability_III
Date: 2025-01-15
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configurar logging
logger = logging.getLogger(__name__)


def analyze_single_image_activation(
    model: nn.Module,
    image: torch.Tensor,
    target_layers: List[str],
    threshold: float = 0.1,
    device: torch.device = torch.device('cpu'),
    hook_class=None  # Se importará dinámicamente
) -> Tuple[Dict, int]:
    """
    Analiza qué neuronas se activan para una imagen específica.

    Args:
        model: Modelo de PyTorch en modo eval
        image: Imagen tensor [C, H, W] o [1, C, H, W]
        target_layers: Lista de nombres de capas a analizar
        threshold: Umbral para considerar neurona "activa" (default: 0.1)
        device: Device donde ejecutar (CPU/GPU)
        hook_class: Clase ActivationHook (debe ser pasada para evitar imports circulares)

    Returns:
        Tuple de:
            - Diccionario con resumen de activaciones por capa
            - Clase predicha (int)

    Example:
        >>> from utils.hooks import ActivationHook
        >>> summary, pred = analyze_single_image_activation(
        ...     model, image, ['relu', 'layer4.1.relu'],
        ...     hook_class=ActivationHook
        ... )
        >>> print(f"Predicción: {pred}")
        >>> print(f"Neuronas activas en relu: {summary['relu']['num_active']}")
    """
    if hook_class is None:
        raise ValueError(
            "Debe proporcionar hook_class (ActivationHook). "
            "Import: from utils.hooks import ActivationHook"
        )

    # Asegurar que la imagen tiene batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # Crear hook temporal
    logger.info(f"Registrando hooks en {len(target_layers)} capas")
    hook = hook_class(model, target_layers=target_layers)
    hook.register_hooks()

    # Forward pass
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        pred_class = output.argmax(dim=1).item()

    logger.info(f"Predicción del modelo: clase {pred_class}")

    # Obtener activaciones
    activations = hook.get_activations()

    # Limpiar hooks
    hook.remove_hooks()

    # Analizar qué neuronas están activas
    activation_summary = {}

    for layer_name, activation in activations.items():
        # activation shape: [1, C, H, W] o [1, C]
        act_np = activation.cpu().numpy()[0]  # Remover batch dim

        # Calcular activación promedio por neurona
        if len(act_np.shape) == 3:  # Conv layer [C, H, W]
            # Promedio espacial por canal
            act_per_neuron = act_np.mean(axis=(1, 2))
        else:  # FC layer [C]
            act_per_neuron = act_np

        # Identificar neuronas activas
        active_mask = act_per_neuron > threshold
        active_indices = np.where(active_mask)[0]

        activation_summary[layer_name] = {
            'activations': act_per_neuron,
            'active_indices': active_indices,
            'num_active': len(active_indices),
            'total_neurons': len(act_per_neuron),
            'activation_rate': len(active_indices) / len(act_per_neuron) * 100,
            'max_activation': float(act_per_neuron.max()),
            'mean_activation': float(act_per_neuron.mean()),
            'mean_active_only': float(act_per_neuron[active_mask].mean()) if len(active_indices) > 0 else 0.0
        }

        logger.debug(
            f"{layer_name}: {len(active_indices)}/{len(act_per_neuron)} activas "
            f"({activation_summary[layer_name]['activation_rate']:.1f}%)"
        )

    return activation_summary, pred_class


def plot_neuron_activation_map(
    image: torch.Tensor,
    activation_summary: Dict,
    pred_class: int,
    class_names: List[str],
    true_label: Optional[int] = None,
    threshold: float = 0.1,
    figsize: Tuple[int, int] = None,
    max_neurons_display: int = 64
) -> plt.Figure:
    """
    Visualiza el mapa de activación neuronal para una imagen.

    Crea una figura con:
    - Imagen original en la parte superior
    - Para cada capa: gráfico de barras de activaciones + histograma

    Args:
        image: Imagen tensor [C, H, W] o [1, C, H, W]
        activation_summary: Diccionario retornado por analyze_single_image_activation
        pred_class: Clase predicha (índice)
        class_names: Lista de nombres de clases
        true_label: Clase real (opcional, para comparación)
        threshold: Umbral usado para marcar neuronas activas
        figsize: Tamaño de la figura (ancho, alto). Si None, se calcula automáticamente
        max_neurons_display: Máximo de neuronas a mostrar en gráficos de barras

    Returns:
        Figura de matplotlib

    Example:
        >>> fig = plot_neuron_activation_map(
        ...     image, summary, pred_class=5,
        ...     class_names=['cat', 'dog', ...],
        ...     true_label=5
        ... )
        >>> plt.show()
    """
    num_layers = len(activation_summary)

    # Calcular figsize automáticamente si no se proporciona
    if figsize is None:
        figsize = (16, max(10, num_layers * 1.5))

    fig = plt.figure(figsize=figsize)

    # Grid: imagen arriba, mapas de neuronas abajo
    gs = fig.add_gridspec(
        num_layers + 1, 2,
        height_ratios=[2] + [1] * num_layers,
        hspace=0.4,
        wspace=0.3
    )

    # ============ IMAGEN ORIGINAL ============
    ax_img = fig.add_subplot(gs[0, :])

    # Denormalizar imagen para visualización
    img_to_show = _denormalize_image(image)

    ax_img.imshow(img_to_show.permute(1, 2, 0).numpy())
    ax_img.axis('off')

    # Título con predicción y label real
    title = f"Predicción: {class_names[pred_class]}"
    if true_label is not None:
        title += f" | Real: {class_names[true_label]}"
        if pred_class == true_label:
            title += " ✅"
        else:
            title += " ❌"
    ax_img.set_title(title, fontsize=16, fontweight='bold')

    # ============ MAPAS DE ACTIVACIÓN POR CAPA ============
    layer_names = list(activation_summary.keys())

    for idx, layer_name in enumerate(layer_names):
        summary = activation_summary[layer_name]

        # Subplots para esta capa
        ax_bar = fig.add_subplot(gs[idx + 1, 0])
        ax_hist = fig.add_subplot(gs[idx + 1, 1])

        # Extraer datos
        activations = summary['activations']
        active_indices = summary['active_indices']
        total_neurons = summary['total_neurons']

        # ---- GRÁFICO DE BARRAS (izquierda) ----
        _plot_activation_bars(
            ax_bar, activations, active_indices, total_neurons,
            layer_name, summary, max_neurons_display
        )

        # ---- HISTOGRAMA (derecha) ----
        _plot_activation_histogram(
            ax_hist, activations, threshold
        )

    # Título general
    plt.suptitle(
        '🔥 Mapa de Activación Neuronal',
        fontsize=18,
        fontweight='bold',
        y=0.995
    )

    return fig


def _denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Denormaliza una imagen normalizada con ImageNet stats.

    Args:
        image: Imagen tensor [C, H, W] o [1, C, H, W]

    Returns:
        Imagen denormalizada [C, H, W] en rango [0, 1]
    """
    img = image.cpu()
    if img.dim() == 4:
        img = img[0]

    # ImageNet normalization stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    return img


def _plot_activation_bars(
    ax: plt.Axes,
    activations: np.ndarray,
    active_indices: np.ndarray,
    total_neurons: int,
    layer_name: str,
    summary: Dict,
    max_display: int
):
    """
    Dibuja gráfico de barras de activaciones.

    Args:
        ax: Axes de matplotlib
        activations: Array de activaciones por neurona
        active_indices: Índices de neuronas activas
        total_neurons: Total de neuronas en la capa
        layer_name: Nombre de la capa
        summary: Diccionario con estadísticas
        max_display: Máximo de neuronas a mostrar
    """
    # Colores: rojo para activas, gris para inactivas
    colors = ['red' if i in active_indices else 'lightgray'
              for i in range(total_neurons)]

    # Si hay muchas neuronas, mostrar solo las más activas
    if total_neurons > max_display:
        top_k = min(max_display, total_neurons)
        top_indices = np.argsort(activations)[-top_k:]

        subset_activations = activations[top_indices]
        subset_colors = [colors[i] for i in top_indices]

        ax.bar(
            range(len(subset_activations)),
            subset_activations,
            color=subset_colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        ax.set_xlabel(f'Neurona (top {top_k} de {total_neurons})', fontsize=9)
    else:
        ax.bar(
            range(total_neurons),
            activations,
            color=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        ax.set_xlabel('Neurona', fontsize=9)

    ax.set_ylabel('Activación', fontsize=9)
    ax.set_title(
        f"{layer_name}\n"
        f"Activas: {summary['num_active']}/{total_neurons} "
        f"({summary['activation_rate']:.1f}%)",
        fontsize=10,
        fontweight='bold'
    )
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)


def _plot_activation_histogram(
    ax: plt.Axes,
    activations: np.ndarray,
    threshold: float
):
    """
    Dibuja histograma de distribución de activaciones.

    Args:
        ax: Axes de matplotlib
        activations: Array de activaciones
        threshold: Umbral de activación
    """
    # Solo valores > 0 (excluir sparsity)
    active_values = activations[activations > 0]

    if len(active_values) > 0:
        ax.hist(
            active_values,
            bins=30,
            color='steelblue',
            alpha=0.7,
            edgecolor='black'
        )

        # Línea de threshold
        ax.axvline(
            x=threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Threshold ({threshold})'
        )
        ax.legend(fontsize=8)

    ax.set_xlabel('Valor de Activación', fontsize=9)
    ax.set_ylabel('Frecuencia', fontsize=9)
    ax.set_title(
        'Distribución de Activaciones\n(solo valores > 0)', fontsize=10)
    ax.grid(alpha=0.3)


def print_activation_statistics(
    activation_summary: Dict,
    top_k: int = 5
):
    """
    Imprime estadísticas detalladas de activaciones.

    Args:
        activation_summary: Diccionario retornado por analyze_single_image_activation
        top_k: Número de neuronas top a mostrar por capa

    Example:
        >>> print_activation_statistics(summary, top_k=5)
    """
    print("=" * 80)
    print("📊 ESTADÍSTICAS DE ACTIVACIÓN POR CAPA:")
    print("=" * 80)

    for layer_name, summary in activation_summary.items():
        print(f"\n{layer_name}:")
        print(f"   Total de neuronas:    {summary['total_neurons']}")
        print(
            f"   Neuronas activas:     {summary['num_active']} ({summary['activation_rate']:.1f}%)")
        print(f"   Activación máxima:    {summary['max_activation']:.3f}")
        print(f"   Activación promedio:  {summary['mean_activation']:.3f}")

        if summary['num_active'] > 0:
            print(
                f"   Promedio (solo activas): {summary['mean_active_only']:.3f}")

            # Top K neuronas más activas
            top_indices = np.argsort(summary['activations'])[-top_k:][::-1]
            print(f"   Top {top_k} neuronas más activas:")

            for rank, neuron_idx in enumerate(top_indices, 1):
                activation_value = summary['activations'][neuron_idx]
                print(
                    f"      {rank}. Neurona #{neuron_idx:3d}: {activation_value:.3f}")
        else:
            print(f"   ⚠️  Ninguna neurona activa (threshold muy alto?)")

    print("\n" + "=" * 80)


def compare_activations(
    summary1: Dict,
    summary2: Dict,
    label1: str = "Imagen 1",
    label2: str = "Imagen 2"
):
    """
    Compara activaciones entre dos imágenes.

    Args:
        summary1: Resumen de activaciones de imagen 1
        summary2: Resumen de activaciones de imagen 2
        label1: Etiqueta para imagen 1
        label2: Etiqueta para imagen 2

    Example:
        >>> compare_activations(dog_summary, cat_summary, "Perro", "Gato")
    """
    print("=" * 80)
    print(f"🔍 COMPARACIÓN DE ACTIVACIONES: {label1} vs {label2}")
    print("=" * 80)

    common_layers = set(summary1.keys()) & set(summary2.keys())

    for layer in sorted(common_layers):
        s1 = summary1[layer]
        s2 = summary2[layer]

        active1 = set(s1['active_indices'])
        active2 = set(s2['active_indices'])

        common = active1 & active2
        only1 = active1 - active2
        only2 = active2 - active1

        print(f"\n{layer}:")
        print(f"   {label1}: {len(active1)} activas")
        print(f"   {label2}: {len(active2)} activas")
        print(
            f"   Comunes: {len(common)} ({len(common)/max(len(active1), len(active2))*100:.1f}%)")
        print(f"   Solo {label1}: {len(only1)}")
        print(f"   Solo {label2}: {len(only2)}")

        # Jaccard similarity
        if len(active1) > 0 or len(active2) > 0:
            union = active1 | active2
            jaccard = len(common) / len(union)
            print(f"   Similitud Jaccard: {jaccard:.3f}")

    print("\n" + "=" * 80)


# ==============================================================================
# Funciones auxiliares de análisis
# ==============================================================================

def find_specialized_neurons(
    activation_summaries: List[Dict],
    layer_name: str,
    min_activation_rate: float = 0.8
) -> np.ndarray:
    """
    Encuentra neuronas que se activan consistentemente en múltiples imágenes.

    Args:
        activation_summaries: Lista de resúmenes de activación
        layer_name: Nombre de la capa a analizar
        min_activation_rate: Tasa mínima de activación (0-1)

    Returns:
        Array con índices de neuronas especializadas

    Example:
        >>> # Analizar 10 imágenes de perros
        >>> summaries = [analyze_single_image_activation(...)[0] for _ in range(10)]
        >>> specialized = find_specialized_neurons(summaries, 'layer4.1.relu', 0.8)
        >>> print(f"Neuronas especializadas en perros: {specialized}")
    """
    if not activation_summaries:
        return np.array([])

    # Obtener todas las activaciones para esta capa
    all_active_sets = []
    for summary in activation_summaries:
        if layer_name in summary:
            all_active_sets.append(set(summary[layer_name]['active_indices']))

    if not all_active_sets:
        logger.warning(f"Capa {layer_name} no encontrada en los resúmenes")
        return np.array([])

    # Contar cuántas veces se activa cada neurona
    total_neurons = activation_summaries[0][layer_name]['total_neurons']
    activation_counts = np.zeros(total_neurons)

    for active_set in all_active_sets:
        for neuron_idx in active_set:
            activation_counts[neuron_idx] += 1

    # Calcular tasa de activación
    activation_rates = activation_counts / len(activation_summaries)

    # Encontrar neuronas que se activan frecuentemente
    specialized_indices = np.where(activation_rates >= min_activation_rate)[0]

    logger.info(
        f"Encontradas {len(specialized_indices)} neuronas especializadas "
        f"(tasa >= {min_activation_rate}) en {layer_name}"
    )

    return specialized_indices
