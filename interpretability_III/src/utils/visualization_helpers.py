"""
Visualization Helpers - Herramientas de visualización para Feature Visualization

Funciones para:
- Crear grids de imágenes sintéticas
- Comparar imágenes reales vs sintéticas
- Visualizar curvas de convergencia
- Exportar resultados
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def plot_feature_grid(
    images: List[np.ndarray],
    neuron_indices: List[int],
    layer_name: str,
    ncols: int = 4,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
):
    """
    Visualiza grid de features generadas

    Args:
        images: Lista de imágenes [H, W, 3]
        neuron_indices: Lista de índices de neuronas
        layer_name: Nombre de la capa
        ncols: Número de columnas
        figsize: Tamaño de figura
        save_path: Ruta para guardar (opcional)
    """
    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_images > 1 else [axes]

    for i, (img, neuron_idx) in enumerate(zip(images, neuron_indices)):
        axes[i].imshow(img)
        axes[i].set_title(
            f'Neurona {neuron_idx}',
            fontsize=10,
            fontweight='bold'
        )
        axes[i].axis('off')

    # Ocultar axes vacíos
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.suptitle(
        f'Feature Visualization - {layer_name}\n'
        f'(Imágenes que maximizan activación)',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Grid guardado en: {save_path}")

    plt.show()


def plot_convergence(
    histories: List[Dict],
    neuron_indices: List[int],
    figsize: Tuple[int, int] = (16, 5)
):
    """
    Visualiza curvas de convergencia durante optimización

    Args:
        histories: Lista de historiales con pérdidas
        neuron_indices: Índices de neuronas
        figsize: Tamaño de figura
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Activación
    for history, neuron_idx in zip(histories, neuron_indices):
        axes[0].plot(
            history['activation'],
            label=f'Neurona {neuron_idx}',
            alpha=0.7
        )
    axes[0].set_title('Activación de Neurona', fontweight='bold')
    axes[0].set_xlabel('Iteración')
    axes[0].set_ylabel('Activación')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: L2 Loss
    for history, neuron_idx in zip(histories, neuron_indices):
        axes[1].plot(
            history['l2_loss'],
            label=f'Neurona {neuron_idx}',
            alpha=0.7
        )
    axes[1].set_title('Regularización L2', fontweight='bold')
    axes[1].set_xlabel('Iteración')
    axes[1].set_ylabel('L2 Loss')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Total Variation
    for history, neuron_idx in zip(histories, neuron_indices):
        axes[2].plot(
            history['tv_loss'],
            label=f'Neurona {neuron_idx}',
            alpha=0.7
        )
    axes[2].set_title('Total Variation', fontweight='bold')
    axes[2].set_xlabel('Iteración')
    axes[2].set_ylabel('TV Loss')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        'Convergencia de Feature Optimization',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.show()


def plot_real_vs_synthetic(
    comparison_results: Dict,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Compara imagen real vs sintética y sus activaciones

    Args:
        comparison_results: Resultado de compare_real_vs_synthetic()
        figsize: Tamaño de figura
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Imagen real
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(comparison_results['real_image'])
    ax1.set_title('Imagen Real', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Imagen sintética
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(comparison_results['synthetic_image'])
    ax2.set_title('Imagen Sintética Optimizada',
                  fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Diferencia
    ax3 = fig.add_subplot(gs[0, 2])
    real_gray = np.mean(comparison_results['real_image'], axis=2)
    synth_gray = np.mean(comparison_results['synthetic_image'], axis=2)
    diff = np.abs(real_gray - synth_gray)
    ax3.imshow(diff, cmap='hot')
    ax3.set_title('Diferencia Absoluta', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Comparación de activaciones
    ax4 = fig.add_subplot(gs[1, :])

    real_act = comparison_results['real_activation']
    synth_act = comparison_results['synthetic_activation']
    improvement = comparison_results['improvement']

    bars = ax4.bar(
        ['Imagen Real', 'Imagen Sintética'],
        [real_act, synth_act],
        color=['#3498db', '#e74c3c'],
        alpha=0.7,
        edgecolor='black',
        linewidth=2
    )

    ax4.set_ylabel('Activación', fontsize=11, fontweight='bold')
    ax4.set_title(
        f'Comparación de Activaciones (Mejora: {improvement:.2f}x)',
        fontsize=12,
        fontweight='bold'
    )
    ax4.grid(True, axis='y', alpha=0.3)

    # Añadir valores sobre barras
    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    plt.suptitle(
        '🔍 Real vs Sintética: ¿Qué activa más la neurona?',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    plt.show()

    # Print interpretación
    print(f"\n💡 INTERPRETACIÓN:")
    print(f"   Activación real:      {real_act:.4f}")
    print(f"   Activación sintética: {synth_act:.4f}")
    print(f"   Mejora:               {improvement:.2f}x")

    if improvement > 2:
        print(f"\n   ✅ La imagen sintética activa MUCHO MÁS la neurona")
        print(f"      → La neurona busca patrones específicos presentes en sintética")
    elif improvement > 1.2:
        print(f"\n   ✓ La imagen sintética activa más la neurona")
    else:
        print(f"\n   ⚠️  Activaciones similares")
        print(f"      → La imagen real ya contiene los patrones que busca la neurona")


def plot_layer_comparison(
    layer_results: Dict[str, List[np.ndarray]],
    neuron_idx: int,
    figsize: Tuple[int, int] = (16, 5)
):
    """
    Compara features de la misma neurona en diferentes capas

    Args:
        layer_results: {layer_name: [images]}
        neuron_idx: Índice de neurona a comparar
        figsize: Tamaño de figura
    """
    n_layers = len(layer_results)

    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    axes = axes if n_layers > 1 else [axes]

    for ax, (layer_name, images) in zip(axes, layer_results.items()):
        ax.imshow(images[neuron_idx])
        ax.set_title(
            f'{layer_name}\nNeurona {neuron_idx}',
            fontsize=10,
            fontweight='bold'
        )
        ax.axis('off')

    plt.suptitle(
        f'Comparación entre capas - Neurona {neuron_idx}',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.show()


def save_feature_collection(
    images: List[np.ndarray],
    neuron_indices: List[int],
    save_dir: str,
    layer_name: str,
    prefix: str = "feature"
):
    """
    Guarda colección de features en disco

    Args:
        images: Lista de imágenes
        neuron_indices: Índices de neuronas
        save_dir: Directorio destino
        layer_name: Nombre de capa
        prefix: Prefijo de archivos
    """
    from PIL import Image

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Guardando {len(images)} features en: {save_path}")

    for img, neuron_idx in zip(images, neuron_indices):
        filename = f"{prefix}_{layer_name}_neuron{neuron_idx:04d}.png"
        filepath = save_path / filename

        Image.fromarray(img).save(filepath)

    print(f"✅ Features guardadas")


def plot_diverse_features(
    images: List[np.ndarray],
    neuron_indices: List[int],
    stats: List[Dict],
    layer_name: str,
    figsize: Tuple[int, int] = (18, 10)
):
    """
    Visualiza features con estadísticas de diversidad

    Args:
        images: Lista de imágenes
        neuron_indices: Índices de neuronas
        stats: Estadísticas (mean, std, etc.)
        layer_name: Nombre de capa
        figsize: Tamaño de figura
    """
    n_images = len(images)
    ncols = 4
    nrows = (n_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_images > 1 else [axes]

    for i, (img, neuron_idx, stat) in enumerate(zip(images, neuron_indices, stats)):
        axes[i].imshow(img)

        # Título con stats
        title = (f'Neurona {neuron_idx}\n'
                 f'Act: {stat.get("activation", 0):.3f}')

        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')

    # Ocultar axes vacíos
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.suptitle(
        f'Feature Diversity - {layer_name}',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()
    plt.show()


def create_summary_report(
    results: Dict,
    save_path: Optional[str] = None
):
    """
    Crea reporte resumen de experimento de feature visualization

    Args:
        results: Diccionario con todos los resultados
        save_path: Ruta para guardar reporte
    """
    report = []
    report.append("=" * 70)
    report.append("FEATURE VISUALIZATION - REPORTE RESUMEN")
    report.append("=" * 70)

    report.append(f"\n📊 CONFIGURACIÓN:")
    report.append(f"   Modelo:       {results.get('model_name', 'N/A')}")
    report.append(f"   Capa:         {results.get('layer_name', 'N/A')}")
    report.append(f"   Neuronas:     {results.get('n_neurons', 'N/A')}")
    report.append(f"   Iteraciones:  {results.get('iterations', 'N/A')}")

    if 'statistics' in results:
        stats = results['statistics']
        report.append(f"\n📈 ESTADÍSTICAS:")
        report.append(
            f"   Activación media:  {stats.get('mean_activation', 0):.4f}")
        report.append(
            f"   Activación máxima: {stats.get('max_activation', 0):.4f}")
        report.append(
            f"   Activación mínima: {stats.get('min_activation', 0):.4f}")
        report.append(
            f"   Desviación std:    {stats.get('std_activation', 0):.4f}")

    report.append("\n" + "=" * 70)

    report_text = "\n".join(report)
    print(report_text)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\n💾 Reporte guardado en: {save_path}")


def plot_optimization_progress(
    history: Dict,
    neuron_idx: int,
    figsize: Tuple[int, int] = (16, 4)
):
    """
    Visualiza progreso detallado de optimización de una neurona

    Args:
        history: Historial de una neurona
        neuron_idx: Índice de neurona
        figsize: Tamaño de figura
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Activación
    axes[0].plot(history['activation'], linewidth=2, color='#2ecc71')
    axes[0].set_title('Activación', fontweight='bold')
    axes[0].set_xlabel('Iteración')
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(
        range(len(history['activation'])),
        history['activation'],
        alpha=0.3,
        color='#2ecc71'
    )

    # L2 Loss
    axes[1].plot(history['l2_loss'], linewidth=2, color='#e74c3c')
    axes[1].set_title('L2 Loss', fontweight='bold')
    axes[1].set_xlabel('Iteración')
    axes[1].grid(True, alpha=0.3)

    # TV Loss
    axes[2].plot(history['tv_loss'], linewidth=2, color='#3498db')
    axes[2].set_title('Total Variation', fontweight='bold')
    axes[2].set_xlabel('Iteración')
    axes[2].grid(True, alpha=0.3)

    # Total Loss
    axes[3].plot(history['total_loss'], linewidth=2, color='#9b59b6')
    axes[3].set_title('Total Loss', fontweight='bold')
    axes[3].set_xlabel('Iteración')
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(
        f'Progreso de Optimización - Neurona {neuron_idx}',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.show()


def find_active_neurons(model, target_layer, device, top_k=10, test_iterations=50, visualizer=None):
    """
    Prueba todas las neuronas de una capa y retorna las más activas

    Args:
        model: Modelo
        target_layer: Capa a analizar
        device: Device
        top_k: Cuántas neuronas retornar
        test_iterations: Iteraciones de prueba por neurona
        visualizer: Instancia de FeatureVisualizer (opcional)

    Returns:
        List de (neuron_idx, activacion_promedio)
    """
    print(f"🔍 Analizando capa: {target_layer}")

    # Si se pasa un visualizer, usarlo; si no, crear uno nuevo
    if visualizer is not None and visualizer.target_layer == target_layer:
        vis = visualizer  # ← Usar el existente
        cleanup_after = False
    else:
        from src.utils.feature_visualizer import FeatureVisualizer  # ← Importar la clase
        vis = FeatureVisualizer(model, target_layer, device)  # ← Crear nuevo
        cleanup_after = True

    # Obtener número de neuronas en la capa
    layer_module = dict(model.named_modules())[target_layer]
    num_neurons = layer_module.out_channels

    print(f"   Total neuronas: {num_neurons}")
    print(f"   Probando con {test_iterations} iteraciones cada una...")

    results = []

    for neuron_idx in range(num_neurons):
        # Test rápido
        _, history = vis.generate_feature(
            neuron_idx=neuron_idx,
            iterations=test_iterations,
            lr=0.1,
            verbose=False
        )

        # Guardar activación promedio
        avg_activation = np.mean(history['activation'][-10:])
        results.append((neuron_idx, avg_activation))

        # Progress
        if (neuron_idx + 1) % 10 == 0:
            print(f"   Progreso: {neuron_idx + 1}/{num_neurons}")

    # Ordenar por activación
    results.sort(key=lambda x: x[1], reverse=True)

    # Solo hacer cleanup si creamos uno nuevo
    if cleanup_after:
        vis.cleanup()

    return results[:top_k]


def extract_max_activation_region(model, target_layer, neuron_idx, image_path, device):
    """
    Extrae región de máxima activación de una imagen real

    Args:
        model: Modelo PyTorch
        target_layer: Nombre de la capa objetivo
        neuron_idx: Índice de la neurona
        image_path: Path a la imagen
        device: Device (cpu/cuda)

    Returns:
        dict con: center_x, center_y, max_value, activation_map, 
                  real_image, patch_size, map_shape
    """
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np

    # Cargar y preprocesar imagen
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    # Registrar hook para capturar activaciones
    activations = {}

    def hook_fn(module, input, output):
        activations['target'] = output.detach()

    # Encontrar capa y registrar hook
    target_module = dict(model.named_modules())[target_layer]
    hook = target_module.register_forward_hook(hook_fn)

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(img_tensor)

    # Remover hook
    hook.remove()

    # Extraer mapa de activación
    activation_map = activations['target'][0, neuron_idx].cpu().numpy()

    # Encontrar posición de máxima activación
    max_y, max_x = np.unravel_index(
        activation_map.argmax(), activation_map.shape)
    max_value = activation_map[max_y, max_x]

    # Escalar a coordenadas de imagen 224x224
    scale_h = 224 / activation_map.shape[0]
    scale_w = 224 / activation_map.shape[1]

    center_x = int(max_x * scale_w)
    center_y = int(max_y * scale_h)

    # Calcular tamaño de parche (receptive field aproximado)
    # Regla simple: más profunda la capa, mayor el receptive field
    layer_depth = int(target_layer.split(
        '.')[-1]) if 'features' in target_layer else 0
    patch_size = 32 + (layer_depth * 16)  # 32, 48, 64, 80, 96...
    patch_size = min(patch_size, 96)  # Limitar a 96

    return {
        'center_x': center_x,
        'center_y': center_y,
        'max_value': max_value,
        'activation_map': activation_map,
        'real_image': np.array(img_resized),
        'patch_size': patch_size,
        'map_shape': activation_map.shape
    }


def plot_activation_heatmap_with_roi(real_image, activation_map, center_x, center_y,
                                     patch_size, neuron_idx, layer_name):
    """
    Visualiza heatmap con ROI marcado

    Args:
        real_image: Array numpy [H, W, 3]
        activation_map: Array numpy [H_map, W_map]
        center_x, center_y: Coordenadas del centro
        patch_size: Tamaño del parche
        neuron_idx: Índice de neurona
        layer_name: Nombre de la capa
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import zoom

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Imagen original con ROI
    axes[0].imshow(real_image)

    # Calcular límites del ROI
    x1 = max(0, center_x - patch_size // 2)
    y1 = max(0, center_y - patch_size // 2)
    x2 = min(224, center_x + patch_size // 2)
    y2 = min(224, center_y + patch_size // 2)

    # Dibujar ROI
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                         fill=False, edgecolor='red', linewidth=3)
    axes[0].add_patch(rect)
    axes[0].plot(center_x, center_y, 'r*', markersize=20,
                 markeredgecolor='white', markeredgewidth=2)
    axes[0].set_title(
        f'Imagen Real\nROI en ({center_x}, {center_y})', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Heatmap superpuesto
    # Redimensionar activation_map a 224x224
    zoom_factors = (
        224 / activation_map.shape[0], 224 / activation_map.shape[1])
    heatmap_resized = zoom(activation_map, zoom_factors, order=1)

    axes[1].imshow(real_image)
    im = axes[1].imshow(heatmap_resized, cmap='jet',
                        alpha=0.5, vmin=0, vmax=heatmap_resized.max())
    axes[1].plot(center_x, center_y, 'w*', markersize=20,
                 markeredgecolor='black', markeredgewidth=2)

    # Colorbar
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Activación', fontsize=10)

    axes[1].set_title(
        f'Heatmap de Activación\nNeurona {neuron_idx}', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle(f'Localización de Activación Máxima - {layer_name}',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def overlay_synthetic_on_activation(real_image, synthetic_image, center_x, center_y, patch_size):
    """
    Superpone feature sintética en región activa

    Args:
        real_image: Array numpy [H, W, 3] (224x224)
        synthetic_image: Array numpy [H, W, 3] (224x224)
        center_x, center_y: Coordenadas del centro
        patch_size: Tamaño del parche

    Returns:
        dict con: real_full, real_patch, synthetic_resized, blend
    """
    import numpy as np
    from PIL import Image

    # Calcular límites
    x1 = max(0, center_x - patch_size // 2)
    y1 = max(0, center_y - patch_size // 2)
    x2 = min(224, center_x + patch_size // 2)
    y2 = min(224, center_y + patch_size // 2)

    # Ajustar si está en borde
    actual_w = x2 - x1
    actual_h = y2 - y1

    # Extraer parche de imagen real
    real_patch = real_image[y1:y2, x1:x2]

    # Redimensionar sintética al tamaño del parche
    synthetic_pil = Image.fromarray(synthetic_image)
    synthetic_resized = np.array(synthetic_pil.resize((actual_w, actual_h)))

    # Crear blend (50/50)
    blend = (real_patch.astype(float) * 0.5 +
             synthetic_resized.astype(float) * 0.5).astype(np.uint8)

    return {
        'real_full': real_image,
        'real_patch': real_patch,
        'synthetic_resized': synthetic_resized,
        'blend': blend,
        'roi': (x1, y1, x2, y2)
    }


def plot_overlay_comparison(real_full, real_patch, synthetic_resized, blend,
                            center_x, center_y, patch_size, neuron_idx, layer_name):
    """
    Visualiza comparación con superposición

    Args:
        real_full: Imagen completa [H, W, 3]
        real_patch: Parche real [h, w, 3]
        synthetic_resized: Sintética redimensionada [h, w, 3]
        blend: Mezcla 50/50 [h, w, 3]
        center_x, center_y: Coordenadas del centro
        patch_size: Tamaño del parche
        neuron_idx: Índice de neurona
        layer_name: Nombre de la capa
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Calcular límites para el rectángulo
    x1 = max(0, center_x - patch_size // 2)
    y1 = max(0, center_y - patch_size // 2)
    x2 = min(224, center_x + patch_size // 2)
    y2 = min(224, center_y + patch_size // 2)

    # 1. Imagen completa con ROI
    axes[0].imshow(real_full)
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                         fill=False, edgecolor='red', linewidth=3)
    axes[0].add_patch(rect)
    axes[0].plot(center_x, center_y, 'r*', markersize=20,
                 markeredgecolor='white', markeredgewidth=2)
    axes[0].set_title(f'Imagen Real Completa\nROI: {x2-x1}x{y2-y1}',
                      fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # 2. Parche real
    axes[1].imshow(real_patch)
    axes[1].set_title('Región Real\n(Zona de máxima activación)',
                      fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # 3. Feature sintética redimensionada
    axes[2].imshow(synthetic_resized)
    axes[2].set_title('Patrón Ideal (Sintética)\n(Redimensionada)',
                      fontsize=11, fontweight='bold')
    axes[2].axis('off')

    # 4. Blend
    axes[3].imshow(blend)
    axes[3].set_title('Superposición 50/50\n(Coincidencias visuales)',
                      fontsize=11, fontweight='bold')
    axes[3].axis('off')

    plt.suptitle(f'Comparación: Real vs Ideal - Neurona {neuron_idx} ({layer_name})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    print(f"\n📐 Dimensiones:")
    print(f"   Parche real: {real_patch.shape}")
    print(f"   Sintética redimensionada: {synthetic_resized.shape}")
    print(f"   Centro: ({center_x}, {center_y})")
