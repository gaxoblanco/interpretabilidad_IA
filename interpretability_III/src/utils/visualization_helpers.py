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
    ax2.set_title('Imagen Sintética Optimizada', fontsize=12, fontweight='bold')
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
        report.append(f"   Activación media:  {stats.get('mean_activation', 0):.4f}")
        report.append(f"   Activación máxima: {stats.get('max_activation', 0):.4f}")
        report.append(f"   Activación mínima: {stats.get('min_activation', 0):.4f}")
        report.append(f"   Desviación std:    {stats.get('std_activation', 0):.4f}")
    
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
