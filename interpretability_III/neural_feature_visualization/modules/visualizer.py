"""
===================================================================
VISUALIZER.PY - Generación de Visualizaciones
===================================================================

Este módulo crea todas las visualizaciones de la aplicación:
1. Imágenes con overlays de heatmaps
2. Marcadores de ROI y zonas de activación
3. Comparaciones de 4 paneles (Real vs Sintética)
4. Gráficos de estadísticas de neuronas

Uso:
    viz = Visualizer()
    fig = viz.create_heatmap_overlay(image, heatmap, roi_center)
===================================================================
"""

import torch
from config import (
    HEATMAP_COLORMAP,
    HEATMAP_ALPHA,
    ROI_SIZE,
    ROI_BORDER_COLOR,
    ROI_BORDER_WIDTH,
    ROI_MARKER_SIZE,
    FIGURE_DPI
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import Tuple, Optional, List, Dict
from PIL import Image, ImageDraw
from skimage.transform import resize as sk_resize
# Importar configuración
import sys
sys.path.append('..')


class Visualizer:
    """
    Generador de visualizaciones para la dashboard.

    Crea figuras matplotlib para mostrar activaciones, heatmaps,
    comparaciones y estadísticas.
    """

    def __init__(
        self,
        colormap: str = HEATMAP_COLORMAP,
        alpha: float = HEATMAP_ALPHA,
        dpi: int = FIGURE_DPI
    ):
        """
        Inicializa el visualizador.

        Args:
            colormap: Colormap para heatmaps ('jet', 'hot', etc.)
            alpha: Transparencia de overlays (0-1)
            dpi: DPI para figuras
        """
        self.colormap = colormap
        self.alpha = alpha
        self.dpi = dpi

        print(f"✅ Visualizer inicializado")
        print(f"   Colormap: {colormap}")
        print(f"   Alpha: {alpha}")

    def create_heatmap_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        roi_center: Optional[Tuple[int, int]] = None,
        roi_size: Tuple[int, int] = ROI_SIZE,
        title: str = "Mapa de Activación"
    ) -> plt.Figure:
        """
        Crea overlay de heatmap sobre imagen con marcador de ROI.

        Args:
            image: Array [H, W, 3] en [0, 1]
            heatmap: Array [H, W] en [0, 1]
            roi_center: (y, x) centro del ROI (opcional)
            roi_size: (height, width) del ROI
            title: Título de la figura

        Returns:
            Figura de matplotlib
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)

        # Panel 1: Imagen original con ROI
        axes[0].imshow(image)
        axes[0].set_title('Imagen Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Dibujar ROI si se proporciona
        if roi_center is not None:
            y_center, x_center = roi_center
            h, w = roi_size

            # Rectángulo del ROI
            rect = Rectangle(
                (x_center - w//2, y_center - h//2),
                w, h,
                linewidth=ROI_BORDER_WIDTH,
                edgecolor=ROI_BORDER_COLOR,
                facecolor='none'
            )
            axes[0].add_patch(rect)

            # Marcador de estrella en el centro
            axes[0].scatter(
                x_center, y_center,
                marker='*',
                s=ROI_MARKER_SIZE,
                c='yellow',
                edgecolors='black',
                linewidths=1,
                zorder=10
            )

        # Panel 2: Heatmap overlay
        axes[1].imshow(image)

        # Aplicar heatmap con transparencia
        cmap = plt.get_cmap(self.colormap)
        heatmap_colored = cmap(heatmap)
        axes[1].imshow(heatmap_colored, alpha=self.alpha)

        axes[1].set_title('Mapa de Calor', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Dibujar ROI también en heatmap
        if roi_center is not None:
            y_center, x_center = roi_center
            h, w = roi_size

            rect = Rectangle(
                (x_center - w//2, y_center - h//2),
                w, h,
                linewidth=ROI_BORDER_WIDTH,
                edgecolor=ROI_BORDER_COLOR,
                facecolor='none'
            )
            axes[1].add_patch(rect)

            axes[1].scatter(
                x_center, y_center,
                marker='*',
                s=ROI_MARKER_SIZE,
                c='yellow',
                edgecolors='black',
                linewidths=1,
                zorder=10
            )

        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        return fig

    def create_4panel_comparison(
        self,
        image_full: np.ndarray,
        roi_real: np.ndarray,
        roi_synthetic: np.ndarray,
        roi_center: Tuple[int, int],
        neuron_idx: int,
        real_activation: float,
        synthetic_activation: float
    ) -> plt.Figure:
        """
        Crea comparación de 4 paneles: Real vs Ideal (como en la imagen ejemplo).

        Args:
            image_full: Imagen completa [H, W, 3] en [0, 1]
            roi_real: ROI real extraído [roi_h, roi_w, 3]
            roi_synthetic: Patrón sintético [roi_h, roi_w, 3]
            roi_center: (y, x) del ROI
            neuron_idx: Índice de la neurona
            real_activation: Activación en imagen real
            synthetic_activation: Activación en patrón sintético

        Returns:
            Figura de matplotlib
        """
        fig = plt.figure(figsize=(16, 4), dpi=self.dpi)

        # Título principal
        fig.suptitle(
            f'Comparación: Real vs Ideal - Neurona {neuron_idx}',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        # ---------------------------------------------------------------
        # Panel 1: Imagen completa con ROI marcado
        # ---------------------------------------------------------------
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(image_full)

        # Dibujar ROI
        y_center, x_center = roi_center
        h, w = ROI_SIZE
        rect = Rectangle(
            (x_center - w//2, y_center - h//2),
            w, h,
            linewidth=3,
            edgecolor='red',
            facecolor='none'
        )
        ax1.add_patch(rect)

        # Estrella
        ax1.scatter(
            x_center, y_center,
            marker='*',
            s=150,
            c='yellow',
            edgecolors='black',
            linewidths=1.5,
            zorder=10
        )

        ax1.set_title(
            f'Imagen Real Completa\nROI: {w}x{h}',
            fontsize=11,
            fontweight='bold'
        )
        ax1.axis('off')

        # ---------------------------------------------------------------
        # Panel 2: ROI Real (zona de máxima activación)
        # ---------------------------------------------------------------
        ax2 = plt.subplot(1, 4, 2)

        # Asegurar que roi_real está en [0, 1] o escalar
        if roi_real.max() > 1.0:
            roi_real = roi_real / 255.0

        ax2.imshow(roi_real)
        ax2.set_title(
            'Región Real\n(Zona de máxima activación)',
            fontsize=11,
            fontweight='bold'
        )
        ax2.axis('off')

        # Añadir borde
        for spine in ax2.spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(2)

        # ---------------------------------------------------------------
        # Panel 3: Patrón Sintético (Ideal)
        # ---------------------------------------------------------------
        ax3 = plt.subplot(1, 4, 3)

        # Asegurar que roi_synthetic está en [0, 1] o escalar
        if roi_synthetic.max() > 1.0:
            roi_synthetic = roi_synthetic / 255.0

        ax3.imshow(roi_synthetic)
        ax3.set_title(
            'Patrón Ideal (Sintética)\n(Redimensionada)',
            fontsize=11,
            fontweight='bold'
        )
        ax3.axis('off')

        # Añadir borde
        for spine in ax3.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2)

        # ---------------------------------------------------------------
        # Panel 4: Superposición 50/50
        # ---------------------------------------------------------------
        ax4 = plt.subplot(1, 4, 4)

        # Crear overlay 50/50
        overlay = 0.5 * roi_real + 0.5 * roi_synthetic
        overlay = np.clip(overlay, 0, 1)

        ax4.imshow(overlay)
        ax4.set_title(
            'Superposición 50/50\n(Coincidencias visuales)',
            fontsize=11,
            fontweight='bold'
        )
        ax4.axis('off')

        # ---------------------------------------------------------------
        # Añadir métricas en la parte inferior
        # ---------------------------------------------------------------
        metrics_text = (
            f'Activación Real: {real_activation:.3f}  |  '
            f'Activación Sintética: {synthetic_activation:.3f}  |  '
            f'Mejora: {(synthetic_activation/max(real_activation, 1e-8)):.2f}x'
        )

        fig.text(
            0.5, 0.02,
            metrics_text,
            ha='center',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        return fig

    def create_neuron_ranking(
        self,
        stats: List[Dict],
        top_k: int = 10,
        criterion: str = 'mean'
    ) -> plt.Figure:
        """
        Crea gráfico de ranking de neuronas.

        Args:
            stats: Lista de estadísticas por neurona
            top_k: Número de neuronas a mostrar
            criterion: Criterio de ranking ('mean', 'max', 'std')

        Returns:
            Figura de matplotlib
        """
        # Ordenar por criterio
        sorted_stats = sorted(stats, key=lambda x: x[criterion], reverse=True)
        top_stats = sorted_stats[:top_k]

        # Extraer datos
        neuron_indices = [s['neuron_idx'] for s in top_stats]
        values = [s[criterion] for s in top_stats]

        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        # Barras horizontales
        bars = ax.barh(range(len(neuron_indices)), values, color='steelblue')

        # Colorear la barra más alta
        bars[0].set_color('crimson')

        # Etiquetas
        ax.set_yticks(range(len(neuron_indices)))
        ax.set_yticklabels([f'Neurona {idx}' for idx in neuron_indices])
        ax.set_xlabel(f'Activación ({criterion.capitalize()})', fontsize=11)
        ax.set_title(
            f'Top {top_k} Neuronas Más Activas',
            fontsize=13,
            fontweight='bold'
        )

        # Añadir valores en las barras
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(
                val,
                bar.get_y() + bar.get_height()/2,
                f'  {val:.3f}',
                va='center',
                fontsize=9
            )

        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()

        return fig

    def create_activation_heatmap_grid(
        self,
        image: np.ndarray,
        activations: np.ndarray,
        neuron_indices: List[int],
        max_display: int = 8
    ) -> plt.Figure:
        """
        Crea grid de mapas de activación para múltiples neuronas.

        Args:
            image: Imagen original [H, W, 3]
            activations: Tensor [C, H, W] de activaciones
            neuron_indices: Lista de índices de neuronas a mostrar
            max_display: Máximo de neuronas a mostrar

        Returns:
            Figura de matplotlib
        """
        num_neurons = min(len(neuron_indices), max_display)

        # Calcular grid
        ncols = min(4, num_neurons)
        nrows = (num_neurons + ncols - 1) // ncols + \
            1  # +1 para imagen original

        fig = plt.figure(figsize=(4*ncols, 4*nrows), dpi=self.dpi)

        # Primera fila: Imagen original
        ax = plt.subplot(nrows, ncols, 1)
        ax.imshow(image)
        ax.set_title('Imagen Original', fontsize=11, fontweight='bold')
        ax.axis('off')

        # Resto de filas: Mapas de activación
        for idx, neuron_idx in enumerate(neuron_indices[:max_display], start=2):
            ax = plt.subplot(nrows, ncols, idx)

            # Obtener mapa de activación
            act_map = activations[neuron_idx]

            # Normalizar
            if act_map.max() > act_map.min():
                act_map_norm = (act_map - act_map.min()) / \
                    (act_map.max() - act_map.min())
            else:
                act_map_norm = act_map

            # Mostrar heatmap
            im = ax.imshow(act_map_norm, cmap=self.colormap,
                           interpolation='bilinear')

            # Título con estadísticas
            mean_act = act_map.mean()
            max_act = act_map.max()

            ax.set_title(
                f'Neurona {neuron_idx}\nμ={mean_act:.3f}, max={max_act:.3f}',
                fontsize=10,
                fontweight='bold'
            )
            ax.axis('off')

            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(
            'Mapas de Activación por Neurona',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )

        plt.tight_layout()

        return fig

    def add_roi_to_image(
        self,
        image: np.ndarray,
        roi_center: Tuple[int, int],
        roi_size: Tuple[int, int] = ROI_SIZE
    ) -> np.ndarray:
        """
        Añade marcador visual de ROI a una imagen (sin matplotlib).

        Args:
            image: Array [H, W, 3] en [0, 1] o [0, 255]
            roi_center: (y, x) centro del ROI
            roi_size: (height, width) del ROI

        Returns:
            Imagen con ROI dibujado [H, W, 3]
        """
        # Convertir a PIL para dibujar
        if image.max() <= 1.0:
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            img_pil = Image.fromarray(image.astype(np.uint8))

        draw = ImageDraw.Draw(img_pil)

        # Calcular coordenadas del rectángulo
        y_center, x_center = roi_center
        h, w = roi_size

        x1 = x_center - w // 2
        y1 = y_center - h // 2
        x2 = x_center + w // 2
        y2 = y_center + h // 2

        # Dibujar rectángulo
        draw.rectangle(
            [x1, y1, x2, y2],
            outline='red',
            width=ROI_BORDER_WIDTH
        )

        # Dibujar punto central (aproximación de estrella)
        star_size = 8
        draw.ellipse(
            [x_center-star_size, y_center-star_size,
             x_center+star_size, y_center+star_size],
            fill='yellow',
            outline='black'
        )

        # Convertir de vuelta a numpy
        img_with_roi = np.array(img_pil)

        # Mantener el mismo rango que entrada
        if image.max() <= 1.0:
            img_with_roi = img_with_roi / 255.0

        return img_with_roi

    def create_pattern_overlay_comparison(
        self,
        image_full: np.ndarray,
        synthetic_pattern: np.ndarray,
        roi_center: tuple[int, int],
        neuron_idx: int,
        real_activation: float,
        synthetic_activation: float
    ) -> plt.Figure:
        """
        Crea comparación con patrón sintético repetido sobre toda la imagen.

        Args:
            image_full: Imagen completa [H, W, 3]
            synthetic_pattern: Patrón sintético [H_s, W_s, 3]
            roi_center: Centro del ROI (y, x)
            neuron_idx: Índice de la neurona
            real_activation: Activación en imagen real
            synthetic_activation: Activación en patrón sintético

        Returns:
            Figura de matplotlib
        """
        fig = plt.figure(figsize=(16, 5), dpi=self.dpi)

        # Título principal
        fig.suptitle(
            f'Análisis de Neurona {neuron_idx} - Patrón Ideal Superpuesto',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        # Panel 1: Imagen original con ROI
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(image_full)

        # Marcar ROI
        y_center, x_center = roi_center
        h, w = ROI_SIZE
        rect = Rectangle(
            (x_center - w//2, y_center - h//2),
            w, h,
            linewidth=3,
            edgecolor='red',
            facecolor='none'
        )
        ax1.add_patch(rect)

        ax1.scatter(
            x_center, y_center,
            marker='*',
            s=150,
            c='yellow',
            edgecolors='black',
            linewidths=1.5,
            zorder=10
        )

        ax1.set_title(
            'Imagen Real\n(ROI: zona de máxima activación)',
            fontsize=12,
            fontweight='bold'
        )
        ax1.axis('off')

        # Panel 2: Patrón sintético (grande, para referencia)
        ax2 = plt.subplot(1, 3, 2)

        # Asegurar rango correcto
        if synthetic_pattern.max() > 1.0:
            synthetic_display = synthetic_pattern / 255.0
        else:
            synthetic_display = synthetic_pattern

        ax2.imshow(synthetic_display)
        ax2.set_title(
            'Patrón Ideal\n(Maximiza activación)',
            fontsize=12,
            fontweight='bold'
        )
        ax2.axis('off')

        for spine in ax2.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)

        # Panel 3: Imagen con patrón repetido superpuesto
        ax3 = plt.subplot(1, 3, 3)

        # Crear patrón tiled del tamaño de la imagen
        img_h, img_w = image_full.shape[:2]

        # ESCALAR el patrón a un tamaño apropiado para tiling

        # Redimensionar patrón a tamaño de ROI para que se vea bien repetido
        target_tile_size = ROI_SIZE  # (32, 32) típicamente
        pattern_scaled = sk_resize(
            synthetic_display,
            target_tile_size,
            anti_aliasing=True
        )

        pat_h, pat_w = pattern_scaled.shape[:2]

        # Calcular cuántas repeticiones necesitamos
        n_tiles_y = int(np.ceil(img_h / pat_h)) + 1
        n_tiles_x = int(np.ceil(img_w / pat_w)) + 1

        # Crear patrón tiled
        pattern_tiled = np.tile(pattern_scaled, (n_tiles_y, n_tiles_x, 1))

        # Recortar al tamaño exacto de la imagen
        pattern_tiled = pattern_tiled[:img_h, :img_w, :]

        # Superponer imagen + patrón
        # Imagen base con más peso
        alpha_image = 0.6
        alpha_pattern = 0.4

        overlay = alpha_image * image_full + alpha_pattern * pattern_tiled
        overlay = np.clip(overlay, 0, 1)

        ax3.imshow(overlay)

        # Marcar ROI también aquí
        rect = Rectangle(
            (x_center - w//2, y_center - h//2),
            w, h,
            linewidth=3,
            edgecolor='red',
            facecolor='none',
            linestyle='--'
        )
        ax3.add_patch(rect)

        ax3.set_title(
            'Superposición Global\n(Patrón repetido sobre imagen)',
            fontsize=12,
            fontweight='bold'
        )
        ax3.axis('off')

        # Métricas en la parte inferior
        metrics_text = (
            f'Activación Real: {real_activation:.3f}  |  '
            f'Activación Sintética: {synthetic_activation:.3f}  |  '
            f'Mejora: {(synthetic_activation/max(real_activation, 1e-8)):.2f}x'
        )

        fig.text(
            0.5, 0.02,
            metrics_text,
            ha='center',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        return fig

    def create_heatmap_with_neuron_markers(
        self,
        image: np.ndarray,
        activations: torch.Tensor,
        neuron_stats: List[Dict],
        top_n: int = 10,
        title: str = "Mapa de Activación con Neuronas Marcadas"
    ) -> plt.Figure:
        """
        Crea mapa de calor con marcadores numerados para top neuronas.

        Args:
            image: Imagen original [H, W, 3]
            activations: Tensor de activaciones [1, C, H, W]
            neuron_stats: Lista de estadísticas de neuronas
            top_n: Número de neuronas top a marcar
            title: Título de la figura

        Returns:
            Figura de matplotlib
        """
        from config import IMAGE_SIZE
        from scipy.ndimage import zoom

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=self.dpi)

        # Título principal
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        # Panel 1: Imagen original con marcadores
        ax1.imshow(image)
        ax1.set_title('Imagen Original\n(Neuronas numeradas por activación)',
                      fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Panel 2: Mapa de calor con marcadores
        # Generar heatmap global (máximo entre todas las neuronas)
        if activations.dim() == 4:
            acts = activations[0]  # [C, H, W]
        else:
            acts = activations

        # Heatmap global
        heatmap = acts.max(dim=0)[0].cpu().numpy()  # [H, W]

        # Redimensionar al tamaño de la imagen
        scale_y = IMAGE_SIZE[0] / heatmap.shape[0]
        scale_x = IMAGE_SIZE[1] / heatmap.shape[1]
        heatmap_resized = zoom(heatmap, (scale_y, scale_x), order=1)

        # Normalizar
        heatmap_norm = (heatmap_resized - heatmap_resized.min()) / \
                       (heatmap_resized.max() - heatmap_resized.min() + 1e-8)

        # Importar configuración
        from config import HEATMAP_COLORMAP, HEATMAP_ALPHA
        # Mostrar imagen con overlay
        ax2.imshow(image)
        heatmap_colored = plt.get_cmap(HEATMAP_COLORMAP)(heatmap_norm)
        ax2.imshow(heatmap_colored, alpha=HEATMAP_ALPHA)
        ax2.set_title('Mapa de Calor\n(Posiciones de máxima activación)',
                      fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Ordenar neuronas por activación
        ranked = sorted(neuron_stats, key=lambda x: x['mean'], reverse=True)[
            :top_n]

        # Marcar cada neurona en ambos paneles
        for i, stat in enumerate(ranked, 1):
            neuron_idx = stat['neuron_idx']

            # Obtener mapa de activación de esta neurona específica
            neuron_map = acts[neuron_idx].cpu().numpy()  # [H, W]

            # Encontrar posición de máximo
            max_pos = np.unravel_index(neuron_map.argmax(), neuron_map.shape)

            # Escalar a coordenadas de imagen
            y_img = int(max_pos[0] * scale_y)
            x_img = int(max_pos[1] * scale_x)

            # Color diferente para la neurona #1
            if i == 1:
                marker_color = 'red'
                marker_size = 200
                edge_color = 'yellow'
                edge_width = 3
            else:
                marker_color = 'lime'
                marker_size = 150
                edge_color = 'white'
                edge_width = 2

            # Marcar en ambos paneles
            for ax in [ax1, ax2]:
                # Punto
                ax.scatter(
                    x_img, y_img,
                    s=marker_size,
                    c=marker_color,
                    marker='o',
                    edgecolors=edge_color,
                    linewidths=edge_width,
                    zorder=10,
                    alpha=0.8
                )

                # Número
                ax.text(
                    x_img, y_img,
                    str(i),
                    color='white',
                    fontsize=10 if i == 1 else 8,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    zorder=11,
                    bbox=dict(
                        boxstyle='circle,pad=0.1',
                        facecolor='black',
                        alpha=0.7,
                        edgecolor='none'
                    )
                )

        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='yellow',
                  label='#1 Neurona más activa'),
            Patch(facecolor='lime', edgecolor='white',
                  label='Top 2-10 neuronas')
        ]
        ax2.legend(
            handles=legend_elements,
            loc='upper right',
            framealpha=0.9,
            fontsize=9
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        return fig


# ===================================================================
# FUNCIONES DE UTILIDAD
# ===================================================================

def resize_to_match(
    source: np.ndarray,
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Redimensiona una imagen para que coincida con un shape objetivo.

    Args:
        source: Array a redimensionar
        target_shape: (height, width) objetivo

    Returns:
        Array redimensionado
    """
    from skimage.transform import resize

    if source.ndim == 3:
        return resize(
            source,
            (target_shape[0], target_shape[1], source.shape[2]),
            anti_aliasing=True
        )
    else:
        return resize(
            source,
            target_shape,
            anti_aliasing=True
        )


# ===================================================================
# TESTING
# ===================================================================

if __name__ == "__main__":
    print("🧪 Testing Visualizer...\n")

    # Crear visualizer
    viz = Visualizer()

    # Crear datos de prueba
    print("1️⃣ Creando datos de prueba...")
    image = np.random.rand(224, 224, 3)
    heatmap = np.random.rand(224, 224)
    roi_center = (112, 112)

    # Test 1: Heatmap overlay
    print("\n2️⃣ Creando heatmap overlay...")
    fig1 = viz.create_heatmap_overlay(image, heatmap, roi_center)
    print("   ✓ Figura creada")
    plt.close(fig1)

    # Test 2: ROI en imagen
    print("\n3️⃣ Añadiendo ROI a imagen...")
    img_with_roi = viz.add_roi_to_image(image, roi_center)
    print(f"   ✓ Imagen con ROI: {img_with_roi.shape}")

    # Test 3: Ranking
    print("\n4️⃣ Creando ranking de neuronas...")
    fake_stats = [
        {'neuron_idx': i, 'mean': np.random.rand(), 'max': np.random.rand(),
         'std': 0.1}
        for i in range(20)
    ]
    fig2 = viz.create_neuron_ranking(fake_stats, top_k=10)
    print("   ✓ Figura de ranking creada")
    plt.close(fig2)

    # Test 4: Comparación 4-panel
    print("\n5️⃣ Creando comparación 4-panel...")
    roi_real = np.random.rand(32, 32, 3)
    roi_synthetic = np.random.rand(32, 32, 3)
    fig3 = viz.create_4panel_comparison(
        image, roi_real, roi_synthetic,
        roi_center, neuron_idx=38,
        real_activation=18.3,
        synthetic_activation=24.7
    )
    print("   ✓ Figura de comparación creada")
    plt.close(fig3)

    print("\n✅ Testing completado!")
