# ===================================================================
# ARCHIVO: src/utils/image_analyzer.py
# ===================================================================

"""
Utilidades para analizar activaciones de neuronas con imágenes individuales.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import matplotlib.pyplot as plt

from .hooks import ActivationHook


class SingleImageAnalyzer:
    """
    Analiza las activaciones de un modelo para una imagen individual.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: torch.device = None
    ):
        """
        Inicializa el analizador.

        Args:
            model: Modelo de PyTorch (ej: ResNet-18)
            target_layer: Nombre de la capa a analizar (ej: 'layer1.0.conv1')
            device: Device para cómputo (CPU/GPU)
        """
        self.model = model.to(device)
        self.model.eval()
        self.target_layer = target_layer
        self.device = device if device else torch.device('cpu')

        # Registrar hook
        self.hook = ActivationHook(self.model, [target_layer])
        self.hook.register_hooks()

        # Verificar captura
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)

        captured = self.hook.get_activations()
        if captured:
            self.actual_layer_name = list(captured.keys())[0]
            print(f"✅ SingleImageAnalyzer inicializado")
            print(f"   Capa objetivo: {target_layer}")
            print(f"   Capa capturada: {self.actual_layer_name}")
            print(f"   Device: {self.device}")
        else:
            raise ValueError(f"No se pudo capturar la capa '{target_layer}'")

        self.hook.clear_activations()

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(
            3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(
            3, 1, 1).to(self.device)

    def load_image(
        self,
        image_path: Union[str, Path],
        size: Tuple[int, int] = (224, 224)
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Carga una imagen desde archivo.

        Args:
            image_path: Ruta a la imagen
            size: Tamaño al que redimensionar (H, W)

        Returns:
            Tuple con:
                - Tensor normalizado [1, 3, H, W] para el modelo
                - Array numpy [H, W, 3] para visualización
        """
        # Cargar imagen
        img_pil = Image.open(image_path).convert('RGB')
        img_pil = img_pil.resize(size, Image.BILINEAR)

        # Para visualización
        img_vis = np.array(img_pil).astype(np.float32) / 255.0

        # Para el modelo (normalizado)
        img_array = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Normalizar con ImageNet stats
        img_normalized = (img_tensor - self.mean) / self.std

        return img_normalized, img_vis

    def analyze_image(
        self,
        image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Analiza una imagen y extrae activaciones.

        Args:
            image: Tensor de imagen [1, 3, H, W] normalizado

        Returns:
            Dict con:
                - 'activations': Tensor [1, C, H, W] de activaciones
                - 'prediction': Índice de clase predicha
                - 'confidence': Confianza de la predicción
        """
        with torch.no_grad():
            # Forward pass
            output = self.model(image)

            # Predicción
            probs = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

            # Obtener activaciones
            activations = self.hook.get_activations()
            layer_acts = activations[self.actual_layer_name]

            self.hook.clear_activations()

        return {
            'activations': layer_acts,
            'prediction': prediction.item(),
            'confidence': confidence.item()
        }

    def get_neuron_statistics(
        self,
        activations: torch.Tensor
    ) -> List[Dict]:
        """
        Calcula estadísticas de activación para cada neurona.

        Args:
            activations: Tensor [1, C, H, W] de activaciones

        Returns:
            Lista de diccionarios con estadísticas por neurona
        """
        num_neurons = activations.shape[1]
        stats = []

        for neuron_idx in range(num_neurons):
            neuron_act = activations[0, neuron_idx, :, :]

            stats.append({
                'neuron_idx': neuron_idx,
                'mean': neuron_act.mean().item(),
                'max': neuron_act.max().item(),
                'min': neuron_act.min().item(),
                'std': neuron_act.std().item(),
                'sparsity': (neuron_act == 0).float().mean().item()
            })

        return stats

    def get_top_neurons(
        self,
        stats: List[Dict],
        top_k: int = 10,
        criterion: str = 'mean'
    ) -> List[int]:
        """
        Obtiene los índices de las top-k neuronas más activas.

        Args:
            stats: Lista de estadísticas por neurona
            top_k: Número de neuronas a retornar
            criterion: Criterio de ordenamiento ('mean', 'max', 'std')

        Returns:
            Lista de índices de neuronas
        """
        sorted_stats = sorted(stats, key=lambda x: x[criterion], reverse=True)
        return [s['neuron_idx'] for s in sorted_stats[:top_k]]

    def visualize_neuron_activations(
        self,
        image_vis: np.ndarray,
        activations: torch.Tensor,
        neuron_indices: List[int],
        figsize: Tuple[int, int] = (20, 12),
        cmap: str = 'jet'
    ):
        """
        Visualiza mapas de activación de múltiples neuronas.

        Args:
            image_vis: Imagen original [H, W, 3] para visualización
            activations: Tensor [1, C, H, W] de activaciones
            neuron_indices: Lista de índices de neuronas a visualizar
            figsize: Tamaño de la figura
            cmap: Colormap para heatmaps
        """
        num_neurons = len(neuron_indices)
        num_cols = min(4, num_neurons)
        num_rows = (num_neurons + num_cols - 1) // num_cols + 1

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(num_rows, num_cols, hspace=0.3, wspace=0.3)

        # Fila superior: Imagen original (ocupa múltiples columnas)
        ax_img = fig.add_subplot(gs[0, :])
        ax_img.imshow(image_vis)
        ax_img.set_title('Imagen Original', fontsize=14, fontweight='bold')
        ax_img.axis('off')

        # Filas siguientes: Mapas de activación
        for idx, neuron_idx in enumerate(neuron_indices):
            row = (idx // num_cols) + 1
            col = idx % num_cols

            ax = fig.add_subplot(gs[row, col])

            # Obtener mapa de activación
            act_map = activations[0, neuron_idx, :, :].cpu().numpy()

            # Normalizar para visualización
            if act_map.max() > act_map.min():
                act_map_norm = (act_map - act_map.min()) / \
                    (act_map.max() - act_map.min())
            else:
                act_map_norm = act_map

            # Mostrar heatmap
            im = ax.imshow(act_map_norm, cmap=cmap, interpolation='bilinear')

            # Estadísticas
            mean_act = act_map.mean()
            max_act = act_map.max()

            ax.set_title(f'Neurona {neuron_idx}\n'
                         f'μ={mean_act:.3f}, max={max_act:.3f}',
                         fontsize=10, fontweight='bold')
            ax.axis('off')

            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f'Mapas de Activación - {self.target_layer}',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.show()

    def get_activation_overlay(
        self,
        image_vis: np.ndarray,
        activations: torch.Tensor,
        neuron_idx: int,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Crea overlay de activación sobre imagen original.

        Args:
            image_vis: Imagen original [H, W, 3]
            activations: Tensor [1, C, H, W] de activaciones
            neuron_idx: Índice de neurona
            alpha: Transparencia del overlay (0-1)

        Returns:
            Imagen con overlay [H, W, 3]
        """
        from skimage.transform import resize

        # Obtener mapa de activación
        act_map = activations[0, neuron_idx, :, :].cpu().numpy()

        # Normalizar
        if act_map.max() > act_map.min():
            act_map_norm = (act_map - act_map.min()) / \
                (act_map.max() - act_map.min())
        else:
            act_map_norm = np.zeros_like(act_map)

        # ✅ CAMBIO: Usar skimage resize
        act_map_resized = resize(act_map_norm,
                                 (image_vis.shape[0], image_vis.shape[1]),
                                 order=3,  # Bicubic
                                 mode='reflect',
                                 anti_aliasing=True)

        # Aplicar colormap
        act_map_colored = plt.cm.jet(act_map_resized)[:, :, :3]

        # Overlay
        overlay = image_vis * (1 - alpha) + act_map_colored * alpha
        overlay = np.clip(overlay, 0, 1)

        return overlay

    def cleanup(self):
        """Libera recursos."""
        self.hook.remove_hooks()
        print("✅ Recursos liberados")


# ===================================================================
# Funciones de conveniencia
# ===================================================================

def analyze_single_image(
    model: nn.Module,
    image_path: Union[str, Path],
    target_layer: str = 'layer1.0.conv1',
    device: torch.device = None,
    top_k: int = 12,
    visualize: bool = True
) -> Dict:
    """
    Función de conveniencia para analizar una imagen rápidamente.

    Args:
        model: Modelo de PyTorch
        image_path: Ruta a la imagen
        target_layer: Capa a analizar
        device: Device (CPU/GPU)
        top_k: Número de top neuronas a mostrar
        visualize: Si visualizar resultados

    Returns:
        Dict con resultados del análisis
    """
    # Crear analizador
    analyzer = SingleImageAnalyzer(model, target_layer, device)

    # Cargar imagen
    img_tensor, img_vis = analyzer.load_image(image_path)

    # Analizar
    results = analyzer.analyze_image(img_tensor)

    # Estadísticas
    stats = analyzer.get_neuron_statistics(results['activations'])
    top_neurons = analyzer.get_top_neurons(
        stats, top_k=top_k, criterion='mean')

    # Visualizar
    if visualize:
        analyzer.visualize_neuron_activations(
            img_vis,
            results['activations'],
            top_neurons
        )

    # Cleanup
    analyzer.cleanup()

    return {
        'activations': results['activations'],
        'prediction': results['prediction'],
        'confidence': results['confidence'],
        'stats': stats,
        'top_neurons': top_neurons
    }

# ===================================================================
# AGREGAR AL FINAL DE src/utils/image_analyzer.py
# ===================================================================


def analyze_and_visualize_layer(
    model: nn.Module,
    image_path: Union[str, Path],
    target_layer: str = 'conv1',
    device: torch.device = None,
    top_k: int = 12,
    figsize: Tuple[int, int] = (20, 14),
    cmap: str = 'jet',
    show_image: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Analiza una imagen y visualiza activaciones de una capa con logs detallados.

    Args:
        model: Modelo de PyTorch
        image_path: Ruta a la imagen
        target_layer: Nombre de la capa a analizar
        device: Device (CPU/GPU)
        top_k: Número de neuronas a visualizar
        figsize: Tamaño de la figura
        cmap: Colormap para heatmaps
        show_image: Si mostrar la visualización
        verbose: Si imprimir logs detallados

    Returns:
        Dict con todos los resultados del análisis
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===================================================================
    # LOGS: Inicio del análisis
    # ===================================================================
    if verbose:
        print("\n" + "="*70)
        print("🔍 ANÁLISIS DE ACTIVACIONES DE CAPA")
        print("="*70)
        print(f"📁 Imagen: {Path(image_path).name}")
        print(f"🎯 Capa objetivo: {target_layer}")
        print(f"🖥️  Device: {device}")
        print("="*70)

    # ===================================================================
    # Crear analizador
    # ===================================================================
    if verbose:
        print(f"\n🔧 Inicializando analizador...")

    analyzer = SingleImageAnalyzer(model, target_layer, device)

    # ===================================================================
    # Cargar imagen
    # ===================================================================
    if verbose:
        print(f"📥 Cargando imagen...")

    img_tensor, img_vis = analyzer.load_image(image_path)

    if verbose:
        print(
            f"   ✅ Imagen cargada: {img_vis.shape[0]}×{img_vis.shape[1]} píxeles")

    # ===================================================================
    # Analizar imagen
    # ===================================================================
    if verbose:
        print(f"\n🔄 Procesando forward pass...")

    results = analyzer.analyze_image(img_tensor)
    activations = results['activations']

    if verbose:
        print(f"   ✅ Activaciones capturadas")

    # ===================================================================
    # LOGS: Información de la capa
    # ===================================================================
    if verbose:
        print(f"\n📊 INFORMACIÓN DE LA CAPA '{analyzer.actual_layer_name}':")
        print(f"   ┌─ Shape completo: {activations.shape}")
        print(f"   ├─ Batch size:    {activations.shape[0]}")
        print(f"   ├─ Num neuronas:  {activations.shape[1]}")
        print(f"   ├─ Alto mapa:     {activations.shape[2]} píxeles")
        print(f"   └─ Ancho mapa:    {activations.shape[3]} píxeles")

        total_values = activations.numel()
        print(f"\n   Total de valores: {total_values:,}")
        print(
            f"   Memoria aprox: {total_values * 4 / 1024 / 1024:.2f} MB (float32)")

    # ===================================================================
    # Calcular estadísticas
    # ===================================================================
    if verbose:
        print(f"\n📈 Calculando estadísticas...")

    stats = analyzer.get_neuron_statistics(activations)
    top_neurons = analyzer.get_top_neurons(
        stats, top_k=top_k, criterion='mean')

    # ===================================================================
    # LOGS: Estadísticas globales
    # ===================================================================
    if verbose:
        act_mean = activations.mean().item()
        act_max = activations.max().item()
        act_min = activations.min().item()
        act_std = activations.std().item()
        sparsity = (activations == 0).float().mean().item() * 100

        print(f"\n📊 ESTADÍSTICAS GLOBALES DE ACTIVACIONES:")
        print(f"   ┌─ Media:              {act_mean:10.4f}")
        print(f"   ├─ Máximo:            {act_max:10.4f}")
        print(f"   ├─ Mínimo:            {act_min:10.4f}")
        print(f"   ├─ Desviación std:    {act_std:10.4f}")
        print(f"   └─ Sparsity (% ceros): {sparsity:9.1f}%")

        # Análisis de distribución
        num_positive = (activations > 0).sum().item()
        num_zero = (activations == 0).sum().item()
        num_negative = (activations < 0).sum().item()

        print(f"\n   Distribución de valores:")
        print(
            f"   ┌─ Positivos: {num_positive:10,} ({num_positive/total_values*100:5.1f}%)")
        print(
            f"   ├─ Ceros:    {num_zero:10,} ({num_zero/total_values*100:5.1f}%)")
        print(
            f"   └─ Negativos: {num_negative:10,} ({num_negative/total_values*100:5.1f}%)")

    # ===================================================================
    # LOGS: Top neuronas
    # ===================================================================
    if verbose:
        print(f"\n🏆 TOP {len(top_neurons)} NEURONAS MÁS ACTIVAS:")
        print(
            f"   {'#':>3} | {'Neurona':>8} | {'Media':>10} | {'Máxima':>10} | {'Std':>10} | {'Sparsity':>10}")
        print(f"   {'-'*70}")

        for rank, neuron_idx in enumerate(top_neurons, 1):
            s = stats[neuron_idx]
            print(f"   {rank:3d} | {neuron_idx:8d} | {s['mean']:10.4f} | "
                  f"{s['max']:10.4f} | {s['std']:10.4f} | {s['sparsity']*100:9.1f}%")

    # ===================================================================
    # LOGS: Predicción del modelo
    # ===================================================================
    if verbose:
        print(f"\n🔮 PREDICCIÓN DEL MODELO (ImageNet):")
        print(f"   Clase predicha: #{results['prediction']}")
        print(f"   Confianza:      {results['confidence']:.2%}")

    # ===================================================================
    # Visualización
    # ===================================================================
    if show_image:
        if verbose:
            print(f"\n🎨 Generando visualización...")

        analyzer.visualize_neuron_activations(
            image_vis=img_vis,
            activations=activations,
            neuron_indices=top_neurons,
            figsize=figsize,
            cmap=cmap
        )

        if verbose:
            print(f"   ✅ Visualización completada")

    # ===================================================================
    # Cleanup
    # ===================================================================
    analyzer.cleanup()

    if verbose:
        print("\n" + "="*70)
        print("✅ ANÁLISIS COMPLETADO")
        print("="*70)

    # ===================================================================
    # Retornar resultados
    # ===================================================================
    return {
        'activations': activations,
        'prediction': results['prediction'],
        'confidence': results['confidence'],
        'stats': stats,
        'top_neurons': top_neurons,
        'layer_name': analyzer.actual_layer_name,
        'image_vis': img_vis,
        'image_tensor': img_tensor
    }
