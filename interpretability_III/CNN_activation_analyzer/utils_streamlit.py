"""
Utilidades para análisis de activaciones de CNN con Streamlit.
Adaptado del módulo image_analyzer.py original.
"""

from turtle import st
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import io


class ActivationHook:
    """
    Hook para capturar activaciones de capas específicas.
    Versión simplificada sin logging para uso en Streamlit.
    """

    def __init__(self, model: nn.Module, target_layers: List[str]):
        """
        Inicializa el hook.

        Args:
            model: Modelo PyTorch
            target_layers: Lista de nombres de capas a capturar
        """
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.hooks = []

    def _make_hook(self, name: str):
        """Crea una función hook para una capa específica."""
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def register_hooks(self):
        """Registra los hooks en las capas objetivo."""
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Retorna las activaciones capturadas."""
        return self.activations

    def clear_activations(self):
        """Limpia las activaciones almacenadas."""
        self.activations = {}

    def remove_hooks(self):
        """Remueve todos los hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class StreamlitImageAnalyzer:
    """
    Analizador de activaciones optimizado para Streamlit.
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
            model: Modelo de PyTorch (ResNet-18 o AlexNet)
            target_layer: Nombre de la capa a analizar
            device: Device para cómputo (CPU/GPU)
        """
        self.model = model.to(device)
        self.model.eval()
        self.target_layer = target_layer
        self.device = device if device else torch.device('cpu')

        # Registrar hook
        self.hook = ActivationHook(self.model, [target_layer])
        self.hook.register_hooks()

        # Verificar captura con imagen dummy
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)

        captured = self.hook.get_activations()
        if captured:
            self.actual_layer_name = list(captured.keys())[0]
        else:
            raise ValueError(f"No se pudo capturar la capa '{target_layer}'")

        self.hook.clear_activations()

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(
            3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(
            3, 1, 1).to(self.device)

    def load_image_from_pil(
        self,
        pil_image: Image.Image,
        size: Tuple[int, int] = (224, 224)
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Carga una imagen desde objeto PIL.

        Args:
            pil_image: Imagen PIL
            size: Tamaño al que redimensionar (H, W)

        Returns:
            Tuple con:
                - Tensor normalizado [1, 3, H, W] para el modelo
                - Array numpy [H, W, 3] para visualización
        """
        # Convertir y redimensionar
        img_pil = pil_image.convert('RGB')
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
            Dict con activaciones, predicción y confianza
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
        criterion: str = 'mean',
        activation_weight: float = 0.7,
        min_sparsity: float = 0.0  # Nuevo parámetro
    ) -> List[int]:
        """
        Obtiene los índices de las top-k neuronas más activas y selectivas.

        Args:
            stats: Lista de estadísticas por neurona
            top_k: Número de neuronas a retornar
            criterion: Criterio de selección
            activation_weight: Peso de activación en score balanceado (0-1)
            min_sparsity: Sparsity mínima requerida (0-1) para filtrar neuronas

        Returns:
            Lista de índices de neuronas
        """

        if criterion == 'balanced':
            selectivity_weight = 1.0 - activation_weight

            # Filtrar neuronas con sparsity muy baja (poco selectivas)
            filtered_stats = [
                s for s in stats if s['sparsity'] >= min_sparsity]

            # Si filtramos demasiado, relajar el criterio
            # if len(filtered_stats) < top_k:
            #     filtered_stats = stats

            # Si filtramos demasiado, usar lo que hay (no relajar)
            if len(filtered_stats) < top_k and len(filtered_stats) > 0:
                # Usar las que hay, aunque sean menos de top_k
                pass  # No hacer nada, usar filtered_stats como está

            # Solo si NO hay ninguna, usar todas (caso extremo)
            if len(filtered_stats) == 0:
                filtered_stats = stats

            # Score combinado
            for s in filtered_stats:
                # Normalizar mean a [0, 1]
                max_mean = max([x['mean'] for x in filtered_stats])
                min_mean = min([x['mean'] for x in filtered_stats])
                if max_mean > min_mean:
                    norm_activation = (s['mean'] - min_mean) / \
                        (max_mean - min_mean)
                else:
                    norm_activation = 0.5

                # Selectividad (sparsity alta es bueno)
                selectivity = s['sparsity']

                # Score combinado con pesos ajustables
                s['balanced_score'] = (activation_weight * norm_activation) + \
                    (selectivity_weight * selectivity)

            sorted_stats = sorted(
                filtered_stats, key=lambda x: x['balanced_score'], reverse=True)
        else:
            # Criterio simple (original)
            sorted_stats = sorted(
                stats, key=lambda x: x[criterion], reverse=True)

        return [s['neuron_idx'] for s in sorted_stats[:top_k]]

    def cleanup(self):
        """Limpia los hooks."""
        self.hook.remove_hooks()


def create_activation_heatmap(
    image_vis: np.ndarray,
    activation_map: np.ndarray,
    title: str = "",
    alpha: float = 0.5,
    cmap: str = 'jet',
    figsize: Tuple[int, int] = (5, 5)
) -> plt.Figure:
    """
    Crea un mapa de calor superpuesto sobre la imagen original.

    Args:
        image_vis: Imagen original [H, W, 3]
        activation_map: Mapa de activación [H, W]
        title: Título del plot
        alpha: Transparencia del heatmap
        cmap: Colormap a usar

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(4, 4))

    # Mostrar imagen original
    ax.imshow(image_vis)

    # Redimensionar mapa de activación al tamaño de la imagen
    from scipy.ndimage import zoom
    h, w = image_vis.shape[:2]
    h_act, w_act = activation_map.shape

    if (h_act, w_act) != (h, w):
        zoom_factors = (h / h_act, w / w_act)
        activation_resized = zoom(activation_map, zoom_factors, order=1)
    else:
        activation_resized = activation_map

    # Normalizar activaciones a [0, 1]
    act_min = activation_resized.min()
    act_max = activation_resized.max()
    if act_max > act_min:
        activation_norm = (activation_resized - act_min) / (act_max - act_min)
    else:
        activation_norm = activation_resized

    # Superponer heatmap
    im = ax.imshow(activation_norm, cmap=cmap, alpha=alpha)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # Colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def create_filter_grid(
    activations: torch.Tensor,
    neuron_indices: List[int],
    image_vis: np.ndarray,
    max_cols: int = 6,
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Crea una grilla de mapas de activación numerados.

    Args:
        activations: Tensor [1, C, H, W] de activaciones
        neuron_indices: Lista de índices de neuronas a visualizar
        image_vis: Imagen original [H, W, 3] para referencia
        max_cols: Número máximo de columnas
        cmap: Colormap a usar

    Returns:
        Figura de matplotlib
    """
    num_neurons = len(neuron_indices)
    num_cols = min(max_cols, num_neurons)
    num_rows = (num_neurons + num_cols - 1) // num_cols

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(3*num_cols, 3*num_rows))

    # Asegurar que axes sea siempre un array 2D
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, neuron_idx in enumerate(neuron_indices):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]

        # Obtener mapa de activación
        act_map = activations[0, neuron_idx, :, :].cpu().numpy()

        # Mostrar
        im = ax.imshow(act_map, cmap=cmap)
        ax.set_title(f'Filtro {neuron_idx}', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Añadir número grande en la esquina
        ax.text(0.05, 0.95, str(idx + 1),
                transform=ax.transAxes,
                fontsize=16, fontweight='bold',
                va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Ocultar ejes vacíos
    for idx in range(num_neurons, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    return fig


def get_available_conv_layers(model: nn.Module, model_name: str) -> List[str]:
    """
    Obtiene lista de capas convolucionales disponibles en el modelo.

    Args:
        model: Modelo PyTorch
        model_name: Nombre del modelo ('resnet18' o 'alexnet')

    Returns:
        Lista de nombres de capas convolucionales
    """
    conv_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)

    return conv_layers


def fig_to_image(fig: plt.Figure) -> Image.Image:
    """
    Convierte una figura de matplotlib a imagen PIL.

    Args:
        fig: Figura de matplotlib

    Returns:
        Imagen PIL
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img
