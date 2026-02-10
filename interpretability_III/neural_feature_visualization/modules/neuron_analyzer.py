"""
===================================================================
NEURON_ANALYZER.PY - Análisis de Activaciones Neuronales
===================================================================

Este módulo analiza las activaciones de neuronas en una capa específica
cuando el modelo procesa una imagen.

Funcionalidades principales:
1. Extracción de activaciones mediante hooks
2. Cálculo de estadísticas por neurona
3. Generación de mapas de calor (heatmaps)
4. Identificación de ROI (Region of Interest) de máxima activación
5. Ranking de neuronas por nivel de activación

Uso:
    analyzer = NeuronAnalyzer(model, 'features.0')
    activations = analyzer.extract_activations(image_tensor)
    heatmap = analyzer.compute_heatmap(activations)
===================================================================
"""

from config import ROI_SIZE
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import zoom, center_of_mass
from skimage.transform import resize

# Importar configuración
import sys
sys.path.append('..')


class ActivationHook:
    """
    Clase auxiliar para registrar hooks y capturar activaciones.
    """

    def __init__(self, model: nn.Module, target_layer: str):
        """
        Args:
            model: Modelo de PyTorch
            target_layer: Nombre de la capa a capturar
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = {}
        self.hooks = []
        self.hook_registered = False

    def _hook_fn(self, name: str):
        """Crea función hook para una capa específica."""
        def hook(module, input, output):
            # Guardar con la clave 'target' para acceso consistente
            self.activations['target'] = output.detach()
            # También con el nombre original
            self.activations[name] = output.detach()
        return hook

    def register_hook(self):
        """Registra el hook en la capa objetivo."""
        found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                handle = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(handle)
                self.hook_registered = True
                found = True
                print(f"   ✓ Hook registrado exitosamente en: {name}")
                break

        if not found:
            # Mostrar capas disponibles para debug
            print(f"   ❌ Capa '{self.target_layer}' no encontrada")
            print(f"   Capas disponibles:")
            for name, _ in self.model.named_modules():
                if len(name) > 0 and 'Conv2d' in str(type(_)):
                    print(f"      • {name}")

        return found

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Retorna las activaciones capturadas."""
        return self.activations

    def clear(self):
        """Limpia activaciones y hooks."""
        self.activations.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.hook_registered = False

    def clear_activations_only(self):
        """Limpia solo las activaciones, mantiene los hooks activos."""
        self.activations.clear()


class NeuronAnalyzer:
    """
    Analizador de activaciones neuronales.

    Extrae y analiza las activaciones de una capa específica del modelo
    cuando procesa una imagen.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: Optional[torch.device] = None
    ):
        """
        Inicializa el analizador.

        Args:
            model: Modelo de PyTorch en modo eval
            target_layer: Nombre de la capa a analizar (ej: 'features.0')
            device: Device para computación
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.device = device if device else torch.device('cpu')

        # Hook para capturar activaciones
        self.hook = ActivationHook(model, target_layer)
        success = self.hook.register_hook()

        if not success:
            raise ValueError(
                f"No se pudo registrar hook en capa '{target_layer}'")

        # Verificar dimensiones con un forward pass dummy
        self._verify_layer()

        print(f"✅ NeuronAnalyzer inicializado")
        print(f"   Capa objetivo: {target_layer}")
        print(f"   Device: {self.device}")

    def _verify_layer(self):
        """Verifica que la capa capture activaciones correctamente."""
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)

        acts = self.hook.get_activations()
        if 'target' in acts:
            shape = acts['target'].shape
            print(f"   Shape de activaciones: {shape}")
            self.num_channels = shape[1]
        else:
            available_keys = list(acts.keys())
            raise ValueError(
                f"Capa '{self.target_layer}' no capturó activaciones correctamente. "
                f"Claves disponibles: {available_keys}"
            )

        self.hook.clear_activations_only()

    def extract_activations(
        self,
        image_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Extrae activaciones de la capa objetivo para una imagen.

        Args:
            image_tensor: Tensor [1, 3, H, W] normalizado

        Returns:
            Tensor [1, C, H', W'] con activaciones
        """
        # Asegurar que está en el device correcto
        image_tensor = image_tensor.to(self.device)

        # Asegurar dimensión de batch
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Limpiar SOLO las activaciones, NO los hooks
        self.hook.activations.clear()

        # Forward pass
        with torch.no_grad():
            _ = self.model(image_tensor)

        # Obtener activaciones usando la clave 'target'
        acts = self.hook.get_activations()

        if 'target' not in acts:
            raise RuntimeError(
                f"No se capturaron activaciones para la capa '{self.target_layer}'. "
                f"Claves disponibles: {list(acts.keys())}"
            )

        activations = acts['target']

        return activations

    def compute_neuron_statistics(
        self,
        activations: torch.Tensor
    ) -> List[Dict]:
        """
        Calcula estadísticas para cada neurona (canal).

        Args:
            activations: Tensor [1, C, H, W] de activaciones

        Returns:
            Lista de diccionarios con estadísticas por neurona:
            - neuron_idx: Índice de la neurona
            - mean: Activación promedio
            - max: Activación máxima
            - std: Desviación estándar
            - max_position: (y, x) de activación máxima
        """
        stats = []

        # Remover batch dimension
        acts = activations[0]  # [C, H, W]

        for neuron_idx in range(acts.shape[0]):
            neuron_act = acts[neuron_idx]  # [H, W]

            # Encontrar posición de máxima activación
            max_val, max_idx = neuron_act.flatten().max(0)
            max_pos = np.unravel_index(max_idx.cpu().item(), neuron_act.shape)

            stats.append({
                'neuron_idx': neuron_idx,
                'mean': neuron_act.mean().item(),
                'max': max_val.item(),
                'std': neuron_act.std().item(),
                'max_position': max_pos  # (y, x)
            })

        return stats

    def rank_neurons(
        self,
        stats: List[Dict],
        criterion: str = 'mean',
        top_k: Optional[int] = None
    ) -> List[int]:
        """
        Rankea neuronas por criterio de activación.

        Args:
            stats: Lista de estadísticas por neurona
            criterion: Criterio de ranking ('mean', 'max', 'std')
            top_k: Si especificado, retorna solo top-k neuronas

        Returns:
            Lista de índices de neuronas ordenados (mayor a menor)
        """
        # Ordenar por criterio
        sorted_stats = sorted(stats, key=lambda x: x[criterion], reverse=True)

        # Extraer índices
        ranked_indices = [s['neuron_idx'] for s in sorted_stats]

        # Retornar top-k si se especifica
        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        return ranked_indices

    def get_neuron_activation_map(
        self,
        activations: torch.Tensor,
        neuron_idx: int
    ) -> np.ndarray:
        """
        Obtiene el mapa de activación de una neurona específica.

        Args:
            activations: Tensor [1, C, H, W]
            neuron_idx: Índice de la neurona

        Returns:
            Array numpy [H, W] con mapa de activación normalizado [0, 1]
        """
        # Extraer mapa de la neurona
        act_map = activations[0, neuron_idx].cpu().numpy()

        # Normalizar a [0, 1]
        if act_map.max() > act_map.min():
            act_map = (act_map - act_map.min()) / \
                (act_map.max() - act_map.min())
        else:
            act_map = np.zeros_like(act_map)

        return act_map

    def compute_heatmap(
        self,
        activations: torch.Tensor,
        method: str = 'max'
    ) -> np.ndarray:
        """
        Computa mapa de calor global combinando todas las neuronas.

        Args:
            activations: Tensor [1, C, H, W]
            method: Método de agregación:
                   - 'max': Máximo por posición espacial
                   - 'mean': Promedio por posición
                   - 'weighted': Promedio ponderado por activación máxima

        Returns:
            Array numpy [H, W] normalizado [0, 1]
        """
        acts = activations[0].cpu().numpy()  # [C, H, W]

        if method == 'max':
            heatmap = acts.max(axis=0)
        elif method == 'mean':
            heatmap = acts.mean(axis=0)
        elif method == 'weighted':
            # Ponderar por activación máxima de cada canal
            max_per_channel = acts.max(axis=(1, 2), keepdims=True)
            weights = max_per_channel / (max_per_channel.sum() + 1e-8)
            heatmap = (acts * weights).sum(axis=0)
        else:
            raise ValueError(f"Método '{method}' no soportado")

        # Normalizar
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / \
                (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def resize_heatmap(
        self,
        heatmap: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Redimensiona mapa de calor a tamaño objetivo.

        Args:
            heatmap: Array [H, W]
            target_size: (height, width) objetivo

        Returns:
            Heatmap redimensionado
        """
        return resize(
            heatmap,
            target_size,
            order=3,  # Bicubic
            mode='reflect',
            anti_aliasing=True
        )

    def find_max_activation_region(
        self,
        activation_map: np.ndarray,
        roi_size: Tuple[int, int] = ROI_SIZE
    ) -> Tuple[int, int]:
        """
        Encuentra la región de máxima activación en el mapa.

        Usa el método 'weighted_area' que encuentra el centro de masa
        de la región con alta activación (más robusto que buscar un
        solo píxel máximo).

        Args:
            activation_map: Array [H, W] con mapa de activación
            roi_size: Tamaño del ROI (height, width)

        Returns:
            Tupla (y, x) con coordenadas del centro del ROI
        """
        # Umbral: considerar píxeles con activación > 80% del máximo
        threshold = activation_map.max() * 0.8
        hot_mask = activation_map >= threshold

        # Seguridad: si no hay píxeles calientes, usar máximo simple
        if hot_mask.sum() == 0:
            max_idx = activation_map.argmax()
            max_y, max_x = np.unravel_index(max_idx, activation_map.shape)
            return (max_y, max_x)

        # Calcular centro de masa de la región caliente
        # Esto pondera cada píxel por su valor de activación
        cy, cx = center_of_mass(activation_map * hot_mask)

        # Redondear a coordenadas enteras
        return (int(round(cy)), int(round(cx)))

    def extract_roi(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        roi_size: Tuple[int, int] = ROI_SIZE
    ) -> np.ndarray:
        """
        Extrae ROI de la imagen centrado en una posición.

        Args:
            image: Array [H, W, 3] o [H, W]
            center: (y, x) centro del ROI
            roi_size: (height, width) del ROI

        Returns:
            ROI extraído [roi_h, roi_w, 3] o [roi_h, roi_w]
        """
        cy, cx = center
        h, w = roi_size

        # Calcular límites
        y1 = max(0, cy - h // 2)
        y2 = min(image.shape[0], cy + h // 2)
        x1 = max(0, cx - w // 2)
        x2 = min(image.shape[1], cx + w // 2)

        # Extraer ROI
        if image.ndim == 3:
            roi = image[y1:y2, x1:x2, :]
        else:
            roi = image[y1:y2, x1:x2]

        # Redimensionar si no es del tamaño exacto
        if roi.shape[:2] != roi_size:
            if image.ndim == 3:
                roi = resize(roi, (roi_size[0], roi_size[1], image.shape[2]))
            else:
                roi = resize(roi, roi_size)

        return roi

    def get_roi_bounds(
        self,
        center: Tuple[int, int],
        roi_size: Tuple[int, int],
        image_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calcula los límites del ROI respetando bordes de imagen.

        Args:
            center: (y, x) centro del ROI
            roi_size: (height, width) del ROI
            image_size: (height, width) de la imagen

        Returns:
            Tupla (y1, x1, y2, x2) con límites del ROI
        """
        cy, cx = center
        h, w = roi_size
        img_h, img_w = image_size

        y1 = max(0, cy - h // 2)
        y2 = min(img_h, cy + h // 2)
        x1 = max(0, cx - w // 2)
        x2 = min(img_w, cx + w // 2)

        return (y1, x1, y2, x2)

    def cleanup(self):
        """Limpia hooks y libera recursos."""
        self.hook.clear()
        print("🧹 NeuronAnalyzer limpiado")


# ===================================================================
# FUNCIONES DE UTILIDAD
# ===================================================================

def create_overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Crea un overlay de heatmap sobre imagen.

    Args:
        image: Array [H, W, 3] en [0, 1]
        heatmap: Array [H, W] en [0, 1]
        alpha: Transparencia del heatmap
        colormap: Nombre del colormap de matplotlib

    Returns:
        Imagen con overlay [H, W, 3] en [0, 1]
    """
    import matplotlib.pyplot as plt

    # Aplicar colormap al heatmap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # [H, W, 3]

    # Combinar
    overlay = image * (1 - alpha) + heatmap_colored * alpha

    return np.clip(overlay, 0, 1)


# ===================================================================
# TESTING
# ===================================================================

if __name__ == "__main__":
    print("🧪 Testing NeuronAnalyzer...\n")

    # Cargar modelo de prueba
    print("1️⃣ Cargando modelo AlexNet...")
    from torchvision import models
    model = models.alexnet(pretrained=False)
    model.eval()

    # Crear analyzer
    print("\n2️⃣ Creando NeuronAnalyzer...")
    analyzer = NeuronAnalyzer(model, 'features.0')

    # Crear imagen de prueba
    print("\n3️⃣ Creando imagen de prueba...")
    test_img = torch.randn(1, 3, 224, 224)

    # Extraer activaciones
    print("\n4️⃣ Extrayendo activaciones...")
    activations = analyzer.extract_activations(test_img)
    print(f"   Shape: {activations.shape}")

    # Estadísticas
    print("\n5️⃣ Calculando estadísticas...")
    stats = analyzer.compute_neuron_statistics(activations)
    print(f"   Neuronas analizadas: {len(stats)}")
    print(f"   Ejemplo neurona 0: {stats[0]}")

    # Ranking
    print("\n6️⃣ Rankeando neuronas...")
    top_neurons = analyzer.rank_neurons(stats, top_k=5)
    print(f"   Top 5 neuronas: {top_neurons}")

    # Heatmap
    print("\n7️⃣ Generando heatmap...")
    heatmap = analyzer.compute_heatmap(activations)
    print(f"   Heatmap shape: {heatmap.shape}")
    print(f"   Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    # Cleanup
    print("\n8️⃣ Limpiando...")
    analyzer.cleanup()

    print("\n✅ Testing completado!")
