"""
============================================================================
ACTIVATION HOOKS - Sistema de Hooks para Capturar Activaciones
============================================================================

Este módulo proporciona funcionalidades para:
- Registrar hooks en capas específicas de un modelo
- Capturar activaciones durante el forward pass
- Almacenar y gestionar activaciones de múltiples capas
- Extraer gradientes (para análisis posteriores)
- Utilidades para análisis de activaciones

Clase Principal:
    ActivationHook: Sistema de hooks para capturar activaciones

Uso típico:
    from src.utils.hooks import ActivationHook
    
    # Registrar hooks
    hook = ActivationHook(model, ['conv1', 'layer1.0.conv1'])
    hook.register_hooks()
    
    # Forward pass
    output = model(input)
    
    # Obtener activaciones
    activations = hook.get_activations()

Autor: Proyecto Módulo III
Fecha: 2025-01-15
============================================================================
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Tuple
from collections import OrderedDict
import logging
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActivationHook:
    """
    Clase para registrar hooks y capturar activaciones de capas intermedias.

    Esta clase permite:
    - Registrar hooks en capas específicas
    - Capturar activaciones automáticamente durante forward pass
    - Almacenar activaciones para múltiples capas
    - Gestionar y limpiar hooks
    - Análisis de activaciones

    Attributes:
        model (nn.Module): Modelo de PyTorch
        target_layers (List[str]): Lista de nombres de capas objetivo
        activations (Dict): Diccionario con activaciones capturadas
        hooks (List): Lista de handles de hooks registrados

    Example:
        >>> hook = ActivationHook(model, ['conv1', 'layer1.0.conv1'])
        >>> hook.register_hooks()
        >>> output = model(input_tensor)
        >>> activations = hook.get_activations()
        >>> print(activations['conv1'].shape)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[str]] = None
    ):
        """
        Inicializa el sistema de hooks.

        Args:
            model: Modelo de PyTorch donde registrar hooks
            target_layers: Lista de nombres de capas donde capturar activaciones.
                          Si None, registra hooks en todas las capas Conv2d.

        Example:
            >>> hook = ActivationHook(model, ['conv1', 'layer4.1.conv2'])
        """
        self.model = model
        self.target_layers = target_layers
        self.activations = OrderedDict()
        self.gradients = OrderedDict()
        self.hooks = []

        # Si no se especifican capas, usar todas las Conv2d
        if self.target_layers is None:
            self.target_layers = self._get_all_conv_layers()

        logger.info(
            f"ActivationHook inicializado con {len(self.target_layers)} capas objetivo")

    def _get_all_conv_layers(self) -> List[str]:
        """
        Obtiene los nombres de todas las capas convolucionales del modelo.

        Returns:
            Lista de nombres de capas Conv2d
        """
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and name != '':
                conv_layers.append(name)

        logger.info(f"Encontradas {len(conv_layers)} capas convolucionales")
        return conv_layers

    def _get_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """
        Obtiene una capa específica por su nombre.

        Args:
            layer_name: Nombre de la capa (e.g., 'layer1.0.conv1')

        Returns:
            El módulo correspondiente o None si no existe
        """
        # Dividir el nombre por puntos
        parts = layer_name.split('.')

        # Navegar por la estructura del modelo
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                logger.warning(
                    f"Capa '{layer_name}' no encontrada en el modelo")
                return None

        return module

    def _create_forward_hook(self, layer_name: str) -> Callable:
        """
        Crea una función hook para capturar activaciones.

        Args:
            layer_name: Nombre de la capa

        Returns:
            Función hook
        """
        def hook_fn(module, input, output):
            """
            Hook function que se ejecuta durante el forward pass.

            Args:
                module: El módulo donde está registrado el hook
                input: Input al módulo (tupla)
                output: Output del módulo
            """
            # Almacenar activación (detached para no interferir con gradientes)
            self.activations[layer_name] = output.detach()

        return hook_fn

    def _create_backward_hook(self, layer_name: str) -> Callable:
        """
        Crea una función hook para capturar gradientes.

        Args:
            layer_name: Nombre de la capa

        Returns:
            Función hook
        """
        def hook_fn(module, grad_input, grad_output):
            """
            Hook function que se ejecuta durante el backward pass.

            Args:
                module: El módulo donde está registrado el hook
                grad_input: Gradiente del input (tupla)
                grad_output: Gradiente del output (tupla)
            """
            # Almacenar gradiente
            self.gradients[layer_name] = grad_output[0].detach()

        return hook_fn

    def register_hooks(self, capture_gradients: bool = False):
        """
        Registra hooks en las capas objetivo.

        Args:
            capture_gradients: Si True, también captura gradientes (backward hooks)

        Example:
            >>> hook.register_hooks()
            >>> # Ahora los hooks capturarán activaciones automáticamente
        """
        logger.info("Registrando hooks...")

        registered_count = 0

        for layer_name in self.target_layers:
            # Obtener la capa
            layer = self._get_layer_by_name(layer_name)

            if layer is None:
                logger.warning(f"Saltando capa '{layer_name}' (no encontrada)")
                continue

            # Registrar forward hook
            forward_hook = layer.register_forward_hook(
                self._create_forward_hook(layer_name)
            )
            self.hooks.append(forward_hook)
            registered_count += 1

            # Registrar backward hook si se solicita
            if capture_gradients:
                backward_hook = layer.register_full_backward_hook(
                    self._create_backward_hook(layer_name)
                )
                self.hooks.append(backward_hook)

        logger.info(f"✅ {registered_count} hooks registrados exitosamente")

        if capture_gradients:
            logger.info(f"   (También capturando gradientes)")

    def remove_hooks(self):
        """
        Remueve todos los hooks registrados.

        Example:
            >>> hook.remove_hooks()
            >>> # Ahora el modelo funciona normalmente sin capturar activaciones
        """
        logger.info("Removiendo hooks...")

        for hook_handle in self.hooks:
            hook_handle.remove()

        self.hooks.clear()
        logger.info(f"✅ Todos los hooks removidos")

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """
        Obtiene las activaciones capturadas.

        Returns:
            Diccionario {layer_name: activation_tensor}

        Example:
            >>> activations = hook.get_activations()
            >>> conv1_act = activations['conv1']
            >>> print(conv1_act.shape)  # [batch_size, channels, H, W]
        """
        return self.activations

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Obtiene los gradientes capturados.

        Returns:
            Diccionario {layer_name: gradient_tensor}

        Note:
            Requiere que se hayan registrado hooks con capture_gradients=True
            y que se haya ejecutado backward pass.

        Example:
            >>> hook.register_hooks(capture_gradients=True)
            >>> output = model(input)
            >>> loss = criterion(output, target)
            >>> loss.backward()
            >>> gradients = hook.get_gradients()
        """
        return self.gradients

    def clear_activations(self):
        """
        Limpia las activaciones almacenadas.

        Útil para liberar memoria entre diferentes forward passes.

        Example:
            >>> # Forward pass 1
            >>> output1 = model(input1)
            >>> act1 = hook.get_activations()
            >>> 
            >>> # Limpiar para el siguiente
            >>> hook.clear_activations()
            >>> 
            >>> # Forward pass 2
            >>> output2 = model(input2)
            >>> act2 = hook.get_activations()
        """
        self.activations.clear()
        self.gradients.clear()

    def get_activation_shapes(self) -> Dict[str, Tuple]:
        """
        Obtiene los shapes de las activaciones capturadas.

        Returns:
            Diccionario {layer_name: shape_tuple}

        Example:
            >>> shapes = hook.get_activation_shapes()
            >>> for layer, shape in shapes.items():
            ...     print(f"{layer}: {shape}")
        """
        shapes = {}
        for layer_name, activation in self.activations.items():
            shapes[layer_name] = tuple(activation.shape)
        return shapes

    def get_activation_statistics(self) -> Dict[str, Dict]:
        """
        Calcula estadísticas de las activaciones capturadas.

        Returns:
            Diccionario con estadísticas por capa:
                - mean: Media de la activación
                - std: Desviación estándar
                - min: Valor mínimo
                - max: Valor máximo
                - sparsity: Proporción de valores == 0
                - shape: Shape del tensor

        Example:
            >>> stats = hook.get_activation_statistics()
            >>> print(f"Conv1 mean: {stats['conv1']['mean']:.4f}")
        """
        statistics = {}

        for layer_name, activation in self.activations.items():
            act_flat = activation.cpu().flatten().numpy()

            stats = {
                'mean': float(np.mean(act_flat)),
                'std': float(np.std(act_flat)),
                'min': float(np.min(act_flat)),
                'max': float(np.max(act_flat)),
                'sparsity': float((act_flat == 0).sum() / len(act_flat)),
                'active_neurons': int((act_flat > 0).sum()),
                'total_neurons': int(len(act_flat)),
                'shape': tuple(activation.shape)
            }

            statistics[layer_name] = stats

        return statistics

    def get_layer_names(self) -> List[str]:
        """
        Obtiene la lista de nombres de capas objetivo.

        Returns:
            Lista de nombres de capas

        Example:
            >>> layers = hook.get_layer_names()
            >>> print(layers)
            ['conv1', 'layer1.0.conv1', 'layer2.0.conv1', ...]
        """
        return self.target_layers

    def has_activations(self) -> bool:
        """
        Verifica si hay activaciones capturadas.

        Returns:
            True si hay activaciones, False en caso contrario

        Example:
            >>> if hook.has_activations():
            ...     activations = hook.get_activations()
        """
        return len(self.activations) > 0

    def get_activation_for_layer(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Obtiene la activación de una capa específica.

        Args:
            layer_name: Nombre de la capa

        Returns:
            Tensor de activación o None si no existe

        Example:
            >>> conv1_act = hook.get_activation_for_layer('conv1')
            >>> if conv1_act is not None:
            ...     print(conv1_act.shape)
        """
        return self.activations.get(layer_name, None)

    def save_activations(self, filepath: str):
        """
        Guarda las activaciones en un archivo.

        Args:
            filepath: Ruta del archivo donde guardar

        Example:
            >>> hook.save_activations('activations.pth')
        """
        # Convertir a CPU para guardar
        activations_cpu = {
            name: act.cpu() for name, act in self.activations.items()
        }

        torch.save(activations_cpu, filepath)
        logger.info(f"✅ Activaciones guardadas en {filepath}")

    def load_activations(self, filepath: str):
        """
        Carga activaciones desde un archivo.

        Args:
            filepath: Ruta del archivo a cargar

        Example:
            >>> hook.load_activations('activations.pth')
        """
        self.activations = torch.load(filepath)
        logger.info(f"✅ Activaciones cargadas desde {filepath}")

    def __repr__(self) -> str:
        """Representación en string del ActivationHook."""
        num_hooks = len(self.hooks)
        num_activations = len(self.activations)
        return (f"ActivationHook(layers={len(self.target_layers)}, "
                f"hooks_registered={num_hooks}, "
                f"activations_captured={num_activations})")


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_all_layer_names(model: nn.Module) -> List[str]:
    """
    Obtiene todos los nombres de capas del modelo.

    Args:
        model: Modelo de PyTorch

    Returns:
        Lista de nombres de todas las capas

    Example:
        >>> layer_names = get_all_layer_names(model)
        >>> print(f"Total layers: {len(layer_names)}")
    """
    layer_names = []
    for name, _ in model.named_modules():
        if name != '':  # Saltar el módulo raíz
            layer_names.append(name)
    return layer_names


def get_layer_types(model: nn.Module) -> Dict[str, List[str]]:
    """
    Agrupa las capas del modelo por tipo.

    Args:
        model: Modelo de PyTorch

    Returns:
        Diccionario {tipo_de_capa: [nombres_de_capas]}

    Example:
        >>> layer_types = get_layer_types(model)
        >>> print(f"Conv2d layers: {layer_types['Conv2d']}")
    """
    layer_types = {}

    for name, module in model.named_modules():
        if name == '':
            continue

        module_type = type(module).__name__

        if module_type not in layer_types:
            layer_types[module_type] = []

        layer_types[module_type].append(name)

    return layer_types


def compare_activations(
    activations1: Dict[str, torch.Tensor],
    activations2: Dict[str, torch.Tensor]
) -> Dict[str, Dict]:
    """
    Compara dos conjuntos de activaciones.

    Args:
        activations1: Primer conjunto de activaciones
        activations2: Segundo conjunto de activaciones

    Returns:
        Diccionario con métricas de comparación por capa

    Example:
        >>> output1 = model(input1)
        >>> act1 = hook.get_activations()
        >>> hook.clear_activations()
        >>> output2 = model(input2)
        >>> act2 = hook.get_activations()
        >>> comparison = compare_activations(act1, act2)
    """
    comparison = {}

    # Capas en común
    common_layers = set(activations1.keys()) & set(activations2.keys())

    for layer_name in common_layers:
        act1 = activations1[layer_name]
        act2 = activations2[layer_name]

        # Calcular diferencias
        diff = (act1 - act2).abs()

        comparison[layer_name] = {
            'mean_abs_diff': float(diff.mean()),
            'max_abs_diff': float(diff.max()),
            'cosine_similarity': float(
                torch.nn.functional.cosine_similarity(
                    act1.flatten(),
                    act2.flatten(),
                    dim=0
                )
            ),
            'correlation': float(
                torch.corrcoef(
                    torch.stack([act1.flatten(), act2.flatten()])
                )[0, 1]
            )
        }

    return comparison


def find_dead_neurons(
    activations: Dict[str, torch.Tensor],
    threshold: float = 0.0
) -> Dict[str, List[int]]:
    """
    Identifica neuronas "muertas" (que nunca se activan).

    Args:
        activations: Diccionario de activaciones
        threshold: Umbral para considerar una neurona como "muerta"

    Returns:
        Diccionario {layer_name: [indices_de_neuronas_muertas]}

    Example:
        >>> dead = find_dead_neurons(activations)
        >>> for layer, neurons in dead.items():
        ...     print(f"{layer}: {len(neurons)} neuronas muertas")
    """
    dead_neurons = {}

    for layer_name, activation in activations.items():
        # Activation shape: [batch, channels, height, width]
        if activation.dim() == 4:
            # Calcular activación máxima por canal
            max_per_channel = activation.max(dim=0)[0]  # [channels, H, W]
            max_per_channel = max_per_channel.view(
                max_per_channel.size(0), -1).max(dim=1)[0]

            # Encontrar canales con activación <= threshold
            dead_indices = (max_per_channel <=
                            threshold).nonzero().flatten().tolist()

            if dead_indices:
                dead_neurons[layer_name] = dead_indices

    return dead_neurons


def analyze_sparsity(activations: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Analiza la sparsity de las activaciones.

    Args:
        activations: Diccionario de activaciones

    Returns:
        Diccionario {layer_name: sparsity_percentage}

    Example:
        >>> sparsity = analyze_sparsity(activations)
        >>> for layer, sparse in sparsity.items():
        ...     print(f"{layer}: {sparse:.1f}% sparse")
    """
    sparsity = {}

    for layer_name, activation in activations.items():
        act_flat = activation.flatten()
        zeros = (act_flat == 0).sum().item()
        total = act_flat.numel()
        sparsity[layer_name] = (zeros / total) * 100

    return sparsity


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del ActivationHook.
    """
    print("🪝 EJEMPLO DE USO: ActivationHook")
    print("="*70)

    # 1. Crear un modelo simple para demostración
    print("\n🔄 Creando modelo de ejemplo...")
    import torchvision.models as models

    model = models.resnet18(weights=None)
    model.eval()

    # 2. Definir capas de interés
    target_layers = [
        'conv1',
        'layer1.0.conv1',
        'layer2.0.conv1',
        'layer3.0.conv1',
        'layer4.0.conv1'
    ]

    print(f"\n📋 Capas objetivo: {len(target_layers)}")
    for layer in target_layers:
        print(f"  • {layer}")

    # 3. Crear hook y registrar
    print(f"\n🪝 Registrando hooks...")
    hook = ActivationHook(model, target_layers)
    hook.register_hooks()

    # 4. Forward pass
    print(f"\n⚡ Realizando forward pass...")
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    # 5. Obtener activaciones
    print(f"\n📦 Activaciones capturadas:")
    activations = hook.get_activations()

    for layer_name, activation in activations.items():
        print(f"  • {layer_name}: {tuple(activation.shape)}")

    # 6. Estadísticas
    print(f"\n📊 Estadísticas de activaciones:")
    stats = hook.get_activation_statistics()

    for layer_name, layer_stats in stats.items():
        print(f"\n  {layer_name}:")
        print(f"    Mean: {layer_stats['mean']:.4f}")
        print(f"    Std:  {layer_stats['std']:.4f}")
        print(f"    Sparsity: {layer_stats['sparsity']*100:.1f}%")

    # 7. Limpiar
    print(f"\n🧹 Limpiando hooks...")
    hook.remove_hooks()

    print("\n✅ Ejemplo completado exitosamente")
