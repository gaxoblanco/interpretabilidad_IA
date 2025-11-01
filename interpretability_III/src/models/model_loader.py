"""
============================================================================
MODEL LOADER - Carga y Análisis de Modelos Pre-entrenados
============================================================================

Este módulo proporciona funcionalidades para:
- Cargar modelos pre-entrenados de torchvision (ResNet, VGG, etc.)
- Analizar arquitectura del modelo (capas, parámetros, estructura)
- Extraer información detallada de bloques residuales
- Gestión de device (CPU/GPU)
- Configuración del modelo para inferencia o fine-tuning

Clase Principal:
    ModelLoader: Carga y configura modelos para interpretabilidad

Uso típico:
    from src.models.model_loader import ModelLoader
    
    # Cargar modelo
    loader = ModelLoader(model_name='resnet18', pretrained=True)
    model = loader.load_model()
    
    # Analizar arquitectura
    arch_info = loader.get_architecture_info()
    layers_info = loader.get_layers_info()

Autor: Proyecto Módulo III
Fecha: 2025-01-15
============================================================================
"""

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Clase para cargar y analizar modelos pre-entrenados de torchvision.

    Esta clase facilita:
    - Carga de modelos pre-entrenados
    - Análisis de arquitectura
    - Gestión de device (CPU/GPU)
    - Extracción de información de capas

    Attributes:
        model_name (str): Nombre del modelo a cargar
        pretrained (bool): Si se debe cargar con pesos pre-entrenados
        num_classes (int): Número de clases de salida
        device (torch.device): Device donde se cargará el modelo
        model (nn.Module): El modelo cargado

    Example:
        >>> loader = ModelLoader('resnet18', pretrained=True)
        >>> model = loader.load_model()
        >>> info = loader.get_architecture_info()
        >>> print(f"Total params: {info['total_params']:,}")
    """

    # Mapeo de nombres de modelos soportados
    SUPPORTED_MODELS = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'alexnet': models.alexnet,
        'densenet121': models.densenet121,
        'mobilenet_v2': models.mobilenet_v2,
    }

    def __init__(
        self,
        model_name: str = 'resnet18',
        pretrained: bool = True,
        num_classes: int = 1000,
        device: Optional[str] = None
    ):
        """
        Inicializa el cargador de modelos.

        Args:
            model_name: Nombre del modelo ('resnet18', 'resnet50', etc.)
            pretrained: Si True, carga pesos pre-entrenados en ImageNet
            num_classes: Número de clases de salida (default: 1000 para ImageNet)
            device: Device específico ('cuda', 'cpu') o None para auto-detección

        Raises:
            ValueError: Si el modelo no está soportado
        """
        self.model_name = model_name.lower()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model = None

        # Validar modelo soportado
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Modelo '{model_name}' no soportado. "
                f"Modelos disponibles: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # Configurar device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"ModelLoader inicializado: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Pre-trained: {self.pretrained}")

    def load_model(self) -> nn.Module:
        """
        Carga el modelo especificado con o sin pesos pre-entrenados.

        Returns:
            El modelo cargado en el device especificado

        Example:
            >>> loader = ModelLoader('resnet18')
            >>> model = loader.load_model()
            >>> print(type(model))
            <class 'torchvision.models.resnet.ResNet'>
        """
        logger.info(f"Cargando modelo {self.model_name}...")

        # Obtener función de construcción del modelo
        model_fn = self.SUPPORTED_MODELS[self.model_name]

        # Cargar modelo con o sin pesos
        if self.pretrained:
            logger.info("Descargando pesos pre-entrenados de ImageNet...")
            self.model = model_fn(weights='IMAGENET1K_V1')
        else:
            logger.info("Inicializando modelo sin pesos pre-entrenados...")
            self.model = model_fn(weights=None)

        # Mover modelo al device
        self.model = self.model.to(self.device)

        # Configurar modelo en modo evaluación por defecto
        self.model.eval()

        logger.info(f"✅ Modelo {self.model_name} cargado exitosamente")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Parámetros: {self._count_parameters():,}")

        return self.model

    def _count_parameters(self) -> int:
        """
        Cuenta el número total de parámetros del modelo.

        Returns:
            Número total de parámetros
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def get_architecture_info(self) -> Dict:
        """
        Obtiene información general de la arquitectura del modelo.

        Returns:
            Diccionario con información de la arquitectura:
                - total_params: Total de parámetros
                - trainable_params: Parámetros entrenables
                - frozen_params: Parámetros congelados
                - model_size_mb: Tamaño del modelo en MB
                - num_layers: Número de capas
                - params_by_type: Distribución de parámetros por tipo

        Example:
            >>> info = loader.get_architecture_info()
            >>> print(f"Size: {info['model_size_mb']:.2f} MB")
        """
        if self.model is None:
            raise ValueError(
                "Modelo no cargado. Ejecutar load_model() primero.")

        # Contar parámetros
        total_params = 0
        trainable_params = 0
        frozen_params = 0

        for param in self.model.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            else:
                frozen_params += num_params

        # Calcular tamaño en MB (asumiendo float32 = 4 bytes)
        model_size_mb = (total_params * 4) / (1024 ** 2)

        # Contar número de capas
        num_layers = len(list(self.model.modules()))

        # Distribución de parámetros por tipo de capa
        params_by_type = {}
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            module_params = sum(p.numel()
                                for p in module.parameters(recurse=False))

            if module_params > 0:
                if module_type not in params_by_type:
                    params_by_type[module_type] = 0
                params_by_type[module_type] += module_params

        return {
            'model_name': self.model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'model_size_mb': model_size_mb,
            'num_layers': num_layers,
            'params_by_type': params_by_type
        }

    def get_layers_info(self) -> List[Dict]:
        """
        Obtiene información detallada de cada capa del modelo.

        Returns:
            Lista de diccionarios con información de cada capa:
                - name: Nombre de la capa
                - type: Tipo de capa (Conv2d, Linear, etc.)
                - params: Número de parámetros
                - trainable: Si los parámetros son entrenables
                - output_shape: Shape de salida (si está disponible)

        Example:
            >>> layers = loader.get_layers_info()
            >>> for layer in layers[:5]:
            ...     print(f"{layer['name']}: {layer['params']:,} params")
        """
        if self.model is None:
            raise ValueError(
                "Modelo no cargado. Ejecutar load_model() primero.")

        layers_info = []

        for name, module in self.model.named_modules():
            # Saltar el módulo raíz
            if name == '':
                continue

            # IMPORTANTE: recurse=False significa que NO cuenta
            # parámetros de sub-módulos (hijos)

            # Contar parámetros de esta capa específica (sin recursión)
            params = sum(p.numel() for p in module.parameters(recurse=False))

            # Verificar si los parámetros son entrenables
            trainable = any(
                p.requires_grad for p in module.parameters(recurse=False))

            # Obtener tipo de módulo
            module_type = type(module).__name__

            # Intentar obtener output_shape si es Conv2d o Linear
            output_shape = self._get_output_shape(module)

            layer_info = {
                'name': name,
                'type': module_type,
                'params': params,
                'trainable': trainable,
                'output_shape': output_shape
            }

            layers_info.append(layer_info)

        return layers_info

    def _get_output_shape(self, module) -> Optional[Tuple]:
        """
        Intenta obtener el shape de salida de un módulo.

        Args:
            module: Módulo de PyTorch

        Returns:
            Tuple con el shape o None si no está disponible
        """
        if isinstance(module, nn.Conv2d):
            return (module.out_channels, 'H', 'W')
        elif isinstance(module, nn.Linear):
            return (module.out_features,)
        elif isinstance(module, nn.BatchNorm2d):
            return (module.num_features, 'H', 'W')
        else:
            return None

    def get_residual_blocks_info(self) -> List[Dict]:
        """
        Obtiene información específica de los bloques residuales (para ResNet).

        Returns:
            Lista de diccionarios con información de cada bloque residual:
                - name: Nombre del bloque
                - num_layers: Número de capas en el bloque
                - params: Número de parámetros
                - in_channels: Canales de entrada
                - out_channels: Canales de salida
                - stride: Stride del bloque
                - has_downsample: Si tiene capa de downsampling

        Note:
            Esta función está diseñada específicamente para modelos ResNet.
            Para otros modelos, retornará una lista vacía.

        Example:
            >>> blocks = loader.get_residual_blocks_info()
            >>> for block in blocks:
            ...     print(f"{block['name']}: {block['in_channels']} -> {block['out_channels']}")
        """
        if self.model is None:
            raise ValueError(
                "Modelo no cargado. Ejecutar load_model() primero.")

        if not self.model_name.startswith('resnet'):
            logger.warning(
                f"get_residual_blocks_info() solo funciona con ResNet, no con {self.model_name}")
            return []

        blocks_info = []

        # ResNet tiene 4 layers (layer1, layer2, layer3, layer4)
        for layer_idx in range(1, 5):
            layer_name = f'layer{layer_idx}'

            if not hasattr(self.model, layer_name):
                continue

            layer = getattr(self.model, layer_name)

            # Cada layer tiene varios bloques BasicBlock o Bottleneck
            for block_idx, block in enumerate(layer):
                # Nombre del bloque
                block_name = f'{layer_name}.{block_idx}'

                # Contar parámetros del bloque
                params = sum(p.numel() for p in block.parameters())

                # Número de capas en el bloque
                num_layers = len([m for m in block.modules()
                                 if isinstance(m, (nn.Conv2d, nn.Linear))])

                # Información de canales
                # El primer conv2d del bloque tiene la info de in/out channels
                first_conv = None
                for module in block.modules():
                    if isinstance(module, nn.Conv2d):
                        first_conv = module
                        break

                if first_conv is not None:
                    in_channels = first_conv.in_channels
                    out_channels = first_conv.out_channels
                    stride = first_conv.stride[0] if isinstance(
                        first_conv.stride, tuple) else first_conv.stride
                else:
                    in_channels = None
                    out_channels = None
                    stride = None

                # Verificar si tiene downsample
                has_downsample = hasattr(
                    block, 'downsample') and block.downsample is not None

                block_info = {
                    'name': block_name,
                    'num_layers': num_layers,
                    'params': params,
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'stride': stride,
                    'has_downsample': has_downsample
                }

                blocks_info.append(block_info)

        return blocks_info

    def get_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """
        Obtiene una capa específica por su nombre.

        Args:
            layer_name: Nombre de la capa (e.g., 'layer1.0.conv1')

        Returns:
            El módulo correspondiente o None si no existe

        Example:
            >>> conv1 = loader.get_layer_by_name('conv1')
            >>> print(conv1.weight.shape)
        """
        if self.model is None:
            raise ValueError(
                "Modelo no cargado. Ejecutar load_model() primero.")

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

    def freeze_layers(self, layer_names: Optional[List[str]] = None):
        """
        Congela (freeze) capas específicas del modelo.

        Args:
            layer_names: Lista de nombres de capas a congelar.
                        Si None, congela todas las capas.

        Example:
            >>> loader.freeze_layers(['conv1', 'layer1'])
            >>> # Ahora conv1 y layer1 no se actualizarán durante entrenamiento
        """
        if self.model is None:
            raise ValueError(
                "Modelo no cargado. Ejecutar load_model() primero.")

        if layer_names is None:
            # Congelar todas las capas
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Todas las capas congeladas")
        else:
            # Congelar capas específicas
            for layer_name in layer_names:
                layer = self.get_layer_by_name(layer_name)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = False
                    logger.info(f"Capa '{layer_name}' congelada")

    def unfreeze_layers(self, layer_names: Optional[List[str]] = None):
        """
        Descongela (unfreeze) capas específicas del modelo.

        Args:
            layer_names: Lista de nombres de capas a descongelar.
                        Si None, descongela todas las capas.

        Example:
            >>> loader.unfreeze_layers(['fc'])
            >>> # Ahora solo la capa fc se actualizará durante entrenamiento
        """
        if self.model is None:
            raise ValueError(
                "Modelo no cargado. Ejecutar load_model() primero.")

        if layer_names is None:
            # Descongelar todas las capas
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info("Todas las capas descongeladas")
        else:
            # Descongelar capas específicas
            for layer_name in layer_names:
                layer = self.get_layer_by_name(layer_name)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True
                    logger.info(f"Capa '{layer_name}' descongelada")

    def set_eval_mode(self):
        """
        Configura el modelo en modo evaluación.

        Esto desactiva dropout y batch normalization en modo entrenamiento.
        """
        if self.model is None:
            raise ValueError(
                "Modelo no cargado. Ejecutar load_model() primero.")

        self.model.eval()
        logger.info("Modelo configurado en modo evaluación")

    def set_train_mode(self):
        """
        Configura el modelo en modo entrenamiento.

        Esto activa dropout y batch normalization en modo entrenamiento.
        """
        if self.model is None:
            raise ValueError(
                "Modelo no cargado. Ejecutar load_model() primero.")

        self.model.train()
        logger.info("Modelo configurado en modo entrenamiento")

    def get_model_summary(self) -> str:
        """
        Genera un resumen legible del modelo.

        Returns:
            String con el resumen del modelo

        Example:
            >>> summary = loader.get_model_summary()
            >>> print(summary)
        """
        if self.model is None:
            raise ValueError(
                "Modelo no cargado. Ejecutar load_model() primero.")

        arch_info = self.get_architecture_info()

        summary = f"""
{'='*70}
RESUMEN DEL MODELO: {self.model_name.upper()}
{'='*70}

Configuración:
  • Pre-entrenado: {self.pretrained}
  • Device: {self.device}
  • Número de clases: {self.num_classes}

Parámetros:
  • Total: {arch_info['total_params']:,}
  • Entrenables: {arch_info['trainable_params']:,}
  • Congelados: {arch_info['frozen_params']:,}
  • Tamaño: {arch_info['model_size_mb']:.2f} MB

Arquitectura:
  • Número de capas: {arch_info['num_layers']}
  • Tipos de capas: {len(arch_info['params_by_type'])}

{'='*70}
"""
        return summary

    def __repr__(self) -> str:
        """Representación en string del ModelLoader."""
        return (f"ModelLoader(model='{self.model_name}', "
                f"pretrained={self.pretrained}, "
                f"device='{self.device}')")


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def list_available_models() -> List[str]:
    """
    Lista todos los modelos disponibles para cargar.

    Returns:
        Lista de nombres de modelos soportados

    Example:
        >>> models = list_available_models()
        >>> print(models)
        ['resnet18', 'resnet34', 'resnet50', ...]
    """
    return list(ModelLoader.SUPPORTED_MODELS.keys())


def get_model_info(model_name: str) -> Dict:
    """
    Obtiene información básica de un modelo sin cargarlo.

    Args:
        model_name: Nombre del modelo

    Returns:
        Diccionario con información del modelo

    Example:
        >>> info = get_model_info('resnet18')
        >>> print(info['architecture'])
    """
    model_info = {
        'resnet18': {
            'architecture': 'ResNet',
            'depth': 18,
            'params_millions': 11.7,
            'top1_accuracy': 69.8,
            'top5_accuracy': 89.1
        },
        'resnet50': {
            'architecture': 'ResNet',
            'depth': 50,
            'params_millions': 25.6,
            'top1_accuracy': 76.1,
            'top5_accuracy': 92.9
        },
        # Agregar más modelos según sea necesario
    }

    return model_info.get(model_name.lower(), {
        'architecture': 'Unknown',
        'info': 'No information available'
    })


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del ModelLoader.
    """
    print("🧠 EJEMPLO DE USO: ModelLoader")
    print("="*70)

    # 1. Listar modelos disponibles
    print("\n📋 Modelos disponibles:")
    for model_name in list_available_models():
        print(f"  • {model_name}")

    # 2. Cargar modelo
    print("\n🔄 Cargando ResNet-18...")
    loader = ModelLoader(model_name='resnet18', pretrained=True)
    model = loader.load_model()

    # 3. Mostrar resumen
    print(loader.get_model_summary())

    # 4. Obtener información de arquitectura
    arch_info = loader.get_architecture_info()
    print(f"\n📊 Parámetros totales: {arch_info['total_params']:,}")

    # 5. Obtener información de capas
    layers = loader.get_layers_info()
    print(f"\n🏗️ Número de capas: {len(layers)}")

    # 6. Información de bloques residuales
    blocks = loader.get_residual_blocks_info()
    print(f"\n🔹 Bloques residuales: {len(blocks)}")

    print("\n✅ Ejemplo completado exitosamente")
