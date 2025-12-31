"""
===================================================================
MODEL_MANAGER.PY - Gestión de Modelos Pre-entrenados
===================================================================

Este módulo maneja la carga, configuración y gestión de modelos
de deep learning pre-entrenados.

Funcionalidades principales:
1. Carga de modelos (AlexNet, ResNet, VGG, etc.)
2. Extracción de capas convolucionales
3. Información detallada de capas (canales, dimensiones)
4. Caché de modelos para performance

Uso:
    manager = ModelManager()
    model = manager.load_model('alexnet')
    layers = manager.get_conv_layers(model)
===================================================================
"""

from config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_DEVICE,
    ENABLE_MODEL_CACHE
)
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import sys

# Importar configuración
sys.path.append('..')


class ModelManager:
    """
    Gestor de modelos de deep learning pre-entrenados.

    Maneja la carga, caché y consulta de información sobre modelos
    y sus capas convolucionales.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Inicializa el gestor de modelos.

        Args:
            device: Dispositivo para computación ('cpu', 'cuda', etc.)
                   Si es None, usa DEFAULT_DEVICE de config
        """
        # Determinar device
        if device is None:
            device = DEFAULT_DEVICE

        # Verificar disponibilidad de CUDA
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA no disponible, usando CPU")
            device = 'cpu'

        self.device = torch.device(device)

        # Caché de modelos cargados
        self._model_cache: Dict[str, nn.Module] = {}

        # Información de capas (se construye al cargar modelo)
        self._layer_info_cache: Dict[str, Dict] = {}

        print(f"✅ ModelManager inicializado")
        print(f"   Device: {self.device}")
        print(f"   Caché habilitado: {ENABLE_MODEL_CACHE}")

    def get_available_models(self) -> List[str]:
        """
        Retorna lista de modelos disponibles.

        Returns:
            Lista con nombres de modelos soportados
        """
        return list(AVAILABLE_MODELS.keys())

    def get_model_info(self, model_name: str) -> Dict:
        """
        Obtiene información detallada de un modelo.

        Args:
            model_name: Nombre del modelo (ej: 'alexnet')

        Returns:
            Diccionario con información del modelo

        Raises:
            ValueError: Si el modelo no está soportado
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Modelo '{model_name}' no soportado. "
                f"Disponibles: {self.get_available_models()}"
            )

        return AVAILABLE_MODELS[model_name]

    def load_model(
        self,
        model_name: str,
        pretrained: bool = True,
        force_reload: bool = False
    ) -> nn.Module:
        """
        Carga un modelo pre-entrenado.

        Args:
            model_name: Nombre del modelo ('alexnet', 'resnet18', 'vgg16')
            pretrained: Si cargar pesos pre-entrenados de ImageNet
            force_reload: Forzar recarga incluso si está en caché

        Returns:
            Modelo de PyTorch en modo evaluación

        Raises:
            ValueError: Si el modelo no está soportado
        """
        # Validar modelo
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Modelo '{model_name}' no soportado. "
                f"Disponibles: {self.get_available_models()}"
            )

        # Verificar caché
        if ENABLE_MODEL_CACHE and not force_reload:
            if model_name in self._model_cache:
                print(f"📦 Modelo '{model_name}' cargado desde caché")
                return self._model_cache[model_name]

        print(f"🔄 Cargando modelo '{model_name}'...")
        if pretrained:
            print(f"   Descargando pesos pre-entrenados de ImageNet...")

        # Cargar modelo según tipo
        if model_name == 'alexnet':
            model = models.alexnet(pretrained=pretrained)
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
        else:
            raise ValueError(f"Modelo '{model_name}' no implementado")

        # Mover a device y poner en modo evaluación
        model = model.to(self.device)
        model.eval()

        # Guardar en caché
        if ENABLE_MODEL_CACHE:
            self._model_cache[model_name] = model

        # Contar parámetros
        num_params = sum(p.numel() for p in model.parameters())

        print(f"✅ Modelo '{model_name}' cargado exitosamente")
        print(f"   Parámetros: {num_params:,}")
        print(f"   Device: {self.device}")

        return model

    def get_conv_layers(
        self,
        model: nn.Module,
        model_name: Optional[str] = None
    ) -> List[str]:
        """
        Extrae nombres de todas las capas convolucionales del modelo.

        Args:
            model: Modelo de PyTorch
            model_name: Nombre del modelo (opcional, para mejor output)

        Returns:
            Lista ordenada de nombres de capas convolucionales
        """
        conv_layers = []

        # Recorrer todos los módulos nombrados
        for name, module in model.named_modules():
            # Verificar si es capa convolucional
            if isinstance(module, nn.Conv2d):
                # Solo agregar si tiene nombre (no submódulos vacíos)
                if name:
                    conv_layers.append(name)

        if model_name:
            print(f"\n🔍 Capas convolucionales en '{model_name}':")
            for i, layer in enumerate(conv_layers, 1):
                print(f"   {i:2d}. {layer}")

        return conv_layers

    def get_layer_info(
        self,
        model: nn.Module,
        layer_name: str
    ) -> Dict:
        """
        Obtiene información detallada sobre una capa específica.

        Args:
            model: Modelo de PyTorch
            layer_name: Nombre de la capa (ej: 'features.0')

        Returns:
            Diccionario con información:
            - 'name': Nombre de la capa
            - 'type': Tipo de módulo
            - 'num_channels': Número de canales de salida
            - 'kernel_size': Tamaño del kernel (si es Conv2d)
            - 'stride': Stride (si es Conv2d)
            - 'padding': Padding (si es Conv2d)

        Raises:
            ValueError: Si la capa no existe
        """
        # Verificar caché
        cache_key = f"{id(model)}_{layer_name}"
        if cache_key in self._layer_info_cache:
            return self._layer_info_cache[cache_key]

        # Buscar el módulo
        target_module = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Capa '{layer_name}' no encontrada en el modelo")

        # Extraer información
        info = {
            'name': layer_name,
            'type': type(target_module).__name__
        }

        # Información específica para Conv2d
        if isinstance(target_module, nn.Conv2d):
            info['num_channels'] = target_module.out_channels
            info['in_channels'] = target_module.in_channels
            info['kernel_size'] = target_module.kernel_size
            info['stride'] = target_module.stride
            info['padding'] = target_module.padding

            # Calcular feature map size aproximado (asume input 224x224)
            # Esta es una aproximación simple
            info['approx_output_size'] = self._estimate_output_size(
                input_size=224,
                kernel_size=target_module.kernel_size[0],
                stride=target_module.stride[0],
                padding=target_module.padding[0]
            )

        # Guardar en caché
        self._layer_info_cache[cache_key] = info

        return info

    def _estimate_output_size(
        self,
        input_size: int,
        kernel_size: int,
        stride: int,
        padding: int
    ) -> int:
        """
        Estima el tamaño del feature map de salida.

        Fórmula: floor((input + 2*padding - kernel_size) / stride) + 1

        Args:
            input_size: Tamaño de entrada
            kernel_size: Tamaño del kernel
            stride: Stride
            padding: Padding

        Returns:
            Tamaño estimado de salida
        """
        return ((input_size + 2 * padding - kernel_size) // stride) + 1

    def print_model_summary(
        self,
        model: nn.Module,
        model_name: Optional[str] = None
    ):
        """
        Imprime un resumen del modelo con sus capas principales.

        Args:
            model: Modelo de PyTorch
            model_name: Nombre del modelo (opcional)
        """
        print("\n" + "=" * 70)
        print(
            f"📊 RESUMEN DEL MODELO{' - ' + model_name if model_name else ''}")
        print("=" * 70)

        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)

        print(f"\n💾 Parámetros:")
        print(f"   Total: {total_params:,}")
        print(f"   Entrenables: {trainable_params:,}")

        # Capas convolucionales
        conv_layers = self.get_conv_layers(model, model_name=None)
        print(f"\n🔍 Capas Convolucionales: {len(conv_layers)}")

        # Mostrar primeras capas con detalle
        print(f"\n📋 Primeras capas:")
        for i, layer_name in enumerate(conv_layers[:5], 1):
            info = self.get_layer_info(model, layer_name)
            print(f"   {i}. {layer_name}")
            print(
                f"      Canales: {info.get('in_channels', '?')} → {info.get('num_channels', '?')}")
            if 'kernel_size' in info:
                print(
                    f"      Kernel: {info['kernel_size']}, Stride: {info['stride']}")

        if len(conv_layers) > 5:
            print(f"   ... y {len(conv_layers) - 5} capas más")

        print("\n" + "=" * 70 + "\n")

    def clear_cache(self):
        """
        Limpia el caché de modelos y libera memoria.
        """
        if self._model_cache:
            num_models = len(self._model_cache)
            self._model_cache.clear()
            self._layer_info_cache.clear()

            # Limpiar caché de CUDA si está disponible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"🧹 Caché limpiado ({num_models} modelo(s) removido(s))")
        else:
            print("ℹ️  Caché ya está vacío")


# ===================================================================
# FUNCIONES DE UTILIDAD
# ===================================================================

def get_model_and_layers(model_name: str = DEFAULT_MODEL) -> Tuple[nn.Module, List[str]]:
    """
    Función de conveniencia para cargar modelo y obtener sus capas.

    Args:
        model_name: Nombre del modelo

    Returns:
        Tupla (modelo, lista de capas convolucionales)
    """
    manager = ModelManager()
    model = manager.load_model(model_name)
    layers = manager.get_conv_layers(model, model_name)
    return model, layers


# ===================================================================
# TESTING
# ===================================================================

if __name__ == "__main__":
    print("🧪 Testing ModelManager...\n")

    # Crear manager
    manager = ModelManager()

    # Probar con AlexNet
    print("\n1️⃣ Cargando AlexNet...")
    model = manager.load_model('alexnet')

    # Obtener capas
    print("\n2️⃣ Extrayendo capas convolucionales...")
    layers = manager.get_conv_layers(model, model_name='alexnet')

    # Info de una capa específica
    print("\n3️⃣ Información de capa 'features.0'...")
    info = manager.get_layer_info(model, 'features.0')
    print(f"   Información: {info}")

    # Resumen del modelo
    print("\n4️⃣ Resumen del modelo...")
    manager.print_model_summary(model, 'alexnet')

    # Probar caché
    print("\n5️⃣ Probando caché...")
    model2 = manager.load_model('alexnet')  # Debería cargar desde caché

    print("\n✅ Testing completado!")
