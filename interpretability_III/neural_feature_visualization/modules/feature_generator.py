"""
===================================================================
FEATURE_GENERATOR.PY - Generación de Patrones Sintéticos
===================================================================

Este módulo genera imágenes sintéticas que maximizan la activación
de neuronas específicas mediante técnicas de optimización (gradient ascent).

Técnicas implementadas:
1. Gradient Ascent en el espacio de píxeles
2. Regularizaciones (L2, Total Variation)
3. Transformaciones aleatorias (jitter, rotation, scale)
4. Blur periódico para suavizado

Uso:
    generator = FeatureGenerator(model, 'features.0')
    synthetic_img, history = generator.generate_pattern(neuron_idx=38)
===================================================================
"""

from config import (
    IMAGE_SIZE,
    FEATURE_ITERATIONS,
    FEATURE_LR,
    L2_DECAY,
    TV_WEIGHT,
    JITTER,
    ROTATION_RANGE,
    SCALE_RANGE,
    BLUR_FREQUENCY,
    BLUR_KERNEL_SIZE,
    VERBOSE_FREQUENCY,
    IMAGENET_MEAN,
    IMAGENET_STD
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm

# Importar configuración
import sys
sys.path.append('..')


class FeatureGenerator:
    """
    Generador de patrones sintéticos que maximizan activaciones neuronales.

    Usa gradient ascent para optimizar una imagen desde ruido aleatorio
    hasta una imagen que activa fuertemente una neurona objetivo.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: Optional[torch.device] = None
    ):
        """
        Inicializa el generador.

        Args:
            model: Modelo de PyTorch en modo eval
            target_layer: Nombre de la capa objetivo
            device: Device para computación
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.device = device if device else torch.device('cpu')

        # Hook para capturar activaciones
        self.activations = {}
        self.hooks = []
        self._register_hook()

        # Normalización ImageNet
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(self.device)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(self.device)

        print(f"✅ FeatureGenerator inicializado")
        print(f"   Capa objetivo: {target_layer}")
        print(f"   Device: {self.device}")

    def _register_hook(self):
        """Registra hook para capturar activaciones."""
        def hook_fn(module, input, output):
            self.activations['target'] = output

        # Buscar y registrar hook
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)
                print(f"   ✓ Hook registrado en: {name}")
                return

        raise ValueError(f"❌ Capa '{self.target_layer}' no encontrada")

    def generate_pattern(
        self,
        neuron_idx: int,
        iterations: int = FEATURE_ITERATIONS,
        lr: float = FEATURE_LR,
        l2_decay: float = L2_DECAY,
        tv_weight: float = TV_WEIGHT,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Genera patrón sintético que maximiza activación de neurona.

        Args:
            neuron_idx: Índice de la neurona objetivo
            iterations: Número de iteraciones de optimización
            lr: Learning rate
            l2_decay: Peso de regularización L2
            tv_weight: Peso de Total Variation
            verbose: Mostrar barra de progreso

        Returns:
            Tupla (imagen, historial):
            - imagen: Array [H, W, 3] uint8 en [0, 255]
            - historial: Dict con métricas por iteración
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🎨 Generando patrón para Neurona {neuron_idx}")
            print(f"{'='*70}")
            print(f"Iteraciones: {iterations}")
            print(f"Learning rate: {lr}")
            print(f"L2 decay: {l2_decay}")
            print(f"TV weight: {tv_weight}")

        # ===============================================================
        # PASO 1: Inicializar imagen con ruido
        # ===============================================================

        img_tensor = torch.randn(
            1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1],
            device=self.device,
            requires_grad=True
        )
        img_tensor.data *= 0.1  # Escalar ruido inicial

        # Optimizador
        optimizer = torch.optim.Adam([img_tensor], lr=lr)

        # Historial de métricas
        history = {
            'activation': [],
            'l2_loss': [],
            'tv_loss': [],
            'total_loss': []
        }

        # ===============================================================
        # PASO 2: Loop de optimización (Gradient Ascent)
        # ===============================================================

        iterator = range(iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Optimizando")

        for iteration in iterator:
            # Reiniciar gradientes
            optimizer.zero_grad()

            # Aplicar transformaciones aleatorias
            img_transformed = self._apply_transforms(img_tensor)

            # Forward pass
            _ = self.model(img_transformed)

            # Obtener activación de neurona objetivo
            acts = self.activations['target']  # [1, C, H, W]
            neuron_activation = acts[0, neuron_idx].mean()

            # -------------------------------------------------------
            # Calcular pérdidas
            # -------------------------------------------------------

            # Loss principal: MAXIMIZAR activación (negativo para minimizar)
            activation_loss = -neuron_activation

            # Regularización L2: penaliza valores extremos
            l2_loss = l2_decay * (img_tensor ** 2).mean()

            # Total Variation: suaviza la imagen
            tv_loss = tv_weight * self._total_variation(img_tensor)

            # Loss total
            total_loss = activation_loss + l2_loss + tv_loss

            # -------------------------------------------------------
            # Backward y actualizar
            # -------------------------------------------------------
            total_loss.backward()
            optimizer.step()

            # -------------------------------------------------------
            # Aplicar blur periódicamente
            # -------------------------------------------------------
            if (iteration + 1) % BLUR_FREQUENCY == 0:
                with torch.no_grad():
                    img_tensor.data = self._apply_blur(img_tensor.data)

            # -------------------------------------------------------
            # Registrar historial
            # -------------------------------------------------------
            history['activation'].append(neuron_activation.item())
            history['l2_loss'].append(l2_loss.item())
            history['tv_loss'].append(tv_loss.item())
            history['total_loss'].append(total_loss.item())

            # -------------------------------------------------------
            # Log progreso (solo en modo verbose sin tqdm)
            # -------------------------------------------------------
            if not verbose and (iteration + 1) % VERBOSE_FREQUENCY == 0:
                print(f"Iter {iteration+1:4d} | "
                      f"Act: {neuron_activation.item():7.4f} | "
                      f"L2: {l2_loss.item():7.4f} | "
                      f"TV: {tv_loss.item():7.4f}")

        # ===============================================================
        # PASO 3: Post-procesamiento
        # ===============================================================

        # Convertir a imagen visualizable
        img_final = self._tensor_to_image(img_tensor)

        if verbose:
            print(f"\n✅ Generación completada")
            print(f"   Activación final: {history['activation'][-1]:.4f}")
            print(f"{'='*70}\n")

        return img_final, history

    def _apply_transforms(
        self,
        img: torch.Tensor
    ) -> torch.Tensor:
        """
        Aplica transformaciones aleatorias para robustez.

        Estas transformaciones ayudan a:
        - Evitar overfitting a patrones específicos
        - Generar imágenes más naturales
        - Mejorar generalización

        Args:
            img: Tensor [1, 3, H, W]

        Returns:
            Tensor transformado
        """
        # Jitter: traslación aleatoria
        if JITTER > 0:
            ox = np.random.randint(-JITTER, JITTER + 1)
            oy = np.random.randint(-JITTER, JITTER + 1)
            img = torch.roll(img, shifts=(ox, oy), dims=(2, 3))

        # Rotación aleatoria
        if ROTATION_RANGE > 0:
            angle = np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
            img = transforms.functional.rotate(img, angle)

        # Escala aleatoria
        if SCALE_RANGE[0] < SCALE_RANGE[1]:
            scale = np.random.uniform(*SCALE_RANGE)
            new_size = int(IMAGE_SIZE[0] * scale)

            img = F.interpolate(
                img,
                size=(new_size, new_size),
                mode='bilinear',
                align_corners=False
            )

            # Center crop o pad
            if new_size > IMAGE_SIZE[0]:
                img = transforms.functional.center_crop(img, IMAGE_SIZE)
            else:
                pad = (IMAGE_SIZE[0] - new_size) // 2
                img = F.pad(img, (pad, pad, pad, pad), mode='reflect')

        return img

    def _total_variation(self, img: torch.Tensor) -> torch.Tensor:
        """
        Calcula Total Variation Loss para suavizar la imagen.

        TV mide la variación entre píxeles vecinos:
        - TV alto = imagen ruidosa
        - TV bajo = imagen suave

        Args:
            img: Tensor [1, 3, H, W]

        Returns:
            Scalar con pérdida TV
        """
        # Diferencias horizontales
        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()

        # Diferencias verticales
        tv_v = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()

        return tv_h + tv_v

    def _apply_blur(
        self,
        img: torch.Tensor,
        kernel_size: int = BLUR_KERNEL_SIZE
    ) -> torch.Tensor:
        """
        Aplica Gaussian blur para suavizar.

        Args:
            img: Tensor [1, 3, H, W]
            kernel_size: Tamaño del kernel

        Returns:
            Tensor suavizado
        """
        # Kernel gaussiano simple (promedio uniforme)
        kernel = torch.ones(
            1, 1, kernel_size, kernel_size,
            device=self.device
        )
        kernel = kernel / kernel.sum()

        # Aplicar a cada canal
        blurred = torch.zeros_like(img)
        for c in range(3):
            channel = img[:, c:c+1, :, :]
            blurred[:, c:c+1, :, :] = F.conv2d(
                channel,
                kernel,
                padding=kernel_size // 2
            )

        return blurred

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convierte tensor a imagen visualizable [0, 255].

        Args:
            tensor: Tensor [1, 3, H, W] normalizado

        Returns:
            Array numpy [H, W, 3] uint8 en [0, 255]
        """
        img = tensor.detach().cpu().squeeze(0)  # [3, H, W]

        # Desnormalizar (ImageNet)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = img * std + mean

        # Clip a [0, 1]
        img = torch.clamp(img, 0, 1)

        # A numpy [H, W, 3]
        img = img.permute(1, 2, 0).numpy()

        # A rango [0, 255]
        img = (img * 255).astype(np.uint8)

        return img

    def cleanup(self):
        """Limpia hooks y libera recursos."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        print("🧹 FeatureGenerator limpiado")


# ===================================================================
# FUNCIONES DE UTILIDAD
# ===================================================================

def compare_activations(
    model: nn.Module,
    target_layer: str,
    real_image_tensor: torch.Tensor,
    synthetic_image: np.ndarray,
    neuron_idx: int,
    device: torch.device
) -> Dict:
    """
    Compara activaciones de imagen real vs sintética.

    Args:
        model: Modelo de PyTorch
        target_layer: Capa objetivo
        real_image_tensor: Tensor [1, 3, H, W] de imagen real
        synthetic_image: Array [H, W, 3] de imagen sintética
        neuron_idx: Índice de neurona
        device: Device

    Returns:
        Dict con activaciones y mejora
    """
    # Procesar imagen sintética
    from modules.image_processor import ImageProcessor
    processor = ImageProcessor(device=device)

    # Convertir sintética a tensor
    synthetic_pil = processor.image_to_pil(synthetic_image)
    synthetic_tensor, _ = processor.load_and_preprocess(synthetic_pil)

    # Crear hook temporal
    activations = {}

    def hook_fn(module, input, output):
        activations['target'] = output

    for name, module in model.named_modules():
        if name == target_layer:
            handle = module.register_forward_hook(hook_fn)
            break

    # Obtener activación de imagen real
    with torch.no_grad():
        _ = model(real_image_tensor)
        real_act = activations['target'][0, neuron_idx].mean().item()

    # Obtener activación de imagen sintética
    activations.clear()
    with torch.no_grad():
        _ = model(synthetic_tensor)
        synthetic_act = activations['target'][0, neuron_idx].mean().item()

    # Limpiar hook
    handle.remove()

    return {
        'real_activation': real_act,
        'synthetic_activation': synthetic_act,
        'improvement': (synthetic_act / max(real_act, 1e-8)) * 100
    }


# ===================================================================
# TESTING
# ===================================================================

if __name__ == "__main__":
    print("🧪 Testing FeatureGenerator...\n")

    # Cargar modelo
    print("1️⃣ Cargando modelo AlexNet...")
    from torchvision import models
    model = models.alexnet(pretrained=False)
    model.eval()

    # Crear generator
    print("\n2️⃣ Creando FeatureGenerator...")
    generator = FeatureGenerator(model, 'features.0')

    # Generar patrón (pocas iteraciones para test)
    print("\n3️⃣ Generando patrón sintético...")
    img, history = generator.generate_pattern(
        neuron_idx=0,
        iterations=50,  # Pocas iteraciones para test
        verbose=True
    )

    print(f"\n4️⃣ Resultados:")
    print(f"   Imagen shape: {img.shape}")
    print(f"   Imagen dtype: {img.dtype}")
    print(f"   Imagen range: [{img.min()}, {img.max()}]")
    print(f"   Activación final: {history['activation'][-1]:.4f}")

    # Cleanup
    print("\n5️⃣ Limpiando...")
    generator.cleanup()

    print("\n✅ Testing completado!")
