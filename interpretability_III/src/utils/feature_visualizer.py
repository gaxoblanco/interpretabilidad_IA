"""
Feature Visualizer - Generación de imágenes que maximizan activaciones neuronales

Este módulo implementa técnicas de visualización de features mediante:
- Gradient Ascent en el espacio de píxeles
- Regularizaciones para imágenes naturales
- Transformaciones robustas (jitter, rotación, escala)

Uso:
    visualizer = FeatureVisualizer(model, target_layer, device)
    synthetic_img = visualizer.generate_feature(neuron_idx, iterations=500)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from typing import Dict, Optional, Tuple, List
from PIL import Image


class FeatureVisualizer:
    """
    Genera imágenes sintéticas que maximizan la activación de neuronas específicas

    Atributos:
        model: Red neuronal
        target_layer: Nombre de la capa objetivo
        device: Dispositivo (cuda/cpu)
        hooks: Registros de hooks para capturar activaciones
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: torch.device,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            model: Modelo pre-entrenado
            target_layer: Nombre de la capa a visualizar
            device: torch.device
            input_size: Tamaño de imagen (H, W)
        """
        self.model = model.eval()  # Modo evaluación
        self.target_layer = target_layer
        self.device = device
        self.input_size = input_size

        # Para almacenar activaciones durante forward pass
        self.activations = {}
        self.hooks = []

        # Registrar hook
        self._register_hooks()

        print(f"✅ FeatureVisualizer inicializado")
        print(f"   Capa objetivo: {target_layer}")
        print(f"   Tamaño entrada: {input_size}")

    def _register_hooks(self):
        """Registra hook para capturar activaciones de la capa objetivo"""

        def hook_fn(module, input, output):
            """Función que captura la salida de la capa"""
            self.activations['target'] = output

        # Buscar el módulo correspondiente a target_layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)
                print(f"   ✓ Hook registrado en: {name}")
                return

        raise ValueError(f"❌ Capa '{self.target_layer}' no encontrada")

    def generate_feature(
        self,
        neuron_idx: int,
        iterations: int = 500,
        lr: float = 0.1,
        l2_decay: float = 1e-4,
        tv_weight: float = 1e-2,
        jitter: int = 4,
        rotation_range: float = 5.0,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        blur_freq: int = 4,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Genera imagen que maximiza la activación de una neurona específica

        Args:
            neuron_idx: Índice de la neurona objetivo
            iterations: Número de iteraciones de optimización
            lr: Learning rate
            l2_decay: Peso de regularización L2
            tv_weight: Peso de Total Variation
            jitter: Píxeles de traslación aleatoria
            rotation_range: Rango de rotación en grados
            scale_range: Rango de escala (min, max)
            blur_freq: Cada cuántas iteraciones aplicar blur
            verbose: Mostrar progreso

        Returns:
            imagen: Array [H, W, 3] en rango [0, 255]
            history: Diccionario con historial de pérdidas
        """

        if verbose:
            print(f"\n{'='*70}")
            print(f"🎨 Generando Feature para Neurona {neuron_idx}")
            print(f"{'='*70}")
            print(f"Iteraciones: {iterations}")
            print(f"Learning rate: {lr}")
            print(f"L2 decay: {l2_decay}")
            print(f"TV weight: {tv_weight}")

        # ===================================================================
        # PASO 1: Inicializar imagen con ruido
        # ===================================================================

        # Ruido uniforme normalizado
        img_tensor = torch.randn(
            1, 3, self.input_size[0], self.input_size[1],
            device=self.device
        )
        img_tensor.requires_grad_(True)
        img_tensor.data *= 0.1

        # Optimizer
        optimizer = torch.optim.Adam([img_tensor], lr=lr)

        # Historial
        history = {
            'activation': [],
            'l2_loss': [],
            'tv_loss': [],
            'total_loss': []
        }

        # ===================================================================
        # PASO 2: Loop de optimización (Gradient Ascent)
        # ===================================================================

        for iteration in range(iterations):
            # Pone todos los gradientes a cero antes de calcular nuevos gradientes.
            optimizer.zero_grad()

            # -----------------------------------------------------------
            # Aplicar transformaciones aleatorias (data augmentation)
            # -----------------------------------------------------------
            img_transformed = self._apply_transforms(
                img_tensor,
                jitter=jitter,
                rotation_range=rotation_range,
                scale_range=scale_range
            )

            # -----------------------------------------------------------
            # Forward pass
            # -----------------------------------------------------------
            _ = self.model(img_transformed)  # _ Ignoramos el output final

            # Obtener activación de la neurona objetivo
            activations = self.activations['target']  # [B, C, H, W]

            # Activación de la neurona específica (promedio espacial)
            neuron_activation = activations[0, neuron_idx].mean()

            # -----------------------------------------------------------
            # Calcular pérdidas
            # -----------------------------------------------------------

            # Loss principal: MAXIMIZAR activación (negativo para minimizar)
            activation_loss = -neuron_activation

            # Regularización L2: penaliza valores extremos
            l2_loss = l2_decay * (img_tensor ** 2).mean()

            # Total Variation: suaviza la imagen
            tv_loss = tv_weight * self._total_variation(img_tensor)

            # Loss total
            total_loss = activation_loss + l2_loss + tv_loss

            # -----------------------------------------------------------
            # Backward y actualizar
            # -----------------------------------------------------------
            total_loss.backward()
            optimizer.step()

            # -----------------------------------------------------------
            # Aplicar blur periódicamente (suavizado)
            # -----------------------------------------------------------
            if (iteration + 1) % blur_freq == 0:
                with torch.no_grad():
                    img_tensor.data = self._apply_blur(img_tensor.data)

            # -----------------------------------------------------------
            # Registrar historial
            # -----------------------------------------------------------
            history['activation'].append(neuron_activation.item())
            history['l2_loss'].append(l2_loss.item())
            history['tv_loss'].append(tv_loss.item())
            history['total_loss'].append(total_loss.item())

            # -----------------------------------------------------------
            # Log progreso
            # -----------------------------------------------------------
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iter {iteration+1:4d} | "
                      f"Activación: {neuron_activation.item():7.4f} | "
                      f"L2: {l2_loss.item():7.4f} | "
                      f"TV: {tv_loss.item():7.4f}")

        # ===================================================================
        # PASO 3: Post-procesamiento
        # ===================================================================

        # Convertir a imagen visualizable
        img_final = self._tensor_to_image(img_tensor)

        if verbose:
            print(f"\n✅ Generación completada")
            print(f"   Activación final: {history['activation'][-1]:.4f}")
            print(f"{'='*70}\n")

        return img_final, history

    def _apply_transforms(
        self,
        img: torch.Tensor,
        jitter: int = 4,
        rotation_range: float = 5.0,
        scale_range: Tuple[float, float] = (0.95, 1.05)
    ) -> torch.Tensor:
        """
        Aplica transformaciones aleatorias para robustez

        Estas transformaciones ayudan a:
        - Evitar overfitting a patrones específicos
        - Generar imágenes más naturales
        - Mejorar la generalización
        """

        # Jitter: traslación aleatoria
        if jitter > 0:
            ox = np.random.randint(-jitter, jitter + 1)
            oy = np.random.randint(-jitter, jitter + 1)
            img = torch.roll(img, shifts=(ox, oy), dims=(2, 3))

        # Rotación aleatoria
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            img = transforms.functional.rotate(img, angle)

        # Escala aleatoria
        if scale_range[0] < scale_range[1]:
            scale = np.random.uniform(*scale_range)
            new_size = int(self.input_size[0] * scale)
            img = F.interpolate(
                img,
                size=(new_size, new_size),
                mode='bilinear',
                align_corners=False
            )
            # Center crop
            if new_size > self.input_size[0]:
                img = transforms.functional.center_crop(img, self.input_size)
            else:
                # Pad si es más pequeño
                pad = (self.input_size[0] - new_size) // 2
                img = F.pad(img, (pad, pad, pad, pad), mode='reflect')

        return img

    def _total_variation(self, img: torch.Tensor) -> torch.Tensor:
        """
        Calcula Total Variation Loss para suavizar la imagen

        TV mide la variación entre píxeles vecinos:
        - TV alto = imagen ruidosa
        - TV bajo = imagen suave
        """
        # Diferencias horizontales
        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()

        # Diferencias verticales
        tv_v = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()

        return tv_h + tv_v

    def _apply_blur(self, img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Aplica Gaussian blur para suavizar"""
        # Gaussian kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device)
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
        Convierte tensor a imagen visualizable [0, 255]

        Normalización:
        - ImageNet usa mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        - Revertimos esta normalización
        """
        img = tensor.detach().cpu().squeeze(0)  # [3, H, W]

        # Desnormalizar (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean

        # Clip a [0, 1]
        img = torch.clamp(img, 0, 1)

        # A numpy [H, W, 3]
        img = img.permute(1, 2, 0).numpy()

        # A rango [0, 255]
        img = (img * 255).astype(np.uint8)

        return img

    def generate_grid(
        self,
        neuron_indices: List[int],
        iterations: int = 300,
        lr: float = 0.1,
        verbose: bool = False
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Genera features para múltiples neuronas

        Args:
            neuron_indices: Lista de índices de neuronas
            iterations: Iteraciones por neurona
            lr: Learning rate
            verbose: Mostrar progreso

        Returns:
            images: Lista de imágenes generadas
            histories: Lista de historiales
        """
        print(f"\n🎨 Generando {len(neuron_indices)} features...")

        images = []
        histories = []

        for i, neuron_idx in enumerate(neuron_indices):
            if verbose:
                print(f"\n[{i+1}/{len(neuron_indices)}] Neurona {neuron_idx}")

            img, history = self.generate_feature(
                neuron_idx=neuron_idx,
                iterations=iterations,
                lr=lr,
                verbose=verbose
            )

            images.append(img)
            histories.append(history)

        print(f"\n✅ {len(images)} features generadas")

        return images, histories

    def cleanup(self):
        """Limpia hooks registrados"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("🧹 Hooks limpiados")


def compare_real_vs_synthetic(
    model: nn.Module,
    target_layer: str,
    device: torch.device,
    neuron_idx: int,
    real_image_path: str,
    iterations: int = 500
) -> Dict:
    """
    Compara activación de neurona en imagen real vs sintética

    Args:
        model: Modelo
        target_layer: Capa objetivo
        device: Dispositivo
        neuron_idx: Índice de neurona
        real_image_path: Ruta a imagen real
        iterations: Iteraciones para generar sintética

    Returns:
        Diccionario con resultados de comparación
    """
    from PIL import Image
    from torchvision import transforms

    # Generar imagen sintética
    visualizer = FeatureVisualizer(model, target_layer, device)
    synthetic_img, history = visualizer.generate_feature(
        neuron_idx=neuron_idx,
        iterations=iterations,
        verbose=True
    )

    # Cargar imagen real
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    real_img = Image.open(real_image_path).convert('RGB')
    real_tensor = preprocess(real_img).unsqueeze(0).to(device)

    # Obtener activaciones
    model.eval()
    with torch.no_grad():
        _ = model(real_tensor)
        real_activation = visualizer.activations['target'][0, neuron_idx].mean(
        ).item()

    synthetic_activation = history['activation'][-1]

    visualizer.cleanup()

    return {
        'real_image': np.array(real_img.resize((224, 224))),
        'synthetic_image': synthetic_img,
        'real_activation': real_activation,
        'synthetic_activation': synthetic_activation,
        'improvement': synthetic_activation / max(real_activation, 1e-8)
    }
