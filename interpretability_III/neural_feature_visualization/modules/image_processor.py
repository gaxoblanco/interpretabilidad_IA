"""
===================================================================
IMAGE_PROCESSOR.PY - Procesamiento de Imágenes
===================================================================

Este módulo maneja todas las operaciones relacionadas con imágenes:
- Carga desde archivo o bytes
- Redimensionado manteniendo aspect ratio
- Normalización para modelos (ImageNet)
- Conversión tensor ↔ imagen
- Validación de formato y tamaño

Uso:
    processor = ImageProcessor()
    tensor, img_vis = processor.load_and_preprocess('cat.jpg')
===================================================================
"""

from config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_UPLOAD_SIZE_MB,
    SUPPORTED_FORMATS
)
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Union, Optional
from pathlib import Path
import io

# Importar configuración
import sys
sys.path.append('..')


class ImageProcessor:
    """
    Procesador de imágenes para redes neuronales.

    Maneja la carga, preprocesamiento y conversión de imágenes
    entre diferentes formatos (PIL, numpy, torch).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        image_size: Tuple[int, int] = IMAGE_SIZE
    ):
        """
        Inicializa el procesador de imágenes.

        Args:
            device: Device para tensores de PyTorch
            image_size: Tamaño objetivo (height, width)
        """
        self.device = device if device else torch.device('cpu')
        self.image_size = image_size

        # Tensores de normalización (ImageNet)
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(self.device)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(self.device)

        print(f"✅ ImageProcessor inicializado")
        print(f"   Tamaño objetivo: {image_size}")
        print(f"   Device: {self.device}")

    def validate_image_file(
        self,
        file_path: Union[str, Path, bytes],
        max_size_mb: float = MAX_UPLOAD_SIZE_MB
    ) -> Tuple[bool, str]:
        """
        Valida un archivo de imagen.

        Args:
            file_path: Ruta al archivo o bytes
            max_size_mb: Tamaño máximo permitido en MB

        Returns:
            Tupla (es_válido, mensaje)
        """
        try:
            # Caso 1: Bytes
            if isinstance(file_path, bytes):
                size_mb = len(file_path) / (1024 * 1024)
                if size_mb > max_size_mb:
                    return False, f"Archivo muy grande: {size_mb:.2f} MB (máx: {max_size_mb} MB)"

                # Intentar abrir
                img = Image.open(io.BytesIO(file_path))

            # Caso 2: Path
            else:
                file_path = Path(file_path)

                # Verificar existencia
                if not file_path.exists():
                    return False, f"Archivo no encontrado: {file_path}"

                # Verificar tamaño
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > max_size_mb:
                    return False, f"Archivo muy grande: {size_mb:.2f} MB (máx: {max_size_mb} MB)"

                # Verificar extensión
                ext = file_path.suffix.lower().replace('.', '')
                if ext not in SUPPORTED_FORMATS:
                    return False, f"Formato no soportado: {ext} (soportados: {SUPPORTED_FORMATS})"

                # Intentar abrir
                img = Image.open(file_path)

            # Verificar que sea RGB válida
            img.verify()  # Verifica integridad

            return True, "Imagen válida"

        except Exception as e:
            return False, f"Error al validar imagen: {str(e)}"

    def load_image(
        self,
        source: Union[str, Path, bytes, Image.Image],
        resize: bool = True
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Carga una imagen desde múltiples fuentes.

        Args:
            source: Puede ser:
                   - Ruta como string o Path
                   - Bytes del archivo
                   - Objeto PIL.Image
            resize: Si redimensionar a self.image_size

        Returns:
            Tupla (imagen_pil, imagen_numpy)
            - imagen_pil: PIL.Image en RGB
            - imagen_numpy: Array [H, W, 3] float32 en [0, 1]

        Raises:
            ValueError: Si el source no es válido
        """
        # Cargar según tipo de source
        if isinstance(source, Image.Image):
            img = source.convert('RGB')
        elif isinstance(source, bytes):
            img = Image.open(io.BytesIO(source)).convert('RGB')
        elif isinstance(source, (str, Path)):
            img = Image.open(source).convert('RGB')
        else:
            raise ValueError(f"Tipo de source no soportado: {type(source)}")

        # Redimensionar si es necesario
        if resize:
            img = self.resize_image(img, self.image_size)

        # Convertir a numpy
        img_np = np.array(img, dtype=np.float32) / 255.0

        return img, img_np

    def resize_image(
        self,
        img: Image.Image,
        target_size: Tuple[int, int],
        maintain_aspect: bool = False
    ) -> Image.Image:
        """
        Redimensiona una imagen.

        Args:
            img: Imagen PIL
            target_size: Tamaño objetivo (height, width)
            maintain_aspect: Si mantener aspect ratio (con center crop)

        Returns:
            Imagen redimensionada
        """
        if maintain_aspect:
            # Redimensionar manteniendo aspecto, luego crop
            img.thumbnail(
                (target_size[1] * 2, target_size[0] * 2), Image.LANCZOS)

            # Center crop al tamaño exacto
            width, height = img.size
            left = (width - target_size[1]) // 2
            top = (height - target_size[0]) // 2
            right = left + target_size[1]
            bottom = top + target_size[0]

            img = img.crop((left, top, right, bottom))
        else:
            # Redimensionar directamente (puede distorsionar)
            img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)

        return img

    def preprocess_for_model(
        self,
        img: Union[Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocesa imagen para el modelo (normalización ImageNet).

        Args:
            img: Imagen PIL o numpy array [H, W, 3] en [0, 1]

        Returns:
            Tensor [1, 3, H, W] normalizado y en device
        """
        # Convertir a numpy si es PIL
        if isinstance(img, Image.Image):
            img = np.array(img, dtype=np.float32) / 255.0

        # Asegurar que está en [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        # Convertir a tensor [H, W, 3] → [3, H, W]
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()

        # Agregar batch dimension [3, H, W] → [1, 3, H, W]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Normalizar con stats de ImageNet
        img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor

    def denormalize_tensor(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Revierte la normalización de ImageNet.

        Args:
            tensor: Tensor [1, 3, H, W] o [3, H, W] normalizado

        Returns:
            Tensor desnormalizado en [0, 1]
        """
        # Manejar batch dimension
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # [1, 3, H, W] → [3, H, W]

        # Desnormalizar
        tensor = tensor * self.std + self.mean

        # Clip a [0, 1]
        tensor = torch.clamp(tensor, 0, 1)

        return tensor

    def tensor_to_image(
        self,
        tensor: torch.Tensor,
        denormalize: bool = True
    ) -> np.ndarray:
        """
        Convierte tensor a imagen visualizable.

        Args:
            tensor: Tensor [1, 3, H, W] o [3, H, W]
            denormalize: Si aplicar desnormalización de ImageNet

        Returns:
            Array numpy [H, W, 3] uint8 en [0, 255]
        """
        # Mover a CPU y quitar batch dimension
        tensor = tensor.detach().cpu()
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # [1, 3, H, W] → [3, H, W]

        # Desnormalizar si es necesario
        if denormalize:
            tensor = self.denormalize_tensor(tensor)
        else:
            # Solo asegurar que está en [0, 1]
            tensor = torch.clamp(tensor, 0, 1)

        # Convertir a numpy [3, H, W] → [H, W, 3]
        img = tensor.permute(1, 2, 0).numpy()

        # Escalar a [0, 255]
        img = (img * 255).astype(np.uint8)

        return img

    def image_to_pil(
        self,
        img: Union[np.ndarray, torch.Tensor]
    ) -> Image.Image:
        """
        Convierte numpy array o tensor a PIL Image.

        Args:
            img: Array [H, W, 3] o tensor [3, H, W] / [1, 3, H, W]

        Returns:
            Imagen PIL en modo RGB
        """
        # Si es tensor, convertir a numpy
        if isinstance(img, torch.Tensor):
            img = self.tensor_to_image(img, denormalize=True)

        # Si está en [0, 1], escalar a [0, 255]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        # Asegurar uint8
        img = img.astype(np.uint8)

        # Crear PIL Image
        return Image.fromarray(img, mode='RGB')

    def load_and_preprocess(
        self,
        source: Union[str, Path, bytes, Image.Image]
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Función de conveniencia: carga y preprocesa en un solo paso.

        Args:
            source: Fuente de la imagen (path, bytes, PIL)

        Returns:
            Tupla (tensor_normalizado, imagen_numpy_visual)
            - tensor_normalizado: [1, 3, H, W] listo para el modelo
            - imagen_numpy_visual: [H, W, 3] float32 en [0, 1] para visualizar
        """
        # Cargar imagen
        img_pil, img_np = self.load_image(source, resize=True)

        # Preprocesar para modelo
        img_tensor = self.preprocess_for_model(img_np)

        return img_tensor, img_np

    def create_batch(
        self,
        images: list
    ) -> torch.Tensor:
        """
        Crea un batch de imágenes.

        Args:
            images: Lista de imágenes (PIL, numpy o paths)

        Returns:
            Tensor [B, 3, H, W] con batch de imágenes normalizadas
        """
        tensors = []

        for img in images:
            tensor, _ = self.load_and_preprocess(img)
            tensors.append(tensor)

        # Concatenar en batch
        batch = torch.cat(tensors, dim=0)

        return batch


# ===================================================================
# FUNCIONES DE UTILIDAD
# ===================================================================

def quick_load(image_path: Union[str, Path]) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Función rápida para cargar y preprocesar una imagen.

    Args:
        image_path: Ruta a la imagen

    Returns:
        Tupla (tensor, numpy_array)
    """
    processor = ImageProcessor()
    return processor.load_and_preprocess(image_path)


# ===================================================================
# TESTING
# ===================================================================

if __name__ == "__main__":
    print("🧪 Testing ImageProcessor...\n")

    # Crear procesador
    processor = ImageProcessor()

    # Crear imagen de prueba
    print("1️⃣ Creando imagen de prueba...")
    test_img = Image.new('RGB', (300, 200), color='red')
    print(f"   Tamaño original: {test_img.size}")

    # Cargar y procesar
    print("\n2️⃣ Cargando y preprocesando...")
    tensor, img_np = processor.load_and_preprocess(test_img)
    print(f"   Tensor shape: {tensor.shape}")
    print(f"   Numpy shape: {img_np.shape}")
    print(f"   Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    # Convertir de vuelta
    print("\n3️⃣ Convirtiendo tensor → imagen...")
    img_back = processor.tensor_to_image(tensor)
    print(f"   Imagen shape: {img_back.shape}")
    print(f"   Imagen dtype: {img_back.dtype}")
    print(f"   Imagen range: [{img_back.min()}, {img_back.max()}]")

    # Validación
    print("\n4️⃣ Validando proceso de normalización...")
    # El tensor normalizado debería estar centrado en ~0
    print(f"   Mean del tensor normalizado: {tensor.mean():.3f}")

    print("\n✅ Testing completado!")
