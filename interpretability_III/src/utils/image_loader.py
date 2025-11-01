"""
============================================================================
IMAGE LOADER - Carga y Procesamiento de Datasets de Imágenes
============================================================================

Este módulo proporciona funcionalidades para:
- Cargar datasets de imágenes (CIFAR-10, CIFAR-100, ImageNet subset)
- Aplicar transformaciones y augmentations
- Crear dataloaders para entrenamiento e inferencia
- Normalización y denormalización de imágenes
- Utilidades de visualización

Clase Principal:
    ImageLoader: Carga y procesa datasets de imágenes

Uso típico:
    from src.utils.image_loader import ImageLoader
    
    # Cargar CIFAR-10
    loader = ImageLoader(dataset_name='cifar10', batch_size=32)
    train_loader, test_loader = loader.get_dataloaders()
    
    # Visualizar muestra
    images, labels = next(iter(test_loader))

Autor: Proyecto Módulo III
Fecha: 2025-01-15
============================================================================
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List, Optional
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageLoader:
    """
    Clase para cargar y procesar datasets de imágenes.

    Esta clase facilita:
    - Carga de datasets populares (CIFAR-10, CIFAR-100)
    - Aplicación de transformaciones estándar
    - Creación de dataloaders
    - Normalización con estadísticas de ImageNet
    - Utilidades de visualización

    Attributes:
        dataset_name (str): Nombre del dataset ('cifar10', 'cifar100')
        batch_size (int): Tamaño del batch
        num_workers (int): Número de workers para carga paralela
        data_dir (Path): Directorio donde se almacenan los datos
        train_dataset: Dataset de entrenamiento
        test_dataset: Dataset de prueba

    Example:
        >>> loader = ImageLoader('cifar10', batch_size=64)
        >>> train_loader, test_loader = loader.get_dataloaders()
        >>> print(f"Batches: {len(train_loader)}")
    """

    # Estadísticas de normalización de ImageNet (usadas por modelos pre-entrenados)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Clases de CIFAR-10
    CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # Clases de CIFAR-100 (superclases)
    CIFAR100_SUPERCLASSES = [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers',
        'fruit_and_vegetables', 'household_electrical_devices',
        'household_furniture', 'insects', 'large_carnivores',
        'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
        'large_omnivores_and_herbivores', 'medium_mammals',
        'non-insect_invertebrates', 'people', 'reptiles',
        'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
    ]

    def __init__(
        self,
        dataset_name: str = 'cifar10',
        batch_size: int = 32,
        num_workers: int = 2,
        data_dir: Optional[str] = None,
        download: bool = True,
        shuffle_train: bool = True,
        pin_memory: bool = True
    ):
        """
        Inicializa el cargador de imágenes.

        Args:
            dataset_name: Nombre del dataset ('cifar10', 'cifar100')
            batch_size: Tamaño del batch para los dataloaders
            num_workers: Número de workers para carga paralela
            data_dir: Directorio para almacenar datos (None = './data')
            download: Si True, descarga el dataset si no existe
            shuffle_train: Si True, mezcla el dataset de entrenamiento
            pin_memory: Si True, usa pinned memory para GPU

        Raises:
            ValueError: Si el dataset no está soportado
        """
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory
        self.download = download

        # Configurar directorio de datos
        if data_dir is None:
            self.data_dir = Path('./data')
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Validar dataset soportado
        supported_datasets = ['cifar10', 'cifar100']
        if self.dataset_name not in supported_datasets:
            raise ValueError(
                f"Dataset '{dataset_name}' no soportado. "
                f"Datasets disponibles: {supported_datasets}"
            )

        # Inicializar datasets
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None

        logger.info(f"ImageLoader inicializado: {self.dataset_name}")
        logger.info(f"Directorio de datos: {self.data_dir}")
        logger.info(f"Batch size: {self.batch_size}")

    def _get_transforms(self, train: bool = False) -> transforms.Compose:
        """
        Obtiene las transformaciones apropiadas para el dataset.

        Args:
            train: Si True, aplica augmentations de entrenamiento

        Returns:
            Composición de transformaciones
        """
        if train:
            # Transformaciones para entrenamiento (con data augmentation)
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD
                )
            ]
        else:
            # Transformaciones para test/validación (sin augmentation)
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD
                )
            ]

        return transforms.Compose(transform_list)

    def load_datasets(self):
        """
        Carga los datasets de entrenamiento y prueba.

        Este método descarga (si es necesario) y carga los datasets
        con las transformaciones apropiadas.
        """
        logger.info(f"Cargando dataset {self.dataset_name}...")

        # Obtener transformaciones
        train_transform = self._get_transforms(train=True)
        test_transform = self._get_transforms(train=False)

        # Cargar dataset según el nombre
        if self.dataset_name == 'cifar10':
            self.train_dataset = CIFAR10(
                root=str(self.data_dir),
                train=True,
                download=self.download,
                transform=train_transform
            )
            self.test_dataset = CIFAR10(
                root=str(self.data_dir),
                train=False,
                download=self.download,
                transform=test_transform
            )

        elif self.dataset_name == 'cifar100':
            self.train_dataset = CIFAR100(
                root=str(self.data_dir),
                train=True,
                download=self.download,
                transform=train_transform
            )
            self.test_dataset = CIFAR100(
                root=str(self.data_dir),
                train=False,
                download=self.download,
                transform=test_transform
            )

        logger.info(f"✅ Dataset cargado exitosamente")
        logger.info(f"   Train samples: {len(self.train_dataset):,}")
        logger.info(f"   Test samples: {len(self.test_dataset):,}")

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Crea y retorna los dataloaders de entrenamiento y prueba.

        Returns:
            Tupla (train_loader, test_loader)

        Example:
            >>> loader = ImageLoader('cifar10')
            >>> train_loader, test_loader = loader.get_dataloaders()
            >>> images, labels = next(iter(train_loader))
            >>> print(images.shape)  # [batch_size, 3, 32, 32]
        """
        # Cargar datasets si no están cargados
        if self.train_dataset is None or self.test_dataset is None:
            self.load_datasets()

        # Crear dataloader de entrenamiento
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

        # Crear dataloader de prueba
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

        logger.info(f"✅ Dataloaders creados")
        logger.info(f"   Train batches: {len(self.train_loader)}")
        logger.info(f"   Test batches: {len(self.test_loader)}")

        return self.train_loader, self.test_loader

    def get_dataset_info(self) -> Dict:
        """
        Obtiene información general del dataset.

        Returns:
            Diccionario con información del dataset:
                - name: Nombre del dataset
                - num_classes: Número de clases
                - image_size: Tamaño de las imágenes
                - train_samples: Número de muestras de entrenamiento
                - test_samples: Número de muestras de prueba
                - classes: Lista de nombres de clases

        Example:
            >>> info = loader.get_dataset_info()
            >>> print(f"Classes: {info['num_classes']}")
        """
        # Cargar datasets si no están cargados
        if self.train_dataset is None or self.test_dataset is None:
            self.load_datasets()

        # Obtener información según el dataset
        if self.dataset_name == 'cifar10':
            num_classes = 10
            classes = self.CIFAR10_CLASSES
            image_size = (32, 32)
        elif self.dataset_name == 'cifar100':
            num_classes = 100
            classes = self.CIFAR100_SUPERCLASSES
            image_size = (32, 32)
        else:
            num_classes = 0
            classes = []
            image_size = (0, 0)

        return {
            'name': self.dataset_name.upper(),
            'num_classes': num_classes,
            'image_size': image_size,
            'train_samples': len(self.train_dataset),
            'test_samples': len(self.test_dataset),
            'classes': classes,
            'normalization_mean': self.IMAGENET_MEAN,
            'normalization_std': self.IMAGENET_STD
        }

    def denormalize_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormaliza una imagen tensor para visualización.

        Args:
            tensor: Tensor de imagen normalizada [C, H, W] o [B, C, H, W]

        Returns:
            Array numpy con valores en [0, 1] listo para visualización

        Example:
            >>> images, _ = next(iter(test_loader))
            >>> img = loader.denormalize_image(images[0])
            >>> plt.imshow(img)
        """
        # Si es un batch, tomar solo la primera imagen
        if tensor.dim() == 4:
            tensor = tensor[0]

        # Clonar tensor para no modificar el original
        tensor = tensor.clone()

        # Denormalizar
        for t, m, s in zip(tensor, self.IMAGENET_MEAN, self.IMAGENET_STD):
            t.mul_(s).add_(m)

        # Clip a [0, 1]
        tensor = torch.clamp(tensor, 0, 1)

        # Convertir a numpy y transponer a [H, W, C]
        img = tensor.cpu().numpy().transpose(1, 2, 0)

        return img

    def normalize_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Normaliza una imagen numpy para input al modelo.

        Args:
            img: Array numpy con valores en [0, 255] o [0, 1]
                Shape: [H, W, C]

        Returns:
            Tensor normalizado [C, H, W]

        Example:
            >>> img = Image.open('image.jpg')
            >>> tensor = loader.normalize_image(np.array(img))
            >>> output = model(tensor.unsqueeze(0))
        """
        # Si está en [0, 255], convertir a [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        # Convertir a tensor y transponer a [C, H, W]
        tensor = torch.from_numpy(img).float()
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)

        # Normalizar
        normalize = transforms.Normalize(
            mean=self.IMAGENET_MEAN,
            std=self.IMAGENET_STD
        )
        tensor = normalize(tensor)

        return tensor

    def get_class_distribution(self, split: str = 'train') -> Dict[str, int]:
        """
        Calcula la distribución de clases en el dataset.

        Args:
            split: 'train' o 'test'

        Returns:
            Diccionario {nombre_clase: número_de_muestras}

        Example:
            >>> dist = loader.get_class_distribution('train')
            >>> print(dist['airplane'])
            5000
        """
        # Cargar datasets si no están cargados
        if self.train_dataset is None or self.test_dataset is None:
            self.load_datasets()

        # Seleccionar dataset
        dataset = self.train_dataset if split == 'train' else self.test_dataset

        # Obtener nombres de clases
        if self.dataset_name == 'cifar10':
            class_names = self.CIFAR10_CLASSES
        elif self.dataset_name == 'cifar100':
            class_names = self.CIFAR100_SUPERCLASSES
        else:
            class_names = []

        # Contar muestras por clase
        class_counts = {}
        for _, label in dataset:
            class_name = class_names[label] if label < len(
                class_names) else f"class_{label}"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return class_counts

    def get_sample_images(
        self,
        num_images: int = 16,
        split: str = 'test',
        random: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtiene un conjunto de imágenes de muestra.

        Args:
            num_images: Número de imágenes a obtener
            split: 'train' o 'test'
            random: Si True, selecciona imágenes aleatorias

        Returns:
            Tupla (images, labels) como tensors

        Example:
            >>> images, labels = loader.get_sample_images(num_images=8)
            >>> print(images.shape)  # [8, 3, 32, 32]
        """
        # Cargar datasets si no están cargados
        if self.train_dataset is None or self.test_dataset is None:
            self.load_datasets()

        # Seleccionar dataset
        dataset = self.train_dataset if split == 'train' else self.test_dataset

        # Seleccionar índices
        if random:
            indices = np.random.choice(
                len(dataset), size=num_images, replace=False)
        else:
            indices = range(min(num_images, len(dataset)))

        # Obtener imágenes y labels
        images = []
        labels = []

        for idx in indices:
            img, label = dataset[idx]
            images.append(img)
            labels.append(label)

        # Apilar en tensors
        images = torch.stack(images)
        labels = torch.tensor(labels)

        return images, labels

    def load_custom_image(
        self,
        image_path: str,
        resize: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Carga y procesa una imagen personalizada.

        Args:
            image_path: Ruta a la imagen
            resize: Tamaño opcional para redimensionar (H, W)

        Returns:
            Tensor normalizado [1, C, H, W] listo para el modelo

        Example:
            >>> img_tensor = loader.load_custom_image('my_image.jpg')
            >>> output = model(img_tensor)
        """
        # Cargar imagen
        img = Image.open(image_path).convert('RGB')

        # Redimensionar si es necesario
        if resize is not None:
            img = img.resize(resize, Image.BILINEAR)

        # Aplicar transformaciones
        transform = self._get_transforms(train=False)
        img_tensor = transform(img)

        # Agregar dimensión de batch
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def get_statistics(self) -> Dict:
        """
        Calcula estadísticas del dataset (media, std, min, max).

        Returns:
            Diccionario con estadísticas por canal

        Note:
            Este método puede ser lento para datasets grandes

        Example:
            >>> stats = loader.get_statistics()
            >>> print(stats['mean'])
        """
        # Cargar datasets si no están cargados
        if self.train_dataset is None:
            self.load_datasets()

        logger.info("Calculando estadísticas del dataset (puede tardar)...")

        # Crear dataloader sin normalización
        dataset_raw = type(self.train_dataset)(
            root=str(self.data_dir),
            train=True,
            download=False,
            transform=transforms.ToTensor()
        )

        loader = DataLoader(
            dataset_raw, batch_size=self.batch_size, num_workers=self.num_workers)

        # Acumular valores
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total_samples = 0

        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_samples += batch_samples

        mean /= total_samples
        std /= total_samples

        logger.info("✅ Estadísticas calculadas")

        return {
            'mean': mean.tolist(),
            'std': std.tolist(),
            'channels': ['R', 'G', 'B']
        }

    def __repr__(self) -> str:
        """Representación en string del ImageLoader."""
        return (f"ImageLoader(dataset='{self.dataset_name}', "
                f"batch_size={self.batch_size}, "
                f"num_workers={self.num_workers})")


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def visualize_augmentations(
    dataset: Dataset,
    num_examples: int = 4
) -> None:
    """
    Visualiza ejemplos de data augmentation.

    Args:
        dataset: Dataset con transformaciones
        num_examples: Número de ejemplos a mostrar

    Example:
        >>> loader = ImageLoader('cifar10')
        >>> loader.load_datasets()
        >>> visualize_augmentations(loader.train_dataset)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(num_examples, 5, figsize=(15, num_examples * 3))

    for i in range(num_examples):
        # Obtener una imagen
        idx = np.random.randint(len(dataset))

        # Obtener 5 augmentations diferentes
        for j in range(5):
            img, _ = dataset[idx]

            # Denormalizar para visualización
            img = img.clone()
            for t, m, s in zip(img, ImageLoader.IMAGENET_MEAN, ImageLoader.IMAGENET_STD):
                t.mul_(s).add_(m)
            img = torch.clamp(img, 0, 1)
            img = img.numpy().transpose(1, 2, 0)

            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f'Aug {j+1}')

    plt.tight_layout()
    plt.show()


def calculate_dataset_mean_std(dataloader: DataLoader) -> Tuple[List[float], List[float]]:
    """
    Calcula la media y desviación estándar de un dataset.

    Args:
        dataloader: DataLoader del dataset

    Returns:
        Tupla (mean, std) como listas de 3 elementos (RGB)

    Example:
        >>> mean, std = calculate_dataset_mean_std(train_loader)
        >>> print(f"Mean: {mean}, Std: {std}")
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean.tolist(), std.tolist()


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del ImageLoader.
    """
    print("🖼️ EJEMPLO DE USO: ImageLoader")
    print("="*70)

    # 1. Crear loader
    print("\n🔄 Creando ImageLoader para CIFAR-10...")
    loader = ImageLoader(dataset_name='cifar10', batch_size=32, num_workers=2)

    # 2. Obtener dataloaders
    print("\n📦 Cargando datasets...")
    train_loader, test_loader = loader.get_dataloaders()

    # 3. Información del dataset
    info = loader.get_dataset_info()
    print(f"\n📊 Información del Dataset:")
    print(f"   Nombre: {info['name']}")
    print(f"   Clases: {info['num_classes']}")
    print(f"   Tamaño de imagen: {info['image_size']}")
    print(f"   Muestras train: {info['train_samples']:,}")
    print(f"   Muestras test: {info['test_samples']:,}")

    # 4. Obtener un batch
    print(f"\n🎯 Obteniendo batch de prueba...")
    images, labels = next(iter(test_loader))
    print(f"   Shape de images: {images.shape}")
    print(f"   Shape de labels: {labels.shape}")
    print(f"   Rango de valores: [{images.min():.2f}, {images.max():.2f}]")

    # 5. Distribución de clases
    print(f"\n📊 Distribución de clases (train):")
    dist = loader.get_class_distribution('train')
    for class_name, count in list(dist.items())[:5]:
        print(f"   {class_name}: {count:,}")
    print(f"   ...")

    # 6. Denormalizar imagen
    print(f"\n🎨 Denormalizando imagen para visualización...")
    img = loader.denormalize_image(images[0])
    print(f"   Shape denormalizada: {img.shape}")
    print(f"   Rango: [{img.min():.2f}, {img.max():.2f}]")

    print("\n✅ Ejemplo completado exitosamente")
