"""
===================================================================
HELPERS.PY - Funciones Auxiliares
===================================================================

Este módulo contiene funciones de utilidad general que se usan
en múltiples partes de la aplicación.

Categorías:
1. Conversiones de formato
2. Validaciones
3. Formateo de texto
4. Funciones de IO
===================================================================
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional, List
from pathlib import Path
import json
from datetime import datetime


# ===================================================================
# 1. CONVERSIONES DE FORMATO
# ===================================================================

def ensure_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Asegura que los datos estén en formato numpy.

    Args:
        data: Array numpy o tensor PyTorch

    Returns:
        Array numpy
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def ensure_tensor(
    data: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Asegura que los datos estén en formato tensor.

    Args:
        data: Array numpy o tensor PyTorch
        device: Device objetivo (opcional)

    Returns:
        Tensor de PyTorch
    """
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).float()
    else:
        tensor = data

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def normalize_range(
    data: np.ndarray,
    target_range: Tuple[float, float] = (0, 1)
) -> np.ndarray:
    """
    Normaliza array a un rango específico.

    Args:
        data: Array a normalizar
        target_range: Rango objetivo (min, max)

    Returns:
        Array normalizado
    """
    min_val, max_val = target_range

    # Normalizar a [0, 1] primero
    data_min = data.min()
    data_max = data.max()

    if data_max - data_min > 1e-8:
        normalized = (data - data_min) / (data_max - data_min)
    else:
        normalized = np.zeros_like(data)

    # Escalar al rango objetivo
    scaled = normalized * (max_val - min_val) + min_val

    return scaled


def to_uint8(data: np.ndarray) -> np.ndarray:
    """
    Convierte array a uint8 en rango [0, 255].

    Args:
        data: Array en cualquier rango

    Returns:
        Array uint8 en [0, 255]
    """
    # Normalizar a [0, 1]
    if data.max() > 1.0:
        data = data / 255.0

    # Escalar a [0, 255]
    data = np.clip(data * 255, 0, 255).astype(np.uint8)

    return data


# ===================================================================
# 2. VALIDACIONES
# ===================================================================

def validate_image_shape(
    image: np.ndarray,
    expected_channels: int = 3
) -> bool:
    """
    Valida que una imagen tenga el shape correcto.

    Args:
        image: Array de imagen
        expected_channels: Número esperado de canales

    Returns:
        True si es válida
    """
    if image.ndim not in [2, 3]:
        return False

    if image.ndim == 3 and image.shape[2] != expected_channels:
        return False

    return True


def validate_activation_shape(
    activations: torch.Tensor,
    expected_dims: int = 4
) -> bool:
    """
    Valida que las activaciones tengan el shape correcto.

    Args:
        activations: Tensor de activaciones
        expected_dims: Número esperado de dimensiones

    Returns:
        True si es válida
    """
    return activations.dim() == expected_dims


def is_valid_neuron_idx(
    neuron_idx: int,
    num_channels: int
) -> bool:
    """
    Valida que un índice de neurona sea válido.

    Args:
        neuron_idx: Índice de la neurona
        num_channels: Número total de canales

    Returns:
        True si es válido
    """
    return 0 <= neuron_idx < num_channels


# ===================================================================
# 3. FORMATEO DE TEXTO
# ===================================================================

def format_number(num: float, decimals: int = 3) -> str:
    """
    Formatea un número con decimales específicos.

    Args:
        num: Número a formatear
        decimals: Número de decimales

    Returns:
        String formateado
    """
    return f"{num:.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Formatea un valor como porcentaje.

    Args:
        value: Valor en rango [0, 1]
        decimals: Número de decimales

    Returns:
        String con porcentaje (ej: "75.3%")
    """
    return f"{value * 100:.{decimals}f}%"


def format_layer_name(layer_name: str) -> str:
    """
    Formatea nombre de capa para display.

    Args:
        layer_name: Nombre técnico de la capa (ej: 'features.0')

    Returns:
        Nombre formateado (ej: 'Features Layer 0')
    """
    parts = layer_name.split('.')
    if len(parts) == 2:
        return f"{parts[0].capitalize()} Layer {parts[1]}"
    return layer_name


def format_model_name(model_name: str) -> str:
    """
    Formatea nombre de modelo para display.

    Args:
        model_name: Nombre técnico (ej: 'alexnet')

    Returns:
        Nombre formateado (ej: 'AlexNet')
    """
    # Casos especiales
    special_cases = {
        'alexnet': 'AlexNet',
        'resnet18': 'ResNet-18',
        'resnet50': 'ResNet-50',
        'vgg16': 'VGG-16',
        'vgg19': 'VGG-19'
    }

    return special_cases.get(model_name.lower(), model_name.capitalize())


# ===================================================================
# 4. FUNCIONES DE IO
# ===================================================================

def save_json(data: dict, filepath: Union[str, Path]):
    """
    Guarda diccionario como JSON.

    Args:
        data: Diccionario a guardar
        filepath: Ruta del archivo
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Union[str, Path]) -> dict:
    """
    Carga diccionario desde JSON.

    Args:
        filepath: Ruta del archivo

    Returns:
        Diccionario cargado
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def get_timestamp() -> str:
    """
    Obtiene timestamp actual formateado.

    Returns:
        String con timestamp (ej: '2025-01-15_14-30-45')
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_filename(
    prefix: str,
    suffix: str = '',
    extension: str = 'png',
    include_timestamp: bool = True
) -> str:
    """
    Crea nombre de archivo con formato consistente.

    Args:
        prefix: Prefijo del nombre
        suffix: Sufijo del nombre (opcional)
        extension: Extensión del archivo
        include_timestamp: Si incluir timestamp

    Returns:
        Nombre de archivo (ej: 'heatmap_neuron38_2025-01-15_14-30-45.png')
    """
    parts = [prefix]

    if suffix:
        parts.append(suffix)

    if include_timestamp:
        parts.append(get_timestamp())

    filename = '_'.join(parts) + f'.{extension}'

    return filename


# ===================================================================
# 5. CÁLCULOS MATEMÁTICOS
# ===================================================================

def compute_improvement_percentage(
    baseline: float,
    improved: float
) -> float:
    """
    Calcula porcentaje de mejora.

    Args:
        baseline: Valor base
        improved: Valor mejorado

    Returns:
        Porcentaje de mejora
    """
    if baseline < 1e-8:
        return 0.0

    return ((improved - baseline) / baseline) * 100


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0
) -> float:
    """
    División segura que evita división por cero.

    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor por defecto si denominator es 0

    Returns:
        Resultado de la división o default
    """
    if abs(denominator) < 1e-8:
        return default

    return numerator / denominator


def moving_average(
    data: List[float],
    window_size: int = 5
) -> List[float]:
    """
    Calcula promedio móvil de una serie.

    Args:
        data: Lista de valores
        window_size: Tamaño de la ventana

    Returns:
        Lista con promedios móviles
    """
    if len(data) < window_size:
        return data

    result = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        end = i + 1
        window = data[start:end]
        result.append(sum(window) / len(window))

    return result


# ===================================================================
# 6. FUNCIONES DE DISPLAY
# ===================================================================

def print_section_header(title: str, width: int = 70):
    """
    Imprime un header de sección formateado.

    Args:
        title: Título de la sección
        width: Ancho del header
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


def print_progress(
    current: int,
    total: int,
    prefix: str = '',
    suffix: str = '',
    decimals: int = 1,
    length: int = 50
):
    """
    Imprime barra de progreso en consola.

    Args:
        current: Valor actual
        total: Valor total
        prefix: Texto antes de la barra
        suffix: Texto después de la barra
        decimals: Decimales en porcentaje
        length: Longitud de la barra
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (current / float(total)))
    filled = int(length * current // total)
    bar = '█' * filled + '-' * (length - filled)

    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')

    # Nueva línea al completar
    if current == total:
        print()


# ===================================================================
# 7. FUNCIONES DE GEOMETRÍA
# ===================================================================

def compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> float:
    """
    Calcula Intersection over Union de dos cajas.

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        IoU en [0, 1]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Área de intersección
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Áreas de cada caja
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union
    union_area = box1_area + box2_area - inter_area

    # IoU
    iou = safe_divide(inter_area, union_area)

    return iou


def clip_to_bounds(
    coords: Tuple[int, int],
    max_coords: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Clip coordenadas a límites de imagen.

    Args:
        coords: (y, x) coordenadas
        max_coords: (max_y, max_x) límites

    Returns:
        Coordenadas clipeadas
    """
    y = max(0, min(coords[0], max_coords[0] - 1))
    x = max(0, min(coords[1], max_coords[1] - 1))

    return (y, x)


# ===================================================================
# TESTING
# ===================================================================

if __name__ == "__main__":
    print("🧪 Testing helpers...\n")

    # Test conversiones
    print("1️⃣ Testing conversiones...")
    arr = np.random.rand(3, 224, 224)
    tensor = ensure_tensor(arr)
    print(f"   Array → Tensor: {tensor.shape}")

    arr_back = ensure_numpy(tensor)
    print(f"   Tensor → Array: {arr_back.shape}")

    # Test normalización
    print("\n2️⃣ Testing normalización...")
    data = np.array([1, 2, 3, 4, 5])
    normalized = normalize_range(data, (0, 1))
    print(f"   Original: {data}")
    print(f"   Normalizado: {normalized}")

    # Test formateo
    print("\n3️⃣ Testing formateo...")
    print(f"   Número: {format_number(3.14159265, decimals=2)}")
    print(f"   Porcentaje: {format_percentage(0.753)}")
    print(f"   Layer: {format_layer_name('features.0')}")
    print(f"   Model: {format_model_name('alexnet')}")

    # Test cálculos
    print("\n4️⃣ Testing cálculos...")
    improvement = compute_improvement_percentage(10.0, 15.0)
    print(f"   Mejora: {improvement:.1f}%")

    # Test filename
    print("\n5️⃣ Testing filename...")
    filename = create_filename('heatmap', 'neuron38', include_timestamp=False)
    print(f"   Filename: {filename}")

    print("\n✅ Testing completado!")
