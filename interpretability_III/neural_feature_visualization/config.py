"""
===================================================================
CONFIG.PY - Configuración Global
===================================================================

Este archivo centraliza todos los parámetros configurables de la
aplicación, facilitando el mantenimiento y la experimentación.

Secciones:
1. Configuración de modelos
2. Configuración de imágenes
3. Parámetros de visualización
4. Parámetros de generación de features
5. Configuración de UI
===================================================================
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ===================================================================
# 1. CONFIGURACIÓN DE MODELOS
# ===================================================================

# Modelos disponibles en la aplicación
AVAILABLE_MODELS = {
    'alexnet': {
        'name': 'AlexNet',
        'description': 'Red clásica de ImageNet (8 capas)',
        'size': '~244 MB',
        'layers': ['features.0', 'features.3', 'features.6', 'features.8', 'features.10']
    },
    'resnet18': {
        'name': 'ResNet-18',
        'description': 'Red residual de 18 capas',
        'size': '~44 MB',
        'layers': ['layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer4.0.conv1']
    },
    'vgg16': {
        'name': 'VGG-16',
        'description': 'Red profunda uniforme (16 capas)',
        'size': '~528 MB',
        'layers': ['features.0', 'features.5', 'features.10', 'features.17', 'features.24']
    }
}

# Modelo por defecto al iniciar la app
DEFAULT_MODEL = 'alexnet'

# Device
DEFAULT_DEVICE = 'cpu'  # Cambiar a 'cuda' si hay GPU disponible

# ===================================================================
# 2. CONFIGURACIÓN DE IMÁGENES
# ===================================================================

# Tamaño de entrada para los modelos (ImageNet standard)
IMAGE_SIZE: Tuple[int, int] = (224, 224)

# Tamaño máximo de archivo permitido (en MB)
MAX_UPLOAD_SIZE_MB = 10

# Formatos soportados
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']

# Normalización ImageNet (mean y std por canal RGB)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ===================================================================
# 3. PARÁMETROS DE VISUALIZACIÓN
# ===================================================================

# Mapas de calor
HEATMAP_COLORMAP = 'jet'  # Opciones: 'jet', 'hot', 'viridis', 'plasma'
HEATMAP_ALPHA = 0.5       # Transparencia del overlay (0.0 - 1.0)
HEATMAP_INTERPOLATION = 'bilinear'  # Interpolación al redimensionar

# ROI (Region of Interest)
ROI_SIZE: Tuple[int, int] = (64, 64)  # Tamaño del recorte de máxima activación
ROI_BORDER_COLOR = 'red'              # Color del borde del ROI
ROI_BORDER_WIDTH = 2                  # Grosor del borde en píxeles
ROI_MARKER_SIZE = 100                 # Tamaño de la estrella marcadora

# Figuras matplotlib
FIGURE_DPI = 100
FIGURE_STYLE = 'default'  # Estilo de matplotlib

# ===================================================================
# 4. PARÁMETROS DE GENERACIÓN DE FEATURES (Gradient Ascent)
# ===================================================================

# Optimización
FEATURE_ITERATIONS = 500        # Número de iteraciones
FEATURE_LR = 0.1               # Learning rate
FEATURE_OPTIMIZER = 'adam'     # Tipo de optimizador

# Regularización
L2_DECAY = 1e-4               # Peso de regularización L2 (controla intensidad)
TV_WEIGHT = 1e-2              # Peso de Total Variation (controla suavidad)

# Transformaciones aleatorias (data augmentation)
JITTER = 4                    # Píxeles de traslación aleatoria
ROTATION_RANGE = 5.0          # Rango de rotación en grados
SCALE_RANGE = (0.95, 1.05)    # Rango de escala (min, max)

# Blur periódico
BLUR_FREQUENCY = 4            # Cada cuántas iteraciones aplicar blur
BLUR_KERNEL_SIZE = 3          # Tamaño del kernel de blur

# Progress reporting
VERBOSE_FREQUENCY = 100       # Cada cuántas iteraciones mostrar progreso

# ===================================================================
# 5. CONFIGURACIÓN DE UI (Streamlit)
# ===================================================================

# Ranking de neuronas
MAX_NEURONS_DISPLAY = 10      # Top-K neuronas a mostrar en el ranking
DEFAULT_NEURON_INDEX = 0      # Neurona seleccionada por defecto

# Layout
PAGE_TITLE = "🎨 Neural Feature Visualization Dashboard"
PAGE_ICON = "🧠"
LAYOUT = "wide"               # 'centered' o 'wide'

# Sidebar
SIDEBAR_STATE = "expanded"    # 'expanded' o 'collapsed'

# Tabs
TAB_NAMES = [
    "📤 Carga de Imagen",
    "🔥 Mapa de Calor",
    "🔬 Comparación Real vs Ideal"
]

# Colores para alertas
COLOR_SUCCESS = "#28a745"
COLOR_INFO = "#17a2b8"
COLOR_WARNING = "#ffc107"
COLOR_ERROR = "#dc3545"

# ===================================================================
# 6. PATHS Y DIRECTORIOS
# ===================================================================

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent

# Directorio para assets
ASSETS_DIR = BASE_DIR / "assets"
SAMPLE_IMAGES_DIR = ASSETS_DIR / "sample_images"

# Crear directorios si no existen
ASSETS_DIR.mkdir(exist_ok=True)
SAMPLE_IMAGES_DIR.mkdir(exist_ok=True)

# ===================================================================
# 7. MENSAJES Y TEXTOS
# ===================================================================

WELCOME_MESSAGE = """
Bienvenido al **Neural Feature Visualization Dashboard** 🧠

Esta herramienta te permite:
- 📸 Cargar imágenes y analizar cómo las ve una red neuronal
- 🔥 Visualizar mapas de activación de neuronas específicas
- 🎨 Generar patrones sintéticos que maximizan activaciones
- 🔬 Comparar regiones reales vs patrones ideales
- En capas profundas se vuelve un poco complejo interpretar el filtro sobre la imgen, ¡pero es fascinante!

**¡Comienza subiendo una imagen o usando una de muestra!**
"""

HELP_MODEL_SELECTION = """
**Selección de Modelo:**
- **AlexNet**: Clásica, rápida, buena para comenzar
- **ResNet-18**: Más moderna, usa skip connections
- **VGG-16**: Más profunda, mejor accuracy pero más lenta
"""

HELP_LAYER_SELECTION = """
**Selección de Capa:**
- **Capas tempranas** (conv1, conv2): Detectan bordes, colores, texturas simples
- **Capas medias** (conv3, conv4): Detectan partes de objetos
- **Capas profundas** (conv5): Detectan objetos completos y conceptos abstractos
"""

HELP_NEURON_ACTIVATION = """
**Activación de Neurona:**
La activación indica qué tan "emocionada" está la neurona con la entrada.
- **Alta activación**: La neurona detectó su patrón preferido
- **Baja activación**: El patrón no está presente
"""

HELP_SYNTHETIC_GENERATION = """
**Generación Sintética:**
Usa **Gradient Ascent** para crear una imagen que maximiza la activación de la neurona.
Esto nos muestra qué patrón "busca" la neurona idealmente.

⚠️ La generación puede tardar 10-30 segundos dependiendo de tu hardware.
"""

# ===================================================================
# 8. CONFIGURACIÓN AVANZADA (OPCIONAL)
# ===================================================================

# Cache
ENABLE_MODEL_CACHE = True     # Mantener modelos en memoria
ENABLE_ACTIVATION_CACHE = True  # Cachear activaciones

# Logging
LOG_LEVEL = "INFO"            # DEBUG, INFO, WARNING, ERROR
ENABLE_PROFILING = False      # Perfilar performance

# Experimental
ENABLE_GPU_IF_AVAILABLE = True  # Detectar y usar GPU automáticamente
ENABLE_MIXED_PRECISION = False  # FP16 para mayor velocidad (requiere GPU)

# ===================================================================
# 9. FUNCIONES DE UTILIDAD
# ===================================================================


def get_model_info(model_name: str) -> Dict:
    """
    Retorna información sobre un modelo específico.

    Args:
        model_name: Nombre del modelo (ej: 'alexnet')

    Returns:
        Diccionario con información del modelo
    """
    return AVAILABLE_MODELS.get(model_name, {})


def get_default_layers(model_name: str) -> List[str]:
    """
    Retorna las capas por defecto para un modelo.

    Args:
        model_name: Nombre del modelo

    Returns:
        Lista de nombres de capas
    """
    model_info = get_model_info(model_name)
    return model_info.get('layers', [])


def validate_config():
    """
    Valida que la configuración sea coherente.
    Lanza excepciones si hay problemas.
    """
    assert HEATMAP_ALPHA >= 0.0 and HEATMAP_ALPHA <= 1.0, \
        "HEATMAP_ALPHA debe estar entre 0.0 y 1.0"

    assert FEATURE_ITERATIONS > 0, \
        "FEATURE_ITERATIONS debe ser positivo"

    assert L2_DECAY >= 0 and TV_WEIGHT >= 0, \
        "Pesos de regularización deben ser no-negativos"

    assert DEFAULT_MODEL in AVAILABLE_MODELS, \
        f"DEFAULT_MODEL '{DEFAULT_MODEL}' no está en AVAILABLE_MODELS"

    print("✅ Configuración validada correctamente")


# Validar configuración al importar el módulo
if __name__ != "__main__":
    validate_config()
