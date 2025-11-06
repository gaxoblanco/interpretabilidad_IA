# 📚 FUNCIONES IMPLEMENTADAS EN `src/`

Este documento detalla todas las funciones y clases implementadas en el módulo III del proyecto de interpretabilidad.

---

## 📁 Estructura de Archivos

```
interpretability_III/src/
├── models/
│   ├── __init__.py
│   └── model_loader.py
├── utils/
│   ├── __init__.py
│   ├── image_loader.py
│   ├── analyze_neuron.py
│   ├── neuron_activation.py
│   └── hooks.py
└── (otros módulos pendientes)
```

---

## 🧠 `src/models/model_loader.py`

### **Clase Principal: `ModelLoader`**

Carga y analiza modelos pre-entrenados de torchvision.

#### **Constructor:**
```python
ModelLoader(
    model_name: str = 'resnet18',
    pretrained: bool = True,
    num_classes: int = 1000,
    device: Optional[str] = None
)
```
- **Modelos soportados:** resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19, alexnet, densenet121, mobilenet_v2

#### **Métodos Principales:**

##### `load_model() -> nn.Module`
- Carga el modelo con o sin pesos pre-entrenados
- Mueve el modelo al device (GPU/CPU)
- Configura en modo evaluación por defecto
- **Retorna:** Modelo cargado

##### `get_architecture_info() -> Dict`
- **Retorna diccionario con:**
  - `total_params`: Total de parámetros
  - `trainable_params`: Parámetros entrenables
  - `frozen_params`: Parámetros congelados
  - `model_size_mb`: Tamaño del modelo en MB
  - `num_layers`: Número de capas
  - `params_by_type`: Distribución de parámetros por tipo de capa

##### `get_layers_info() -> List[Dict]`
- **Retorna lista de diccionarios con info de cada capa:**
  - `name`: Nombre de la capa
  - `type`: Tipo de capa (Conv2d, Linear, etc.)
  - `params`: Número de parámetros
  - `trainable`: Si los parámetros son entrenables
  - `output_shape`: Shape de salida (si está disponible)

##### `get_residual_blocks_info() -> List[Dict]`
- **Solo para ResNet**
- **Retorna lista con info de cada bloque residual:**
  - `name`: Nombre del bloque (e.g., 'layer1.0')
  - `num_layers`: Número de capas en el bloque
  - `params`: Número de parámetros
  - `in_channels`: Canales de entrada
  - `out_channels`: Canales de salida
  - `stride`: Stride del bloque
  - `has_downsample`: Si tiene capa de downsampling

##### `get_layer_by_name(layer_name: str) -> Optional[nn.Module]`
- Obtiene una capa específica por su nombre
- **Ejemplo:** `get_layer_by_name('layer1.0.conv1')`

##### `freeze_layers(layer_names: Optional[List[str]] = None)`
- Congela capas específicas para no entrenarlas
- Si `layer_names=None`, congela todas las capas

##### `unfreeze_layers(layer_names: Optional[List[str]] = None)`
- Descongela capas específicas para entrenarlas
- Si `layer_names=None`, descongela todas las capas

##### `set_eval_mode()`
- Configura el modelo en modo evaluación (desactiva dropout, batch norm)

##### `set_train_mode()`
- Configura el modelo en modo entrenamiento

##### `get_model_summary() -> str`
- Genera un resumen legible del modelo con todas las estadísticas

#### **Funciones Auxiliares:**

##### `list_available_models() -> List[str]`
- Lista todos los modelos disponibles para cargar

##### `get_model_info(model_name: str) -> Dict`
- Información básica de un modelo sin cargarlo

---

## 🖼️ `src/utils/image_loader.py`

### **Clase Principal: `ImageLoader`**

Carga y procesa datasets de imágenes (CIFAR-10, CIFAR-100).

#### **Constructor:**
```python
ImageLoader(
    dataset_name: str = 'cifar10',
    batch_size: int = 32,
    num_workers: int = 2,
    data_dir: Optional[str] = None,
    download: bool = True,
    shuffle_train: bool = True,
    pin_memory: bool = True
)
```
- **Datasets soportados:** cifar10, cifar100

#### **Constantes de Clase:**
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Normalización ImageNet
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
```

#### **Métodos Principales:**

##### `load_datasets()`
- Carga los datasets de entrenamiento y prueba
- Aplica transformaciones apropiadas
- Descarga automáticamente si es necesario

##### `get_dataloaders() -> Tuple[DataLoader, DataLoader]`
- Crea y retorna dataloaders de entrenamiento y prueba
- **Retorna:** (train_loader, test_loader)

##### `get_dataset_info() -> Dict`
- **Retorna diccionario con:**
  - `name`: Nombre del dataset
  - `num_classes`: Número de clases
  - `image_size`: Tamaño de las imágenes
  - `train_samples`: Número de muestras de entrenamiento
  - `test_samples`: Número de muestras de prueba
  - `classes`: Lista de nombres de clases
  - `normalization_mean`: Media de normalización
  - `normalization_std`: Desviación estándar de normalización

##### `denormalize_image(tensor: torch.Tensor) -> np.ndarray`
- Denormaliza una imagen tensor para visualización
- **Input:** Tensor [C, H, W] o [B, C, H, W]
- **Output:** Array numpy [H, W, C] con valores en [0, 1]

##### `normalize_image(img: np.ndarray) -> torch.Tensor`
- Normaliza una imagen numpy para input al modelo
- **Input:** Array [H, W, C] con valores en [0, 255] o [0, 1]
- **Output:** Tensor normalizado [C, H, W]

##### `get_class_distribution(split: str = 'train') -> Dict[str, int]`
- Calcula la distribución de clases en el dataset
- **Retorna:** {nombre_clase: número_de_muestras}

##### `get_sample_images(num_images: int = 16, split: str = 'test', random: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`
- Obtiene un conjunto de imágenes de muestra
- **Retorna:** (images, labels) como tensors

##### `load_custom_image(image_path: str, resize: Optional[Tuple[int, int]] = None) -> torch.Tensor`
- Carga y procesa una imagen personalizada
- **Retorna:** Tensor [1, C, H, W] listo para el modelo

##### `get_statistics() -> Dict`
- Calcula estadísticas del dataset (media, std por canal)
- **Nota:** Puede ser lento para datasets grandes

#### **Funciones Auxiliares:**

##### `visualize_augmentations(dataset: Dataset, num_examples: int = 4)`
- Visualiza ejemplos de data augmentation

##### `calculate_dataset_mean_std(dataloader: DataLoader) -> Tuple[List[float], List[float]]`
- Calcula la media y desviación estándar de un dataset

---

## 🪝 `src/utils/hooks.py`

### **Clase Principal: `ActivationHook`**

Sistema de hooks para capturar activaciones de capas intermedias.

#### **Constructor:**
```python
ActivationHook(
    model: nn.Module,
    target_layers: Optional[List[str]] = None
)
```
- Si `target_layers=None`, registra hooks en todas las capas Conv2d

#### **Atributos:**
```python
self.model          # Modelo de PyTorch
self.target_layers  # Lista de nombres de capas objetivo
self.activations    # OrderedDict con activaciones capturadas
self.gradients      # OrderedDict con gradientes capturados
self.hooks          # Lista de handles de hooks registrados
```

#### **Métodos Principales:**

##### `register_hooks(capture_gradients: bool = False)`
- Registra hooks en las capas objetivo
- Si `capture_gradients=True`, también captura gradientes (backward hooks)

##### `remove_hooks()`
- Remueve todos los hooks registrados
- Libera recursos

##### `get_activations() -> Dict[str, torch.Tensor]`
- Obtiene las activaciones capturadas
- **Retorna:** {layer_name: activation_tensor}

##### `get_gradients() -> Dict[str, torch.Tensor]`
- Obtiene los gradientes capturados
- Requiere `capture_gradients=True` en `register_hooks()`
- Requiere haber ejecutado backward pass

##### `clear_activations()`
- Limpia las activaciones almacenadas
- Útil para liberar memoria entre forward passes

##### `get_activation_shapes() -> Dict[str, Tuple]`
- **Retorna:** {layer_name: shape_tuple}

##### `get_activation_statistics() -> Dict[str, Dict]`
- Calcula estadísticas completas de las activaciones
- **Retorna diccionario por capa con:**
  - `mean`: Media de la activación
  - `std`: Desviación estándar
  - `min`: Valor mínimo
  - `max`: Valor máximo
  - `sparsity`: Proporción de valores == 0
  - `active_neurons`: Número de neuronas activas
  - `total_neurons`: Total de neuronas
  - `shape`: Shape del tensor

##### `get_layer_names() -> List[str]`
- Obtiene la lista de nombres de capas objetivo

##### `has_activations() -> bool`
- Verifica si hay activaciones capturadas

##### `get_activation_for_layer(layer_name: str) -> Optional[torch.Tensor]`
- Obtiene la activación de una capa específica

##### `save_activations(filepath: str)`
- Guarda las activaciones en un archivo .pth

##### `load_activations(filepath: str)`
- Carga activaciones desde un archivo .pth

#### **Funciones Auxiliares:**

##### `get_all_layer_names(model: nn.Module) -> List[str]`
- Obtiene todos los nombres de capas del modelo

##### `get_layer_types(model: nn.Module) -> Dict[str, List[str]]`
- Agrupa las capas del modelo por tipo
- **Retorna:** {tipo_de_capa: [nombres_de_capas]}

##### `compare_activations(activations1: Dict, activations2: Dict) -> Dict[str, Dict]`
- Compara dos conjuntos de activaciones
- **Retorna métricas por capa:**
  - `mean_abs_diff`: Diferencia absoluta media
  - `max_abs_diff`: Diferencia absoluta máxima
  - `cosine_similarity`: Similitud coseno
  - `correlation`: Correlación

##### `find_dead_neurons(activations: Dict, threshold: float = 0.0) -> Dict[str, List[int]]`
- Identifica neuronas "muertas" (que nunca se activan)
- **Retorna:** {layer_name: [indices_de_neuronas_muertas]}

##### `analyze_sparsity(activations: Dict) -> Dict[str, float]`
- Analiza la sparsity de las activaciones
- **Retorna:** {layer_name: sparsity_percentage}

---

---

## 🎨 `src/utils/neuron_activation.py`

### **Funciones Principales de Análisis**

Módulo para visualizar y analizar mapas de activación neuronal de imágenes individuales.

#### **Función: `analyze_single_image_activation()`**
```python
analyze_single_image_activation(
    model: nn.Module,
    image: torch.Tensor,
    target_layers: List[str],
    threshold: float = 0.1,
    device: torch.device = torch.device('cpu'),
    hook_class = None
) -> Tuple[Dict, int]
```

**Propósito:** Analiza qué neuronas se activan para una imagen específica

**Parámetros:**
- `model`: Modelo de PyTorch en modo eval
- `image`: Imagen tensor [C, H, W] o [1, C, H, W]
- `target_layers`: Lista de nombres de capas a analizar
- `threshold`: Umbral para considerar neurona "activa" (default: 0.1)
- `device`: Device donde ejecutar (CPU/GPU)
- `hook_class`: Clase ActivationHook (debe ser pasada)

**Retorna:** 
- Tupla de (activation_summary: Dict, pred_class: int)
- `activation_summary` contiene por capa:
  - `activations`: Array con valores por neurona
  - `active_indices`: Índices de neuronas activas
  - `num_active`: Número de neuronas activas
  - `total_neurons`: Total de neuronas
  - `activation_rate`: Porcentaje de activación
  - `max_activation`: Activación máxima
  - `mean_activation`: Activación promedio
  - `mean_active_only`: Promedio solo de neuronas activas

---

#### **Función: `plot_neuron_activation_map()`**
```python
plot_neuron_activation_map(
    image: torch.Tensor,
    activation_summary: Dict,
    pred_class: int,
    class_names: List[str],
    true_label: Optional[int] = None,
    threshold: float = 0.1,
    figsize: Tuple[int, int] = None,
    max_neurons_display: int = 64
) -> plt.Figure
```

**Propósito:** Visualiza el mapa de activación neuronal para una imagen

**Crea una figura con:**
- Imagen original en la parte superior
- Para cada capa: gráfico de barras de activaciones + histograma

**Parámetros:**
- `image`: Imagen tensor [C, H, W] o [1, C, H, W]
- `activation_summary`: Dict retornado por analyze_single_image_activation
- `pred_class`: Clase predicha (índice)
- `class_names`: Lista de nombres de clases
- `true_label`: Clase real (opcional, para comparación)
- `threshold`: Umbral usado para marcar neuronas activas
- `figsize`: Tamaño de la figura (ancho, alto)
- `max_neurons_display`: Máximo de neuronas a mostrar

**Retorna:** Figura de matplotlib

---

#### **Función: `print_activation_statistics()`**
```python
print_activation_statistics(
    activation_summary: Dict,
    top_k: int = 5
)
```

**Propósito:** Imprime estadísticas detalladas de activaciones

**Muestra por capa:**
- Total de neuronas
- Neuronas activas (número y porcentaje)
- Activación máxima, promedio y promedio de activas
- Top K neuronas más activas

---

#### **Función: `compare_activations()`**
```python
compare_activations(
    summary1: Dict,
    summary2: Dict,
    label1: str = "Imagen 1",
    label2: str = "Imagen 2"
)
```

**Propósito:** Compara activaciones entre dos imágenes

**Muestra por capa:**
- Número de neuronas activas en cada imagen
- Neuronas comunes
- Neuronas exclusivas de cada imagen
- Similitud de Jaccard

---

#### **Función: `find_specialized_neurons()`**
```python
find_specialized_neurons(
    activation_summaries: List[Dict],
    layer_name: str,
    min_activation_rate: float = 0.8
) -> np.ndarray
```

**Propósito:** Encuentra neuronas que se activan consistentemente en múltiples imágenes

**Parámetros:**
- `activation_summaries`: Lista de resúmenes de activación
- `layer_name`: Nombre de la capa a analizar
- `min_activation_rate`: Tasa mínima de activación (0-1)

**Retorna:** Array con índices de neuronas especializadas

**Uso típico:** Identificar "detectores" especializados en una clase (ej: detectores de perros)

---

### **Funciones Auxiliares Internas**

#### `_denormalize_image(image: torch.Tensor) -> torch.Tensor`
- Denormaliza imagen con stats de ImageNet
- **Input:** [C, H, W] o [1, C, H, W]
- **Output:** [C, H, W] en rango [0, 1]

#### `_plot_activation_bars(ax, activations, active_indices, total_neurons, layer_name, summary, max_display)`
- Dibuja gráfico de barras de activaciones
- Colores: rojo para activas, gris para inactivas
- Si >64 neuronas, muestra solo las más activas

#### `_plot_activation_histogram(ax, activations, threshold)`
- Dibuja histograma de distribución
- Solo valores > 0 (excluye sparsity)
- Línea vertical en threshold

---

## 🔬 `src/utils/analyze_neuron.py`

### **Funciones de Análisis Avanzado de Neuronas**

Funciones para análisis detallado de comportamiento de neuronas individuales.

#### **Función: `analyze_spatial_bias()` (static)**
```python
@staticmethod
analyze_spatial_bias(
    neuron_index: int,
    layer_name: str,
    concatenated_activations: dict,
    num_samples: int = 50,
    verbose: bool = True
) -> dict
```

**Propósito:** Analiza si una neurona tiene sesgo espacial (izquierda vs derecha, arriba vs abajo)

**Parámetros:**
- `neuron_index`: Índice de la neurona a analizar
- `layer_name`: Nombre de la capa (ej: 'layer3.1.relu')
- `concatenated_activations`: Dict con activaciones capturadas
- `num_samples`: Número de imágenes a analizar
- `verbose`: Si True, imprime resultados detallados

**Retorna:** Dict con análisis completo:
- `neuron_info`: Información básica de la neurona
- `horizontal_bias`: Análisis izquierda/derecha
  - `bias_type`: Tipo de sesgo ("FUERTE hacia IZQUIERDA", etc.)
  - `left_mean`, `right_mean`: Activaciones promedio
  - `ratio`: Ratio izquierda/derecha
  - `left_dominant_pct`, `right_dominant_pct`: Porcentajes de dominancia
- `vertical_bias`: Análisis arriba/abajo
  - `bias_type`: Tipo de sesgo
  - `top_mean`, `bottom_mean`: Activaciones promedio
  - `ratio`: Ratio arriba/abajo
- `dominant_bias`: Sesgo dominante general
- `_raw_data`: Datos crudos para análisis adicional

**Categorías de sesgo:**
- Sin sesgo: ratio entre 0.91 y 1.1
- Moderado: ratio entre 0.77-0.91 o 1.1-1.3
- Fuerte: ratio < 0.77 o > 1.3

**Uso típico:** Identificar si una neurona detecta features en posiciones específicas de la imagen

---

#### **Función: `analyze_neuron_correlation_with_visual_features()` (static)**
```python
@staticmethod
analyze_neuron_correlation_with_visual_features(
    neuron_idx: int,
    layer_name: str,
    concatenated_activations: dict,
    test_images: torch.Tensor,
    num_samples: int = 50,
    plot: bool = True
) -> dict
```

**Propósito:** Analiza correlación entre activaciones de neurona y características visuales de imagen

**Características analizadas:**
1. **Brillo promedio**: Luminosidad general de la imagen
2. **Contraste**: Diferencia entre píxeles claros y oscuros
3. **Bordes (Sobel)**: Cantidad de bordes detectados
4. **Varianza**: Variabilidad de píxeles
5. **Entropía**: Complejidad de la imagen
6. **Gradiente resized (2×2)**: Transiciones de intensidad

**Parámetros:**
- `neuron_idx`: Índice de neurona a analizar
- `layer_name`: Nombre de capa
- `concatenated_activations`: Dict con activaciones
- `test_images`: Tensor de imágenes [B, C, H, W]
- `num_samples`: Número de imágenes
- `plot`: Si True, muestra gráficos

**Retorna:** Dict con:
- `neuron_info`: Información básica
- `features`: Dict con valores de cada característica visual
- `correlations`: Dict con correlaciones de Pearson
  - `sorted`: Lista ordenada de (feature, correlación)
  - Por feature: valor de correlación
- `interpretation`: Interpretación automática
- `_raw_visualizations`: Datos para gráficos

**Interpretación automática:**
- Correlación alta (+0.5 a +1.0): Relación positiva fuerte
- Correlación moderada (+0.3 a +0.5): Relación positiva moderada
- Sin correlación (-0.3 a +0.3): Sin relación clara
- Correlación negativa (-0.5 a -0.3): Relación inversa moderada
- Correlación muy negativa (-1.0 a -0.5): Relación inversa fuerte

**Visualizaciones generadas (si plot=True):**
1. Comparación de imágenes con características visuales
2. Gráfico de barras de correlaciones

**Uso típico:** 
- Entender qué características visuales activan una neurona
- Identificar "qué busca" cada neurona (bordes, contraste, texturas, etc.)

---

#### **Funciones Auxiliares en analyze_neuron.py:**

##### `visualize_neuron_activation_map()` (versión simplificada)
- Similar a la función principal pero más básica
- Retorna activation_summary y pred_class

##### `plot_activation_map()`
- Visualiza mapa con imagen + barras + histogramas por capa
- Versión completa con múltiples subplots

---

## 📝 Ejemplos de Uso

### **Cargar Modelo:**
```python
from src.models.model_loader import ModelLoader

loader = ModelLoader('resnet18', pretrained=True)
model = loader.load_model()
arch_info = loader.get_architecture_info()
```

### **Cargar Dataset:**
```python
from src.utils.image_loader import ImageLoader

img_loader = ImageLoader('cifar10', batch_size=32)
train_loader, test_loader = img_loader.get_dataloaders()
dataset_info = img_loader.get_dataset_info()
```

### **Capturar Activaciones:**
```python
from src.utils.hooks import ActivationHook

target_layers = ['conv1', 'layer1.0.conv1', 'layer4.0.conv1']
hook = ActivationHook(model, target_layers)
hook.register_hooks()

# Forward pass
output = model(input_tensor)

# Obtener activaciones
activations = hook.get_activations()
stats = hook.get_activation_statistics()

# Limpiar
hook.clear_activations()
hook.remove_hooks()
```

---

## ✅ Estado de Implementación

| Módulo | Archivo | Estado | Funciones |
|--------|---------|--------|-----------|
| **models** | model_loader.py | ✅ Completo | 13 métodos |
| **utils** | image_loader.py | ✅ Completo | 12 métodos |
| **utils** | hooks.py | ✅ Completo | 16 métodos |
| **interpretability** | (pendiente) | ❌ Por implementar | - |
| **visualization** | (pendiente) | ❌ Por implementar | - |

---

## 🎯 Próximas Implementaciones (Notebook 02+)

### **`src/interpretability/`** (Pendiente)
- `activation_extractor.py`: Extraer activaciones de todas las capas
- `feature_visualizer.py`: Generar feature visualizations
- `neuron_probe.py`: Probing classifiers
- `activation_analyzer.py`: Análisis estadístico avanzado

### **`src/visualization/`** (Pendiente)
- `heatmap_viz.py`: Mapas de calor de activaciones
- `filter_viz.py`: Visualización de filtros
- `layer_viz.py`: Visualización por capas

---

## 📌 Notas Importantes

1. **Todas las funciones están documentadas** con docstrings completos
2. **Logging integrado** para debugging
3. **Manejo de errores** con mensajes descriptivos
4. **Type hints** en todos los métodos
5. **Ejemplos de uso** al final de cada archivo

---

**Última actualización:** 2025-01-15
**Versión del proyecto:** Módulo III - Notebook 01 Completado