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