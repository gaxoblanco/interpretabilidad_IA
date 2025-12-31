# 🎨 Neural Feature Visualization Dashboard

Dashboard interactiva para visualizar y explorar qué patrones aprenden las redes neuronales convolucionales.

![Dashboard Preview](assets/preview.png)

## 🌟 Características

- **📸 Carga de Imágenes**: Sube tus propias imágenes o usa muestras predefinidas
- **🔥 Mapas de Calor**: Visualiza qué regiones activan más las neuronas
- **🎯 Análisis por Neurona**: Explora neuronas individuales y sus patrones
- **🎨 Feature Synthesis**: Genera patrones sintéticos que maximizan activaciones
- **🔬 Comparación Real vs Ideal**: Compara regiones reales con patrones sintéticos
- **⚡ Múltiples Modelos**: Soporta AlexNet, ResNet-18, VGG-16 y más

## 🚀 Inicio Rápido

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/neural-dashboard.git
cd neural-dashboard

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución

```bash
# Ejecutar la dashboard
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📖 Uso

### 1. Cargar Imagen

- **Opción A**: Sube tu propia imagen (JPG, PNG)
- **Opción B**: Usa una imagen de muestra

### 2. Configurar Modelo

Selecciona:
- **Modelo**: AlexNet, ResNet-18 o VGG-16
- **Capa**: Cualquier capa convolucional del modelo

💡 **Tip**: Las capas tempranas detectan patrones simples (bordes, colores), mientras que las capas profundas detectan conceptos más abstractos.

### 3. Generar Mapa de Calor

Haz click en "🔍 Generar Mapa de Calor" para:
- Ver qué regiones de la imagen activan más las neuronas
- Obtener un ranking de las neuronas más activas
- Identificar la región de interés (ROI)

### 4. Comparación Real vs Sintética

1. Selecciona una neurona del ranking
2. Haz click en "🎨 Generar Comparación"
3. Observa:
   - **Región Real**: Parte de la imagen que más activa la neurona
   - **Patrón Sintético**: Imagen ideal que maximiza la activación
   - **Superposición 50/50**: Comparación visual directa

## 🏗️ Arquitectura

```
neural_dashboard/
├── app.py                     # Aplicación principal (Streamlit)
├── config.py                  # Configuración global
├── requirements.txt           # Dependencias
│
├── modules/                   # Módulos funcionales
│   ├── model_manager.py      # Gestión de modelos
│   ├── image_processor.py    # Procesamiento de imágenes
│   ├── neuron_analyzer.py    # Análisis de activaciones
│   ├── feature_generator.py  # Generación de patrones sintéticos
│   └── visualizer.py         # Creación de visualizaciones
│
├── utils/                     # Utilidades
│   ├── cache_manager.py      # Gestión de caché
│   └── helpers.py            # Funciones auxiliares
│
└── assets/                    # Recursos estáticos
    └── sample_images/        # Imágenes de muestra
```

## 🧠 Conceptos Técnicos

### Feature Visualization

La visualización de features usa **gradient ascent** para generar imágenes que maximizan la activación de neuronas específicas:

```
x* = argmax_x a_i^l(x)
```

Donde:
- `x*` es la imagen óptima
- `a_i^l` es la activación de la neurona `i` en la capa `l`

### Regularizaciones

Para obtener imágenes interpretables, aplicamos:

1. **L2 Decay**: Penaliza valores extremos
   ```
   L_l2 = λ₁ ||x||²
   ```

2. **Total Variation**: Suaviza la imagen
   ```
   L_tv = λ₂ Σ|x_{i,j} - x_{i+1,j}| + |x_{i,j} - x_{i,j+1}|
   ```

3. **Transformaciones Aleatorias**: Jitter, rotación, escala para robustez

## ⚙️ Configuración

Edita `config.py` para personalizar:

```python
# Modelos
AVAILABLE_MODELS = ['alexnet', 'resnet18', 'vgg16']

# Feature Generation
FEATURE_ITERATIONS = 500   # Iteraciones de optimización
FEATURE_LR = 0.1          # Learning rate
L2_DECAY = 1e-4           # Regularización L2
TV_WEIGHT = 1e-2          # Total Variation

# Visualización
HEATMAP_COLORMAP = 'jet'  # Colormap: 'jet', 'hot', 'viridis'
HEATMAP_ALPHA = 0.5       # Transparencia
```

## 📊 Ejemplos de Uso

### Analizar Detectores de Bordes (Capas Tempranas)

```
1. Cargar imagen con bordes claros
2. Seleccionar AlexNet / features.0
3. Observar neuronas que detectan orientaciones específicas
```

### Explorar Detectores de Objetos (Capas Profundas)

```
1. Cargar imagen con objetos reconocibles
2. Seleccionar AlexNet / features.10
3. Comparar patrones sintéticos con regiones de objetos
```

## 🔧 Módulos Principales

### ModelManager

Gestiona carga y consulta de modelos:

```python
from modules.model_manager import ModelManager

manager = ModelManager()
model = manager.load_model('alexnet')
layers = manager.get_conv_layers(model)
```

### NeuronAnalyzer

Analiza activaciones neuronales:

```python
from modules.neuron_analyzer import NeuronAnalyzer

analyzer = NeuronAnalyzer(model, 'features.0')
activations = analyzer.extract_activations(image_tensor)
heatmap = analyzer.compute_heatmap(activations)
```

### FeatureGenerator

Genera patrones sintéticos:

```python
from modules.feature_generator import FeatureGenerator

generator = FeatureGenerator(model, 'features.0')
synthetic_img, history = generator.generate_pattern(neuron_idx=38)
```

## 🎯 Roadmap

- [ ] Soporte para más arquitecturas (EfficientNet, Vision Transformers)
- [ ] Exportación de resultados (PDF, PNG)
- [ ] Análisis comparativo entre modelos
- [ ] Modo batch para múltiples imágenes
- [ ] Integración con datasets públicos (ImageNet, COCO)
- [ ] Visualización de rutas de activación completas

## 🐛 Troubleshooting

### Problema: Imágenes sintéticas muy ruidosas

**Solución**: Aumentar `TV_WEIGHT` en `config.py`

```python
TV_WEIGHT = 5e-2  # Mayor suavizado
```

### Problema: Activación no aumenta durante optimización

**Solución**: Aumentar learning rate o iteraciones

```python
FEATURE_LR = 0.2
FEATURE_ITERATIONS = 1000
```

### Problema: Out of Memory (GPU)

**Solución**: La app detecta automáticamente y usa CPU. Para forzar CPU:

```python
DEFAULT_DEVICE = 'cpu'
```

## 📚 Referencias

- Olah et al. (2017) - [Feature Visualization](https://distill.pub/2017/feature-visualization/)
- Mordvintsev et al. (2015) - [DeepDream](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
- Simonyan et al. (2013) - [Deep Inside Convolutional Networks](https://arxiv.org/abs/1312.6034)

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 👥 Autores

- **Neural Viz Team** - *Desarrollo inicial*

## 🙏 Agradecimientos

- PyTorch Team por el framework
- Streamlit Team por la plataforma de dashboards
- Distill.pub por la investigación en interpretabilidad

---

**¿Preguntas o feedback?** Abre un issue en GitHub o contacta al equipo.

**¡Disfruta explorando las redes neuronales! 🧠✨**