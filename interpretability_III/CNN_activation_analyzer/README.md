# 🔬 Analizador de Activaciones de CNN

Aplicación interactiva desarrollada con **Streamlit** para visualizar y analizar las activaciones internas de redes neuronales convolucionales (ResNet18 y AlexNet). Permite entender qué patrones detecta cada capa de la red y cómo se activan los filtros convolucionales ante diferentes imágenes.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Funcionalidades Principales](#-funcionalidades-principales)
- [Interpretación de Resultados](#-interpretación-de-resultados)
- [Ejemplos de Uso](#-ejemplos-de-uso)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)

---

## ✨ Características

### 🎯 Análisis Profundo de Activaciones
- **Múltiples modelos**: Soporte para ResNet18 y AlexNet preentrenados en ImageNet
- **Selección de capas**: Analiza cualquier capa convolucional del modelo
- **Filtrado inteligente**: Sistema de scoring balanceado que prioriza neuronas selectivas sobre fondos uniformes
- **Visualización RGB**: Descomposición de filtros en canales R, G, B individuales

### 📊 Visualizaciones Interactivas
- **6 pestañas especializadas** con diferentes perspectivas del análisis
- **Heatmaps superpuestos**: Mapas de calor sobre la imagen original
- **Grids de filtros**: Visualización en grilla de múltiples activaciones
- **Análisis detallado por filtro**: Regiones de activación, patrones RGB y estadísticas
- **Predicción del modelo**: Información sobre la clase detectada (ImageNet)

### 🔧 Controles Personalizables
- **Criterios de selección**: Balanced, mean, max, std
- **Parámetros ajustables**: Sparsity mínima, peso activación vs. selectividad
- **Visualización flexible**: Transparencia de heatmaps, colormaps, número de filtros
- **Carga de imágenes**: Desde URL o subida local

---

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/cnn-activation-analyzer.git
cd cnn-activation-analyzer
```

### Paso 2: Crear entorno virtual (recomendado)

```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Linux/Mac:
source venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

**Contenido de `requirements.txt`:**
```
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=9.5.0
scipy>=1.10.0
```

---

## 💻 Uso

### Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Flujo de trabajo básico

1. **Seleccionar modelo** (ResNet18 o AlexNet) en el sidebar
2. **Elegir capa** a analizar (ej: `layer2.0.conv1`)
3. **Configurar parámetros**:
   - Número de neuronas a visualizar (6 recomendado)
   - Criterio de selección (recomendado: `balanced`)
   - Sparsity mínima (recomendado: 0.15 para capas medias)
4. **Cargar imagen** (por URL o archivo local)
5. **Presionar "Analizar Activaciones"**
6. **Explorar resultados** en las 6 pestañas disponibles

---

## 📁 Estructura del Proyecto

```
cnn-activation-analyzer/
│
├── app.py                      # Aplicación principal de Streamlit
├── utils_streamlit.py          # Utilidades para análisis de activaciones
├── filter_visualization.py     # Funciones de visualización de filtros
├── imagenet_classes.py         # (Opcional) Diccionario de clases ImageNet
│
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Este archivo
│
└── examples/                   # (Opcional) Imágenes de ejemplo
    ├── cat.jpg
    ├── dog.jpg
    └── airplane.jpg
```

---

## 🎨 Funcionalidades Principales

### 📊 Tab 0: Resultados Generales
Visión global del análisis con métricas clave:
- **Predicción del modelo**: Clase detectada y nivel de confianza
- **Información de la capa**: Nombre, número de neuronas, dimensiones
- **Estadísticas globales**: Media, máximo, desviación estándar, sparsity
- **Top neuronas**: Tabla con las 6 neuronas más activas/selectivas

### 🔥 Tab 1: Heatmaps Superpuestos
Visualización de mapas de calor sobre la imagen original:
- **Grid 2×3**: Muestra 6 heatmaps simultáneamente
- **Selector individual**: Explora cada filtro en detalle
- **Parámetros ajustables**: Transparencia (alpha) y colormap

### 🎨 Tab 2: Grid de Filtros
Vista panorámica de todos los filtros analizados:
- **Grid compacto**: Hasta 24 filtros en una sola visualización
- **Numeración clara**: Cada filtro con su índice y ranking

### 🔬 Tab 3: Análisis Detallado ⭐
Análisis profundo de cada filtro con sub-pestañas (6 filtros):

**Por cada filtro:**
- **🖼️ Regiones + Patrón RGB**: 
  - Imagen con cajas coloreadas marcando regiones de activación
  - Regiones ordenadas por interés (prioriza patrones específicos)
  - Patrón RGB del filtro (kernel 7×7)

- **🎨 Descomposición RGB**:
  - Canal Rojo individual (visualizado en rojo)
  - Canal Verde individual (visualizado en verde)
  - Canal Azul individual (visualizado en azul)
  - Combinado RGB (mezcla real de colores)

- **🌡️ Mapa de Calor**: Heatmap detallado de activación

- **🔍 Panel de Análisis** (columna derecha):
  - Métricas: Media, Desv. Std, Máxima, Sparsity
  - Explicación textual del comportamiento
  - Advertencias de interpretación
  - Evaluación de coincidencia de patrón

### 🎯 Tab 4: Visualización de Filtros
Comparación de patrones vs. detecciones:
- **🎨 Patrones RGB**: Grid de kernels 7×7 de los 6 filtros top
- **📸 Patches detectados**: 3 regiones de imagen real que activaron cada filtro

### 🤖 Tab 5: Predicción del Modelo
Información sobre la clasificación ImageNet:
- **Predicción principal**: Clase + confianza
- **Interpretación**: Nivel de certeza del modelo
- **Información de ImageNet**: Contexto sobre las 1000 categorías

---

## 📖 Interpretación de Resultados

### Criterios de Selección

#### `balanced` ⭐ (Recomendado)
Combina activación y selectividad:
- **Activación alta**: El filtro responde fuertemente
- **Sparsity alta**: El filtro es selectivo
- **Evita fondos**: Filtra sparsity <10%

**Parámetros clave:**
- `Peso Activación vs Selectividad`: 0.5 = balance
- `Sparsity Mínima`: 0.15 = excluye filtros no selectivos

### Niveles de Sparsity

| Sparsity | Interpretación | Ejemplo |
|----------|----------------|---------|
| >70% | 🎯 Muy selectivo | Ojos, rayas específicas |
| 30-70% | ⚖️ Moderado | Texturas, bordes |
| 10-30% | 🌊 Poco selectivo | Colores comunes |
| <10% | ⚠️ No selectivo | Fondo, iluminación |

### Interpretación de Regiones

Ordenadas por **score de interés** (intensidad × selectividad):

- **⭐ Región 1**: Más interesante (patrón específico)
- **🔸 Pequeña**: <5% imagen (patrón específico)
- **🔹 Mediana**: 5-15% imagen  
- **🔷 Grande**: >30% imagen (posible fondo)

---

## 🎓 Ejemplos de Uso

### Caso 1: Entender un filtro específico

**Pasos**:
1. `ResNet18` → `layer2.0.conv1`
2. Criterio: `balanced`, Sparsity: 0.15
3. Cargar imagen de gato
4. **Tab 3** → **Filtro 38**

**Observar**:
- **Descomposición RGB**: Azul + Naranja
- **Regiones**: Transiciones pelaje/fondo
- **Sparsity**: 71% → Muy selectivo

### Caso 2: Filtrar fondos uniformes

**Problema**: Filtros detectan fondo azul

**Solución**:
1. Criterio: `balanced`
2. Sparsity mínima: 0.20-0.30
3. Peso activación: 0.3-0.4

**Resultado**: Solo patrones específicos del objeto

---

## 🔧 Tecnologías Utilizadas

- **Streamlit**: Framework de aplicaciones web
- **PyTorch**: Deep learning
- **Torchvision**: Modelos preentrenados
- **NumPy**: Operaciones numéricas
- **Matplotlib**: Visualizaciones
- **SciPy**: Procesamiento de imágenes
- **Pillow**: Manipulación de imágenes

---

## 📝 Notas Técnicas

### Limitaciones

1. **Visualización RGB**: Solo capas tempranas con entrada RGB directa
2. **Memoria**: ResNet18/AlexNet funcionan bien en CPU
3. **Clases ImageNet**: Diccionario con ~50 clases comunes incluidas

### GPU (Opcional)

Detección automática:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---
