📋 Planificado - Por definir duración


**Información actual:**
- **Objetivo:** Entender qué representan neuronas individuales en redes profundas
- **Técnicas:** Feature visualization, Activation maximization, Neuron probing
- **Modelos:** Por definir (CNN o Transformer según hallazgos del Proyecto 1)
- **Stack Tecnológico:** PyTorch/TensorFlow, Captum, Plotly, PIL

---

## 🎯 **PROPUESTA DETALLADA PARA EL MÓDULO III**

### **1. Objetivo Principal**

> **"¿Qué están aprendiendo las neuronas individuales en mi modelo de Deep Learning, y cómo puedo visualizarlo?"**

Este módulo se enfoca en **interpretabilidad a nivel de arquitectura interna**, complementando los módulos anteriores:
- **Módulo I:** Interpretabilidad de features (datos tabulares)
- **Módulo II:** Interpretabilidad de tokens/palabras (NLP)
- **Módulo III:** Interpretabilidad de neuronas (arquitectura interna)

---

### **2. Técnicas a Implementar**

#### **A) Feature Visualization (Visualización de Features)**
- **Qué es:** Generar imágenes sintéticas que maximalmente activan una neurona específica
- **Herramienta:** Lucid (TensorFlow) o pytorch-cnn-visualizations
- **Output:** "Esta neurona detecta ojos", "Esta neurona detecta texturas"

#### **B) Activation Maximization (Maximización de Activación)**
- **Qué es:** Optimización para encontrar el input que maximiza la activación
- **Herramienta:** Captum (PyTorch)
- **Output:** Patrones abstractos que la red "busca"

#### **C) Neuron Probing (Sondeo de Neuronas)**
- **Qué es:** Entrenar clasificadores lineales sobre activaciones para ver qué información codifican
- **Técnica:** Probing classifiers
- **Output:** "La capa 3 codifica formas, la capa 5 codifica objetos"

#### **D) Activation Patterns (Patrones de Activación)**
- **Qué es:** Visualizar qué partes de la imagen activan cada neurona
- **Herramienta:** Activation maps, heatmaps
- **Output:** Mapas de calor de activaciones por capa

---

### **3. Modelo y Dataset Propuestos**

#### **Modelo:**
**ResNet-18 o ResNet-50 pre-entrenado en ImageNet**

**Razones:**
- Arquitectura bien estudiada y documentada
- Pre-entrenado (no requiere training costoso)
- Suficientemente complejo para ser interesante
- Capas residuales facilitan análisis de flujo de información

#### **Dataset:**
**ImageNet subset o CIFAR-10**

**Opción 1 - ImageNet (100 clases):**
- Más realista y rico
- Clases interpretables (perros, gatos, vehículos)
- Puede ser costoso computacionalmente

**Opción 2 - CIFAR-10:**
- 10 clases simples (aviones, coches, pájaros, gatos, etc.)
- Más rápido para experimentar
- Suficiente para aprender las técnicas

**Recomendación:** Empezar con CIFAR-10 para prototipado rápido, luego escalar a ImageNet subset si es necesario

---

## **4. Estructura del Proyecto**

interpretability_III/
├── notebooks/
│   ├── 01_model_loading.ipynb           # Cargar ResNet pre-entrenado
│   ├── 02_activation_analysis.ipynb      # Analizar activaciones por capa
│   ├── 03_feature_visualization.ipynb    # Generar imágenes que activan neuronas
│   ├── 04_neuron_probing.ipynb          # Clasificadores de sondeo
│   └── 05_case_studies.ipynb            # Casos de estudio interesantes
│
├── src/
│   ├── models/
│   │   └── model_loader.py              # Cargar ResNet con hooks
│   │
│   ├── interpretability/
│   │   ├── activation_extractor.py      # Extraer activaciones de capas
│   │   ├── feature_visualizer.py        # Generar feature visualizations
│   │   ├── neuron_probe.py              # Probing classifiers
│   │   └── activation_analyzer.py       # Análisis estadístico de activaciones
│   │
│   ├── visualization/
│   │   ├── heatmap_viz.py               # Mapas de calor de activaciones
│   │   ├── filter_viz.py                # Visualización de filtros
│   │   └── layer_viz.py                 # Visualización por capas
│   │
│   └── utils/
│       ├── image_loader.py              # Cargar y procesar imágenes
│       └── hooks.py                     # PyTorch hooks para capas
│
├── app.py                               # Dashboard Streamlit interactivo
├── requirements.txt
├── LEARNINGS.md
└── README.md

---
## Plan de Trabajo 
Step 1: Setup y Fundamentos

Investigar teoría de feature visualization
Configurar entorno con PyTorch + Captum
Cargar ResNet pre-entrenado
Implementar hooks para extraer activaciones
Entregable: Notebook de carga del modelo + documentación teórica

Step 2: Extracción de Activaciones

Implementar ActivationExtractor class
Extraer activaciones de todas las capas para un batch de imágenes
Analizar estadísticas de activaciones (media, varianza, sparsity)
Visualizar distribuciones de activaciones
Entregable: Sistema de extracción funcionando + análisis exploratorio

Step 3: Feature Visualization

Implementar técnica de activation maximization
Generar imágenes sintéticas para neuronas individuales
Visualizar qué detectan los filtros de conv1, conv2, etc.
Identificar neuronas de "alto nivel" vs "bajo nivel"
Entregable: Galería de feature visualizations por capa

Step 4: Neuron Probing

Entrenar clasificadores lineales sobre activaciones
Probar qué información codifica cada capa (colores, formas, objetos)
Análisis de "emergencia" de conceptos en capas profundas
Entregable: Análisis de qué codifica cada capa

Step 5: Activation Patterns y Heatmaps

Crear mapas de calor de activaciones por imagen
Visualizar qué regiones activan cada neurona
Comparar patrones entre clases (gatos vs perros)
Entregable: Sistema de visualización de activation maps

Step 6: Dashboard y Casos de Estudio

Crear dashboard interactivo con Streamlit
Permitir al usuario subir imagen y explorar activaciones
Documentar 5-10 casos de estudio interesantes
Entregable: Dashboard funcionando + casos documentados

Step 7: Consolidación y Preparación Módulo IV

Documentar limitaciones encontradas
Crear LEARNINGS.md con insights clave
Diseñar transición a Módulo IV (GradCAM)
Entregable: Reporte final + roadmap Módulo IV


## Criterios de Éxito

 Sistema funcionando para extraer y visualizar activaciones de cualquier capa
 Galería de al menos 20 feature visualizations de neuronas interesantes
 Análisis de probing que muestre qué codifica cada capa
 Dashboard interactivo donde se pueda subir una imagen y explorar
 Identificación de al menos 3 insights sorprendentes sobre la red
 Documentación completa de limitaciones y aprendizajes
 Base sólida para transicionar a Módulo IV (GradCAM)