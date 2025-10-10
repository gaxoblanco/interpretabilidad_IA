# Roadmap de Interpretabilidad de IA
## Proyecto 1: Análisis de Modelos de Clasificación de Texto

---

## 📚 Estructura del Libro

### **Parte I: Fundamentos Teóricos**
- **Capítulo 1:** Introducción a la Interpretabilidad en IA
- **Capítulo 2:** Arquitecturas de Transformers y DistilBERT
- **Capítulo 3:** Teoría de SHAP y LIME

### **Parte II: Implementación Práctica**
- **Capítulo 4:** Configuración del Entorno y Estructura del Proyecto
- **Capítulo 5:** Carga y Evaluación del Modelo Base
- **Capítulo 6:** Implementación de SHAP para NLP
- **Capítulo 7:** Implementación de LIME para Texto
- **Capítulo 8:** Visualización y Comparación de Resultados

### **Parte III: Análisis Avanzado**
- **Capítulo 9:** Validación de Explicaciones
- **Capítulo 10:** Casos de Estudio y Patrones
- **Capítulo 11:** Preparación para el Proyecto 2

---

## 🎯 Objetivos de Aprendizaje

### Conceptuales
- ✅ Entender cómo los transformers procesan y representan texto
- ✅ Comprender la diferencia entre interpretabilidad global vs local
- ✅ Familiarizarse con métricas de importancia de features
- ✅ Aprender a validar explicaciones de modelos

### Técnicos
- ✅ Manejo de modelos pre-entrenados con HuggingFace
- ✅ Implementación de SHAP y LIME para NLP
- ✅ Visualización de resultados de interpretabilidad
- ✅ Preprocesamiento y análisis de datasets de texto

---

## 🛠️ Stack Tecnológico

### Librerías Core
```python
# Modelo y procesamiento
transformers==4.21.0      # HuggingFace transformers
torch==1.12.0             # PyTorch backend
datasets==2.4.0           # Manejo de datasets

# Interpretabilidad
shap==0.41.0              # SHAP explanations
lime==0.2.0.1             # LIME explanations

# Análisis y visualización
pandas==1.4.3            # Manipulación de datos
numpy==1.21.6             # Operaciones numéricas
matplotlib==3.5.2         # Visualización básica
seaborn==0.11.2           # Visualización estadística
plotly==5.10.0            # Visualizaciones interactivas

# Utilidades
scikit-learn==1.1.2      # Métricas y preprocesamiento
tqdm==4.64.0              # Barras de progreso
```

---

## 📁 Estructura del Proyecto

```
proyecto_interpretabilidad/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py      # Carga y configuración del modelo
│   │   └── predictor.py         # Wrapper para predicciones
│   ├── interpretability/
│   │   ├── __init__.py
│   │   ├── shap_analyzer.py     # Implementación SHAP
│   │   ├── lime_analyzer.py     # Implementación LIME
│   │   └── base_explainer.py    # Clase base común
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── text_viz.py          # Visualización de texto
│   │   └── metrics_viz.py       # Gráficos de métricas
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py       # Carga de datasets
│       └── preprocessing.py     # Limpieza de texto
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_interpretability_analysis.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── requirements.txt
└── README.md
```

---

## 💻 Pseudocódigo de Componentes Principales

### 1. Cargador de Modelo
```python
class ModelLoader:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Inicializa el modelo pre-entrenado
        - Carga tokenizer y modelo
        - Configura device (CPU/GPU)
        """
    
    def load_model(self):
        """
        Retorna: (model, tokenizer)
        """
    
    def predict(self, texts):
        """
        Input: lista de textos
        Output: predicciones y probabilidades
        """
```

### 2. Analizador SHAP
```python
class SHAPAnalyzer:
    def __init__(self, model, tokenizer):
        """
        Configura explainer para transformers
        - Utiliza shap.Explainer
        - Configura masking strategy
        """
    
    def explain_instance(self, text):
        """
        Input: texto individual
        Output: valores SHAP por token
        """
    
    def explain_batch(self, texts):
        """
        Input: batch de textos
        Output: valores SHAP agregados
        """
    
    def get_global_importance(self, dataset):
        """
        Input: dataset completo
        Output: importancia global de features
        """
```

### 3. Analizador LIME
```python
class LIMEAnalyzer:
    def __init__(self, model, tokenizer):
        """
        Configura LimeTextExplainer
        - Define función de predicción
        - Configura parámetros de sampling
        """
    
    def explain_instance(self, text, num_features=10):
        """
        Input: texto y número de features
        Output: explicación LIME
        """
    
    def compare_predictions(self, original_text, perturbed_text):
        """
        Input: texto original y perturbado
        Output: comparación de predicciones
        """
```

### 4. Visualizador
```python
class TextVisualizer:
    def plot_token_importance(self, text, importance_scores):
        """
        Crea heatmap de importancia por token
        """
    
    def plot_comparison(self, shap_scores, lime_scores):
        """
        Compara resultados de SHAP vs LIME
        """
    
    def create_interactive_explanation(self, text, explanations):
        """
        Genera visualización interactiva con Plotly
        """
```

---

## 📖 Roadmap Detallado por Capítulos

### **Capítulo 1: Introducción a la Interpretabilidad en IA** (Semana 1)
**Duración estimada:** 3-4 días

**Objetivos:**
- Entender qué es la interpretabilidad y por qué es importante
- Diferenciar entre explicabilidad, interpretabilidad y transparencia
- Conocer el panorama actual de herramientas de interpretabilidad

**Actividades:**
- [ ] Lectura de papers fundamentales (Ribeiro et al. 2016 - LIME, Lundberg & Lee 2017 - SHAP)
- [ ] Investigar casos de uso reales de interpretabilidad en la industria
- [ ] Crear glosario de términos clave
- [ ] Definir métricas de evaluación para el proyecto

**Entregables:**
- Documento de definiciones y conceptos clave
- Lista de casos de uso identificados
- Criterios de éxito para el proyecto

---

### **Capítulo 2: Arquitecturas de Transformers y DistilBERT** (Semana 1-2)
**Duración estimada:** 4-5 días

**Objetivos:**
- Comprender la arquitectura de transformers
- Entender cómo DistilBERT difiere de BERT
- Familiarizarse con conceptos de attention y embeddings

**Actividades:**
- [ ] Estudiar el paper "Attention is All You Need"
- [ ] Analizar la arquitectura de DistilBERT
- [ ] Explorar modelos pre-entrenados en HuggingFace
- [ ] Crear diagramas de la arquitectura

**Entregables:**
- Documento técnico sobre transformers
- Diagrama de arquitectura de DistilBERT
- Comparación BERT vs DistilBERT

---

### **Capítulo 3: Teoría de SHAP y LIME** (Semana 2)
**Duración estimada:** 3-4 días

**Objetivos:**
- Entender la base matemática de SHAP (valores de Shapley)
- Comprender el algoritmo de LIME
- Identificar ventajas y limitaciones de cada método

**Actividades:**
- [ ] Estudiar teoría de juegos cooperativos (valores de Shapley)
- [ ] Analizar el algoritmo de perturbación de LIME
- [ ] Comparar propiedades teóricas de ambos métodos
- [ ] Implementar ejemplos toy en datos tabulares

**Entregables:**
- Documento de fundamentos matemáticos
- Implementación de ejemplo simple
- Tabla comparativa SHAP vs LIME

---

### **Capítulo 4: Configuración del Entorno y Estructura del Proyecto** (Semana 3)
**Duración estimada:** 2-3 días

**Objetivos:**
- Configurar entorno de desarrollo
- Implementar estructura modular del proyecto
- Establecer buenas prácticas de código

**Actividades:**
- [ ] Crear ambiente virtual y instalar dependencias
- [ ] Implementar estructura de carpetas
- [ ] Configurar logging y manejo de errores
- [ ] Crear scripts de setup y configuración

**Entregables:**
- Proyecto base funcionando
- Documentación de setup
- Scripts de instalación automatizada

---

### **Capítulo 5: Carga y Evaluación del Modelo Base** (Semana 3)
**Duración estimada:** 3-4 días

**Objetivos:**
- Cargar modelo DistilBERT pre-entrenado
- Evaluar performance en dataset de prueba
- Implementar pipeline de predicción

**Actividades:**
- [ ] Implementar `ModelLoader` class
- [ ] Cargar dataset IMDb reviews
- [ ] Evaluar accuracy, precision, recall, F1
- [ ] Crear pipeline de predicción end-to-end
- [ ] Análisis de errores del modelo

**Entregables:**
- Módulo `model_loader.py` completo
- Reporte de evaluación del modelo
- Pipeline de predicción funcional

---

### **Capítulo 6: Implementación de SHAP para NLP** (Semana 4)
**Duración estimada:** 4-5 días

**Objetivos:**
- Implementar SHAP para modelos de texto
- Calcular importancia de tokens
- Generar explicaciones globales y locales

**Actividades:**
- [ ] Implementar `SHAPAnalyzer` class
- [ ] Configurar `shap.Explainer` para transformers
- [ ] Calcular valores SHAP para instancias individuales
- [ ] Agregar valores SHAP para análisis global
- [ ] Optimizar performance para batches grandes

**Entregables:**
- Módulo `shap_analyzer.py` completo
- Notebook con ejemplos de uso
- Análisis de performance y timing

---

### **Capítulo 7: Implementación de LIME para Texto** (Semana 4-5)
**Duración estimada:** 4-5 días

**Objetivos:**
- Implementar LIME para clasificación de texto
- Generar explicaciones locales interpretables
- Comparar resultados con SHAP

**Actividades:**
- [ ] Implementar `LIMEAnalyzer` class
- [ ] Configurar `LimeTextExplainer`
- [ ] Experimentar con diferentes estrategias de perturbación
- [ ] Analizar estabilidad de explicaciones
- [ ] Implementar métricas de comparación

**Entregables:**
- Módulo `lime_analyzer.py` completo
- Análisis comparativo SHAP vs LIME
- Métricas de estabilidad y fidelidad

---

### **Capítulo 8: Visualización y Comparación de Resultados** (Semana 5-6)
**Duración estimada:** 4-5 días

**Objetivos:**
- Crear visualizaciones informativas
- Desarrollar dashboards interactivos
- Implementar comparaciones side-by-side

**Actividades:**
- [ ] Implementar `TextVisualizer` class
- [ ] Crear heatmaps de importancia de tokens
- [ ] Desarrollar visualizaciones interactivas con Plotly
- [ ] Implementar comparaciones SHAP vs LIME
- [ ] Crear dashboard de análisis

**Entregables:**
- Módulos de visualización completos
- Dashboard interactivo
- Galería de visualizaciones ejemplo

---

### **Capítulo 9: Validación de Explicaciones** (Semana 6)
**Duración estimada:** 3-4 días

**Objetivos:**
- Implementar métricas de validación
- Evaluar fidelidad y estabilidad
- Realizar tests de sanidad

**Actividades:**
- [ ] Implementar métricas de fidelidad
- [ ] Calcular estabilidad de explicaciones
- [ ] Crear tests de perturbación
- [ ] Validar con casos conocidos
- [ ] Documentar limitaciones encontradas

**Entregables:**
- Suite de métricas de validación
- Reporte de calidad de explicaciones
- Documentación de limitaciones

---

### **Capítulo 10: Casos de Estudio y Patrones** (Semana 7)
**Duración estimada:** 4-5 días

**Objetivos:**
- Analizar casos específicos interesantes
- Identificar patrones en las explicaciones
- Documentar hallazgos clave

**Actividades:**
- [ ] Seleccionar casos de estudio representativos
- [ ] Analizar patrones de importancia por categoría
- [ ] Identificar tokens más influyentes globalmente
- [ ] Estudiar casos de desacuerdo SHAP vs LIME
- [ ] Documentar insights obtenidos

**Entregables:**
- Casos de estudio documentados
- Análisis de patrones identificados
- Lista de insights y hallazgos

---

### **Capítulo 11: Preparación para el Proyecto 2** (Semana 7-8)
**Duración estimada:** 3-4 días

**Objetivos:**
- Consolidar aprendizajes del Proyecto 1
- Diseñar arquitectura para el Proyecto 2
- Identificar skills faltantes

**Actividades:**
- [ ] Crear reporte final del Proyecto 1
- [ ] Identificar limitaciones y áreas de mejora
- [ ] Investigar herramientas para activación de neuronas
- [ ] Diseñar estructura para el Proyecto 2
- [ ] Crear roadmap del siguiente proyecto

**Entregables:**
- Reporte final completo
- Presentación de resultados
- Roadmap del Proyecto 2

---

## 📊 Información Relevante

### Conceptos Clave a Dominar

**SHAP (SHapley Additive exPlanations)**
- Basado en teoría de juegos cooperativos
- Proporciona valores de contribución aditivos
- Garantiza propiedades como eficiencia y simetría
- Mejor para análisis global de modelos

**LIME (Local Interpretable Model-agnostic Explanations)**
- Explica predicciones individuales
- Usa modelos interpretables localmente
- Funciona perturbando inputs y observando cambios
- Mejor para entender decisiones específicas

**DistilBERT Architecture**
- Versión destilada de BERT (50% menos parámetros)
- 6 layers, 768 hidden units, 12 attention heads
- Mantiene 97% del performance de BERT
- Ideal para interpretabilidad por su menor complejidad

### Métricas de Validación
- **Fidelidad:** ¿Las explicaciones reflejan el comportamiento real del modelo?
- **Estabilidad:** ¿Explicaciones similares para inputs similares?
- **Comprensibilidad:** ¿Los humanos pueden entender las explicaciones?

### Dataset Sugerido
**IMDb Movie Reviews**
- 50k reviews balanceados (25k pos, 25k neg)
- Textos de longitud variable (promedio ~200 palabras)
- Disponible en HuggingFace datasets
- Permite validación intuitiva de explicaciones

---

## ⏱️ Timeline General

| Semana | Capítulos | Enfoque Principal |
|--------|-----------|-------------------|
| 1 | 1-2 | Fundamentos teóricos |
| 2 | 2-3 | Arquitecturas y matemáticas |
| 3 | 4-5 | Setup y modelo base |
| 4 | 6-7 | Implementación SHAP |
| 5 | 7-8 | Implementación LIME y visualización |
| 6 | 8-9 | Visualización avanzada y validación |
| 7 | 10-11 | Casos de estudio y preparación |
| 8 | 11 | Consolidación y siguiente proyecto |

**Duración total estimada:** 7-8 semanas (1.5-2 meses)

---

## 🎯 Criterios de Éxito

- [ ] Implementación funcional de SHAP y LIME para texto
- [ ] Visualizaciones claras e informativas
- [ ] Análisis comparativo robusto entre métodos
- [ ] Documentación completa y reproducible
- [ ] Identificación de al menos 5 insights clave sobre el modelo
- [ ] Base sólida para el Proyecto 2 (activación de neuronas)