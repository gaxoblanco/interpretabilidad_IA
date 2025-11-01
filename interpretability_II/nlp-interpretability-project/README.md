# 🔍 Módulo II - Interpretabilidad en NLP con Transformers

Proyecto educativo enfocado en dominar técnicas de interpretabilidad (SHAP y LIME) para modelos de lenguaje natural.

---

## 📂 Estructura del Proyecto

```
interpretability_II/
│
├── nlp-interpretability-project/     # 📚 NOTEBOOKS DE APRENDIZAJE
│   ├── notebooks/
│   │   ├── 02_model_evaluation.ipynb      # Evaluación del modelo base
│   │   ├── 03_shap_analysis.ipynb         # Implementación y análisis SHAP
│   │   └── 04_lime_analysis.ipynb         # Implementación y análisis LIME
│   │
│   ├── data/                               # Datasets y cache de modelos
│   ├── explainability_analysis/            # Resultados de análisis
│   ├── models_cache/                       # Modelos descargados
│   ├── results/                            # Outputs de experimentos
│   └── venv/                               # Entorno virtual local
│
└── nlp-interpretability-dashboard/   # 🚀 DASHBOARD DEPLOYADO
    ├── app.py                              # Aplicación Streamlit principal
    ├── requirements.txt                    # Dependencias para deployment
    ├── README.md                           # Documentación del dashboard
    │
    └── src/
        ├── config/
        │   └── config.yaml                 # Configuración centralizada
        │
        ├── models/
        │   └── model_loader.py             # Carga de modelos Transformer
        │
        └── utils/
            ├── dashboard.py                # Funciones de visualización
            ├── data_loader.py              # Carga de datos
            ├── fidelity_explanation.py     # Validación de explicaciones
            └── [otros módulos]
```

---

## 🎯 Objetivo del Proyecto

Responder la pregunta:
> **"¿Qué método de interpretabilidad (SHAP o LIME) proporciona explicaciones más útiles para modelos Transformer en análisis de sentimientos, y en qué situaciones usar cada uno?"**

---

## 📊 Componentes del Proyecto

### 1️⃣ **Notebooks de Aprendizaje** (`nlp-interpretability-project/`)

Notebooks Jupyter donde se implementaron y probaron las técnicas:

#### **02_model_evaluation.ipynb**
- Carga y evaluación de DistilBERT pre-entrenado
- Métricas: Accuracy, Precision, Recall, F1-Score
- Análisis de errores y casos extremos
- Dataset: SST-2 (Stanford Sentiment Treebank)

#### **03_shap_analysis.ipynb**
- Implementación de SHAP para transformers
- Cálculo de valores de Shapley para tokens
- Visualizaciones: waterfall plots, force plots
- Análisis de importancia global de palabras

#### **04_lime_analysis.ipynb**
- Implementación de LIME para texto
- Configuración de perturbaciones locales
- Comparación de estabilidad entre ejecuciones
- Trade-offs: velocidad vs precisión

---

### 2️⃣ **Dashboard Interactivo** (`nlp-interpretability-dashboard/`)

Aplicación Streamlit deployada en Hugging Face Spaces.

**🔗 Demo en vivo:** [https://huggingface.co/spaces/gaxoblanco/nlp-interpretability-dashboard](https://huggingface.co/spaces/gaxoblanco/nlp-interpretability-dashboard)

#### **Funcionalidades:**

**Modelos disponibles:**
- DistilBERT (sentimientos binarios)
- RoBERTa (sentimientos binarios)
- DistilRoBERTa (6 emociones)
- BERT Emotion (6 emociones)

**Métodos de explicación:**
- Solo SHAP
- Solo LIME
- Ambos (comparación lado a lado)

**Visualizaciones:**
- Predicción con confianza
- Importancia de palabras por método
- Comparación SHAP vs LIME
- Métricas de validación (fidelidad, correlación)

**Casos de uso:**
- Input personalizado del usuario
- 5 ejemplos predefinidos (positivo, negativo, mixto, sarcástico, neutral)

---

## 🛠️ Stack Tecnológico

### Core ML
```
transformers==4.35.2    # HuggingFace Transformers
torch==2.0.0            # PyTorch backend
datasets==2.18.0        # Datasets de HuggingFace
```

### Interpretabilidad
```
shap==0.42.0            # SHAP explainer
lime==0.2.0.1           # LIME explainer
```

### Visualización & Dashboard
```
streamlit==1.22.0       # Framework del dashboard
matplotlib==3.7.1       # Visualizaciones básicas
seaborn==0.12.2         # Visualizaciones estadísticas
plotly==5.14.1          # Gráficos interactivos
```

### Data Science
```
pandas==2.0.3           # Manipulación de datos
numpy==1.24.3           # Operaciones numéricas
scikit-learn==1.2.2     # Métricas y utilidades
```

---

## 🚀 Inicio Rápido

### **Opción 1: Probar el Dashboard Online** (Recomendado)

Visita directamente: [https://huggingface.co/spaces/gaxoblanco/nlp-interpretability-dashboard](https://huggingface.co/spaces/gaxoblanco/nlp-interpretability-dashboard)

---

### **Opción 2: Ejecutar Localmente**

#### **A) Dashboard Interactivo**

```bash
# 1. Clonar el repositorio
git clone [tu-repo]
cd interpretability_II/nlp-interpretability-dashboard

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar dashboard
streamlit run app.py
```

El dashboard abrirá en: `http://localhost:8501`

---

#### **B) Notebooks de Aprendizaje**

```bash
# 1. Ir a la carpeta de notebooks
cd interpretability_II/nlp-interpretability-project

# 2. Activar entorno virtual
source venv/bin/activate

# 3. Lanzar Jupyter
jupyter notebook

# 4. Abrir cualquier notebook:
#    - 02_model_evaluation.ipynb
#    - 03_shap_analysis.ipynb
#    - 04_lime_analysis.ipynb
```

---

## 📚 Roadmap de Desarrollo (Completado)

| Semana | Objetivo | Entregable | Estado |
|--------|----------|------------|--------|
| **1-2** | Fundamentos teóricos | Documentación conceptual | ✅ |
| **3** | Setup y evaluación | `02_model_evaluation.ipynb` | ✅ |
| **4** | Implementación SHAP | `03_shap_analysis.ipynb` | ✅ |
| **5** | Implementación LIME | `04_lime_analysis.ipynb` | ✅ |
| **6** | Dashboard | `app.py` deployado | ✅ |
| **7** | Validación | Métricas de fidelidad | ✅ |


---

## 🎓 Aprendizajes Clave

### **SHAP vs LIME: ¿Cuándo usar cada uno?**

#### **Usa SHAP cuando:**
- ✅ Necesitas garantías matemáticas formales
- ✅ Quieres explicaciones globales (todo el dataset)
- ✅ Importa más la precisión que la velocidad
- ✅ Necesitas consistencia perfecta entre ejecuciones

#### **Usa LIME cuando:**
- ✅ Necesitas explicaciones rápidas (producción)
- ✅ Solo te interesan explicaciones locales (instancia específica)
- ✅ Quieres explorar múltiples perturbaciones
- ✅ El modelo es completamente black-box

#### **Usa ambos cuando:**
- ✅ Quieres validar cruzada de explicaciones
- ✅ Estás en fase de research/análisis
- ✅ Necesitas detectar artefactos del método

---

## 👨‍💻 Autor

**Proyecto Educativo - Módulo II**
- Plan de estudio en interpretabilidad de ML
- Desarrollado por Gaston Blanco
---

## 📄 Licencia

MIT License - Proyecto educativo de código abierto

---
