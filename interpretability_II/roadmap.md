# 📘 Roadmap Módulo II: Interpretabilidad en NLP
## DistilBERT + SHAP + LIME (7 semanas)

---

## 🎯 Objetivo
Implementar SHAP y LIME para explicar predicciones de un modelo DistilBERT en análisis de sentimientos, comparando ambas técnicas.

---

## 📅 Plan por Semanas

### **Semana 1: Fundamentos Teóricos**
**Objetivo:** Entender qué es interpretabilidad y cómo funcionan Transformers

**Actividades:**
- [ ] Leer sobre interpretabilidad en ML (1 paper: Ribeiro LIME 2016)
- [ ] Estudiar arquitectura de Transformers (attention mechanism)
- [ ] Entender qué es DistilBERT y cómo difiere de BERT
- [ ] Crear documento con conceptos clave

**Entregable:**
```
docs/01_fundamentos.md
  - Definición de interpretabilidad
  - Diagrama simple de Transformer
  - Cuándo usar SHAP vs LIME
```

**Tiempo:** 6-8 horas

---

### **Semana 2: Teoría SHAP y LIME**
**Objetivo:** Entender matemáticamente cómo funcionan ambos métodos

**Actividades:**
- [ ] Estudiar valores de Shapley (teoría de juegos)
- [ ] Entender algoritmo de perturbación de LIME
- [ ] Implementar ejemplo TOY en datos tabulares simples
- [ ] Comparar resultados en el ejemplo toy

**Entregable:**
```
notebooks/01_shap_lime_toy_example.ipynb
  - Ejemplo con RandomForest en datos simples
  - Tabla comparativa SHAP vs LIME
```

**Código ejemplo (toy):**
```python
# Datos tabulares simples (ejemplo: aprobar/rechazar préstamo)
from sklearn.ensemble import RandomForestClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer

# Entrenar modelo simple
model = RandomForestClassifier().fit(X_train, y_train)

# SHAP
explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(X_test[0])

# LIME
explainer_lime = LimeTabularExplainer(X_train)
lime_exp = explainer_lime.explain_instance(X_test[0], model.predict_proba)

# Comparar visualmente
```

**Tiempo:** 6-8 horas

---

### **Semana 3: Setup del Proyecto**
**Objetivo:** Estructura modular y carga del modelo DistilBERT

**Actividades:**
- [ ] Crear estructura de carpetas modular
- [ ] Implementar `ModelLoader` (carga DistilBERT)
- [ ] Implementar `DataLoader` (carga IMDb dataset)
- [ ] Crear pipeline básico de predicción
- [ ] Evaluar modelo base (accuracy, F1)

**Entregable:**
```
src/
├── models/model_loader.py
├── utils/data_loader.py
└── config/config.yaml

notebooks/02_model_evaluation.ipynb
  - Métricas del modelo base
  - Ejemplos de predicciones
```

**Pseudocódigo:**
```python
# src/models/model_loader.py
class ModelLoader:
    """Carga y configura DistilBERT pre-entrenado"""
    
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        # Cargar tokenizer y modelo
        # Configurar device (CPU/GPU)
        pass
    
    def predict(self, texts: list) -> dict:
        # Input: lista de textos
        # Output: {"predictions": [...], "probabilities": [...]}
        pass
```

**Tiempo:** 8-10 horas

---

### **Semana 4: Implementación SHAP**
**Objetivo:** Explicar predicciones con SHAP values

**Actividades:**
- [ ] Implementar `SHAPAnalyzer` class
- [ ] Calcular SHAP values para instancias individuales
- [ ] Calcular importancia global de palabras
- [ ] Crear visualizaciones básicas (waterfall, bar plot)

**Entregable:**
```
src/interpretability/shap_analyzer.py

notebooks/03_shap_analysis.ipynb
  - Ejemplos de explicaciones SHAP
  - Palabras más importantes globalmente
  - Top 10 palabras positivas/negativas
```

**Pseudocódigo:**
```python
# src/interpretability/shap_analyzer.py
class SHAPAnalyzer:
    """Implementa SHAP para modelos de texto"""
    
    def __init__(self, model, tokenizer):
        # Configurar shap.Explainer para transformers
        pass
    
    def explain_instance(self, text: str) -> dict:
        # Input: texto individual
        # Output: {"tokens": [...], "shap_values": [...]}
        pass
    
    def get_global_importance(self, texts: list) -> dict:
        # Input: lista de textos
        # Output: palabras más importantes globalmente
        pass
```

**Tiempo:** 10-12 horas

---

### **Semana 5: Implementación LIME**
**Objetivo:** Explicar predicciones con LIME

**Actividades:**
- [ ] Implementar `LIMEAnalyzer` class
- [ ] Configurar estrategia de perturbación para texto
- [ ] Generar explicaciones locales
- [ ] Comparar con SHAP en mismos ejemplos

**Entregable:**
```
src/interpretability/lime_analyzer.py

notebooks/04_lime_analysis.ipynb
  - Ejemplos de explicaciones LIME
  - Comparación LIME vs SHAP (5 casos)
```

**Pseudocódigo:**
```python
# src/interpretability/lime_analyzer.py
class LIMEAnalyzer:
    """Implementa LIME para modelos de texto"""
    
    def __init__(self, model, tokenizer):
        # Configurar LimeTextExplainer
        pass
    
    def explain_instance(self, text: str, num_features=10) -> dict:
        # Input: texto y número de features
        # Output: {"words": [...], "importance": [...]}
        pass
```

**Tiempo:** 10-12 horas

---

### **Semana 6: Visualización y Comparación**
**Objetivo:** Crear visualizaciones claras y dashboard interactivo

**Actividades:**
- [ ] Implementar visualizador de importancia de tokens (heatmap)
- [ ] Crear comparación side-by-side SHAP vs LIME
- [ ] Desarrollar dashboard Streamlit simple
- [ ] Permitir input de usuario en tiempo real

**Entregable:**
```
src/visualization/text_viz.py
app.py  # Dashboard Streamlit

notebooks/05_visualization.ipynb
  - Galería de visualizaciones
```

**Dashboard features:**
```python
# app.py
import streamlit as st

st.title("🔍 Explicador de Sentimientos")

# Input del usuario
text = st.text_area("Escribe una review:")

if st.button("Explicar"):
    # Predecir
    prediction = model.predict(text)
    
    # Explicar con SHAP
    shap_exp = shap_analyzer.explain(text)
    
    # Explicar con LIME
    lime_exp = lime_analyzer.explain(text)
    
    # Visualizar lado a lado
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SHAP")
        plot_shap(shap_exp)
    with col2:
        st.subheader("LIME")
        plot_lime(lime_exp)
```

**Tiempo:** 8-10 horas

---

### **Semana 7: Validación y Casos de Estudio**
**Objetivo:** Evaluar calidad de explicaciones y documentar hallazgos

**Actividades:**
- [ ] Implementar métricas de validación (fidelidad, estabilidad)
- [ ] Analizar 10 casos interesantes (acuerdo/desacuerdo SHAP-LIME)
- [ ] Identificar patrones (palabras más influyentes)
- [ ] Documentar limitaciones encontradas
- [ ] Crear documento LEARNINGS.md

**Entregable:**
```
notebooks/06_validation.ipynb
  - Métricas de fidelidad
  - Análisis de estabilidad
  - 10 casos de estudio documentados

LEARNINGS.md
  - Insights clave
  - Cuándo usar SHAP vs LIME
  - Limitaciones encontradas
  - Preparación para Módulo III
```

**Métricas a calcular:**
```python
# Fidelidad: ¿Las explicaciones reflejan el modelo?
def compute_fidelity(explanations, model_predictions):
    # Eliminar top-K features y ver cambio en predicción
    pass

# Estabilidad: ¿Explicaciones similares para textos similares?
def compute_stability(text1, text2, exp1, exp2):
    # Medir correlación entre explicaciones
    pass
```

**Tiempo:** 8-10 horas

---

## 📊 Resumen de Entregables

| Semana | Entregable Principal | Tipo |
|--------|---------------------|------|
| 1 | `docs/01_fundamentos.md` | Documentación |
| 2 | `notebooks/01_toy_example.ipynb` | Código + Análisis |
| 3 | `src/models/` + `notebooks/02_eval.ipynb` | Código |
| 4 | `src/interpretability/shap_analyzer.py` | Código |
| 5 | `src/interpretability/lime_analyzer.py` | Código |
| 6 | `app.py` (Dashboard) | Aplicación |
| 7 | `LEARNINGS.md` + Validación | Documentación |

---

## ✅ Criterios de Éxito

- [ ] Implementación funcional de SHAP y LIME para DistilBERT
- [ ] Dashboard interactivo que acepta texto del usuario
- [ ] Comparación clara entre ambos métodos (al menos 5 casos)
- [ ] Identificación de al menos 3 insights clave sobre el modelo
- [ ] Documentación de limitaciones y siguiente paso (Módulo III)

---

## 🛠️ Stack Tecnológico Mínimo

```txt
# requirements.txt
transformers==4.30.0
torch==2.0.0
datasets==2.12.0
shap==0.42.0
lime==0.2.0.1
pandas==2.0.0
matplotlib==3.7.0
streamlit==1.22.0
scikit-learn==1.2.0
```

---

## ⏱️ Tiempo Total Estimado

**56-70 horas** distribuidas en **7 semanas** (8-10 horas/semana)

---

## 🎯 Pregunta Final a Responder

**"Para este modelo de sentimientos, ¿qué método (SHAP o LIME) me da explicaciones más útiles y en qué situaciones?"**