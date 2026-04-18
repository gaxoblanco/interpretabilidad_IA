python -m notebook

# Interpretabilidad en Machine Learning

Programa de estudio progresivo enfocado en técnicas de interpretabilidad para modelos de ML. Cada módulo implementa métodos específicos sobre diferentes tipos de datos, documentando decisiones técnicas y resultados experimentales.

---

## Estructura del Repositorio

```
interpretability_ml/
├── interpretability_I/     # XGBoost + SHAP (Datos Tabulares)
├── interpretability_II/    # DistilBERT + SHAP + LIME (NLP)
├── interpretability_III/   # Análisis de Activaciones Neuronales
├── interpretability_IV/    # CNN + GradCAM (Computer Vision)
├── roadmap.md             # Planificación detallada por semanas
└── roadmap_study.md       # Referencias y notas de estudio
```

Cada carpeta de módulo contiene:
- `notebooks/`: Análisis exploratorio, entrenamiento, evaluación
- `LEARNINGS.md`: Hallazgos, decisiones técnicas, reflexiones
- `README.md`: Documentación específica del experimento

---

## Módulo I: Clasificación Tabular con XGBoost

**Dataset:** German Credit Data (1000 instancias, 20 features)  
**Problema:** Clasificación binaria de riesgo crediticio

### Objetivo
Implementar y comparar métodos model-agnostic de interpretabilidad en un clasificador de gradient boosting.

### Modelos
- **Baseline:** Logistic Regression (interpretable por diseño)
- **Target:** XGBoost Classifier

### Técnicas de Interpretabilidad
1. **SHAP (SHapley Additive exPlanations)**
   - TreeExplainer para modelos tree-based
   - Valores de Shapley para atribución de features
   - Visualizaciones: waterfall, force plots, dependence plots

2. **Counterfactual Explanations**
   - Generación de instancias contrafactuales via optimización
   - Distancia euclidiana para encontrar la instancia más cercana en clase opuesta

### Métricas de Evaluación
- **Modelo:** Accuracy, Precision, Recall, ROC-AUC
- **Interpretabilidad:** Consistencia SHAP vs Feature Importance, coverage de counterfactuals

### Artefactos
- 4 notebooks: EDA, modelado, SHAP analysis, counterfactuals
- Dashboard Streamlit para predicción + explicación
- Modelos serializados (pickle)

**Estado:** Completado

---

## Módulo II: Clasificación de Texto con Transformers

**Dataset:** IMDb Reviews (50k reviews, clasificación binaria sentimiento)  
**Problema:** Análisis de sentimientos

### Objetivo
Adaptar técnicas de interpretabilidad model-agnostic (SHAP, LIME) a modelos de lenguaje pre-entrenados y comparar con métodos específicos de Transformers (attention weights).

### Modelo
- **DistilBERT** fine-tuned para clasificación binaria
- 6 capas, 12 attention heads por capa
- Tokenización WordPiece

### Técnicas de Interpretabilidad

#### 1. SHAP para Texto
- **Kernel SHAP** con perturbed sampling de tokens
- Atribución a nivel de token individual
- **Desafío:** Manejo de contexto secuencial (tokens no independientes)

#### 2. LIME (Local Interpretable Model-agnostic Explanations)
- Perturbación de texto mediante eliminación de palabras
- Modelo sustituto lineal local
- **Comparación:** SHAP vs LIME en términos de estabilidad/fidelidad

#### 3. Attention Visualization
- Extracción de attention weights de capas específicas
- Visualización de patrones de atención token-to-token
- **Limitación:** Attention ≠ explanation (debate abierto en literatura)

### Plan de Implementación (3 Fases - 7-8 semanas)

**Fase I: Fundamentos (Semanas 1-2)**
- Setup de modelo pre-entrenado + fine-tuning pipeline
- Implementación de SHAP para texto
- Validación con casos sintéticos

**Fase II: Implementación Modular (Semanas 3-5)**
- `ModelLoader`: carga de DistilBERT + tokenizer
- `SHAPAnalyzer`: wrapper para Kernel SHAP con sampling estratégico
- `LIMEAnalyzer`: perturbador de texto + modelo lineal local
- `AttentionVisualizer`: extractor de attention matrices

**Fase III: Análisis Comparativo (Semanas 6-8)**
- Métricas de fidelidad: ¿qué método predice mejor la salida del modelo?
- Análisis de casos donde SHAP/LIME/Attention divergen
- Documentación de trade-offs (costo computacional vs interpretabilidad)

**Estado:** Planificado

---

## Módulo III: Análisis de Activaciones Neuronales

**Objetivo:** Investigar qué representan neuronas individuales en capas ocultas de redes profundas.

### Enfoque (a definir según hallazgos de Módulo II)
- **Opción A:** Probing tasks sobre representaciones de DistilBERT
- **Opción B:** Feature visualization en CNN (si se pivotea a visión)

### Técnicas Candidatas
1. **Neuron Probing**
   - Entrenar clasificadores lineales sobre activaciones de capas específicas
   - Evaluar qué información lingüística/visual captura cada capa

2. **Activation Maximization**
   - Optimización de inputs para maximizar activación de neurona específica
   - Visualización de features que activan cada neurona

3. **Causal Intervention**
   - Ablación de neuronas/atención heads
   - Medición de impacto en performance downstream

**Estado:** Pendiente de definición

---

## Módulo IV: Interpretabilidad en Computer Vision

**Dataset:** Por definir (ImageNet subset, CIFAR-10, o dominio específico)  
**Problema:** Clasificación de imágenes

### Objetivo
Implementar métodos de visualización de saliency para entender qué regiones de la imagen influyen en la predicción del modelo.

### Modelo
- CNN pre-entrenada (ResNet, EfficientNet, o ViT)

### Técnicas
1. **GradCAM (Gradient-weighted Class Activation Mapping)**
   - Gradientes de la clase predicha respecto a feature maps
   - Heatmap de importancia espacial

2. **Integrated Gradients**
   - Integración de gradientes a lo largo de un path (baseline → input)
   - Atribución pixel-level

3. **Saliency Maps**
   - Gradiente de la predicción respecto al input
   - Variantes: SmoothGrad, Guided Backprop

### Evaluación
- **Sanity checks:** Comparación con random model/input
- **Deletion/Insertion curves:** Métrica de fidelidad de saliency maps

**Estado:** Planificado

---

## Progreso General

| Módulo | Tipo de Datos | Modelo | Técnicas | Estado |
|--------|--------------|--------|----------|--------|
| **I** | Tabular | XGBoost | SHAP, Counterfactuals | ✅ Completado |
| **II** | Texto | DistilBERT | SHAP, LIME, Attention | 📋 Planificado (8 sem) |
| **III** | Variable | TBD | Neuron Probing, Activation Max | 📋 Por definir |
| **IV** | Imágenes | CNN/ViT | GradCAM, Integrated Gradients | 📋 Planificado |

---

## Stack Técnico por Módulo

| Módulo | ML Framework | Interpretabilidad | Visualización |
|--------|-------------|-------------------|---------------|
| I | XGBoost, Scikit-learn | SHAP | Matplotlib, Streamlit |
| II | Transformers (HF), PyTorch | SHAP, LIME, BertViz | Plotly, Streamlit |
| III | PyTorch | Captum (TBD) | Matplotlib |
| IV | PyTorch/TensorFlow | Captum, tf-explain | Matplotlib, PIL |

---

## Setup de Entorno

```bash
# Clonar repositorio
git clone <repo-url>
cd interpretability_ml

# Navegar a módulo específico
cd interpretability_I

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebooks o dashboard
jupyter notebook
# o
streamlit run app.py
```

### Limpieza de Notebooks (pre-commit)
```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

---

## Recursos de Referencia

- **Roadmap detallado:** `roadmap.md` - Desglose por capítulos y semanas
- **Notas de estudio:** `roadmap_study.md` - Papers, recursos teóricos
- **Aprendizajes por módulo:** `<modulo>/LEARNINGS.md` - Decisiones técnicas y hallazgos

---

## Principios del Proyecto

1. **Reproducibilidad:** Seeds fijados, versiones de librerías especificadas
2. **Modularidad:** Código reutilizable entre experimentos
3. **Documentación:** Decisiones técnicas justificadas en LEARNINGS.md
4. **Rigor:** Comparación con baselines y métricas de fidelidad

**Última actualización:** Octubre 2025 - Módulo I completado