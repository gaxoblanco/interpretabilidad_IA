# 🔍 Interpretabilidad en NLP con DistilBERT

Proyecto educativo del **Módulo II** enfocado en explicar predicciones de modelos Transformer usando **SHAP** y **LIME**.

# Para iniciar el dashboard interactivo:
```bash
streamlit run app.py
```
---

## 🎯 Objetivo del Proyecto

Implementar y comparar dos técnicas de interpretabilidad (SHAP y LIME) para explicar las predicciones de un modelo **DistilBERT** en la tarea de **análisis de sentimientos**.

### Pregunta Central
> **"Para este modelo de sentimientos, ¿qué método (SHAP o LIME) me da explicaciones más útiles y en qué situaciones?"**

---

## 📊 Dataset

- **Fuente:** IMDb Movie Reviews (HuggingFace Datasets)
- **Tarea:** Clasificación binaria de sentimientos (Positivo/Negativo)
- **Tamaño:** 50,000 reviews (25k train / 25k test)
- **Promedio de palabras por review:** ~230 tokens

---

## 🧠 Modelo Base

**DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`)
- Versión destilada de BERT (66% menos parámetros)
- Pre-entrenado en SST-2 (Stanford Sentiment Treebank)
- **Arquitectura:**
  - 6 capas Transformer
  - 12 attention heads por capa
  - 768 dimensiones de embedding
  - 66M parámetros totales

**Métricas esperadas en IMDb:**
- Accuracy: ~91-93%
- F1-Score: ~0.92

---

## 🛠️ Stack Tecnológico

```
Python 3.9+
transformers 4.30.0    # HuggingFace Transformers
torch 2.0.0            # PyTorch
datasets 2.12.0        # HuggingFace Datasets
shap 0.42.0            # SHAP explainer
lime 0.2.0.1           # LIME explainer
streamlit 1.22.0       # Dashboard interactivo
pandas 2.0.0           # Manipulación de datos
matplotlib 3.7.0       # Visualizaciones
scikit-learn 1.2.0     # Métricas y utilidades
```

---

## 📁 Estructura del Proyecto

```
nlp-interpretability-project/
│
├── src/
│   ├── models/
│   │   └── model_loader.py          # Carga y predicción con DistilBERT
│   │
│   ├── interpretability/
│   │   ├── shap_analyzer.py         # Explicaciones SHAP
│   │   └── lime_analyzer.py         # Explicaciones LIME
│   │
│   ├── utils/
│   │   ├── data_loader.py           # Carga del dataset IMDb
│   │   └── text_preprocessor.py     # Preprocesamiento de texto
│   │
│   ├── visualization/
│   │   └── text_viz.py              # Visualizaciones de explicaciones
│   │
│   └── config/
│       └── config.yaml              # Configuraciones centralizadas
│
├── notebooks/
│   ├── 01_shap_lime_toy_example.ipynb     # Ejemplo toy (datos tabulares)
│   ├── 02_model_evaluation.ipynb          # Evaluación del modelo base
│   ├── 03_shap_analysis.ipynb             # Análisis con SHAP
│   ├── 04_lime_analysis.ipynb             # Análisis con LIME
│   ├── 05_visualization.ipynb             # Galería de visualizaciones
│   └── 06_validation.ipynb                # Métricas de validación
│
├── docs/
│   ├── fundamentos.md                     # Conceptos teóricos respondidos
│   └── LEARNINGS.md                       # Insights finales del proyecto
│
├── data/
│   └── cache/                             # Cache de modelos y datasets
│
├── app.py                                 # Dashboard Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

## 📖 Uso Básico

### Predicción Simple

```python
from src.models.model_loader import ModelLoader

# Cargar modelo
model = ModelLoader(model_name="distilbert-base-uncased-finetuned-sst-2-english")

# Predecir sentimiento
text = "This movie was absolutely fantastic!"
result = model.predict(text)

print(f"Predicción: {result['predictions'][0]}")  # 'POSITIVE'
print(f"Confianza: {result['probabilities'][0]:.2%}")  # 98.5%
```

### Explicar con SHAP

```python
from src.interpretability.shap_analyzer import SHAPAnalyzer

# Inicializar analizador
shap_analyzer = SHAPAnalyzer(model.model, model.tokenizer)

# Obtener explicación
explanation = shap_analyzer.explain_instance(text)

# Visualizar
shap_analyzer.plot_waterfall(explanation)
```

### Explicar con LIME

```python
from src.interpretability.lime_analyzer import LIMEAnalyzer

# Inicializar analizador
lime_analyzer = LIMEAnalyzer(model.model, model.tokenizer)

# Obtener explicación
explanation = lime_analyzer.explain_instance(text, num_features=10)

# Visualizar
lime_analyzer.plot_explanation(explanation)
```

---

## 🎨 Dashboard Interactivo

Ejecutar la aplicación Streamlit:

```bash
streamlit run app.py
```

Funcionalidades:
- ✍️ Ingresar texto personalizado para análisis
- 🔮 Ver predicción y probabilidades
- 📊 Comparar explicaciones SHAP vs LIME lado a lado
- 🎯 Identificar palabras más influyentes
- 📈 Visualizar importancia global de tokens

---

## 📚 Roadmap de Desarrollo (7 Semanas)

| Semana | Objetivo | Entregable |
|--------|----------|------------|
| **1** | Fundamentos teóricos | `docs/fundamentos.md` |
| **2** | Teoría SHAP/LIME | `notebooks/01_toy_example.ipynb` |
| **3** | Setup del proyecto | `src/models/` + evaluación base |
| **4** | Implementación SHAP | `src/interpretability/shap_analyzer.py` |
| **5** | Implementación LIME | `src/interpretability/lime_analyzer.py` |
| **6** | Visualización | `app.py` (Dashboard) |
| **7** | Validación y análisis | `LEARNINGS.md` |

---

## 🔬 Metodología de Comparación

### Métricas de Evaluación

1. **Fidelidad**: ¿Las explicaciones reflejan fielmente el modelo?
2. **Estabilidad**: ¿Explicaciones consistentes para textos similares?
3. **Eficiencia**: Tiempo de cómputo por explicación
4. **Interpretabilidad**: Facilidad de comprensión humana

### Casos de Estudio

Analizaremos 10 casos que incluyen:
- Sentimientos claramente positivos/negativos
- Casos ambiguos o sarcásticos
- Textos largos vs cortos
- Acuerdo y desacuerdo entre SHAP y LIME

---

## 🎓 Conceptos Clave Aprendidos

### SHAP (SHapley Additive exPlanations)
- Base en teoría de juegos (valores de Shapley)
- Garantías matemáticas formales
- Explicaciones globales + locales
- Más lento pero más riguroso

### LIME (Local Interpretable Model-agnostic Explanations)
- Aproximación lineal local
- Basado en perturbaciones del input
- Solo explicaciones locales
- Más rápido pero estocástico

### Transformers
- Mecanismo de self-attention
- Positional embeddings
- Multi-head attention
- Fine-tuning de modelos pre-entrenados

---

## ⚠️ Limitaciones Conocidas

1. **SHAP en texto es lento**: ~30-60 seg por explicación
2. **LIME es estocástico**: Resultados varían entre ejecuciones
3. **Contexto limitado**: DistilBERT tiene límite de 512 tokens
4. **Masking vs Removal**: Diferentes estrategias de perturbación
5. **OOD (Out-of-distribution)**: Perturbaciones pueden crear textos irreales

---

## 📝 Resultados Esperados

Al finalizar el proyecto, serás capaz de:
- ✅ Explicar predicciones de cualquier modelo Transformer
- ✅ Identificar qué palabras influyen más en las decisiones
- ✅ Comparar ventajas/desventajas de SHAP vs LIME
- ✅ Crear visualizaciones interpretables
- ✅ Validar calidad de explicaciones

---

## 🔗 Referencias

### Papers Fundamentales
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017)
- [LIME Paper](https://arxiv.org/abs/1602.04938) - Ribeiro et al. (2016)
- [SHAP Paper](https://arxiv.org/abs/1705.07874) - Lundberg & Lee (2017)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108) - Sanh et al. (2019)

### Recursos Adicionales
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [HuggingFace Course](https://huggingface.co/course)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)

---

## 👨‍💻 Autor

Proyecto educativo del **Módulo II: Interpretabilidad en NLP**

---

## 📄 Licencia

MIT License - Proyecto educativo de código abierto

---

## 🤝 Contribuciones

Este es un proyecto educativo. Si encuentras errores o tienes sugerencias:
1. Abre un Issue describiendo el problema
2. Propón mejoras mediante Pull Requests
3. Comparte tus propios experimentos

---

## ⏭️ Próximos Pasos (Módulo III)

- Interpretabilidad en modelos generativos (GPT)
- Attention visualization
- Probing tasks
- Adversarial examples

---
