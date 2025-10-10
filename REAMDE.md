# 🧠 Interpretabilidad en Machine Learning

Proyecto educativo progresivo para dominar técnicas de interpretabilidad en ML, siguiendo el roadmap estructurado en [`roadmap.md`](roadmap.md).

---

## 📂 Módulos

### ✅ `interpretability_I/` - German Credit Risk (XGBoost + SHAP)
**Estado:** Completado | **Tipo:** Datos Tabulares

Análisis de riesgo crediticio con modelos de clasificación tabular.

- **Modelo:** XGBoost (78% accuracy, 0.809 ROC-AUC)
- **Dataset:** German Credit Data (1000 clientes, 20 features)
- **Técnicas:** SHAP values, Waterfall plots, Counterfactual explanations
- **Dashboard:** Streamlit interactivo con predicciones en tiempo real

**Aprendizajes clave:**
- SHAP > Feature Importance para interpretabilidad real
- Counterfactuals: explicabilidad accionable ("qué cambiar para aprobar")
- Trade-offs: Recall vs Precision en clasificación desbalanceada
- Comunicación: Traducir valores SHAP a lenguaje de negocio

📁 **Contenido:**
```
interpretability_I/
├── notebooks/           # 4 notebooks: EDA, Modelado, SHAP, Counterfactuals
├── app.py              # Dashboard Streamlit interactivo
├── models/             # Modelos y encoders guardados
├── data/               # Dataset procesado
├── LEARNINGS.md        # Insights y decisiones documentadas
└── README.md           # Documentación específica del módulo
```

---

### 🔜 `interpretability_II/` - Text Classification (DistilBERT + SHAP + LIME)
**Estado:** Próximamente | **Tipo:** NLP

Interpretabilidad en modelos de clasificación de texto con Transformers.

**Roadmap detallado:** Ver [`roadmap.md`](roadmap.md) - Proyecto 1

- **Modelo:** DistilBERT fine-tuned para análisis de sentimientos
- **Dataset:** IMDb Reviews (50k reviews balanceados)
- **Técnicas:** SHAP para texto, LIME, Attention visualization
- **Fases:**
  - **Parte I:** Fundamentos teóricos (Transformers, SHAP, LIME)
  - **Parte II:** Implementación (ModelLoader, SHAPAnalyzer, LIMEAnalyzer)
  - **Parte III:** Análisis avanzado (Validación, casos de estudio)

**Duración estimada:** 7-8 semanas

---

### 🔮 `interpretability_III/` - Neuron Activation Analysis
**Estado:** Planificado | **Tipo:** Análisis Interno de Modelos

Interpretabilidad mediante activación de neuronas (Proyecto 2 según roadmap).

- **Objetivo:** Entender qué representan neuronas individuales en redes profundas
- **Técnicas:** Feature visualization, Activation maximization, Neuron probing
- **Modelos:** Por definir (CNN o Transformer según hallazgos del Proyecto 1)

---

### 🔮 `interpretability_IV/` - Computer Vision (CNN + GradCAM)
**Estado:** Planificado | **Tipo:** Computer Vision

Interpretabilidad en clasificación de imágenes.

- **Modelo:** ResNet / EfficientNet
- **Dataset:** Por definir
- **Técnicas:** GradCAM, Integrated Gradients, Saliency Maps
- **Objetivo:** Visualizar qué regiones de imágenes activan el modelo

---

## 🗺️ Roadmap General

| Módulo | Enfoque | Estado | Duración |
|--------|---------|--------|----------|
| **I** | XGBoost + SHAP (Tabular) | ✅ Completado | 3 semanas |
| **II** | DistilBERT + LIME (NLP) | 🔄 Siguiente | 7-8 semanas |
| **III** | Neuron Activation Analysis | 📋 Planificado | Por definir |
| **IV** | CNN + GradCAM (Vision) | 📋 Planificado | Por definir |

📘 **Roadmap completo:** [`roadmap.md`](roadmap.md) | 📊 **Notas de estudio:** [`roadmap_study.md`](roadmap_study.md)

---

## 🛠️ Stack Tecnológico

| Categoría | Módulo I | Módulo II (Planeado) | Módulo III-IV (Planeado) |
|-----------|----------|----------------------|--------------------------|
| **ML Frameworks** | XGBoost, Scikit-learn | Transformers, PyTorch | PyTorch, TensorFlow |
| **Interpretabilidad** | SHAP | SHAP, LIME | Captum, GradCAM |
| **Visualización** | Matplotlib, Plotly, Streamlit | Plotly, Streamlit | Plotly, PIL |
| **Data** | Pandas, NumPy | Datasets (HF), Pandas | PIL, OpenCV |

---

## 🚀 Inicio Rápido

```bash
# Clonar repositorio
git clone <repo-url>

# Navegar a módulo deseado
cd interpretability_I

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar dashboard (ejemplo módulo I)
streamlit run app.py
```

---

## 📈 Progreso

- [x] **Módulo I:** German Credit + SHAP + Counterfactuals + Dashboard
- [ ] **Módulo II:** DistilBERT + SHAP + LIME (según roadmap.md)
  - [ ] Parte I: Fundamentos teóricos (3 capítulos)
  - [ ] Parte II: Implementación práctica (5 capítulos)
  - [ ] Parte III: Análisis avanzado (3 capítulos)
- [ ] **Módulo III:** Neuron Activation Analysis
- [ ] **Módulo IV:** CNN + GradCAM

---

## 📚 Recursos

- 📘 **Roadmap completo:** [`roadmap.md`](roadmap.md) - Guía detallada por capítulos y semanas
- 📊 **Study notes:** [`roadmap_study.md`](roadmap_study.md) - Notas de estudio y referencias
- 📝 **Learnings por módulo:** Ver `LEARNINGS.md` en cada carpeta del módulo

---

**Última actualización:** Módulo I completado - Octubre 2025