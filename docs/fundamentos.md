# Rama 1 — Fundamentos
## Técnicas clásicas de interpretabilidad en ML

---

## ¿De qué trata esta rama?

La interpretabilidad clásica responde una pregunta central: **¿por qué el modelo tomó esta decisión?**

Las herramientas de esta rama son model-agnostic (funcionan con cualquier modelo) o model-specific (aprovechan la estructura interna de una arquitectura). El foco está en atribuir importancia a features, tokens, píxeles o neuronas — y en comunicar esa importancia de forma comprensible.

Esta rama es el prerequisito de las otras dos. Sin entender bien cómo funcionan SHAP, LIME, GradCAM y el análisis de activaciones, es difícil criticarlos (Rama 2) o ir más profundo (Rama 3).

---

## Mapa de módulos

| Módulo | Tipo de datos | Modelo | Técnicas principales | Estado |
|--------|--------------|--------|----------------------|--------|
| I | Tabular | XGBoost | SHAP (TreeExplainer), Counterfactuals | ✅ Completado |
| II | Texto / NLP | DistilBERT | SHAP (KernelExplainer), LIME, Attention | ✅ Completado |
| III | Imágenes (activaciones) | ResNet / AlexNet | Activation maps, Feature visualization, Causal intervention | 🟡 En progreso |
| IV | Imágenes (saliency) | ResNet / ViT | GradCAM, Integrated Gradients, Saliency Maps | ⬜ Planificado |

---

## Módulo I — Interpretabilidad en datos tabulares

**Pregunta guía:** ¿Qué features del dataset empuja al modelo hacia una u otra predicción?

### Técnicas
- **SHAP TreeExplainer** — valores de Shapley optimizados para modelos tree-based. Atribución exacta, no aproximada.
- **Counterfactual Explanations** — ¿qué cambio mínimo en los inputs invertiría la predicción?
- **Feature importance global vs local** — importancia promedio del dataset vs importancia para una instancia específica.

### Herramientas
- `shap` (TreeExplainer, waterfall plots, force plots, dependence plots)
- `scikit-learn` (Logistic Regression como baseline interpretable)
- `xgboost`
- `streamlit` (dashboard de predicción + explicación)

### Entregables
- [ ] Notebook EDA del dataset
- [ ] Notebook de modelado (XGBoost vs Logistic Regression baseline)
- [ ] Notebook de análisis SHAP con visualizaciones
- [ ] Notebook de counterfactuals
- [ ] Dashboard Streamlit: input → predicción → explicación SHAP

### Conceptos clave a dominar
- Diferencia entre interpretabilidad **global** (comportamiento del modelo en todo el dataset) y **local** (explicación de una instancia)
- Propiedades teóricas de SHAP: eficiencia, simetría, linealidad, dummy
- Por qué Logistic Regression es el baseline interpretable de referencia
- Limitaciones de counterfactuals (pueden ser no realistas, no únicos)

### Recursos
- 📄 **Paper fundacional SHAP:** Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions* — [arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)
- 📄 **Paper counterfactuals:** Wachter et al. (2017) — *Counterfactual Explanations Without Opening the Black Box* — [arxiv.org/abs/1711.00399](https://arxiv.org/abs/1711.00399)
- 📖 **Libro:** Christoph Molnar — *Interpretable Machine Learning* (capítulos 5 y 9) — [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book)
- 🎥 **Video:** SHAP explicado visualmente — [youtube.com/watch?v=VB9uV-x0gtg](https://youtube.com/watch?v=VB9uV-x0gtg)

**Estado:** ✅ Completado — Dataset: German Credit Data (1000 instancias, 20 features)

---

## Módulo II — Interpretabilidad en NLP / Transformers

**Pregunta guía:** ¿Qué tokens (palabras) del texto son responsables de la predicción del modelo?

### Técnicas
- **SHAP KernelExplainer** — perturbación de tokens para estimar contribuciones. Más lento que TreeExplainer pero funciona con cualquier modelo.
- **LIME para texto** — modelo sustituto lineal entrenado localmente alrededor de una instancia. Elimina palabras y observa cambios en la predicción.
- **Attention weights** — visualización de qué tokens "mira" el modelo en cada capa. Importante: attention ≠ explanation (debate abierto en literatura).

### Herramientas
- `transformers` (HuggingFace) + `torch`
- `shap` (KernelExplainer, TextMasker)
- `lime` (LimeTextExplainer)
- `bertviz` (visualización de attention heads)
- `plotly` (visualizaciones interactivas)

### Entregables
- [ ] Módulo `model_loader.py` — carga de DistilBERT + tokenizer + pipeline de predicción
- [ ] Módulo `shap_analyzer.py` — wrapper de KernelExplainer con sampling estratégico
- [ ] Módulo `lime_analyzer.py` — LimeTextExplainer con métricas de estabilidad
- [ ] Notebook comparativo SHAP vs LIME vs Attention
- [ ] Análisis de casos donde los tres métodos divergen

### Conceptos clave a dominar
- Por qué SHAP en texto es más difícil que en tabular (tokens no independientes, contexto secuencial)
- Diferencia entre perturbación a nivel de token (SHAP) vs palabra (LIME)
- El debate "attention is not explanation" — cuándo las attention weights sí son informativas
- Métricas de comparación: fidelidad, estabilidad, comprensibilidad

### Recursos
- 📄 **Paper LIME:** Ribeiro et al. (2016) — *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* — [arxiv.org/abs/1602.04938](https://arxiv.org/abs/1602.04938)
- 📄 **Paper attention:** Jain & Wallace (2019) — *Attention is not Explanation* — [arxiv.org/abs/1902.10186](https://arxiv.org/abs/1902.10186)
- 📄 **Paper respuesta:** Wiegreffe & Pinter (2019) — *Attention is not not Explanation* — [arxiv.org/abs/1908.04626](https://arxiv.org/abs/1908.04626)
- 📄 **DistilBERT:** Sanh et al. (2019) — *DistilBERT, a distilled version of BERT* — [arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)
- 🔧 **BertViz:** herramienta de visualización de attention — [github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)

**Estado:** ✅ Completado — Dataset: IMDb Reviews (50k reviews, clasificación binaria de sentimiento)

---

## Módulo III — Análisis de activaciones neuronales

**Pregunta guía:** ¿Qué aprenden las neuronas individuales de una CNN? ¿Qué detecta cada filtro?

### Técnicas
- **PyTorch Hooks** — captura de activaciones de cualquier capa en tiempo de inferencia sin modificar el modelo.
- **Feature visualization / Activation maximization** — generar imágenes sintéticas que maximizan la activación de una neurona específica. Muestra el "patrón ideal" que busca el filtro.
- **Activation maps / Heatmaps** — superponer las activaciones sobre la imagen original. Muestra *dónde* reacciona el filtro.
- **Causal intervention** — knockout (anular neuronas), isolation (dejar solo una activa), amplification (multiplicar activación). Mide el impacto causal de cada neurona en la predicción.
- **Neuron Probing** — entrenar un clasificador lineal simple sobre las activaciones de una capa para medir qué información está codificada ahí. *(Pendiente)*

### Herramientas
- `torch` + `torchvision` (ResNet-18, AlexNet pre-entrenados en ImageNet)
- `captum` (Integrated Gradients, Feature Ablation)
- `matplotlib` + `PIL` (visualización de filtros y heatmaps)
- `streamlit` (dashboard interactivo)
- `scikit-learn` (clasificadores lineales para probing)

### Entregables
- [x] `model_loader.py` — carga de ResNet/AlexNet con hooks
- [x] `hooks.py` — sistema de captura de activaciones por capa
- [x] `image_loader.py` — carga y preprocesamiento de imágenes
- [x] App `neural_feature_visualization` — activation maximization con AlexNet
- [x] App `CNN_activation_analyzer` — heatmaps, knockout, isolation, amplification
- [ ] `neuron_probe.py` — probing classifiers por capa
- [ ] Análisis: ¿qué codifica cada capa? (colores → bordes → formas → objetos)
- [ ] `LEARNINGS.md` completo con insights y limitaciones

### Conceptos clave a dominar
- Diferencia entre **pesos del filtro** (patrón buscado, fijo) y **activaciones** (respuesta a una imagen específica)
- Por qué capas tempranas detectan features simples (bordes, texturas) y capas profundas detectan conceptos complejos
- Qué significa la **sparsity** de activaciones (ReLU, neuronas muertas)
- Limitaciones de activation maximization: las imágenes sintéticas no siempre son interpretables para humanos
- Transfer learning: por qué congelar capas (requires_grad=False) es correcto para preservar features de ImageNet

### Learnings ya capturados
- Los filtros de capas tempranas son fáciles de interpretar en el heatmap (bordes, contornos). Los filtros profundos producen patrones demasiado complejos para identificar visualmente.
- Posicionar el filtro sobre la imagen original y observar el mapa de calor resultante revela mucho sobre el tamaño y posición del patrón buscado.
- Congelar parámetros (requires_grad=False) no es solo una optimización — es conceptualmente correcto cuando se quiere analizar las representaciones aprendidas, no modificarlas.

### Recursos
- 📄 **Feature visualization:** Olah et al. (2017) — *Feature Visualization* — [distill.pub/2017/feature-visualization](https://distill.pub/2017/feature-visualization)
- 📄 **Zoom In:** Olah et al. (2020) — *Zoom In: An Introduction to Circuits* — [distill.pub/2020/circuits/zoom-in](https://distill.pub/2020/circuits/zoom-in)
- 📄 **DeepDream:** Mordvintsev et al. (2015) — *Inceptionism: Going Deeper into Neural Networks* — [ai.googleblog.com](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
- 📄 **Simonyan et al. (2013)** — *Deep Inside Convolutional Networks* — [arxiv.org/abs/1312.6034](https://arxiv.org/abs/1312.6034)
- 🔧 **Captum:** librería de interpretabilidad de PyTorch — [captum.ai](https://captum.ai)

**Estado:** 🟡 En progreso — Steps 1-3 completados (setup, activaciones, feature viz). Steps 4-7 pendientes.

---

## Módulo IV — Saliency maps en Computer Vision

**Pregunta guía:** ¿Qué regiones de la imagen influyeron en la predicción del modelo? ¿Está mirando lo correcto?

### Técnicas
- **GradCAM** — usa los gradientes de la clase predicha respecto a los feature maps de la última capa convolucional. Produce un heatmap grueso de importancia espacial.
- **Guided Backpropagation** — backprop modificado que solo deja pasar gradientes positivos. Más detallado que GradCAM pero menos fiel a la predicción.
- **Integrated Gradients** — integra los gradientes a lo largo de un camino lineal desde un baseline (imagen negra) hasta la imagen real. Satisface propiedades de atribución más fuertes que GradCAM.
- **SmoothGrad** — reduce el ruido de los saliency maps promediando gradientes sobre versiones ruidosas de la imagen.
- **Sanity checks** — comparar explicaciones contra un modelo con pesos aleatorios. Si los mapas se ven igual, la explicación no está capturando nada real.

### Herramientas
- `torch` + `torchvision`
- `captum` (IntegratedGradients, GradCAM, SmoothGrad, GuidedBackprop)
- `grad-cam` (librería de Jacob Gildenblat — más fácil de usar que captum para GradCAM)
- `matplotlib` + `PIL`
- `streamlit` (dashboard interactivo)

### Entregables
- [ ] Módulo `gradcam.py` — implementación de GradCAM con selección de capa target
- [ ] Módulo `integrated_gradients.py` — IG con baseline configurable
- [ ] Módulo `saliency_viz.py` — visualización side-by-side de métodos
- [ ] Notebook comparativo: GradCAM vs Guided Backprop vs IG sobre mismas imágenes
- [ ] Sanity checks implementados y documentados
- [ ] Dashboard: subir imagen → comparar 3 métodos de saliency simultáneamente
- [ ] Análisis de casos donde GradCAM mira "lo incorrecto" y por qué

### Conceptos clave a dominar
- Por qué GradCAM es grueso (resolución del feature map) y cómo Guided GradCAM mejora eso
- La diferencia entre **atribución** (qué pixels importaron) y **saliency** (qué pixels el modelo fue sensible a)
- Integrated Gradients y por qué satisface el axioma de completeness: la suma de todas las atribuciones iguala la diferencia entre la predicción real y la del baseline
- Por qué los sanity checks son obligatorios antes de confiar en cualquier saliency map
- Deletion/insertion curves: métrica cuantitativa de calidad de explicaciones

### Recursos
- 📄 **GradCAM:** Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* — [arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
- 📄 **Integrated Gradients:** Sundararajan et al. (2017) — *Axiomatic Attribution for Deep Networks* — [arxiv.org/abs/1703.01365](https://arxiv.org/abs/1703.01365)
- 📄 **SmoothGrad:** Smilkov et al. (2017) — *SmoothGrad: removing noise by adding noise* — [arxiv.org/abs/1706.03825](https://arxiv.org/abs/1706.03825)
- 📄 **Sanity checks:** Adebayo et al. (2018) — *Sanity Checks for Saliency Maps* — [arxiv.org/abs/1810.03292](https://arxiv.org/abs/1810.03292)
- 🔧 **pytorch-grad-cam:** [github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

**Estado:** ⬜ Planificado — no iniciado

---

## Criterios de graduación de esta rama

Para considerar la Rama 1 completa:

- [ ] Los 4 módulos tienen sus entregables de código y notebooks
- [ ] Cada módulo tiene su `LEARNINGS.md` documentado
- [ ] Se puede comparar al menos dos técnicas dentro de cada módulo (ej: SHAP vs LIME, GradCAM vs IG)
- [ ] Se identificaron al menos 2 limitaciones concretas por módulo
- [ ] El código de cada módulo es reutilizable: clases con interfaz limpia, sin hardcoding

---

*Última actualización: Módulo III en progreso*
