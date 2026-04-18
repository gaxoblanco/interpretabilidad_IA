# Syllabus de papers — Lectura obligatoria por rama y nivel

Cada paper tiene una etiqueta de prioridad:
- 🔴 **Obligatorio** — no avanzar al siguiente nivel sin haberlo leído
- 🟡 **Recomendado** — complementa el obligatorio, leer antes de implementar
- 🔵 **Referencia** — consultar cuando la técnica aparece en la práctica, no leer completo

Formato de cada entrada:
> **Título corto** — Autores (año)
> Por qué leerlo / qué aporta
> `[arxiv]` `[web]` `[tiempo estimado]`

---

## 🧱 RAMA 1 — Fundamentos

### Módulo I — Tabular / XGBoost

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 1 | SHAP unificado | 🔴 | Antes de implementar SHAP |
| 2 | LIME original | 🔴 | Antes de implementar LIME |
| 3 | Counterfactuals Wachter | 🟡 | Antes del notebook de counterfactuals |
| 4 | Stop explaining black boxes (Rudin) | 🟡 | Al terminar el módulo |
| 5 | Shapley values teoría de juegos | 🔵 | Si querés entender la matemática base |

---

**1.** 🔴 **A Unified Approach to Interpreting Model Predictions**
Lundberg & Lee (2017) — *el paper que define SHAP*
Explica cómo unificar LIME, DeepLIFT y Shapley values en un único framework aditivo. La Sección 2 es la más importante: define las propiedades que cualquier explicación debería cumplir. La Sección 3 presenta TreeExplainer.
`[arxiv.org/abs/1705.07874]` `[~45 min, leer completo]`

**2.** 🔴 **"Why Should I Trust You?": Explaining the Predictions of Any Classifier**
Ribeiro, Singh & Guestrin (2016) — *el paper original de LIME*
Define el problema de explicabilidad local. Secciones 1-3 son esenciales. Sección 4 tiene la implementación. Sección 5 tiene los experimentos de evaluación humana — muy útil para entender cómo medir si una explicación es "buena".
`[arxiv.org/abs/1602.04938]` `[~50 min, leer completo]`

**3.** 🟡 **Counterfactual Explanations Without Opening the Black Box**
Wachter, Mittelstadt & Russell (2017)
Formaliza el concepto de explicación contrafáctica. La Sección 2 define las propiedades que debe tener un contrafáctico válido (proximidad, accionabilidad). Útil antes de implementar tu propio generador.
`[arxiv.org/abs/1711.00399]` `[~30 min, secciones 1-3]`

**4.** 🟡 **Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead**
Rudin (2019) — *Nature Machine Intelligence*
Artículo de opinión de la investigadora más citada del campo. Argumento central: en decisiones de alto impacto, los modelos post-hoc explicados son inferiores a modelos intrínsecamente interpretables. Leerlo al terminar el módulo cambia la perspectiva sobre cuándo tiene sentido usar SHAP.
`[arxiv.org/abs/1811.10154]` `[~25 min, leer completo]`

**5.** 🔵 **Shapley Values — Teoría de juegos cooperativos**
Shapley (1953) — *Contributions to the Theory of Games*
El paper matemático original. Solo si querés entender de dónde vienen las propiedades de SHAP (eficiencia, simetría, dummy, aditividad). No es necesario para implementar.
`[acceso via JSTOR o Wikipedia entry de Shapley value]` `[~1h, lectura técnica]`

---

### Módulo II — NLP / Transformers

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 6 | Attention is All You Need | 🔴 | Antes de trabajar con DistilBERT |
| 7 | DistilBERT | 🔴 | Antes de cargar el modelo |
| 8 | Attention is not Explanation | 🔴 | Antes de visualizar attention weights |
| 9 | Attention is not not Explanation | 🟡 | Inmediatamente después del 8 |
| 10 | BERT original | 🟡 | Para entender de dónde viene DistilBERT |

---

**6.** 🔴 **Attention Is All You Need**
Vaswani et al. (2017) — *el paper fundacional de Transformers*
Secciones 1, 2 y 3 son obligatorias: definen self-attention, multi-head attention y la arquitectura completa. Sección 4 en adelante son experimentos, se puede omitir. Sin entender este paper, analizar attention weights no tiene sentido.
`[arxiv.org/abs/1706.03762]` `[~1h, secciones 1-3 obligatorias]`

**7.** 🔴 **DistilBERT, a distilled version of BERT**
Sanh et al. (2019)
Paper corto (~6 páginas). Explica qué se eliminó de BERT y por qué el modelo mantiene 97% del performance con 40% menos parámetros. Crítico para entender las limitaciones del modelo que se analiza.
`[arxiv.org/abs/1910.01108]` `[~20 min, leer completo]`

**8.** 🔴 **Attention is not Explanation**
Jain & Wallace (2019)
Demuestra empíricamente que las attention weights NO son explicaciones confiables: se pueden cambiar los pesos de atención sin cambiar la predicción. Fundamental antes de confiar en cualquier visualización de attention.
`[arxiv.org/abs/1902.10186]` `[~40 min, leer completo]`

**9.** 🟡 **Attention is not not Explanation**
Wiegreffe & Pinter (2019)
Respuesta al paper anterior. Argumento: la afirmación de Jain & Wallace es demasiado fuerte. En ciertas condiciones, attention sí es informativa. Leer en conjunto con el 8 para tener una visión equilibrada del debate.
`[arxiv.org/abs/1908.04626]` `[~40 min, leer completo]`

**10.** 🟡 **BERT: Pre-training of Deep Bidirectional Transformers**
Devlin et al. (2018)
El paper de BERT del que deriva DistilBERT. Leer secciones 1-3 para entender el pre-training con MLM y NSP. No es obligatorio si ya se entiende DistilBERT, pero da contexto importante.
`[arxiv.org/abs/1810.04805]` `[~40 min, secciones 1-3]`

---

### Módulo III — Activaciones neuronales / CNN

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 11 | Feature Visualization (Distill) | 🔴 | Antes de implementar activation maximization |
| 12 | Zoom In: An Introduction to Circuits | 🔴 | Al empezar el módulo |
| 13 | Deep Inside Convolutional Networks | 🟡 | Antes de implementar gradient-based viz |
| 14 | Network Dissection | 🟡 | Antes de neuron probing |
| 15 | DeepDream / Inceptionism | 🔵 | Para contexto histórico |

---

**11.** 🔴 **Feature Visualization**
Olah et al. (2017) — *Distill.pub*
Artículo interactivo, no paper tradicional. Explica activation maximization con visualizaciones en vivo. La sección sobre regularización es crítica para entender por qué las imágenes sintéticas sin regularizar son ruido. Uno de los mejores recursos de comunicación científica que existen.
`[distill.pub/2017/feature-visualization]` `[~1h, leer completo + explorar interactivos]`

**12.** 🔴 **Zoom In: An Introduction to Circuits**
Olah et al. (2020) — *Distill.pub*
Presenta los tres claims centrales del campo: features son reales, circuits son reales, la universalidad es posible. Leer antes de cualquier análisis de activaciones para tener el marco conceptual correcto.
`[distill.pub/2020/circuits/zoom-in]` `[~45 min, leer completo]`

**13.** 🟡 **Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps**
Simonyan, Vedaldi & Zisserman (2013)
El paper que introduce gradient-based saliency maps. Sección 2 define la técnica. Útil antes de implementar cualquier visualización basada en gradientes.
`[arxiv.org/abs/1312.6034]` `[~30 min, secciones 1-3]`

**14.** 🟡 **Network Dissection: Quantifying Interpretability of Deep Visual Representations**
Bau et al. (2017)
Metodología para medir cuántos conceptos humanos corresponden a neuronas individuales. Muy relevante para neuron probing: establece el framework de "una neurona = un concepto" y sus límites.
`[arxiv.org/abs/1704.05796]` `[~45 min, secciones 1-4]`

**15.** 🔵 **Inceptionism: Going Deeper into Neural Networks**
Mordvintsev et al. (2015) — *Google AI Blog*
El post original de DeepDream. Referencia histórica para entender cómo llegamos a activation maximization. Lectura rápida, sin matemáticas.
`[ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html]` `[~15 min]`

---

### Módulo IV — Saliency Maps / Computer Vision

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 16 | GradCAM | 🔴 | Antes de implementar GradCAM |
| 17 | Integrated Gradients | 🔴 | Antes de implementar IG |
| 18 | Sanity Checks for Saliency Maps | 🔴 | Antes de confiar en cualquier resultado |
| 19 | SmoothGrad | 🟡 | Cuando los saliency maps sean muy ruidosos |
| 20 | RISE | 🔵 | Para entender deletion/insertion curves |

---

**16.** 🔴 **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**
Selvaraju et al. (2017)
Define GradCAM y Guided GradCAM. Secciones 1-3 son esenciales. La Figura 1 es el resumen visual más claro de cómo funciona el método. Indispensable antes de implementar.
`[arxiv.org/abs/1610.02391]` `[~40 min, secciones 1-4]`

**17.** 🔴 **Axiomatic Attribution for Deep Networks (Integrated Gradients)**
Sundararajan, Taly & Yan (2017)
Define los axiomas que debería cumplir cualquier método de atribución (completeness, sensitivity, implementation invariance). Muestra que gradientes simples los violan. IG es la solución. Sección 2 y 3 son críticas.
`[arxiv.org/abs/1703.01365]` `[~45 min, secciones 1-4]`

**18.** 🔴 **Sanity Checks for Saliency Maps**
Adebayo et al. (2018)
*Lectura obligatoria antes de publicar o confiar en cualquier saliency map.* Demuestra que Guided Backprop y Guided GradCAM fallan el test más básico: sus mapas son insensibles a los pesos del modelo. Uno de los papers más importantes de la rama de Crítica también.
`[arxiv.org/abs/1810.03292]` `[~45 min, leer completo]`

**19.** 🟡 **SmoothGrad: Removing Noise by Adding Noise**
Smilkov et al. (2017)
Técnica simple y efectiva para reducir ruido en saliency maps. Promediar gradientes sobre versiones ruidosas del input. Sección 2 explica el método en dos párrafos.
`[arxiv.org/abs/1706.03825]` `[~20 min, secciones 1-3]`

**20.** 🔵 **RISE: Randomized Input Sampling for Explanation of Black-box Models**
Petsiuk et al. (2018)
Introduce las deletion/insertion curves como métrica cuantitativa de calidad de explicaciones. Útil cuando necesitás comparar métodos numéricamente.
`[arxiv.org/abs/1806.07421]` `[~35 min, secciones 1-3]`

---

## 🔬 RAMA 2 — Crítica

### Nivel 1 — Métricas de evaluación

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 21 | Towards Robust Interpretability (Alvarez-Melis) | 🔴 | Al empezar la rama |
| 22 | Explaining the Explainer (LIME teoría) | 🔴 | Antes de implementar métricas de LIME |
| 23 | Quantus paper | 🟡 | Antes de usar la librería |

---

**21.** 🔴 **Towards Robust Interpretability with Self-Explaining Neural Networks**
Alvarez-Melis & Jaakkola (2018)
Formaliza las propiedades que deberían tener las explicaciones: explicitness, faithfulness, stability. Define estabilidad en términos de Lipschitz continuity. La Sección 2 es el marco teórico más limpio que existe para evaluar explicaciones.
`[arxiv.org/abs/1806.07538]` `[~45 min, secciones 1-3]`

**22.** 🔴 **Explaining the Explainer: A First Theoretical Analysis of LIME**
Garreau & von Luxburg (2020)
Análisis matemático de cuándo LIME da resultados correctos y cuándo no. Demuestra que la elección del kernel y el tamaño del vecindario afectan dramáticamente el resultado. Fundamental antes de confiar en explicaciones de LIME.
`[arxiv.org/abs/2001.03447]` `[~50 min, secciones 1-4]`

**23.** 🟡 **Quantus: An Explainability Toolbox for Responsible Evaluation of Neural Network Explanations**
Hedström et al. (2022)
Paper que acompaña la librería Quantus. Categoriza todas las métricas de evaluación de explicaciones existentes. Útil como referencia cuando elegís qué métricas implementar.
`[arxiv.org/abs/2202.06861]` `[~30 min, tabla de métricas en sección 3]`

---

### Nivel 2 — Sanity checks

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 18 | Sanity Checks (Adebayo) | 🔴 | *Ya listado arriba — releer en este contexto* |
| 24 | The (Un)reliability of Saliency Methods | 🔴 | Al empezar el nivel |

---

**24.** 🔴 **The (Un)reliability of Saliency Methods**
Kindermans et al. (2019)
Demuestra el input invariance test: agregar una constante a todos los pixels no debería cambiar la explicación, pero muchos métodos populares fallan esto. Corto y devastadoramente claro.
`[arxiv.org/abs/1711.00867]` `[~25 min, leer completo]`

---

### Nivel 3 — Divergencias entre métodos

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 25 | The Disagreement Problem in XAI | 🔴 | Al empezar el nivel |
| 26 | Fooling LIME and SHAP | 🔴 | Antes de confiar ciegamente en ningún método |

---

**25.** 🔴 **The Disagreement Problem in Explainable Machine Learning**
Krishna et al. (2022)
Documenta sistemáticamente que distintos métodos de XAI dan explicaciones contradictorias para los mismos modelos e instancias. Cuantifica el grado de desacuerdo. La pregunta "¿cuál tiene razón?" no tiene respuesta fácil.
`[arxiv.org/abs/2202.01602]` `[~45 min, leer completo]`

**26.** 🔴 **Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods**
Slack et al. (2020)
Demuestra que se puede entrenar un modelo que da predicciones normales pero engaña deliberadamente a LIME y SHAP cuando lo interrogan. Rompe la ilusión de que estos métodos son seguros contra manipulación.
`[arxiv.org/abs/1911.02508]` `[~40 min, leer completo]`

---

### Nivel 4 — Sesgos y Clever Hans

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 27 | Unmasking Clever Hans Predictors | 🔴 | Al empezar el nivel |
| 28 | Annotation Artifacts in NLI | 🟡 | Para ver Clever Hans en NLP |
| 29 | Right for the Right Reasons | 🟡 | Para entender cómo corregir RFWR |

---

**27.** 🔴 **Unmasking Clever Hans Predictors and Assessing What Machines Really Learn**
Lapuschkin et al. (2019) — *Nature Communications*
El paper que populariza el término "Clever Hans" en ML. Demuestra que un clasificador de ImageNet famoso aprendió a detectar una marca de agua en lugar del objeto. Imprescindible.
`[arxiv.org/abs/1902.10178]` `[~50 min, leer completo]`

**28.** 🟡 **Annotation Artifacts in Natural Language Inference Data**
Gururangan et al. (2018)
Muestra que modelos de NLI pueden alcanzar alta accuracy usando solo la hipótesis, sin leer la premisa — gracias a correlaciones espurias en los datos de anotación. Ejemplo clásico de Clever Hans en NLP.
`[arxiv.org/abs/1803.02324]` `[~30 min, secciones 1-4]`

**29.** 🟡 **Right for the Right Reasons: Training Differentiably through a Tree**
Ross, Hughes & Doshi-Velez (2017)
Define formalmente el problema RFWR (Right for Wrong Reasons) y propone un método para penalizar modelos que usan features incorrectos. Útil para pensar cómo corregir, no solo detectar.
`[arxiv.org/abs/1703.03717]` `[~40 min, secciones 1-4]`

---

## ⚙️ RAMA 3 — Circuitos internos

### Nivel 1 — Superposición y representaciones

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 30 | Toy Models of Superposition | 🔴 | Primer paper de la rama, leer completo |
| 31 | Linear Representation Hypothesis | 🔴 | Antes de hacer probing |
| 32 | Linguistic Regularities (Word2Vec) | 🟡 | Para contexto histórico de representaciones lineales |

---

**30.** 🔴 **Toy Models of Superposition**
Elhage et al. (2022) — *Anthropic*
*El paper más importante de esta rama.* Define superposición, polisemanticity, y presenta toy models reproducibles en pocas líneas de código. Tiene jupyter notebooks que acompañan el paper. Leer completo incluyendo el apéndice matemático.
`[transformer-circuits.pub/2022/toy_model/index.html]` `[~3-4h, leer completo + correr código]`

**31.** 🔴 **The Linear Representation Hypothesis and the Geometry of Large Language Models**
Park et al. (2023)
Formaliza matemáticamente la hipótesis de que los modelos representan conceptos como direcciones lineales. Define cuándo y por qué la linealidad aparece. Secciones 1-3 son esenciales.
`[arxiv.org/abs/2311.03658]` `[~50 min, secciones 1-4]`

**32.** 🟡 **Linguistic Regularities in Continuous Space Word Representations**
Mikolov et al. (2013) — *el paper de king - man + woman = queen*
Referencia histórica que introduce la idea de aritmética vectorial en embeddings. Corto e influyente.
`[aclanthology.org/N13-1090]` `[~20 min, leer completo]`

---

### Nivel 2 — Circuits

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 33 | Thread: Circuits (Distill) | 🔴 | Al empezar el nivel |
| 34 | In-context Learning and Induction Heads | 🔴 | Antes de analizar induction heads |
| 35 | Interpretability in the Wild (IOI) | 🔴 | Primer ejercicio práctico de circuit discovery |
| 36 | Logit Lens | 🟡 | Antes de implementar logit lens |

---

**33.** 🔴 **Thread: Circuits**
Cammarata et al. (2020) — *Distill.pub*
Artículo que lanza el programa de investigación de circuits. Presenta los primeros resultados: curve detectors, high-low frequency detectors, multimodal neurons. El marco conceptual que guía toda la Rama 3.
`[distill.pub/2020/circuits]` `[~1.5h, leer el artículo principal + explorar los threads]`

**34.** 🔴 **In-context Learning and Induction Heads**
Olsson et al. (2022) — *Anthropic*
Demuestra que los induction heads son el mecanismo central del in-context learning. Introduce activation patching de forma clara. Paper largo pero bien estructurado — las Secciones 1-4 son las más importantes.
`[transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html]` `[~2h, secciones 1-5]`

**35.** 🔴 **Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 small**
Wang et al. (2022)
El primer circuit discovery completo en un LLM real. Analiza paso a paso cómo GPT-2 resuelve "John gave Mary the ball. He gave it to ___". Usa path patching. Es la referencia práctica de cómo se hace circuit discovery.
`[arxiv.org/abs/2211.00593]` `[~1.5h, leer completo]`

**36.** 🟡 **Interpreting GPT: The Logit Lens**
nostalgebraist (2020) — *LessWrong*
Post que introduce la técnica de proyectar activaciones intermedias al espacio de vocabulario. No es un paper académico formal pero es la referencia estándar del método.
`[lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens]` `[~30 min]`

---

### Nivel 3 — Mechanistic interpretability en Transformers

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 37 | Mathematical Framework for Transformer Circuits | 🔴 | *El paper más técnico — base matemática de todo* |
| 38 | Transformer Feed-Forward Layers as Key-Value Memories | 🔴 | Antes de analizar MLPs |
| 39 | Open Problems in Mechanistic Interpretability | 🟡 | Para mapear el campo completo |

---

**37.** 🔴 **A Mathematical Framework for Transformer Circuits**
Elhage et al. (2021) — *Anthropic*
Define el lenguaje matemático del campo: residual stream, QK circuits, OV circuits, composición de heads. Sin este framework, el análisis de Transformers es ad-hoc. Es denso — leer en múltiples sesiones con papel y lápiz.
`[transformer-circuits.pub/2021/framework/index.html]` `[~4-6h, leer completo — es fundamental]`

**38.** 🔴 **Transformer Feed-Forward Layers Are Key-Value Memories**
Geva et al. (2021)
Demuestra empíricamente que las capas MLP actúan como bases de datos de conocimiento factual. Las neuronas en la primera capa son "keys" y en la segunda son "values". Cambia cómo se interpreta el rol de los MLPs.
`[arxiv.org/abs/2012.14913]` `[~45 min, leer completo]`

**39.** 🟡 **Open Problems in Mechanistic Interpretability**
Sharkey et al. (2025)
Review del estado del arte y los problemas abiertos del campo, escrito colectivamente por los principales investigadores. Útil para tener un mapa de lo que no se sabe todavía.
`[arxiv.org/abs/2501.16496]` `[~1h, leer intro + secciones de interés]`

---

### Nivel 4 — Causalidad y ablación

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 40 | ROME: Locating and Editing Factual Associations | 🔴 | Al empezar el nivel |
| 41 | Localizing Model Behavior with Path Patching | 🔴 | Antes de implementar path patching |
| 42 | Causal Abstractions of Neural Networks | 🟡 | Para el framework teórico de causalidad |

---

**40.** 🔴 **Locating and Editing Factual Associations in GPT (ROME)**
Meng et al. (2022)
Introduce causal tracing: cómo identificar exactamente qué capa y qué posición contiene un hecho específico. El experimento de "Paris → Rome" es el ejemplo canónico. Secciones 1-4 obligatorias.
`[arxiv.org/abs/2202.05262]` `[~1h, secciones 1-4]`

**41.** 🔴 **Localizing Model Behavior with Path Patching**
Goldowsky-Dill et al. (2023)
Extiende activation patching a rutas completas de información. Permite responder "¿cómo fluye la información de A a B a través de C?". Más poderoso pero más complejo que activation patching simple.
`[arxiv.org/abs/2304.05969]` `[~1h, secciones 1-4]`

**42.** 🟡 **Causal Abstractions of Neural Networks**
Geiger et al. (2021)
Framework teórico basado en la teoría de causalidad de Judea Pearl, adaptado a redes neuronales. Más abstracto pero da el fundamento riguroso para hacer afirmaciones causales.
`[arxiv.org/abs/2106.02997]` `[~1h, secciones 1-3]`

---

### Nivel 5 — Sparse Autoencoders

| # | Paper | Prioridad | Cuándo leerlo |
|---|-------|-----------|---------------|
| 43 | Towards Monosemanticity (Dictionary Learning) | 🔴 | *Leer antes de cualquier implementación de SAE* |
| 44 | Scaling and Evaluating Sparse Autoencoders | 🔴 | Después del 43 |
| 45 | On the Biology of a Large Language Model | 🔴 | El estado del arte — Anthropic 2025 |
| 46 | Language Models Can Explain Neurons | 🟡 | Para automated interpretability |
| 47 | Scaling Monosemanticity (Claude 3 Sonnet) | 🟡 | Para ver SAEs a escala real |

---

**43.** 🔴 **Towards Monosemanticity: Decomposing Language Models With Dictionary Learning**
Bricken et al. (2023) — *Anthropic*
El paper que lanza los SAEs como herramienta central del campo. Entrena un SAE sobre una capa de un one-layer transformer y encuentra features individuales interpretables: "the", nombres en hebreo, código Python. Tiene código reproducible.
`[transformer-circuits.pub/2023/monosemanticity/index.html]` `[~3h, leer completo + explorar features]`

**44.** 🔴 **Scaling and Evaluating Sparse Autoencoders**
Templeton et al. (2024) — *Anthropic*
Escala los SAEs a Claude 3 Sonnet. Encuentra features para conceptos abstractos como "Inner Conflict" y "The Dark Side of Human Nature". Define métricas para evaluar SAEs. La sección de evaluación es la más importante.
`[anthropic.com/research/scaling-sparse-autoencoders]` `[~2h, leer completo]`

**45.** 🔴 **On the Biology of a Large Language Model**
Lindsey et al. (2025) — *Anthropic*
El paper más reciente y avanzado del campo. Introduce attribution graphs: mapas completos de cómo fluye la información en Claude para producir una respuesta específica. Estado del arte absoluto.
`[transformer-circuits.pub/2025/attribution-graphs/biology.html]` `[~3h, leer completo]`

**46.** 🟡 **Language Models Can Explain Neurons in Language Models**
Bills et al. (2023) — *OpenAI*
Método para usar GPT-4 para generar descripciones automáticas de los features descubiertos. Define un pipeline de automated interpretability con métricas de evaluación de las descripciones.
`[openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html]` `[~45 min, secciones 1-4]`

**47.** 🟡 **Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet**
Templeton et al. (2024) — *Anthropic*
Versión abreviada/blog del paper 44. Más accesible, con los hallazgos más sorprendentes destacados. Leer primero si el paper completo resulta denso.
`[anthropic.com/news/scaling-monosemanticity]` `[~45 min]`

---

## 📊 Resumen de carga de lectura

| Rama | Papers obligatorios 🔴 | Papers recomendados 🟡 | Total aprox. |
|------|----------------------|----------------------|--------------|
| Fundamentos | 13 | 7 | ~14h de lectura |
| Crítica | 7 | 3 | ~6h de lectura |
| Circuitos internos | 11 | 5 | ~22h de lectura |
| **Total** | **31** | **15** | **~42h** |

---

## 📅 Orden de lectura recomendado (secuencia lineal)

Si se quiere una secuencia que minimice el backtracking:

```
SHAP (1) → LIME (2) → Attention is All You Need (6) →
DistilBERT (7) → Feature Visualization (11) → Zoom In Circuits (12) →
Attention not Explanation (8) → Attention not not (9) →
GradCAM (16) → Integrated Gradients (17) → Sanity Checks (18) →
[PAUSA — terminar Fundamentos en código]
Alvarez-Melis fidelidad (21) → Explainer explicado (22) →
Disagreement Problem (25) → Fooling LIME and SHAP (26) →
Clever Hans (27) →
[PAUSA — terminar Crítica en código]
Toy Models of Superposition (30) → Mathematical Framework (37) →
Thread Circuits (33) → Induction Heads (34) → IOI (35) →
ROME (40) → Path Patching (41) →
Towards Monosemanticity (43) → Scaling SAEs (44) → Biology of LLM (45)
```

---

*Última actualización: abril 2025 — 47 papers catalogados*
