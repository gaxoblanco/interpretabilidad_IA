# Rama 3 — Circuitos internos
## Mechanistic interpretability

---

## ¿De qué trata esta rama?

Las ramas anteriores responden "¿qué features importaron para esta predicción?". Esta rama hace una pregunta diferente y más profunda: **¿qué operación matemática concreta está implementando el modelo para llegar a esa predicción?**

En lugar de observar inputs y outputs, mechanistic interpretability mira *adentro* del modelo con el objetivo de reconstruir el algoritmo que implementa. El trabajo no es solo visualizar — es entender la red con la misma precisión con la que se entiende un programa de computadora.

Este campo es relativamente nuevo (2020-2025) y está siendo liderado principalmente por investigadores de Anthropic (Chris Olah, Neel Nanda), DeepMind y grupos universitarios. Los resultados ya son sorprendentes: se han descubierto circuitos específicos en GPT-2 que implementan operaciones como "detectar nombres propios", "calcular la posición en una secuencia" o "buscar el sujeto de una oración".

**Por qué importa para el objetivo de "AI master":** esta rama es la frontera del campo. Entenderla — aunque sea parcialmente — es lo que distingue a alguien que aplica herramientas de alguien que entiende los modelos a nivel fundamental.

**Prerequisito:** Fundamentos completo (especialmente Módulo III de activaciones). Álgebra lineal y cálculo a nivel universitario. Familiaridad con la arquitectura interna de Transformers.

---

## Mapa de niveles

| Nivel | Tema central | Técnicas clave | Estado |
|-------|-------------|----------------|--------|
| 1 | Superposición y geometría de representaciones | Feature geometry, superposition hypothesis | ⬜ |
| 2 | Circuits — encontrar algoritmos en redes neuronales | Subgraph discovery, activation patching | ⬜ |
| 3 | Mechanistic interpretability en Transformers | Attention head analysis, MLP as key-value memory | ⬜ |
| 4 | Causalidad y ablación sistemática | Causal tracing, path patching | ⬜ |
| 5 | Fronteras del campo — Sparse Autoencoders y dictionary learning | SAEs, feature decomposition | ⬜ |

---

## Nivel 1 — Superposición y geometría de representaciones

**Pregunta guía:** ¿Cómo organiza una red neuronal la información en sus activaciones? ¿Hay más conceptos que dimensiones disponibles?

Este nivel introduce la idea de que las redes neuronales no representan features de forma simple (un feature = una neurona). La realidad es mucho más compleja y más interesante.

### Técnicas y conceptos

**Linear representation hypothesis**
La hipótesis central del campo: los modelos representan conceptos como *direcciones* en el espacio de activaciones. "Rey" y "reina" no son neuronas — son vectores en el embedding space. La operación famosa: `king - man + woman ≈ queen` es evidencia de esto.

**Superposition hypothesis**
Una red con N neuronas puede representar *más de N features* si los features son suficientemente raros (sparse). La red "superpone" múltiples features en el mismo espacio dimensional, usando interferencia controlada.
- Implicación crítica: el análisis neurona-por-neurona es insuficiente. Una sola neurona puede estar codificando múltiples conceptos a la vez.
- Toy model de superposición: red 2D que representa 5 features — el paper de Elhage et al. (2022) tiene una implementación en 50 líneas de código.

**Probing como herramienta de geometría**
Los probing classifiers del Módulo III adquieren un nuevo significado aquí: miden qué información está codificada linealmente en las representaciones. Si un clasificador lineal puede predecir "el sujeto de esta oración" a partir de las activaciones de la capa 4, hay una dirección lineal en esa capa que codifica el concepto de sujeto.

**PCA y visualización del espacio de representaciones**
Proyectar activaciones de múltiples instancias y analizar la estructura del espacio. ¿Se agrupan conceptos similares? ¿Hay estructura geométrica interpretable?

### Herramientas
- `torch` + `numpy` — cálculo de representaciones y análisis lineal
- `scikit-learn` — PCA, clasificadores lineales para probing
- `matplotlib` — visualización de espacios de alta dimensión proyectados

### Entregables
- [ ] Reproducir el toy model de superposición de Elhage et al. (red 2D, 5 features)
- [ ] Visualizar el espacio de embeddings de DistilBERT (del Módulo II) con PCA/t-SNE
- [ ] Probing classifiers sobre capas de DistilBERT: ¿qué información lingüística codifica cada capa?
- [ ] Análisis: ¿hay evidencia de superposición en los modelos ya entrenados?

### Recursos
- 📄 **Paper central:** Elhage et al. (2022) — *Toy Models of Superposition* — [transformer-circuits.pub/2022/toy_model/index.html](https://transformer-circuits.pub/2022/toy_model/index.html) *(lectura obligatoria)*
- 📄 **Linear representations:** Mikolov et al. (2013) — *Linguistic Regularities in Continuous Space Word Representations* — [aclanthology.org/N13-1090](https://aclanthology.org/N13-1090)
- 📄 **Geometry of representations:** Park et al. (2023) — *The Linear Representation Hypothesis and the Geometry of Large Language Models* — [arxiv.org/abs/2311.03658](https://arxiv.org/abs/2311.03658)
- 🎥 **Introducción accesible:** Neel Nanda — *A Mechanistic Interpretability Analysis of Grokking* (video) — YouTube

---

## Nivel 2 — Circuits: encontrar algoritmos en redes neuronales

**Pregunta guía:** ¿Se puede identificar un subgrafo específico de la red que implementa una operación concreta?

Un "circuit" es un subconjunto de neuronas y conexiones de la red que implementa un algoritmo específico. El objetivo es reconstruir ese algoritmo — no solo saber que existe, sino entender qué operación realiza.

### Técnicas y conceptos

**Circuit discovery**
Proceso sistemático para identificar qué parte de la red es responsable de un comportamiento específico:
1. Definir una tarea concreta y medible (ej: "completar el patrón A, B, C, ?")
2. Identificar las capas y neuronas más relevantes (ablación, patching)
3. Verificar que el subgrafo identificado reproduce el comportamiento en aislamiento

**Activation patching (Causal mediation analysis)**
Técnica central del campo. Pasos:
1. Correr el modelo en el input original → guardar todas las activaciones
2. Correr el modelo en un input alternativo (ej: misma oración con el sujeto cambiado)
3. Parchear las activaciones de componentes específicos del paso 2 en la forward pass del paso 1
4. Medir cuánto cambia el output → eso mide el efecto causal de ese componente

**Logit lens / Tuned lens**
Proyectar las activaciones de capas intermedias directamente al espacio de vocabulario para ver qué "piensa" el modelo en cada capa. Permite rastrear cómo se construye una predicción capa por capa.

**Casos de estudio históricos**
- **Induction heads** — heads de atención que implementan el patrón "si viste [A][B] antes, después de [A] predecí [B]". Mecanismo fundamental de in-context learning.
- **Indirect object identification** — circuito en GPT-2 que identifica el receptor de una acción en oraciones del tipo "John gave Mary the ball. He gave it to ___"

### Herramientas
- `transformer_lens` (Neel Nanda) — la herramienta estándar del campo para mechanistic interp en Transformers
- `baukit` — toolkit de activación y patching
- `torch` (hooks + gradientes para análisis manual)

### Entregables
- [ ] Instalar y familiarizarse con `transformer_lens` sobre GPT-2 small
- [ ] Reproducir la detección de induction heads (tutorial de Neel Nanda)
- [ ] Implementar activation patching sobre un comportamiento concreto de GPT-2
- [ ] Logit lens: visualizar cómo evoluciona la predicción capa por capa en 5 ejemplos
- [ ] Documentar un "circuit" simple identificado: qué componentes, qué operación implementan

### Recursos
- 📄 **Paper fundacional de circuits:** Cammarata et al. (2020) — *Thread: Circuits* — [distill.pub/2020/circuits](https://distill.pub/2020/circuits) *(lectura obligatoria)*
- 📄 **Induction heads:** Olsson et al. (2022) — *In-context Learning and Induction Heads* — [transformer-circuits.pub/2022/in-context-learning-and-induction-heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- 📄 **Indirect object identification:** Wang et al. (2022) — *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small* — [arxiv.org/abs/2211.00593](https://arxiv.org/abs/2211.00593)
- 📄 **Logit lens:** nostalgebraist (2020) — *interpreting GPT: the logit lens* — [lesswrong.com](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- 🔧 **TransformerLens:** [github.com/neelnanda-io/TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- 🎥 **Tutorial de Neel Nanda:** *Mechanistic Interpretability Tutorial* — YouTube / ARENA curriculum

---

## Nivel 3 — Mechanistic interpretability en Transformers

**Pregunta guía:** ¿Qué hace exactamente cada componente de un Transformer (attention heads, MLPs, layer norm)?

Este nivel analiza los componentes del Transformer no como cajas negras sino como operaciones matemáticas con roles interpretables.

### Técnicas y conceptos

**Attention heads como operaciones sobre el residual stream**
El residual stream es la representación que fluye a través del Transformer. Cada capa *lee* del stream, *procesa*, y *escribe* de vuelta. Analizar qué subespacio del stream lee y escribe cada head.
- Cada attention head tiene una "función" identificable: algunos copian tokens, algunos buscan nombre propio → verbo, algunos detectan posición.

**MLPs como memoria key-value**
Los MLPs en Transformers pueden interpretarse como bases de datos: las neuronas en la primera capa actúan como "keys" (patrones a matchear) y las neuronas en la segunda capa actúan como "values" (información a devolver si el key matchea).
- El paper de Geva et al. (2021) demuestra esto empíricamente.

**QK y OV circuits**
Descomponer el mecanismo de atención en dos circuitos separados:
- **QK circuit:** determina *a qué tokens prestar atención* (pattern of attention)
- **OV circuit:** determina *qué información mover* del token atendido al token query

**Superposición en MLPs**
Las neuronas individuales de MLP a menudo responden a múltiples conceptos no relacionados (polisemia neuronal). Un solo neurón puede activarse para "strings de código Python" Y "composiciones musicales en Do mayor".

### Herramientas
- `transformer_lens` — análisis de residual stream, decomposición de atención por componente
- `circuitsvis` — visualización de patterns de atención e intervenciones
- `plotly` — visualización de matrices QK y OV

### Entregables
- [ ] Analizar las attention heads de GPT-2 capa por capa: clasificar cada head según su función predominante
- [ ] Reproducir el análisis MLP como key-value memory sobre GPT-2
- [ ] Descomponer un head en sus circuitos QK y OV y documentar qué hace cada uno
- [ ] Identificar un neurón polisémico y documentar los conceptos que activan

### Recursos
- 📄 **Mathematical framework:** Elhage et al. (2021) — *A Mathematical Framework for Transformer Circuits* — [transformer-circuits.pub/2021/framework/index.html](https://transformer-circuits.pub/2021/framework/index.html) *(lectura obligatoria)*
- 📄 **MLPs como memoria:** Geva et al. (2021) — *Transformer Feed-Forward Layers Are Key-Value Memories* — [arxiv.org/abs/2012.14913](https://arxiv.org/abs/2012.14913)
- 📄 **Polisemia neuronal:** Elhage et al. (2022) — en el paper de superposición, sección sobre polysemanticity
- 🔧 **CircuitsVis:** [github.com/alan-cooney/CircuitsVis](https://github.com/alan-cooney/CircuitsVis)

---

## Nivel 4 — Causalidad y ablación sistemática

**Pregunta guía:** ¿Cómo probar que un componente identificado *causa* un comportamiento, en lugar de simplemente correlacionar con él?

La diferencia entre correlación y causalidad es crítica en interpretabilidad. Un componente puede estar activo cuando ocurre un comportamiento sin ser la causa de ese comportamiento.

### Técnicas y conceptos

**Causal tracing (ROME)**
Desarrollado en el contexto de edición de conocimiento factual en LLMs. Método:
1. Identificar qué capas contienen el conocimiento factual (ej: "La torre Eiffel está en ___")
2. Demostrar causalidad mediante restauración: corromper el input, luego restaurar las activaciones de una capa específica → ¿vuelve la predicción correcta?

**Path patching**
Extensión de activation patching que traza rutas causales completas a través del modelo. En lugar de parchear un componente, se parchean caminos entre componentes para entender el flujo de información.

**Interchange intervention**
Método formal para establecer causalidad en redes neuronales. Basado en la teoría de intervenciones de Judea Pearl adaptada a representaciones distribuidas.

**Knock-out sistemático a escala**
A diferencia del knockout del Módulo III (individual o grupal sobre pocas neuronas), aquí el objetivo es una ablación sistemática de *todos* los componentes para construir un mapa de causalidad completo.

### Herramientas
- `transformer_lens` (activation patching, path patching)
- `rome` / `memit` — implementaciones de causal tracing para edición de modelos
- `pyvene` — librería para intervenciones causales en representaciones

### Entregables
- [ ] Implementar causal tracing sobre un hecho factual en GPT-2 (ej: capital de un país)
- [ ] Reproducir un resultado de path patching del paper de Wang et al. (2022)
- [ ] Mapa de causalidad completo de un comportamiento simple: qué componentes están causalmente involucrados, en qué orden
- [ ] Comparar causal tracing con activation patching simple: ¿qué adiciona la perspectiva causal?

### Recursos
- 📄 **Causal tracing:** Meng et al. (2022) — *Locating and Editing Factual Associations in GPT* (ROME) — [arxiv.org/abs/2202.05262](https://arxiv.org/abs/2202.05262)
- 📄 **Path patching:** Goldowsky-Dill et al. (2023) — *Localizing Model Behavior with Path Patching* — [arxiv.org/abs/2304.05969](https://arxiv.org/abs/2304.05969)
- 📄 **Interchange intervention:** Geiger et al. (2021) — *Causal Abstractions of Neural Networks* — [arxiv.org/abs/2106.02997](https://arxiv.org/abs/2106.02997)
- 🔧 **pyvene:** [github.com/stanfordnlp/pyvene](https://github.com/stanfordnlp/pyvene)

---

## Nivel 5 — Sparse Autoencoders y dictionary learning

**Pregunta guía:** ¿Puede el modelo ser "traducido" a un espacio de features humano-interpretables de forma sistemática y escalable?

Este es el estado del arte actual (2024-2025). La idea central: los modelos operan en representaciones superpuestas que son difíciles de analizar directamente. Sparse Autoencoders (SAEs) aprenden a "decodificar" esas representaciones en features individuales, sparse y más interpretables.

### Técnicas y conceptos

**Sparse Autoencoders (SAEs)**
Un autoencoder con un bottleneck muy grande (mucho más ancho que la representación original) entrenado con regularización L1 (sparsity). El encoder aprende a mapear las activaciones del modelo a features individuales sparse. El decoder aprende a reconstruir las activaciones originales desde esos features.
- Resultado: un "diccionario" de miles o decenas de miles de features individuales, cada uno activándose para un concepto específico y raramente para otros.
- Escala: Anthropic entrenó SAEs con ~34 millones de features sobre Claude 3 Sonnet.

**Dictionary learning**
Framework más general del que SAEs son una instancia. El objetivo: encontrar una base de "átomos" (features) tal que cualquier activación del modelo pueda reconstruirse como una combinación sparse de esos átomos.

**Feature steering**
Una vez identificados features individuales con SAEs, se puede *intervenir* en el modelo amplificando o suprimiendo features específicos y observar el cambio en el comportamiento. Permite verificar que el feature identificado causa el comportamiento observado.

**Automated interpretability**
Usar LLMs para generar automáticamente descripciones de los features descubiertos. El proceso: para un feature F, mostrar los textos que más activan F a GPT-4 y pedirle que genere una descripción. Luego verificar la descripción con más ejemplos.

### Herramientas
- `sae_lens` (Joseph Bloom) — librería estándar para entrenar y analizar SAEs
- `transformer_lens` — para las activaciones del modelo base
- `neuronpedia` — plataforma web con SAEs pre-entrenados e interfaz de exploración

### Entregables
- [ ] Entrenar un SAE pequeño sobre las activaciones de una capa de GPT-2 small
- [ ] Explorar al menos 20 features descubiertos: ¿son interpretables?
- [ ] Implementar feature steering: amplificar un feature y documentar el cambio en el output
- [ ] Automated interpretability: usar un LLM para describir features automáticamente y evaluar calidad
- [ ] Análisis crítico: ¿qué no puede capturar un SAE? ¿Cuáles son sus limitaciones?

### Recursos
- 📄 **SAEs en Anthropic:** Templeton et al. (2024) — *Scaling and evaluating sparse autoencoders* — [anthropic.com/research/scaling-sparse-autoencoders](https://www.anthropic.com/research/scaling-sparse-autoencoders)
- 📄 **SAEs en Claude Sonnet:** Lindsey et al. (2025) — *On the Biology of a Large Language Model* — [transformer-circuits.pub/2025/attribution-graphs/biology.html](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- 📄 **Dictionary learning:** Bricken et al. (2023) — *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* — [transformer-circuits.pub/2023/monosemanticity](https://transformer-circuits.pub/2023/monosemanticity/index.html) *(lectura obligatoria)*
- 📄 **Automated interpretability:** Bills et al. (2023) — *Language models can explain neurons in language models* — [openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
- 🔧 **SAELens:** [github.com/jbloomaus/SAELens](https://github.com/jbloomaus/SAELens)
- 🌐 **Neuronpedia:** [neuronpedia.org](https://neuronpedia.org) — explorar SAEs pre-entrenados sin código

---

## Criterios de graduación de esta rama

Para considerar la Rama 3 completa:

- [ ] Los 5 niveles tienen entregables implementados y documentados
- [ ] Se reprodujo al menos un resultado de un paper del estado del arte (Nivel 2 o superior)
- [ ] Se entrenó y analizó un SAE propio (Nivel 5)
- [ ] Se identificó al menos un circuit completo con evidencia causal (Niveles 2-4)
- [ ] Existe un `LEARNINGS.md` que conecta los hallazgos de esta rama con los de Fundamentos: ¿cómo cambia la perspectiva sobre SHAP y GradCAM después de entender circuits?

---

## Contexto del campo (2025)

Esta rama está en plena ebullición. En los últimos 3 años el campo pasó de ser un nicho académico a ser una prioridad de seguridad en los principales labs de IA. Algunos hitos recientes:

- **2022** — Toy Models of Superposition (Anthropic) establece el framework teórico
- **2023** — Monosemanticity paper (Anthropic): SAEs con features interpretables a escala
- **2024** — Scaling SAEs a Claude 3 Sonnet: 34M features, con hallazgos sorprendentes (features para "Inner Conflict", "Slavery and Oppression", etc.)
- **2025** — Attribution graphs en Claude: mapa de cómo fluye la información para producir una respuesta específica

El campo se mueve rápido. Los papers más recientes de Anthropic en transformer-circuits.pub son la mejor fuente para estar al día.

---

*Última actualización: rama no iniciada — prerequisito: Fundamentos completo + Álgebra lineal*
