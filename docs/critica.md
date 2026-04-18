# Rama 2 — Crítica
## Validación y meta-análisis de explicaciones

---

## ¿De qué trata esta rama?

Saber *usar* SHAP o GradCAM no es suficiente. La pregunta más importante es: **¿cuándo podemos confiar en una explicación?**

Las herramientas de interpretabilidad pueden dar respuestas convincentes pero incorrectas. Una saliency map puede resaltar regiones irrelevantes con perfecta seguridad. SHAP puede asignar importancia alta a features que el modelo aprendió a usar por correlación espuria, no por causalidad. Dos métodos aplicados al mismo modelo y la misma instancia pueden contradecirse completamente.

Esta rama desarrolla el criterio crítico para evaluar, comparar y rechazar explicaciones cuando no son confiables. Es la diferencia entre alguien que *aplica* herramientas de interpretabilidad y alguien que *entiende* lo que está midiendo.

**Prerequisito:** haber completado al menos los Módulos I y II de Fundamentos con experiencia práctica en SHAP y LIME.

---

## Mapa de niveles

| Nivel | Tema central | Técnicas clave | Estado |
|-------|-------------|----------------|--------|
| 1 | Métricas de evaluación de explicaciones | Fidelidad, estabilidad, comprensibilidad | ⬜ |
| 2 | Sanity checks y tests de adversarial robustness | Randomization tests, input invariance | ⬜ |
| 3 | Análisis comparativo y diagnóstico de divergencias | SHAP vs LIME vs Attention — cuándo divergen y por qué | ⬜ |
| 4 | Sesgos, correlaciones espurias y explicaciones engañosas | Spurious correlations, Clever Hans phenomenon | ⬜ |

---

## Nivel 1 — Métricas para evaluar explicaciones

**Pregunta guía:** ¿Cómo medimos cuán buena es una explicación sin depender del juicio humano?

Una explicación no puede evaluarse solo visualmente. Se necesitan métricas cuantitativas que midan si la explicación captura el comportamiento real del modelo.

### Técnicas y conceptos

**Fidelidad (Faithfulness)**
Mide si la explicación refleja realmente lo que el modelo hace. La prueba concreta: si eliminamos los features/tokens/píxeles que la explicación marca como importantes, ¿la predicción cambia significativamente?
- *Deletion curve:* quitar features por orden de importancia, medir caída de accuracy.
- *Insertion curve:* agregar features por orden de importancia desde una baseline vacía, medir subida de accuracy.
- Un método con alta fidelidad tiene una deletion curve que cae rápido y una insertion curve que sube rápido.

**Estabilidad (Stability / Consistency)**
Instancias similares deberían tener explicaciones similares. Si dos reviews casi idénticos reciben explicaciones completamente distintas, el método no es confiable.
- Medir con perturbaciones pequeñas: ¿cómo cambia la explicación si se agrega ruido mínimo al input?
- *Lipschitz continuity* como criterio formal.

**Comprensibilidad (Comprehensibility)**
¿Cuántos features necesita incluir una explicación para ser útil? Menos es mejor, pero con un umbral de fidelidad mínima.
- Medir como el número mínimo de features necesarios para alcanzar X% de fidelidad.

**Complejidad (Complexity)**
Relacionado con comprensibilidad pero desde el lado del modelo sustituto (en LIME): ¿qué tan complejo es el modelo local que aproxima al original?

### Herramientas
- `quantus` — librería especializada en métricas de evaluación de explicaciones (Python)
- `captum` — tiene métricas de fidelidad integradas
- Implementación custom de deletion/insertion curves con cualquier framework

### Entregables
- [ ] Implementar deletion curve y insertion curve para SHAP y LIME sobre el mismo modelo y dataset
- [ ] Implementar stability metric: distribución de cambios en explicaciones ante perturbaciones pequeñas
- [ ] Tabla comparativa: SHAP vs LIME en fidelidad y estabilidad sobre los modelos de Fundamentos
- [ ] Notebook con visualizaciones de las curvas y análisis de resultados

### Recursos
- 📄 **Definición formal de fidelidad:** Alvarez-Melis & Jaakkola (2018) — *Towards Robust Interpretability with Self-Explaining Neural Networks* — [arxiv.org/abs/1806.07538](https://arxiv.org/abs/1806.07538)
- 📄 **Deletion/Insertion curves:** Petsiuk et al. (2018) — *RISE: Randomized Input Sampling for Explanation of Black-box Models* — [arxiv.org/abs/1806.07421](https://arxiv.org/abs/1806.07421)
- 📄 **Estabilidad de LIME:** Garreau & von Luxburg (2020) — *Explaining the Explainer: A First Theoretical Analysis of LIME* — [arxiv.org/abs/2001.03447](https://arxiv.org/abs/2001.03447)
- 🔧 **Quantus:** [github.com/understandingai/quantus](https://github.com/understandingai/quantus)

---

## Nivel 2 — Sanity checks

**Pregunta guía:** ¿La explicación captura algo real sobre el modelo, o es una ilusión visual convincente?

Este es el nivel más importante y el más frecuentemente omitido. Antes de confiar en cualquier explicación, hay que demostrar que no pasaría el mismo test con un modelo aleatorio.

### Técnicas y conceptos

**Model parameter randomization test**
Reemplazar los pesos del modelo con valores aleatorios y generar las explicaciones de nuevo. Si los saliency maps o valores SHAP *se ven igual* con pesos aleatorios, la explicación no está capturando el comportamiento del modelo — está capturando algo del input o del método en sí.
- Referencia directa: Adebayo et al. (2018) demostraron que Guided Backprop y Guided GradCAM fallan este test. Son insensibles a los pesos del modelo.

**Data randomization test**
Entrenar el modelo con labels aleatorizados y generar explicaciones. Un método que supera este test debería producir explicaciones distintas entre el modelo real y el modelo con labels al azar.

**Input invariance test**
Agregar una constante a todos los pixels de la imagen (shift uniforme) no debería cambiar la saliency map — el modelo no debería ser sensible a ese shift. Métodos que fallan este test tienen problemas fundamentales de atribución.

**Cascading randomization**
Aleatorizar capa por capa desde la última hasta la primera y observar cómo cambian las explicaciones. Permite identificar a qué capas es sensible el método.

### Entregables
- [ ] Implementar model parameter randomization test sobre GradCAM e Integrated Gradients
- [ ] Implementar data randomization test para SHAP (tabular y NLP)
- [ ] Documentar qué métodos de Fundamentos pasan y cuáles fallan los tests
- [ ] Notebook con comparación visual de explicaciones reales vs aleatorias
- [ ] `LEARNINGS.md` con conclusiones sobre qué métodos son confiables y bajo qué condiciones

### Recursos
- 📄 **El paper de sanity checks:** Adebayo et al. (2018) — *Sanity Checks for Saliency Maps* — [arxiv.org/abs/1810.03292](https://arxiv.org/abs/1810.03292) *(lectura obligatoria)*
- 📄 **Input invariance:** Kindermans et al. (2019) — *The (Un)reliability of Saliency Methods* — [arxiv.org/abs/1711.00867](https://arxiv.org/abs/1711.00867)
- 📄 **Cascading randomization:** en el mismo paper de Adebayo et al.

---

## Nivel 3 — Diagnóstico de divergencias

**Pregunta guía:** Cuando SHAP, LIME y Attention weights se contradicen, ¿cuál tiene razón? ¿Por qué divergen?

### Técnicas y conceptos

**Análisis sistemático de casos de divergencia**
Identificar instancias donde dos métodos asignan importancias opuestas al mismo token/feature. Analizar:
- ¿Es el input ambiguo?
- ¿Hay features correlacionados que se "reparten" la importancia?
- ¿El modelo sustituto de LIME es una mala aproximación local?
- ¿La linearización de SHAP pierde información no lineal?

**Correlación entre métodos como métrica de confianza**
Si SHAP y LIME acuerdan, hay más evidencia de que la explicación es correcta. Si divergen, ambas son sospechosas. Medir correlación (Kendall tau, Spearman) entre rankings de importancia de distintos métodos.

**Análisis de estabilidad local de LIME**
LIME es un método local — su explicación depende del vecindario que define alrededor del punto. Variar el radio del vecindario y medir cuánto cambia la explicación. Alta variación = la explicación es frágil.

**Casos de estudio documentados**
Construir un catálogo de instancias donde los métodos divergen con análisis de por qué.

### Entregables
- [ ] Script para detectar automáticamente instancias de alta divergencia entre métodos
- [ ] Análisis de al menos 10 casos de divergencia documentados con hipótesis de causa
- [ ] Métrica de acuerdo inter-método implementada (Kendall tau entre rankings SHAP y LIME)
- [ ] Notebook de análisis de estabilidad de LIME en función del parámetro de vecindario

### Recursos
- 📄 **Divergencias entre métodos:** Krishna et al. (2022) — *The Disagreement Problem in Explainable Machine Learning* — [arxiv.org/abs/2202.01602](https://arxiv.org/abs/2202.01602)
- 📄 **Limitaciones de LIME:** Slack et al. (2020) — *Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods* — [arxiv.org/abs/1911.02508](https://arxiv.org/abs/1911.02508)
- 📖 **Capítulo de comparación:** Molnar — *Interpretable ML Book* — capítulo de Evaluation of Interpretability

---

## Nivel 4 — Sesgos y explicaciones engañosas

**Pregunta guía:** ¿Puede el modelo estar usando razones "incorrectas" para predecir correctamente? ¿Puede una explicación ocultar ese problema?

Este es el nivel más crítico. Un modelo puede tener alta accuracy por razones completamente equivocadas (correlaciones espurias), y sus explicaciones pueden verse perfectamente razonables.

### Técnicas y conceptos

**Clever Hans phenomenon**
El nombre viene de un caballo que "sabía matemáticas" pero en realidad leía las señales corporales de su dueño. En ML: el modelo aprendió un atajo no causal. Ejemplo clásico: un clasificador de imágenes de lobos que en realidad detecta nieve de fondo.
- Detectar: analizar qué features son más importantes globalmente y verificar si tienen sentido causal.

**Spurious correlations**
Features que predicen la label en el dataset de entrenamiento pero no por causalidad real. Ej: en un dataset de análisis de sentimiento, reviews largas pueden ser más frecuentemente positivas — el modelo aprende longitud como proxy.
- Medir con datasets de test contrafácticos (contradict the spurious correlation).

**Adversarial attacks a explicaciones**
Slack et al. (2020) demostraron que se puede entrenar un modelo que da predicciones normales pero cuando LIME o SHAP lo interrogan, devuelve un modelo sustituto diferente al real. El modelo engaña al método de explicación.
- Implicación: las explicaciones pueden ser manipuladas deliberadamente.

**Right for Wrong Reasons (RFWR)**
El modelo predice correctamente pero por features incorrectos. Las explicaciones pueden confirmar el resultado sin detectar el problema. Solo los sanity checks y los tests causales pueden revelarlo.

### Entregables
- [ ] Reproducir un caso de Clever Hans en un dataset controlado (introducir correlación espuria artificialmente)
- [ ] Demostrar que SHAP/LIME "no detectan" la correlación espuria en ese caso
- [ ] Implementar un contradict-test: evaluar el modelo en instancias donde la correlación espuria se rompe
- [ ] Documento de análisis: ¿cuáles de los modelos entrenados en Fundamentos podrían tener RFWR?

### Recursos
- 📄 **Clever Hans en ML:** Lapuschkin et al. (2019) — *Unmasking Clever Hans Predictors and Assessing What Machines Really Learn* — [arxiv.org/abs/1902.10178](https://arxiv.org/abs/1902.10178)
- 📄 **Atacar LIME y SHAP:** Slack et al. (2020) — *Fooling LIME and SHAP* — [arxiv.org/abs/1911.02508](https://arxiv.org/abs/1911.02508)
- 📄 **Right for Wrong Reasons:** Ross et al. (2017) — *Right for the Right Reasons: Training Differentiably through a Tree* — [arxiv.org/abs/1703.03717](https://arxiv.org/abs/1703.03717)
- 📄 **Spurious correlations en NLP:** Gururangan et al. (2018) — *Annotation Artifacts in Natural Language Inference Data* — [arxiv.org/abs/1803.02324](https://arxiv.org/abs/1803.02324)

---

## Criterios de graduación de esta rama

Para considerar la Rama 2 completa:

- [ ] Los 4 niveles tienen sus entregables implementados
- [ ] Se aplicaron sanity checks a los modelos de Fundamentos y se documentaron resultados
- [ ] Se construyó al menos un caso reproducible de explicación engañosa
- [ ] Existe un documento de conclusiones: ¿qué métodos de Fundamentos son confiables bajo qué condiciones?
- [ ] El conocimiento de esta rama modifica retrospectivamente algún LEARNINGS.md de Fundamentos

---

*Última actualización: rama no iniciada — prerequisito: Fundamentos Módulos I y II completados*
