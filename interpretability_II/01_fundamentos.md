# 🎯 Guía de Preguntas - Fundamentos de Interpretabilidad en NLP

## Instrucciones
Responde cada pregunta con tus propias palabras. Puedes usar diagramas, ejemplos o código cuando sea útil.

---

## 📚 BLOQUE 1: Interpretabilidad en Machine Learning

### Conceptos Fundamentales

**1.1** ¿Qué es interpretabilidad en machine learning? ¿Por qué es importante?

**1.2** ¿Cuál es la diferencia entre interpretabilidad, explicabilidad y transparencia?

**1.3** Menciona 3 casos de uso reales donde la interpretabilidad es crítica (ej: medicina, finanzas, legal)

**1.4** ¿Qué es una "caja negra" en ML? ¿Todos los modelos son cajas negras?

---

### Tipos de Interpretabilidad

**1.5** ¿Qué es interpretabilidad **global** vs **local**? Da un ejemplo de cada una.

**1.6** ¿Qué son métodos **model-agnostic** vs **model-specific**? ¿Cuándo usar cada uno?

**1.7** ¿Qué es una explicación **post-hoc**? Contrasta con modelos inherentemente interpretables.

**1.8** ¿Feature importance tradicional es suficiente? ¿Qué limitaciones tiene?

---

### Trade-offs y Limitaciones

**1.9** ¿Existe un trade-off entre performance y interpretabilidad? Explica con ejemplos.

**1.10** ¿Qué riesgos hay en confiar ciegamente en las explicaciones de un modelo?

**1.11** ¿Cómo validamos que una explicación es "correcta" o "útil"?

---

## 🤖 BLOQUE 2: Arquitectura de Transformers

### Conceptos Básicos

**2.1** ¿Qué problema resuelven los Transformers que las RNN/LSTM no podían resolver bien?

**2.2** ¿Qué es el mecanismo de **atención** (attention)? Explica con un ejemplo de traducción.

**2.3** ¿Qué es **self-attention**? ¿En qué se diferencia de attention tradicional?

**2.4** Dibuja un diagrama simple de la arquitectura Transformer (encoder-decoder)

---

### Componentes Clave

**2.5** ¿Qué son los **positional embeddings**? ¿Por qué son necesarios?

**2.6** ¿Qué es **multi-head attention**? ¿Por qué usar múltiples "cabezas"?

**2.7** ¿Qué hace la capa **Feed-Forward** en cada bloque del Transformer?

**2.8** ¿Qué es **layer normalization** y por qué se usa?

---

### BERT y DistilBERT

**2.9** ¿BERT es encoder-only, decoder-only o encoder-decoder? ¿Por qué?

**2.10** ¿Qué es el token `[CLS]` y para qué se usa en clasificación?

**2.11** ¿Qué es **DistilBERT**? ¿Cómo se "destila" conocimiento de BERT?

**2.12** ¿Cuántas capas tiene DistilBERT vs BERT? ¿Cuánto performance se pierde?

**2.13** Para clasificación de sentimientos, ¿qué parte del modelo DistilBERT usamos?

---

### Fine-tuning

**2.14** ¿Qué significa hacer **fine-tuning** de un modelo pre-entrenado?

**2.15** ¿Qué capas se entrenan durante fine-tuning para clasificación?

**2.16** ¿Por qué usar un modelo pre-entrenado en vez de entrenar desde cero?

---

## 🔍 BLOQUE 3: SHAP (SHapley Additive exPlanations)

### Fundamentos Teóricos

**3.1** ¿Qué son los **valores de Shapley** en teoría de juegos?

**3.2** Explica la analogía de "coaliciones" en Shapley values aplicado a features de ML

**3.3** ¿Qué significa que SHAP tiene la propiedad de **eficiencia**? ¿Y **simetría**?

**3.4** ¿Qué es la propiedad **dummy** en SHAP? Da un ejemplo.

---

### SHAP en la Práctica

**3.5** ¿Qué es **KernelSHAP**? ¿Cómo aproxima los valores de Shapley?

**3.6** Para un modelo de texto, ¿qué representan las "features" en SHAP? (palabras, tokens, embeddings?)

**3.7** ¿Cómo se calculan SHAP values para un Transformer? ¿Qué estrategia de masking se usa?

**3.8** ¿SHAP values son siempre positivos? ¿Qué significa un valor negativo?

**3.9** ¿Cómo se interpreta un SHAP value de +0.3 para la palabra "excelente"?

---

### Visualizaciones SHAP

**3.10** ¿Qué muestra un **waterfall plot** de SHAP?

**3.11** ¿Qué muestra un **bar plot** de importancia global?

**3.12** ¿Qué diferencia hay entre analizar una instancia vs el dataset completo con SHAP?

---

### Ventajas y Limitaciones

**3.13** ¿Cuáles son las 3 principales ventajas de SHAP sobre otros métodos?

**3.14** ¿Cuál es la principal desventaja de SHAP? (pista: complejidad computacional)

**3.15** ¿SHAP funciona para cualquier tipo de modelo? (árboles, redes neuronales, etc.)

---

## 🍋 BLOQUE 4: LIME (Local Interpretable Model-agnostic Explanations)

### Fundamentos del Algoritmo

**4.1** ¿Qué significa que LIME es **model-agnostic**?

**4.2** ¿Qué significa que LIME genera explicaciones **locales**?

**4.3** Explica el algoritmo de LIME en 4 pasos:
   - Paso 1: ¿Qué hace con el input?
   - Paso 2: ¿Qué modelo entrena?
   - Paso 3: ¿Cómo pondera los samples?
   - Paso 4: ¿Qué retorna?

**4.4** ¿Por qué LIME usa un modelo lineal si el modelo original es complejo?

---

### LIME para Texto

**4.5** Para texto, ¿cómo "perturba" LIME una oración? Da un ejemplo concreto.

**4.6** Si la oración es "This movie is absolutely fantastic", ¿qué perturbaciones crearía LIME?

**4.7** ¿Cuántas perturbaciones genera LIME típicamente? ¿Más es siempre mejor?

**4.8** ¿Qué es el parámetro `num_features` en LIME? ¿Cómo elegir su valor?

---

### Interpretación de Resultados

**4.9** Si LIME dice que "terrible" tiene peso -0.45, ¿qué significa?

**4.10** ¿Las explicaciones de LIME son siempre consistentes para el mismo input?

**4.11** ¿Qué pasa si ejecuto LIME dos veces sobre el mismo texto? ¿Obtengo el mismo resultado?

---

### Ventajas y Limitaciones

**4.12** ¿Cuál es la principal ventaja de LIME sobre SHAP?

**4.13** ¿LIME garantiza alguna propiedad teórica (como SHAP)? ¿Por qué sí o no?

**4.14** ¿Qué significa que LIME es "inestable"? Da un ejemplo.

**4.15** ¿Cuándo preferirías LIME sobre SHAP?

---

## ⚖️ BLOQUE 5: SHAP vs LIME

### Comparación Conceptual

**5.1** Resume en una tabla las diferencias clave entre SHAP y LIME:
| Aspecto | SHAP | LIME |
|---------|------|------|
| Base teórica | ? | ? |
| Scope (global/local) | ? | ? |
| Velocidad | ? | ? |
| Garantías formales | ? | ? |

**5.2** ¿Cuándo usarías SHAP en vez de LIME?

**5.3** ¿Cuándo usarías LIME en vez de SHAP?

**5.4** ¿Pueden SHAP y LIME dar explicaciones contradictorias para el mismo input?

---

### Aplicación a NLP

**5.5** Para un modelo de análisis de sentimientos, ¿qué método sería mejor para:
   - Entender qué palabras son más importantes globalmente?
   - Explicar una predicción específica a un usuario?
   - Depurar por qué el modelo falla en ciertos casos?

**5.6** ¿Cómo medirías si SHAP o LIME da "mejores" explicaciones?

**5.7** Si SHAP y LIME identifican palabras diferentes como importantes, ¿a cuál le creerías? ¿Por qué?

---

## 🎯 BLOQUE 6: Métricas de Validación

### Evaluación de Explicaciones

**6.1** ¿Qué es **fidelidad** de una explicación? ¿Cómo se mide?

**6.2** ¿Qué es **estabilidad** de una explicación? Da un ejemplo de explicación inestable.

**6.3** ¿Qué es **comprensibilidad**? ¿Cómo la evaluarías?

**6.4** Propón un experimento para validar que una explicación es "correcta"

---

### Tests de Sanidad

**6.5** Si eliminas la palabra más importante según SHAP/LIME, ¿qué debería pasar con la predicción?

**6.6** Si duplicas una palabra importante, ¿cómo debería cambiar su SHAP value?

**6.7** ¿Cómo verificarías que SHAP/LIME no está dando explicaciones aleatorias?

---

## 🚀 BLOQUE 7: Aplicación al Proyecto

### Dataset y Modelo

**7.1** ¿Por qué usamos el dataset **IMDb** para este proyecto?

**7.2** ¿Qué características tiene el dataset IMDb? (tamaño, balance, longitud promedio)

**7.3** ¿Por qué elegimos **DistilBERT** en vez de BERT completo?

**7.4** ¿El modelo que usaremos está pre-entrenado o entrenaremos desde cero?

---

### Plan de Implementación

**7.5** Lista los 5 componentes principales que implementaremos:
   1. ?
   2. ?
   3. ?
   4. ?
   5. ?

**7.6** ¿Qué herramientas/librerías usaremos para:
   - Cargar el modelo?
   - Implementar SHAP?
   - Implementar LIME?
   - Visualizar resultados?

**7.7** ¿Qué visualizaciones queremos crear al final del proyecto?

---

### Preguntas de Investigación

**7.8** ¿Qué queremos aprender de este proyecto? Lista 3 preguntas de investigación.

**7.9** ¿Qué considerarías un "éxito" al finalizar este módulo?

**7.10** ¿Qué habilidades nuevas habrás adquirido después de completar este módulo?

---

## ✅ Checklist de Completitud

Marca cuando hayas respondido cada bloque:

- [ ] BLOQUE 1: Interpretabilidad en ML (11 preguntas)
- [ ] BLOQUE 2: Arquitectura Transformers (16 preguntas)
- [ ] BLOQUE 3: SHAP (15 preguntas)
- [ ] BLOQUE 4: LIME (15 preguntas)
- [ ] BLOQUE 5: SHAP vs LIME (7 preguntas)
- [ ] BLOQUE 6: Métricas de Validación (7 preguntas)
- [ ] BLOQUE 7: Aplicación al Proyecto (10 preguntas)

**Total: 81 preguntas**

---

## 📚 Recursos Recomendados

Para responder estas preguntas, consulta:

1. **Papers clave:**
   - Ribeiro et al. (2016) - "Why Should I Trust You?" (LIME)
   - Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions" (SHAP)
   - Vaswani et al. (2017) - "Attention is All You Need" (Transformers)

2. **Documentación:**
   - HuggingFace Transformers: https://huggingface.co/docs/transformers
   - SHAP documentation: https://shap.readthedocs.io
   - LIME documentation: https://lime-ml.readthedocs.io

3. **Tutoriales:**
   - Illustrated Transformer: http://jalammar.github.io/illustrated-transformer/
   - SHAP examples: https://github.com/slundberg/shap

---

## 🎯 Siguiente Paso

Una vez respondidas todas las preguntas:
1. Crea un documento `docs/01_fundamentos_respuestas.md` con tus respuestas
2. Revisa tus respuestas con los recursos mencionados
3. Identifica conceptos que necesitas reforzar
4. ¡Avanza a la Semana 2 (implementación toy)!