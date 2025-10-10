# 🎯 Guía de Preguntas - Fundamentos Módulo II (Solo Conceptos Nuevos)

## Instrucciones
Estas son las preguntas sobre conceptos **nuevos** que no vimos en Módulo I. Enfócate en lo que necesitas para NLP con Transformers.

---

## 🤖 BLOQUE 1: Arquitectura de Transformers (LO NUEVO)

### Conceptos Esenciales

**1.1** ¿Qué es el mecanismo de **self-attention**? Explica con un ejemplo simple de una oración.

**1.2** ¿Qué es **multi-head attention**? ¿Por qué usar 12 "cabezas" en vez de 1?

**1.3** Dibuja un diagrama simplificado: ¿Cómo fluye el texto desde input hasta predicción en DistilBERT?

**1.4** ¿Qué es el token `[CLS]` y por qué lo usamos para clasificación?

**1.5** ¿Qué son **positional embeddings**? ¿Por qué los Transformers los necesitan?

---

### BERT vs DistilBERT

**1.6** ¿Cuál es la diferencia principal entre BERT y DistilBERT? (capas, parámetros, velocidad)

**1.7** ¿Qué significa "destilar" conocimiento de un modelo grande a uno pequeño?

**1.8** Para nuestro proyecto, ¿por qué elegimos DistilBERT en vez de BERT completo?

---

### Fine-tuning (Específico para Clasificación)

**1.9** ¿Qué significa hacer **fine-tuning** de un modelo pre-entrenado?

**1.10** ¿Qué capa se agrega encima de DistilBERT para hacer clasificación de sentimientos?

**1.11** ¿Qué ventaja tiene usar un modelo pre-entrenado vs entrenar desde cero?

---

## 🔍 BLOQUE 2: SHAP para NLP (LO NUEVO)

### Diferencias con SHAP Tabular

**2.1** En Módulo I usamos SHAP con XGBoost (datos tabulares). ¿Qué cambia al aplicarlo a texto?

**2.2** Para un modelo de texto, ¿las "features" son palabras, tokens o algo más?

**2.3** ¿Cómo "enmascara" SHAP palabras en una oración para calcular importancia?

**2.4** ¿Qué estrategia de masking se usa con Transformers? (¿tokens en blanco, [MASK], eliminación?)

---

### SHAP con Transformers

**2.5** ¿Usaremos `TreeExplainer` (como en Módulo I) o `Explainer` genérico? ¿Por qué?

**2.6** ¿Qué significa un SHAP value de +0.5 para la palabra "excelente" en análisis de sentimientos?

**2.7** ¿Cómo agregamos SHAP values de múltiples ejemplos para obtener importancia global de palabras?

---

## 🍋 BLOQUE 3: LIME para Texto (LO NUEVO)

### Algoritmo de Perturbación

**3.1** Para texto, ¿cómo "perturba" LIME una oración? Da un ejemplo concreto.

Ejemplo: `"This movie is absolutely fantastic"`
¿Qué perturbaciones crearía LIME?

**3.2** ¿Por qué LIME usa un modelo lineal local si DistilBERT es súper complejo?

**3.3** ¿Cuántas perturbaciones genera LIME típicamente para una explicación? ¿Más es mejor?

---

### Configuración para Transformers

**3.4** ¿Qué es el parámetro `num_features` en LIME? ¿Cuánto usaremos (5, 10, 20)?

**3.5** ¿LIME da siempre las mismas explicaciones para el mismo input? ¿Por qué sí o no?

**3.6** ¿Qué diferencia hay entre aplicar LIME a un texto corto (1 línea) vs largo (párrafo)?

---

## ⚖️ BLOQUE 4: SHAP vs LIME en NLP

### Comparación Específica para Texto

**4.1** Resume en una tabla para tu proyecto:
| Aspecto | SHAP (Transformers) | LIME (Texto) |
|---------|---------------------|--------------|
| Velocidad típica | ? segundos | ? segundos |
| ¿Estable? | Sí/No | Sí/No |
| Mejor para... | ? | ? |

**4.2** Si SHAP dice "excelente" es importante (+0.5) pero LIME dice "fantastic" (+0.8), ¿cómo interpretas eso?

**4.3** ¿Cuándo preferirías SHAP sobre LIME en este proyecto de sentimientos?

**4.4** ¿Para qué usarás cada uno en tu dashboard final?

---

## 🎯 BLOQUE 5: Validación de Explicaciones (LO NUEVO)

### Métricas que Implementarás

**5.1** ¿Qué es **fidelidad** de una explicación? Describe cómo la medirás en código.

**5.2** Propón un test simple: Si elimino la palabra más importante según SHAP, ¿qué debería pasar?

**5.3** ¿Cómo medirías si SHAP y LIME "están de acuerdo" en un ejemplo?

**5.4** ¿Qué experimento harías para verificar que no están dando explicaciones aleatorias?

---

## 🚀 BLOQUE 6: Implementación Práctica

### Tu Stack Técnico

**6.1** ¿Qué librería usarás para cargar DistilBERT? (nombre exacto)

**6.2** ¿Qué dataset de HuggingFace usarás? ¿Cuántos ejemplos?

**6.3** Lista las 3 clases principales que implementarás:
   - `ModelLoader`: ¿Qué hace?
   - `SHAPAnalyzer`: ¿Qué hace?
   - `LIMEAnalyzer`: ¿Qué hace?

---

### Visualizaciones

**6.4** Describe 2 visualizaciones que crearás:
   - Para SHAP: ?
   - Para LIME: ?

**6.5** En tu dashboard Streamlit, ¿qué podrá hacer el usuario?

---

## ✅ Checklist de Completitud

- [ ] **BLOQUE 1:** Transformers (11 preguntas) - Arquitectura que no viste en Módulo I
- [ ] **BLOQUE 2:** SHAP para NLP (7 preguntas) - Diferencias con SHAP tabular
- [ ] **BLOQUE 3:** LIME para Texto (6 preguntas) - Método completamente nuevo
- [ ] **BLOQUE 4:** Comparación específica (4 preguntas) - Para tu proyecto
- [ ] **BLOQUE 5:** Validación (4 preguntas) - Métricas nuevas
- [ ] **BLOQUE 6:** Implementación (5 preguntas) - Tu código específico

**Total: 37 preguntas** (enfocadas en lo nuevo)

---

## 📚 Recursos Mínimos Necesarios

**Para Transformers:**
- Tutorial: "The Illustrated Transformer" (jalammar.github.io)
- Paper: "Attention is All You Need" (solo sección 3)
- Docs: HuggingFace DistilBERT model card

**Para SHAP/LIME en NLP:**
- SHAP docs: Sección "Text models"
- LIME docs: `LimeTextExplainer` API
- Tutorial: SHAP text examples en GitHub oficial

---

## 🎯 Siguiente Paso

1. Responde estas 37 preguntas (guarda en `docs/01_fundamentos_respuestas.md`)
2. Identifica las 3-5 preguntas más difíciles
3. Busca recursos específicos para esas
4. ¡Pasa a Semana 2 (ejemplo toy)!

---

## 💡 Tip: Estrategia de Estudio

**Prioridad ALTA (responde primero):**
- Bloque 1: Preguntas 1.1, 1.3, 1.4, 1.9
- Bloque 2: Preguntas 2.1, 2.3, 2.5
- Bloque 3: Preguntas 3.1, 3.2
- Bloque 6: Todas

**Prioridad MEDIA:**
- El resto de Bloques 2 y 3
- Bloque 4 completo

**Prioridad BAJA (puedes aprender haciendo):**
- Bloque 5 (aprenderás implementando)