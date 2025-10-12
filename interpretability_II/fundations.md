# 🎯 Guía de Preguntas - Fundamentos Módulo II (Solo Conceptos Nuevos)

## Instrucciones
Estas son las preguntas sobre conceptos **nuevos** que no vimos en Módulo I. Enfócate en lo que necesitas para NLP con Transformers.

---

## 🤖 BLOQUE 1: Arquitectura de Transformers (LO NUEVO)

### Conceptos Esenciales

**1.1** ¿Qué es el mecanismo de **self-attention**?
- Permite que una palabra "mire/compare" con otras en la misma oración. Con esto podemos darle un peso a cada palabra, igual que en interretavility_I las values tienen un "peso" que cuanto afectan para la decision final.

**1.2** ¿Qué es **multi-head attention**? ¿Por qué usar 12 "cabezas" en vez de 1?
- Utilizar multiples "cabezas" permite al modelo capturar diferentes tipos de relaciones entre palabras en paralelo. Cada cabeza puede enfocarse en distintos aspectos del contexto. Pero segun cada caso se puede usar mas o menos cabezas.

**1.3** ¿Cómo fluye el texto desde input hasta predicción en un Transformer encoder para clasificación?
INPUT TEXT
    ↓
[TOKENIZER] → [CLS] word1 word2 ... [SEP]
    ↓
[EMBEDDINGS] → Vectores + Posiciones
    ↓
[LAYER 1] → Multi-Head Attention + FFN
    ↓
[LAYER 2] → Multi-Head Attention + FFN
    ↓
    ...
    ↓
[LAYER N] → Multi-Head Attention + FFN
    ↓
[EXTRACT [CLS]] → Vector de 768 dims
    ↓
[LINEAR + SOFTMAX] → [prob_clase1, prob_clase2, ...]
    ↓
PREDICTION

[CLS] vector (768) → Linear → [2.3, -1.8] → Softmax → [0.92, 0.08]
                                                         ↑      ↑
                                                      Positivo Negativo

**1.4** ¿Qué es el token `[CLS]` y por qué lo usamos para clasificación?
- Seria el puntero para marcar el inicio de la oracion, y es el que se usa para clasificacion porque contiene informacion de toda la oracion. Utiliza por dentro self-attention para "mirar" todas las palabras y resumir su significado en un solo vector.

**1.41** ¿Qué es el token `[SEP]` y por qué lo usamos?
- Se usa para marcar el final de una oracion o separar inputs en tareas de pares de oraciones (ej. pregunta-respuesta). y asi facilitar que el modelo entienda los limites del texto.

**1.5** ¿Qué son **positional embeddings**? ¿Por qué los Transformers los necesitan?
- Como trabajamos en paralelo, necesitamos contar con la posicion de cada palabra en la oracion. Los positional embeddings son vectores que se suman a los word embeddings para darle al modelo informacion sobre el orden de las palabras.
   # Pseudocódigo simplificado
   tokens = ["[CLS]", "This", "movie", "is", "great", "[SEP]"]

   # Token IDs
   token_ids = [101, 2023, 3185, 2003, 2307, 102]

   # Token embeddings (768 dims cada uno)
   token_embeds = embedding_layer(token_ids)  # Shape: [6, 768]

   # Positional embeddings (una por posición)
   positions = [0, 1, 2, 3, 4, 5]
   pos_embeds = position_embedding_layer(positions)  # Shape: [6, 768]

   # Sumar ambos
   final_embeds = token_embeds + pos_embeds  # Shape: [6, 768]
---

### BERT vs DistilBERT

**1.6** ¿Cuál es la diferencia principal entre BERT y DistilBERT? (capas, parámetros, velocidad)

**1.7** ¿Qué significa "destilar" conocimiento de un modelo grande a uno pequeño?

**1.8** Para nuestro proyecto, ¿por qué elegimos DistilBERT en vez de BERT completo?

---

### Fine-tuning (Específico para Clasificación)

**1.9** ¿Qué significa hacer **fine-tuning** de un modelo pre-entrenado?
- Pre-entrenamiento: Aprende lenguaje (semanas, millones de textos)
- Fine-tuning: Aprende tu tarea específica (horas, miles de ejemplos)

**1.10** ¿Qué capa se agrega encima de DistilBERT para hacer clasificación de sentimientos?

**1.11** ¿Qué ventaja tiene usar un modelo pre-entrenado vs entrenar desde cero?

---

## 🔍 BLOQUE 2: SHAP para NLP (LO NUEVO)

### Diferencias con SHAP Tabular

**2.1** En Módulo I usamos SHAP con XGBoost (datos tabulares). ¿Qué cambia al aplicarlo a texto?
- En texto, las "features" son palabras o tokens, no columnas numéricas. Al tener N° variables (palabras), el espacio de combinaciones es mucho mayor. Usamos token masking en vez de valores nulos. Y el modelo es un Transformer, no un árbol. Por esto ultimo usamos SHAP Explainer genérico, no TreeExplainer.
**2.2** Para un modelo de texto, ¿las "features" son palabras, tokens o algo más?
- Son tokens, que pueden ser palabras completas o sub-palabras (subwords) dependiendo del tokenizador usado. Por ejemplo, "fantástico" puede dividirse en "fan", "tás" y "tico".

**2.3** ¿Cómo "enmascara" SHAP palabras en una oración para calcular importancia?
- Reemplaza palabras con un token especial (ej. `[MASK]`) o las elimina, y observa cómo cambia la predicción del modelo. Esto ayuda a medir la contribución de cada palabra.

**2.4** ¿Qué estrategia de masking se usa con Transformers? (¿tokens en blanco, [MASK], eliminación?)
- Se usa el token `[MASK]` para reemplazar palabras, ya que los Transformers están entrenados para manejar este token y pueden inferir el contexto faltante.

**2.4.1** ¿Que tipos de Tokenizers existen y cuál usaremos? (word-level, subword, byte-level)
- Word-level: Cada palabra es una feature. Problemas con OOV y vocab grande.
- Subword: Divide palabras en partes (WordPiece, BPE). Balance entre vocab y OOV. Usaremos WordPiece.
- Byte-level: Cada byte es una feature. Resuelve OOV y multilingüismo, pero es lento y complejo.
- subword (WordPiece) es la mejor opción para nuestro proyecto.

- El tema de tokenización es crucial para NLP. Cuanto mas se optimice este paso, mejores resultados se obtendrán en las tareas posteriores a la hora de interpretar los modelos.

---

### SHAP con Transformers

**2.5** ¿Usaremos `TreeExplainer` (como en Módulo I) o `Explainer` genérico? ¿Por qué?
- Usaremos `Explainer` genérico porque los Transformers no son modelos basados en árboles. `TreeExplainer` está optimizado para modelos como XGBoost o Random Forest, pero no funciona bien con redes neuronales complejas como DistilBERT.

**2.6** ¿Qué significa un SHAP value de +0.5 para la palabra "excelente" en análisis de sentimientos?
- Significa que la palabra "excelente" contribuye positivamente a la predicción de sentimiento positivo, aumentando la probabilidad de esa clase en 0.5 unidades en la escala log-odds del modelo.

**2.7** ¿Cómo agregamos SHAP values de múltiples ejemplos para obtener importancia global de palabras?
- Calculamos el valor absoluto promedio de SHAP para cada palabra a lo largo de todos los ejemplos. Esto nos da una medida de la importancia global de cada palabra en el conjunto de datos.

---

## 🍋 BLOQUE 3: LIME para Texto (LO NUEVO)

### Algoritmo de Perturbación

**3.1** Para texto, ¿cómo "perturba" LIME una oración? Da un ejemplo concreto.

Ejemplo: `"This movie is absolutely fantastic"`
¿Qué perturbaciones crearía LIME?
- `"This movie is absolutely [MASK]"`
- Para generar la perturbación, LIME enmascara palabras al azar en la oración original.

**3.2** ¿Por qué LIME usa un modelo lineal local si DistilBERT es súper complejo?
- Porque LIME asume que cerca del punto de interés (la oración original), el comportamiento del modelo puede aproximarse linealmente. Esto simplifica la interpretación y permite entender qué palabras son más importantes para esa predicción específica.

**3.3** ¿Cuántas perturbaciones genera LIME típicamente para una explicación? ¿Más es mejor?
- Usualmente entre 500 y 1000 perturbaciones. Más perturbaciones pueden mejorar la estabilidad de la explicación, pero también aumentan el tiempo de cómputo. Hay un punto de rendimientos decrecientes donde agregar más perturbaciones no mejora significativamente la explicación.
- Opciones: Feature Subtitution, Random Deletion, Synonym Replacement.
---

### Configuración para Transformers

**3.4** ¿Qué es el parámetro `num_features` en LIME? ¿Cuánto usaremos (5, 10, 20)?
- `num_features` define cuántas palabras (features) se incluirán en la explicación final. Usaremos 10 para equilibrar detalle y claridad, ya que demasiadas pueden hacer la explicación confusa.

**3.5** ¿LIME da siempre las mismas explicaciones para el mismo input? ¿Por qué sí o no?
- No, LIME puede dar explicaciones ligeramente diferentes en cada ejecución debido a la naturaleza aleatoria de las perturbaciones que genera. Sin embargo, con un número suficiente de perturbaciones, las explicaciones tienden a ser consistentes.
- Se analiza la estabilidad de las explicaciones ejecutando LIME varias veces y comparando los resultados.

**3.6** ¿Qué diferencia hay entre aplicar LIME a un texto corto (1 línea) vs largo (párrafo)?
- En textos cortos, cada palabra tiene un impacto más significativo en la predicción, por lo que las explicaciones pueden ser más claras y directas. En textos largos, la importancia de cada palabra puede diluirse, y LIME puede identificar más palabras como relevantes, lo que puede complicar la interpretación.
- Ejemplo, una palabra que se repite en un párrafo largo puede tener una importancia acumulada mayor que en una oración corta, y no ncesariamente debería ser la más relevante para la predicción.
---

## ⚖️ BLOQUE 4: SHAP vs LIME en NLP

### Comparación Específica para Texto

**4.1** Resume en una tabla para tu proyecto:
| Aspecto | SHAP (Transformers) | LIME (Texto) |
|---------|---------------------|--------------|
| Velocidad típica | ? segundos | ? segundos |
| Aspecto | SHAP (Transformers) | LIME (Texto) |
|---------|---------------------|--------------|
| **Base teórica** | Teoría de juegos (Shapley values) | Aproximación lineal local |
| **Scope** | Global + Local | Solo Local |
| **Perturbación** | Masking `[MASK]` | Removal (eliminar palabras) |
| **Num. perturbaciones** | Todas las coaliciones (aprox.) | 5,000 - 10,000 muestras |
| **Velocidad** | 30-60 seg/texto | 45-90 seg/texto (5k samples) |
| **Estabilidad** | ✅ Alta (determinístico) | 🟡 Media (estocástico) |
| **Garantías matemáticas** | ✅ Sí (propiedades formales) | ❌ No |
| **Out-of-distribution** | 🟡 Moderado (BERT conoce [MASK]) | ❌ Alto (oraciones incompletas) |
| **Mejor para** | Importancia global + análisis riguroso | Explicación rápida individual |
| **Limitación** | Muy lento | Inestable, sin garantías |

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