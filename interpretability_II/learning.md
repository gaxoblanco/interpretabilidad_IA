
## Datos vs Modelo
- Para poder analizar un modelo, o conjunto de datos no basta con esperar que todo cuadre.
- Debemos conocer los valores que acepta y devuelve el modelo, tanto en inputs como variables para la evaluación.
- En este punto debemos cuestionarnos, el dataset que tenemos es para evaludar un modelo o para evaluar al dataset? suena raro, pero es importante.
- Según la naturaleza del objetivo podremos adaptar el dataset al modelo o el modelo al dataset.

## Futuro de Tokenización

- Modelos byte-level (ByT5, CANINE) resuelven multilingüismo y OOV
- Limitación principal: 3-5x más lentos sin mejoras claras en accuracy
- Para este proyecto, subword (WordPiece) sigue siendo la mejor opción
- Byte-level relevante para: idiomas raros, texto ruidoso, code-switching

## Validación de Explicaciones

- Fidelidad: mide qué tan bien una explicación refleja el comportamiento del modelo
- Test de eliminación: eliminar la palabra más importante según SHAP debería reducir la confianza en la predicción. Pero no esta simple. Se debe definir "eliminar" y "palabra" correctamente.
- Acuerdo entre métodos: usar correlación de Spearman para comparar rankings de importancia entre SHAP y LIME
- Al buscar experimentar con los cambios de palabras fue donde mas variables a considerar encontre. Llegado al punto de porque un #. tiene mas valor en SHAP que una Terrible. Y ocurre que segun el tokenizador, podrias obtener tokens que por si solo no tienen sentido, pero en conjunto si, ocacionando que un #. le gane a #Ter o #rible. 
- Este proble del tokenizador deriba en que uno debe definir en que nivel se va a trabajar, si en tokens, palabras, oraciones, etc. y como se va a eliminar una palabra, si reemplazandola por un token especial o por una palabra aleatorea.
- Resumen: 
    - Define tokenización y eliminación claramente
    - Nivel de análisis: tokens, palabras, oraciones
    - Usa métricas cuantitativas y cualitativas

## Dashboard Interactivo
- No tengo idea de como luce uno similar, pero lo arme lo mas simple posible y amigable para pruebas "rapidas".
- Como trabajo con texto, ingresar texto manualmente es clave.
- Permitir seleccionar modelos abre la puerta a comprender mucho mejor las explicaciones.
- Comparar SHAP y LIME visualmente ayuda a entender diferencias y similitudes.
- Validación de explicaciones integrada permite evaluar calidad de explicaciones en tiempo real.


## 1. Los modelos binarios ocultan información crítica

**Problema descubierto:**
Los modelos de sentimiento binario (positivo/negativo) dificultan el análisis 
porque colapsan información compleja en una sola dimensión.

**Hallazgo inesperado:**
SHAP mostró que la presencia de un punto final (.) correlaciona fuertemente 
con predicciones positivas. Esto sugiere que el modelo aprendió patterns 
sintácticos en lugar de semánticos.

**Implicación práctica:**
Si tu objetivo es interpretar sentimientos, considera modelos multi-clase 
(joy, anger, sadness) que exponen más información sobre el razonamiento interno.

---

## 2. La tokenización rompe la interpretabilidad humana

**El problema:**
- SHAP devuelve importancia por **token**, no por **palabra**
- "Terrible" → ["Ter", "##rible"] 
- Cada fragmento tiene SHAP value bajo individualmente
- Enmascarar "Ter" o "##rible" por separado → poco impacto en la predicción

**¿Por qué importa?**
La pregunta "¿cuál es la palabra más importante?" se vuelve ambigua:
- ¿Es la palabra completa "Terrible"?
- ¿O el token "Ter" que técnicamente tiene mayor SHAP value?

**Solución implementada:**
Agregué una función para re-agregar tokens en palabras completas antes de 
visualizar, sumando sus SHAP values.

---

## 3. Definir "eliminar" cambia toda la interpretación

**Pregunta clave que NO me hice al inicio:**
Cuando testeo importancia, ¿qué significa "eliminar" una palabra?

**Opciones encontradas:**
1. Reemplazar con token [MASK]
2. Reemplazar con palabra aleatoria
3. Simplemente borrar (cambia estructura sintáctica)
4. Reemplazar con sinónimo neutral

**Resultado:**
Cada método da explicaciones diferentes. No hay "correcta", depende de 
qué quieres medir:
- [MASK] → "¿Qué pasa si el modelo no ve esta info?"
- Aleatoria → "¿Esta palabra aporta más que ruido?"
- Borrar → "¿La posición/estructura importa?"

**Mi decisión:** Usé [MASK] porque quería medir "ausencia de información".

---

## 4. LIME vs SHAP: No comparan manzanas con manzanas

**El problema sutil:**
Al comparar resultados, descubrí que usan escalas opuestas:

| Método | "Terrible" en review negativa | Interpretación |
|--------|-------------------------------|----------------|
| SHAP   | +20.1464                      | Contribuye +20 a que sea NEGATIVA |
| LIME   | -0.0969                       | Contribuye -0.09 al sentimiento (← hacia negativo) |

**¿Por qué confunde?**
- SHAP: valores positivos = hacia la clase predicha (en este caso, NEGATIVA)
- LIME: valores negativos = hacia sentimiento negativo

**Solución:**
Normalicé las visualizaciones para que ambos usen la misma convención:
- Verde = palabras que empujan hacia POSITIVO
- Rojo = palabras que empujan hacia NEGATIVO

Sin este ajuste, las visualizaciones lado a lado son engañosas.

---

## 5. LIME sobresale en simplicidad, SHAP en modelos complejos

**Observación:**
- **LIME:** Rápido, devuelve palabras completas, fácil de explicar a no-técnicos
- **SHAP:** Lento, devuelve tokens, pero más consistente en modelos complejos

**Cuándo usar cada uno:**
- LIME → Prototipado rápido, explicaciones para stakeholders, modelos simples
- SHAP → Análisis profundo, validación científica, modelos multi-clase

**Trade-off descubierto:**
LIME empieza a fallar cuando el modelo tiene muchas capas. Ahí SHAP es más confiable, 
pero paga el costo en tiempo de cómputo.

---

## 6. Local vs Global: Define tu nivel de análisis

**4 niveles de interpretabilidad posibles:**

| Nivel | Pregunta que responde | Ejemplo |
|-------|----------------------|---------|
| **Local - Token** | ¿Qué token específico cambió la predicción? | "Ter" tiene SHAP=15 |
| **Local - Input** | ¿Qué palabras hacen que ESTE texto sea positivo? | "amazing" y "loved" |
| **Global - Input** | ¿Qué patterns ve el modelo en TODOS los textos? | Exclamaciones = positivo |
| **Global - Modelo** | ¿Qué características generales usa el modelo? | Palabras negativas = negativo |

**Mi error inicial:**
Mezclar niveles. Comparaba importancia de tokens (local) con patterns 
generales (global) sin separarlos explícitamente.

**Aprendizaje:**
Cada análisis responde una pregunta diferente. El secreto está en visualizar 
los datos correctamente para cada nivel y **escribir una frase en lenguaje 
natural** que explique qué significa cada número.

---

## 7. Conclusión: La visualización es interpretabilidad

**Insight final:**
Los números crudos (SHAP=20.1464) no sirven de nada sin contexto.

**Lo que realmente importa:**
Transformar `Terrible SHAP=20.1464` en:
> "La palabra 'Terrible' aumenta en un 20% la confianza del modelo de que 
> esta review es negativa, siendo el factor más determinante en esta predicción."

**Mi takeaway:**
Cada proyecto de interpretabilidad necesita:
1. Datos numéricos (SHAP/LIME values)
2. Visualización intuitiva (colores, gráficos)
3. **Traducción a lenguaje humano** ← esto es lo que más falta en la literatura
