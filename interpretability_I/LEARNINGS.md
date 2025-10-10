# Learnings - interpretability_I
## Estrutura del documento
## Fase ...
### Hallazgos Clave:
### Preguntas Resueltas
### Lo que aprendi


## Fase 1: Exploración de Datos ✅

### Hallazgos Clave:

1. **Desbalanceo de clases (2.33:1)**
   - 700 clientes "Good Risk" (70%)
   - 300 clientes "Bad Risk" (30%)
   - ⚠️ Implicación: Necesitamos manejar esto en el modelo
   - Solución propuesta: `scale_pos_weight` en XGBoost

2. **Calidad de datos: Excelente**
   - ✅ Sin valores faltantes
   - ✅ Sin necesidad de imputación

3. **Variables más prometedoras** (basado en visualizaciones):
   - La mejor relacion la tienen duration = credit_amount
   - Las categorias salvo por Top porpouse, tienden a tener saltos cuntitativos entre una opcion y otra

4. **Variables categóricas vs numéricas**
   - 13 categóricas → Necesitarán encoding
   - 7 numéricas → Verificar si necesitan scaling

## ####################################################################
## Fase 2: Modelado con XGBoost ✅

### Hallazgos Clave:

#### 1. **Overfitting en modelo inicial**
   - Modelo baseline (max_depth=5): Diferencia Train-Test de 21.9%
   - ⚠️ Implicación: El modelo memorizaba los datos en lugar de aprender patrones
   - Solución implementada: Reducir complejidad del modelo

#### 2. **Importancia de la simplicidad**
   - Reducir `max_depth` de 5 → 3 fue el cambio más impactante pero sutil
   - Árboles más simples generalizan mejor en datasets pequeños (1000 registros)
   - ✅ Resultado: Overfitting bajó de 21.9% a 8.0%

#### 3. **Compensación entre hiperparámetros**
   - `learning_rate` bajo (0.05) + más árboles (200) = aprendizaje estable
   - Esta combinación compensó la menor complejidad de los árboles
   - Test Accuracy mejoró de 75.5% a 78.0%

#### 4. **Recall mejoró significativamente (+10%)**
   - De 61.7% a 71.7%
   - Ahora detectamos ~72 de cada 100 clientes Bad Risk (vs 62 antes)
   - Importante para el contexto de negocio: reducir pérdidas por malos clientes

#### 5. **ROC-AUC = 0.809 indica buen modelo**
   - Rango "Bueno" (0.80-0.89)
   - El modelo discrimina bien entre Good y Bad Risk
   - Comparable con benchmarks reportados en literatura para este dataset

### Decisiones Tomadas:

- [x] Usar Label Encoding (suficiente para XGBoost, no requiere One-Hot)
- [x] Configuración final: max_depth=3, lr=0.05, n_est=200
- [x] Mantener scale_pos_weight=2.33 (balance entre Recall y Precision)
- [x] Split estratificado 80/20 mantiene proporción del desbalanceo

### Preguntas Resueltas:

- ✅ ¿Cómo reducir overfitting? → Reducir max_depth
- ✅ ¿Qué hiperparámetros son más importantes? → max_depth, learning_rate, n_estimators
- ✅ ¿El modelo generaliza bien? → Sí, con diferencia Train-Test de 8%
- ✅ ¿Qué métrica priorizar? → Recall para detectar Bad Risk

### Lo Que Aprendí:

1. **No existe almuerzo gratis:** Mejorar una métrica (Recall) puede afectar otras (Precision, Accuracy)

2. **Simplicidad funciona:** Árboles simples (depth=3) + muchos árboles (200) > Árboles complejos (depth=5) + pocos árboles (100)

3. **Context matters:** Para riesgo crediticio, es más importante detectar malos clientes (Recall) que la precisión total (Accuracy)

4. **Iteración es clave:** El primer modelo rara vez es el mejor. Experimentar con hiperparámetros lleva a mejoras significativas.

5. **ROC-AUC > Accuracy:** Para datasets desbalanceados, ROC-AUC es más informativa que Accuracy

### Próximo paso: Interpretabilidad con SHAP

Ahora que tenemos un modelo sólido (78% accuracy, 8% overfitting, 0.809 AUC), es momento de entender **por qué** hace las predicciones que hace.

## ####################################################################

## Fase 3: Interpretabilidad con SHAP ✅

### Hallazgos Clave:

#### 1. **SHAP revela importancia REAL vs Feature Importance de XGBoost**
   - XGBoost sobreestimó: `residence_since` (#9), `own_telephone` (#8), `property_magnitude` (#2)
   - SHAP reveló las verdaderas top 3: `checking_status` (0.210), `duration` (0.124), `credit_amount` (0.096)
   - ⚠️ Implicación: Feature Importance cuenta splits, SHAP mide impacto real en predicciones

#### 2. **Waterfall plots explican decisiones individuales**
   - Cliente Good (99.3%): Ahorros buenos (-0.78), duración corta (-1.30), cuenta sana (-0.98)
   - Cliente Bad (95%): Cuenta mala (+0.84), historial malo (+0.60), plazo largo (+0.37)
   - ✅ El modelo razona correctamente: penaliza señales de riesgo.

#### 3. **Dependence plots revelan interacciones**
   - `duration` + `savings_status`: Plazos largos son menos riesgosos si tienes ahorros
   - `credit_amount` + `purpose`: Montos altos solo son riesgosos para ciertos propósitos
   - Relaciones no lineales: `checking_status` tiene impacto lineal, `credit_amount` es cuadrático

#### 4. **Análisis de errores (False Negatives)**
   - Cliente #4 (Bad aprobado): Ahorros moderados (-0.59) y buen historial (-0.38) ocultaron señales débiles de riesgo
   - Factores engañosos: Features "buenas" dominaron sobre señales sutiles (edad joven, duración media)
   - ⚠️ Implicación: El modelo confía demasiado en historial superficial (joven con historial corto lo califica como bueno)

#### 5. **Feature Selection no mejoró el modelo**
   - Eliminar 8 features de bajo SHAP (<0.02) causó pérdidas:
     - Test Accuracy: -7.5% (78% → 70.5%)
     - ROC-AUC: -0.024 (0.809 → 0.785)
     - Precision: -10.8% (0.614 → 0.506)
   - ✅ Conclusión: Features con bajo SHAP individual SÍ aportan en interacciones combinadas
   - Dataset pequeño (1000 registros, 20 features) se beneficia de mantener todas las variables.
   - En datasets grandes (miles de features), feature selection podria ser más útil.

### Decisiones Tomadas:

- [x] Usar SHAP como método principal de interpretabilidad (más confiable que Feature Importance)
- [x] Mantener las 20 features originales (feature selection empeoró resultados)
- [x] Documentar patrones de error para mejorar proceso de aprobación
- [x] Priorizar explicaciones con Waterfall plots (más claras para no técnicos)

### Preguntas Resueltas:

- ✅ ¿Por qué el modelo rechaza/aprueba clientes? → Waterfall plots desglosan cada decisión
- ✅ ¿Qué features importan más? → SHAP: checking_status, duration, credit_amount (no property_magnitude)
- ✅ ¿Hay interacciones entre features? → Sí, duration + savings, credit_amount + purpose
- ✅ ¿Se puede simplificar el modelo eliminando features? → NO para este dataset

### Lo Que Aprendí:

1. **SHAP > Feature Importance:** Feature Importance mide uso (splits), SHAP mide impacto real. Para decisiones de negocio, confiar en SHAP.

2. **SHAP bajo ≠ inútil:** Features con SHAP < 0.02 pueden ser importantes en combinaciones.

3. **Explicabilidad es clave:** Entender POR QUÉ el modelo falla (False Negatives) es más valioso que solo mejorar accuracy.

4. **Dataset pequeños son especiales:** Con 1000 registros y 20 features bien curadas, cada feature aporta. En datasets grandes, podria aportar valor hacer selección.

5. **Trade-offs de simplicidad:** Reducir features de 20 a 12 (-40%) para "simplificar" costó -7.5% accuracy. El beneficio no justifica la pérdida.

6. **Validación matemática de SHAP:** Base value + Σ(SHAP values) = predicción. Esta propiedad aditiva garantiza explicaciones consistentes.

### Próximos Pasos:

- [ ] Implementar dashboard interactivo con Streamlit
- [ ] Crear sistema de explicaciones para clientes rechazados
- [ ] Explorar ajuste de threshold (0.5 → 0.4) para reducir False Negatives
- [ ] Aplicar aprendizajes a proyecto SIFEN

## ####################################################################

## Fase 4: Counterfactual Explanations ✅

### Hallazgos Clave:

#### 1. **Algoritmo greedy para encontrar cambios mínimos**
   - Estrategia: Probar cambios en features modificables ordenadas por importancia
   - Aplicar solo cambios que mejoren la probabilidad (approach incremental)
   - ✅ Resultado: Encontrar soluciones con 1-3 cambios que logran aprobación

#### 2. **Clasificación de features por modificabilidad**
   - **Modificables (9):** savings_status, employment, duration, credit_amount, property_magnitude, housing, job, own_telephone, other_payment_plans
   - **No modificables (11):** checking_status, credit_history, age, personal_status, purpose, etc.
   - ⚠️ Implicación: Solo ~45% de features son realistas de cambiar

#### 3. **Patrones en soluciones encontradas**
   - Feature más modificada: `duration` (reducir plazo del crédito)
   - Segunda más común: `savings_status` (aumentar ahorros)
   - Tercera: `property_magnitude` (mejorar situación de propiedad)
   - ✅ Coincide con top features de SHAP (duration #2, savings #4)

#### 4. **Efectividad de contrafactuales**
   - Casos recuperables (55-75% Bad Risk): ~67% logran aprobación con 1-3 cambios
   - Mejora promedio: 15-30% reducción en probabilidad de Bad Risk
   - Casos extremos (>80% Bad Risk): Difíciles de revertir incluso con 3 cambios
   - ✅ Validación: Las recomendaciones son accionables y efectivas

#### 5. **Limitaciones identificadas**
   - El algoritmo asume independencia entre cambios (no considera interacciones complejas, tener una propiedad puede facilitar los pagos de un crédito)
   - Usa valores "más comunes" de clientes aprobados (puede no ser óptimo, por ejemplo el caso de `duration` implicaria una cuota mas alta)
   - No valida si los cambios son realistas para el cliente individual ( un joven no puede simplemente aumentar sus ahorros de 0 a 1000 en un mes u obtener una propiedad)
   - No considera el "costo" o dificultad de cada cambio

### Decisiones Tomadas:

- [x] Implementar algoritmo greedy (balance entre optimalidad y velocidad)
- [x] Limitar a 3 cambios máximos (principio de parsimonia)
- [x] Usar valores de clientes aprobados como referencia
- [x] Generar recomendaciones en lenguaje natural para clientes

### Preguntas Resueltas:

- ✅ ¿Qué debe cambiar un cliente rechazado para ser aprobado? → 1-3 cambios específicos en features críticas
- ✅ ¿Son realistas las recomendaciones? → Sí y no, enfocadas en features modificables pero pueden estar sesgadas
- ✅ ¿Cuánto mejora cada cambio? → Cuantificado: duration (-28%), savings (-15%), etc.
- ✅ ¿Se pueden automatizar las recomendaciones? → Sí, con texto personalizado por cliente podriamos guiarlos a un prestamo aprobado

### Lo Que Aprendí:

1. **Counterfactuals = Explicabilidad accionable:** No solo dicen "por qué rechazamos", sino "cómo aprobar". Más útil para clientes que SHAP.

2. **Greedy es suficiente:** Encontrar la solución óptima global es NP-hard. Un algoritmo greedy (mejor mejora en cada paso) da soluciones buenas en tiempo razonable.

3. **Contexto de negocio importa:** No todos los cambios son iguales. Reducir `duration` es negociable con el banco (fácil), aumentar `savings_status` requiere meses (difícil).

4. **Múltiples soluciones posibles:** Un mismo cliente puede tener 3-5 combinaciones diferentes de cambios que logran aprobación. Ofrecer opciones es mejor que una sola solución.

5. **Trade-off num_changes vs mejora:** 1 cambio → 10% mejora, 2 cambios → 25%, 3 cambios → 30%. Rendimientos decrecientes sugieren priorizar cambios mínimos. Para este caso de uso, lo mejor seria modificar duration para que el cliente pueda acceder a un credito aprobado.

### Próximos Pasos:

- [ ] Implementar búsqueda exhaustiva de top-K soluciones (ofrecer alternativas)
- [ ] Agregar "costos" a cambios (fácil/medio/difícil) para priorizar soluciones realistas
- [ ] Validar con expertos de negocio si las recomendaciones son viables
- [ ] Integrar counterfactuals al dashboard de aprobación