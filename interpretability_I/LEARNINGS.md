# Learnings - interpretability_I

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


## Fase 2: Modelado con XGBoost ✅

### Fecha: [Fecha]
### Tiempo invertido: ~3-4 horas

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