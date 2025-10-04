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

