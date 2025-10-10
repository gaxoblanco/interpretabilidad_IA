# Interpretability I - German Credit Data
- Para levantar el Dashboard: `streamlit run app.py`
- Autor: [@gaxoblanco]
- Fecha: 2025-10-10

## 🎯 Objetivo
Entender interpretabilidad en Machine Learning mediante análisis de riesgo crediticio.

## 📊 Dataset
- **Nombre:** German Credit Data
- **Tipo:** Clasificación binaria
- **Target:** risk (0=Good, 1=Bad)
- **Registros:** 1000 clientes
- **Features:** 20 variables (7 numéricas, 13 categóricas)
- **Desbalanceo:** 70/30 (ratio 2.33:1)

## 🛠️ Stack Tecnológico
- Python 3.8+
- XGBoost (modelo)
- SHAP (interpretabilidad)
- Streamlit (dashboard)
- Pandas, Matplotlib, Seaborn (análisis)

## 📂 Estructura
- `data/` - Datasets
- `notebooks/` - Análisis exploratorio y experimentación
  - ✅ 01_exploracion.ipynb - EDA completado
  - ✅ 02_modelado.ipynb - completado
  - ✅ 03_shap_analysis.ipynb - completado
  - ✅ 04_counterfactual_explanations.ipynb
- `models/` - Modelos entrenados
- `app.py` - Dashboard interactivo simple con Streamlit

## 📈 Progreso
- [x] Fase 1: Exploración de datos
- [x] Fase 2: Modelado XGBoost
- [x] Fase 3: Interpretabilidad SHAP
- [x] Fase 4: Dashboard

## 🚀 Próximos Pasos
1. ✅ Exploración completada
2. ✅ Entrenamiento modelo XGBoost
3. ✅ Implementación SHAP
4. ✅ Dashboard Streamlit