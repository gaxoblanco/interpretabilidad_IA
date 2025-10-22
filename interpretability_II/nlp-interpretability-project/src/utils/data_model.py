def verificar_entrada_salida_modelo(model):
    """
    Función que verifica qué tipo de datos entran y salen del modelo.

    Args:
        model: Instancia del ModelLoader cargado en la celda anterior

    Returns:
        dict: Diccionario con información de entrada y salida
    """
    print("="*60)
    print("🔍 VERIFICACIÓN DE ENTRADA/SALIDA DEL MODELO")
    print("="*60)

    # Texto de prueba
    texto_prueba = "This movie is absolutely amazing!"

    # ============ ENTRADA ============
    print("\n📥 ENTRADA AL MODELO:")
    print("-"*40)
    print(f"  • Tipo esperado: {type(texto_prueba).__name__} (string)")
    print(f"  • Ejemplo: '{texto_prueba}'")

    # ============ SALIDA ============
    print("\n📤 SALIDA DEL MODELO:")
    print("-"*40)

    # Hacer una predicción de prueba
    resultado = model.predict(texto_prueba)

    print(f"  • Tipo de salida: {type(resultado).__name__}")
    print(f"  • Campos disponibles: {list(resultado.keys())}")
    print(f"\n  • Detalle de cada campo:")
    for campo, valor in resultado.items():
        print(f"    - '{campo}': {type(valor).__name__} = {valor}")

    # ============ RESUMEN ============
    print("\n✅ RESUMEN:")
    print("-"*40)
    print(f"  ENTRADA → String de texto")
    print(f"  SALIDA  → Diccionario con:")
    print(
        f"           • prediction: {type(resultado['prediction']).__name__} ('{resultado['prediction']}')")
    print(
        f"           • label_id: {type(resultado.get('label_id', 'N/A')).__name__} ({resultado.get('label_id', 'N/A')})")

    return {
        'entrada': 'string',
        'salida_prediction': type(resultado['prediction']).__name__,
        'salida_label_id': type(resultado.get('label_id', 'N/A')).__name__,
        'valores_prediction': ['NEGATIVE', 'POSITIVE'],
        'valores_label_id': [0, 1]
    }

# Ejecutar esta función en la celda siguiente:
# info_io = verificar_entrada_salida_modelo(model)

# ============================================================
# DIAGNÓSTICO: ¿EL PROBLEMA ES EL MODELO O EL DATASET?
# ============================================================
# Esta función determinará definitivamente dónde está el problema
# ============================================================


def diagnosticar_problema_completo(model, dataset_validacion):
    """
    Diagnóstica si el problema está en el modelo o en el dataset.

    Args:
        model: El ModelLoader cargado
        dataset_validacion: El dataset de validación original
    """
    print("🔬 DIAGNÓSTICO DEFINITIVO: MODELO vs DATASET")
    print("="*60)

    # ----------------------------------------------------------
    # PASO 1: VERIFICAR EL MAPEO DEL DATASET
    # ----------------------------------------------------------
    print("\n1️⃣ MAPEO DEL DATASET SST-2:")
    print("-"*40)

    # Tomar algunos ejemplos del dataset
    ejemplos_dataset = []
    for i in range(min(5, len(dataset_validacion))):
        ejemplo = dataset_validacion[i]
        ejemplos_dataset.append(ejemplo)
        print(f"\nEjemplo {i+1}:")
        print(f"  • Texto: '{ejemplo['sentence'][:60]}...'")
        print(f"  • Label (número): {ejemplo['label']}")
        print(f"  • Significado según dataset: 0=negative, 1=positive")

        # Interpretación humana
        if "great" in ejemplo['sentence'].lower() or "amazing" in ejemplo['sentence'].lower() or "wonderful" in ejemplo['sentence'].lower():
            print(f"  • Interpretación humana: POSITIVO ✓")
        elif "terrible" in ejemplo['sentence'].lower() or "awful" in ejemplo['sentence'].lower() or "bad" in ejemplo['sentence'].lower():
            print(f"  • Interpretación humana: NEGATIVO ✗")

    # ----------------------------------------------------------
    # PASO 2: VERIFICAR LAS PREDICCIONES DEL MODELO
    # ----------------------------------------------------------
    print("\n2️⃣ PREDICCIONES DEL MODELO:")
    print("-"*40)

    for i, ejemplo in enumerate(ejemplos_dataset, 1):
        texto = ejemplo['sentence']
        label_real = ejemplo['label']

        # Predicción del modelo
        pred = model.predict(texto)

        print(f"\nEjemplo {i}:")
        print(f"  • Texto: '{texto[:60]}...'")
        print(
            f"  • Label real (dataset): {label_real} ({'positive' if label_real == 1 else 'negative'})")
        print(
            f"  • Predicción modelo: '{pred['prediction']}' (label_id: {pred['label_id']})")
        print(
            f"  • Probabilidades: NEGATIVE={pred['probabilities']['NEGATIVE']:.3f}, POSITIVE={pred['probabilities']['POSITIVE']:.3f}")

        # Análisis de coherencia
        prob_positiva = pred['probabilities']['POSITIVE']
        if prob_positiva > 0.5:
            print(f"  • Mayor probabilidad: POSITIVE ({prob_positiva:.1%})")
        else:
            print(
                f"  • Mayor probabilidad: NEGATIVE ({pred['probabilities']['NEGATIVE']:.1%})")

    # ----------------------------------------------------------
    # PASO 3: DIAGNÓSTICO FINAL
    # ----------------------------------------------------------
    print("\n3️⃣ DIAGNÓSTICO FINAL:")
    print("="*60)

    # Hacer una prueba con textos inequívocos
    textos_prueba = [
        ("This is absolutely fantastic! Best thing ever!", "POSITIVO"),
        ("Terrible, horrible, the worst thing I've seen", "NEGATIVO"),
    ]

    print("\nPrueba con textos inequívocos:")
    for texto, esperado in textos_prueba:
        pred = model.predict(texto)
        print(f"\n• Texto: '{texto}'")
        print(f"  Esperado: {esperado}")
        print(
            f"  Modelo dice: {pred['prediction']} (label_id: {pred['label_id']})")
        print(
            f"  Probabilidades: POS={pred['probabilities']['POSITIVE']:.1%}, NEG={pred['probabilities']['NEGATIVE']:.1%}")

        # Verificar coherencia
        if pred['probabilities']['POSITIVE'] > 0.9 and pred['prediction'] == 'NEGATIVE':
            print("  ⚠️ INCOHERENCIA: Alta prob POSITIVE pero prediction NEGATIVE")
        elif pred['probabilities']['NEGATIVE'] > 0.9 and pred['prediction'] == 'POSITIVE':
            print("  ⚠️ INCOHERENCIA: Alta prob NEGATIVE pero prediction POSITIVE")

    # ----------------------------------------------------------
    # PASO 4: VERIFICAR EL MAPEO INTERNO DEL MODELO
    # ----------------------------------------------------------
    print("\n4️⃣ CONFIGURACIÓN INTERNA DEL MODELO:")
    print("-"*40)

    info = model.get_model_info()
    print(f"  • id2label: {info.get('id2label', 'No disponible')}")
    print(f"  • label2id: {info.get('label2id', 'No disponible')}")

    print("\n" + "="*60)
    print("💡 CONCLUSIÓN:")
    print("-"*40)
    print("Revisa los resultados arriba para determinar:")
    print("1. Si las probabilidades coinciden con el sentimiento real → Problema en 'prediction'")
    print("2. Si 'label_id' coincide con el dataset (0=neg, 1=pos) → Usar label_id")
    print("3. Si todo está invertido → El modelo fue entrenado con mapeo invertido")

    return {
        'dataset_mapping': {'negative': 0, 'positive': 1},
        'model_output_fields': ['prediction', 'label_id', 'probabilities']
    }

# ----------------------------------------------------------
# EJECUTAR ASÍ:
# ----------------------------------------------------------
# diagnostico = diagnosticar_problema_completo(model, dataset['validation'])
#
# Luego, según los resultados, decidir si usar:
# - pred['label_id'] directamente
# - pred['probabilities'] para calcular la etiqueta
# - Invertir el mapeo de pred['prediction']
# ----------------------------------------------------------


# ============================================================
# AGREGAR ESTA FUNCIÓN AL ARCHIVO: src/utils/data_format.py
# ============================================================
# Va después de la función validate_and_fix_labels() que ya tienes
# ============================================================

def corregir_predicciones_modelo(predictions, threshold=0.5):
    """
    Corrige las predicciones del modelo usando las PROBABILIDADES.

    El diagnóstico mostró que:
    - El campo 'prediction' está MAL (siempre NEGATIVE)
    - El campo 'label_id' está MAL (siempre 0)
    - Las PROBABILIDADES están CORRECTAS ✓

    Por lo tanto, usamos las probabilidades para determinar la clase.

    Args:
        predictions: Lista de diccionarios devueltos por model.predict_batch()
                    Cada dict contiene: 'text', 'prediction', 'label_id',
                    'confidence', 'probabilities'
        threshold: Umbral para clasificar como positivo (default 0.5)

    Returns:
        list: Lista de etiquetas numéricas corregidas (0=negative, 1=positive)

    Raises:
        ValueError: Si las predicciones no tienen el formato esperado

    Example:
        >>> predictions = model.predict_batch(test_texts)
        >>> predicted_labels = corregir_predicciones_modelo(predictions)
        >>> # predicted_labels ahora tiene las etiquetas correctas basadas en probabilidades
    """
    print("\n🔧 Corrigiendo predicciones del modelo...")
    print("-"*40)

    # Validar entrada
    if not predictions or len(predictions) == 0:
        raise ValueError("No hay predicciones para corregir")

    # Verificar estructura del primer elemento
    first_pred = predictions[0]
    if 'probabilities' not in first_pred:
        raise ValueError("Las predicciones no contienen probabilidades")

    # Mostrar información sobre el problema detectado
    print("  📍 Problema detectado: label_id y prediction están incorrectos")
    print("  ✅ Solución: Usar PROBABILIDADES que están correctas")
    print(f"  📊 Threshold usado: {threshold}")

    # USAR PROBABILIDADES QUE ESTÁN CORRECTAS
    predicted_labels = []

    for pred in predictions:
        prob_positive = pred['probabilities']['POSITIVE']

        # Si la probabilidad de POSITIVE es mayor o igual al threshold, es 1
        if prob_positive >= threshold:
            predicted_labels.append(1)  # POSITIVE
        else:
            predicted_labels.append(0)  # NEGATIVE

    # Mostrar información de la corrección
    print(f"\n  • Total de predicciones procesadas: {len(predicted_labels)}")

    # Mostrar muestra de la corrección (primeros 3)
    if len(predictions) >= 3:
        print(f"\n  • Ejemplos de corrección:")
        for i in range(min(3, len(predictions))):
            pred = predictions[i]
            prob_pos = pred['probabilities']['POSITIVE']
            label_asignado = 1 if prob_pos >= threshold else 0
            print(
                f"    - Prob(POSITIVE): {prob_pos:.4f} → Label: {label_asignado}")

    # Verificar la distribución
    count_neg = predicted_labels.count(0)
    count_pos = predicted_labels.count(1)
    print(f"\n  • Distribución de predicciones corregidas:")
    print(
        f"    - Negativos (0): {count_neg} ({count_neg/len(predicted_labels)*100:.1f}%)")
    print(
        f"    - Positivos (1): {count_pos} ({count_pos/len(predicted_labels)*100:.1f}%)")

    # Validación de salida
    unique_values = set(predicted_labels)
    if not unique_values.issubset({0, 1}):
        print(
            f"  ⚠️ ADVERTENCIA: Se encontraron valores inesperados: {unique_values - {0, 1}}")
    else:
        print(f"  ✅ Todas las etiquetas son válidas (0 o 1)")

    # Advertencia si la distribución es muy desbalanceada
    if count_neg > 0:
        ratio = count_pos / count_neg
        if ratio < 0.2 or ratio > 5:
            print(
                f"\n  ⚠️ Nota: Distribución desbalanceada (ratio pos/neg: {ratio:.2f})")
            print(f"     Considera ajustar el threshold (actual: {threshold})")

    return predicted_labels


def corregir_predicciones_con_probabilidades(predictions, threshold=0.5):
    """
    Función alternativa: Corrige las predicciones usando las probabilidades directamente.
    Útil si no confías en label_id o quieres ajustar el threshold de decisión.

    Args:
        predictions: Lista de diccionarios devueltos por model.predict_batch()
        threshold: Umbral para clasificar como positivo (default 0.5)
                  - Valores > threshold → Positivo (1)
                  - Valores <= threshold → Negativo (0)

    Returns:
        list: Lista de etiquetas numéricas (0=negative, 1=positive)

    Example:
        >>> # Usar threshold default de 0.5
        >>> predicted_labels = corregir_predicciones_con_probabilidades(predictions)
        >>> 
        >>> # Usar threshold más conservador (más difícil ser positivo)
        >>> predicted_labels = corregir_predicciones_con_probabilidades(predictions, 0.7)
    """
    print(
        f"\n🔧 Corrigiendo predicciones usando probabilidades (threshold={threshold})...")
    print("-"*40)

    if not predictions or len(predictions) == 0:
        raise ValueError("No hay predicciones para corregir")

    # Verificar que tengamos probabilidades
    if 'probabilities' not in predictions[0]:
        raise ValueError("Las predicciones no contienen probabilidades")

    predicted_labels = []
    for pred in predictions:
        # Obtener la probabilidad de la clase POSITIVE
        prob_positive = pred['probabilities'].get('POSITIVE', 0)

        # Aplicar threshold
        if prob_positive >= threshold:
            predicted_labels.append(1)  # Positive
        else:
            predicted_labels.append(0)  # Negative

    # Mostrar estadísticas
    count_neg = predicted_labels.count(0)
    count_pos = predicted_labels.count(1)

    print(f"  • Total de predicciones: {len(predicted_labels)}")
    print(f"  • Threshold usado: {threshold}")
    print(
        f"  • Negativos (prob < {threshold}): {count_neg} ({count_neg/len(predicted_labels)*100:.1f}%)")
    print(
        f"  • Positivos (prob >= {threshold}): {count_pos} ({count_pos/len(predicted_labels)*100:.1f}%)")

    return predicted_labels
