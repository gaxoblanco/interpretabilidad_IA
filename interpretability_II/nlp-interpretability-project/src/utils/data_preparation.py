# ============================================================
# ARCHIVO: src/utils/data_preparation.py
# ============================================================
# Funciones para preparar y dividir los datos del dataset
# ============================================================

def preparar_datos_para_evaluacion(dataset_raw, split_test='validation', sample_size=None, random_seed=42):
    """
    Prepara los datos del dataset para evaluación del modelo.
    Extrae textos y etiquetas del split especificado.

    Args:
        dataset_raw: Dataset de HuggingFace con los splits cargados
        split_test: Nombre del split a usar para test ('validation' o 'test')
        sample_size: Si se especifica, toma una muestra aleatoria de este tamaño
        random_seed: Semilla para reproducibilidad del muestreo

    Returns:
        tuple: (test_texts, test_labels)
            - test_texts: Lista de strings con los textos
            - test_labels: Lista de integers con las etiquetas (0=neg, 1=pos)

    Example:
        >>> test_texts, test_labels = preparar_datos_para_evaluacion(dataset_raw, 'validation')
        >>> print(f"Textos: {len(test_texts)}, Labels: {len(test_labels)}")
    """
    import random

    print(f"📦 PREPARACIÓN DE DATOS PARA EVALUACIÓN")
    print("="*60)

    # Validar que el split existe
    if split_test not in dataset_raw:
        raise ValueError(
            f"El split '{split_test}' no existe. Disponibles: {list(dataset_raw.keys())}")

    # Obtener el dataset del split especificado
    dataset_split = dataset_raw[split_test]
    total_samples = len(dataset_split)

    print(f"\n📊 Información del split '{split_test}':")
    print(f"  • Total de muestras: {total_samples:,}")

    # Determinar el tamaño de la muestra
    if sample_size is not None and sample_size < total_samples:
        print(f"  • Tomando muestra de: {sample_size:,} ejemplos")
        print(f"  • Seed aleatorio: {random_seed}")

        # Crear índices aleatorios
        random.seed(random_seed)
        indices = random.sample(range(total_samples), sample_size)
        indices.sort()  # Ordenar para mantener cierto orden
    else:
        print(f"  • Usando dataset completo")
        indices = list(range(total_samples))

    # Extraer textos y etiquetas
    print(f"\n🔄 Extrayendo datos...")
    print("-"*40)

    test_texts = []
    test_labels = []

    # Identificar el nombre correcto de la columna de texto
    text_column = None
    for col in ['sentence', 'text', 'review', 'input']:
        if col in dataset_split.column_names:
            text_column = col
            break

    if text_column is None:
        raise ValueError(
            f"No se encontró columna de texto. Columnas disponibles: {dataset_split.column_names}")

    print(f"  • Columna de texto: '{text_column}'")
    print(f"  • Columna de etiquetas: 'label'")

    # Extraer los datos
    for idx in indices:
        example = dataset_split[idx]
        test_texts.append(example[text_column])
        test_labels.append(example['label'])

    # Validar los datos extraídos
    print(f"\n✅ Datos extraídos correctamente:")
    print(f"  • Textos: {len(test_texts)} ejemplos")
    print(f"  • Etiquetas: {len(test_labels)} ejemplos")

    # Verificar tipos de datos
    if test_texts:
        print(f"  • Tipo de textos: {type(test_texts[0]).__name__}")
        print(f"  • Ejemplo de texto: '{test_texts[0][:50]}...'")

    if test_labels:
        print(f"  • Tipo de etiquetas: {type(test_labels[0]).__name__}")
        unique_labels = set(test_labels)
        print(f"  • Valores únicos de etiquetas: {sorted(unique_labels)}")

        # Contar distribución
        count_neg = test_labels.count(0)
        count_pos = test_labels.count(1)
        print(f"\n📊 Distribución de clases:")
        print(
            f"  • Clase 0 (Negative): {count_neg:,} ({count_neg/len(test_labels)*100:.1f}%)")
        print(
            f"  • Clase 1 (Positive): {count_pos:,} ({count_pos/len(test_labels)*100:.1f}%)")

        # Verificar balance
        ratio = count_pos / count_neg if count_neg > 0 else float('inf')
        if 0.8 <= ratio <= 1.2:
            print(f"  • ✅ Dataset balanceado (ratio: {ratio:.2f})")
        else:
            print(f"  • ⚠️ Dataset desbalanceado (ratio: {ratio:.2f})")

    # Verificación final
    assert len(test_texts) == len(
        test_labels), "Error: Número diferente de textos y etiquetas"
    assert all(isinstance(text, str)
               for text in test_texts[:10]), "Error: Los textos deben ser strings"
    assert all(label in [0, 1] for label in test_labels[:10]
               ), "Error: Las etiquetas deben ser 0 o 1"

    print("\n" + "="*60)
    print("✅ Datos preparados exitosamente para evaluación")

    return test_texts, test_labels


def dividir_dataset_train_val_test(dataset_raw, train_size=0.8, val_size=0.1, test_size=0.1, random_seed=42):
    """
    Divide un dataset en train, validation y test.
    Útil cuando el dataset original no tiene los splits necesarios.

    Args:
        dataset_raw: Dataset completo
        train_size: Proporción para entrenamiento (default 0.8)
        val_size: Proporción para validación (default 0.1)
        test_size: Proporción para test (default 0.1)
        random_seed: Semilla para reproducibilidad

    Returns:
        dict: Diccionario con 'train', 'validation', 'test' splits
    """
    from sklearn.model_selection import train_test_split

    print(f"🔄 DIVISIÓN DE DATASET EN TRAIN/VAL/TEST")
    print("="*60)

    # Validar proporciones
    assert abs(train_size + val_size + test_size -
               1.0) < 0.001, "Las proporciones deben sumar 1.0"

    print(f"\n📊 Proporciones definidas:")
    print(f"  • Train: {train_size:.0%}")
    print(f"  • Validation: {val_size:.0%}")
    print(f"  • Test: {test_size:.0%}")

    # Implementación de la división...
    # (código de división aquí si es necesario)

    return dataset_raw  # Por ahora retorna el mismo si ya tiene los splits
