# Funciones para visualizar los datos antes de procesarlos en modelos ML/NLP

import numpy as np


def view_distributions(dataset_raw):
    # 1. Estructura básica
    print("\n1️⃣ ESTRUCTURA DEL DATASET:")
    print("-"*40)
    for split_name, split_data in dataset_raw.items():
        print(f"\n{split_name.upper()}:")
        print(f"  • Tamaño: {len(split_data)} ejemplos")
        print(f"  • Columnas: {split_data.column_names}")
        print(f"  • Features: {split_data.features}")

    # 2. Muestras de datos
    print("\n2️⃣ EJEMPLOS DE DATOS:")
    print("-"*40)

    # Ver primeros ejemplos de train
    print("\nPrimeros 5 ejemplos de TRAIN:")
    for i in range(min(5, len(dataset_raw['train']))):
        example = dataset_raw['train'][i]
        label_text = "POSITIVE" if example['label'] == 1 else "NEGATIVE"
        print(f"{i+1}. [{label_text}] '{example['sentence'][:80]}...'")

    # Ver primeros ejemplos de validation
    print("\nPrimeros 5 ejemplos de VALIDATION:")
    for i in range(min(5, len(dataset_raw['validation']))):
        example = dataset_raw['validation'][i]
        label_text = "POSITIVE" if example['label'] == 1 else "NEGATIVE"
        print(f"{i+1}. [{label_text}] '{example['sentence'][:80]}...'")

    # 3. Análisis de longitudes
    print("\n3️⃣ ANÁLISIS DE LONGITUD DE TEXTOS:")
    print("-"*40)

    import numpy as np

    # Calcular longitudes para train
    train_lengths = [len(ex['sentence'].split())
                     for ex in dataset_raw['train']]
    val_lengths = [len(ex['sentence'].split())
                   for ex in dataset_raw['validation']]

    print(f"\nTRAIN ({len(train_lengths)} ejemplos):")
    print(f"  • Promedio: {np.mean(train_lengths):.1f} palabras")
    print(f"  • Mediana: {np.median(train_lengths):.1f} palabras")
    print(f"  • Mínimo: {np.min(train_lengths)} palabras")
    print(f"  • Máximo: {np.max(train_lengths)} palabras")
    print(f"  • Std: {np.std(train_lengths):.1f}")

    print(f"\nVALIDATION ({len(val_lengths)} ejemplos):")
    print(f"  • Promedio: {np.mean(val_lengths):.1f} palabras")
    print(f"  • Mediana: {np.median(val_lengths):.1f} palabras")
    print(f"  • Mínimo: {np.min(val_lengths)} palabras")
    print(f"  • Máximo: {np.max(val_lengths)} palabras")
    print(f"  • Std: {np.std(val_lengths):.1f}")

    # 4. Balance de clases
    print("\n4️⃣ BALANCE DE CLASES:")
    print("-"*40)

    # Para train
    train_labels = [ex['label'] for ex in dataset_raw['train']]
    train_neg = train_labels.count(0)
    train_pos = train_labels.count(1)

    print(f"\nTRAIN:")
    print(
        f"  • Negativos (0): {train_neg:,} ({train_neg/len(train_labels)*100:.1f}%)")
    print(
        f"  • Positivos (1): {train_pos:,} ({train_pos/len(train_labels)*100:.1f}%)")
    print(f"  • Ratio Pos/Neg: {train_pos/train_neg:.2f}")

    # Para validation
    val_labels = [ex['label'] for ex in dataset_raw['validation']]
    val_neg = val_labels.count(0)
    val_pos = val_labels.count(1)

    print(f"\nVALIDATION:")
    print(f"  • Negativos (0): {val_neg} ({val_neg/len(val_labels)*100:.1f}%)")
    print(f"  • Positivos (1): {val_pos} ({val_pos/len(val_labels)*100:.1f}%)")
    print(f"  • Ratio Pos/Neg: {val_pos/val_neg:.2f}")
