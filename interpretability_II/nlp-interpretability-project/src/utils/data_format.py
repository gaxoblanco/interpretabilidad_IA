
def validate_and_fix_labels(dataset_raw):
    """
    Valida y corrige automáticamente los labels del dataset para que sean compatibles
    con el modelo DistilBERT-SST-2 que espera: 0 = NEGATIVE, 1 = POSITIVE

    Args:
        dataset_raw: Dataset de HuggingFace con los splits cargados

    Returns:
        dataset_raw: Dataset con labels corregidos si fue necesario
    """
    print("🔍 VALIDACIÓN Y CORRECCIÓN DE LABELS")
    print("="*60)

    print("\n📋 Formato esperado por el modelo:")
    print("  • 0 = NEGATIVE")
    print("  • 1 = POSITIVE")

    # Verificar cada split
    splits_to_check = list(dataset_raw.keys())
    needs_correction = False
    correction_map = {}

    print("\n🔍 Verificando labels en cada split...")
    print("-"*40)

    for split_name in splits_to_check:
        split_data = dataset_raw[split_name]

        # Obtener muestra de labels
        sample_size = min(100, len(split_data))
        sample_labels = [split_data[i]['label'] for i in range(sample_size)]
        unique_labels = set(sample_labels)

        print(f"\n{split_name.upper()}:")
        print(f"  • Tamaño: {len(split_data)} ejemplos")
        print(f"  • Valores únicos en muestra: {unique_labels}")
        print(f"  • Tipo de dato: {type(sample_labels[0])}")

        # Verificar si los labels son correctos
        if unique_labels == {0, 1} or unique_labels == {0} or unique_labels == {1}:
            print(f"  ✅ Labels en formato correcto (0/1)")
        else:
            print(f"  ⚠️ Labels NO están en formato esperado")
            needs_correction = True

            # Intentar identificar el mapeo correcto
            if unique_labels == {-1, 1}:
                correction_map = {-1: 0, 1: 1}
                print(f"     Detectado formato: -1/1 → Se convertirá a 0/1")
            elif all(isinstance(x, str) for x in unique_labels):
                # Labels son strings
                str_labels = list(unique_labels)
                if 'negative' in [s.lower() for s in str_labels]:
                    # Formato string lowercase
                    correction_map = {'negative': 0, 'positive': 1}
                elif 'NEGATIVE' in str_labels:
                    # Formato string uppercase
                    correction_map = {'NEGATIVE': 0, 'POSITIVE': 1}
                print(f"     Detectado formato string → Se convertirá a 0/1")
            else:
                print(f"     ⚠️ Formato no reconocido: {unique_labels}")

    # Aplicar corrección si es necesaria
    if needs_correction and correction_map:
        print("\n🔧 APLICANDO CORRECCIONES...")
        print("-"*40)

        from datasets import Dataset, DatasetDict

        corrected_dataset = {}

        for split_name in splits_to_check:
            split_data = dataset_raw[split_name]

            # Crear lista de datos corregidos
            corrected_data = []
            for example in split_data:
                corrected_example = example.copy()
                # Corregir el label
                if 'label' in corrected_example:
                    original_label = corrected_example['label']
                    if original_label in correction_map:
                        corrected_example['label'] = correction_map[original_label]
                corrected_data.append(corrected_example)

            # Crear nuevo Dataset con datos corregidos
            corrected_dataset[split_name] = Dataset.from_list(corrected_data)

            # Verificar corrección
            sample_corrected = [corrected_dataset[split_name][i]['label']
                                for i in range(min(10, len(corrected_dataset[split_name])))]
            print(
                f"\n{split_name}: Labels corregidos - Muestra: {sample_corrected}")

        dataset_raw = DatasetDict(corrected_dataset)
        print("\n✅ Corrección completada exitosamente")

    else:
        if not needs_correction:
            print(
                "\n✅ No se necesitan correcciones - Labels ya están en formato correcto")
        else:
            print(
                "\n⚠️ Se detectaron problemas pero no se pudo determinar la corrección automática")

    # Verificación final
    print("\n📊 VERIFICACIÓN FINAL:")
    print("-"*40)

    for split_name in splits_to_check:
        split_data = dataset_raw[split_name]
        labels = [split_data[i]['label'] for i in range(len(split_data))]
        unique = set(labels)
        count_0 = labels.count(0)
        count_1 = labels.count(1)

        print(f"\n{split_name.upper()}:")
        print(f"  • Valores únicos: {unique}")
        print(
            f"  • Cantidad de 0s (NEGATIVE): {count_0:,} ({count_0/len(labels)*100:.1f}%)")
        print(
            f"  • Cantidad de 1s (POSITIVE): {count_1:,} ({count_1/len(labels)*100:.1f}%)")

        if unique == {0, 1}:
            print(f"  ✅ Formato correcto para el modelo")
        else:
            print(f"  ⚠️ ADVERTENCIA: Aún hay problemas con el formato")

    print("\n" + "="*60)
    print("✅ Validación y corrección completadas")

    return dataset_raw
