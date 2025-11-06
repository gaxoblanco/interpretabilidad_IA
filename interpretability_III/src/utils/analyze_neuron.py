import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters
from scipy.ndimage import generic_filter
from skimage.transform import resize
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from utils.image_loader import ImageLoader

# ============================================================================
# 4️⃣ ANÁLISIS POSICION: ¿Esta neurona tiene sesgo de posición?
# ============================================================================


@staticmethod
def analyze_spatial_bias(
    neuron_index: int,
    layer_name: str,
    concatenated_activations: dict,
    num_samples: int = 50,
    verbose: bool = True
) -> dict:
    """
    Analiza si una neurona tiene sesgo espacial (izquierda vs derecha, arriba vs abajo).

    Args:
        neuron_index: Índice de la neurona a analizar
        layer_name: Nombre de la capa (ej: 'layer3.1.relu')
        concatenated_activations: Dict con activaciones capturadas
        num_samples: Número de imágenes a analizar
        verbose: Si True, imprime resultados detallados

    Returns:
        Dict con análisis completo de sesgo espacial
    """
    import numpy as np
    import torch

    # Verificar que la capa existe
    if layer_name not in concatenated_activations:
        raise ValueError(f"Capa '{layer_name}' no encontrada en activaciones")

    # ========================================================================
    # PASO 1: Extraer activaciones de la neurona
    # ========================================================================

    neuron_activations = concatenated_activations[layer_name][:num_samples,
                                                              neuron_index, :, :]
    # Shape: [num_samples, H, W]

    activation_shape = neuron_activations.shape
    num_images = activation_shape[0]
    height = activation_shape[1]
    width = activation_shape[2]

    # ========================================================================
    # PASO 2: Calcular activaciones por región espacial
    # ========================================================================

    # Dividir en izquierda/derecha
    left_half = neuron_activations[:, :, :width//2]
    right_half = neuron_activations[:, :, width//2:]

    left_activations = left_half.mean(dim=(1, 2)).cpu().numpy()
    right_activations = right_half.mean(dim=(1, 2)).cpu().numpy()

    # Dividir en arriba/abajo
    top_half = neuron_activations[:, :height//2, :]
    bottom_half = neuron_activations[:, height//2:, :]

    top_activations = top_half.mean(dim=(1, 2)).cpu().numpy()
    bottom_activations = bottom_half.mean(dim=(1, 2)).cpu().numpy()

    # ========================================================================
    # PASO 3: Calcular estadísticas de sesgo horizontal (izq/der)
    # ========================================================================

    left_mean = float(left_activations.mean())
    right_mean = float(right_activations.mean())
    left_std = float(left_activations.std())
    right_std = float(right_activations.std())

    horizontal_ratio = left_mean / (right_mean + 1e-8)
    horizontal_diff = left_mean - right_mean

    # Contar dominancia
    left_dominant_count = int((left_activations > right_activations).sum())
    right_dominant_count = num_images - left_dominant_count

    left_dominant_pct = (left_dominant_count / num_images) * 100
    right_dominant_pct = (right_dominant_count / num_images) * 100

    # Determinar sesgo horizontal
    if horizontal_ratio > 1.3:
        horizontal_bias = "FUERTE hacia IZQUIERDA"
    elif horizontal_ratio > 1.1:
        horizontal_bias = "MODERADO hacia IZQUIERDA"
    elif horizontal_ratio < 0.77:  # 1/1.3
        horizontal_bias = "FUERTE hacia DERECHA"
    elif horizontal_ratio < 0.91:  # 1/1.1
        horizontal_bias = "MODERADO hacia DERECHA"
    else:
        horizontal_bias = "SIN SESGO (balanceado)"

    # ========================================================================
    # PASO 4: Calcular estadísticas de sesgo vertical (arriba/abajo)
    # ========================================================================

    top_mean = float(top_activations.mean())
    bottom_mean = float(bottom_activations.mean())
    top_std = float(top_activations.std())
    bottom_std = float(bottom_activations.std())

    vertical_ratio = top_mean / (bottom_mean + 1e-8)
    vertical_diff = top_mean - bottom_mean

    # Contar dominancia
    top_dominant_count = int((top_activations > bottom_activations).sum())
    bottom_dominant_count = num_images - top_dominant_count

    top_dominant_pct = (top_dominant_count / num_images) * 100
    bottom_dominant_pct = (bottom_dominant_count / num_images) * 100

    # Determinar sesgo vertical
    if vertical_ratio > 1.3:
        vertical_bias = "FUERTE hacia ARRIBA"
    elif vertical_ratio > 1.1:
        vertical_bias = "MODERADO hacia ARRIBA"
    elif vertical_ratio < 0.77:
        vertical_bias = "FUERTE hacia ABAJO"
    elif vertical_ratio < 0.91:
        vertical_bias = "MODERADO hacia ABAJO"
    else:
        vertical_bias = "SIN SESGO (balanceado)"

    # ========================================================================
    # PASO 5: Determinar sesgo dominante
    # ========================================================================

    horizontal_strength = abs(horizontal_ratio - 1.0)
    vertical_strength = abs(vertical_ratio - 1.0)

    if horizontal_strength > vertical_strength * 1.5:
        dominant_bias = "HORIZONTAL (izquierda/derecha)"
    elif vertical_strength > horizontal_strength * 1.5:
        dominant_bias = "VERTICAL (arriba/abajo)"
    elif max(horizontal_strength, vertical_strength) > 0.2:
        dominant_bias = "MIXTO (ambos ejes)"
    else:
        dominant_bias = "NINGUNO (neurona espacialmente uniforme)"

    # ========================================================================
    # PASO 6: Compilar resultados
    # ========================================================================

    results = {
        'neuron_info': {
            'index': neuron_index,
            'layer': layer_name,
            'activation_shape': list(activation_shape),
            'num_samples_analyzed': num_images
        },
        'horizontal_bias': {
            'bias_type': horizontal_bias,
            'left_mean': left_mean,
            'right_mean': right_mean,
            'left_std': left_std,
            'right_std': right_std,
            'ratio': horizontal_ratio,
            'difference': horizontal_diff,
            'left_dominant_count': left_dominant_count,
            'right_dominant_count': right_dominant_count,
            'left_dominant_pct': left_dominant_pct,
            'right_dominant_pct': right_dominant_pct
        },
        'vertical_bias': {
            'bias_type': vertical_bias,
            'top_mean': top_mean,
            'bottom_mean': bottom_mean,
            'top_std': top_std,
            'bottom_std': bottom_std,
            'ratio': vertical_ratio,
            'difference': vertical_diff,
            'top_dominant_count': top_dominant_count,
            'bottom_dominant_count': bottom_dominant_count,
            'top_dominant_pct': top_dominant_pct,
            'bottom_dominant_pct': bottom_dominant_pct
        },
        'dominant_bias': dominant_bias,
        '_raw_data': {
            'left_activations': left_activations,
            'right_activations': right_activations,
            'top_activations': top_activations,
            'bottom_activations': bottom_activations
        }
    }

    # ========================================================================
    # PASO 7: Imprimir resultados (si verbose=True)
    # ========================================================================

    if verbose:
        print("="*80)
        print(f"📍 ANÁLISIS DE SESGO ESPACIAL: Neurona #{neuron_index}")
        print("="*80)

        print(f"\n🔍 Información básica:")
        print(f"   Capa: {layer_name}")
        print(f"   Shape de activación: {list(activation_shape)}")
        print(f"   Imágenes analizadas: {num_images}")

        print(f"\n↔️  SESGO HORIZONTAL (Izquierda vs Derecha):")
        print(f"   Lado IZQUIERDO:  {left_mean:.4f} ± {left_std:.4f}")
        print(f"   Lado DERECHO:    {right_mean:.4f} ± {right_std:.4f}")
        print(f"   Ratio (Izq/Der): {horizontal_ratio:.2f}x")
        print(f"   Diferencia:      {horizontal_diff:+.4f}")
        print(f"\n   Dominancia:")
        print(
            f"   - Izquierda dominante: {left_dominant_count}/{num_images} ({left_dominant_pct:.1f}%)")
        print(
            f"   - Derecha dominante:   {right_dominant_count}/{num_images} ({right_dominant_pct:.1f}%)")
        print(f"\n   → {horizontal_bias}")

        print(f"\n↕️  SESGO VERTICAL (Arriba vs Abajo):")
        print(f"   Parte SUPERIOR:  {top_mean:.4f} ± {top_std:.4f}")
        print(f"   Parte INFERIOR:  {bottom_mean:.4f} ± {bottom_std:.4f}")
        print(f"   Ratio (Arr/Abj): {vertical_ratio:.2f}x")
        print(f"   Diferencia:      {vertical_diff:+.4f}")
        print(f"\n   Dominancia:")
        print(
            f"   - Arriba dominante: {top_dominant_count}/{num_images} ({top_dominant_pct:.1f}%)")
        print(
            f"   - Abajo dominante:  {bottom_dominant_count}/{num_images} ({bottom_dominant_pct:.1f}%)")
        print(f"\n   → {vertical_bias}")

        print(f"\n🎯 CONCLUSIÓN:")
        print(f"   Sesgo dominante: {dominant_bias}")

        # Interpretación adicional
        if "HORIZONTAL" in dominant_bias:
            if "IZQUIERDA" in horizontal_bias:
                print(
                    f"   💡 Esta neurona prefiere detectar objetos/contenido en el LADO IZQUIERDO")
            elif "DERECHA" in horizontal_bias:
                print(
                    f"   💡 Esta neurona prefiere detectar objetos/contenido en el LADO DERECHO")
        elif "VERTICAL" in dominant_bias:
            if "ARRIBA" in vertical_bias:
                print(
                    f"   💡 Esta neurona prefiere detectar objetos/contenido en la PARTE SUPERIOR")
            elif "ABAJO" in vertical_bias:
                print(
                    f"   💡 Esta neurona prefiere detectar objetos/contenido en la PARTE INFERIOR")
        elif "NINGUNO" in dominant_bias:
            print(f"   💡 Esta neurona procesa toda la imagen de forma uniforme")

        print("="*80)

    return results


@staticmethod
def visualize_spatial_bias(results: dict, figsize=(18, 12)):
    """
    Visualiza los resultados del análisis de sesgo espacial.

    Args:
        results: Dict retornado por analyze_spatial_bias
        figsize: Tamaño de la figura
    """
    import matplotlib.pyplot as plt
    import numpy as np

    neuron_idx = results['neuron_info']['index']
    num_images = results['neuron_info']['num_samples_analyzed']

    h_bias = results['horizontal_bias']
    v_bias = results['vertical_bias']
    raw = results['_raw_data']

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # ========== 1. SCATTER: Izquierda vs Derecha ==========

    ax1 = fig.add_subplot(gs[0, 0])

    x = np.arange(num_images)
    ax1.scatter(x, raw['left_activations'], alpha=0.6, s=50,
                color='red', label='Lado Izquierdo')
    ax1.scatter(x, raw['right_activations'], alpha=0.6, s=50,
                color='blue', label='Lado Derecho')
    ax1.axhline(h_bias['left_mean'], color='red', linestyle='--',
                alpha=0.5, label=f"Media Izq ({h_bias['left_mean']:.3f})")
    ax1.axhline(h_bias['right_mean'], color='blue', linestyle='--',
                alpha=0.5, label=f"Media Der ({h_bias['right_mean']:.3f})")

    ax1.set_xlabel('Índice de Imagen')
    ax1.set_ylabel('Activación Promedio')
    ax1.set_title(f'Neurona #{neuron_idx}: Activación Izquierda vs Derecha')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)

    # ========== 2. BARRAS: Diferencia Horizontal ==========

    ax2 = fig.add_subplot(gs[0, 1])

    difference_h = raw['left_activations'] - raw['right_activations']
    colors_h = ['red' if d > 0 else 'blue' for d in difference_h]

    ax2.bar(x, difference_h, color=colors_h, alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Índice de Imagen')
    ax2.set_ylabel('Diferencia (Izq - Der)')
    ax2.set_title(
        'Sesgo Horizontal por Imagen\n(Rojo=Izq dominante, Azul=Der dominante)')
    ax2.grid(alpha=0.3, axis='y')

    # ========== 3. SCATTER: Arriba vs Abajo ==========

    ax3 = fig.add_subplot(gs[1, 0])

    ax3.scatter(x, raw['top_activations'], alpha=0.6, s=50,
                color='green', label='Parte Superior')
    ax3.scatter(x, raw['bottom_activations'], alpha=0.6, s=50,
                color='orange', label='Parte Inferior')
    ax3.axhline(v_bias['top_mean'], color='green', linestyle='--',
                alpha=0.5, label=f"Media Arr ({v_bias['top_mean']:.3f})")
    ax3.axhline(v_bias['bottom_mean'], color='orange', linestyle='--',
                alpha=0.5, label=f"Media Abj ({v_bias['bottom_mean']:.3f})")

    ax3.set_xlabel('Índice de Imagen')
    ax3.set_ylabel('Activación Promedio')
    ax3.set_title(f'Neurona #{neuron_idx}: Activación Arriba vs Abajo')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(alpha=0.3)

    # ========== 4. BARRAS: Diferencia Vertical ==========

    ax4 = fig.add_subplot(gs[1, 1])

    difference_v = raw['top_activations'] - raw['bottom_activations']
    colors_v = ['green' if d > 0 else 'orange' for d in difference_v]

    ax4.bar(x, difference_v, color=colors_v, alpha=0.7, edgecolor='black')
    ax4.axhline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Índice de Imagen')
    ax4.set_ylabel('Diferencia (Arr - Abj)')
    ax4.set_title(
        'Sesgo Vertical por Imagen\n(Verde=Arr dominante, Naranja=Abj dominante)')
    ax4.grid(alpha=0.3, axis='y')

    # ========== 5. RESUMEN: Gráfico de dominancia ==========

    ax5 = fig.add_subplot(gs[2, :])

    categories = ['Izquierda', 'Derecha', 'Arriba', 'Abajo']
    percentages = [
        h_bias['left_dominant_pct'],
        h_bias['right_dominant_pct'],
        v_bias['top_dominant_pct'],
        v_bias['bottom_dominant_pct']
    ]
    colors_bar = ['red', 'blue', 'green', 'orange']

    bars = ax5.bar(categories, percentages, color=colors_bar,
                   alpha=0.7, edgecolor='black')
    ax5.axhline(50, color='gray', linestyle='--',
                linewidth=1, label='50% (sin sesgo)')
    ax5.set_ylabel('Porcentaje de Dominancia (%)')
    ax5.set_title(f'Resumen: Dominancia Espacial de Neurona #{neuron_idx}')
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')

    # Añadir valores en las barras
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                 f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle(f'Análisis Completo de Sesgo Espacial - Neurona #{neuron_idx}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.show()

    # ========== Gráfico adicional: Indicador de sesgo ==========

    fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(14, 5))

    # Horizontal
    h_ratio = h_bias['ratio']
    ax_h.barh(['Sesgo Horizontal'], [h_ratio],
              color='purple', alpha=0.7, edgecolor='black')
    ax_h.axvline(1.0, color='gray', linestyle='--',
                 linewidth=2, label='Sin sesgo (1.0)')
    ax_h.axvline(1.1, color='orange', linestyle='--',
                 linewidth=1, label='Moderado (1.1)')
    ax_h.axvline(1.3, color='red', linestyle='--',
                 linewidth=1, label='Fuerte (1.3)')
    ax_h.set_xlabel('Ratio Izquierda/Derecha', fontsize=12, fontweight='bold')
    ax_h.set_title(
        f'Sesgo Horizontal: {h_bias["bias_type"]}', fontsize=12, fontweight='bold')
    ax_h.legend(loc='best')
    ax_h.grid(alpha=0.3, axis='x')
    ax_h.text(h_ratio, 0, f'  {h_ratio:.2f}x',
              va='center', fontsize=12, fontweight='bold')

    # Vertical
    v_ratio = v_bias['ratio']
    ax_v.barh(['Sesgo Vertical'], [v_ratio],
              color='teal', alpha=0.7, edgecolor='black')
    ax_v.axvline(1.0, color='gray', linestyle='--',
                 linewidth=2, label='Sin sesgo (1.0)')
    ax_v.axvline(1.1, color='orange', linestyle='--',
                 linewidth=1, label='Moderado (1.1)')
    ax_v.axvline(1.3, color='red', linestyle='--',
                 linewidth=1, label='Fuerte (1.3)')
    ax_v.set_xlabel('Ratio Arriba/Abajo', fontsize=12, fontweight='bold')
    ax_v.set_title(
        f'Sesgo Vertical: {v_bias["bias_type"]}', fontsize=12, fontweight='bold')
    ax_v.legend(loc='best')
    ax_v.grid(alpha=0.3, axis='x')
    ax_v.text(v_ratio, 0, f'  {v_ratio:.2f}x',
              va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

# ============================================================================
# 5️⃣ ANÁLISIS POR CLASE: ¿Esta neurona es selectiva a alguna clase?
# ============================================================================


@staticmethod
def analyze_class_selectivity(
    neuron_index: int,
    layer_name: str,
    concatenated_activations: dict,
    labels: np.ndarray,
    class_names: list,
    num_samples: int = 50,
    verbose: bool = True
) -> dict:
    """
    Analiza si una neurona es selectiva a alguna clase específica.

    Args:
        neuron_index: Índice de la neurona a analizar
        layer_name: Nombre de la capa (ej: 'layer3.1.relu')
        concatenated_activations: Dict con activaciones capturadas
        labels: Array con labels de las imágenes [num_images]
        class_names: Lista con nombres de las clases
        num_samples: Número de muestras a analizar
        verbose: Si True, imprime resultados detallados

    Returns:
        Dict con análisis de selectividad por clase
    """

    # Verificar que la capa existe
    if layer_name not in concatenated_activations:
        raise ValueError(f"Capa '{layer_name}' no encontrada en activaciones")

    # ========================================================================
    # PASO 1: Extraer activaciones de la neurona
    # ========================================================================

    # Obtener activaciones de TODAS las neuronas
    layer_activations = concatenated_activations[layer_name][:num_samples]
    # Shape: [num_samples, num_channels, H, W]

    # Extraer activaciones de la neurona específica
    neuron_activations = layer_activations[:, neuron_index, :, :]
    # Shape: [num_samples, H, W]

    # Calcular activación promedio espacial por imagen
    avg_activations = neuron_activations.mean(dim=(1, 2)).cpu().numpy()
    # Shape: [num_samples]

    # Truncar labels al mismo tamaño
    labels = labels[:num_samples]

    # ========================================================================
    # PASO 2: Calcular activación promedio por clase
    # ========================================================================

    class_statistics = {}

    for class_id, class_name in enumerate(class_names):
        # Obtener índices de imágenes de esta clase
        class_mask = labels == class_id

        if class_mask.sum() > 0:
            # Activaciones de esta clase
            class_acts = avg_activations[class_mask]

            class_statistics[class_name] = {
                'mean': float(class_acts.mean()),
                'std': float(class_acts.std()),
                'median': float(np.median(class_acts)),
                'max': float(class_acts.max()),
                'min': float(class_acts.min()),
                'count': int(class_mask.sum())
            }

    # ========================================================================
    # PASO 3: Identificar clase preferida
    # ========================================================================

    # Ordenar por activación media
    sorted_classes = sorted(
        class_statistics.items(),
        key=lambda x: x[1]['mean'],
        reverse=True
    )

    top_class = sorted_classes[0][0]
    top_activation = sorted_classes[0][1]['mean']
    second_class = sorted_classes[1][0] if len(sorted_classes) > 1 else None
    second_activation = sorted_classes[1][1]['mean'] if len(
        sorted_classes) > 1 else 0

    # Calcular diferencia relativa
    activation_diff = top_activation - second_activation
    activation_ratio = top_activation / (second_activation + 1e-8)

    # Determinar grado de selectividad
    if activation_ratio > 2.0:
        selectivity_level = "MUY ALTA"
        selectivity_description = f"Altamente especializada en '{top_class}'"
    elif activation_ratio > 1.5:
        selectivity_level = "ALTA"
        selectivity_description = f"Fuerte preferencia por '{top_class}'"
    elif activation_ratio > 1.2:
        selectivity_level = "MODERADA"
        selectivity_description = f"Preferencia moderada por '{top_class}'"
    else:
        selectivity_level = "BAJA"
        selectivity_description = "No muestra selectividad clara (generalista)"

    # ========================================================================
    # PASO 4: Compilar resultados
    # ========================================================================

    results = {
        'neuron_info': {
            'index': neuron_index,
            'layer': layer_name,
            'num_samples_analyzed': num_samples
        },
        'class_statistics': class_statistics,
        'selectivity': {
            'level': selectivity_level,
            'description': selectivity_description,
            'top_class': top_class,
            'top_activation': top_activation,
            'second_class': second_class,
            'second_activation': second_activation,
            'activation_difference': activation_diff,
            'activation_ratio': activation_ratio
        },
        'sorted_classes': sorted_classes,
        '_raw_data': {
            'avg_activations': avg_activations,
            'labels': labels
        }
    }

    # ========================================================================
    # PASO 5: Imprimir resultados (si verbose=True)
    # ========================================================================

    if verbose:
        print("="*80)
        print(f"🎨 SELECTIVIDAD POR CLASE: Neurona #{neuron_index}")
        print("="*80)

        # Crear DataFrame para mejor visualización
        df = pd.DataFrame(class_statistics).T
        df = df.sort_values('mean', ascending=False)

        print(f"\n📊 Activación promedio por clase (ordenado):")
        print("-"*80)
        display(df)

        print(f"\n🏆 RESULTADO:")
        print(f"   Clase preferida: {top_class.upper()}")
        print(f"   Activación promedio: {top_activation:.4f}")

        if second_class:
            print(f"\n📊 COMPARACIÓN:")
            print(f"   Segunda clase: {second_class}")
            print(f"   Activación: {second_activation:.4f}")
            print(f"   Diferencia: {activation_diff:.4f}")
            print(f"   Ratio: {activation_ratio:.2f}x")

        print(f"\n💡 SELECTIVIDAD: {selectivity_level}")
        print(f"   {selectivity_description}")

        print("="*80)

    return results


@staticmethod
def visualize_class_selectivity(results: dict, figsize=(15, 10)):
    """
    Visualiza los resultados del análisis de selectividad por clase.

    Args:
        results: Dict retornado por analyze_class_selectivity
        figsize: Tamaño de la figura
    """

    neuron_idx = results['neuron_info']['index']
    class_stats = results['class_statistics']
    sorted_classes = results['sorted_classes']
    top_class = results['selectivity']['top_class']

    # Crear DataFrame
    df = pd.DataFrame(class_stats).T
    df = df.sort_values('mean', ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ========== 1. Gráfico de barras: Media por clase ==========

    colors = ['red' if cls == top_class else 'steelblue' for cls in df.index]
    bars = axes[0, 0].bar(range(len(df)), df['mean'],
                          yerr=df['std'], capsize=5,
                          alpha=0.7, color=colors, edgecolor='black')

    axes[0, 0].set_xticks(range(len(df)))
    axes[0, 0].set_xticklabels(df.index, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Activación Promedio')
    axes[0, 0].set_title(
        f'Selectividad de Neurona #{neuron_idx} por Clase\n(Rojo = clase preferida)')
    axes[0, 0].grid(alpha=0.3, axis='y')

    # Añadir valores
    for i, (bar, val) in enumerate(zip(bars, df['mean'])):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # ========== 2. Box plot: Distribución por clase ==========

    class_names = df.index.tolist()
    raw_data = results['_raw_data']
    labels = raw_data['labels']
    avg_acts = raw_data['avg_activations']

    # Preparar datos para box plot
    box_data = []
    for cls in class_names:
        class_id = list(results['class_statistics'].keys()).index(cls)
        class_mask = labels == class_id
        box_data.append(avg_acts[class_mask])

    bp = axes[0, 1].boxplot(box_data, labels=class_names, patch_artist=True)

    # Colorear boxes
    for patch, cls in zip(bp['boxes'], class_names):
        if cls == top_class:
            patch.set_facecolor('red')
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Activación')
    axes[0, 1].set_title('Distribución de Activaciones por Clase')
    axes[0, 1].grid(alpha=0.3, axis='y')

    # ========== 3. Scatter: Todas las activaciones ==========

    colors_scatter = [list(results['class_statistics'].keys())[int(lbl)]
                      for lbl in labels]
    color_map = {cls: i for i, cls in enumerate(class_names)}
    color_indices = [color_map[c] for c in colors_scatter]

    scatter = axes[1, 0].scatter(range(len(avg_acts)), avg_acts,
                                 c=color_indices, cmap='tab10',
                                 alpha=0.6, s=50, edgecolors='black')

    axes[1, 0].set_xlabel('Índice de Imagen')
    axes[1, 0].set_ylabel('Activación')
    axes[1, 0].set_title(f'Activaciones de Neurona #{neuron_idx} por Imagen')
    axes[1, 0].grid(alpha=0.3)

    # ========== 4. Ratio de selectividad ==========

    # Calcular ratio respecto a la mejor clase
    top_mean = df.iloc[0]['mean']
    ratios = df['mean'] / top_mean

    colors_ratio = ['red' if cls ==
                    top_class else 'steelblue' for cls in df.index]
    axes[1, 1].barh(df.index, ratios, color=colors_ratio,
                    alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(1.0, color='black', linestyle='--',
                       linewidth=2, label='Clase top')
    axes[1, 1].set_xlabel('Ratio (relativo a clase preferida)')
    axes[1, 1].set_title(f'Selectividad Relativa\n(1.0 = {top_class})')
    axes[1, 1].grid(alpha=0.3, axis='x')
    axes[1, 1].legend()

    # Añadir porcentajes
    for i, (idx, ratio) in enumerate(zip(df.index, ratios)):
        axes[1, 1].text(ratio, i, f' {ratio:.2f}',
                        va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    # ========== Gráfico adicional: Resumen de selectividad ==========

    fig, ax = plt.subplots(figsize=(10, 6))

    selectivity_level = results['selectivity']['level']
    activation_ratio = results['selectivity']['activation_ratio']

    # Indicador visual de selectividad
    if selectivity_level == "MUY ALTA":
        color_level = 'darkgreen'
    elif selectivity_level == "ALTA":
        color_level = 'green'
    elif selectivity_level == "MODERADA":
        color_level = 'orange'
    else:
        color_level = 'red'

    ax.barh(['Selectividad'], [activation_ratio],
            color=color_level, alpha=0.7, edgecolor='black')
    ax.axvline(1.0, color='gray', linestyle='--',
               linewidth=1, label='Sin preferencia')
    ax.axvline(1.5, color='orange', linestyle='--',
               linewidth=1, label='Moderada')
    ax.axvline(2.0, color='green', linestyle='--', linewidth=1, label='Alta')
    ax.set_xlabel('Ratio de Activación (Top / Segunda)',
                  fontsize=12, fontweight='bold')
    ax.set_title(f'Nivel de Selectividad: {selectivity_level}\n'
                 f'Neurona #{neuron_idx} → Preferencia por "{top_class.upper()}"',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')

    # Añadir valor
    ax.text(activation_ratio, 0, f'  {activation_ratio:.2f}x',
            va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

# ============================================================================
# 6️⃣ ANÁLISIS DE TEXTURAS: Función modular
# ============================================================================


def analyze_neuron_texture_and_features(
    neuron_index: int,
    image_index: int,
    layer_name: str,
    concatenated_activations: dict,
    test_loader: DataLoader,
    image_loader: ImageLoader,
    dataset_info: dict,
    verbose: bool = True
) -> dict:
    """
    Analiza qué características visuales (textura, color, forma, contorno) 
    detecta una neurona específica en una imagen.

    Args:
        neuron_index: Índice de la neurona a analizar
        image_index: Índice de la imagen a analizar
        layer_name: Nombre de la capa (ej: 'layer3.1.relu')
        concatenated_activations: Dict con activaciones capturadas
        test_loader: DataLoader del test set
        image_loader: Instancia de ImageLoader
        dataset_info: Dict con info del dataset
        verbose: Si True, imprime resultados detallados

    Returns:
        Dict con todos los resultados del análisis
    """

    # Verificar que la capa existe
    if layer_name not in concatenated_activations:
        raise ValueError(f"Capa '{layer_name}' no encontrada en activaciones")

    # ========================================================================
    # PASO 1: Obtener imagen y activación
    # ========================================================================

    # Obtener imagen
    test_images, test_labels = next(iter(test_loader))
    image = test_images[image_index]
    label = test_labels[image_index]
    class_name = dataset_info['classes'][label]

    # Denormalizar para visualización
    image_denorm = image_loader.denormalize_image(image)

    # Obtener activación de la neurona
    neuron_activation = concatenated_activations[layer_name][image_index, neuron_index].cpu(
    ).numpy()
    activation_shape = neuron_activation.shape

    # ========================================================================
    # PASO 2: Análisis de COLOR
    # ========================================================================

    color_analysis = {
        'Red': {
            'mean': float(image_denorm[:, :, 0].mean()),
            'std': float(image_denorm[:, :, 0].std()),
            'max': float(image_denorm[:, :, 0].max())
        },
        'Green': {
            'mean': float(image_denorm[:, :, 1].mean()),
            'std': float(image_denorm[:, :, 1].std()),
            'max': float(image_denorm[:, :, 1].max())
        },
        'Blue': {
            'mean': float(image_denorm[:, :, 2].mean()),
            'std': float(image_denorm[:, :, 2].std()),
            'max': float(image_denorm[:, :, 2].max())
        }
    }

    dominant_channel = max(
        color_analysis, key=lambda x: color_analysis[x]['mean'])

    # ========================================================================
    # PASO 3: Análisis de TEXTURA
    # ========================================================================

    gray = image_denorm.mean(axis=2)  # Escala de grises

    # Textura: varianza local
    texture = generic_filter(gray, np.std, size=3)
    texture_mean = float(texture.mean())
    texture_std = float(texture.std())

    # Clasificar tipo de textura
    if texture_mean > 0.15:
        texture_type = "RUGOSA/COMPLEJA"
    elif texture_mean > 0.08:
        texture_type = "MODERADA"
    else:
        texture_type = "SUAVE/UNIFORME"

    # ========================================================================
    # PASO 4: Análisis de FORMA y CONTORNO
    # ========================================================================

    # Bordes (Sobel)
    edges_x = filters.sobel_h(gray)
    edges_y = filters.sobel_v(gray)
    edges_magnitude = np.sqrt(edges_x**2 + edges_y**2)
    edge_intensity = float(edges_magnitude.mean())

    # Clasificar complejidad de forma
    if edge_intensity > 0.15:
        shape_complexity = "COMPLEJA (muchos bordes)"
    elif edge_intensity > 0.08:
        shape_complexity = "MODERADA"
    else:
        shape_complexity = "SIMPLE (pocos bordes)"

    # Gradiente (cambios de intensidad)
    gradient = np.gradient(gray)
    gradient_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
    gradient_intensity = float(gradient_magnitude.mean())

    # ========================================================================
    # PASO 5: CORRELACIÓN con activación de neurona
    # ========================================================================

    # Redimensionar características al tamaño de la activación
    edges_resized = resize(
        edges_magnitude, activation_shape, anti_aliasing=True)
    texture_resized = resize(texture, activation_shape, anti_aliasing=True)
    gradient_resized = resize(
        gradient_magnitude, activation_shape, anti_aliasing=True)
    red_resized = resize(
        image_denorm[:, :, 0], activation_shape, anti_aliasing=True)
    green_resized = resize(
        image_denorm[:, :, 1], activation_shape, anti_aliasing=True)
    blue_resized = resize(
        image_denorm[:, :, 2], activation_shape, anti_aliasing=True)

    # Aplanar para correlación
    neuron_flat = neuron_activation.flatten()

    correlations = {
        'Bordes (Contorno)': float(pearsonr(neuron_flat, edges_resized.flatten())[0]),
        'Textura (Rugosidad)': float(pearsonr(neuron_flat, texture_resized.flatten())[0]),
        'Gradiente (Transiciones)': float(pearsonr(neuron_flat, gradient_resized.flatten())[0]),
        'Canal Rojo': float(pearsonr(neuron_flat, red_resized.flatten())[0]),
        'Canal Verde': float(pearsonr(neuron_flat, green_resized.flatten())[0]),
        'Canal Azul': float(pearsonr(neuron_flat, blue_resized.flatten())[0]),
    }

    # Ordenar por correlación absoluta
    sorted_correlations = sorted(
        correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_feature = sorted_correlations[0]

    # Determinar fuerza de correlación
    if abs(top_feature[1]) > 0.7:
        correlation_strength = "FUERTE"
    elif abs(top_feature[1]) > 0.4:
        correlation_strength = "MODERADA"
    else:
        correlation_strength = "DÉBIL"

    # ========================================================================
    # PASO 6: Compilar resultados
    # ========================================================================

    results = {
        'image_info': {
            'index': image_index,
            'class': class_name,
            'label_id': int(label)
        },
        'neuron_info': {
            'index': neuron_index,
            'layer': layer_name,
            'activation_mean': float(neuron_activation.mean()),
            'activation_max': float(neuron_activation.max()),
            'activation_shape': list(activation_shape)
        },
        'color_analysis': {
            'channels': color_analysis,
            'dominant_channel': dominant_channel
        },
        'texture_analysis': {
            'mean': texture_mean,
            'std': texture_std,
            'type': texture_type
        },
        'shape_analysis': {
            'edge_intensity': edge_intensity,
            'complexity': shape_complexity,
            'gradient_intensity': gradient_intensity
        },
        'correlations': {
            'all': correlations,
            'sorted': sorted_correlations,
            'top_feature': top_feature[0],
            'top_correlation': top_feature[1],
            'correlation_strength': correlation_strength
        },
        # Datos para visualización
        '_raw_data': {
            'image_denorm': image_denorm,
            'gray': gray,
            'edges': edges_magnitude,
            'texture': texture,
            'gradient': gradient_magnitude,
            'neuron_activation': neuron_activation,
            'edges_resized': edges_resized,
            'texture_resized': texture_resized,
            'gradient_resized': gradient_resized,
            'red_resized': red_resized,
            'green_resized': green_resized,
            'blue_resized': blue_resized
        }
    }

    # ========================================================================
    # PASO 7: Imprimir resultados (si verbose=True)
    # ========================================================================

    if verbose:
        print("="*80)
        print(
            f"🔬 ANÁLISIS DE NEURONA #{neuron_index} - IMAGEN #{image_index} ({class_name.upper()})")
        print("="*80)

        print(f"\n🎨 ANÁLISIS DE COLOR:")
        for channel, stats in color_analysis.items():
            print(
                f"   {channel:6s}: Media={stats['mean']:.3f}, Std={stats['std']:.3f}")
        print(f"   → Color dominante: {dominant_channel}")

        print(f"\n🧵 ANÁLISIS DE TEXTURA:")
        print(f"   Varianza promedio: {texture_mean:.4f} ± {texture_std:.4f}")
        print(f"   → Tipo: {texture_type}")

        print(f"\n📐 ANÁLISIS DE FORMA Y CONTORNO:")
        print(f"   Intensidad de bordes: {edge_intensity:.4f}")
        print(f"   Intensidad de gradiente: {gradient_intensity:.4f}")
        print(f"   → Complejidad: {shape_complexity}")

        print(f"\n🧠 CORRELACIÓN NEURONA vs CARACTERÍSTICAS:")
        for feature, corr in sorted_correlations:
            bar = '█' * int(abs(corr) * 20)
            sign = '+' if corr >= 0 else ''
            print(f"   {feature:25s}: {sign}{corr:6.3f}  {bar}")

        print(f"\n💡 INTERPRETACIÓN:")
        print(
            f"   La neurona #{neuron_index} tiene correlación {correlation_strength}")
        print(f"   con: {top_feature[0]} (r={top_feature[1]:.3f})")

        if top_feature[1] > 0:
            print(f"   → A mayor {top_feature[0].lower()}, MAYOR activación")
        else:
            print(f"   → A mayor {top_feature[0].lower()}, MENOR activación")

        print("="*80)

    return results

# ========================================================================
# Función auxiliar para visualización
# ========================================================================


def visualize_texture_analysis(results: dict, figsize=(20, 10)):
    """
    Visualiza los resultados del análisis de texturas.

    Args:
        results: Dict retornado por analyze_neuron_texture_and_features
        figsize: Tamaño de la figura
    """
    import matplotlib.pyplot as plt

    raw = results['_raw_data']
    corr = results['correlations']['all']
    neuron_idx = results['neuron_info']['index']
    class_name = results['image_info']['class']

    fig, axes = plt.subplots(2, 4, figsize=figsize)

    # ========== FILA 1: Características de la imagen ==========

    # Original
    axes[0, 0].imshow(raw['image_denorm'])
    axes[0, 0].set_title(
        f'Original\n{class_name.upper()}', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')

    # Bordes
    im1 = axes[0, 1].imshow(raw['edges'], cmap='hot')
    axes[0, 1].set_title('Bordes (Sobel)\nDetección de Contorno', fontsize=10)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Textura
    im2 = axes[0, 2].imshow(raw['texture'], cmap='viridis')
    axes[0, 2].set_title('Textura\n(Varianza Local)', fontsize=10)
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Gradiente
    im3 = axes[0, 3].imshow(raw['gradient'], cmap='plasma')
    axes[0, 3].set_title('Gradiente\n(Cambios de Intensidad)', fontsize=10)
    axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

    # ========== FILA 2: Características redimensionadas + Activación ==========

    # Bordes resized
    im4 = axes[1, 0].imshow(raw['edges_resized'], cmap='hot')
    axes[1, 0].set_title(
        f'Bordes (2×2)\nr={corr["Bordes (Contorno)"]:.2f}', fontsize=10)
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    # Textura resized
    im5 = axes[1, 1].imshow(raw['texture_resized'], cmap='viridis')
    axes[1, 1].set_title(
        f'Textura (2×2)\nr={corr["Textura (Rugosidad)"]:.2f}', fontsize=10)
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

    # Gradiente resized
    im6 = axes[1, 2].imshow(raw['gradient_resized'], cmap='plasma')
    axes[1, 2].set_title(
        f'Gradiente (2×2)\nr={corr["Gradiente (Transiciones)"]:.2f}', fontsize=10)
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    # Activación de neurona
    im7 = axes[1, 3].imshow(raw['neuron_activation'], cmap='YlOrRd')
    axes[1, 3].set_title(
        f'Activación\nNeurona #{neuron_idx}', fontweight='bold', fontsize=10)
    axes[1, 3].axis('off')
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046)

    plt.tight_layout()
    plt.show()

    # ========== Gráfico de correlaciones ==========

    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_corr = results['correlations']['sorted']
    features = [f[0] for f in sorted_corr]
    corr_values = [f[1] for f in sorted_corr]
    colors_bar = ['green' if c > 0 else 'red' for c in corr_values]

    ax.barh(features, corr_values, color=colors_bar,
            alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linewidth=1.5)
    ax.set_xlabel('Correlación de Pearson', fontsize=12, fontweight='bold')
    ax.set_title(f'Correlaciones: Neurona #{neuron_idx} vs Características Visuales\n(Verde=Positiva, Rojo=Negativa)',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()
