
## Parámetros entrenables
- param.requires_grad = True
- Los gradientes se calculan para estos parámetros
- Se actualizan con el optimizador

## Parámetros congelados
- param.requires_grad = False
- No se calculan gradientes (ahorra memoria y tiempo)
- No se actualizan durante el entrenamiento
- Útil para transfer learning y evitar sobreajuste
- ❄️Congelar el modelo: param.requires_grad = False

## Congelar capas:
- Lo usamos para cuando reentrenamos el modelo buscando agregar una nueva capa o modificar parte del modelo sin tener que re entrenar todo el modelo desde cero. Ahorrando tiempo de computo.
- Entrenamiento más rápido: No calculas gradientes para capas congeladas
- Menos memoria: No almacenas gradientes
- Transfer learning: Reutilizas features aprendidas en ImageNet
- Evitas overfitting: Mantienes representaciones generales

Arquitecturas de Redes Neuronales Convolucionales (CNN) comunes:

## 📊 **TABLA COMPARATIVA**

| Arquitectura | Año | Parámetros | Top-1 Acc | Innovación Clave | Cuándo usar |
|--------------|-----|------------|-----------|------------------|-------------|
| **AlexNet** | 2012 | 60M | 57% | ReLU + Dropout | Historia, no usar |
| **VGGNet** | 2014 | 138M | 71% | Muchas conv pequeñas | Transfer learning simple |
| **ResNet** | 2015 | 25M | 76% | **Shortcuts** | **Uso general (mejor opción)** |
| **Inception** | 2015 | 24M | 78% | Multi-escala | Eficiencia + precisión |
| **DenseNet** | 2017 | 20M | 77% | Conexiones densas | Feature reuse importante |
| **MobileNet** | 2017 | 4M | 70% | Depthwise conv | **Móviles/Edge devices** |
| **EfficientNet** | 2019 | 66M | 84% | Compound scaling | **Estado del arte** |
| **ViT** | 2020 | 86M | 88% | Self-attention | **Muchos datos (>10M imgs)** |


## Pesos 
- ✅ **Qué son:** Los **parámetros aprendidos** del filtro (el "ADN" del filtro)
- ✅ **Dependen de:** El entrenamiento (NO cambian con cada imagen)
- ✅ **Muestran:** Qué PATRÓN busca el filtro en CUALQUIER imagen
- ✅ **Ejemplo:** "El filtro 5 está programado para detectar bordes horizontales"
- ❌ **No muestran:** Qué activación específica produjo el filtro para UNA imagen dada

| Aspecto | PESOS (Celda 14) | ACTIVACIONES (Celda 12) |
|---------|------------------|-------------------------|
| **¿Qué es?** | Plantilla/patrón que busca | Resultado de aplicar la plantilla |
| **Cuándo se crea** | Durante entrenamiento | Durante forward pass |
| **Cambia?** | NO (fijos después de entrenar) | SÍ (cada imagen diferente) |
| **Shape** | `[64, 3, 7, 7]` | `[1, 64, 16, 16]` |
| **Muestra** | Qué busca el filtro | Qué encontró el filtro |
| **Visualización** | Colores RGB del kernel | Mapa de calor de activación |