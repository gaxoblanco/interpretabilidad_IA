# 📚 Guía de Estudio - Módulo III: Neuron Activation Analysis

## 🎯 Objetivo del Módulo
Entender **qué representan las neuronas individuales** en redes profundas y cómo visualizar/analizar sus activaciones.

---

## 📖 Temas de Estudio por Orden de Prioridad

### 🔴 **NIVEL 1: FUNDAMENTALES (Estudiar ANTES del Notebook 01)**

#### 1.1 Redes Neuronales Convolucionales (CNN)
**¿Qué estudiar?**
- Cómo funcionan las **capas convolucionales** (filtros, kernels, stride, padding)
- Concepto de **feature maps** / **activation maps**
- **Pooling layers** (max pooling, average pooling)
- **Fully connected layers** al final
- Flujo de información: entrada → conv → activación → pool → fc → salida

**Recursos:**
- [ ] Video: "CNN Explained" de StatQuest (~20 min)
- [ ] Artículo: "A Beginner's Guide to CNNs" en Medium
- [ ] Práctica: Visualizar filtros de una CNN simple

**Conceptos clave:**
- ✅ Un filtro detecta un patrón específico
- ✅ Las primeras capas detectan bordes/texturas
- ✅ Las capas profundas detectan objetos complejos
- ✅ Cada neurona tiene un "campo receptivo"

---

#### 1.2 Arquitectura ResNet
**¿Qué estudiar?**
- **Conexiones residuales** (skip connections): ¿Por qué existen?
- Problema del **vanishing gradient** que ResNet soluciona
- Estructura de **bloques residuales** (BasicBlock, Bottleneck)
- Diferencia entre ResNet-18, 50, 101, etc.

**Recursos:**
- [ ] Paper original: "Deep Residual Learning for Image Recognition" (2015) - Leer solo Sección 3
- [ ] Video: "ResNet Explained" de Yannic Kilcher (~15 min)
- [ ] Diagrama: Dibujar un bloque residual con skip connection

**Conceptos clave:**
- ✅ F(x) + x permite flujo directo de gradientes
- ✅ ResNet-18 tiene 4 "layers" con múltiples bloques cada uno
- ✅ Los bloques pueden tener "downsample" (reducir resolución)

**Pregunta para ti:**
> ¿Por qué una red de 152 capas puede entrenarse mejor con skip connections que una de 34 sin ellas?

---

#### 1.3 Forward Pass y Activaciones
**¿Qué estudiar?**
- ¿Qué es una **activación**? (output de una neurona/capa)
- Funciones de activación: **ReLU**, Sigmoid, Tanh
- Shape de las activaciones: [batch, channels, height, width]
- Diferencia entre **pre-activación** y **post-activación**

**Recursos:**
- [ ] Implementar forward pass manualmente en NumPy
- [ ] Visualizar activaciones de una capa simple

**Conceptos clave:**
- ✅ ReLU(x) = max(0, x) → introduce no-linealidad
- ✅ Las activaciones cambian con cada input
- ✅ Una neurona "se activa" cuando su output > 0

**Ejercicio:**
```python
# Si tengo una imagen 32x32x3 (CIFAR-10)
# Y aplico Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
# ¿Qué shape tiene la activación resultante?
# Respuesta: [batch, 64, 16, 16]
```

---

### 🟡 **NIVEL 2: CONCEPTOS CLAVE (Estudiar DURANTE el Notebook 01-02)**

#### 2.1 PyTorch Hooks
**¿Qué estudiar?**
- ¿Qué es un **hook** en PyTorch?
- **Forward hooks** vs **Backward hooks**
- `register_forward_hook()` - Capturar activaciones
- `register_backward_hook()` - Capturar gradientes

**Recursos:**
- [ ] PyTorch Docs: "torch.nn.Module.register_forward_hook"
- [ ] Tutorial práctico: Capturar activaciones de una capa

**Conceptos clave:**
- ✅ Un hook es una función callback que se ejecuta automáticamente
- ✅ Se "engancha" a una capa específica
- ✅ Captura activaciones sin modificar el modelo

**Ejemplo mental:**
```python
# Sin hook: output = model(input)  # No veo activaciones internas
# Con hook: output = model(input)  # Hook captura conv1, layer1, etc.
```

---

#### 2.2 Visualización de Activaciones
**¿Qué estudiar?**
- Cómo interpretar un **heatmap de activación**
- Normalización de activaciones para visualización
- Diferencia entre visualizar **pesos** vs **activaciones**
- Colormap apropiados (viridis, jet, etc.)

**Recursos:**
- [ ] Ejemplos de visualizaciones en papers de interpretabilidad
- [ ] Matplotlib: imshow() para visualizar matrices

**Conceptos clave:**
- ✅ Valores altos (brillantes) = neurona muy activada
- ✅ Valores bajos/cero (oscuros) = neurona inactiva
- ✅ Cada filtro produce un "mapa" diferente

---

#### 2.3 Estadísticas de Activaciones
**¿Qué estudiar?**
- **Sparsity**: % de neuronas con activación = 0
- **Dead neurons**: Neuronas que nunca se activan
- **Mean, Std, Min, Max** de activaciones
- Distribuciones de activaciones (histogramas)

**Conceptos clave:**
- ✅ Alta sparsity = muchas neuronas inactivas (normal con ReLU)
- ✅ Dead neurons = posible problema de entrenamiento
- ✅ Estadísticas varían por capa (primeras capas más densas)

---

### 🟢 **NIVEL 3: TÉCNICAS AVANZADAS (Estudiar ANTES del Notebook 03-05)**

#### 3.1 Feature Visualization
**¿Qué estudiar?**
- **Activation Maximization**: Optimizar input para maximizar activación de neurona
- **DeepDream**: Variante de activation maximization
- Regularización en feature visualization (suavizado, prior natural)
- Técnicas de optimización (gradient ascent)

**Recursos:**
- [ ] Paper: "Feature Visualization" (Olah et al., 2017) - Distill.pub
- [ ] Video: "Deep Visualization" de Two Minute Papers
- [ ] Implementación: Lucid library o Captum

**Conceptos clave:**
- ✅ Crear una imagen que "engañe" a una neurona
- ✅ Muestra qué patrón busca esa neurona
- ✅ Requiere optimización iterativa

**Pregunta:**
> Si maximizo la activación del filtro 23 en conv1, ¿qué tipo de patrón esperarías ver?
> Respuesta: Probablemente bordes o texturas simples

---

#### 3.2 Neuron Probing
**¿Qué estudiar?**
- **Probing classifiers**: Entrenar clasificador lineal sobre activaciones
- ¿Qué información codifica cada capa?
- Concepto de **emergencia** de conceptos en capas profundas
- Análisis de representaciones aprendidas

**Recursos:**
- [ ] Paper: "What do you learn from context?" (Peters et al., 2018)
- [ ] Paper: "Network Dissection" (Bau et al., 2017)

**Conceptos clave:**
- ✅ Si un clasificador lineal puede predecir "color" desde layer1, esa capa codifica color
- ✅ Capas tempranas = features simples
- ✅ Capas profundas = conceptos abstractos

**Ejercicio mental:**
> Si entreno un clasificador para predecir "¿es un gato?" usando activaciones de layer4, ¿tendrá mejor accuracy que usando conv1?
> Respuesta: Sí, porque layer4 tiene representaciones de alto nivel

---

#### 3.3 Activation Patterns y Class Activation Maps
**¿Qué estudiar?**
- **GradCAM** (Gradient-weighted Class Activation Mapping)
- Diferencia entre visualizar activaciones vs importancia
- Mapas de calor de atención
- Regiones de la imagen que activan cada neurona

**Recursos:**
- [ ] Paper: "Grad-CAM" (Selvaraju et al., 2017)
- [ ] Tutorial: Implementar GradCAM en PyTorch

**Conceptos clave:**
- ✅ GradCAM muestra "dónde mira" el modelo
- ✅ Combina gradientes + activaciones
- ✅ Útil para debugging de modelos

---

## 🧪 Conceptos Matemáticos Necesarios

### Álgebra Lineal Básica
- [ ] Multiplicación de matrices
- [ ] Convolución 2D (operación matemática)
- [ ] Broadcasting en NumPy/PyTorch

### Cálculo
- [ ] Derivadas parciales (para entender backprop)
- [ ] Gradiente de una función
- [ ] Gradient ascent vs descent

### Estadística
- [ ] Media, varianza, desviación estándar
- [ ] Distribuciones (normal, uniforme)
- [ ] Correlación

---

## 📝 Checklist de Preparación por Notebook

### ✅ Antes del Notebook 01:
- [ ] Entender CNNs básicas
- [ ] Conocer arquitectura ResNet
- [ ] Saber qué es una activación
- [ ] Concepto de forward pass
- [ ] Familiaridad con PyTorch básico

### ✅ Antes del Notebook 02:
- [ ] PyTorch hooks
- [ ] Estadísticas de activaciones (mean, std, sparsity)
- [ ] Visualización de heatmaps

### ✅ Antes del Notebook 03:
- [ ] Feature visualization
- [ ] Activation maximization
- [ ] Gradient ascent

### ✅ Antes del Notebook 04:
- [ ] Neuron probing
- [ ] Clasificadores lineales
- [ ] Representaciones aprendidas

### ✅ Antes del Notebook 05:
- [ ] GradCAM
- [ ] Class activation maps
- [ ] Integración de todas las técnicas

---

## 🎓 Recursos Recomendados (Orden de Prioridad)

### Videos (Más Rápido)
1. **StatQuest**: "Neural Networks Explained" (~30 min)
2. **3Blue1Brown**: "What is a neural network?" (~20 min)
3. **Yannic Kilcher**: "ResNet Explained" (~15 min)
4. **Two Minute Papers**: "Deep Visualization" (~5 min)

### Artículos (Profundidad Media)
1. **Distill.pub**: "Feature Visualization" ⭐⭐⭐⭐⭐
2. **Distill.pub**: "The Building Blocks of Interpretability"
3. **CS231n**: Lecture Notes on CNNs
4. **PyTorch Docs**: Hooks Tutorial

### Papers (Más Profundo)
1. **ResNet**: "Deep Residual Learning" (2015) - Sección 3 solamente
2. **Network Dissection** (Bau et al., 2017) - Introducción + Figuras
3. **Grad-CAM** (Selvaraju et al., 2017) - Metodología

---

## 💡 Consejos de Estudio

### Estrategia 80/20:
- **80% práctica** (ejecutar código, modificar, experimentar)
- **20% teoría** (leer papers, ver videos)

### Plan de 3 Días:
**Día 1 (2 horas):**
- Ver videos sobre CNNs y ResNet
- Leer artículo de Distill.pub sobre Feature Visualization
- Ejecutar Notebook 01 celdas 1-7

**Día 2 (2 horas):**
- Estudiar PyTorch hooks (tutorial + ejemplos)
- Ejecutar Notebook 01 celdas 8-15
- Experimentar con diferentes capas

**Día 3 (2 horas):**
- Revisar conceptos de estadísticas de activaciones
- Analizar resultados del Notebook 01
- Documentar hallazgos en LEARNINGS.md

---

## ❓ Preguntas de Auto-Evaluación

### Nivel Básico:
1. ¿Qué es un filtro convolucional y qué detecta?
2. ¿Por qué ResNet usa skip connections?
3. ¿Qué es una activación en una red neuronal?
4. ¿Qué hace la función ReLU?

### Nivel Intermedio:
1. ¿Cómo funciona un forward hook en PyTorch?
2. ¿Qué significa que una capa tenga 50% de sparsity?
3. ¿Por qué las primeras capas detectan bordes y las últimas objetos?
4. ¿Cómo interpretar un heatmap de activaciones?

### Nivel Avanzado:
1. ¿Cómo generar una imagen que maximice una neurona específica?
2. ¿Qué revela un probing classifier sobre una capa?
3. ¿Cuál es la diferencia entre visualizar pesos vs activaciones?
4. ¿Cómo funciona GradCAM internamente?

---

## 🎯 Resultado Esperado

Después de estudiar estos temas, deberías poder:

✅ Explicar qué hace cada capa de ResNet  
✅ Capturar activaciones de cualquier capa  
✅ Interpretar visualizaciones de activaciones  
✅ Identificar qué detecta una neurona específica  
✅ Analizar estadísticas de activaciones  
✅ Implementar técnicas básicas de feature visualization  
✅ Diseñar experimentos para entender qué aprendió tu modelo  

---

## 📌 TL;DR - Mínimo Necesario

Si solo tienes **1 hora**, estudia:
1. ✅ CNNs básicas (qué es un filtro)
2. ✅ ResNet (skip connections)
3. ✅ Forward pass y activaciones
4. ✅ PyTorch hooks (cómo capturar activaciones)

Esto es suficiente para empezar con el Notebook 01.

---

**¿Listo para comenzar? 🚀**

Siguiente paso: Ejecutar `python verify_setup.py` y abrir el Notebook 01.