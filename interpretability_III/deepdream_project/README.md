---
title: DeepDream Explorer
emoji: 🧠
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# 🧠 DeepDream Explorer

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

Dashboard interactivo para explorar el algoritmo **DeepDream** usando PyTorch.
Visualizá qué aprendió cada capa de InceptionV3 y AlexNet amplificando
los patrones que las redes neuronales detectan en tus imágenes.

---

## 🎯 ¿Qué es DeepDream?

DeepDream es una técnica de **visualización de redes neuronales** desarrollada
por Google en 2015. En lugar de entrenar el modelo, usa **gradient ascent**
para modificar una imagen de forma que maximice las activaciones de una capa.

**Resultado:** la red "amplifica" lo que ya detectaba, generando patrones
psicodélicos que revelan qué aprendió cada capa durante su entrenamiento.

### Comparado con entrenamiento normal:

| Entrenamiento | DeepDream |
|---|---|
| Gradient **descent** | Gradient **ascent** |
| Minimizar pérdida | Maximizar activaciones |
| Modificar **pesos** | Modificar **imagen** |
| Los pesos cambian | Los pesos son fijos |

---

## 🚀 Modos de uso

### ⚡ Preview (rápido)
5 iteraciones × 3 octavas. Feedback visual en ~15-20 segundos en CPU.
Ideal para explorar combinaciones de modelos y capas antes del proceso completo.

### 🎨 Completo
Parámetros completos con progreso visual octava a octava.
Al finalizar: comparación side-by-side original vs resultado + descarga PNG.

### 🔬 Comparar 3 capas
Ejecuta DeepDream con los mismos parámetros en 3 capas diferentes simultáneamente.
Muestra cómo cambian los patrones según la profundidad de la capa en la red.

---

## 🔬 Conceptos implementados

### Pirámide de octavas (multi-escala)
```
Imagen original → ×(1/1.4) → ×(1/1.4)² → ...
                                            ↓ DeepDream
← resultado final ← upscale + detalle ← resultado pequeño
```
Aplicamos DeepDream a múltiples resoluciones. Los patrones aparecen a varias
escalas → el efecto "fractal" característico de DeepDream.

### Gradient Ascent con Jitter
```python
# ASCENSO (no descenso): maximizar activaciones
imagen = imagen + lr * gradiente

# Jitter: desplazamiento aleatorio antes de cada paso
# Reduce artefactos de alta frecuencia
imagen_shifted = np.roll(imagen, jitter_x, axis=1)
```

### Hook de PyTorch para captura de activaciones
```python
class CapturaActivaciones:
    def __call__(self, modulo, input, output):
        self.activaciones = output  # ← espiar activaciones sin modificar el modelo

hook = capa.register_forward_hook(captura)
modelo(imagen)
activaciones = captura.activaciones
hook.remove()  # SIEMPRE limpiar
```

---

## 📁 Estructura del proyecto

```
deepdream_space/
├── app.py                 # Dashboard Streamlit — solo UI
├── deepdream_engine.py    # Algoritmo DeepDream — sin UI
├── requirements.txt       # Dependencias pinneadas (CPU)
├── Dockerfile             # Para HF Spaces con Docker
├── README.md              # Este archivo
└── assets/
    └── ejemplo.jpg        # Imagen de ejemplo incluida
```

**Separación de responsabilidades:**
- `deepdream_engine.py` — algoritmo puro, testeable independientemente
- `app.py` — UI que importa del engine, sin lógica de DeepDream

---

## ⚙️ Modelos y capas

### InceptionV3
| Capa | Detecta |
|---|---|
| Mixed_5b | Bordes y formas geométricas simples |
| Mixed_5d | Formas básicas y contornos |
| Mixed_6e | Objetos reconocibles (**recomendada**) |
| Mixed_7c | Máxima abstracción |

### AlexNet
| Capa | Detecta |
|---|---|
| features.0 | Bordes y colores crudos |
| features.6 | Patrones y mosaicos |
| features.10 | Objetos de alto nivel (**recomendada**) |

---

## 🛠️ Instalación local

```bash
# Clonar el repo
git clone https://huggingface.co/spaces/TU_USUARIO/deepdream-explorer
cd deepdream-explorer

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
streamlit run app.py
```

### Con Docker
```bash
docker build -t deepdream-space .
docker run -p 7860:7860 deepdream-space
# Abrir http://localhost:7860
```

---

## 📚 Referencias

- [Inceptionism: Going Deeper into Neural Networks](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) — Google (2015)
- [Feature Visualization](https://distill.pub/2017/feature-visualization/) — Distill.pub (2017)
- [PyTorch Hooks](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html)

---

## 📝 Licencia

MIT — libre para usar, modificar y distribuir.
