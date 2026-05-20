# Ruta de aprendizaje — Interpretabilidad en IA

Este repositorio es una carrera de estudio progresivo sobre interpretabilidad de modelos de machine learning. Está organizado en **3 ramas temáticas independientes pero complementarias**, cada una con sus propios niveles, técnicas, entregables y recursos.

Las ramas no son estrictamente secuenciales: se puede avanzar en paralelo, aunque se recomienda tener una base sólida en **Fundamentos** antes de profundizar en las otras dos.

---

## Estructura general

```
roadmap/
├── README.md                ← este archivo
├── fundamentos.md           ← Rama 1: técnicas clásicas de interpretabilidad
├── critica.md               ← Rama 2: validación y crítica de explicaciones
└── circuitos_internos.md    ← Rama 3: mechanistic interpretability
```

Cada rama tiene su propio archivo `.md` con:
- Descripción del área y por qué importa
- Niveles de profundidad (Nivel 1 → Nivel N)
- Técnicas y herramientas por nivel
- Entregables concretos
- Papers y recursos recomendados

---

## Las 3 ramas

### 🧱 Rama 1 — Fundamentos
**Archivo:** `fundamentos.md`

El punto de entrada. Cubre las técnicas establecidas de interpretabilidad aplicadas a los cuatro tipos de datos principales: tabular, texto, activaciones neuronales e imágenes.

El foco es aprender a *usar* las herramientas existentes correctamente: SHAP, LIME, GradCAM, Integrated Gradients, análisis de activaciones.

**Cuándo empezar:** desde el primer día. Es el prerequisito implícito de las otras dos ramas.

**Estado actual:** Módulos I (tabular), II (NLP) y III (activaciones) completados. Módulo IV (computer vision / GradCAM) en planificación.

---

### 🔬 Rama 2 — Crítica
**Archivo:** `critica.md`

El segundo nivel. No basta con saber aplicar SHAP o GradCAM — hay que saber cuándo sus resultados son confiables y cuándo no.

Esta rama cubre el análisis meta de las explicaciones: métricas de fidelidad, tests de sanidad, casos donde los métodos se contradicen entre sí, y cómo evaluar si una explicación realmente refleja el comportamiento del modelo o es un artefacto.

**Cuándo empezar:** después de completar al menos dos módulos de Fundamentos con diferentes tipos de datos.

**Estado actual:** no iniciada.

---

### ⚙️ Rama 3 — Circuitos internos
**Archivo:** `circuitos_internos.md`

El nivel más avanzado y el más cercano al estado del arte del campo. En lugar de explicar *outputs*, busca entender la *computación interna* del modelo: qué circuitos implementan un comportamiento, cómo se organizan las representaciones, dónde vive la información dentro de la red.

Basada en el campo emergente de **mechanistic interpretability** (Anthropic, DeepMind, Neel Nanda).

**Cuándo empezar:** después de completar Fundamentos en su totalidad y tener familiaridad con la Rama 2.

**Estado actual:** no iniciada.

---

## Mapa de progreso general

| Rama | Niveles totales | Nivel actual | Estado |
|------|----------------|--------------|--------|
| Fundamentos | 4 módulos | Módulo III (en curso) | 🟡 En progreso |
| Crítica | 4 niveles | — | ⬜ No iniciada |
| Circuitos internos | 5 niveles | — | ⬜ No iniciada |

---

## Principios del proyecto

1. **Documentar antes de codear** — cada módulo o nivel arranca con su README y plan antes de escribir código.
2. **Código modular y comentado** — cada implementación es reutilizable y explicada.
3. **Learnings capturados** — cada módulo termina con un `LEARNINGS.md` con hallazgos, sorpresas y limitaciones encontradas.
4. **Reproducibilidad** — seeds fijados, versiones de librerías especificadas.
5. **Rigor sobre velocidad** — mejor entender bien una técnica que correr diez sin comprenderlas.

---

RAMA 1 — FUNDAMENTOS
├── Módulo I   — Tabular/XGBoost        ✅ completo
├── Módulo II  — NLP/DistilBERT         ✅ completo  
├── Módulo III — Activaciones/CNN       🟡 Acá
│   ├── Steps 1-3  ✅ hecho + 2 apps + DeepDream deployado
│   └── Steps 4-7  ⬜ pendiente (Neuron Probing → cierre)
└── Módulo IV  — Saliency/GradCAM      ⬜ no iniciado

RAMA 2 — CRÍTICA                        ⬜ no iniciada
RAMA 3 — CIRCUITOS INTERNOS            ⬜ no iniciada

*Última actualización: en curso — Módulo III de Fundamentos*
