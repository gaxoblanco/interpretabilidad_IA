"""
================================================================
APP.PY — Dashboard Streamlit de DeepDream
================================================================

ROL EN LA ARQUITECTURA:
    Este archivo contiene ÚNICAMENTE la lógica de interfaz de usuario.
    No implementa ningún algoritmo de DeepDream — eso está en
    deepdream_engine.py.

    Responsabilidades de app.py:
      - Renderizar la sidebar con controles
      - Gestionar session_state para persistir resultados
      - Llamar a deepdream_engine con los parámetros configurados
      - Mostrar progreso, comparaciones y botones de descarga

    Patrón de importación:
      from deepdream_engine import (
          cargar_modelo_interno, preprocesar_imagen,
          deepdream_universal, numpy_a_pil,
          CAPAS_POR_MODELO, TAMAÑOS_MODELO, INTENSIDADES
      )

MODOS DE EJECUCIÓN:
    - Preview  : 5 iter, 3 octavas. Feedback rápido (~15s en CPU)
    - Completo : parámetros completos, progreso por octava, descarga
    - Comparar : misma imagen en 3 capas distintas simultáneamente

================================================================
"""

import io
import requests
import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# ---- Importar el engine (lógica pura, sin UI) ----
from deepdream_engine import (
    cargar_modelo_interno,
    preprocesar_imagen,
    deepdream_universal,
    numpy_a_pil,
    CAPAS_POR_MODELO,
    TAMAÑOS_MODELO,
    INTENSIDADES,
)


# ================================================================
# CONFIGURACIÓN GLOBAL DE PÁGINA
# ================================================================

st.set_page_config(
    page_title="DeepDream Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# CACHÉ DE MODELOS
# ================================================================

@st.cache_resource
def cargar_modelo(nombre: str):
    """
    Carga el modelo una sola vez por sesión de Streamlit.

    ¿POR QUÉ @st.cache_resource?
        InceptionV3 pesa ~100MB, AlexNet ~230MB.
        Sin caché, cada vez que el usuario mueve un slider
        Streamlit re-ejecuta el script completo y recarga
        los pesos desde cero → insoportablemente lento.
    """
    with st.spinner(f"Cargando modelo {nombre}... (primera vez puede tardar)"):
        return cargar_modelo_interno(nombre)


# ================================================================
# INICIALIZACIÓN DEL SESSION STATE
# ================================================================

def init_session_state():
    """
    Inicializa las variables del session state con valores por defecto.

    ¿QUÉ ES session_state EN STREAMLIT?
        Streamlit re-ejecuta el script completo en cada interacción.
        st.session_state es un diccionario persistente entre re-runs
        que nos permite "recordar" cosas entre ejecuciones.
    """
    defaults = {
        "ultimo_resultado":    None,
        "ultimo_original":     None,
        "imagen_seleccionada": "🌄 Paisaje",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ================================================================
# SIDEBAR
# ================================================================

def render_sidebar() -> dict:
    """
    Renderiza todos los controles en la sidebar y retorna la configuración.
    """
    st.sidebar.title("🧠 DeepDream Explorer")
    st.sidebar.markdown("---")

    # ------------------------------------------------------------------
    # MODELO
    # ------------------------------------------------------------------
    st.sidebar.subheader("📦 Modelo")
    modelo_nombre = st.sidebar.radio(
        "Red neuronal:",
        options=["inception", "alexnet"],
        format_func=lambda x: "InceptionV3 (complejo)" if x == "inception" else "AlexNet (simple)",
        help="InceptionV3 produce patrones más orgánicos. AlexNet es más rápido.",
    )
    st.sidebar.markdown("---")

    # ------------------------------------------------------------------
    # SELECCIÓN DE IMAGEN
    # ------------------------------------------------------------------
    st.sidebar.subheader("🖼️ Imagen de entrada")

    if "fuente_imagen" not in st.session_state:
        st.session_state["fuente_imagen"] = "ejemplo"

    fuente = st.sidebar.radio(
        "Fuente:",
        options=["ejemplo", "url", "subir"],
        format_func=lambda x: {
            "ejemplo": "🖼️ Ejemplos",
            "url":     "🔗 URL",
            "subir":   "📁 Subir archivo",
        }[x],
        key="fuente_imagen",
        horizontal=True,
    )

    if fuente == "ejemplo":
        EJEMPLOS = {
            "🌄 Paisaje":       "https://picsum.photos/seed/macro65/600/400",
            "🌿 Bosque difuso": "https://picsum.photos/seed/macro59/600/400",
            "🌿 Zoom Plantas": "https://picsum.photos/seed/macro57/600/400",
        }
        seleccion = st.sidebar.selectbox("Elegí una:", list(EJEMPLOS.keys()))
        url = EJEMPLOS[seleccion]
        try:
            resp = requests.get(url, timeout=10)
            imagen_pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
            st.sidebar.image(imagen_pil, use_column_width=True)
        except Exception:
            st.sidebar.error("No se pudo cargar la imagen de ejemplo.")
            imagen_pil = _crear_imagen_ejemplo()

    elif fuente == "url":
        url_input = st.sidebar.text_input(
            "URL de la imagen:",
            placeholder="https://ejemplo.com/imagen.jpg",
        )
        if url_input:
            try:
                resp = requests.get(url_input, timeout=10)
                imagen_pil = Image.open(
                    io.BytesIO(resp.content)).convert("RGB")
                st.sidebar.image(
                    imagen_pil, caption="✓ Cargada", use_column_width=True)
            except Exception:
                st.sidebar.error("No se pudo cargar. Verificá la URL.")
                imagen_pil = _crear_imagen_ejemplo()
        else:
            st.sidebar.info("Pegá una URL de imagen JPG o PNG")
            imagen_pil = _crear_imagen_ejemplo()

    else:  # subir
        archivo = st.sidebar.file_uploader(
            "JPG o PNG:",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if archivo is not None:
            imagen_pil = Image.open(archivo).convert("RGB")
            st.sidebar.image(imagen_pil, caption="✓ Cargada",
                             use_column_width=True)
        else:
            st.sidebar.info("Subí una imagen para empezar")
            imagen_pil = _crear_imagen_ejemplo()
    st.sidebar.markdown("---")

    # ------------------------------------------------------------------
    # CAPA OBJETIVO
    # ------------------------------------------------------------------
    st.sidebar.subheader("🔬 Capa objetivo")
    capas_disponibles = CAPAS_POR_MODELO[modelo_nombre]
    capa_default = "Mixed_6e" if modelo_nombre == "inception" else "features.10"

    capa_seleccionada = st.sidebar.selectbox(
        "Capa:",
        options=list(capas_disponibles.keys()),
        index=list(capas_disponibles.keys()).index(capa_default),
        format_func=lambda c: f"{c} — {capas_disponibles[c]}",
        help="Capas tempranas → texturas. Capas profundas → objetos complejos.",
    )
    st.sidebar.info(f"🔍 {capas_disponibles[capa_seleccionada]}")

    # ------------------------------------------------------------------
    # INTENSIDAD
    # ------------------------------------------------------------------
    st.sidebar.subheader("⚡ Intensidad")
    intensidad_nombre = st.sidebar.radio(
        "Nivel:",
        options=list(INTENSIDADES.keys()),
        index=1,
        help="Determina el learning rate y el número de iteraciones por octava.",
    )
    config_intensidad = INTENSIDADES[intensidad_nombre]
    iterations = config_intensidad["iterations"]
    # lr depende del modelo: Inception=0.04, AlexNet=0.03 (igual que notebook)
    lr = 0.04 if modelo_nombre == "inception" else 0.03

    if intensidad_nombre == "Extremo" and modelo_nombre == "inception":
        st.sidebar.warning(
            "⚠️ **Tiempo estimado:** 4-8 minutos en CPU.\n"
            "Considerá usar 'Intenso' para resultados similares más rápido."
        )

    # ------------------------------------------------------------------
    # PIRÁMIDE DE OCTAVAS
    # ------------------------------------------------------------------
    st.sidebar.subheader("🔭 Pirámide de octavas")
    num_octavas = st.sidebar.slider(
        "Número de octavas:",
        min_value=3,
        max_value=8,
        value=4,
        help="Más octavas = patrones a más escalas (efecto fractal), pero más lento.",
    )

    # ------------------------------------------------------------------
    # MODO DE EJECUCIÓN
    # ------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Modo de ejecución")
    modo = st.sidebar.radio(
        "Modo:",
        options=["preview", "completo", "comparar"],
        format_func=lambda m: {
            "preview":  "⚡ Preview (rápido, ~15s)",
            "completo": "🎨 Completo (con progreso)",
            "comparar": "🔬 Comparar 3 capas",
        }[m],
    )

    # ------------------------------------------------------------------
    # SELECTOR DE CAPAS PARA MODO COMPARAR
    # ------------------------------------------------------------------
    capas_comparar = None
    if modo == "comparar":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔬 Capas a comparar")
        capas_modelo = list(CAPAS_POR_MODELO[modelo_nombre].keys())
        descripciones = CAPAS_POR_MODELO[modelo_nombre]
        n = len(capas_modelo)
        defaults = [capas_modelo[0], capas_modelo[n // 2], capas_modelo[-1]]

        def fmt(c): return f"{c} — {descripciones[c]}"

        c1 = st.sidebar.selectbox("Capa 1 (temprana):", capas_modelo,
                                  index=capas_modelo.index(defaults[0]),
                                  format_func=fmt, key="capa_comparar_1")
        c2 = st.sidebar.selectbox("Capa 2 (media):", capas_modelo,
                                  index=capas_modelo.index(defaults[1]),
                                  format_func=fmt, key="capa_comparar_2")
        c3 = st.sidebar.selectbox("Capa 3 (profunda):", capas_modelo,
                                  index=capas_modelo.index(defaults[-1]),
                                  format_func=fmt, key="capa_comparar_3")
        capas_comparar = [c1, c2, c3]

    st.sidebar.markdown("---")

    return {
        "modelo_nombre":  modelo_nombre,
        "capa":           capa_seleccionada,
        "lr":             lr,
        "iterations":     iterations,
        "num_octavas":    num_octavas,
        "scale_factor":   1.3,   # igual que notebook original
        "modo":           modo,
        "imagen_pil":     imagen_pil,
        "capas_comparar": capas_comparar,
    }


# ================================================================
# MODOS DE EJECUCIÓN
# ================================================================

def modo_preview(imagen_pil: Image.Image, modelo, config: dict):
    """
    Ejecuta DeepDream en modo rápido: 3 octavas × 5 iteraciones.
    Objetivo: feedback visual en ~15-20 segundos en CPU.
    """
    st.subheader("⚡ Preview — Resultado rápido")
    st.caption(
        "3 octavas × 5 iteraciones. Para el resultado completo, usá Modo Completo.")

    config_preview = {**config, "iterations": 5, "num_octavas": 3}

    tamaño = TAMAÑOS_MODELO[config["modelo_nombre"]]
    img_tensor = preprocesar_imagen(imagen_pil, tamaño)

    with st.spinner("Procesando preview..."):
        resultado = deepdream_universal(
            img_tensor=img_tensor,
            modelo=modelo,
            nombre_capa=config["capa"],
            config=config_preview,
            device=DEVICE,
            callback=None,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.image(imagen_pil, caption="Original", use_column_width=True)
    with col2:
        st.image(resultado, caption="DeepDream (Preview)",
                 use_column_width=True)


def modo_completo(imagen_pil: Image.Image, modelo, config: dict):
    """
    Ejecuta DeepDream con parámetros completos y muestra progreso por octava.
    La imagen original se muestra fija a la izquierda durante todo el proceso.
    """
    st.subheader("🎨 Modo Completo")

    tamaño = TAMAÑOS_MODELO[config["modelo_nombre"]]
    img_tensor = preprocesar_imagen(imagen_pil, tamaño)

    # Dos columnas: original fija izquierda, progreso/resultado derecha
    col_orig, col_dream = st.columns(2)

    with col_orig:
        st.markdown("**Original**")
        st.image(imagen_pil, use_column_width=True)

    with col_dream:
        st.markdown("**DeepDream**")
        placeholder_progreso = st.empty()
        placeholder_progreso.info("⏳ Procesando...")

    barra_progreso = st.progress(0)
    texto_estado = st.empty()

    def actualizar_progreso(octava_actual: int, total_octavas: int, img_parcial: Image.Image):
        porcentaje = int((octava_actual / total_octavas) * 100)
        barra_progreso.progress(porcentaje)
        texto_estado.text(
            f"Octava {octava_actual} / {total_octavas} completada")
        with col_dream:
            placeholder_progreso.image(
                img_parcial,
                caption=f"Octava {octava_actual}/{total_octavas}",
                use_column_width=True,
            )

    config_completo = {k: config[k] for k in
                       ["lr", "iterations", "num_octavas", "scale_factor"]}

    with st.spinner("Procesando..."):
        resultado = deepdream_universal(
            img_tensor=img_tensor,
            modelo=modelo,
            nombre_capa=config["capa"],
            config=config_completo,
            device=DEVICE,
            callback=actualizar_progreso,
        )

    barra_progreso.empty()
    texto_estado.empty()
    placeholder_progreso.image(
        resultado, caption="Resultado final ✓", use_column_width=True)

    st.session_state["ultimo_resultado"] = resultado
    st.session_state["ultimo_original"] = imagen_pil

    _mostrar_boton_descarga(
        resultado, nombre_archivo="deepdream_resultado.png")


def modo_comparar(imagen_pil: Image.Image, modelo, config: dict):
    """
    Ejecuta DeepDream en 3 capas distintas y muestra los resultados en columnas.
    Objetivo educativo: ver cómo la profundidad de la capa cambia los patrones.
    """
    st.subheader("🔬 Comparar 3 Capas")
    st.caption(
        "Mismos parámetros, distinta capa. "
        "Observá cómo la profundidad cambia el tipo de patrón generado."
    )

    descripciones = CAPAS_POR_MODELO[config["modelo_nombre"]]
    capas_elegidas = config["capas_comparar"]
    tamaño = TAMAÑOS_MODELO[config["modelo_nombre"]]
    img_tensor = preprocesar_imagen(imagen_pil, tamaño)

    config_base = {k: config[k] for k in
                   ["lr", "iterations", "num_octavas", "scale_factor"]}

    cols = st.columns(3)
    resultados = []

    for i, capa in enumerate(capas_elegidas):
        with cols[i]:
            st.markdown(f"**{capa}**")
            st.caption(descripciones[capa])
            placeholder = st.empty()
            placeholder.info("⏳ Procesando...")

        resultado = deepdream_universal(
            img_tensor=img_tensor,
            modelo=modelo,
            nombre_capa=capa,
            config=config_base,
            device=DEVICE,
            callback=None,
        )
        resultados.append(resultado)

        with cols[i]:
            placeholder.image(resultado, use_column_width=True)

    st.markdown("---")
    st.markdown("**Descargar resultados:**")
    cols_dl = st.columns(3)
    for i, (capa, resultado) in enumerate(zip(capas_elegidas, resultados)):
        with cols_dl[i]:
            _mostrar_boton_descarga(
                resultado,
                nombre_archivo=f"deepdream_{capa.replace('.', '_')}.png",
                label=f"⬇️ {capa}",
            )


# ================================================================
# COMPONENTES REUTILIZABLES
# ================================================================

def mostrar_comparacion(original: Image.Image, resultado: Image.Image, capa: str):
    st.markdown("---")
    st.subheader("📊 Comparación")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(original, use_column_width=True)
    with col2:
        st.markdown(f"**DeepDream** — capa `{capa}`")
        st.image(resultado, use_column_width=True)


def _mostrar_boton_descarga(imagen_pil, nombre_archivo="deepdream.png",
                            label="⬇️ Descargar resultado (PNG)"):
    """
    Descarga en PNG usando BytesIO como buffer en memoria.
    st.download_button espera bytes o file-like object.
    """
    buffer = io.BytesIO()
    imagen_pil.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button(label=label, data=buffer,
                       file_name=nombre_archivo, mime="image/png")


def _crear_imagen_ejemplo() -> Image.Image:
    """Degradado colorido de fallback si no hay conexión ni assets."""
    ancho, alto = 400, 400
    arr = np.zeros((alto, ancho, 3), dtype=np.uint8)
    for y in range(alto):
        for x in range(ancho):
            arr[y, x, 0] = int(255 * y / alto)
            arr[y, x, 1] = int(255 * x / ancho)
            arr[y, x, 2] = int(255 * (x + y) / (ancho + alto))
    return Image.fromarray(arr)


# ================================================================
# MAIN
# ================================================================

def main():
    init_session_state()

    # ── 1. Título e info educativa — se muestra ANTES de cargar el modelo ──
    st.title("🧠 DeepDream Explorer")
    st.markdown(
        "Tomás una foto, la pasás por una red neuronal, y en lugar de clasificarla "
        "la hacés **soñar**: la red amplifica lo que ya detectaba, generando patrones "
        "alucinógenos que revelan qué aprendió cada capa."
    )

    with st.expander("📖 Cómo leer los resultados", expanded=False):
        st.markdown("""
            **La imagen de entrada importa** — DeepDream amplifica lo que *ya existe* en la foto:

            | Si la imagen tiene... | DeepDream genera... |
            |---|---|
            | 🌤️ Cielo / nubes | Torres, pagodas, estructuras verticales |
            | 🌿 Hojas / vegetación | Animales, insectos, ojos |
            | 🪨 Rocas / terreno | Edificios, arcos, formas geométricas |

            **El modelo cambia el estilo:**
            - **InceptionV3** → máxima variedad, el efecto psicodélico clásico de Google
            - **AlexNet** → más rápido, patrones más repetitivos (sesgo a gatos y perros)
                    
            **La capa elegida define el tipo de patrón:**
            - Capas tempranas → texturas y bordes simples
            - Capas intermedias → partes de objetos (ojos, plumas, escamas)
            - Capas profundas → objetos completos (perros, edificios, caras)

            **Las intensidades en la práctica:**
            - **Micro** → casi imperceptible, la imagen original domina. Útil para ver qué detecta la capa sin distorsionar.
            - **Normal** → el balance clásico entre imagen original y sueño
            - **Extremo** → la red "gana" sobre la imagen. Puede tardar 5+ minutos en CPU.
                    
            **Las octavas controlan la escala de los patrones:**

            Antes de soñar, la imagen se reduce N veces formando una pirámide.
            DeepDream procesa de la más pequeña a la original — cada escala
            aporta patrones de distinto tamaño que se acumulan en el resultado final.

            | Octavas | Efecto |
            |---|---|
            | 3 | Patrones grandes y simples — rápido |
            | 5 | Balance entre detalle y estructura — recomendado |
            | 8 | Patrones a múltiples escalas, efecto fractal completo — lento |

            Más octavas = más niveles de detalle simultáneos, no más intensidad.
            La intensidad la controla el nivel (Micro → Extremo).
                    
        """)

    st.markdown("---")

    # ── 2. Sidebar + carga del modelo ──────────────────────────────────────
    # El usuario ya tiene contenido para leer mientras esto carga
    config = render_sidebar()
    modelo = cargar_modelo(config["modelo_nombre"])

    # ── 3. Estado del sistema + controles ──────────────────────────────────
    device_str = "🖥️ GPU" if DEVICE.type == "cuda" else "💻 CPU"
    st.caption(
        f"{device_str} | Modelo: `{config['modelo_nombre']}` | Capa: `{config['capa']}`")

    boton_label = {
        "preview":  "⚡ Ejecutar Preview",
        "completo": "🎨 Ejecutar DeepDream Completo",
        "comparar": "🔬 Comparar 3 Capas",
    }[config["modo"]]

    if st.button(boton_label, type="primary", use_container_width=True):
        imagen = config["imagen_pil"]
        modo = config["modo"]

        if modo == "preview":
            modo_preview(imagen, modelo, config)
        elif modo == "completo":
            modo_completo(imagen, modelo, config)
        elif modo == "comparar":
            modo_comparar(imagen, modelo, config)

    elif st.session_state["ultimo_resultado"] is not None:
        st.info("💡 Último resultado. Presioná el botón para generar uno nuevo.")
        mostrar_comparacion(
            st.session_state["ultimo_original"],
            st.session_state["ultimo_resultado"],
            config["capa"],
        )
        _mostrar_boton_descarga(st.session_state["ultimo_resultado"],
                                nombre_archivo="deepdream_ultimo.png")
    else:
        st.info("👈 Configurá los parámetros en la sidebar y presioná el botón.")
        tamaño = TAMAÑOS_MODELO[config["modelo_nombre"]]
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(config["imagen_pil"], use_column_width=True)
            st.caption(
                f"Se redimensionará a {tamaño[0]}×{tamaño[1]}px antes de procesar")


if __name__ == "__main__":
    main()
