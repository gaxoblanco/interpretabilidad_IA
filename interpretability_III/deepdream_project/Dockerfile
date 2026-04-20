# ================================================================
# DOCKERFILE — Contenedor para Hugging Face Spaces
# ================================================================
#
# ROL:
#   Define el entorno de ejecución completo del dashboard.
#   HF Spaces con Docker usa este archivo para construir y
#   lanzar la aplicación automáticamente al hacer push.
#
# FORMATO REQUERIDO POR HF SPACES:
#   - Puerto: 7860 (obligatorio para HF Spaces con Docker)
#   - CMD debe lanzar el servidor en 0.0.0.0 (no localhost)
#
# FLUJO DE BUILD:
#   1. Imagen base: python:3.10-slim (liviana, sin extras innecesarios)
#   2. Instalar dependencias del sistema (libGL para PIL)
#   3. Instalar dependencias Python desde requirements.txt
#   4. Copiar código fuente
#   5. Lanzar Streamlit en el puerto 7860
#
# USO LOCAL (para probar antes de subir a HF):
#   docker build -t deepdream-space .
#   docker run -p 7860:7860 deepdream-space
#   → Abrir http://localhost:7860
#
# ================================================================

# Imagen base: slim para reducir tamaño final del contenedor
FROM python:3.12.12-slim-bookworm

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# ---- Dependencias del sistema ----
# libGL y libglib son requeridas por OpenCV/PIL para operaciones de imagen
# En imágenes slim estas librerías no vienen incluidas
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# rm -rf limpia la caché de apt → reduce tamaño de imagen final

# ---- Dependencias Python ----
# Copiamos requirements.txt ANTES del código fuente.
# Esto permite que Docker cachee la capa de pip install:
# si solo cambia el código (no las dependencias), no re-instala todo.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir: no guarda caché de pip → reduce tamaño de imagen

# ---- Código fuente ----
# Copiar todo el proyecto al contenedor
COPY . .

# ---- Crear directorio de assets si no existe ----
# (en caso de que ejemplo.jpg no esté en el repo por .gitignore)
RUN mkdir -p assets

# Puerto que expone el contenedor (requerido por HF Spaces)
EXPOSE 7860

# ---- Comando de inicio ----
# --server.port=7860 : puerto requerido por HF Spaces
# --server.address=0.0.0.0 : escuchar en todas las interfaces
#                             (necesario en Docker, no solo localhost)
# --server.headless=true : no abrir browser automáticamente
CMD ["streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.runOnSave=true"]