"""
================================================================
DEEPDREAM_ENGINE.PY — Motor de DeepDream (Lógica Pura)
================================================================

Implementacion fiel al notebook deepdream_tutorial_universal.ipynb.

Diferencias clave respecto a implementaciones comunes:
  - La imagen se mantiene en rango [0, 1] SIN normalizacion ImageNet.
    El notebook usa solo ToTensor() — normalizar con mean/std distorsiona
    los gradientes y produce resultados incorrectos.
  - La piramide usa tensores y F.interpolate (no PIL resize en numpy).
  - El blur se aplica cada 4 iteraciones dentro del loop (no post-octava).
  - El detalle entre octavas se acumula en tensor y se suma a img_base.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps


# ----------------------------------------------------------------
# CONSTANTES
# ----------------------------------------------------------------

# Minimo de pixeles en la piramide (evita RuntimeError de kernel)
MIN_SIZE = 75

TAMAÑOS_MODELO = {
    "inception": (299, 299),
    "alexnet":   (512, 512),
}

CAPAS_POR_MODELO = {
    "inception": {
        "Mixed_5b": "Bordes, colores y formas geometricas simples",
        "Mixed_5c": "Texturas y patrones repetitivos",
        "Mixed_5d": "Formas basicas y contornos",
        "Mixed_6a": "Estructuras intermedias y gradientes",
        "Mixed_6b": "Figuras complejas en formacion",
        "Mixed_6c": "Patrones tipo ojo y espiral",
        "Mixed_6d": "Estructuras tipo animal o planta",
        "Mixed_6e": "Objetos reconocibles (recomendada)",
        "Mixed_7a": "Faces, animales, conceptos abstractos",
        "Mixed_7b": "Alta abstraccion — muy dreamlike",
        "Mixed_7c": "Maxima abstraccion de InceptionV3",
    },
    "alexnet": {
        "features.0":  "Conv1 — Bordes y colores crudos",
        "features.3":  "Conv2 — Texturas simples",
        "features.6":  "Conv3 — Patrones y mosaicos",
        "features.8":  "Conv4 — Formas complejas",
        "features.10": "Conv5 — Objetos de alto nivel (recomendada)",
    },
}

# Parametros del notebook original.
# lr por modelo: Inception=0.04, AlexNet=0.03
# Las iteraciones determinan la intensidad del efecto.
INTENSIDADES = {
    "Micro":   {"iterations": 3},   # apenas visible — patrones sutiles
    "Suave":   {"iterations": 10},
    "Normal":  {"iterations": 20},
    "Intenso": {"iterations": 30},
    "Extremo": {"iterations": 50},
}


# ================================================================
# CLASE: CapturaActivaciones
# ================================================================

class CapturaActivaciones:
    """
    Hook de PyTorch para capturar activaciones de una capa.

    Implementacion identica al notebook: busca la capa por nombre
    usando modelo.named_modules() y registra el hook directamente.

    Uso:
        capturador = CapturaActivaciones(modelo, "Mixed_6e")
        modelo(img)
        acts = capturador.obtener_activaciones()
        capturador.remover_hook()
    """

    def __init__(self, modelo: nn.Module, nombre_capa: str):
        self.activaciones = None

        capa_encontrada = False
        for nombre, modulo in modelo.named_modules():
            if nombre == nombre_capa:
                self.hook = modulo.register_forward_hook(self._hook_fn)
                capa_encontrada = True
                break

        if not capa_encontrada:
            raise ValueError(
                f"Capa '{nombre_capa}' no encontrada en el modelo")

    def _hook_fn(self, modulo, input, output):
        self.activaciones = output

    def obtener_activaciones(self):
        return self.activaciones

    def remover_hook(self):
        self.hook.remove()


# ================================================================
# FUNCIONES: modelo y preprocesamiento
# ================================================================

def cargar_modelo_interno(nombre: str) -> nn.Module:
    """
    Carga modelo pre-entrenado en eval() con parametros congelados.

    aux_logits=False: desactiva la cabeza auxiliar de InceptionV3.
    En DeepDream no queremos esa salida adicional — genera gradientes
    que interfieren con el efecto.
    """
    if nombre == "inception":
        modelo = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT)
        modelo.aux_logits = False
    elif nombre == "alexnet":
        modelo = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    else:
        raise ValueError(
            f"Modelo '{nombre}' no soportado. Usar 'inception' o 'alexnet'.")

    modelo.eval()
    for param in modelo.parameters():
        param.requires_grad_(False)
    return modelo


def preprocesar_imagen(pil_img: Image.Image, tamanio: tuple) -> torch.Tensor:
    """
    PIL Image -> tensor [1, 3, H, W] en rango [0, 1].

    IMPORTANTE: NO aplicamos normalizacion ImageNet (sin mean/std).
    El notebook usa solo ToTensor() que convierte [0,255] -> [0,1].
    Normalizar con ImageNet distorsiona los gradientes en DeepDream
    porque el modelo espera esa distribucion en inferencia normal,
    pero en DeepDream estamos optimizando la imagen directamente.

    Aplica correccion EXIF para fotos de iPhone/Android.
    """
    pil_img = ImageOps.exif_transpose(pil_img)  # fix rotacion de camara
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(tamanio, Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,255] -> [0,1], sin normalizacion
    ])
    return transform(pil_img).unsqueeze(0)  # [1, 3, H, W]


def numpy_a_pil(img_numpy: np.ndarray) -> Image.Image:
    """numpy float32 [H,W,3] [0,1] -> PIL Image RGB."""
    img_uint8 = np.clip(img_numpy * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_uint8)


def tensor_a_pil(img_tensor: torch.Tensor) -> Image.Image:
    """tensor [1,3,H,W] o [3,H,W] en [0,1] -> PIL Image RGB."""
    if img_tensor.ndim == 4:
        img_tensor = img_tensor.squeeze(0)
    img_np = img_tensor.detach().cpu().clamp(0, 1).numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    return Image.fromarray((img_np * 255).astype(np.uint8))


# ================================================================
# FUNCION PRINCIPAL: deepdream_universal
# ================================================================

def deepdream_universal(
    img_tensor: torch.Tensor,
    modelo: nn.Module,
    nombre_capa: str,
    config: dict,
    device: torch.device,
    callback=None,
) -> Image.Image:
    """
    DeepDream con piramide de octavas — implementacion fiel al notebook.

    ALGORITMO (igual que el notebook):

    1. Calcular lista de tamanios decrecientes (piramide).
       Empezar desde el tamanio original y dividir por scale_factor.
       Invertir la lista: procesar de pequeno a grande.

    2. Por cada octava (tamanio):
       a. Escalar img_tensor al tamanio de esta octava (F.interpolate)
       b. Si hay detalle acumulado de la octava anterior, sumarlo
       c. Loop de gradient ascent:
          - Jitter aleatorio (torch.roll)
          - Forward pass -> capturar activaciones via hook
          - loss = norma de activaciones
          - Backward -> gradiente respecto a la imagen
          - Normalizar gradiente por std
          - img = img + lr * grad  (ASCENSO, no descenso)
          - Deshacer jitter
          - Cada 4 iteraciones: blur gaussiano suave
          - Clamp a [0, 1]
       d. detail = img - img_base  (lo que agrego DeepDream)

    3. Retornar la imagen final como PIL.

    Args:
        img_tensor:  tensor [1, 3, H, W] en [0,1] — salida de preprocesar_imagen
        modelo:      modelo en eval() con parametros congelados
        nombre_capa: nombre de la capa a maximizar
        config:      dict con iterations, num_octavas, scale_factor, lr
        device:      torch.device
        callback:    funcion opcional callback(octava_actual, total, pil_img)

    Returns:
        PIL Image con DeepDream aplicado
    """
    iterations = config.get("iterations", 20)
    num_octavas = config.get("num_octavas", 4)
    scale_factor = config.get("scale_factor", 1.3)
    lr = config.get("lr", 0.04)

    modelo = modelo.to(device)
    img_tensor = img_tensor.to(device)

    # Hook para capturar activaciones de la capa objetivo
    capturador = CapturaActivaciones(modelo, nombre_capa)

    # Blur suave para regularizacion (aplicado cada 4 iteraciones)
    blur = GaussianBlur(kernel_size=3, sigma=0.5)

    try:
        # ----------------------------------------------------------
        # PIRAMIDE: tamanios decrecientes desde original
        # ----------------------------------------------------------
        # Igual que el notebook: partir del tamanio original,
        # dividir por scale_factor, cortar si alguna dimension < MIN_SIZE.
        # Luego invertir para procesar de pequeno a grande.

        base_h, base_w = img_tensor.shape[-2:]
        tamanios = []

        h, w = base_h, base_w
        for _ in range(num_octavas):
            tamanios.append((h, w))
            h = int(h / scale_factor)
            w = int(w / scale_factor)
            if h < MIN_SIZE or w < MIN_SIZE:
                break

        tamanios.reverse()  # pequeno -> grande
        num_octavas_real = len(tamanios)

        # ----------------------------------------------------------
        # GRADIENT ASCENT POR OCTAVA
        # ----------------------------------------------------------
        detail = None  # acumula el "sueno" entre octavas

        for octave_idx, (h, w) in enumerate(tamanios):

            # Escalar la imagen original a este tamanio
            img_base = F.interpolate(
                img_tensor,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )

            # Sumar detalle de la octava anterior (si existe)
            # El detalle es lo que DeepDream agrego — lo transferimos
            # a la siguiente escala para que los patrones sean coherentes
            if detail is not None:
                detail_resized = F.interpolate(
                    detail,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                )
                img_base = img_base + detail_resized

            img = img_base.clone()

            # Loop de gradient ascent
            for i in range(iterations):

                # Jitter: desplazamiento aleatorio para reducir artefactos
                ox = np.random.randint(-24, 25)
                oy = np.random.randint(-24, 25)
                img = torch.roll(img, shifts=(ox, oy), dims=(2, 3))

                # Requiere gradiente respecto a la imagen
                img = img.detach().requires_grad_(True)

                # Forward pass: el hook captura activaciones
                output = modelo(img)
                if isinstance(output, tuple):
                    output = output[0]

                activaciones = capturador.obtener_activaciones()

                # Maximizar norma de activaciones = la red "ve mas" en la imagen
                loss = activaciones.norm()
                loss.backward()

                grad = img.grad.data
                # Normalizar por std -> learning rate efectivo estable
                grad = grad / (grad.std() + 1e-8)

                # ASCENSO: sumar gradiente (en entrenamiento se resta)
                img = img.data + lr * grad

                # Deshacer jitter
                img = torch.roll(img, shifts=(-ox, -oy), dims=(2, 3))

                # Blur suave cada 4 iteraciones (regularizacion)
                if (i + 1) % 4 == 0:
                    img = blur(img)

                img = torch.clamp(img, 0, 1)

            # Guardar el detalle que agrego DeepDream en esta octava
            detail = img.detach() - img_base.detach()

            # Callback para mostrar progreso en la UI
            if callback is not None:
                callback(octave_idx + 1, num_octavas_real, tensor_a_pil(img))

        return tensor_a_pil(img)

    finally:
        capturador.remover_hook()
