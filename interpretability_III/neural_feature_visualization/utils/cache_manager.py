"""
===================================================================
CACHE_MANAGER.PY - Gestión de Caché
===================================================================

Este módulo gestiona el cacheo de resultados para optimizar performance:
- Modelos cargados
- Activaciones computadas
- Imágenes procesadas
- Patrones sintéticos generados

Uso:
    cache = CacheManager()
    cache.set('model_alexnet', model)
    model = cache.get('model_alexnet')
===================================================================
"""

from config import ENABLE_MODEL_CACHE, ENABLE_ACTIVATION_CACHE
from typing import Any, Optional, Dict
import hashlib
import pickle
import time
from pathlib import Path

# Importar configuración
import sys
sys.path.append('..')


class CacheManager:
    """
    Gestor de caché en memoria para resultados computacionales.

    Permite cachear objetos para evitar recálculos costosos.
    """

    def __init__(
        self,
        max_size: int = 100,
        enable_model_cache: bool = ENABLE_MODEL_CACHE,
        enable_activation_cache: bool = ENABLE_ACTIVATION_CACHE
    ):
        """
        Inicializa el gestor de caché.

        Args:
            max_size: Número máximo de items en caché
            enable_model_cache: Habilitar caché de modelos
            enable_activation_cache: Habilitar caché de activaciones
        """
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._access_count: Dict[str, int] = {}

        self.max_size = max_size
        self.enable_model_cache = enable_model_cache
        self.enable_activation_cache = enable_activation_cache

        print(f"✅ CacheManager inicializado")
        print(f"   Max size: {max_size}")
        print(f"   Model cache: {enable_model_cache}")
        print(f"   Activation cache: {enable_activation_cache}")

    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un item del caché.

        Args:
            key: Clave del item

        Returns:
            Item cacheado o None si no existe
        """
        if key in self._cache:
            # Actualizar contador de accesos
            self._access_count[key] += 1
            return self._cache[key]

        return None

    def set(self, key: str, value: Any) -> bool:
        """
        Almacena un item en el caché.

        Args:
            key: Clave del item
            value: Valor a cachear

        Returns:
            True si se almacenó correctamente
        """
        # Verificar si debemos cachear según tipo
        if not self._should_cache(key):
            return False

        # Si el caché está lleno, hacer espacio
        if len(self._cache) >= self.max_size:
            self._evict_lru()

        # Almacenar
        self._cache[key] = value
        self._timestamps[key] = time.time()
        self._access_count[key] = 0

        return True

    def _should_cache(self, key: str) -> bool:
        """
        Determina si un item debe ser cacheado según su tipo.

        Args:
            key: Clave del item

        Returns:
            True si debe ser cacheado
        """
        # Modelos
        if 'model_' in key:
            return self.enable_model_cache

        # Activaciones
        if 'activation_' in key:
            return self.enable_activation_cache

        # Otros items: cachear por defecto
        return True

    def _evict_lru(self):
        """
        Elimina el item menos recientemente usado (LRU).
        """
        if not self._cache:
            return

        # Encontrar el item con menor número de accesos
        lru_key = min(self._access_count.items(), key=lambda x: x[1])[0]

        # Eliminar
        del self._cache[lru_key]
        del self._timestamps[lru_key]
        del self._access_count[lru_key]

        print(f"🗑️  Cache evicted: {lru_key}")

    def has(self, key: str) -> bool:
        """
        Verifica si una clave existe en el caché.

        Args:
            key: Clave a verificar

        Returns:
            True si existe
        """
        return key in self._cache

    def remove(self, key: str) -> bool:
        """
        Elimina un item del caché.

        Args:
            key: Clave del item

        Returns:
            True si se eliminó
        """
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
            del self._access_count[key]
            return True

        return False

    def clear(self):
        """
        Limpia todo el caché.
        """
        count = len(self._cache)
        self._cache.clear()
        self._timestamps.clear()
        self._access_count.clear()

        print(f"🧹 Cache cleared: {count} items removed")

    def clear_by_prefix(self, prefix: str):
        """
        Limpia items del caché que coincidan con un prefijo.

        Args:
            prefix: Prefijo de las claves a eliminar
        """
        keys_to_remove = [
            k for k in self._cache.keys() if k.startswith(prefix)]

        for key in keys_to_remove:
            self.remove(key)

        print(
            f"🧹 Cache cleared: {len(keys_to_remove)} items with prefix '{prefix}'")

    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas del caché.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'utilization': len(self._cache) / self.max_size * 100,
            'keys': list(self._cache.keys()),
            'access_counts': self._access_count.copy()
        }

    def print_stats(self):
        """
        Imprime estadísticas del caché.
        """
        stats = self.get_stats()

        print("\n" + "="*70)
        print("📊 ESTADÍSTICAS DEL CACHÉ")
        print("="*70)
        print(f"Tamaño: {stats['size']}/{stats['max_size']} "
              f"({stats['utilization']:.1f}% utilizado)")
        print(f"\nItems en caché:")

        for key in stats['keys']:
            accesos = stats['access_counts'][key]
            print(f"  • {key}: {accesos} accesos")

        print("="*70 + "\n")


class ImageHashCache:
    """
    Caché especializado que usa hashes de imágenes como claves.

    Útil para cachear activaciones o resultados basados en imágenes.
    """

    def __init__(self):
        """Inicializa el caché de hashes."""
        self._cache: Dict[str, Any] = {}

    @staticmethod
    def compute_hash(image_array) -> str:
        """
        Computa hash de una imagen.

        Args:
            image_array: Array numpy de la imagen

        Returns:
            Hash string
        """
        # Convertir a bytes y hashear
        image_bytes = image_array.tobytes()
        return hashlib.sha256(image_bytes).hexdigest()[:16]

    def get(self, image_array) -> Optional[Any]:
        """
        Obtiene resultado cacheado para una imagen.

        Args:
            image_array: Array numpy de la imagen

        Returns:
            Resultado cacheado o None
        """
        key = self.compute_hash(image_array)
        return self._cache.get(key)

    def set(self, image_array, value: Any):
        """
        Cachea resultado para una imagen.

        Args:
            image_array: Array numpy de la imagen
            value: Valor a cachear
        """
        key = self.compute_hash(image_array)
        self._cache[key] = value

    def clear(self):
        """Limpia el caché."""
        self._cache.clear()


# ===================================================================
# INSTANCIA GLOBAL (SINGLETON)
# ===================================================================

# Instancia global del cache manager
_global_cache = None


def get_cache() -> CacheManager:
    """
    Obtiene la instancia global del cache manager.

    Returns:
        Instancia de CacheManager
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = CacheManager()

    return _global_cache


# ===================================================================
# TESTING
# ===================================================================

if __name__ == "__main__":
    print("🧪 Testing CacheManager...\n")

    # Crear cache
    print("1️⃣ Creando CacheManager...")
    cache = CacheManager(max_size=5)

    # Añadir items
    print("\n2️⃣ Añadiendo items...")
    cache.set('model_alexnet', 'fake_model_object')
    cache.set('activation_layer1', 'fake_activation')
    cache.set('image_processed', 'fake_image')

    # Obtener items
    print("\n3️⃣ Obteniendo items...")
    model = cache.get('model_alexnet')
    print(f"   Model: {model}")

    # Verificar existencia
    print("\n4️⃣ Verificando existencia...")
    exists = cache.has('model_alexnet')
    print(f"   Existe 'model_alexnet': {exists}")

    # Estadísticas
    print("\n5️⃣ Estadísticas:")
    cache.print_stats()

    # Limpiar
    print("\n6️⃣ Limpiando caché...")
    cache.clear()
    cache.print_stats()

    # Test ImageHashCache
    print("\n7️⃣ Testing ImageHashCache...")
    import numpy as np
    img_cache = ImageHashCache()

    test_img = np.random.rand(224, 224, 3)
    img_cache.set(test_img, 'cached_result')

    result = img_cache.get(test_img)
    print(f"   Resultado cacheado: {result}")

    print("\n✅ Testing completado!")
