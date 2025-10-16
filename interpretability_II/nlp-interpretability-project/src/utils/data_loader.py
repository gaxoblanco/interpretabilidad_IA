"""
============================================================================
DATA LOADER - Carga y Procesamiento del Dataset IMDb
============================================================================

Este módulo proporciona la clase DataLoader para:
1. Cargar el dataset IMDb desde HuggingFace
2. Aplicar preprocessing (limpiar HTML, URLs, caracteres especiales)
3. Crear splits de train/validation/test
4. Filtrar por longitud de texto
5. Gestionar cache eficientemente

Flujo de uso:
    from src.utils.data_loader import DataLoader
    from src.config import Config
    
    config = Config()
    data = DataLoader(config)
    
    # Obtener datos de train
    train_data = data.get_train_data()
    
    # Obtener muestra aleatoria
    sample = data.get_random_sample(n=5)

DataLoader
├── Inicialización
│   ├── _load_dataset()            # Carga desde HuggingFace
│   ├── _preprocess_text()         # Limpia 1 texto
│   ├── _preprocess_dataset()      # Limpia todo el dataset
│   ├── _filter_by_length()        # Filtra por palabras
│   ├── _create_validation_split() # Crea split validación
│   └── _create_splits()           # Organiza splits finales
│
├── Acceso a Datos
│   ├── get_train_data()           # Retorna train
│   ├── get_validation_data()      # Retorna validation
│   ├── get_test_data()            # Retorna test
│   ├── get_sample()               # N ejemplos consecutivos
│   ├── get_random_sample()        # N ejemplos aleatorios
│   └── get_by_label()             # Ejemplos de 1 label
│
└── Utilidades
    ├── get_dataset_info()         # Estadísticas
    └── __repr__()                 # Representación
============================================================================
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict
import random
import html


# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# CLASE PRINCIPAL: DataLoader
# ----------------------------------------------------------------------------

class DataLoader:
    """
    Clase para cargar y procesar el dataset IMDb.

    Esta clase encapsula toda la lógica de:
    - Carga del dataset desde HuggingFace
    - Preprocessing de textos (HTML, URLs, etc.)
    - Creación de splits (train/validation/test)
    - Filtrado por longitud
    - Gestión de cache

    Attributes:
        config: Objeto de configuración del proyecto
        dataset_name (str): Nombre del dataset en HuggingFace
        dataset (DatasetDict): Dataset cargado
        train_data (Dataset): Datos de entrenamiento
        validation_data (Dataset): Datos de validación
        test_data (Dataset): Datos de test

    Example:
        >>> from src.config import Config
        >>> config = Config()
        >>> loader = DataLoader(config)
        >>> train = loader.get_train_data()
        >>> print(f"Train size: {len(train)}")
        Train size: 22500
    """

    def __init__(self, config):
        """
        Inicializa el DataLoader con configuración.

        Args:
            config: Objeto Config con todas las configuraciones del proyecto

        Raises:
            RuntimeError: Si falla la carga del dataset
        """
        self.config = config
        self.dataset_name = config.dataset.name

        # Configuraciones de preprocessing
        self.lowercase = config.dataset.lowercase
        self.remove_html = config.dataset.remove_html
        self.remove_urls = config.dataset.remove_urls
        self.remove_special_chars = config.dataset.remove_special_chars
        self.min_length = config.dataset.min_length
        self.max_length = config.dataset.max_length

        # Configuraciones de splits
        self.train_size = config.dataset.train_size
        self.test_size = config.dataset.test_size
        self.validation_split = config.dataset.validation_split

        # Cache
        self.cache_dir = Path(config.paths.cache_dir) / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"🚀 Inicializando DataLoader con dataset: {self.dataset_name}")

        # Cargar dataset
        self.dataset = self._load_dataset()

        # Aplicar preprocessing
        self.dataset = self._preprocess_dataset()

        # Crear splits
        self._create_splits()

        logger.info("✅ DataLoader inicializado exitosamente")

    # ------------------------------------------------------------------------
    # MÉTODOS PRIVADOS: Carga y Procesamiento
    # ------------------------------------------------------------------------

    def _load_dataset(self) -> DatasetDict:
        """
        Carga el dataset IMDb desde HuggingFace.
        """
        try:
            logger.info(f"📥 Cargando dataset: {self.dataset_name}")

            # Cargar dataset completo
            dataset = load_dataset(
                self.dataset_name,
                cache_dir=str(self.cache_dir)
            )

            logger.info(f"✅ Dataset cargado:")
            logger.info(f"   - Train: {len(dataset['train'])} ejemplos")
            logger.info(f"   - Test: {len(dataset['test'])} ejemplos")

            # Aplicar límites de tamaño si están configurados
            if self.train_size is not None:
                logger.info(f"   Limitando train a {self.train_size} ejemplos")
                dataset['train'] = dataset['train'].select(
                    range(self.train_size))

            # ⭐ NUEVO: Subset estratificado usando train_test_split
            if self.test_size is not None:
                logger.info(
                    f"   Limitando test a {self.test_size} ejemplos (estratificado)")

                # Calcular fracción a mantener
                fraction = self.test_size / len(dataset['test'])

                # Hacer split estratificado
                # Nota: train_test_split retorna {'train': subset_pequeño, 'test': resto}
                # Usamos 'train' porque es el subset que queremos (de tamaño test_size)
                split = dataset['test'].train_test_split(
                    train_size=fraction,
                    stratify_by_column='label',  # ⭐ Clave: estratificar por label
                    seed=42  # Reproducibilidad
                )

                # Usar el subset pequeño como test
                dataset['test'] = split['train']

                # Verificar balance
                test_labels = dataset['test']['label']
                num_pos = sum(test_labels)
                num_neg = len(test_labels) - num_pos
                logger.info(
                    f"   ✅ Test balanceado: {num_neg} negativos, {num_pos} positivos")

            return dataset

        except Exception as e:
            error_msg = f"❌ Error al cargar dataset: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _preprocess_text(self, text: str) -> str:
        """
        Aplica preprocessing a un texto individual.

        Pasos de preprocessing (según configuración):
        1. Decodificar entidades HTML (ej: &amp; -> &)
        2. Eliminar tags HTML (ej: <br />, <b>, etc.)
        3. Eliminar URLs
        4. Eliminar caracteres especiales (opcional)
        5. Convertir a minúsculas (opcional)
        6. Limpiar espacios múltiples

        Args:
            text: Texto a procesar

        Returns:
            Texto procesado

        Example:
            >>> text = "<br />This is <b>great</b>! http://example.com"
            >>> processed = loader._preprocess_text(text)
            >>> print(processed)
            "this is great!"
        """
        if not text:
            return ""

        # 1. Decodificar entidades HTML
        text = html.unescape(text)

        # 2. Eliminar tags HTML
        if self.remove_html:
            # Patrón para tags HTML: <...>
            text = re.sub(r'<[^>]+>', ' ', text)
            # Casos especiales: <br/>, <br />, etc.
            text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)

        # 3. Eliminar URLs
        if self.remove_urls:
            # Patrón para URLs (http, https, www)
            text = re.sub(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                ' ',
                text
            )
            text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', ' ', text)

        # 4. Eliminar caracteres especiales (opcional)
        if self.remove_special_chars:
            # Mantener solo letras, números, espacios y puntuación básica
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'-]', ' ', text)

        # 5. Convertir a minúsculas (opcional)
        if self.lowercase:
            text = text.lower()

        # 6. Limpiar espacios múltiples y espacios al inicio/final
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _preprocess_dataset(self) -> DatasetDict:
        """
        Aplica preprocessing a todo el dataset.

        Returns:
            DatasetDict con textos procesados
        """
        logger.info("🔄 Aplicando preprocessing al dataset...")

        def preprocess_example(example):
            """Función auxiliar para procesar un ejemplo"""
            example['text'] = self._preprocess_text(example['text'])
            return example

        # Aplicar preprocessing a todos los splits
        processed_dataset = self.dataset.map(
            preprocess_example,
            desc="Preprocessing",
            num_proc=self.config.model.num_workers  # Procesamiento paralelo
        )

        logger.info("✅ Preprocessing completado")

        return processed_dataset

    def _filter_by_length(self, dataset: Dataset) -> Dataset:
        """
        Filtra el dataset por longitud de texto (tokens, no palabras).

        Args:
            dataset: Dataset a filtrar

        Returns:
            Dataset filtrado
        """
        logger.info(f"🔍 Filtrando por longitud de tokens (<512)")

        def filter_example(example):
            """Función auxiliar para filtrar por longitud EN TOKENS"""
            # Contar tokens en vez de palabras
            # Aproximación: 1.3 palabras ≈ 1 token en promedio
            num_words = len(example['text'].split())
            estimated_tokens = int(num_words * 1.3)

            # Filtro conservador: max 380 palabras ≈ 494 tokens
            return num_words <= 380

        # Aplicar filtro
        filtered = dataset.filter(
            filter_example,
            desc="Filtering by token length"
        )

        removed = len(dataset) - len(filtered)
        if removed > 0:
            logger.info(
                f"   Removidos {removed} ejemplos muy largos ({removed/len(dataset):.1%})")

        return filtered

    def _create_validation_split(self, train_dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """
        Crea un split de validación desde el conjunto de train.

        Args:
            train_dataset: Dataset de entrenamiento

        Returns:
            Tuple[Dataset, Dataset]: (train_reducido, validation)
        """
        if self.validation_split <= 0:
            logger.info("⚠️  Sin split de validación (validation_split = 0)")
            return train_dataset, None

        logger.info(
            f"✂️  Creando split de validación ({self.validation_split:.0%} del train)")

        # Calcular tamaño del split
        total_size = len(train_dataset)
        val_size = int(total_size * self.validation_split)
        train_size = total_size - val_size

        # Crear splits usando índices aleatorios
        indices = list(range(total_size))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Seleccionar ejemplos
        new_train = train_dataset.select(train_indices)
        validation = train_dataset.select(val_indices)

        logger.info(f"   Train: {len(new_train)} ejemplos")
        logger.info(f"   Validation: {len(validation)} ejemplos")

        return new_train, validation

    def _create_splits(self) -> None:
        """
        Crea y asigna los splits finales (train/validation/test).

        Este método:
        1. Filtra datasets por longitud
        2. Crea split de validación desde train
        3. Asigna a atributos de la clase
        """
        logger.info("✂️  Creando splits finales...")

        # Filtrar por longitud
        filtered_train = self._filter_by_length(self.dataset['train'])
        filtered_test = self._filter_by_length(self.dataset['test'])

        # Crear split de validación
        self.train_data, self.validation_data = self._create_validation_split(
            filtered_train
        )

        # Asignar test
        self.test_data = filtered_test

        logger.info("✅ Splits creados:")
        logger.info(f"   Train: {len(self.train_data)} ejemplos")
        if self.validation_data:
            logger.info(f"   Validation: {len(self.validation_data)} ejemplos")
        logger.info(f"   Test: {len(self.test_data)} ejemplos")

    # ------------------------------------------------------------------------
    # MÉTODOS PÚBLICOS: Acceso a Datos
    # ------------------------------------------------------------------------

    def get_train_data(self) -> Dataset:
        """
        Obtiene el conjunto de entrenamiento.

        Returns:
            Dataset de entrenamiento

        Example:
            >>> train = loader.get_train_data()
            >>> print(train[0])
            {'text': 'this movie is great!', 'label': 1}
        """
        return self.train_data

    def get_validation_data(self) -> Optional[Dataset]:
        """
        Obtiene el conjunto de validación.

        Returns:
            Dataset de validación o None si no existe
        """
        return self.validation_data

    def get_test_data(self) -> Dataset:
        """
        Obtiene el conjunto de test.

        Returns:
            Dataset de test
        """
        return self.test_data

    def get_sample(
        self,
        n: int = 5,
        split: str = 'train',
        start_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Obtiene una muestra de N ejemplos consecutivos.

        Args:
            n: Número de ejemplos a obtener
            split: Split del cual obtener ('train', 'validation', 'test')
            start_idx: Índice inicial

        Returns:
            Lista de diccionarios con ejemplos

        Example:
            >>> sample = loader.get_sample(n=3, split='train')
            >>> for example in sample:
            ...     print(f"{example['text'][:50]}... - {example['label']}")
        """
        # Seleccionar dataset
        if split == 'train':
            dataset = self.train_data
        elif split == 'validation':
            if self.validation_data is None:
                raise ValueError("❌ No existe split de validación")
            dataset = self.validation_data
        elif split == 'test':
            dataset = self.test_data
        else:
            raise ValueError(f"❌ Split inválido: {split}")

        # Validar índices
        if start_idx >= len(dataset):
            raise ValueError(
                f"❌ start_idx ({start_idx}) >= tamaño del dataset ({len(dataset)})")

        end_idx = min(start_idx + n, len(dataset))

        # Obtener ejemplos
        sample = dataset.select(range(start_idx, end_idx))

        return [
            {
                'text': example['text'],
                'label': example['label'],
                'label_name': 'POSITIVE' if example['label'] == 1 else 'NEGATIVE'
            }
            for example in sample
        ]

    def get_random_sample(
        self,
        n: int = 5,
        split: str = 'train',
        seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene una muestra aleatoria de N ejemplos.

        Args:
            n: Número de ejemplos a obtener
            split: Split del cual obtener ('train', 'validation', 'test')
            seed: Semilla para reproducibilidad (opcional)

        Returns:
            Lista de diccionarios con ejemplos aleatorios

        Example:
            >>> sample = loader.get_random_sample(n=5, split='test', seed=42)
            >>> print(f"Obtenidos {len(sample)} ejemplos aleatorios")
        """
        # Seleccionar dataset
        if split == 'train':
            dataset = self.train_data
        elif split == 'validation':
            if self.validation_data is None:
                raise ValueError("❌ No existe split de validación")
            dataset = self.validation_data
        elif split == 'test':
            dataset = self.test_data
        else:
            raise ValueError(f"❌ Split inválido: {split}")

        # Validar n
        if n > len(dataset):
            logger.warning(
                f"⚠️  n ({n}) > tamaño del dataset ({len(dataset)}), usando todo el dataset")
            n = len(dataset)

        # Generar índices aleatorios
        if seed is not None:
            random.seed(seed)

        indices = random.sample(range(len(dataset)), n)

        # Obtener ejemplos
        sample = dataset.select(indices)

        return [
            {
                'text': example['text'],
                'label': example['label'],
                'label_name': 'POSITIVE' if example['label'] == 1 else 'NEGATIVE'
            }
            for example in sample
        ]

    def get_by_label(
        self,
        label: int,
        n: int = 5,
        split: str = 'train',
        random_sample: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Obtiene ejemplos de una etiqueta específica.

        Args:
            label: Etiqueta a filtrar (0=NEGATIVE, 1=POSITIVE)
            n: Número de ejemplos
            split: Split del cual obtener
            random_sample: Si True, selección aleatoria

        Returns:
            Lista de ejemplos con la etiqueta especificada

        Example:
            >>> positives = loader.get_by_label(label=1, n=10, split='test')
            >>> print(f"Obtenidos {len(positives)} ejemplos positivos")
        """
        # Seleccionar dataset
        if split == 'train':
            dataset = self.train_data
        elif split == 'validation':
            if self.validation_data is None:
                raise ValueError("❌ No existe split de validación")
            dataset = self.validation_data
        elif split == 'test':
            dataset = self.test_data
        else:
            raise ValueError(f"❌ Split inválido: {split}")

        # Filtrar por label
        filtered = dataset.filter(lambda x: x['label'] == label)

        if len(filtered) == 0:
            logger.warning(f"⚠️  No se encontraron ejemplos con label={label}")
            return []

        # Limitar n
        n = min(n, len(filtered))

        # Seleccionar ejemplos
        if random_sample:
            indices = random.sample(range(len(filtered)), n)
            sample = filtered.select(indices)
        else:
            sample = filtered.select(range(n))

        return [
            {
                'text': example['text'],
                'label': example['label'],
                'label_name': 'POSITIVE' if example['label'] == 1 else 'NEGATIVE'
            }
            for example in sample
        ]

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Obtiene información detallada del dataset.

        Returns:
            Dict con estadísticas del dataset

        Example:
            >>> info = loader.get_dataset_info()
            >>> print(f"Train size: {info['train_size']}")
            >>> print(f"Avg words: {info['avg_length']:.1f}")
        """
        # Calcular estadísticas de longitud (palabras)
        train_lengths = [len(x['text'].split()) for x in self.train_data]
        test_lengths = [len(x['text'].split()) for x in self.test_data]

        # Contar labels
        train_labels = [x['label'] for x in self.train_data]
        test_labels = [x['label'] for x in self.test_data]

        info = {
            'dataset_name': self.dataset_name,
            'train_size': len(self.train_data),
            'validation_size': len(self.validation_data) if self.validation_data else 0,
            'test_size': len(self.test_data),
            'total_size': len(self.train_data) + len(self.test_data),
            'train_positive': sum(train_labels),
            'train_negative': len(train_labels) - sum(train_labels),
            'test_positive': sum(test_labels),
            'test_negative': len(test_labels) - sum(test_labels),
            'avg_length_train': sum(train_lengths) / len(train_lengths),
            'avg_length_test': sum(test_lengths) / len(test_lengths),
            'min_length': self.min_length,
            'max_length': self.max_length,
            'preprocessing': {
                'lowercase': self.lowercase,
                'remove_html': self.remove_html,
                'remove_urls': self.remove_urls,
                'remove_special_chars': self.remove_special_chars
            }
        }

        return info

    def __repr__(self) -> str:
        """Representación legible del DataLoader"""
        return (
            f"DataLoader(dataset={self.dataset_name}, "
            f"train={len(self.train_data)}, "
            f"test={len(self.test_data)})"
        )

    def __len__(self) -> int:
        """Retorna el tamaño total del dataset"""
        total = len(self.train_data) + len(self.test_data)
        if self.validation_data:
            total += len(self.validation_data)
        return total


# ----------------------------------------------------------------------------
# EJEMPLO DE USO
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Ejemplo de uso del DataLoader.
    Ejecutar: python -m src.utils.data_loader
    """

    from src.config import setup_project

    print("\n" + "="*60)
    print("🧪 EJEMPLO DE USO: DataLoader")
    print("="*60 + "\n")

    # 1. Setup del proyecto
    config = setup_project()

    # 2. Inicializar DataLoader
    loader = DataLoader(config)

    # 3. Información del dataset
    print("\n📊 INFORMACIÓN DEL DATASET:")
    info = loader.get_dataset_info()
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  • {key}:")
            for k, v in value.items():
                print(f"      - {k}: {v}")
        else:
            print(f"  • {key}: {value}")

    # 4. Muestra de ejemplos
    print("\n📝 MUESTRA DE EJEMPLOS (Train):")
    sample = loader.get_sample(n=3, split='train')
    for i, example in enumerate(sample, 1):
        print(f"\n{i}. [{example['label_name']}]")
        print(f"   {example['text'][:100]}...")

    # 5. Muestra aleatoria
    print("\n🎲 MUESTRA ALEATORIA (Test):")
    random_sample = loader.get_random_sample(n=3, split='test', seed=42)
    for i, example in enumerate(random_sample, 1):
        print(f"\n{i}. [{example['label_name']}]")
        print(f"   {example['text'][:100]}...")

    # 6. Ejemplos por label
    print("\n✅ EJEMPLOS POSITIVOS:")
    positives = loader.get_by_label(label=1, n=2, split='test')
    for example in positives:
        print(f"   {example['text'][:80]}...")

    print("\n❌ EJEMPLOS NEGATIVOS:")
    negatives = loader.get_by_label(label=0, n=2, split='test')
    for example in negatives:
        print(f"   {example['text'][:80]}...")

    print("\n" + "="*60)
    print("✅ Ejemplo completado")
    print("="*60 + "\n")
