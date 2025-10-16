"""
============================================================================
CONFIG LOADER - Helper para Cargar Configuraciones
============================================================================

Este módulo proporciona utilidades para:
1. Cargar el archivo config.yaml
2. Validar configuraciones
3. Crear directorios necesarios
4. Acceso tipo objeto a configuraciones

Uso:
    from src.config import load_config, Config
    
    # Opción 1: Cargar como diccionario
    config = load_config()
    model_name = config['model']['name']
    
    # Opción 2: Cargar como objeto (recomendado)
    config = Config()
    model_name = config.model.name

Autor: Proyecto Módulo II
Fecha: 2025-01-15
============================================================================
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# CONFIGURACIÓN DE LOGGING
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# CONSTANTES
# ----------------------------------------------------------------------------
# Ruta por defecto del archivo de configuración
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Rutas obligatorias que deben existir o crearse
REQUIRED_DIRS = [
    "data/cache",
    "data/raw",
    "data/processed",
    "data/models",
    "checkpoints",
    "results/plots",
    "results/logs",
    "notebooks"
]


# ----------------------------------------------------------------------------
# CLASES DE CONFIGURACIÓN (Acceso tipo objeto)
# ----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuración del modelo DistilBERT"""
    name: str
    max_length: int
    truncation: bool
    padding: str
    return_tensors: str
    device: str
    batch_size: int
    num_workers: int
    cache_model: bool
    use_fast_tokenizer: bool


@dataclass
class DatasetConfig:
    """Configuración del dataset IMDb"""
    name: str
    train_split: str
    test_split: str
    train_size: Optional[int]
    test_size: Optional[int]
    validation_split: float
    lowercase: bool
    remove_html: bool
    remove_urls: bool
    remove_special_chars: bool
    min_length: int
    max_length: int
    cache_dataset: bool


@dataclass
class SHAPConfig:
    """Configuración de SHAP Explainer"""
    explainer_type: str
    num_samples: int
    max_evals: int
    masking_strategy: str
    mask_token: str
    aggregation: str
    max_display: int
    plot_type: str
    batch_size: int
    use_cache: bool
    cache_dir: str
    timeout_seconds: int


@dataclass
class LIMEConfig:
    """Configuración de LIME Explainer"""
    num_samples: int
    num_features: int
    perturbation_strategy: str
    kernel_width: int
    distance_metric: str
    feature_selection: str
    alpha: float
    split_expression: str
    bow: bool
    random_state: int
    use_cache: bool
    cache_dir: str
    timeout_seconds: int


@dataclass
class PathsConfig:
    """Configuración de rutas del proyecto"""
    root: str
    data_dir: str
    cache_dir: str
    raw_data: str
    processed_data: str
    models_dir: str
    checkpoints_dir: str
    results_dir: str
    plots_dir: str
    logs_dir: str
    notebooks_dir: str
    streamlit_cache: str


# ----------------------------------------------------------------------------
# CLASE PRINCIPAL DE CONFIGURACIÓN
# ----------------------------------------------------------------------------

class Config:
    """
    Clase principal para acceder a todas las configuraciones del proyecto.
    
    Proporciona acceso tipo objeto a las configuraciones:
        config = Config()
        model_name = config.model.name
    
    Attributes:
        model (ModelConfig): Configuración del modelo
        dataset (DatasetConfig): Configuración del dataset
        shap (SHAPConfig): Configuración de SHAP
        lime (LIMEConfig): Configuración de LIME
        paths (PathsConfig): Configuración de rutas
        raw (dict): Configuración raw completa
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa la configuración.
        
        Args:
            config_path: Ruta al archivo config.yaml (opcional)
        """
        # Cargar configuración raw
        self.raw = load_config(config_path)
        
        # Crear objetos de configuración tipados
        self.model = self._create_model_config()
        self.dataset = self._create_dataset_config()
        self.shap = self._create_shap_config()
        self.lime = self._create_lime_config()
        self.paths = self._create_paths_config()
        
        # Validar configuración
        self.validate()
        
        logger.info("✅ Configuración cargada exitosamente")
    
    def _create_model_config(self) -> ModelConfig:
        """Crea objeto ModelConfig desde dict"""
        return ModelConfig(**self.raw['model'])
    
    def _create_dataset_config(self) -> DatasetConfig:
        """Crea objeto DatasetConfig desde dict"""
        return DatasetConfig(**self.raw['dataset'])
    
    def _create_shap_config(self) -> SHAPConfig:
        """Crea objeto SHAPConfig desde dict"""
        return SHAPConfig(**self.raw['shap'])
    
    def _create_lime_config(self) -> LIMEConfig:
        """Crea objeto LIMEConfig desde dict"""
        return LIMEConfig(**self.raw['lime'])
    
    def _create_paths_config(self) -> PathsConfig:
        """Crea objeto PathsConfig desde dict"""
        return PathsConfig(**self.raw['paths'])
    
    def validate(self) -> bool:
        """
        Valida que la configuración sea correcta.
        
        Returns:
            bool: True si la configuración es válida
        
        Raises:
            ValueError: Si hay errores en la configuración
        """
        errors = []
        
        # Validar modelo
        if self.model.max_length <= 0 or self.model.max_length > 512:
            errors.append(f"model.max_length debe estar entre 1 y 512, got {self.model.max_length}")
        
        if self.model.device not in ["auto", "cpu", "cuda", "mps"]:
            errors.append(f"model.device debe ser 'auto', 'cpu', 'cuda' o 'mps', got {self.model.device}")
        
        # Validar dataset
        if self.dataset.validation_split < 0 or self.dataset.validation_split > 1:
            errors.append(f"dataset.validation_split debe estar entre 0 y 1, got {self.dataset.validation_split}")
        
        # Validar SHAP
        if self.shap.num_samples <= 0:
            errors.append(f"shap.num_samples debe ser > 0, got {self.shap.num_samples}")
        
        if self.shap.masking_strategy not in ["mask_token", "remove", "zero"]:
            errors.append(f"shap.masking_strategy inválida: {self.shap.masking_strategy}")
        
        # Validar LIME
        if self.lime.num_samples <= 0:
            errors.append(f"lime.num_samples debe ser > 0, got {self.lime.num_samples}")
        
        if self.lime.num_features <= 0:
            errors.append(f"lime.num_features debe ser > 0, got {self.lime.num_features}")
        
        if self.lime.perturbation_strategy not in ["removal", "replacement"]:
            errors.append(f"lime.perturbation_strategy inválida: {self.lime.perturbation_strategy}")
        
        # Si hay errores, lanzar excepción
        if errors:
            error_msg = "❌ Errores en configuración:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
        
        return True
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración usando notación de punto.
        
        Args:
            key_path: Ruta al valor (ej: "model.name", "shap.num_samples")
            default: Valor por defecto si no existe la clave
        
        Returns:
            Valor de configuración o default
        
        Example:
            >>> config = Config()
            >>> config.get("model.name")
            "distilbert-base-uncased-finetuned-sst-2-english"
        """
        keys = key_path.split('.')
        value = self.raw
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def __repr__(self) -> str:
        """Representación legible de la configuración"""
        return f"Config(model={self.model.name}, dataset={self.dataset.name})"


# ----------------------------------------------------------------------------
# FUNCIONES PRINCIPALES
# ----------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carga el archivo de configuración YAML.
    
    Args:
        config_path: Ruta al archivo config.yaml (opcional)
                    Si no se proporciona, usa la ubicación por defecto
    
    Returns:
        dict: Diccionario con todas las configuraciones
    
    Raises:
        FileNotFoundError: Si el archivo no existe
        yaml.YAMLError: Si hay error al parsear el YAML
    
    Example:
        >>> config = load_config()
        >>> print(config['model']['name'])
        distilbert-base-uncased-finetuned-sst-2-english
    """
    # Determinar ruta del archivo
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)
    
    # Validar que existe
    if not config_path.exists():
        raise FileNotFoundError(
            f"❌ Archivo de configuración no encontrado: {config_path}\n"
            f"   Asegúrate de que existe src/config/config.yaml"
        )
    
    # Cargar YAML
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✅ Configuración cargada desde: {config_path}")
        return config
    
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"❌ Error al parsear archivo YAML: {e}\n"
            f"   Revisa la sintaxis en {config_path}"
        )


def create_directories(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Crea todos los directorios necesarios para el proyecto.
    
    Args:
        config: Diccionario de configuración (opcional)
                Si no se proporciona, carga desde archivo
    
    Example:
        >>> create_directories()
        ✅ Directorios creados: data/cache, data/raw, ...
    """
    if config is None:
        config = load_config()
    
    # Obtener rutas desde config
    paths = config.get('paths', {})
    root = Path(paths.get('root', '.'))
    
    # Directorios a crear
    dirs_to_create = [
        root / paths.get('data_dir', 'data'),
        root / paths.get('cache_dir', 'data/cache'),
        root / paths.get('raw_data', 'data/raw'),
        root / paths.get('processed_data', 'data/processed'),
        root / paths.get('models_dir', 'data/models'),
        root / paths.get('checkpoints_dir', 'checkpoints'),
        root / paths.get('results_dir', 'results'),
        root / paths.get('plots_dir', 'results/plots'),
        root / paths.get('logs_dir', 'results/logs'),
        root / paths.get('notebooks_dir', 'notebooks'),
        root / Path(config.get('shap', {}).get('cache_dir', 'data/cache/shap')),
        root / Path(config.get('lime', {}).get('cache_dir', 'data/cache/lime')),
    ]
    
    # Crear directorios
    created = []
    for directory in dirs_to_create:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(str(directory))
    
    if created:
        logger.info(f"✅ Directorios creados: {', '.join(created)}")
    else:
        logger.info("✅ Todos los directorios ya existen")


def setup_project() -> Config:
    """
    Setup completo del proyecto:
    1. Carga configuración
    2. Valida configuración
    3. Crea directorios necesarios
    4. Retorna objeto Config listo para usar
    
    Returns:
        Config: Objeto de configuración validado y listo
    
    Example:
        >>> from src.config import setup_project
        >>> config = setup_project()
        >>> print(config.model.name)
    """
    logger.info("🚀 Iniciando setup del proyecto...")
    
    # Cargar y validar configuración
    config = Config()
    
    # Crear directorios
    create_directories(config.raw)
    
    logger.info("✅ Setup completado exitosamente")
    return config


# ----------------------------------------------------------------------------
# UTILIDADES ADICIONALES
# ----------------------------------------------------------------------------

def print_config_summary(config: Config) -> None:
    """
    Imprime un resumen legible de la configuración.
    
    Args:
        config: Objeto Config
    """
    print("\n" + "="*60)
    print("📋 RESUMEN DE CONFIGURACIÓN")
    print("="*60)
    
    print(f"\n🤖 MODELO:")
    print(f"  • Nombre: {config.model.name}")
    print(f"  • Device: {config.model.device}")
    print(f"  • Max length: {config.model.max_length} tokens")
    print(f"  • Batch size: {config.model.batch_size}")
    
    print(f"\n📊 DATASET:")
    print(f"  • Nombre: {config.dataset.name}")
    print(f"  • Train size: {config.dataset.train_size or 'completo'}")
    print(f"  • Test size: {config.dataset.test_size or 'completo'}")
    print(f"  • Validación: {config.dataset.validation_split:.0%}")
    
    print(f"\n🔍 SHAP:")
    print(f"  • Num samples: {config.shap.num_samples:,}")
    print(f"  • Masking: {config.shap.masking_strategy}")
    print(f"  • Cache: {'✅' if config.shap.use_cache else '❌'}")
    
    print(f"\n🎯 LIME:")
    print(f"  • Num samples: {config.lime.num_samples:,}")
    print(f"  • Num features: {config.lime.num_features}")
    print(f"  • Perturbation: {config.lime.perturbation_strategy}")
    
    print("\n" + "="*60 + "\n")


# ----------------------------------------------------------------------------
# EXPORTS
# ----------------------------------------------------------------------------

__all__ = [
    'Config',
    'ModelConfig',
    'DatasetConfig',
    'SHAPConfig',
    'LIMEConfig',
    'PathsConfig',
    'load_config',
    'create_directories',
    'setup_project',
    'print_config_summary'
]


# ----------------------------------------------------------------------------
# EJEMPLO DE USO
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Ejemplo de uso del módulo de configuración.
    Ejecutar: python -m src.config
    """
    
    # Setup completo
    config = setup_project()
    
    # Imprimir resumen
    print_config_summary(config)
    
    # Ejemplos de acceso
    print("📝 EJEMPLOS DE ACCESO:")
    print(f"  config.model.name = {config.model.name}")
    print(f"  config.shap.num_samples = {config.shap.num_samples}")
    print(f"  config.get('model.device') = {config.get('model.device')}")
    print(f"  config.get('missing.key', 'default') = {config.get('missing.key', 'default')}")