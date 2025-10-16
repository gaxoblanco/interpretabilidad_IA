"""
============================================================================
UTILS MODULE - Utilidades del Proyecto
============================================================================

Este módulo proporciona utilidades para:
- Carga y procesamiento de datasets (IMDb)
- Preprocessing de texto
- Gestión de datos

Componentes principales:
    - DataLoader: Clase para cargar y procesar el dataset IMDb

Uso:
    from src.utils import DataLoader
    from src.config import Config
    
    config = Config()
    data = DataLoader(config)
    train = data.get_train_data()

Autor: Proyecto Módulo II
Fecha: 2025-01-15
============================================================================
"""

# Importar clase principal
from .data_loader import DataLoader

# Definir qué se exporta cuando se hace "from src.utils import *"
__all__ = [
    'DataLoader'
]

# Información del módulo
__version__ = '1.0.0'
__author__ = 'Estudiante Módulo II'

# Logging
import logging
logger = logging.getLogger(__name__)
logger.info("✅ Módulo 'utils' cargado correctamente")
