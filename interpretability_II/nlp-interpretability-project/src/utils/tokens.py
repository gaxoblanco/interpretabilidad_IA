"""
🔧 Utilidades para Manejo de Tokens
Módulo centralizado para filtrar y procesar tokens no semánticos
"""

from typing import List, Set, Tuple
import numpy as np


class TokenFilter:
    """
    Clase para gestionar el filtrado de tokens no semánticos
    """

    # Conjuntos de tokens a filtrar (definidos una sola vez)
    PUNCTUATION = {
        ',', '.', '!', '?', ';', ':', '-', '(', ')', '[', ']',
        '"', "'", '...', '--', '``', "''", '/', '\\', '{', '}',
        '<', '>', '|', '~', '`', '@', '#', '$', '%', '^', '&', '*',
        '+', '=', '_'
    }

    CONNECTORS = {
        # Artículos
        'the', 'a', 'an',

        # Preposiciones
        'of', 'to', 'in', 'on', 'at', 'for', 'with', 'by', 'from',
        'as', 'into', 'about', 'after', 'before', 'between', 'through',

        # Conjunciones
        'and', 'or', 'but', 'nor', 'so', 'yet',

        # Verbos auxiliares comunes
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had',

        # Pronombres comunes (opcional - puedes comentar si quieres mantenerlos)
        'it', 'this', 'that', 'these', 'those'
    }

    # Tokens especiales de diferentes tokenizadores
    SPECIAL_TOKENS = {
        '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]',  # BERT
        '<s>', '</s>', '<pad>', '<unk>', '<mask>',      # RoBERTa
        'Ġ', '##', '▁'                                  # Prefijos
    }

    @classmethod
    def is_semantic(cls, token: str) -> bool:
        """
        Determina si un token es semánticamente significativo

        Args:
            token: Token a evaluar

        Returns:
            bool: True si el token es semántico, False si debe filtrarse
        """
        # Limpiar prefijos de tokenizador
        clean_token = token.replace('Ġ', '').replace(
            '##', '').replace('▁', '').lower().strip()

        # Verificar si es token vacío
        if not clean_token:
            return False

        # Verificar si es puntuación
        if clean_token in cls.PUNCTUATION:
            return False

        # Verificar si es conector
        if clean_token in cls.CONNECTORS:
            return False

        # Verificar si es token especial
        if clean_token in cls.SPECIAL_TOKENS:
            return False

        # Si pasa todas las pruebas, es semántico
        return True

    @classmethod
    def filter_importance_values(cls, tokens: List[str], importance_values: np.ndarray) -> np.ndarray:
        """
        Filtra valores de importancia poniendo en 0 los tokens no semánticos

        Args:
            tokens: Lista de tokens
            importance_values: Array de valores de importancia

        Returns:
            np.ndarray: Array filtrado con 0s en posiciones no semánticas
        """
        filtered_values = importance_values.copy()

        for i, token in enumerate(tokens):
            if i >= len(filtered_values):
                break

            if not cls.is_semantic(token):
                filtered_values[i] = 0.0

        return filtered_values

    @classmethod
    def get_semantic_indices(cls, tokens: List[str]) -> List[int]:
        """
        Obtiene índices de tokens semánticamente significativos

        Args:
            tokens: Lista de tokens

        Returns:
            List[int]: Lista de índices de tokens semánticos
        """
        return [i for i, token in enumerate(tokens) if cls.is_semantic(token)]

    @classmethod
    def filter_token_list(cls, tokens: List[str]) -> List[str]:
        """
        Filtra una lista de tokens, manteniendo solo los semánticos

        Args:
            tokens: Lista de tokens original

        Returns:
            List[str]: Lista filtrada de tokens semánticos
        """
        return [token for token in tokens if cls.is_semantic(token)]

    @classmethod
    def get_top_k_semantic_indices(cls, tokens: List[str], importance_values: np.ndarray,
                                   k: int) -> Tuple[List[int], np.ndarray]:
        """
        Obtiene los top-k índices de tokens semánticos más importantes

        Args:
            tokens: Lista de tokens
            importance_values: Array de valores de importancia
            k: Número de tokens a seleccionar

        Returns:
            Tuple[List[int], np.ndarray]: (índices top-k, valores filtrados)
        """
        # Primero filtrar valores poniendo 0 en no semánticos
        filtered_values = cls.filter_importance_values(
            tokens, importance_values)

        # Obtener índices de tokens semánticos
        semantic_indices = cls.get_semantic_indices(tokens)

        # Si hay menos tokens semánticos que k, ajustar k
        k_adjusted = min(k, len(semantic_indices))

        if k_adjusted == 0:
            return [], filtered_values

        # Obtener top-k solo de tokens semánticos
        # Filtrar solo valores semánticos para argsort
        semantic_values = filtered_values[semantic_indices]

        # Obtener índices relativos en el array semántico
        top_k_relative = np.argsort(np.abs(semantic_values))[-k_adjusted:]

        # Convertir a índices absolutos en el array original
        top_k_absolute = [semantic_indices[i] for i in top_k_relative]

        return top_k_absolute, filtered_values

    @classmethod
    def clean_token(cls, token: str) -> str:
        """
        Limpia un token de prefijos de tokenizador

        Args:
            token: Token a limpiar

        Returns:
            str: Token limpio
        """
        return token.replace('Ġ', '').replace('##', '').replace('▁', '').strip()

    @classmethod
    def add_to_filters(cls, new_punctuation: Set[str] = None,
                       new_connectors: Set[str] = None):
        """
        Permite agregar tokens adicionales a los filtros

        Args:
            new_punctuation: Nuevos tokens de puntuación
            new_connectors: Nuevos conectores
        """
        if new_punctuation:
            cls.PUNCTUATION.update(new_punctuation)

        if new_connectors:
            cls.CONNECTORS.update(new_connectors)
