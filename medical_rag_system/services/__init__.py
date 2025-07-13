"""
Capa de servicios - Lógica de negocio médico
"""

from .medical_processor import MedicalProcessor
from .document_manager import DocumentManager
from .fallback_processor import FallbackProcessor

__all__ = [
    'MedicalProcessor',
    'DocumentManager', 
    'FallbackProcessor'
]
