"""
Modelos de datos para el sistema médico
"""

from .patient import PatientData
from .medical import ClinicalEntity, MedicalProcessingResult
from .document import DocumentMetadata, ProcessingStatus

__all__ = [
    'PatientData',
    'ClinicalEntity', 
    'MedicalProcessingResult',
    'DocumentMetadata',
    'ProcessingStatus'
]
