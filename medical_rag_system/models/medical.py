"""
Modelos médicos con validación Pydantic
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class ClinicalEntity(BaseModel):
    """Entidad clínica con validación"""
    entity: str = Field(..., description="Nombre de la entidad clínica")
    type: str = Field(..., description="Tipo de entidad")
    severity: Optional[str] = Field(None, description="Severidad")
    location: Optional[str] = Field(None, description="Ubicación anatómica")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confianza de la extracción")
    
    @validator('entity')
    def validate_entity(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Entity debe tener al menos 2 caracteres")
        return v.strip()
    
    @validator('type')
    def validate_type(cls, v):
        valid_types = [
            'diagnosis', 'symptom', 'medication', 'procedure', 
            'condition', 'finding', 'anatomical_location', 'general'
        ]
        if v.lower() not in valid_types:
            return 'general'  # Valor por defecto
        return v.lower()


class MedicalProcessingResult(BaseModel):
    """Resultado del procesamiento médico"""
    
    # Texto procesado
    cleaned_text: str
    original_text: str = ""
    
    # Diagnósticos
    main_diagnosis: str
    secondary_diagnoses: List[str] = Field(default_factory=list)
    
    # Codificación y entidades
    cie10_codes: List[str] = Field(default_factory=list)
    expanded_acronyms: Dict[str, str] = Field(default_factory=dict)
    clinical_entities: List[ClinicalEntity] = Field(default_factory=list)
    
    # Información clínica
    symptoms: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    anatomical_location: Optional[str] = None
    
    # Métricas de calidad
    confidence_scores: List[float] = Field(default_factory=list)
    processing_notes: List[str] = Field(default_factory=list)
    
    # Metadatos de procesamiento
    processing_method: str = "unknown"
    processing_time: Optional[str] = None
    model_version: Optional[str] = None
    success: bool = True
    
    @validator('confidence_scores')
    def validate_confidence(cls, v):
        return [max(0, min(1, score)) for score in v]
    
    @validator('cie10_codes')
    def validate_cie10(cls, v):
        # Validar formato básico de códigos CIE-10
        valid_codes = []
        for code in v:
            if isinstance(code, str) and len(code.strip()) >= 3:
                valid_codes.append(code.strip().upper())
        return valid_codes
    
    def get_overall_confidence(self) -> float:
        """Obtener confianza general del procesamiento"""
        if not self.confidence_scores:
            return 0.5  # Confianza por defecto
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    def add_processing_note(self, note: str):
        """Agregar nota de procesamiento"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.processing_notes.append(f"[{timestamp}] {note}")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Crear resumen del resultado"""
        return {
            "main_diagnosis": self.main_diagnosis,
            "cie10_codes": self.cie10_codes,
            "entities_count": len(self.clinical_entities),
            "symptoms_count": len(self.symptoms),
            "confidence": self.get_overall_confidence(),
            "method": self.processing_method,
            "success": self.success
        }


class ProcessingRequest(BaseModel):
    """Solicitud de procesamiento médico"""
    text: str = Field(..., min_length=10, description="Texto médico a procesar")
    patient_age: Optional[int] = Field(None, ge=0, le=150)
    patient_sex: Optional[str] = None
    service: Optional[str] = None
    use_vector_store: bool = True
    max_retries: int = Field(default=3, ge=1, le=10)
    
    @validator('text')
    def validate_text(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Texto debe tener al menos 10 caracteres")
        return v.strip()
    
    @validator('patient_sex')
    def validate_sex(cls, v):
        if v is None:
            return v
        if v.upper() in ['M', 'F', 'MALE', 'FEMALE']:
            return v.upper()
        return None
