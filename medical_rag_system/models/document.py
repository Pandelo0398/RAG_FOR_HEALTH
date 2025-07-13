"""
Modelos para gestión de documentos
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class ProcessingStatus(str, Enum):
    """Estados de procesamiento de documentos"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DocumentType(str, Enum):
    """Tipos de documentos médicos"""
    CLINICAL_NOTE = "clinical_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    RADIOLOGY_REPORT = "radiology_report"
    PATHOLOGY_REPORT = "pathology_report"
    LAB_RESULT = "lab_result"
    PRESCRIPTION = "prescription"
    PROGRESS_NOTE = "progress_note"
    GENERAL = "general"


class DocumentMetadata(BaseModel):
    """Metadatos de documento médico"""
    
    # Identificación
    document_id: str = Field(..., description="ID único del documento")
    document_type: DocumentType = DocumentType.GENERAL
    title: Optional[str] = None
    
    # Información temporal
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None
    document_date: Optional[datetime] = None
    
    # Información del paciente (opcional)
    patient_id: Optional[str] = None
    patient_age: Optional[int] = Field(None, ge=0, le=150)
    patient_sex: Optional[str] = None
    
    # Información médica
    service: Optional[str] = None
    department: Optional[str] = None
    physician: Optional[str] = None
    
    # Estado de procesamiento
    status: ProcessingStatus = ProcessingStatus.PENDING
    processing_notes: List[str] = Field(default_factory=list)
    
    # Métricas
    text_length: Optional[int] = None
    chunk_count: Optional[int] = None
    processing_time_seconds: Optional[float] = None
    
    # Información adicional
    tags: List[str] = Field(default_factory=list)
    priority: int = Field(default=0, ge=0, le=10)
    language: str = "es"
    
    # Datos personalizados
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    def add_tag(self, tag: str):
        """Agregar tag si no existe"""
        if tag.strip() and tag not in self.tags:
            self.tags.append(tag.strip())
    
    def update_status(self, status: ProcessingStatus, note: Optional[str] = None):
        """Actualizar estado con nota opcional"""
        self.status = status
        self.modified_at = datetime.now()
        if note:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.processing_notes.append(f"[{timestamp}] {note}")
    
    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convertir a metadatos para ChromaDB"""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type.value,
            "created_at": self.created_at.isoformat(),
            "patient_age": self.patient_age,
            "patient_sex": self.patient_sex,
            "service": self.service,
            "department": self.department,
            "language": self.language,
            "priority": self.priority,
            "text_length": self.text_length,
            "chunk_count": self.chunk_count,
            **{f"custom_{k}": v for k, v in self.custom_fields.items() if isinstance(v, (str, int, float, bool))}
        }


class DocumentChunk(BaseModel):
    """Chunk de documento para almacenamiento vectorial"""
    
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    
    # Metadatos específicos del chunk
    chunk_type: str = "content"  # content, header, footer, etc.
    tokens_count: Optional[int] = None
    
    # Ubicación en el documento original
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    # Información semántica
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    
    # Relación con otros chunks
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = Field(default_factory=list)
    
    def __str__(self) -> str:
        return f"Chunk({self.chunk_id}, {len(self.text)} chars)"
