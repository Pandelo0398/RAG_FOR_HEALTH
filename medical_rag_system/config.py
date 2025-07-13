"""
Configuración centralizada del Sistema RAG Médico
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class MedicalSystemConfig:
    """Configuración centralizada del sistema médico"""
    
    # Configuración de embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM: int = 384
    EMBEDDING_CACHE_DIR: str = "./embedding_cache"
    
    # Configuración de chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Configuración de ChromaDB
    CHROMA_PERSIST_DIR: str = "./medical_chroma_db"
    CHROMA_COLLECTION_NAME: str = "medical_documents"
    
    # Configuración de retrieval
    SIMILARITY_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Configuración del LLM Qwen
    LLM_MODEL: str = "Qwen/Qwen3-14B"
    LLM_API_BASE: str = "http://localhost:8000"
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.1
    LLM_CONTEXT_WINDOW: int = 32000
    LLM_TIMEOUT: int = 30
    LLM_MAX_RETRIES: int = 3
    
    # Configuración de logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "medical_rag.log"
    
    # Configuración de la aplicación
    APP_NAME: str = "Medical RAG System"
    APP_VERSION: str = "1.0.0"
    APP_DEBUG: bool = False
    
    @classmethod
    def from_env(cls) -> 'MedicalSystemConfig':
        """Crear configuración desde variables de entorno"""
        return cls(
            EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", cls.EMBEDDING_MODEL),
            CHROMA_PERSIST_DIR=os.getenv("CHROMA_PERSIST_DIR", cls.CHROMA_PERSIST_DIR),
            LLM_API_BASE=os.getenv("LLM_API_BASE", cls.LLM_API_BASE),
            LLM_MODEL=os.getenv("LLM_MODEL", cls.LLM_MODEL),
            LOG_LEVEL=os.getenv("LOG_LEVEL", cls.LOG_LEVEL),
            APP_DEBUG=os.getenv("APP_DEBUG", "false").lower() == "true"
        )
    
    def validate(self) -> bool:
        """Validar configuración"""
        try:
            # Validar directorios
            Path(self.CHROMA_PERSIST_DIR).parent.mkdir(parents=True, exist_ok=True)
            Path(self.EMBEDDING_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            
            # Validar valores numéricos
            assert self.CHUNK_SIZE > 0, "CHUNK_SIZE debe ser positivo"
            assert self.SIMILARITY_TOP_K > 0, "SIMILARITY_TOP_K debe ser positivo"
            assert 0 <= self.SIMILARITY_THRESHOLD <= 1, "SIMILARITY_THRESHOLD debe estar entre 0 y 1"
            
            return True
        except Exception as e:
            print(f"❌ Error validando configuración: {e}")
            return False


# Instancia global de configuración
config = MedicalSystemConfig.from_env()
