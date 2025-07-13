"""
Sistema RAG Médico - Clase principal de orquestación
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from ..config import config
from ..models.patient import PatientData
from ..models.medical import ProcessingRequest, MedicalProcessingResult
from ..models.document import DocumentType
from ..infrastructure.llm_client import QwenClient, FallbackLLMClient
from ..infrastructure.vector_store import ChromaVectorStore
from ..infrastructure.embedding_service import EmbeddingService
from ..services.medical_processor import MedicalProcessor
from ..services.document_manager import DocumentManager
from ..services.fallback_processor import FallbackProcessor


class MedicalRAGSystem:
    """Sistema RAG Médico completo - Clase principal de orquestación"""
    
    def __init__(self, 
                 use_qwen: bool = True,
                 custom_config: Optional[Dict[str, Any]] = None):
        """
        Inicializar sistema RAG médico
        
        Args:
            use_qwen: Si usar cliente Qwen real o fallback
            custom_config: Configuración personalizada
        """
        
        self.config = config
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        print(f"🏥 Inicializando Sistema RAG Médico v{self.config.APP_VERSION}")
        print("=" * 60)
        
        # Validar configuración
        if not self.config.validate():
            raise ValueError("Configuración inválida")
        
        # Inicializar componentes de infraestructura
        self._initialize_infrastructure(use_qwen)
        
        # Inicializar servicios
        self._initialize_services()
        
        # Estadísticas del sistema
        self.system_stats = {
            "initialized_at": datetime.now().isoformat(),
            "documents_processed": 0,
            "diagnoses_processed": 0
        }
        
        print("✅ Sistema RAG Médico inicializado exitosamente")
    
    def _initialize_infrastructure(self, use_qwen: bool):
        """Inicializar componentes de infraestructura"""
        
        print("🔧 Inicializando infraestructura...")
        
        # Embedding Service
        self.embedding_service = EmbeddingService()
        
        # Vector Store
        try:
            self.vector_store = ChromaVectorStore(
                embedding_service=self.embedding_service
            )
            print("✅ Vector store inicializado")
        except Exception as e:
            print(f"⚠️ Error inicializando vector store: {e}")
            self.vector_store = None
        
        # LLM Client
        if use_qwen:
            self.llm_client = QwenClient()
            if not self.llm_client.is_available():
                print("⚠️ Qwen no disponible, usando fallback")
                self.llm_client = FallbackLLMClient()
        else:
            print("🛠️ Usando cliente LLM de fallback")
            self.llm_client = FallbackLLMClient()
    
    def _initialize_services(self):
        """Inicializar servicios de negocio"""
        
        print("⚙️ Inicializando servicios...")
        
        # Fallback Processor
        self.fallback_processor = FallbackProcessor()
        
        # Medical Processor
        self.medical_processor = MedicalProcessor(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            fallback_processor=self.fallback_processor
        )
        
        # Document Manager
        if self.vector_store:
            self.document_manager = DocumentManager(self.vector_store)
        else:
            self.document_manager = None
        
        print("✅ Servicios inicializados")
    
    async def process_diagnosis(self,
                               text: str,
                               patient_age: Optional[int] = None,
                               patient_sex: Optional[str] = None,
                               service: Optional[str] = None,
                               use_vector_store: bool = True) -> MedicalProcessingResult:
        """
        Procesar diagnóstico médico
        
        Args:
            text: Texto del diagnóstico
            patient_age: Edad del paciente
            patient_sex: Sexo del paciente
            service: Servicio médico
            use_vector_store: Si usar contexto del vector store
        """
        
        # Crear solicitud
        request = ProcessingRequest(
            text=text,
            patient_age=patient_age,
            patient_sex=patient_sex,
            service=service,
            use_vector_store=use_vector_store and self.vector_store is not None
        )
        
        # Procesar
        result = await self.medical_processor.process_diagnosis(request)
        
        # Actualizar estadísticas
        self.system_stats["diagnoses_processed"] += 1
        
        return result
    
    def add_medical_document(self,
                            text: str,
                            document_type: DocumentType = DocumentType.GENERAL,
                            title: Optional[str] = None,
                            patient_id: Optional[str] = None,
                            patient_age: Optional[int] = None,
                            patient_sex: Optional[str] = None,
                            service: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Agregar documento médico al sistema
        
        Returns:
            Document ID si se agrega exitosamente, None si error
        """
        
        if not self.document_manager:
            print("❌ Document manager no disponible")
            return None
        
        try:
            doc_id = self.document_manager.add_document(
                text=text,
                document_type=document_type,
                title=title,
                patient_id=patient_id,
                patient_age=patient_age,
                patient_sex=patient_sex,
                service=service,
                metadata=metadata
            )
            
            self.system_stats["documents_processed"] += 1
            return doc_id
            
        except Exception as e:
            print(f"❌ Error agregando documento: {e}")
            return None
    
    def search_medical_documents(self,
                                query: str,
                                top_k: int = 5,
                                document_type: Optional[DocumentType] = None,
                                service: Optional[str] = None) -> List[Dict[str, Any]]:
        """Buscar documentos médicos similares"""
        
        if not self.document_manager:
            print("❌ Document manager no disponible")
            return []
        
        return self.document_manager.search_documents(
            query=query,
            top_k=top_k,
            document_type=document_type,
            service=service
        )
    
    async def test_system_health(self) -> Dict[str, Any]:
        """Ejecutar diagnóstico de salud del sistema"""
        
        print("🔍 Ejecutando diagnóstico de salud del sistema...")
        
        health_status = {
            "system_healthy": True,
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Test LLM
        try:
            llm_test = await self.llm_client.test_connection() if hasattr(self.llm_client, 'test_connection') else {"success": True}
            health_status["components"]["llm"] = {
                "available": self.llm_client.is_available(),
                "test_result": llm_test
            }
        except Exception as e:
            health_status["components"]["llm"] = {
                "available": False,
                "error": str(e)
            }
            health_status["system_healthy"] = False
        
        # Test Vector Store
        if self.vector_store:
            try:
                stats = self.vector_store.get_stats()
                health_status["components"]["vector_store"] = {
                    "available": True,
                    "stats": stats
                }
            except Exception as e:
                health_status["components"]["vector_store"] = {
                    "available": False,
                    "error": str(e)
                }
                health_status["system_healthy"] = False
        else:
            health_status["components"]["vector_store"] = {
                "available": False,
                "error": "Not initialized"
            }
        
        # Test Embedding Service
        try:
            embedding_info = self.embedding_service.get_model_info()
            health_status["components"]["embedding_service"] = {
                "available": True,
                "info": embedding_info
            }
        except Exception as e:
            health_status["components"]["embedding_service"] = {
                "available": False,
                "error": str(e)
            }
        
        # Test Medical Processor
        try:
            processor_stats = self.medical_processor.get_processing_stats()
            health_status["components"]["medical_processor"] = {
                "available": True,
                "stats": processor_stats
            }
        except Exception as e:
            health_status["components"]["medical_processor"] = {
                "available": False,
                "error": str(e)
            }
        
        return health_status
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas del sistema"""
        
        stats = {
            "system": self.system_stats,
            "config": {
                "llm_model": self.config.LLM_MODEL,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "chunk_size": self.config.CHUNK_SIZE,
                "top_k": self.config.SIMILARITY_TOP_K
            }
        }
        
        # Estadísticas de procesamiento
        if hasattr(self.medical_processor, 'get_processing_stats'):
            stats["processing"] = self.medical_processor.get_processing_stats()
        
        # Estadísticas de documentos
        if self.document_manager:
            stats["documents"] = self.document_manager.get_stats()
        
        # Estadísticas de vector store
        if self.vector_store:
            stats["vector_store"] = self.vector_store.get_stats()
        
        return stats
    
    def reset_system_stats(self):
        """Resetear estadísticas del sistema"""
        
        self.system_stats = {
            "initialized_at": datetime.now().isoformat(),
            "documents_processed": 0,
            "diagnoses_processed": 0
        }
        
        if hasattr(self.medical_processor, 'reset_stats'):
            self.medical_processor.reset_stats()
        
        print("✅ Estadísticas del sistema reseteadas")
    
    def shutdown(self):
        """Cerrar sistema limpiamente"""
        
        print("🛑 Cerrando sistema RAG médico...")
        
        # Cerrar conexiones si es necesario
        if hasattr(self.llm_client, 'shutdown'):
            self.llm_client.shutdown()
        
        print("✅ Sistema cerrado correctamente")
