"""
Sistema RAG Médico Mejorado con Integraciones LlamaIndex
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from ..config import config
from ..models.patient import PatientData
from ..models.medical import ProcessingRequest, MedicalProcessingResult
from ..models.document import DocumentType
from ..infrastructure.llama_index_integration import LlamaIndexMedicalRAG
from ..services.medical_agent import MedicalAgentSystem
from ..services.evaluation_service import MedicalEvaluationService

logger = logging.getLogger(__name__)


class EnhancedMedicalRAGSystem:
    """Sistema RAG Médico Mejorado con todas las integraciones de LlamaIndex"""
    
    def __init__(self, 
                 use_llama_index: bool = True,
                 enable_agents: bool = True,
                 enable_evaluation: bool = True,
                 custom_config: Optional[Dict[str, Any]] = None):
        """
        Inicializar sistema RAG médico mejorado
        
        Args:
            use_llama_index: Si usar integración completa con LlamaIndex
            enable_agents: Si habilitar sistema de agentes
            enable_evaluation: Si habilitar sistema de evaluación
            custom_config: Configuración personalizada
        """
        
        self.config = config
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        print(f"🏥 Inicializando Sistema RAG Médico Mejorado v{self.config.APP_VERSION}")
        print("=" * 70)
        
        # Validar configuración
        if not self.config.validate():
            raise ValueError("Configuración inválida")
        
        # Componentes principales
        self.llama_rag = None
        self.agent_system = None
        self.evaluation_service = None
        
        # Estadísticas del sistema
        self.system_stats = {
            "initialized_at": datetime.now().isoformat(),
            "documents_processed": 0,
            "diagnoses_processed": 0,
            "agent_queries": 0,
            "evaluations_performed": 0
        }
        
        # Inicializar componentes
        self._initialize_components(use_llama_index, enable_agents, enable_evaluation)
        
        print("✅ Sistema RAG Médico Mejorado inicializado exitosamente")
    
    def _initialize_components(self, use_llama_index: bool, enable_agents: bool, enable_evaluation: bool):
        """Inicializar componentes del sistema"""
        
        print("🔧 Inicializando componentes mejorados...")
        
        # 1. Sistema RAG con LlamaIndex
        if use_llama_index:
            try:
                self.llama_rag = LlamaIndexMedicalRAG(
                    persist_dir=self.config.CHROMA_PERSIST_DIR,
                    collection_name=self.config.CHROMA_COLLECTION_NAME,
                    embedding_model=self.config.EMBEDDING_MODEL,
                    llm_config={
                        "model": self.config.LLM_MODEL,
                        "api_base": self.config.LLM_API_BASE,
                        "max_tokens": self.config.LLM_MAX_TOKENS,
                        "temperature": self.config.LLM_TEMPERATURE,
                        "timeout": self.config.LLM_TIMEOUT
                    }
                )
                print("✅ Sistema RAG con LlamaIndex inicializado")
            except Exception as e:
                print(f"❌ Error inicializando LlamaIndex RAG: {e}")
                self.llama_rag = None
        
        # 2. Sistema de Agentes
        if enable_agents and self.llama_rag:
            try:
                self.agent_system = MedicalAgentSystem(self.llama_rag)
                print("✅ Sistema de Agentes Médicos inicializado")
            except Exception as e:
                print(f"❌ Error inicializando agentes: {e}")
                self.agent_system = None
        
        # 3. Sistema de Evaluación
        if enable_evaluation and self.llama_rag:
            try:
                self.evaluation_service = MedicalEvaluationService(self.llama_rag)
                print("✅ Sistema de Evaluación inicializado")
            except Exception as e:
                print(f"❌ Error inicializando evaluación: {e}")
                self.evaluation_service = None
        
        print("✅ Componentes inicializados")
    
    async def process_diagnosis_enhanced(self,
                                        text: str,
                                        patient_age: Optional[int] = None,
                                        patient_sex: Optional[str] = None,
                                        service: Optional[str] = None,
                                        use_agents: bool = True,
                                        evaluate_response: bool = True) -> MedicalProcessingResult:
        """
        Procesar diagnóstico médico con capacidades mejoradas
        
        Args:
            text: Texto del diagnóstico
            patient_age: Edad del paciente
            patient_sex: Sexo del paciente
            service: Servicio médico
            use_agents: Si usar agentes especializados
            evaluate_response: Si evaluar la respuesta
        """
        
        start_time = datetime.now()
        
        try:
            # Crear solicitud
            request = ProcessingRequest(
                text=text,
                patient_age=patient_age,
                patient_sex=patient_sex,
                service=service,
                use_vector_store=True
            )
            
            # Procesar con agentes si están disponibles
            if use_agents and self.agent_system:
                result = await self.agent_system.run_medical_workflow(request)
                self.system_stats["agent_queries"] += 1
            elif self.llama_rag:
                # Usar sistema RAG básico
                rag_result = await self.llama_rag.query_medical_system(
                    query=text,
                    patient_context={
                        "age": patient_age,
                        "sex": patient_sex,
                        "service": service
                    }
                )
                
                result = MedicalProcessingResult(
                    original_text=text,
                    processed_text=text,
                    diagnosis_analysis=rag_result.get("response", ""),
                    treatment_suggestions="",
                    research_findings="",
                    confidence_score=0.8,
                    processing_time=0.0,
                    metadata={
                        "llama_index_used": True,
                        "retrieved_nodes": len(rag_result.get("retrieved_nodes", [])),
                        "workflow_used": rag_result.get("workflow_used", False)
                    }
                )
            else:
                # Fallback básico
                result = MedicalProcessingResult(
                    original_text=text,
                    processed_text=text,
                    diagnosis_analysis="Sistema no disponible",
                    treatment_suggestions="",
                    research_findings="",
                    confidence_score=0.0,
                    processing_time=0.0,
                    metadata={"error": "No hay sistema disponible"}
                )
            
            # Evaluar respuesta si está habilitado
            if evaluate_response and self.evaluation_service:
                evaluation = await self.evaluation_service.evaluate_medical_response(
                    query=text,
                    response=result.diagnosis_analysis,
                    context_nodes=result.metadata.get("retrieved_nodes", [])
                )
                
                result.metadata["evaluation"] = evaluation
                result.confidence_score = evaluation.get("average_score", result.confidence_score)
                self.system_stats["evaluations_performed"] += 1
            
            # Calcular tiempo de procesamiento
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            # Actualizar estadísticas
            self.system_stats["diagnoses_processed"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error procesando diagnóstico: {e}")
            return MedicalProcessingResult(
                original_text=text,
                processed_text=text,
                diagnosis_analysis=f"Error en procesamiento: {str(e)}",
                treatment_suggestions="",
                research_findings="",
                confidence_score=0.0,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    async def add_medical_document_enhanced(self,
                                           text: str,
                                           document_type: DocumentType = DocumentType.GENERAL,
                                           title: Optional[str] = None,
                                           patient_id: Optional[str] = None,
                                           patient_age: Optional[int] = None,
                                           patient_sex: Optional[str] = None,
                                           service: Optional[str] = None,
                                           metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Agregar documento médico usando LlamaIndex"""
        
        if not self.llama_rag:
            print("❌ Sistema LlamaIndex no disponible")
            return None
        
        try:
            doc_id = await self.llama_rag.add_medical_document(
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
    
    async def query_with_agent(self,
                              query: str,
                              agent_type: str = "diagnosis",
                              patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Consultar usando agentes especializados"""
        
        if not self.agent_system:
            return {
                "error": "Sistema de agentes no disponible",
                "available_agents": []
            }
        
        try:
            result = await self.agent_system.process_medical_query(
                query=query,
                agent_type=agent_type,
                patient_context=patient_context
            )
            
            self.system_stats["agent_queries"] += 1
            return result
            
        except Exception as e:
            logger.error(f"❌ Error consultando con agente: {e}")
            return {
                "error": str(e),
                "agent_used": agent_type,
                "timestamp": datetime.now().isoformat()
            }
    
    def search_medical_documents_enhanced(self,
                                         query: str,
                                         top_k: int = 5,
                                         document_type: Optional[DocumentType] = None,
                                         service: Optional[str] = None) -> List[Dict[str, Any]]:
        """Buscar documentos médicos usando LlamaIndex"""
        
        if not self.llama_rag:
            print("❌ Sistema LlamaIndex no disponible")
            return []
        
        return self.llama_rag.search_medical_documents(
            query=query,
            top_k=top_k,
            document_type=document_type,
            service=service
        )
    
    async def evaluate_system_performance(self,
                                         test_queries: List[str],
                                         expected_responses: List[str] = None) -> Dict[str, Any]:
        """Evaluar rendimiento del sistema"""
        
        if not self.evaluation_service:
            return {
                "error": "Sistema de evaluación no disponible"
            }
        
        return await self.evaluation_service.evaluate_system_performance(
            test_queries=test_queries,
            expected_responses=expected_responses
        )
    
    async def test_system_health_enhanced(self) -> Dict[str, Any]:
        """Ejecutar diagnóstico de salud del sistema mejorado"""
        
        print("🔍 Ejecutando diagnóstico de salud del sistema mejorado...")
        
        health_status = {
            "system_healthy": True,
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Test LlamaIndex RAG
        if self.llama_rag:
            try:
                llama_stats = self.llama_rag.get_system_stats()
                health_status["components"]["llama_index_rag"] = {
                    "available": True,
                    "stats": llama_stats
                }
            except Exception as e:
                health_status["components"]["llama_index_rag"] = {
                    "available": False,
                    "error": str(e)
                }
                health_status["system_healthy"] = False
        else:
            health_status["components"]["llama_index_rag"] = {
                "available": False,
                "error": "No inicializado"
            }
        
        # Test Agentes
        if self.agent_system:
            try:
                agent_stats = self.agent_system.get_agent_stats()
                health_status["components"]["agents"] = {
                    "available": True,
                    "stats": agent_stats
                }
            except Exception as e:
                health_status["components"]["agents"] = {
                    "available": False,
                    "error": str(e)
                }
        else:
            health_status["components"]["agents"] = {
                "available": False,
                "error": "No inicializado"
            }
        
        # Test Evaluación
        if self.evaluation_service:
            try:
                eval_stats = self.evaluation_service.get_evaluation_stats()
                health_status["components"]["evaluation"] = {
                    "available": True,
                    "stats": eval_stats
                }
            except Exception as e:
                health_status["components"]["evaluation"] = {
                    "available": False,
                    "error": str(e)
                }
        else:
            health_status["components"]["evaluation"] = {
                "available": False,
                "error": "No inicializado"
            }
        
        return health_status
    
    def get_system_stats_enhanced(self) -> Dict[str, Any]:
        """Obtener estadísticas completas del sistema mejorado"""
        
        stats = {
            "system": self.system_stats,
            "config": {
                "llm_model": self.config.LLM_MODEL,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "chunk_size": self.config.CHUNK_SIZE,
                "top_k": self.config.SIMILARITY_TOP_K
            }
        }
        
        # Estadísticas de LlamaIndex
        if self.llama_rag:
            stats["llama_index"] = self.llama_rag.get_system_stats()
        
        # Estadísticas de agentes
        if self.agent_system:
            stats["agents"] = self.agent_system.get_agent_stats()
        
        # Estadísticas de evaluación
        if self.evaluation_service:
            stats["evaluation"] = self.evaluation_service.get_evaluation_stats()
        
        return stats
    
    def get_available_features(self) -> Dict[str, bool]:
        """Obtener características disponibles del sistema"""
        
        return {
            "llama_index_integration": self.llama_rag is not None,
            "agent_system": self.agent_system is not None,
            "evaluation_service": self.evaluation_service is not None,
            "enhanced_processing": self.llama_rag is not None,
            "medical_agents": self.agent_system is not None,
            "quality_evaluation": self.evaluation_service is not None
        }
    
    def reset_system_stats(self):
        """Resetear estadísticas del sistema"""
        
        self.system_stats = {
            "initialized_at": datetime.now().isoformat(),
            "documents_processed": 0,
            "diagnoses_processed": 0,
            "agent_queries": 0,
            "evaluations_performed": 0
        }
        
        print("✅ Estadísticas del sistema reseteadas")
    
    def shutdown(self):
        """Cerrar sistema limpiamente"""
        
        print("🛑 Cerrando sistema RAG médico mejorado...")
        
        # Cerrar componentes si es necesario
        if hasattr(self.llama_rag, 'shutdown'):
            self.llama_rag.shutdown()
        
        print("✅ Sistema cerrado correctamente") 