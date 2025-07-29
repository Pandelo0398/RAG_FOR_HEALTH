"""
Sistema de Agentes Médicos Inteligentes usando LlamaIndex
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.workflow import Workflow, step, Context
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer

from llama_index.agent.openai import OpenAIAgent
from llama_index.tools import BaseTool

from ..infrastructure.llama_index_integration import LlamaIndexMedicalRAG
from ..models.medical import ProcessingRequest, MedicalProcessingResult
from ..models.document import DocumentType, ProcessingStatus

logger = logging.getLogger(__name__)


class MedicalAgentSystem:
    """Sistema de agentes médicos inteligentes"""
    
    def __init__(self, llama_rag: LlamaIndexMedicalRAG):
        """
        Inicializar sistema de agentes médicos
        
        Args:
            llama_rag: Instancia del sistema RAG con LlamaIndex
        """
        
        self.llama_rag = llama_rag
        self.agents = {}
        self.tools = {}
        
        self._setup_medical_tools()
        self._setup_specialized_agents()
    
    def _setup_medical_tools(self):
        """Configurar herramientas médicas especializadas"""
        
        # Herramienta para búsqueda de documentos médicos
        def search_medical_docs(query: str, document_type: str = None, service: str = None) -> str:
            """Buscar documentos médicos relevantes"""
            try:
                doc_type = DocumentType(document_type) if document_type else None
                results = self.llama_rag.search_medical_documents(
                    query=query,
                    document_type=doc_type,
                    service=service
                )
                
                if not results:
                    return "No se encontraron documentos médicos relevantes."
                
                response = f"Encontrados {len(results)} documentos relevantes:\n\n"
                for i, result in enumerate(results[:3], 1):
                    response += f"{i}. {result['text'][:200]}...\n"
                    response += f"   Similitud: {result['similarity_score']:.3f}\n"
                    response += f"   Tipo: {result['metadata'].get('document_type', 'N/A')}\n\n"
                
                return response
            except Exception as e:
                return f"Error en búsqueda: {str(e)}"
        
        # Herramienta para análisis de diagnósticos
        def analyze_diagnosis(diagnosis_text: str, patient_age: int = None, patient_sex: str = None) -> str:
            """Analizar diagnóstico médico y proporcionar insights"""
            try:
                # Construir contexto del paciente
                patient_context = ""
                if patient_age:
                    patient_context += f"Paciente de {patient_age} años"
                if patient_sex:
                    patient_context += f", sexo {patient_sex}"
                
                # Buscar diagnósticos similares
                similar_diagnoses = self.llama_rag.search_medical_documents(
                    query=diagnosis_text,
                    document_type=DocumentType.DIAGNOSIS
                )
                
                # Generar análisis
                analysis_prompt = f"""
                Analiza el siguiente diagnóstico médico:
                
                DIAGNÓSTICO: {diagnosis_text}
                CONTEXTO PACIENTE: {patient_context or 'No especificado'}
                
                DIAGNÓSTICOS SIMILARES ENCONTRADOS: {len(similar_diagnoses)}
                
                Proporciona:
                1. Análisis de los síntomas principales
                2. Posibles diagnósticos diferenciales
                3. Recomendaciones de seguimiento
                4. Urgencia del caso
                """
                
                # Usar el LLM para análisis
                response = self.llama_rag.llm.complete(analysis_prompt)
                return response.text
                
            except Exception as e:
                return f"Error analizando diagnóstico: {str(e)}"
        
        # Herramienta para recomendaciones de tratamiento
        def suggest_treatment(diagnosis: str, patient_context: str = "") -> str:
            """Sugerir tratamientos basados en diagnóstico"""
            try:
                # Buscar tratamientos similares
                treatment_docs = self.llama_rag.search_medical_documents(
                    query=f"tratamiento {diagnosis}",
                    document_type=DocumentType.TREATMENT
                )
                
                if not treatment_docs:
                    return "No se encontraron protocolos de tratamiento específicos para este diagnóstico."
                
                # Generar recomendaciones
                treatment_prompt = f"""
                Basándote en el diagnóstico y los protocolos encontrados, sugiere un plan de tratamiento:
                
                DIAGNÓSTICO: {diagnosis}
                CONTEXTO PACIENTE: {patient_context}
                
                PROTOCOLOS ENCONTRADOS: {len(treatment_docs)}
                
                Proporciona:
                1. Tratamiento farmacológico recomendado
                2. Tratamiento no farmacológico
                3. Seguimiento recomendado
                4. Precauciones y contraindicaciones
                """
                
                response = self.llama_rag.llm.complete(treatment_prompt)
                return response.text
                
            except Exception as e:
                return f"Error sugiriendo tratamiento: {str(e)}"
        
        # Herramienta para validación de medicamentos
        def validate_medication(medication: str, patient_age: int = None, patient_sex: str = None) -> str:
            """Validar medicamento y posibles interacciones"""
            try:
                # Buscar información del medicamento
                med_info = self.llama_rag.search_medical_documents(
                    query=f"medicamento {medication} dosis interacciones",
                    document_type=DocumentType.MEDICATION
                )
                
                validation_prompt = f"""
                Valida el siguiente medicamento para el paciente:
                
                MEDICAMENTO: {medication}
                EDAD: {patient_age or 'No especificada'}
                SEXO: {patient_sex or 'No especificado'}
                
                INFORMACIÓN ENCONTRADA: {len(med_info)} documentos
                
                Proporciona:
                1. Dosis recomendada según edad/sexo
                2. Posibles efectos secundarios
                3. Interacciones medicamentosas
                4. Contraindicaciones
                5. Recomendaciones de administración
                """
                
                response = self.llama_rag.llm.complete(validation_prompt)
                return response.text
                
            except Exception as e:
                return f"Error validando medicamento: {str(e)}"
        
        # Registrar herramientas
        self.tools = {
            "search_medical_docs": FunctionTool.from_defaults(
                fn=search_medical_docs,
                name="search_medical_docs",
                description="Buscar documentos médicos relevantes en la base de datos"
            ),
            "analyze_diagnosis": FunctionTool.from_defaults(
                fn=analyze_diagnosis,
                name="analyze_diagnosis", 
                description="Analizar diagnóstico médico y proporcionar insights clínicos"
            ),
            "suggest_treatment": FunctionTool.from_defaults(
                fn=suggest_treatment,
                name="suggest_treatment",
                description="Sugerir tratamientos basados en diagnóstico y protocolos médicos"
            ),
            "validate_medication": FunctionTool.from_defaults(
                fn=validate_medication,
                name="validate_medication",
                description="Validar medicamento, dosis e interacciones para el paciente"
            )
        }
        
        logger.info(f"✅ Configuradas {len(self.tools)} herramientas médicas")
    
    def _setup_specialized_agents(self):
        """Configurar agentes especializados"""
        
        # Agente de diagnóstico
        diagnosis_tools = [
            self.tools["search_medical_docs"],
            self.tools["analyze_diagnosis"]
        ]
        
        self.agents["diagnosis"] = ReActAgent.from_tools(
            tools=diagnosis_tools,
            llm=self.llama_rag.llm,
            verbose=True,
            system_prompt="""
            Eres un agente médico especializado en diagnóstico. Tu objetivo es:
            1. Analizar síntomas y signos clínicos
            2. Buscar diagnósticos similares en la base de datos
            3. Proporcionar análisis clínico detallado
            4. Sugerir diagnósticos diferenciales
            5. Indicar la urgencia del caso
            
            Siempre mantén un tono profesional y médico.
            """
        )
        
        # Agente de tratamiento
        treatment_tools = [
            self.tools["search_medical_docs"],
            self.tools["suggest_treatment"],
            self.tools["validate_medication"]
        ]
        
        self.agents["treatment"] = ReActAgent.from_tools(
            tools=treatment_tools,
            llm=self.llama_rag.llm,
            verbose=True,
            system_prompt="""
            Eres un agente médico especializado en tratamientos. Tu objetivo es:
            1. Analizar diagnósticos para sugerir tratamientos
            2. Buscar protocolos de tratamiento en la base de datos
            3. Validar medicamentos y dosis
            4. Considerar interacciones medicamentosas
            5. Proporcionar planes de seguimiento
            
            Siempre prioriza la seguridad del paciente.
            """
        )
        
        # Agente de investigación médica
        research_tools = [
            self.tools["search_medical_docs"]
        ]
        
        self.agents["research"] = ReActAgent.from_tools(
            tools=research_tools,
            llm=self.llama_rag.llm,
            verbose=True,
            system_prompt="""
            Eres un agente de investigación médica. Tu objetivo es:
            1. Buscar información médica relevante
            2. Analizar casos similares
            3. Proporcionar evidencia científica
            4. Identificar tendencias en diagnósticos
            5. Sugerir áreas de investigación
            
            Siempre cita las fuentes y mantén rigor científico.
            """
        )
        
        logger.info(f"✅ Configurados {len(self.agents)} agentes especializados")
    
    async def process_medical_query(self,
                                   query: str,
                                   agent_type: str = "diagnosis",
                                   patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Procesar consulta médica usando agentes especializados
        
        Args:
            query: Consulta médica
            agent_type: Tipo de agente a usar (diagnosis, treatment, research)
            patient_context: Contexto del paciente
        """
        
        try:
            if agent_type not in self.agents:
                return {
                    "error": f"Agente no disponible: {agent_type}",
                    "available_agents": list(self.agents.keys())
                }
            
            agent = self.agents[agent_type]
            
            # Construir consulta con contexto
            if patient_context:
                context_str = f"Contexto del paciente: {patient_context}\n\n"
                full_query = context_str + query
            else:
                full_query = query
            
            # Ejecutar agente
            response = await agent.achat(full_query)
            
            return {
                "response": response.response,
                "agent_used": agent_type,
                "tools_used": response.sources,
                "patient_context": patient_context,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error procesando consulta con agente: {e}")
            return {
                "error": str(e),
                "agent_used": agent_type,
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_medical_workflow(self,
                                  request: ProcessingRequest) -> MedicalProcessingResult:
        """
        Ejecutar workflow médico completo usando agentes
        
        Args:
            request: Solicitud de procesamiento médico
        """
        
        try:
            # Paso 1: Análisis de diagnóstico
            diagnosis_result = await self.process_medical_query(
                query=request.text,
                agent_type="diagnosis",
                patient_context={
                    "age": request.patient_age,
                    "sex": request.patient_sex,
                    "service": request.service
                }
            )
            
            # Paso 2: Sugerir tratamiento
            treatment_result = await self.process_medical_query(
                query=f"Basándome en el diagnóstico: {request.text}",
                agent_type="treatment",
                patient_context={
                    "age": request.patient_age,
                    "sex": request.patient_sex,
                    "service": request.service
                }
            )
            
            # Paso 3: Investigación adicional
            research_result = await self.process_medical_query(
                query=f"Buscar información adicional sobre: {request.text}",
                agent_type="research"
            )
            
            # Construir resultado completo
            result = MedicalProcessingResult(
                original_text=request.text,
                processed_text=request.text,
                diagnosis_analysis=diagnosis_result.get("response", ""),
                treatment_suggestions=treatment_result.get("response", ""),
                research_findings=research_result.get("response", ""),
                confidence_score=0.85,  # Calcular basado en resultados
                processing_time=0.0,  # Calcular tiempo real
                metadata={
                    "agent_workflow": True,
                    "diagnosis_agent": diagnosis_result.get("tools_used", []),
                    "treatment_agent": treatment_result.get("tools_used", []),
                    "research_agent": research_result.get("tools_used", [])
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en workflow médico: {e}")
            return MedicalProcessingResult(
                original_text=request.text,
                processed_text=request.text,
                diagnosis_analysis=f"Error en procesamiento: {str(e)}",
                treatment_suggestions="",
                research_findings="",
                confidence_score=0.0,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de los agentes"""
        
        return {
            "total_agents": len(self.agents),
            "available_agents": list(self.agents.keys()),
            "total_tools": len(self.tools),
            "available_tools": list(self.tools.keys()),
            "system_status": "active"
        } 