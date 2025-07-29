"""
Integración completa con LlamaIndex para el sistema RAG médico
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings, 
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    RecursiveCharacterTextSplitter
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import TextNode, NodeWithScore

from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.workflow import (
    Event, 
    Workflow, 
    StartEvent, 
    StopEvent, 
    step, 
    Context
)

from ..config import config
from ..models.document import DocumentMetadata, DocumentType, ProcessingStatus

logger = logging.getLogger(__name__)


class LlamaIndexMedicalRAG:
    """Sistema RAG médico optimizado con LlamaIndex"""
    
    def __init__(self, 
                 persist_dir: str = None,
                 collection_name: str = None,
                 embedding_model: str = None,
                 llm_config: Dict[str, Any] = None):
        """
        Inicializar sistema RAG con LlamaIndex
        
        Args:
            persist_dir: Directorio de persistencia
            collection_name: Nombre de la colección
            embedding_model: Modelo de embeddings
            llm_config: Configuración del LLM
        """
        
        self.persist_dir = Path(persist_dir or config.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.llm_config = llm_config or {}
        
        # Componentes LlamaIndex
        self.embedding_model_obj = None
        self.llm = None
        self.vector_store = None
        self.index = None
        self.retriever = None
        self.query_engine = None
        
        # Workflow para procesamiento médico
        self.medical_workflow = None
        
        self._initialize_components()
        self._setup_medical_workflow()
    
    def _initialize_components(self):
        """Inicializar componentes de LlamaIndex"""
        
        logger.info("🔧 Inicializando componentes LlamaIndex...")
        
        # 1. Configurar embeddings
        self.embedding_model_obj = HuggingFaceEmbedding(
            model_name=self.embedding_model,
            cache_folder=str(Path(config.EMBEDDING_CACHE_DIR))
        )
        Settings.embed_model = self.embedding_model_obj
        
        # 2. Configurar LLM
        self.llm = OpenAILike(
            model=self.llm_config.get("model", config.LLM_MODEL),
            api_base=self.llm_config.get("api_base", config.LLM_API_BASE),
            api_key="dummy",  # No se usa para modelos locales
            max_tokens=self.llm_config.get("max_tokens", config.LLM_MAX_TOKENS),
            temperature=self.llm_config.get("temperature", config.LLM_TEMPERATURE),
            timeout=self.llm_config.get("timeout", config.LLM_TIMEOUT)
        )
        Settings.llm = self.llm
        
        # 3. Configurar vector store
        self._setup_vector_store()
        
        # 4. Configurar index y retriever
        self._setup_index_and_retriever()
        
        logger.info("✅ Componentes LlamaIndex inicializados")
    
    def _setup_vector_store(self):
        """Configurar ChromaDB vector store con LlamaIndex"""
        
        try:
            # Crear directorio si no existe
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Vector store de ChromaDB
            self.vector_store = ChromaVectorStore(
                chroma_collection=self.collection_name,
                persist_dir=str(self.persist_dir)
            )
            
            logger.info(f"✅ Vector store configurado: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Error configurando vector store: {e}")
            raise
    
    def _setup_index_and_retriever(self):
        """Configurar index y retriever"""
        
        try:
            # Intentar cargar index existente
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            index_path = self.persist_dir / "index"
            if index_path.exists():
                self.index = load_index_from_storage(
                    storage_context=storage_context,
                    index_id="medical_index"
                )
                logger.info("✅ Index existente cargado")
            else:
                # Crear nuevo index
                self.index = VectorStoreIndex(
                    vector_store=self.vector_store,
                    storage_context=storage_context
                )
                # Guardar index
                self.index.storage_context.persist(persist_dir=str(index_path))
                logger.info("✅ Nuevo index creado")
            
            # Configurar retriever
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=config.SIMILARITY_TOP_K,
                similarity_cutoff=config.SIMILARITY_THRESHOLD
            )
            
            # Configurar query engine
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                structured_answer_filtering=True
            )
            
            self.query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
                response_synthesizer=response_synthesizer
            )
            
            logger.info("✅ Index y retriever configurados")
            
        except Exception as e:
            logger.error(f"❌ Error configurando index: {e}")
            raise
    
    def _setup_medical_workflow(self):
        """Configurar workflow específico para procesamiento médico"""
        
        @step()
        def preprocess_medical_text(context: Context, text: str) -> str:
            """Preprocesar texto médico"""
            # Limpiar y normalizar texto médico
            text = text.strip()
            # Expandir abreviaciones médicas
            medical_expansions = {
                " DM ": " diabetes mellitus ",
                " HTA ": " hipertensión arterial ",
                " IAM ": " infarto agudo miocardio ",
                " ACV ": " accidente cerebrovascular ",
                " EPOC ": " enfermedad pulmonar obstructiva crónica ",
                " FA ": " fibrilación auricular ",
                " IC ": " insuficiencia cardíaca ",
            }
            
            for abbrev, expansion in medical_expansions.items():
                text = text.replace(abbrev, expansion)
            
            context["preprocessed_text"] = text
            return text
        
        @step()
        def create_medical_document(context: Context, text: str, metadata: Dict[str, Any]) -> Document:
            """Crear documento médico con metadatos"""
            
            # Configurar parser específico para documentos médicos
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Crear documento con metadatos médicos
            document = Document(
                text=text,
                metadata={
                    **metadata,
                    "document_type": "medical",
                    "processed_at": datetime.now().isoformat(),
                    "chunk_size": config.CHUNK_SIZE,
                    "chunk_overlap": config.CHUNK_OVERLAP
                }
            )
            
            context["document"] = document
            return document
        
        @step()
        def retrieve_medical_context(context: Context, query: str) -> List[NodeWithScore]:
            """Recuperar contexto médico relevante"""
            
            # Usar retriever configurado
            nodes = self.retriever.retrieve(query)
            
            # Filtrar por relevancia médica
            medical_nodes = []
            for node in nodes:
                if node.score >= config.SIMILARITY_THRESHOLD:
                    medical_nodes.append(node)
            
            context["retrieved_nodes"] = medical_nodes
            return medical_nodes
        
        @step()
        def generate_medical_response(context: Context, query: str, nodes: List[NodeWithScore]) -> str:
            """Generar respuesta médica usando LLM"""
            
            # Construir contexto
            context_text = "\n\n".join([node.text for node in nodes])
            
            # Prompt médico especializado
            medical_prompt = f"""
            Eres un asistente médico especializado. Analiza la siguiente consulta y contexto médico para proporcionar una respuesta precisa y profesional.
            
            CONSULTA: {query}
            
            CONTEXTO MÉDICO:
            {context_text}
            
            INSTRUCCIONES:
            1. Proporciona una respuesta clara y estructurada
            2. Cita las fuentes relevantes del contexto
            3. Si no hay información suficiente, indícalo claramente
            4. Mantén un tono profesional médico
            
            RESPUESTA:
            """
            
            # Generar respuesta
            response = self.llm.complete(medical_prompt)
            
            context["response"] = response.text
            return response.text
        
        # Crear workflow
        self.medical_workflow = Workflow(
            steps=[
                preprocess_medical_text,
                create_medical_document,
                retrieve_medical_context,
                generate_medical_response
            ]
        )
        
        logger.info("✅ Workflow médico configurado")
    
    async def add_medical_document(self,
                                  text: str,
                                  document_type: DocumentType = DocumentType.GENERAL,
                                  title: Optional[str] = None,
                                  patient_id: Optional[str] = None,
                                  patient_age: Optional[int] = None,
                                  patient_sex: Optional[str] = None,
                                  service: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Agregar documento médico usando LlamaIndex"""
        
        try:
            # Preparar metadatos
            doc_metadata = {
                "document_type": document_type.value,
                "title": title,
                "patient_id": patient_id,
                "patient_age": patient_age,
                "patient_sex": patient_sex,
                "service": service,
                "text_length": len(text),
                "chunk_count": 0,
                "status": ProcessingStatus.PROCESSING.value,
                **(metadata or {})
            }
            
            # Crear documento
            document = Document(
                text=text,
                metadata=doc_metadata
            )
            
            # Insertar en index
            self.index.insert(document)
            
            # Actualizar metadatos
            doc_metadata["status"] = ProcessingStatus.COMPLETED.value
            doc_metadata["chunk_count"] = len(document.get_nodes())
            
            logger.info(f"✅ Documento médico agregado: {title or 'Sin título'}")
            return document.doc_id
            
        except Exception as e:
            logger.error(f"❌ Error agregando documento médico: {e}")
            raise
    
    async def query_medical_system(self,
                                  query: str,
                                  patient_context: Optional[Dict[str, Any]] = None,
                                  use_workflow: bool = True) -> Dict[str, Any]:
        """Consultar sistema médico usando LlamaIndex"""
        
        try:
            if use_workflow and self.medical_workflow:
                # Usar workflow completo
                context = Context()
                context["query"] = query
                context["patient_context"] = patient_context
                
                result = await self.medical_workflow.run(context)
                
                return {
                    "response": context.get("response", ""),
                    "retrieved_nodes": context.get("retrieved_nodes", []),
                    "preprocessed_query": context.get("preprocessed_text", query),
                    "workflow_used": True
                }
            else:
                # Usar query engine directo
                response = await self.query_engine.aquery(query)
                
                return {
                    "response": response.response,
                    "retrieved_nodes": response.source_nodes,
                    "workflow_used": False
                }
                
        except Exception as e:
            logger.error(f"❌ Error en consulta médica: {e}")
            return {
                "response": f"Error procesando consulta: {str(e)}",
                "retrieved_nodes": [],
                "workflow_used": False,
                "error": str(e)
            }
    
    def search_medical_documents(self,
                                query: str,
                                top_k: int = None,
                                document_type: Optional[DocumentType] = None,
                                service: Optional[str] = None) -> List[Dict[str, Any]]:
        """Buscar documentos médicos usando LlamaIndex"""
        
        try:
            # Configurar filtros
            filters = {}
            if document_type:
                filters["document_type"] = document_type.value
            if service:
                filters["service"] = service
            
            # Usar retriever con filtros
            nodes = self.retriever.retrieve(
                query,
                filters=filters if filters else None
            )
            
            # Formatear resultados
            results = []
            for node in nodes[:top_k or config.SIMILARITY_TOP_K]:
                results.append({
                    "text": node.text,
                    "similarity_score": node.score,
                    "metadata": node.metadata,
                    "node_id": node.node_id
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error buscando documentos: {e}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema LlamaIndex"""
        
        try:
            # Estadísticas del index
            index_stats = {
                "total_nodes": len(self.index.docstore.docs),
                "embedding_model": self.embedding_model,
                "vector_store_type": "ChromaDB",
                "collection_name": self.collection_name
            }
            
            # Estadísticas del retriever
            retriever_stats = {
                "similarity_top_k": config.SIMILARITY_TOP_K,
                "similarity_threshold": config.SIMILARITY_THRESHOLD
            }
            
            return {
                "llama_index_version": "latest",
                "index": index_stats,
                "retriever": retriever_stats,
                "workflow_available": self.medical_workflow is not None
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def clear_index(self) -> bool:
        """Limpiar todo el index"""
        
        try:
            # Eliminar archivos de persistencia
            import shutil
            if self.persist_dir.exists():
                shutil.rmtree(self.persist_dir)
            
            # Recrear componentes
            self._setup_vector_store()
            self._setup_index_and_retriever()
            
            logger.info("✅ Index limpiado completamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error limpiando index: {e}")
            return False 