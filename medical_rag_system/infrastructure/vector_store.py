"""
Vector Store basado en ChromaDB para documentos médicos
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB no disponible. Instalar con: pip install chromadb")

from ..config import config
from ..models.document import DocumentMetadata, DocumentChunk, ProcessingStatus
from .embedding_service import EmbeddingService


class ChromaVectorStore:
    """Gestor de base vectorial usando ChromaDB para documentos médicos"""
    
    def __init__(self,
                 persist_dir: str = None,
                 collection_name: str = None,
                 embedding_service: EmbeddingService = None):
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB no está disponible. Instalar con: pip install chromadb")
        
        self.persist_dir = Path(persist_dir or config.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.embedding_service = embedding_service or EmbeddingService()
        
        self.client = None
        self.collection = None
        
        self._initialize()
    
    def _initialize(self):
        """Inicializar ChromaDB y crear colección"""
        
        print(f"🗄️ Inicializando ChromaDB en: {self.persist_dir}")
        
        # Crear directorio si no existe
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Cliente ChromaDB persistente
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # Crear o obtener colección
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"✅ Colección existente cargada: {self.collection.count()} documentos")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Base vectorial de documentos médicos",
                    "embedding_model": config.EMBEDDING_MODEL,
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            )
            print(f"✅ Nueva colección creada: {self.collection_name}")
    
    def add_document(self,
                    text: str,
                    document_id: str,
                    metadata: DocumentMetadata,
                    chunk_size: int = None) -> List[str]:
        """Agregar documento con chunking automático"""
        
        chunk_size = chunk_size or config.CHUNK_SIZE
        
        print(f"📄 Procesando documento: {document_id}")
        
        # Actualizar metadatos del documento
        metadata.text_length = len(text)
        metadata.update_status(ProcessingStatus.PROCESSING, "Iniciando chunking")
        
        # Crear chunks
        chunks = self._create_chunks(text, document_id, chunk_size)
        metadata.chunk_count = len(chunks)
        
        print(f"🔄 Creados {len(chunks)} chunks")
        
        # Generar embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        
        try:
            embeddings = self.embedding_service.encode_medical_text(
                chunk_texts,
                preprocess=True,
                show_progress=True
            )
            
            # Preparar datos para ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                chunk_metadata = metadata.to_chroma_metadata()
                chunk_metadata.update({
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "tokens_count": chunk.tokens_count or len(chunk.text.split()),
                    "processed_at": datetime.now().isoformat()
                })
                metadatas.append(chunk_metadata)
            
            # Insertar en ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=chunk_texts,
                metadatas=metadatas,
                ids=ids
            )
            
            metadata.update_status(ProcessingStatus.COMPLETED, f"Agregados {len(chunks)} chunks")
            print(f"✅ Documento agregado exitosamente")
            
            return ids
            
        except Exception as e:
            metadata.update_status(ProcessingStatus.FAILED, f"Error: {str(e)}")
            print(f"❌ Error agregando documento: {e}")
            raise
    
    def _create_chunks(self, 
                      text: str, 
                      document_id: str, 
                      chunk_size: int) -> List[DocumentChunk]:
        """Crear chunks del documento"""
        
        chunks = []
        overlap = config.CHUNK_OVERLAP
        
        # Split por párrafos primero
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Si el párrafo actual cabe en el chunk
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # Guardar chunk actual si no está vacío
                if current_chunk.strip():
                    chunk = DocumentChunk(
                        chunk_id=f"{document_id}_chunk_{chunk_index}",
                        document_id=document_id,
                        chunk_index=chunk_index,
                        text=current_chunk.strip(),
                        chunk_type="content"
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Iniciar nuevo chunk
                current_chunk = paragraph + "\n\n"
                
                # Si el párrafo es muy largo, dividirlo
                if len(current_chunk) > chunk_size:
                    sub_chunks = self._split_long_text(current_chunk, chunk_size, overlap)
                    for sub_text in sub_chunks:
                        chunk = DocumentChunk(
                            chunk_id=f"{document_id}_chunk_{chunk_index}",
                            document_id=document_id,
                            chunk_index=chunk_index,
                            text=sub_text.strip(),
                            chunk_type="content"
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    current_chunk = ""
        
        # Agregar último chunk si queda texto
        if current_chunk.strip():
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                chunk_index=chunk_index,
                text=current_chunk.strip(),
                chunk_type="content"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_long_text(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Dividir texto largo en chunks con overlap"""
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_size
            
            # Buscar punto de corte natural (espacio, punto, etc.)
            if end < len(text):
                for sep in ['. ', '.\n', ' ', '\n']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break
            
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(chunk_text)
            
            start = max(start + 1, end - overlap)
        
        return chunks
    
    def search(self,
              query: str,
              top_k: int = None,
              similarity_threshold: float = None,
              filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Buscar documentos similares"""
        
        top_k = top_k or config.SIMILARITY_TOP_K
        similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        
        print(f"🔍 Buscando: {query[:100]}...")
        
        try:
            # Generar embedding de la query
            query_embedding = self.embedding_service.encode_medical_text([query], preprocess=True)
            
            if len(query_embedding) == 0:
                return []
            
            # Buscar en ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k * 2,  # Obtener más para filtrar después
                where=filter_metadata,
                include=["metadatas", "documents", "distances"]
            )
            
            # Procesar resultados
            relevant_results = []
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )):
                # Convertir distancia a similitud
                similarity = 1 - distance
                
                if similarity >= similarity_threshold:
                    result = {
                        "text": doc,
                        "similarity_score": similarity,
                        "metadata": metadata,
                        "rank": i + 1
                    }
                    relevant_results.append(result)
            
            # Limitar a top_k
            relevant_results = relevant_results[:top_k]
            
            print(f"✅ Encontrados {len(relevant_results)} resultados relevantes")
            return relevant_results
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Obtener todos los chunks de un documento"""
        
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas", "documents"]
            )
            
            chunks = []
            for doc, metadata in zip(results["documents"], results["metadatas"]):
                chunks.append({
                    "text": doc,
                    "metadata": metadata
                })
            
            # Ordenar por chunk_index
            chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
            return chunks
            
        except Exception as e:
            print(f"❌ Error obteniendo chunks: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Eliminar documento y todos sus chunks"""
        
        try:
            # Obtener IDs de chunks del documento
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if not results["ids"]:
                print(f"⚠️ No se encontraron chunks para documento: {document_id}")
                return False
            
            # Eliminar chunks
            self.collection.delete(ids=results["ids"])
            
            print(f"✅ Eliminado documento {document_id} ({len(results['ids'])} chunks)")
            return True
            
        except Exception as e:
            print(f"❌ Error eliminando documento: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la base vectorial"""
        
        try:
            count = self.collection.count()
            
            # Obtener algunos metadatos para estadísticas
            sample = self.collection.get(limit=min(100, count), include=["metadatas"])
            
            document_ids = set()
            services = set()
            document_types = set()
            
            for metadata in sample["metadatas"]:
                if "document_id" in metadata:
                    document_ids.add(metadata["document_id"])
                if "service" in metadata and metadata["service"]:
                    services.add(metadata["service"])
                if "document_type" in metadata:
                    document_types.add(metadata["document_type"])
            
            return {
                "total_chunks": count,
                "unique_documents": len(document_ids),
                "services": list(services),
                "document_types": list(document_types),
                "collection_name": self.collection_name,
                "persist_dir": str(self.persist_dir),
                "embedding_model": config.EMBEDDING_MODEL
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def clear_all(self) -> bool:
        """Limpiar toda la colección"""
        
        try:
            self.client.delete_collection(self.collection_name)
            self._initialize()
            print("✅ Colección limpiada completamente")
            return True
        except Exception as e:
            print(f"❌ Error limpiando colección: {e}")
            return False
