"""
Gestor de documentos médicos
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..models.document import DocumentMetadata, DocumentType, ProcessingStatus
from ..infrastructure.vector_store import ChromaVectorStore


class DocumentManager:
    """Gestor de documentos médicos con almacenamiento vectorial"""
    
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store
        self.document_registry: Dict[str, DocumentMetadata] = {}
    
    def add_document(self,
                    text: str,
                    document_type: DocumentType = DocumentType.GENERAL,
                    title: Optional[str] = None,
                    patient_id: Optional[str] = None,
                    patient_age: Optional[int] = None,
                    patient_sex: Optional[str] = None,
                    service: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Agregar documento médico al sistema"""
        
        # Generar ID único
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        
        # Crear metadatos
        doc_metadata = DocumentMetadata(
            document_id=document_id,
            document_type=document_type,
            title=title or f"Documento {document_type.value}",
            patient_id=patient_id,
            patient_age=patient_age,
            patient_sex=patient_sex,
            service=service,
            text_length=len(text),
            custom_fields=metadata or {}
        )
        
        print(f"📄 Agregando documento: {document_id}")
        print(f"   Tipo: {document_type.value}")
        print(f"   Longitud: {len(text)} caracteres")
        
        try:
            # Almacenar en vector store
            chunk_ids = self.vector_store.add_document(
                text=text,
                document_id=document_id,
                metadata=doc_metadata
            )
            
            # Registrar documento
            self.document_registry[document_id] = doc_metadata
            
            print(f"✅ Documento agregado exitosamente ({len(chunk_ids)} chunks)")
            return document_id
            
        except Exception as e:
            doc_metadata.update_status(ProcessingStatus.FAILED, f"Error: {str(e)}")
            print(f"❌ Error agregando documento: {e}")
            raise
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Obtener metadatos de documento"""
        return self.document_registry.get(document_id)
    
    def search_documents(self,
                        query: str,
                        top_k: int = 5,
                        document_type: Optional[DocumentType] = None,
                        service: Optional[str] = None,
                        patient_age_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Buscar documentos similares"""
        
        print(f"🔍 Buscando documentos: {query[:50]}...")
        
        # Crear filtros de metadatos
        filters = {}
        
        if document_type:
            filters["document_type"] = document_type.value
        
        if service:
            filters["service"] = service
        
        if patient_age_range:
            min_age, max_age = patient_age_range
            # ChromaDB no soporta rangos directamente, así que filtraremos después
        
        try:
            # Buscar en vector store
            results = self.vector_store.search(
                query=query,
                top_k=top_k * 2,  # Obtener más para filtrar
                filter_metadata=filters if filters else None
            )
            
            # Filtrar por edad si se especifica
            if patient_age_range:
                min_age, max_age = patient_age_range
                filtered_results = []
                
                for result in results:
                    age = result["metadata"].get("patient_age")
                    if age is None or (min_age <= age <= max_age):
                        filtered_results.append(result)
                
                results = filtered_results[:top_k]
            
            print(f"✅ Encontrados {len(results)} documentos relevantes")
            return results
            
        except Exception as e:
            print(f"❌ Error buscando documentos: {e}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Obtener todos los chunks de un documento"""
        try:
            return self.vector_store.get_document_chunks(document_id)
        except Exception as e:
            print(f"❌ Error obteniendo chunks: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Eliminar documento del sistema"""
        
        try:
            # Eliminar del vector store
            success = self.vector_store.delete_document(document_id)
            
            # Eliminar del registro
            if document_id in self.document_registry:
                del self.document_registry[document_id]
            
            if success:
                print(f"✅ Documento {document_id} eliminado exitosamente")
            else:
                print(f"⚠️ No se pudo eliminar documento {document_id}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error eliminando documento: {e}")
            return False
    
    def list_documents(self,
                      document_type: Optional[DocumentType] = None,
                      service: Optional[str] = None,
                      status: Optional[ProcessingStatus] = None) -> List[DocumentMetadata]:
        """Listar documentos con filtros opcionales"""
        
        documents = list(self.document_registry.values())
        
        # Aplicar filtros
        if document_type:
            documents = [d for d in documents if d.document_type == document_type]
        
        if service:
            documents = [d for d in documents if d.service == service]
        
        if status:
            documents = [d for d in documents if d.status == status]
        
        # Ordenar por fecha de creación (más recientes primero)
        documents.sort(key=lambda d: d.created_at, reverse=True)
        
        return documents
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del gestor de documentos"""
        
        # Estadísticas del vector store
        vector_stats = self.vector_store.get_stats()
        
        # Estadísticas locales
        docs_by_type = {}
        docs_by_service = {}
        docs_by_status = {}
        
        for doc in self.document_registry.values():
            # Por tipo
            doc_type = doc.document_type.value
            docs_by_type[doc_type] = docs_by_type.get(doc_type, 0) + 1
            
            # Por servicio
            if doc.service:
                docs_by_service[doc.service] = docs_by_service.get(doc.service, 0) + 1
            
            # Por estado
            status = doc.status.value
            docs_by_status[status] = docs_by_status.get(status, 0) + 1
        
        return {
            "total_documents": len(self.document_registry),
            "documents_by_type": docs_by_type,
            "documents_by_service": docs_by_service,
            "documents_by_status": docs_by_status,
            "vector_store_stats": vector_stats
        }
    
    def import_from_directory(self,
                             directory_path: str,
                             document_type: DocumentType = DocumentType.GENERAL,
                             file_patterns: List[str] = None) -> List[str]:
        """Importar documentos desde un directorio"""
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directorio no existe: {directory_path}")
        
        file_patterns = file_patterns or ["*.txt", "*.md"]
        imported_docs = []
        
        print(f"📂 Importando documentos desde: {directory_path}")
        
        for pattern in file_patterns:
            for file_path in directory.glob(pattern):
                try:
                    print(f"📄 Procesando: {file_path.name}")
                    
                    # Leer archivo
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if len(content.strip()) < 50:
                        print(f"⚠️ Archivo muy corto, saltando: {file_path.name}")
                        continue
                    
                    # Agregar documento
                    doc_id = self.add_document(
                        text=content,
                        document_type=document_type,
                        title=file_path.stem,
                        metadata={"source_file": str(file_path)}
                    )
                    
                    imported_docs.append(doc_id)
                    
                except Exception as e:
                    print(f"❌ Error procesando {file_path.name}: {e}")
        
        print(f"✅ Importados {len(imported_docs)} documentos")
        return imported_docs
    
    def export_document_list(self, output_path: str) -> bool:
        """Exportar lista de documentos a archivo"""
        
        try:
            output = Path(output_path)
            
            with open(output, 'w', encoding='utf-8') as f:
                f.write("# Lista de Documentos Médicos\n\n")
                
                for doc in self.list_documents():
                    f.write(f"## {doc.title}\n")
                    f.write(f"- **ID**: {doc.document_id}\n")
                    f.write(f"- **Tipo**: {doc.document_type.value}\n")
                    f.write(f"- **Estado**: {doc.status.value}\n")
                    f.write(f"- **Creado**: {doc.created_at}\n")
                    if doc.service:
                        f.write(f"- **Servicio**: {doc.service}\n")
                    if doc.patient_age:
                        f.write(f"- **Edad paciente**: {doc.patient_age}\n")
                    f.write(f"- **Longitud**: {doc.text_length} caracteres\n")
                    f.write(f"- **Chunks**: {doc.chunk_count}\n\n")
            
            print(f"✅ Lista exportada a: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error exportando lista: {e}")
            return False
