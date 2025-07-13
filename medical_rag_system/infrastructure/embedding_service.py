"""
Servicio de embeddings para el sistema médico
"""

import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer

from ..config import config


class EmbeddingService:
    """Servicio para generar embeddings de texto médico"""
    
    def __init__(self, 
                 model_name: str = None,
                 cache_dir: str = None,
                 device: str = None):
        
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.cache_dir = Path(cache_dir or config.EMBEDDING_CACHE_DIR)
        self.device = device or "cpu"
        
        self.model: Optional[SentenceTransformer] = None
        self._model_loaded = False
        
        # Crear directorio de cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self):
        """Cargar modelo de embeddings de forma lazy"""
        if self._model_loaded:
            return
        
        try:
            print(f"📊 Cargando modelo de embeddings: {self.model_name}")
            
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir),
                device=self.device
            )
            
            self._model_loaded = True
            print(f"✅ Modelo de embeddings cargado exitosamente")
            
        except Exception as e:
            print(f"❌ Error cargando modelo de embeddings: {e}")
            raise
    
    def encode(self, 
               texts: List[str], 
               batch_size: int = 32,
               show_progress: bool = False,
               normalize: bool = True) -> np.ndarray:
        """Generar embeddings para lista de textos"""
        
        if not texts:
            return np.array([])
        
        self._load_model()
        
        try:
            # Filtrar textos vacíos
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            
            if not valid_texts:
                return np.array([])
            
            print(f"🔄 Generando embeddings para {len(valid_texts)} textos...")
            
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            
            print(f"✅ Embeddings generados: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generando embeddings: {e}")
            raise
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generar embedding para un solo texto"""
        if not text or not text.strip():
            return np.array([])
        
        embeddings = self.encode([text.strip()], normalize=normalize)
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def compute_similarity(self, 
                          embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """Calcular similitud coseno entre dos embeddings"""
        try:
            # Normalizar si es necesario
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception:
            return 0.0
    
    def find_most_similar(self, 
                         query_embedding: np.ndarray,
                         candidate_embeddings: List[np.ndarray],
                         top_k: int = 5) -> List[tuple]:
        """Encontrar embeddings más similares al query"""
        
        if not candidate_embeddings:
            return []
        
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        info = {
            "model_name": self.model_name,
            "cache_dir": str(self.cache_dir),
            "device": self.device,
            "model_loaded": self._model_loaded
        }
        
        if self._model_loaded and self.model:
            try:
                info.update({
                    "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                    "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown')
                })
            except:
                pass
        
        return info
    
    def preprocess_medical_text(self, text: str) -> str:
        """Preprocesar texto médico para mejores embeddings"""
        
        # Limpiar texto básico
        text = text.strip()
        
        # Expandir algunas abreviaciones comunes
        medical_expansions = {
            " DM ": " diabetes mellitus ",
            " HTA ": " hipertensión arterial ",
            " IAM ": " infarto agudo miocardio ",
            " ACV ": " accidente cerebrovascular ",
            " EPOC ": " enfermedad pulmonar obstructiva crónica ",
            " FA ": " fibrilación auricular ",
            " IC ": " insuficiencia cardíaca ",
        }
        
        text_upper = text.upper()
        for abbrev, expansion in medical_expansions.items():
            if abbrev in text_upper:
                # Reemplazar manteniendo el caso original aproximado
                text = text.replace(abbrev.strip(), expansion.strip())
                text = text.replace(abbrev.strip().lower(), expansion.strip())
                text = text.replace(abbrev.strip().upper(), expansion.strip())
        
        return text
    
    def encode_medical_text(self, 
                           texts: List[str],
                           preprocess: bool = True,
                           **kwargs) -> np.ndarray:
        """Generar embeddings específicamente para texto médico"""
        
        if preprocess:
            processed_texts = [self.preprocess_medical_text(text) for text in texts]
        else:
            processed_texts = texts
        
        return self.encode(processed_texts, **kwargs)
