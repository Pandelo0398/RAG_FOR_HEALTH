"""
Procesador médico principal con integración LLM y fallback
"""

import json
import re
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..models.patient import PatientData
from ..models.medical import MedicalProcessingResult, ProcessingRequest, ClinicalEntity
from ..infrastructure.llm_client import LLMClientInterface
from ..infrastructure.vector_store import ChromaVectorStore
from .fallback_processor import FallbackProcessor


class MedicalProcessor:
    """Procesador médico principal con capacidades de LLM y fallback"""
    
    def __init__(self,
                 llm_client: LLMClientInterface,
                 vector_store: Optional[ChromaVectorStore] = None,
                 fallback_processor: Optional[FallbackProcessor] = None):
        
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.fallback_processor = fallback_processor or FallbackProcessor()
        
        self.processing_stats = {
            "total_processed": 0,
            "llm_successful": 0,
            "fallback_used": 0,
            "errors": 0
        }
    
    async def process_diagnosis(self, request: ProcessingRequest) -> MedicalProcessingResult:
        """Procesar diagnóstico médico de forma resiliente"""
        
        print(f"🏥 PROCESAMIENTO MÉDICO RESILIENTE")
        print(f"📝 Texto: {request.text[:100]}...")
        print("-" * 60)
        
        start_time = datetime.now()
        
        # Crear datos del paciente
        patient_data = PatientData(
            age=request.patient_age or 50,
            sex=request.patient_sex or "M",
            service=request.service or "general"
        )
        
        # Intentar procesamiento con LLM
        llm_result = await self._try_llm_processing(request, patient_data)
        
        if llm_result:
            self.processing_stats["llm_successful"] += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            
            llm_result.processing_time = datetime.now().isoformat()
            llm_result.add_processing_note(f"Procesado con LLM en {processing_time:.2f}s")
            
            self.processing_stats["total_processed"] += 1
            return llm_result
        
        # Fallback a procesamiento basado en reglas
        print("🔄 Usando procesamiento de fallback...")
        
        try:
            fallback_result = self.fallback_processor.process_diagnosis(
                request.text, patient_data
            )
            
            self.processing_stats["fallback_used"] += 1
            self.processing_stats["total_processed"] += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            fallback_result.add_processing_note(f"Fallback completado en {processing_time:.2f}s")
            
            return fallback_result
            
        except Exception as e:
            self.processing_stats["errors"] += 1
            print(f"❌ Error en fallback: {e}")
            
            # Resultado de error mínimo
            return self._create_error_result(request.text, str(e))
    
    async def _try_llm_processing(self, 
                                 request: ProcessingRequest,
                                 patient_data: PatientData) -> Optional[MedicalProcessingResult]:
        """Intentar procesamiento con LLM"""
        
        if not self.llm_client.is_available():
            print("❌ LLM no disponible")
            return None
        
        # Obtener contexto relevante del vector store
        relevant_context = ""
        if self.vector_store and request.use_vector_store:
            try:
                query = f"diagnóstico médico {patient_data.service} {request.text[:200]}"
                contexts = self.vector_store.search(query, top_k=3)
                if contexts:
                    relevant_context = "\n".join([ctx['text'] for ctx in contexts[:2]])[:800]
                    print(f"📚 Contexto relevante obtenido: {len(relevant_context)} caracteres")
            except Exception as e:
                print(f"⚠️ Error obteniendo contexto: {e}")
        
        # Crear prompt optimizado
        prompt = self._create_medical_prompt(request.text, patient_data, relevant_context)
        
        # Ejecutar LLM con reintentos
        response = await self._execute_llm_with_retries(prompt, request.max_retries)
        
        if not response:
            return None
        
        # Procesar respuesta JSON
        try:
            return self._parse_llm_response(response, request.text)
        except Exception as e:
            print(f"❌ Error parseando respuesta LLM: {e}")
            return None
    
    def _create_medical_prompt(self, 
                              diagnosis_text: str,
                              patient_data: PatientData,
                              context: str = "") -> str:
        """Crear prompt médico optimizado"""
        
        prompt = f"""
Analiza este caso médico y devuelve JSON válido:

PACIENTE: {patient_data.age} años, {patient_data.sex}, {patient_data.service}
DIAGNÓSTICO: {diagnosis_text}

{f"CONTEXTO MÉDICO RELEVANTE: {context}" if context else ""}

Analiza el caso y devuelve SOLO este JSON (sin markdown ni explicaciones):
{{
  "main_diagnosis": "diagnóstico principal extraído",
  "secondary_diagnoses": ["diagnóstico secundario si existe"],
  "cie10_codes": ["código CIE-10 relevante"],
  "expanded_acronyms": {{"DM": "Diabetes Mellitus", "HTA": "Hipertensión Arterial"}},
  "clinical_entities": [
    {{"entity": "entidad médica", "type": "diagnosis", "severity": "leve/moderada/severa"}}
  ],
  "symptoms": ["síntoma1", "síntoma2"],
  "risk_factors": ["factor de riesgo"],
  "anatomical_location": "ubicación anatómica si aplica",
  "confidence": 0.9
}}

Instrucciones:
- Usa códigos CIE-10 válidos cuando sea posible
- Expande acrónimos médicos comunes (DM, HTA, IAM, etc.)
- Identifica síntomas y factores de riesgo
- Clasifica entidades como: diagnosis, symptom, medication, procedure
- Asigna confianza entre 0.7-0.95
"""
        
        return prompt.strip()
    
    async def _execute_llm_with_retries(self, 
                                       prompt: str, 
                                       max_retries: int) -> Optional[str]:
        """Ejecutar LLM con reintentos automáticos"""
        
        for attempt in range(max_retries):
            try:
                print(f"🧠 Intento LLM {attempt + 1}/{max_retries}...")
                
                response = await self.llm_client.complete(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=1500
                )
                
                if response and len(response.strip()) > 20:
                    print("✅ LLM respondió exitosamente")
                    return response
                else:
                    print("⚠️ Respuesta LLM vacía o muy corta")
                
            except Exception as e:
                print(f"❌ Error LLM intento {attempt + 1}: {str(e)[:100]}...")
            
            # Esperar antes del siguiente intento
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
        
        return None
    
    def _parse_llm_response(self, 
                           response: str, 
                           original_text: str) -> MedicalProcessingResult:
        """Parsear respuesta JSON del LLM"""
        
        # Limpiar respuesta
        json_text = self._clean_json_response(response)
        
        try:
            data = json.loads(json_text)
            
            # Procesar entidades clínicas
            clinical_entities = []
            if "clinical_entities" in data:
                for entity_data in data["clinical_entities"]:
                    if isinstance(entity_data, dict):
                        entity = ClinicalEntity(
                            entity=entity_data.get("entity", ""),
                            type=entity_data.get("type", "general"),
                            severity=entity_data.get("severity"),
                            location=entity_data.get("location"),
                            confidence=entity_data.get("confidence", 0.8)
                        )
                        clinical_entities.append(entity)
            
            # Crear resultado
            result = MedicalProcessingResult(
                cleaned_text=original_text.strip(),
                original_text=original_text,
                main_diagnosis=data.get("main_diagnosis", original_text[:100]),
                secondary_diagnoses=data.get("secondary_diagnoses", []),
                cie10_codes=data.get("cie10_codes", []),
                expanded_acronyms=data.get("expanded_acronyms", {}),
                clinical_entities=clinical_entities,
                symptoms=data.get("symptoms", []),
                risk_factors=data.get("risk_factors", []),
                anatomical_location=data.get("anatomical_location"),
                confidence_scores=[data.get("confidence", 0.8)],
                processing_method="llm_processing",
                success=True
            )
            
            return result
            
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido: {e}")
        except Exception as e:
            raise ValueError(f"Error procesando respuesta: {e}")
    
    def _clean_json_response(self, response: str) -> str:
        """Limpiar y extraer JSON de respuesta LLM"""
        
        # Buscar JSON en la respuesta
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No se encontró JSON válido en la respuesta")
        
        json_text = response[start_idx:end_idx]
        
        # Limpiar caracteres problemáticos
        json_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_text)
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        return json_text
    
    def _create_error_result(self, text: str, error_msg: str) -> MedicalProcessingResult:
        """Crear resultado de error mínimo"""
        
        return MedicalProcessingResult(
            cleaned_text=text,
            original_text=text,
            main_diagnosis=f"Error procesando: {text[:50]}...",
            processing_method="error_fallback",
            success=False,
            processing_notes=[f"Error: {error_msg}"]
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de procesamiento"""
        
        total = self.processing_stats["total_processed"]
        
        return {
            **self.processing_stats,
            "success_rate": (total - self.processing_stats["errors"]) / max(total, 1),
            "llm_usage_rate": self.processing_stats["llm_successful"] / max(total, 1),
            "fallback_usage_rate": self.processing_stats["fallback_used"] / max(total, 1),
            "llm_available": self.llm_client.is_available()
        }
    
    def reset_stats(self):
        """Resetear estadísticas"""
        self.processing_stats = {
            "total_processed": 0,
            "llm_successful": 0, 
            "fallback_used": 0,
            "errors": 0
        }
