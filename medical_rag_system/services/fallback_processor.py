"""
Procesador de fallback basado en reglas para cuando LLM no está disponible
"""

import re
from typing import Dict, Any, List
from datetime import datetime

from ..models.patient import PatientData
from ..models.medical import MedicalProcessingResult, ClinicalEntity


class FallbackProcessor:
    """Procesador de fallback basado en reglas médicas"""
    
    def __init__(self):
        self.acronyms_map = {
            "dm": "Diabetes Mellitus",
            "hta": "Hipertensión Arterial", 
            "iam": "Infarto Agudo de Miocardio",
            "acv": "Accidente Cerebrovascular",
            "epoc": "Enfermedad Pulmonar Obstructiva Crónica",
            "fa": "Fibrilación Auricular",
            "ic": "Insuficiencia Cardíaca",
            "fx": "Fractura",
            "itu": "Infección del Tracto Urinario",
            "tea": "Trastorno del Espectro Autista",
            "tbc": "Tuberculosis",
            "vih": "Virus de Inmunodeficiencia Humana",
            "sida": "Síndrome de Inmunodeficiencia Adquirida"
        }
        
        self.cie10_map = {
            "diabetes": ["E11.9", "E10.9"],
            "hipertension": ["I10"],
            "infarto": ["I21.9"],
            "cerebrovascular": ["I64"],
            "fractura": ["S72.9"],
            "neumonia": ["J18.9"],
            "epoc": ["J44.1"],
            "fibrilacion": ["I48"],
            "insuficiencia cardiaca": ["I50.9"],
            "tuberculosis": ["A15.9"],
            "cancer": ["C80.1"],
            "anemia": ["D64.9"],
            "asma": ["J45.9"]
        }
        
        self.symptom_keywords = {
            "dolor": ["dolor", "doloroso", "molestia", "algia"],
            "fiebre": ["fiebre", "febril", "hipertermia", "temperatura"],
            "disnea": ["disnea", "dificultad respiratoria", "falta de aire"],
            "tos": ["tos", "tos seca", "tos productiva"],
            "nausea": ["nausea", "vomito", "emesis"],
            "fatiga": ["fatiga", "cansancio", "astenia"],
            "mareo": ["mareo", "vertigo", "inestabilidad"],
            "cefalea": ["cefalea", "dolor de cabeza", "migraña"]
        }
        
        self.risk_factors_keywords = {
            "tabaquismo": ["fumador", "tabaco", "cigarrillo", "tabaquismo"],
            "alcoholismo": ["alcohol", "alcoholismo", "bebedor"],
            "sedentarismo": ["sedentario", "inactividad", "sin ejercicio"],
            "obesidad": ["obeso", "obesidad", "sobrepeso"],
            "hipertension": ["hipertenso", "tension alta", "hta"],
            "diabetes": ["diabetico", "diabetes", "dm"]
        }
    
    def process_diagnosis(self, 
                         diagnosis_text: str, 
                         patient_data: PatientData) -> MedicalProcessingResult:
        """Procesar diagnóstico usando reglas de fallback"""
        
        print("🛠️ Ejecutando procesamiento de fallback...")
        
        text_lower = diagnosis_text.lower()
        
        # Expandir acrónimos
        expanded_acronyms = self._expand_acronyms(text_lower)
        
        # Detectar códigos CIE-10
        cie10_codes = self._detect_cie10_codes(text_lower)
        
        # Extraer síntomas
        symptoms = self._extract_symptoms(text_lower)
        
        # Detectar factores de riesgo
        risk_factors = self._detect_risk_factors(text_lower, patient_data)
        
        # Extraer entidades clínicas básicas
        clinical_entities = self._extract_clinical_entities(diagnosis_text)
        
        # Determinar diagnóstico principal
        main_diagnosis = self._extract_main_diagnosis(diagnosis_text)
        
        # Crear resultado
        result = MedicalProcessingResult(
            cleaned_text=diagnosis_text.strip(),
            original_text=diagnosis_text,
            main_diagnosis=main_diagnosis,
            secondary_diagnoses=[],
            cie10_codes=cie10_codes,
            expanded_acronyms=expanded_acronyms,
            clinical_entities=clinical_entities,
            symptoms=symptoms,
            risk_factors=risk_factors,
            confidence_scores=[0.6],  # Confianza moderada para fallback
            processing_method="fallback_rules",
            processing_time=datetime.now().isoformat(),
            success=True
        )
        
        result.add_processing_note("Procesado con reglas de fallback")
        result.add_processing_note(f"Paciente: {patient_data.age} años, {patient_data.sex}")
        
        return result
    
    def _expand_acronyms(self, text: str) -> Dict[str, str]:
        """Expandir acrónimos médicos"""
        expanded = {}
        
        for acronym, expansion in self.acronyms_map.items():
            # Buscar acrónimo como palabra completa
            pattern = rf'\b{re.escape(acronym)}\b'
            if re.search(pattern, text, re.IGNORECASE):
                expanded[acronym.upper()] = expansion
        
        return expanded
    
    def _detect_cie10_codes(self, text: str) -> List[str]:
        """Detectar códigos CIE-10 basados en condiciones"""
        codes = []
        
        for condition, condition_codes in self.cie10_map.items():
            if condition in text:
                codes.extend(condition_codes)
        
        # Detectar códigos CIE-10 explícitos en el texto
        cie10_pattern = r'\b[A-Z]\d{2}\.?\d?\b'
        explicit_codes = re.findall(cie10_pattern, text.upper())
        codes.extend(explicit_codes)
        
        return list(set(codes))  # Remover duplicados
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extraer síntomas del texto"""
        symptoms = []
        
        for symptom, keywords in self.symptom_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    symptoms.append(symptom)
                    break
        
        return list(set(symptoms))
    
    def _detect_risk_factors(self, text: str, patient_data: PatientData) -> List[str]:
        """Detectar factores de riesgo"""
        risk_factors = []
        
        # Factores basados en edad
        if patient_data.age >= 65:
            risk_factors.append("edad_avanzada")
        if patient_data.age >= 75:
            risk_factors.append("edad_muy_avanzada")
        
        # Factores basados en texto
        for risk_factor, keywords in self.risk_factors_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    risk_factors.append(risk_factor)
                    break
        
        # Factores específicos por género
        if patient_data.sex == 'M' and patient_data.age > 45:
            if any(word in text for word in ["cardiaco", "infarto", "coronario"]):
                risk_factors.append("riesgo_cardiovascular_masculino")
        
        return list(set(risk_factors))
    
    def _extract_clinical_entities(self, text: str) -> List[ClinicalEntity]:
        """Extraer entidades clínicas básicas"""
        entities = []
        
        # Patrones para diagnósticos
        diagnosis_patterns = [
            r'\b(diabetes|hipertension|infarto|fractura|neumonia)\b',
            r'\b(cancer|tumor|neoplasia)\b',
            r'\b(anemia|asma|epoc)\b'
        ]
        
        for pattern in diagnosis_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = ClinicalEntity(
                    entity=match.group(0),
                    type="diagnosis",
                    confidence=0.7
                )
                entities.append(entity)
        
        # Patrones para síntomas
        symptom_patterns = [
            r'\b(dolor|fiebre|tos|nausea|mareo)\b',
            r'\b(disnea|fatiga|cefalea)\b'
        ]
        
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = ClinicalEntity(
                    entity=match.group(0),
                    type="symptom",
                    confidence=0.6
                )
                entities.append(entity)
        
        # Remover duplicados
        unique_entities = []
        seen_entities = set()
        
        for entity in entities:
            entity_key = (entity.entity.lower(), entity.type)
            if entity_key not in seen_entities:
                unique_entities.append(entity)
                seen_entities.add(entity_key)
        
        return unique_entities
    
    def _extract_main_diagnosis(self, text: str) -> str:
        """Extraer diagnóstico principal del texto"""
        
        # Buscar patrones de diagnóstico principal
        main_diagnosis_patterns = [
            r'diagnóstico:?\s*([^.]+)',
            r'dx:?\s*([^.]+)',
            r'impresión diagnóstica:?\s*([^.]+)',
            r'^([^.]+?)(?:con|presenta|cursa)'
        ]
        
        for pattern in main_diagnosis_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                diagnosis = match.group(1).strip()
                if len(diagnosis) > 10:  # Filtrar diagnósticos muy cortos
                    return diagnosis
        
        # Si no se encuentra patrón específico, usar las primeras palabras
        sentences = text.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 20:
                return first_sentence
        
        # Fallback: usar el texto completo truncado
        return text[:100].strip() + ("..." if len(text) > 100 else "")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del procesador de fallback"""
        return {
            "processor_type": "fallback_rules",
            "acronyms_count": len(self.acronyms_map),
            "cie10_mappings": len(self.cie10_map),
            "symptom_categories": len(self.symptom_keywords),
            "risk_factor_categories": len(self.risk_factors_keywords),
            "confidence_range": "0.5 - 0.7",
            "capabilities": [
                "acronym_expansion",
                "cie10_detection", 
                "symptom_extraction",
                "risk_factor_assessment",
                "entity_recognition"
            ]
        }
